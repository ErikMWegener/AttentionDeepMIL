from PIL import Image
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from torchvision import transforms
import dataset_manager
import argparse
import multiprocessing as mp
import sys
import os

from collections import defaultdict

POS_PATH = '../datasets/drone/2026_wheat_heads_datasets/2026-06-08_BBG/DJI_202606081251_161_BBGnullspike'
NEG_PATH = '../datasets/drone/2026_wheat_heads_datasets/2026-05-26_BBG/DJI_202605261135_140_BBGnullspike'


def build_transform(grayscale=True):
    if grayscale:
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # In Graustufen umwandeln
            transforms.ToTensor(),                      # In einen PyTorch-Tensor umwandeln
            transforms.Normalize((0.5,), (0.5,))        # Normalisieren (wie bei MNIST üblich)
        ])
    return transforms.Compose([
        transforms.ToTensor(),                      # In einen PyTorch-Tensor umwandeln
        transforms.Normalize((0.5,), (0.5,))        # Normalisieren (wie bei MNIST üblich)
    ])


def load_metadata():
    """Load the image file names and the wheat-head middle points.

    Positive images each have a matching ``.txt`` file that contains one
    ``x y`` middle point of a wheat head per line. Negative images have no
    wheat heads. Returns the positive/negative png file names and a mapping
    ``{positive_png: np.ndarray of shape (N, 2)}`` of middle points.
    """
    pos_jpgs = sorted([f for f in os.listdir(POS_PATH) if f.endswith('.JPG')])
    neg_jpgs = sorted([f for f in os.listdir(NEG_PATH) if f.endswith('.JPG')])

    return pos_jpgs, neg_jpgs

def load_point_annotations(annotation_path):
    """
    Load point annotations from a text file.
    Each line in the file should contain two numbers (x, y) representing the coordinates of a point.
    """
    points = np.loadtxt(annotation_path, ndmin=2)   # default delimiter splits on any whitespace
    return points

# ---------------------------------------------------------------------------
# New approach: patch definitions + randomized bags (like create_gwhd_bags.py)
# ---------------------------------------------------------------------------

def create_patch_definitions(pos_jpgs, neg_jpgs, patch_size=128, stride=128):
    """Create lightweight patch definitions without loading pixel data.

    Positive patches are cut directly centered on the wheat-head middle points
    of the positive images. Negative patches are taken from a regular grid over
    the negative images only. Each definition stores the source image path and
    the crop coordinates so the actual images are only opened/cropped later,
    when the bags are written.
    """
    positive_patches = []
    negative_patches = []

    half = patch_size // 2

    print(f'Generating patch definitions with patch size {patch_size} and stride {stride}...')

    # Positive patches: one patch centered on every wheat-head middle point.
    for jpg in tqdm(pos_jpgs, desc='Positive patch definitions'):
            image_path = f'{POS_PATH}/{jpg}'
            img_w, img_h = Image.open(image_path).size
            for y in range(0, img_h - patch_size + 1, stride):
                for x in range(0, img_w - patch_size + 1, stride):
                    coords = (x, y, x + patch_size, y + patch_size)
                    positive_patches.append({'image_path': image_path, 'coords': coords, 'label': 1})

    # Negative patches: regular grid over the negative images only.
    for jpg in tqdm(neg_jpgs, desc='Negative patch definitions'):
        image_path = f'{NEG_PATH}/{jpg}'
        img_w, img_h = Image.open(image_path).size
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                coords = (x, y, x + patch_size, y + patch_size)
                negative_patches.append({'image_path': image_path, 'coords': coords, 'label': 0})

    print(f'Generated {len(positive_patches)} positive patches and '
          f'{len(negative_patches)} negative patches.')

    return positive_patches, negative_patches


def fetch_patch_from_image(image_path, coords):
    with Image.open(image_path) as img:
        patch = img.crop(coords)
        return patch


def _crop_patches_from_image(task):
    """Decode one image a single time and cut all requested patches out of it.

    A drone JPEG is ~5280x3956 px and PIL always decodes the whole file, so
    opening it once per patch dominates the runtime. Everything that is needed
    from one image is therefore collected first and cropped in one go.
    """
    image_path, coords_list, grayscale = task
    patches = []
    with Image.open(image_path) as img:
        img.load()
        for coords in coords_list:
            patch = img.crop(coords)
            if grayscale:
                patch = patch.convert('L')
            patches.append(np.asarray(patch, dtype=np.uint8))
    return image_path, np.stack(patches)


def _materialize_patches(bags_instances, grayscale, workers, desc):
    """Fetch the pixel data for a list of bags, one decode per source image.

    ``bags_instances`` is a list of per-bag instance lists. Returns a list of
    uint8 arrays, one per bag, with shape ``(bag_len, H, W)`` for grayscale and
    ``(bag_len, H, W, 3)`` otherwise.
    """
    # image_path -> [(bag index, slot index, coords), ...]
    groups = defaultdict(list)
    for bag_idx, instances in enumerate(bags_instances):
        for slot_idx, instance in enumerate(instances):
            groups[instance['image_path']].append((bag_idx, slot_idx, instance['coords']))

    tasks = [(path, [c for _, _, c in entries], grayscale) for path, entries in groups.items()]
    slots = [[None] * len(instances) for instances in bags_instances]

    def scatter(result):
        image_path, patches = result
        for (bag_idx, slot_idx, _), patch in zip(groups[image_path], patches):
            slots[bag_idx][slot_idx] = patch

    if workers > 1 and len(tasks) > 1:
        with mp.Pool(min(workers, len(tasks))) as pool:
            for result in tqdm(pool.imap_unordered(_crop_patches_from_image, tasks),
                               total=len(tasks), desc=desc):
                scatter(result)
    else:
        for task in tqdm(tasks, desc=desc):
            scatter(_crop_patches_from_image(task))

    return [np.stack(bag) for bag in slots]


def _to_bag_tensor(patches):
    """Turn a uint8 patch stack into the normalized tensor the writer expects.

    Equivalent to ``ToTensor`` + ``Normalize((0.5,), (0.5,))`` of
    :func:`build_transform`, but applied to the whole bag at once.
    """
    tensor = torch.from_numpy(patches)
    if tensor.ndim == 3:                       # (N, H, W) grayscale
        tensor = tensor.unsqueeze(1)
    else:                                      # (N, H, W, C) rgb
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor.float().div_(255.0).sub_(0.5).div_(0.5)


def create_bags(num_bags, mean_bag_len, var_bag_len, positive_patches, negative_patches,
                output_path, dataset_name='drone_bags', split='train', bag_ratio=0.5,
                seed=0, grayscale=True, workers=None, chunk_bags=200):
    """Create randomized bags from patch definitions and write them to file.

    Positive bags contain a shuffled mix of positive and negative patches (with
    at least one positive patch); negative bags contain only negative patches.

    The bag compositions are sampled first, then the pixel data is fetched per
    source image instead of per patch. Bags are processed in chunks of
    ``chunk_bags`` so that only a bounded amount of patch data is held in memory.
    """
    num_pos_bags = int(num_bags * bag_ratio)
    num_neg_bags = num_bags - num_pos_bags

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    r = np.random.RandomState(seed)
    print(f'Creating {num_pos_bags} positive and {num_neg_bags} negative bags with '
          f'mean length {mean_bag_len} and variance {var_bag_len}...')

    # ---- 1. Sample the bag compositions (no pixel data touched yet). --------
    plans = []

    # Positive bags with a shuffled mix of positive and negative patches.
    for i in range(num_pos_bags):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_positive = r.randint(1, bag_length) if bag_length > 1 else 1  # at least one positive patch
        num_negative = bag_length - num_positive

        pos_idx = r.choice(len(positive_patches), num_positive, replace=True)
        instances = [positive_patches[j] for j in pos_idx]

        if num_negative > 0:
            neg_idx = r.choice(len(negative_patches), num_negative, replace=True)
            instances += [negative_patches[j] for j in neg_idx]

        order = r.permutation(len(instances))
        instances = [instances[j] for j in order]

        plans.append({'image_id': f'{split}_{i}', 'label': 1, 'count': num_positive,
                      'instances': instances})

    # Negative bags with only negative patches.
    for i in range(num_neg_bags):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_negative = bag_length  # all patches in negative bags are negative

        neg_idx = r.choice(len(negative_patches), num_negative, replace=True)
        instances = [negative_patches[j] for j in neg_idx]

        order = r.permutation(len(instances))
        instances = [instances[j] for j in order]

        plans.append({'image_id': f'{split}_{i+num_pos_bags}', 'label': 0, 'count': 0,
                      'instances': instances})

    # ---- 2. Fetch pixels chunk by chunk and write the bags. -----------------
    writer = dataset_manager.DatasetWriter(output_path)
    num_chunks = (len(plans) + chunk_bags - 1) // chunk_bags

    for chunk_nr, start in enumerate(range(0, len(plans), chunk_bags), start=1):
        chunk = plans[start:start + chunk_bags]
        bags = _materialize_patches([p['instances'] for p in chunk], grayscale, workers,
                                    desc=f'Cropping {split} chunk {chunk_nr}/{num_chunks}')

        for plan, patches in tqdm(list(zip(chunk, bags)),
                                  desc=f'Writing {split} chunk {chunk_nr}/{num_chunks}'):
            instance_labels = [instance['label'] for instance in plan['instances']]
            writer.write(dataset_name,
                         plan['image_id'],
                         None,
                         label=plan['label'],
                         patches=_to_bag_tensor(patches),
                         count=plan['count'],
                         instance_label=torch.tensor(instance_labels),
                         split=split)

    print(f'Finished creating {num_bags} {split} bags and saved them to {output_path}.')


# ---------------------------------------------------------------------------
# Legacy approach: one bag per image, dense grid of patches (kept as an option)
# ---------------------------------------------------------------------------

def create_split_bags(output_path, dataset_name, jpgs, patch_size=128, stride=128, grayscale=True, split='train', positive=True):

    transform = build_transform(grayscale)

    base_path = POS_PATH if positive else NEG_PATH

    for image_nr in tqdm(range(len(jpgs)), desc=f"Creating positive {split} synthetic bags" if positive else f"Creating negative {split} synthetic bags"):
        image = Image.open(f'{base_path}/{jpgs[image_nr]}')
        img_w, img_h = image.size
        patches = []
        patch_coords_list = []
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_coords = (x, y, x + patch_size, y + patch_size)
                patch = image.crop(patch_coords)
                patch_tensor = transform(patch)
                patches.append(patch_tensor)
                patch_coords_list.append(patch_coords)
        bag_tensor = torch.stack(patches)
        dataset_manager.DatasetWriter(output_path).write(dataset_name,
                                                    f'{split}_{image_nr if positive else image_nr + len(jpgs)}',
                                                    None,
                                                    label = 1 if positive else 0,
                                                    patches=bag_tensor,
                                                    count = 0,
                                                    instance_label=torch.zeros(bag_tensor.size(0)),
                                                    split=split)


def run_legacy(args, pos_jpgs, neg_jpgs):
    n0, n1, n2 = args.num_bags[0], args.num_bags[1], args.num_bags[2]

    create_split_bags(args.output_path, args.dataset_name, pos_jpgs[:n0//2], args.patch_size, args.stride, args.grayscale, 'train', positive=True)
    create_split_bags(args.output_path, args.dataset_name, neg_jpgs[:n0//2], args.patch_size, args.stride, args.grayscale, 'train', positive=False)

    create_split_bags(args.output_path, args.dataset_name, pos_jpgs[n0//2+1:n0//2 + n1//2], args.patch_size, args.stride, args.grayscale, 'test', positive=True)
    create_split_bags(args.output_path, args.dataset_name, neg_jpgs[n0//2+1:n0//2 + n1//2], args.patch_size, args.stride, args.grayscale, 'test', positive=False)

    create_split_bags(args.output_path, args.dataset_name, pos_jpgs[n0//2 + n1//2+1:n0//2 + n1//2 + n2//2], args.patch_size, args.stride, args.grayscale, 'val', positive=True)
    create_split_bags(args.output_path, args.dataset_name, neg_jpgs[n0//2 + n1//2+1:n0//2 + n1//2 + n2//2], args.patch_size, args.stride, args.grayscale, 'val', positive=False)


def run_bags(args, pos_jpgs, neg_jpgs):
    positive_patches, negative_patches = create_patch_definitions(
        pos_jpgs, neg_jpgs, patch_size=args.patch_size, stride=args.stride)

    total = sum(args.num_bags)
    train_pos_index = int(len(positive_patches) * args.num_bags[0] / total)
    train_neg_index = int(len(negative_patches) * args.num_bags[0] / total)
    val_pos_index = int(len(positive_patches) * args.num_bags[1] / total)
    val_neg_index = int(len(negative_patches) * args.num_bags[1] / total)

    create_bags(args.num_bags[0], args.mean_bag_len, args.var_bag_len,
                positive_patches[:train_pos_index],
                negative_patches[:train_neg_index],
                args.output_path, args.dataset_name, split='train',
                bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale,
                workers=args.workers, chunk_bags=args.chunk_bags)

    create_bags(args.num_bags[1], args.mean_bag_len, args.var_bag_len,
                positive_patches[train_pos_index:train_pos_index+val_pos_index],
                negative_patches[train_neg_index:train_neg_index+val_neg_index],
                args.output_path, args.dataset_name, split='validation',
                bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale,
                workers=args.workers, chunk_bags=args.chunk_bags)

    create_bags(args.num_bags[2], args.mean_bag_len, args.var_bag_len,
                positive_patches[train_pos_index+val_pos_index:],
                negative_patches[train_neg_index+val_neg_index:],
                args.output_path, args.dataset_name, split='test',
                bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale,
                workers=args.workers, chunk_bags=args.chunk_bags)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create synthetic bags from images and annotations.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the drone bags.')
    parser.add_argument('--dataset_name', type=str, required=True, help='Name of the dataset.')
    parser.add_argument('--num_bags', nargs='+', type=int, default=[1000, 200, 200], help='Number of bags to create (train, val, test).')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of each patch.')
    parser.add_argument('--stride', type=int, default=128, help='Stride for patch extraction.')
    parser.add_argument('--grayscale', action='store_true', help='Convert images to grayscale.')
    parser.add_argument('--mean_bag_len', type=int, default=100, help='Mean number of instances per bag (randomized bags).')
    parser.add_argument('--var_bag_len', type=int, default=10, help='Variance of the number of instances per bag (randomized bags).')
    parser.add_argument('--bag_ratio', type=float, default=0.5, help='Ratio of positive to negative bags (randomized bags).')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (randomized bags).')
    parser.add_argument('--legacy', action='store_true', help='Use the legacy one-bag-per-image approach instead of randomized bags.')
    parser.add_argument('--workers', type=int, default=None, help='Processes used to decode and crop the source images (default: CPU count - 1).')
    parser.add_argument('--chunk_bags', type=int, default=200, help='Number of bags whose patches are held in memory at once.')

    args = parser.parse_args()

    pos_jpgs, neg_jpgs = load_metadata()

    if args.legacy:
        run_legacy(args, pos_jpgs, neg_jpgs)
    else:
        run_bags(args, pos_jpgs, neg_jpgs)

    print('All done!')
