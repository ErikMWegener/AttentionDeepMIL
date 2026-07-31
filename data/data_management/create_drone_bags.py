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


def create_bags(num_bags, mean_bag_len, var_bag_len, positive_patches, negative_patches,
                output_path, dataset_name='drone_bags', split='train', bag_ratio=0.5,
                seed=0, grayscale=True):
    """Create randomized bags from patch definitions and write them to file.

    Positive bags contain a shuffled mix of positive and negative patches (with
    at least one positive patch); negative bags contain only negative patches.
    Images are opened and cropped here, at write time.
    """
    transform = build_transform(grayscale)

    num_pos_bags = int(num_bags * bag_ratio)
    num_neg_bags = num_bags - num_pos_bags

    r = np.random.RandomState(seed)
    print(f'Creating {num_pos_bags} positive and {num_neg_bags} negative bags with '
          f'mean length {mean_bag_len} and variance {var_bag_len}...')

    # Positive bags with a shuffled mix of positive and negative patches.
    for i in tqdm(range(num_pos_bags), desc=f"Creating positive {split} bags"):
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

        patches = []
        instance_labels = []
        for instance in instances:
            patch = fetch_patch_from_image(instance['image_path'], instance['coords'])
            patches.append(transform(patch))
            instance_labels.append(instance['label'])

        bag_tensor = torch.stack(patches)
        dataset_manager.DatasetWriter(output_path).write(dataset_name,
                                                    f'{split}_{i}',
                                                    None,
                                                    label=1,
                                                    patches=bag_tensor,
                                                    count=num_positive,
                                                    instance_label=torch.tensor(instance_labels),
                                                    split=split)

    # Negative bags with only negative patches.
    for i in tqdm(range(num_neg_bags), desc=f"Creating negative {split} bags"):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_negative = bag_length  # all patches in negative bags are negative

        neg_idx = r.choice(len(negative_patches), num_negative, replace=True)
        instances = [negative_patches[j] for j in neg_idx]

        order = r.permutation(len(instances))
        instances = [instances[j] for j in order]

        patches = []
        instance_labels = []
        for instance in instances:
            patch = fetch_patch_from_image(instance['image_path'], instance['coords'])
            patches.append(transform(patch))
            instance_labels.append(instance['label'])

        bag_tensor = torch.stack(patches)
        dataset_manager.DatasetWriter(output_path).write(dataset_name,
                                                    f'{split}_{i+num_pos_bags}',
                                                    None,
                                                    label=0,
                                                    patches=bag_tensor,
                                                    count=0,
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
                bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale)

    create_bags(args.num_bags[1], args.mean_bag_len, args.var_bag_len,
                positive_patches[train_pos_index:train_pos_index+val_pos_index],
                negative_patches[train_neg_index:train_neg_index+val_neg_index],
                args.output_path, args.dataset_name, split='validation',
                bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale)

    create_bags(args.num_bags[2], args.mean_bag_len, args.var_bag_len,
                positive_patches[train_pos_index+val_pos_index:],
                negative_patches[train_neg_index+val_neg_index:],
                args.output_path, args.dataset_name, split='test',
                bag_ratio=args.bag_ratio, seed=args.seed, grayscale=args.grayscale)


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

    args = parser.parse_args()

    pos_jpgs, neg_jpgs = load_metadata()

    if args.legacy:
        run_legacy(args, pos_jpgs, neg_jpgs)
    else:
        run_bags(args, pos_jpgs, neg_jpgs)

    print('All done!')
