"""Bags from the GWHD 2021 images (thesis sections "GWHD dataset",
"Patch extraction and labelling" and "Bag construction").

1. Patch definitions: a regular grid of patch_size x patch_size crops over each
   image; a patch is positive if a bounding box covers at least
   ``--overlap_threshold`` of its area, otherwise negative.
2. Splits: ``--original_split`` keeps the official GWHD train/val/test images;
   otherwise all images are shuffled (``--seed``) and the patch pools are split in
   proportion to ``--num_bags``.
3. Bags: per split, half of the bags are positive with a number of positive
   patches drawn uniformly from 1..(bag length - 1); the bag length is drawn from a
   normal distribution with mean ``--mean_bag_len`` and standard deviation
   ``--var_bag_len``. Patches are sampled with replacement from the split's pools.
   Patches are converted to grayscale with ``--grayscale`` and normalised to [-1, 1].

Run from data/data_management/, for example
    python create_gwhd_bags.py --output_path ../datasets/bags/my_gwhd_bags.h5 \
        --dataset_name my_gwhd_bags --num_bags 1000 200 200 --patch_size 64 \
        --overlap_threshold 0.7 --grayscale
"""
from collections import defaultdict
from importlib.resources import path
import os
from PIL import Image
from tqdm import tqdm

import torch
import numpy as np
import pandas as pd
from torchvision import datasets, transforms
import dataset_manager
import argparse
import multiprocessing as mp
import sys
import os

# Read CSV-File for GWHD dataset

TRAIN_CSV_PATH = '../datasets/gwhd_2021/competition_train.csv'
TEST_CSV_PATH = '../datasets/gwhd_2021/competition_test.csv'
VAL_CSV_PATH = '../datasets/gwhd_2021/competition_val.csv'

IMAGE_PATH = '../datasets/gwhd_2021/images'

def load_full_metadata():
    """Bounding boxes of the official train/val/test images as three dicts {image_name: [boxes]}."""
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)
    df_val = pd.read_csv(VAL_CSV_PATH)

    metadata_train = load_metadata(df_train)
    metadata_val = load_metadata(df_val)
    metadata_test = load_metadata(df_test)

    return metadata_train, metadata_val, metadata_test

def load_metadata(df):
    """Parse the ``BoxesString`` column of a GWHD competition CSV into {image_name: [(x1, y1, x2, y2), ...]}."""
    metadata = {}
    for _, row in df.iterrows():
        image_id = row['image_name']
        boxes = []
        for s in row['BoxesString'].split(';'):
            if s == 'no_box':
                continue
            if s.strip():
                x1, y1, x2, y2 = map(int, s.split(' '))
                boxes.append((x1, y1, x2, y2))
        metadata[image_id] = boxes
    print(f'{len(metadata)} images found')
    return metadata
# Compute the overlap between a patch and a bounding box 

def compute_overlap(patch_box, bbox):
    """Fraction of the patch area covered by the bounding box (intersection / patch area)."""
    x1 = max(patch_box[0], bbox[0])
    y1 = max(patch_box[1], bbox[1])
    x2 = min(patch_box[2], bbox[2])
    y2 = min(patch_box[3], bbox[3])

    if x1 >= x2 and y1 >= y2:
        return 0.0
    
    intersection = (x2 - x1) * (y2 -y1)
    patch_area = (patch_box[2] - patch_box[0]) * (patch_box[3] - patch_box[1])

    overlap = intersection / patch_area
    return overlap

# Generate patch list with coordinates, the correspinding image ID and the label for each patch based on the overlap with the bounding boxes

def extract_box_patches(bbox, img_w, img_h, patch_size=28, stride=28, threshold=1.0):
    """Grid of patches centred on one bounding box ("centred" patch definition).

    The grid has max(1, floor(s_box * (2 - threshold) / stride)) patches per axis,
    so every box yields at least one patch and the grid grows with the box size
    (formula in the thesis section "Patch extraction and labelling"). The grid is
    shifted back into the image where it would cross the border.
    """
    x1, y1, x2, y2 = bbox

    box_h = y2 - y1
    box_w = x2 - x1

    center_x = x1 + box_w // 2
    center_y = y1 + box_h // 2

    num_patch_in_width = max(1, int(box_w * (1 + 1 - threshold) / (stride)))
    num_patch_in_height = max(1, int(box_h * (1 + 1 - threshold) / (stride)))

    start_x = max(0, center_x - (num_patch_in_width * stride) // 2)
    start_y = max(0, center_y - (num_patch_in_height * stride) // 2)
    
    if start_x + num_patch_in_width * stride > img_w:
        start_x = img_w - num_patch_in_width * stride - 1
    if start_y + num_patch_in_height * stride > img_h:
        start_y = img_h - num_patch_in_height * stride - 1
    coords = []

    for y in range(min(start_y, img_h - patch_size - 1), min(img_h - 1, start_y + num_patch_in_height * stride), stride):
        if y + patch_size > img_h:
            break
        for x in range(min(start_x, img_w - patch_size - 1), min(img_w - 1, start_x + num_patch_in_width * stride), stride):
            if x + patch_size > img_w:
                break
            patch_coords = (x, y, x + patch_size, y + patch_size)
            
            coords.append(patch_coords)

    return coords

def generate_patch_definitions(metadata, patch_size=28, stride=28, dense=False, overlap_threshold=0.5):
    """Positive and negative patch pools (image, crop coordinates, label) without pixel data.

    Grid mode (``dense=False``, used from the command line): a patch on the regular
    grid is positive if a box covers at least ``overlap_threshold`` of it.
    ``dense=True`` adds centred patches per box (``extract_box_patches``); it is not
    reachable from the command line in the current version (see README, "Notes").
    """
    positive_patches = []
    negative_patches = []

    print(f'Generating patch definitions with patch size {patch_size} and stride {stride}...')

    for image_id, boxes in metadata.items():

        img_w, img_h = Image.open(f'{IMAGE_PATH}/{image_id}').size
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_coords = (x, y, x + patch_size, y + patch_size)

                patch_label = 0
                for bbox in boxes:
                    overlap = compute_overlap(patch_coords, bbox)
                    if dense and overlap > 0:
                        patch_label = 1
                    elif overlap >= overlap_threshold:
                        patch_label = 1
                        break
                if not dense and patch_label == 1:
                    positive_patches.append({'image_id': image_id, 'coords': patch_coords, 'label': patch_label})
                else:
                    negative_patches.append({'image_id': image_id, 'coords': patch_coords, 'label': patch_label})

        if dense:

            for bbox in boxes:
                dense_coords = extract_box_patches(bbox, img_w, img_h, patch_size, stride, threshold=overlap_threshold)
                for patch_coords in dense_coords:
                    positive_patches.append({'image_id': image_id, 'coords': patch_coords, 'label': 1})

    print(f'Generated {len(positive_patches)} positive patches and {len(negative_patches)} negative patches.')
    
    return positive_patches, negative_patches

def fetch_patch_from_image(image_path, coords):
    with Image.open(image_path) as img:
        patch = img.crop(coords)
        return patch


def _crop_patches_from_image(task):
    """Decode one image a single time and cut all requested patches out of it.

    PIL always decodes the whole file, so opening an image once per patch
    dominates the runtime. Everything that is needed from one image is
    therefore collected first and cropped in one go.
    """
    image_id, coords_list, grayscale = task
    patches = []
    with Image.open(f'{IMAGE_PATH}/{image_id}') as img:
        img.load()
        for coords in coords_list:
            patch = img.crop(coords)
            if grayscale:
                patch = patch.convert('L')
            patches.append(np.asarray(patch, dtype=np.uint8))
    return image_id, np.stack(patches)


def _materialize_patches(bags_instances, grayscale, workers, desc):
    """Fetch the pixel data for a list of bags, one decode per source image.

    ``bags_instances`` is a list of per-bag instance lists. Returns a list of
    uint8 arrays, one per bag, with shape ``(bag_len, H, W)`` for grayscale and
    ``(bag_len, H, W, 3)`` otherwise.
    """
    # image_id -> [(bag index, slot index, coords), ...]
    groups = defaultdict(list)
    for bag_idx, instances in enumerate(bags_instances):
        for slot_idx, instance in enumerate(instances):
            groups[instance['image_id']].append((bag_idx, slot_idx, instance['coords']))

    tasks = [(image_id, [c for _, _, c in entries], grayscale) for image_id, entries in groups.items()]
    slots = [[None] * len(instances) for instances in bags_instances]

    def scatter(result):
        image_id, patches = result
        for (bag_idx, slot_idx, _), patch in zip(groups[image_id], patches):
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

    Equivalent to ``ToTensor`` + ``Normalize((0.5,), (0.5,))`` of the
    torchvision transform, but applied to the whole bag at once.
    """
    tensor = torch.from_numpy(patches)
    if tensor.ndim == 3:                       # (N, H, W) grayscale
        tensor = tensor.unsqueeze(1)
    else:                                      # (N, H, W, C) rgb
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
    return tensor.float().div_(255.0).sub_(0.5).div_(0.5)

# Create bags from the patch definitions and write them to an H5 file using the DatasetWriter class. 
# Each bag corresponds to an image and contains all patches extracted from that image along with their labels.

def create_bags(num_bags, mean_bag_len, var_bag_len, positive_patches, negative_patches, output_path, dataset_name='gwhd_bags', split='train', bag_ratio=0.5, seed=0, grayscale=False, workers=None, chunk_bags=200):
    """Create randomized bags from patch definitions and write them to file.

    The bag compositions are sampled first, then the pixel data is fetched per
    source image instead of per patch. Bags are processed in chunks of
    ``chunk_bags`` so that only a bounded amount of patch data is held in memory.
    """
    num_pos_bags = int(num_bags * bag_ratio)
    num_neg_bags = num_bags - num_pos_bags

    if workers is None:
        workers = max(1, (os.cpu_count() or 1) - 1)

    r = np.random.RandomState(seed)
    print(f'Creating {num_pos_bags} positive and {num_neg_bags} negative bags with mean length {mean_bag_len} and variance {var_bag_len}...')

    # ---- 1. Sample the bag compositions (no pixel data touched yet). --------
    plans = []

    # Positive bags with varying number of positive and negative patches
    for i in range(num_pos_bags):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_positive = r.randint(1, bag_length) if bag_length > 1 else 1  # Ensure at least one positive patch
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

    # Negative bags with only negative patches
    for i in range(num_neg_bags):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_negative = bag_length  # All patches in negative bags are negative

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

    print(f'Finished creating {num_bags} bags and saved them to {output_path}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create bags for GWHD dataset')

    parser.add_argument('--output_path', type=str, default='../datasets/bags/gwhd_bags.h5', help='Path to the output H5 file for the bags')
    parser.add_argument('--dataset_name', type=str, default='gwhd_bags', help='Name of the dataset to create within the H5 file')
    parser.add_argument('--num_bags', type=int, nargs='+', default=[1000,200,200], help='Total number of bags to create')
    parser.add_argument('--mean_bag_len', type=int, default=100, help='Mean number of instances per bag')
    parser.add_argument('--var_bag_len', type=int, default=10, help='Variance of the number of instances per bag')
    parser.add_argument('--bag_ratio', type=float, default=0.5, help='Ratio of positive to negative bags (default: 0.5)')
    parser.add_argument('--patch_size', type=int, default=28, help='Size of the patches to extract from the images (default: 28)')
    parser.add_argument('--overlap_threshold', type=float, default=0.5, help='Overlap threshold for labeling patches as positive (default: 0.5)')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Activate grayscale images')
    parser.add_argument('--dense', action='store_true', default=False, help='Activate dense patch generation (label all patches with any overlap as positive)')
    parser.add_argument('--original_split', action='store_true', default=False, help='Use the original train/val/test split instead of shuffling and splitting the dataset')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)') 
    parser.add_argument('--workers', type=int, default=None, help='Processes used to decode and crop the source images (default: CPU count - 1)')
    parser.add_argument('--chunk_bags', type=int, default=200, help='Number of bags whose patches are held in memory at once')
    args = parser.parse_args()

    metadata_train, metadata_val, metadata_test = load_full_metadata()

    if args.original_split:
        pos_patches_train, neg_patches_train = generate_patch_definitions(metadata_train, patch_size=args.patch_size, stride=args.patch_size, overlap_threshold=args.overlap_threshold)
        pos_patches_val, neg_patches_val = generate_patch_definitions(metadata_val, patch_size=args.patch_size, stride=args.patch_size, overlap_threshold=args.overlap_threshold)
        pos_patches_test, neg_patches_test = generate_patch_definitions(metadata_test, patch_size=args.patch_size, stride=args.patch_size, overlap_threshold=args.overlap_threshold)
    else:
        merged = {**metadata_train, **metadata_val, **metadata_test}
        image_ids = list(merged)
        np.random.RandomState(args.seed).shuffle(image_ids)  # Shuffle the images
        metadata = {image_id: merged[image_id] for image_id in image_ids}

        positive_patches, negative_patches = generate_patch_definitions(metadata, patch_size=args.patch_size, stride=args.patch_size, overlap_threshold=args.overlap_threshold)

        train_pos_index = int(len(positive_patches) * args.num_bags[0] / sum(args.num_bags))
        train_neg_index = int(len(negative_patches) * args.num_bags[0] / sum(args.num_bags))

        val_pos_index = int(len(positive_patches) * args.num_bags[1] / sum(args.num_bags))
        val_neg_index = int(len(negative_patches) * args.num_bags[1] / sum(args.num_bags))

        pos_patches_train = positive_patches[:train_pos_index]
        neg_patches_train = negative_patches[:train_neg_index]

        pos_patches_val = positive_patches[train_pos_index:train_pos_index+val_pos_index]
        neg_patches_val = negative_patches[train_neg_index:train_neg_index+val_neg_index]

        pos_patches_test = positive_patches[train_pos_index+val_pos_index:]
        neg_patches_test = negative_patches[train_neg_index+val_neg_index:]

    # Create train bags
    create_bags(args.num_bags[0], 
                args.mean_bag_len, 
                args.var_bag_len, 
                pos_patches_train, 
                neg_patches_train, 
                args.output_path, 
                args.dataset_name,
                split='train', 
                bag_ratio=args.bag_ratio, 
                seed=args.seed,
                grayscale=args.grayscale,
                workers=args.workers,
                chunk_bags=args.chunk_bags) 

    # Create validation bags  
    create_bags(args.num_bags[1], 
                args.mean_bag_len, 
                args.var_bag_len, 
                pos_patches_val, 
                neg_patches_val, 
                args.output_path, 
                args.dataset_name,
                split='validation', 
                bag_ratio=args.bag_ratio, 
                seed=args.seed,
                grayscale=args.grayscale,
                workers=args.workers,
                chunk_bags=args.chunk_bags)    
    
    # Create test bags
    create_bags(args.num_bags[2], 
                args.mean_bag_len, 
                args.var_bag_len, 
                pos_patches_test, 
                neg_patches_test, 
                args.output_path, 
                args.dataset_name,
                split='test', 
                bag_ratio=args.bag_ratio, 
                seed=args.seed,
                grayscale=args.grayscale,
                workers=args.workers,
                chunk_bags=args.chunk_bags)

    print('All done!')