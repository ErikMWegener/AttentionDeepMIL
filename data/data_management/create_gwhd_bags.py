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
import sys
import os

# Read CSV-File for GWHD dataset

TRAIN_CSV_PATH = '../datasets/gwhd_2021/competition_train.csv'
TEST_CSV_PATH = '../datasets/gwhd_2021/competition_test.csv'
VAL_CSV_PATH = '../datasets/gwhd_2021/competition_val.csv'

IMAGE_PATH = '../datasets/gwhd_2021/images'

def load_metadata(seed=0):
    df_train = pd.read_csv(TRAIN_CSV_PATH)
    df_test = pd.read_csv(TEST_CSV_PATH)
    df_val = pd.read_csv(VAL_CSV_PATH)

    df = pd.concat([df_train, df_test, df_val], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # Shuffle the DataFrame

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

# Create bags from the patch definitions and write them to an H5 file using the DatasetWriter class. 
# Each bag corresponds to an image and contains all patches extracted from that image along with their labels.

def create_bags(num_bags, mean_bag_len, var_bag_len, positive_patches, negative_patches, output_path, dataset_name='gwhd_bags', split='train', bag_ratio=0.5, seed=0, grayscale=False):

    if grayscale:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1), # In Graustufen umwandeln
            transforms.ToTensor(),                      # In einen PyTorch-Tensor umwandeln
            transforms.Normalize((0.5,), (0.5,))        # Normalisieren (wie bei MNIST üblich)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),                      # In einen PyTorch-Tensor umwandeln
            transforms.Normalize((0.5,), (0.5,))        # Normalisieren (wie bei MNIST üblich)
        ])

    num_pos_bags = int(num_bags * bag_ratio)
    num_neg_bags = num_bags - num_pos_bags

    r = np.random.RandomState(seed)
    print(f'Creating {num_pos_bags} positive and {num_neg_bags} negative bags with mean length {mean_bag_len} and variance {var_bag_len}...')
    
    # Positive bags with varying number of positive and negative patches

    for i in tqdm(range(num_pos_bags), desc="Creating positive bags"):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        num_positive = r.randint(1, bag_length)  # Ensure at least one positive patch
        num_negative = bag_length - num_positive

        pos_idx = r.choice(len(positive_patches), num_positive, replace=True)
        instances = [positive_patches[i] for i in pos_idx]

        if num_negative > 0:
            neg_idx = r.choice(len(negative_patches), num_negative, replace=True)
            instances += [negative_patches[i] for i in neg_idx]
        
        order = r.permutation(len(instances))
        instances = [instances[i] for i in order]

        patches = []
        instance_labels = []
        for instance in instances:
            patch = fetch_patch_from_image(f'{IMAGE_PATH}/{instance["image_id"]}', instance["coords"])
            patch_tensor = transform(patch)
            patches.append(patch_tensor)
            instance_labels.append(instance['label'])

        bag_tensor = torch.stack(patches)
        dataset_manager.DatasetWriter(output_path).write(dataset_name, 
                                                    f'{split}_{i}' , 
                                                    None, 
                                                    label = 1, 
                                                    patches=bag_tensor,
                                                    count = num_positive, 
                                                    instance_label=torch.tensor(instance_labels), 
                                                    split=split)
        
    # Negative bags with only negative patches
    for i in tqdm(range(num_neg_bags), desc="Creating negative bags"):
        bag_length = max(1, int(r.normal(mean_bag_len, var_bag_len)))
        
        num_negative = bag_length # All patches in negative bags are negative

        neg_idx = r.choice(len(negative_patches), num_negative, replace=True)
        instances = [negative_patches[i] for i in neg_idx]

        order = r.permutation(len(instances))
        instances = [instances[i] for i in order]

        patches = []
        instance_labels = []
        for instance in instances:
            patch = fetch_patch_from_image(f'{IMAGE_PATH}/{instance["image_id"]}', instance["coords"])
            patch_tensor = transform(patch)
            patches.append(patch_tensor)
            instance_labels.append(instance['label'])

        bag_tensor = torch.stack(patches)
        dataset_manager.DatasetWriter(output_path).write(dataset_name, 
                                                    f'{split}_{i+num_pos_bags}', 
                                                    None, 
                                                    label = 0, 
                                                    patches=bag_tensor,
                                                    count = 0, 
                                                    instance_label=torch.tensor(instance_labels), 
                                                    split=split)

    print(f'Finished creating {num_bags} bags and saved them to {output_path}.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create bags for GWHD dataset')

    parser.add_argument('--output_path', type=str, default='../datasets/bags/gwhd_bags.h5', help='Path to the output H5 file for the bags')
    parser.add_argument('--dataset_name', type=str, default='gwhd_bags', help='Name of the dataset to create within the H5 file')
    parser.add_argument('--num_bags', type=int, nargs='+', default=[100,10,10], help='Total number of bags to create')
    parser.add_argument('--mean_bag_len', type=int, default=100, help='Mean number of instances per bag')
    parser.add_argument('--var_bag_len', type=int, default=10, help='Variance of the number of instances per bag')
    parser.add_argument('--bag_ratio', type=float, default=0.5, help='Ratio of positive to negative bags (default: 0.5)')
    parser.add_argument('--patch_size', type=int, default=28, help='Size of the patches to extract from the images (default: 28)')
    parser.add_argument('--overlap_threshold', type=float, default=0.5, help='Overlap threshold for labeling patches as positive (default: 0.5)')
    parser.add_argument('--grayscale', action='store_true', default=False, help='Activate grayscale images')
    parser.add_argument('--dense', action='store_true', default=False, help='Activate dense patch generation (label all patches with any overlap as positive)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)') 
    args = parser.parse_args()

    metadata = load_metadata(args.seed if args.seed is not None else 0)
    positive_patches, negative_patches = generate_patch_definitions(metadata, patch_size=args.patch_size, stride=args.patch_size, overlap_threshold=args.overlap_threshold)
    
    train_pos_index = int(len(positive_patches) * args.num_bags[0] / sum(args.num_bags))
    train_neg_index = int(len(negative_patches) * args.num_bags[0] / sum(args.num_bags))

    val_pos_index = int(len(positive_patches) * args.num_bags[1] / sum(args.num_bags))
    val_neg_index = int(len(negative_patches) * args.num_bags[1] / sum(args.num_bags))

    # Create train bags
    create_bags(args.num_bags[0], 
                args.mean_bag_len, 
                args.var_bag_len, 
                positive_patches[:train_pos_index], 
                negative_patches[:train_neg_index], 
                args.output_path, 
                args.dataset_name,
                split='train', 
                bag_ratio=args.bag_ratio, 
                seed=args.seed,
                grayscale=args.grayscale) 

    # Create validation bags  
    create_bags(args.num_bags[1], 
                args.mean_bag_len, 
                args.var_bag_len, 
                positive_patches[train_pos_index:train_pos_index+val_pos_index], 
                negative_patches[train_neg_index:train_neg_index+val_neg_index], 
                args.output_path, 
                args.dataset_name,
                split='validation', 
                bag_ratio=args.bag_ratio, 
                seed=args.seed,
                grayscale=args.grayscale)
    
    # Create test bags
    create_bags(args.num_bags[2], 
                args.mean_bag_len, 
                args.var_bag_len, 
                positive_patches[train_pos_index+val_pos_index:], 
                negative_patches[train_neg_index+val_neg_index:], 
                args.output_path, 
                args.dataset_name,
                split='test', 
                bag_ratio=args.bag_ratio, 
                seed=args.seed,
                grayscale=args.grayscale)

 

    print('All done!')