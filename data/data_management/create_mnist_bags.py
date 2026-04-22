from collections import defaultdict

import torch
import numpy as np
from torchvision import datasets, transforms
import dataset_manager

# Create bags from MNIST dataset with set bag length and target number.     
# The bags are formed such that they contain at least one third does not contain instance of the target number.
# Number of instances in each bag must fit into a quadratical grid to form a pseudo image. 
# The bags are stored in a H5 file with the bag label (1 if it contains the target number, 0 otherwise),  
# the indices of the instances in the original dataset and the coordinates of the instances in the pseudo image.

def form_bags_to_h5(path, target_number, pseudo_image_width, pseudo_image_height, num_bag, stride = 28, seed=0, train=True):
    if train:
        loader = datasets.MNIST('../datasets',
                                train=True,
                                download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))
    else:
        loader = datasets.MNIST('../datasets',
                                train=False,
                                download=True,
                                transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))]))

    label_to_imgs = defaultdict(list)
    for img, label in loader:
        label_to_imgs[label].append(img.squeeze().numpy())  # (28, 28)

    # Creating bags based on Potts-Model sampling of the label grid and building pseudo images from the sampled grids.
    # The bags are labeled as positive if they contain at least one instance of the target number, otherwise they are labeled as negative.
    # The bags are stored in a list along with their labels and the indices of the instances in the original dataset.
    # The bags are created until the desired number of bags is reached or the maximum number of attempts is exceeded to avoid infinite loops.

    collected = {"positive": 0, "negative": 0}
    needed = {"positive": num_bag // 2, "negative": num_bag // 2}
    attempts = 0
    max_attempts = num_bag * 10

    while collected ["positive"] < needed["positive"] or collected["negative"] < needed["negative"]:
        if attempts >= max_attempts:
            print("Reached maximum number of attempts. Stopping generation.")
            break
        attempts += 1
        beta = 2.0
        n_sweeps = 30
        label_grid = sample_potts_grid(pseudo_image_width, pseudo_image_height, beta=beta, n_sweeps=n_sweeps)
        count = get_bag_label(label_grid, target_number)

        bag, coords = [], []

        if count > 0 and collected["positive"] < needed["positive"]:
            bag, coords = build_patch_tensor(label_grid, label_to_imgs)
            collected["positive"] += 1
            sum_collected = collected["positive"] + collected["negative"]
            dataset_manager.DatasetWriter(path).write('mnist_bags', 
                                                    f'train_{sum_collected}' if train else f'test_{sum_collected}', 
                                                    torch.tensor(coords), 
                                                    label = 1 if count > 0 else 0, 
                                                    patches=torch.tensor(bag),
                                                    count = count, 
                                                    instance_label=label_grid, split='train' if train else 'test')
        elif count == 0 and collected["negative"] < needed["negative"]:
            bag, coords = build_patch_tensor(label_grid, label_to_imgs)
            collected["negative"] += 1
            sum_collected = collected["positive"] + collected["negative"]
            dataset_manager.DatasetWriter(path).write('mnist_bags', 
                                                    f'train_{sum_collected}' if train else f'test_{sum_collected}', 
                                                    torch.tensor(coords), 
                                                    label = 0, 
                                                    patches=torch.tensor(bag),
                                                    count = count, 
                                                    instance_label=label_grid, split='train' if train else 'test')
        else:
            pass  

def sample_potts_grid(grid_h, grid_w, n_classes=10, beta=2.0, n_sweeps=50):
    """
    Gibbs-Sampling für das Potts-Modell.
    
    beta: Clustering-Stärke
      β=0  → zufällig (keine Korrelation)
      β=1  → moderate Cluster
      β=2+ → starke Cluster (Ähren-ähnlich)
    """
    grid = np.random.randint(0, n_classes, (grid_h, grid_w))
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for sweep in range(n_sweeps):
        # Zufällige Reihenfolge für ergodisches Sampling
        order = np.random.permutation(grid_h * grid_w)
        for idx in order:
            i, j = divmod(idx, grid_w)
            
            # Nachbar-Klassenzählung
            counts = np.zeros(n_classes)
            for di, dj in neighbors:
                ni, nj = i + di, j + dj
                if 0 <= ni < grid_h and 0 <= nj < grid_w:
                    counts[grid[ni, nj]] += 1
            
            # Potts-Energie: P(k) ∝ exp(β * Anzahl_Nachbarn_mit_k)
            log_probs = beta * counts
            log_probs -= log_probs.max()  # numerische Stabilität
            probs = np.exp(log_probs)
            probs /= probs.sum()
            
            grid[i, j] = np.random.choice(n_classes, p=probs)
    
    return grid

def build_patch_tensor(label_grid, label_to_imgs, tile_size=28):
    H, W = label_grid.shape
    patches = []
    coords = []
    
    for i in range(H):
        for j in range(W):
            label = label_grid[i, j]
            tile = label_to_imgs[label][np.random.randint(len(label_to_imgs[label]))]
            patches.append(tile)
            coords.append((i * tile_size, j * tile_size))  # Top-Left-Koordinate des Patches
    
    patches = np.stack(patches)  # (H*W, 28, 28)
    patches = np.expand_dims(patches, axis=1)  # (H*W, 1, 28, 28) für CNNs

    return patches, coords  # (H*W, 28, 28), [(i, j), ...]

def get_bag_label(label_grid, target_digit):
    """Zählt Instanzen des Ziel-Digits (Regression) oder: vorhanden? (Klassifikation)"""
    count = np.sum(label_grid == target_digit)
    return count  # oder: int(count > 0) für binäre MIL

if __name__ == "__main__":
    # Argument parser for command line execution of form_bags function with parameters for target number, pseudo image dimensions, number of bags, stride and seed.
    import argparse
    parser = argparse.ArgumentParser(description='Generate MNIST bags for MIL.')
    parser.add_argument('--path', type=str, default='mnist_bags.h5', help='Path to save the generated bags (default: mnist_bags.h5)')
    parser.add_argument('--target_number', type=int, default=9, help='Target number for positive bags (default: 9)')
    parser.add_argument('--pseudo_image_width', type=int, default=10, help='Width of the pseudo image (default: 10)')
    parser.add_argument('--pseudo_image_height', type=int, default=10, help='Height of the pseudo image (default: 10)')
    parser.add_argument('--num_bag', type=int, default=100, help='Total number of bags to generate (default: 100)')
    parser.add_argument('--stride', type=int, default=28, help='Stride for placing instances in the pseudo  image (default: 28)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility (default: 0)')
    args = parser.parse_args() 

    form_bags_to_h5(args.path, args.target_number, args.pseudo_image_width, args.pseudo_image_height, args.num_bag, args.stride, args.seed, train=True) 
    form_bags_to_h5(args.path, args.target_number, args.pseudo_image_width, args.pseudo_image_height, args.num_bag // 10, args.stride, args.seed, train=False)