import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from collections import defaultdict
from scipy.ndimage import gaussian_filter

# ─── 1. MNIST laden & nach Label indizieren ───────────────────────────────────

mnist = datasets.MNIST(root='./data', train=True, download=True,
                       transform=transforms.ToTensor())

label_to_imgs = defaultdict(list)
for img, label in mnist:
    label_to_imgs[label].append(img.squeeze().numpy())  # (28, 28)


# ─── 2a. Potts-Modell via Gibbs Sampling ──────────────────────────────────────

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


# ─── 2b. Alternative: Gaussian Random Field (schneller) ───────────────────────

def sample_grf_grid(grid_h, grid_w, n_classes=10, smoothness=3.0):
    """
    Glatter Zufallsfeld → diskretisierte Labels.
    smoothness: Korrelationslänge (in Grid-Zellen)
    """
    # Unabhängige Felder pro Klasse, dann argmax
    fields = np.stack([
        gaussian_filter(np.random.randn(grid_h, grid_w), sigma=smoothness)
        for _ in range(n_classes)
    ])
    return np.argmax(fields, axis=0)


# ─── 3. Pseudo-Image zusammensetzen ───────────────────────────────────────────

def build_pseudo_image(label_grid, label_to_imgs, tile_size=28):
    H, W = label_grid.shape
    canvas = np.zeros((H * tile_size, W * tile_size))
    
    for i in range(H):
        for j in range(W):
            label = label_grid[i, j]
            # Zufälliges MNIST-Bild mit passendem Label
            tile = label_to_imgs[label][np.random.randint(len(label_to_imgs[label]))]
            canvas[i*tile_size:(i+1)*tile_size,
                   j*tile_size:(j+1)*tile_size] = tile
    return canvas


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
    
    return np.stack(patches), coords  # (H*W, 28, 28), [(i, j), ...]

# ─── 4. Bag-Labels für MIL ────────────────────────────────────────────────────

def get_bag_label(label_grid, target_digit):
    """Zählt Instanzen des Ziel-Digits (Regression) oder: vorhanden? (Klassifikation)"""
    count = np.sum(label_grid == target_digit)
    return count  # oder: int(count > 0) für binäre MIL


def generate_mil_dataset(n_positive_bags, n_negative_bags, target_digit, grid_size=(10,10), beta=2.0, n_sweeps=30, max_attempts=1000):
    bags, labels, grid = [], [], []

    collected = {"positive": 0, "negative": 0}
    needed = {"positive": n_positive_bags, "negative": n_negative_bags}
    attempts = 0

    while collected["positive"] < needed["positive"] or collected["negative"] < needed["negative"]:
        if attempts >= max_attempts:
            print("Maximale Anzahl an Versuchen erreicht. Generierung abgebrochen.")
            break
        attempts += 1
        
        label_grid = sample_potts_grid(*grid_size, beta=beta, n_sweeps=n_sweeps)
        bag_label = get_bag_label(label_grid, target_digit)

        if bag_label > 0 and collected["positive"] < needed["positive"]:
            bags.append(build_pseudo_image(label_grid, label_to_imgs))
            labels.append(1)  # positive Bag
            grid.append(label_grid)
            collected["positive"] += 1
        elif bag_label == 0 and collected["negative"] < needed["negative"]:
            bags.append(build_pseudo_image(label_grid, label_to_imgs))
            labels.append(0)  # negative Bag
            grid.append(label_grid)
            collected["negative"] += 1
        
        

    print(f"Generierung abgeschlossen: {collected['positive']} positive und {collected['negative']} negative Bags in {attempts} Versuchen.")
    return bags, labels, grid

# ─── 5. Beispiel ──────────────────────────────────────────────────────────────

GRID_H, GRID_W = 15, 15
TARGET_DIGIT   = 9

# Potts-Grid sampeln
label_grid = sample_potts_grid(GRID_H, GRID_W, n_classes=10, beta=1.5, n_sweeps=30)

# Pseudo-Image erzeugen
pseudo_img = build_pseudo_image(label_grid, label_to_imgs)
bag_label  = get_bag_label(label_grid, TARGET_DIGIT)

# Visualisierung
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(label_grid, cmap='tab10', vmin=0, vmax=9)
axes[0].set_title('Label-Grid (Potts-Sampling)')
axes[0].figure.colorbar(axes[0].images[0], ax=axes[0])

axes[1].imshow(pseudo_img, cmap='gray')
axes[1].set_title(f'Pseudo-Image | #{TARGET_DIGIT} Count: {bag_label}')
plt.tight_layout()
plt.show()