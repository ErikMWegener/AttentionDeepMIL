import os

import numpy as np
from PIL import Image
import pandas as pd

import torch 
from torchvision import transforms
from sklearn.linear_model import LinearRegression 
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from scipy import stats


# Create a transformation pipeline for the image patches identical to the one used durgin bag creation.
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

def full_image_count(image_path, patch_size, model, grayscale=True, threshold=None):
    """
    Read the image, split it into a regular grid of patches
    and get the per patch signals for further evaluation.

    Args:
        image_path: Pfad des vollen Bildes (jedes von PIL lesbare Format).
        patch_size: Kantenlaenge der Patches; Stride ist identisch (kein Overlap).
        model: CLAM-Modell mit ``count_positive_instances``.
        grayscale: muss zur Transform-Pipeline der Bag-Erzeugung passen.
        threshold: Zaehl-Threshold. None nutzt den (ggf. kalibrierten) Wert
            aus dem ``count_threshold``-Buffer des Modells.

    Returns:
        (count, inst_probs, patch_coords_list) mit inst_probs als 1-D numpy array.
    """
    image = Image.open(image_path)
    img_w, img_h = image.size
    transform = build_transform(grayscale)
    patches = []
    patch_coords_list = []
    # Loop over a regular grid of patches in the image
    for y in range(0, img_h - patch_size + 1, patch_size):
        for x in range(0, img_w - patch_size + 1, patch_size):
            patch_coords = (x, y, x + patch_size, y + patch_size)
            patch = image.crop(patch_coords)
            patch_tensor = transform(patch)
            patches.append(patch_tensor)
            patch_coords_list.append(patch_coords)

    if not patches:
        raise ValueError(f'{image_path}: {img_w}x{img_h} ist kleiner als patch_size={patch_size}.')

    # Auf das Device des Modells schieben, sonst bricht der Forward-Pass auf der GPU ab.
    bag_tensor = torch.stack(patches).to(next(model.parameters()).device)

    count, inst_probs = model.count_positive_instances(bag_tensor.unsqueeze(0), threshold=threshold)

    return count, inst_probs.detach().cpu().numpy(), patch_coords_list

def load_point_annotations(annotation_path):
    """
    Load point annotations from a text file.
    Each line in the file should contain two numbers (x, y) representing the coordinates of a point.
    """
    points = np.loadtxt(annotation_path, ndmin=2)   # default delimiter splits on any whitespace
    return points


def points_per_patch(points, patch_coords):
    """
    Takes arrays points and patch coords 
    Calculates the number of points per patch
    Counts the points that fall outside the grid
    Does not work for a stride other than patch_size
    """
    ppp = np.zeros(len(patch_coords))
    patch_size = patch_coords[0][2]-patch_coords[0][0]
    grid_w, grid_h  = patch_coords[-1][2], patch_coords[-1][3]
    cols, rows = grid_w//patch_size, grid_h//patch_size

    out_of_grid = 0

    for x,y in points:
        col = int(x // patch_size)
        row = int(y // patch_size)

        if not (0 <= col < cols and 0 <= row < rows):
            out_of_grid += 1
            continue

        ppp[row * cols + col] += 1

    return ppp, out_of_grid

def aggregate(inst_probs, count, threshold=None):
    """
    Aggregates the patch signals to a hard_count (threshold) and soft_count (sum).

    Returns:
        (hard_count, soft_count, n_patches) als int/float/int, damit die Werte
        ohne Umwege in einen DataFrame und nach MLflow wandern koennen.
    """
    if threshold is not None:
        count = int((inst_probs > threshold).sum())
    soft_count = float(inst_probs.sum())
    return int(count), soft_count, int(len(inst_probs))

def evaluate_image_set(model, image_ids, path, patch_size, grayscale=True, threshold=None,
                       img_ext='.JPG', ann_ext='.txt'):
    """
    Wertet einen Satz voller Bilder aus und sammelt die Ergebnisse in einem DataFrame.

    Erwartet Bild und Annotation im selben Verzeichnis unter gleichem Stamm,
    z.B. ``<path>/<image_id>.JPG`` und ``<path>/<image_id>.txt``.

    Hinweis zu den Zaehlspalten: ``gt_count`` zaehlt nur Punkte innerhalb des
    Patch-Rasters (das ist der faire Vergleich, das Modell sieht den Randstreifen
    rechts/unten nie), ``gt_total`` alle annotierten Punkte.
    """
    rows = []
    for image_id in image_ids:
        hard_count, inst_probs, coords = full_image_count(
            os.path.join(path, image_id + img_ext), patch_size, model, grayscale, threshold)
        hard_count, soft_count, n_patches = aggregate(inst_probs, hard_count, threshold)
        points = load_point_annotations(os.path.join(path, image_id + ann_ext))
        ppp, out_of_grid = points_per_patch(points, coords)
        rows.append({
            'image_id': image_id,
            'hard_count': hard_count,
            'soft_count': soft_count,
            'gt_count': int(ppp.sum()),
            'gt_total': int(len(points)),
            'n_patches': n_patches,
            'out_of_grid': out_of_grid,
            'patch_signals': inst_probs,
            'points_per_patch': ppp,
            'hard_frac': hard_count / n_patches if n_patches > 0 else float('nan'),
            'soft_frac': soft_count / n_patches if n_patches > 0 else float('nan'),
        })
    return pd.DataFrame(rows)

def fit_calibration(pred_counts, gt_counts, kind='linear'):
    """
    Fit a calibration mapping predicted counts -> ground truth counts.

    Returns:
        Callable f(pred) -> kalibrierte Counts, nimmt 1-D Input und liefert
        1-D Output (fuer beide kinds identisch). Der gefittete Estimator bleibt
        als ``f.estimator`` erreichbar, die Art als ``f.kind``.
    """
    p = np.asarray(pred_counts, dtype=float).ravel()
    g = np.asarray(gt_counts, dtype=float).ravel()

    if kind == 'linear':
        est = LinearRegression().fit(p.reshape(-1, 1), g)
        cal_fn = lambda x: est.predict(np.asarray(x, dtype=float).reshape(-1, 1)).ravel()
    elif kind == 'isotonic':
        est = IsotonicRegression(increasing=True, out_of_bounds='clip').fit(p, g)
        cal_fn = lambda x: est.predict(np.asarray(x, dtype=float).ravel())
    else:
        raise ValueError("Invalid kind. Choose 'linear' or 'isotonic'.")

    cal_fn.estimator = est
    cal_fn.kind = kind
    return cal_fn

def plot_pred_vs_gt(df, count_col='hard_count', gt_col='gt_count', cal_fn=None, frac_col='hard_frac', title=None):

    c = np.asarray(df[count_col], dtype=float).ravel()
    g = np.asarray(df[gt_col], dtype=float).ravel()

    if c.size < 3:
        raise ValueError("Not enough data points to plot. Need at least 3.")

    # -- Correlation

    rho, rho_p = stats.spearmanr(c, g)
    r, _ = stats.pearsonr(c, g)

    n_panels = 2 if cal_fn is not None else 1
    fig, axs = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    axs = np.atleast_1d(axs)
    ax = axs[0]

    # -- Panel 1: Scatter plot of predicted vs ground truth counts

    use_frac = frac_col is not None and frac_col in getattr(df, 'columns', [])
    if use_frac:
        frac = np.asarray(df[frac_col], dtype=float).ravel()
        sc = ax.scatter(c, g, c=frac, cmap='viridis', vmin=0, vmax=1, s=60, edgecolor='k', linewidth=0.5, zorder=3)
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Proportion of positive patches")

    else:
        ax.scatter(c, g, color='blue', s=60, edgecolor='k', linewidth=0.5, zorder=3)

    if cal_fn is not None:
        grid = np.linspace(c.min(), c.max(), 200)
        ax.plot(grid, np.asarray(cal_fn(grid)).ravel(), color='red', lw=2, zorder=2, label="Calibration f(count)")
        ax.legend(loc='upper left', fontsize=9)

    ax.set_xlabel(f"Aggregated predicted count ({count_col})")
    ax.set_ylabel(f"Ground truth count ({gt_col})")
    ax.set_title(f"Spearman rho={rho:.3f} (p={rho_p:.3f}), Pearson r={r:.3f} | n={c.size}", fontsize=10)
    ax.grid(alpha=0.3)

    # -- Panel 2: Residuals (only with calibration)
    if cal_fn is not None:
        pred = np.asarray(cal_fn(c), dtype=float).ravel()
        resid = pred - g
        mae = np.abs(resid).mean()
        bias = resid.mean()

        ax2 = axs[1]
        ax2.axhline(0, color='k', lw=1, zorder=1)
        ax2.scatter(g, resid, color="tab:orange", s=60, edgecolor='k', linewidth=0.5, zorder=3)

        if np.ptp(g) > 0:
            a, b = np.polyfit(g, resid, 1)
            grid_y = np.linspace(g.min(), g.max(), 100)
            ax2.plot(grid_y, a * grid_y + b, color='tab:red', ls="--", lw=1.5, zorder=2, label=f"Trend line: y={a:.3f}x+{b:.3f}")
            ax2.legend(loc='upper right', fontsize=9)

        ax2.set_xlabel(f"Ground truth count ({gt_col})")
        ax2.set_ylabel("Residuals (predicted - gt)")
        ax2.set_title(f"MAE={mae:.3f}| Bias={bias:.3f}", fontsize=10)
        ax2.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout()

    return fig

