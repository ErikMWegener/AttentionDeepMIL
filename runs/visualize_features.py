"""
visualize_features.py
Modul zur Visualisierung von MIL Feature-Vektoren (H) und Attention-Gewichten (A).
Kann standalone oder als Import in main_mlflow.py verwendet werden.

DataLoader-Format erwartet: (patches, coords, label, count, instance_label)
Modell benötigt: extract_features(x) → (H_numpy, A_numpy)

Cluster-Hinweis: Vor dem Start MPLBACKEND=Agg setzen, z.B.:
    export MPLBACKEND=Agg
"""

import numpy as np
import matplotlib.pyplot as plt
import torch

from sklearn.manifold import TSNE

try:
    import umap as umap_lib
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


# ── Feature-Extraktion ───────────────────────────────────────────────────────

def collect_features_from_loader(model, loader, max_bags=None, device="cpu", target_digit=9):
    """
    Sammelt H-Vektoren, Attention-Gewichte, Bag- und Instanz-Labels aus dem DataLoader.

    Args:
        model:        Trainiertes Modell mit extract_features(x)-Methode.
        loader:       DataLoader mit Format (patches, coords, label, count, instance_label).
        max_bags:     Maximale Anzahl zu verarbeitender Bags.
        device:       Torch-Device ('cpu' oder 'cuda').
        target_digit: Klasse, die als positive Instanz gilt (für Multi-Class-Labels).

    Returns:
        H_all:           np.ndarray [N, M]  – Feature-Vektoren
        A_all:           np.ndarray [N]     – Attention-Gewichte (gemittelt über Branches)
        bag_labels:      np.ndarray [N]     – Bag-Label (0/1) pro Instanz
        instance_labels: np.ndarray [N]     – binarisiertes Instanz-Label (0/1)
        bag_ids:         np.ndarray [N]     – Bag-Index pro Instanz
    """
    model.eval()
    H_list, A_list = [], []
    bag_label_list, inst_label_list, bag_id_list = [], [], []

    with torch.no_grad():
        for bag_idx, (patches, coords, label, count, instance_label) in enumerate(loader):
            if max_bags is not None and bag_idx >= max_bags:
                break

            patches = patches.squeeze(0).to(device)  # [K, C, H, W]
            H, A = model.extract_features(patches)   # [K, M], [K] als numpy-Arrays
            K = H.shape[0]

            # Instanz-Labels binarisieren falls Multi-Class
            inst = instance_label.cpu().numpy().flatten()
            if len(set(inst.tolist())) > 2:
                inst = (inst == target_digit).astype(int)
            else:
                inst = inst.astype(int)

            # Längen angleichen (instance_label kann abweichen)
            min_len = min(K, len(inst))

            H_list.append(H[:min_len])
            A_list.append(A[:min_len])
            bag_label_list.append(np.full(min_len, int(label.item())))
            inst_label_list.append(inst[:min_len])
            bag_id_list.append(np.full(min_len, bag_idx))

    if not H_list:
        raise RuntimeError("Keine Features gesammelt – DataLoader leer oder max_bags=0.")

    return (
        np.concatenate(H_list),
        np.concatenate(A_list),
        np.concatenate(bag_label_list),
        np.concatenate(inst_label_list),
        np.concatenate(bag_id_list),
    )


# ── Dimensionsreduktion ──────────────────────────────────────────────────────

def reduce_dimensions(H, method="umap"):
    """
    Reduziert H auf 2D via t-SNE oder UMAP.

    Args:
        H:      np.ndarray [N, M]
        method: 'tsne' oder 'umap'

    Returns:
        np.ndarray [N, 2]
    """
    n = H.shape[0]
    print(f"  Dimensionsreduktion: {method.upper()} auf {n} Instanzen...")

    if method == "umap":
        if HAS_UMAP:
            reducer = umap_lib.UMAP(n_components=2, random_state=42, verbose=False)
            return reducer.fit_transform(H)
        else:
            print("  UMAP nicht installiert (pip install umap-learn), Fallback auf t-SNE.")

    # t-SNE: perplexity muss < n sein
    perplexity = min(30, max(5, n // 10))
    reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    return reducer.fit_transform(H)


# ── Plot-Funktionen ──────────────────────────────────────────────────────────

def plot_per_seed(H_2d, A_all, bag_labels, instance_labels, title_suffix=""):
    """
    Drei-Panel-Visualisierung für einen einzelnen Seed:
      1. Nach Bag-Label (Pos./Neg. Bag)
      2. Nach Instanz-Label Ground Truth (ermöglicht Diagnose der Feature-Diskriminierbarkeit)
      3. Positive Bags: Punktgröße ∝ Attention-Gewicht

    Args:
        H_2d:            np.ndarray [N, 2]  – reduzierte Features
        A_all:           np.ndarray [N]     – Attention-Gewichte
        bag_labels:      np.ndarray [N]     – Bag-Labels (0/1)
        instance_labels: np.ndarray [N]     – Instanz-Labels (0/1)
        title_suffix:    String für den Titel (z.B. "Seed 42")

    Returns:
        matplotlib.Figure – Caller ist für plt.close(fig) verantwortlich.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    if title_suffix:
        fig.suptitle(f"Feature-Visualisierung – {title_suffix}", fontsize=13, y=1.02)

    # ── Panel 1: Bag-Label ───────────────────────────────────────────────────
    ax = axes[0]
    for lbl, color, name in zip([0, 1], ["#4477AA", "#EE6677"], ["Neg. Bag", "Pos. Bag"]):
        mask = bag_labels == lbl
        if mask.any():
            ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                       c=color, s=6, alpha=0.5, label=name, rasterized=True)
    ax.set_title("Nach Bag-Label")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 2: Instanz-Label (Ground Truth) ────────────────────────────────
    # Wenn Pos./Neg. hier nicht trennbar → Feature-Extraktor ist das Bottleneck
    ax = axes[1]
    for lbl, color, name in zip([0, 1], ["#4477AA", "#EE6677"],
                                 ["Neg. Instanz", "Pos. Instanz"]):
        mask = instance_labels == lbl
        if mask.any():
            ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                       c=color, s=6, alpha=0.5, label=name, rasterized=True)
    ax.set_title("Nach Instanz-Label (GT)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 3: Attention-Gewicht in positiven Bags ─────────────────────────
    # Wenn hoch-attended Instanzen nicht mit Pos.-Cluster übereinstimmen → Attention falsch
    ax = axes[2]
    neg_mask = bag_labels == 0
    pos_mask = bag_labels == 1
    if neg_mask.any():
        ax.scatter(H_2d[neg_mask, 0], H_2d[neg_mask, 1],
                   c="#CCCCCC", s=4, alpha=0.3, label="Neg. Bags", rasterized=True)
    if pos_mask.any():
        a = A_all[pos_mask]
        sizes = 5 + 80 * (a - a.min()) / (a.max() - a.min() + 1e-8)
        sc = ax.scatter(H_2d[pos_mask, 0], H_2d[pos_mask, 1],
                        c=a, cmap="hot", s=sizes, alpha=0.7,
                        label="Pos. Bags", rasterized=True)
        plt.colorbar(sc, ax=ax, label="Attention-Gewicht")
    ax.set_title("Pos. Bags: Größe = Attention")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    return fig


def plot_aggregated(H_2d, A_all, bag_labels, instance_labels, seed_ids):
    """
    Vier-Panel-Visualisierung aggregiert über alle Seeds.
    Zusätzliches Panel zeigt Seed-Zugehörigkeit (prüft Reproduzierbarkeit der Features).

    Args:
        H_2d:            np.ndarray [N, 2]
        A_all:           np.ndarray [N]
        bag_labels:      np.ndarray [N]
        instance_labels: np.ndarray [N]
        seed_ids:        np.ndarray [N]  – welcher Seed jede Instanz erzeugt hat

    Returns:
        matplotlib.Figure – Caller ist für plt.close(fig) verantwortlich.
    """
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    fig.suptitle("Feature-Visualisierung – Aggregiert über alle Seeds", fontsize=13, y=1.02)

    # ── Panel 1: Bag-Label ───────────────────────────────────────────────────
    ax = axes[0]
    for lbl, color, name in zip([0, 1], ["#4477AA", "#EE6677"], ["Neg. Bag", "Pos. Bag"]):
        mask = bag_labels == lbl
        if mask.any():
            ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                       c=color, s=4, alpha=0.4, label=name, rasterized=True)
    ax.set_title("Nach Bag-Label")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 2: Instanz-Label (Ground Truth) ────────────────────────────────
    ax = axes[1]
    for lbl, color, name in zip([0, 1], ["#4477AA", "#EE6677"],
                                 ["Neg. Instanz", "Pos. Instanz"]):
        mask = instance_labels == lbl
        if mask.any():
            ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                       c=color, s=4, alpha=0.4, label=name, rasterized=True)
    ax.set_title("Nach Instanz-Label (GT)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 3: Attention-Gewicht (Pos. Bags) ───────────────────────────────
    ax = axes[2]
    neg_mask = bag_labels == 0
    pos_mask = bag_labels == 1
    if neg_mask.any():
        ax.scatter(H_2d[neg_mask, 0], H_2d[neg_mask, 1],
                   c="#CCCCCC", s=3, alpha=0.3, label="Neg. Bags", rasterized=True)
    if pos_mask.any():
        a = A_all[pos_mask]
        sizes = 3 + 50 * (a - a.min()) / (a.max() - a.min() + 1e-8)
        sc = ax.scatter(H_2d[pos_mask, 0], H_2d[pos_mask, 1],
                        c=a, cmap="hot", s=sizes, alpha=0.6, rasterized=True)
        plt.colorbar(sc, ax=ax, label="Attention-Gewicht")
    ax.set_title("Attention-Gewicht (Pos. Bags)")
    ax.legend(markerscale=2, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    # ── Panel 4: Seed-Zugehörigkeit ──────────────────────────────────────────
    # Wenn Seeds stark separate Cluster bilden → Instabilität des Trainings
    ax = axes[3]
    unique_seeds = np.unique(seed_ids).astype(int)
    cmap = plt.cm.get_cmap("tab10", max(len(unique_seeds), 1))
    for i, s in enumerate(unique_seeds):
        mask = seed_ids == s
        ax.scatter(H_2d[mask, 0], H_2d[mask, 1],
                   color=cmap(i), s=4, alpha=0.4, label=f"Seed {s}", rasterized=True)
    ax.set_title("Nach Seed (Stabilitätscheck)")
    ax.legend(markerscale=3, fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    return fig


# ── Standalone-Verwendung ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    import torch.utils.data as data_utils
    from data.data_management.dataset_manager import DatasetReader
    from models.model import Attention, GatedAttention

    parser = argparse.ArgumentParser(description="Standalone Feature-Visualisierung für MIL-Modelle")
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Pfad zu model checkpoint (.pth)')
    parser.add_argument('--path', type=str, required=True,
                        help='Pfad zur H5-Datei')
    parser.add_argument('--dataset', type=str, default='mnist_bags')
    parser.add_argument('--model', type=str, default='attention',
                        choices=['attention', 'gated_attention'])
    parser.add_argument('--method', type=str, default='tsne', choices=['tsne', 'umap'])
    parser.add_argument('--max_bags', type=int, default=150)
    parser.add_argument('--target_digit', type=int, default=9)
    parser.add_argument('--output', type=str, default='feature_visualization.pdf')
    parser.add_argument('--no-cuda', action='store_true', default=False)
    sa = parser.parse_args()

    device = "cuda" if not sa.no_cuda and torch.cuda.is_available() else "cpu"

    ModelClass = Attention if sa.model == "attention" else GatedAttention
    model = ModelClass().to(device)
    model.load_state_dict(torch.load(sa.checkpoint, map_location=device))
    print(f"Modell geladen: {sa.checkpoint}")

    loader = data_utils.DataLoader(
        DatasetReader(sa.path, dataset_name=sa.dataset, split='test'),
        batch_size=1, shuffle=False
    )

    H, A, bag_lbls, inst_lbls, _ = collect_features_from_loader(
        model, loader, max_bags=sa.max_bags, device=device, target_digit=sa.target_digit
    )
    H_2d = reduce_dimensions(H, method=sa.method)
    fig = plot_per_seed(H_2d, A, bag_lbls, inst_lbls, title_suffix="Standalone")
    fig.savefig(sa.output, dpi=150, bbox_inches="tight")
    print(f"Gespeichert: {sa.output}")
    plt.close(fig)