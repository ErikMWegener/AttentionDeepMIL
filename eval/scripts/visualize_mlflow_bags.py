import json
import math
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.data_management.dataset_manager import DatasetReader

def visualize_attention_weights(json_path, h5_path, dataset_name, seed=1, bag_id=1, softmax = False):

    df = pd.read_json(json_path, orient='split')
    
    bag = df[(df["seeds"] == seed) & (df["bag_ids"] == bag_id)]
    if bag.empty:
        print(f"Keine Daten für Seed {seed} und Bag ID {bag_id} gefunden.")
        return

    print(bag)
    print(f"Lade Bag {bag_id} aus dem Test-Split von {h5_path}...")
    reader = DatasetReader(h5_path, dataset_name=dataset_name, split='test')

    patches, coords, label, count, instance_label = reader[bag_id]

    instance_labels = []
    if 'mnist' in dataset_name.lower():
        # Flatten the 2D grid to 1D
        if torch.is_tensor(instance_label):
            instance_label_flat = instance_label.flatten().numpy()
        else:
            instance_label_flat = np.array(instance_label).flatten()
        
        # Convert to binary: 1 if class 9 (target), 0 otherwise
        instance_labels = (instance_label_flat == 9).astype(int)
    else:
        # For other datasets, assume instance_label is already 1D binary
        instance_labels = np.array([int(l.item() if torch.is_tensor(l) else l) for l in instance_label])
       

    wheights_np = np.array(bag["attention_weights"].values[0]).flatten()

    K = len(patches) # Anzahl der Patches
    if len(wheights_np) != K:
        print("Warnung: Anzahl der Patches stimmt nicht mit der Anzahl der Attention-Gewichte überein!")
    
    threshold = bag["count_threshold"].values[0] 
    if softmax:
        threshold = 1.0 / K  # Bei Softmax ist die natürliche Schwelle für "positiv" 1/K

    grid_size = math.ceil(math.sqrt(K))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for i in range(K):
        ax = axes[i]

        # Patch formatieren: [C, H, W] -> [H, W, C]
        img = patches[i].numpy()
        if img.shape[0] in [1, 3]:  
            img = img.transpose(1, 2, 0)
        
        # Farbbild / Graustufen unterscheiden
        if img.shape[-1] == 1:
            img = img.squeeze(-1)
            ax.imshow(img, cmap='gray')
        else:
            img = (img - img.min()) / (img.max() - img.min() + 1e-5)
            ax.imshow(img)

        # Bewertungslogik anwenden
        weight = wheights_np[i]
        is_counted = weight > threshold
        
        # Ground Truth Label extrahieren (falls als Tensor vorliegend)
        gt_label = instance_labels[i] if i < len(instance_labels) else 0  # Fallback zu 0, falls Labels fehlen
        # Farb- und Stil-Logik (TP, FP, TN, FN)
        if is_counted and gt_label == 1:
            # True Positive
            color = '#00FF00'  # Starkes Grün
            linewidth = 4
            linestyle = 'solid'
            status_text = "TP"
        elif is_counted and gt_label == 0:
            # False Positive
            color = '#FF00FF'  # Magenta (leichter von Rot zu unterscheiden)
            linewidth = 4
            linestyle = 'solid'
            status_text = "FP"
        elif not is_counted and gt_label == 1:
            # False Negative
            color = '#FF8C00'  # Orange
            linewidth = 3
            linestyle = 'dashed'
            status_text = "FN"
        else:
            # True Negative
            color = '#404040'  # Dunkelgrau
            linewidth = 2
            linestyle = 'dotted'
            status_text = "TN"

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(linewidth)
            spine.set_linestyle(linestyle)
            
        ax.set_title(f"A: {weight:.4f} | {status_text}", color=color, fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(K, len(axes)):
        axes[j].axis('off')
        
    # Titel erweitern um die True-Positive Rate info etc. (Optional, aber hilfreich)
    predicted_count = sum(w > threshold for w in wheights_np)
    actual_positive_instances = sum(int(il.item() if torch.is_tensor(il) else il) for il in instance_labels)
    
    fig.suptitle(
        f"Bag ID: {bag['bag_ids'].values[0]} | Seed: {seed} | Threshold: {threshold:.4f}\n"
        f"Bag Label (Truth/Pred): {bag['truth'].values[0]} / {bag['predicted'].values[0]} | "
        f"Count (Truth/Pred): {bag['count_truth'].values[0]} / {predicted_count}\n"
        f"Echte positive Patches (GT): {actual_positive_instances} (Grün/Orange)", 
        fontsize=14
    )
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # --- ANPASSEN ---
    # Pfad zur JSON-Datei aus deinem spezifischen MLflow Run (z.B. in mlruns/...)
    JSON_FILE = "/home/erik/AttentionDeepMIL/runs/mlruns/0/889712d626f04eaa8f1285dbe8a9b80f/artifacts/aggregated_run_results.json" 
    
    # Pfad zu deinem Original-Datensatz
    H5_FILE = "../../data/datasets/bags/mnist_bags.h5" 
    DATASET_NAME = "mnist_bags_beta2.0"
    # ----------------
    
    if os.path.exists(JSON_FILE):
        visualize_attention_weights(JSON_FILE, H5_FILE, dataset_name=DATASET_NAME, seed=1, bag_id=0, softmax=False)
    else:
        print(f"JSON-Datei nicht gefunden: {JSON_FILE}")