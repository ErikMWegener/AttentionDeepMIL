# eval/scripts/visualize_confusion_analysis.py

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def analyze_confusion_distribution(json_path, h5_path, dataset_name, output_dir='./'):
    """
    Visualisiert die Verteilung von TP/FP/TN/FN über alle Bags und Seeds.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    df = pd.read_json(json_path, orient='split')
    
    # TP/FP/TN/FN pro Bag berechnen
    tp_list, fp_list, tn_list, fn_list = [], [], [], []
    bag_counts = []
    seeds = []
    
    for idx, row in df.iterrows():
        weights = np.array(row['attention_weights']).flatten()
        threshold = float(row['count_threshold'])
        predicted = (weights > threshold).astype(int)
        
        # Echte Labels laden
        from data.data_management.dataset_manager import DatasetReader
        reader = DatasetReader(h5_path, dataset_name=dataset_name, split='test')
        _, _, _, _, instance_label = reader[int(row['bag_ids'])]
        if 'mnist' in dataset_name.lower():
            # Flatten the 2D grid to 1D
            if torch.is_tensor(instance_label):
                instance_label_flat = instance_label.flatten().numpy()
            else:
                instance_label_flat = np.array(instance_label).flatten()
            
            # Convert to binary: 1 if class 9 (target), 0 otherwise
            instance_label = (instance_label_flat == 9).astype(int)
        else:
            # For other datasets, assume instance_label is already 1D binary
            instance_label = np.array([int(l.item() if torch.is_tensor(l) else l) for l in instance_label])
        
        tp = np.sum((predicted == 1) & (instance_label == 1))
        fp = np.sum((predicted == 1) & (instance_label == 0))
        tn = np.sum((predicted == 0) & (instance_label == 0))
        fn = np.sum((predicted == 0) & (instance_label == 1))
        
        tp_list.append(tp)
        fp_list.append(fp)
        tn_list.append(tn)
        fn_list.append(fn)
        bag_counts.append(len(instance_label))
        seeds.append(row['seeds'])
    
    # ========== CONFUSION MATRIX COUNTS ==========
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(tp_list, bins=20, color='green', edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('True Positives (TP) Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Count per Bag')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].text(0.98, 0.98, f'Mean={np.mean(tp_list):.2f}\nStd={np.std(tp_list):.2f}', 
                    transform=axes[0, 0].transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[0, 1].hist(fp_list, bins=20, color='red', edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('False Positives (FP) Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Count per Bag')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].text(0.98, 0.98, f'Mean={np.mean(fp_list):.2f}\nStd={np.std(fp_list):.2f}', 
                    transform=axes[0, 1].transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1, 0].hist(fn_list, bins=20, color='orange', edgecolor='black', alpha=0.7)
    axes[1, 0].set_title('False Negatives (FN) Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Count per Bag')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].text(0.98, 0.98, f'Mean={np.mean(fn_list):.2f}\nStd={np.std(fn_list):.2f}', 
                    transform=axes[1, 0].transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    axes[1, 1].hist(tn_list, bins=20, color='gray', edgecolor='black', alpha=0.7)
    axes[1, 1].set_title('True Negatives (TN) Distribution', fontweight='bold')
    axes[1, 1].set_xlabel('Count per Bag')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].text(0.98, 0.98, f'Mean={np.mean(tn_list):.2f}\nStd={np.std(tn_list):.2f}', 
                    transform=axes[1, 1].transAxes, ha='right', va='top', 
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_distribution.png', dpi=300)
    plt.close()
    print(f"✓ Confusion distribution saved")
    
    # ========== SCATTER: FP vs FN (zeigt Error-Pattern) ==========
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(fp_list, fn_list, c=seeds, cmap='tab10', s=100, alpha=0.7, edgecolors='black')
    ax.set_xlabel('False Positives (FP)', fontsize=12)
    ax.set_ylabel('False Negatives (FN)', fontsize=12)
    ax.set_title('False Positive vs False Negative: Error Pattern Analysis', fontweight='bold', fontsize=14)
    plt.colorbar(scatter, ax=ax, label='Seed')
    ax.grid(alpha=0.3)
    
    # Diagonal: ideale Linie (FP=FN)
    max_val = max(max(fp_list), max(fn_list))
    ax.plot([0, max_val], [0, max_val], 'r--', label='FP=FN Line', linewidth=2)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fp_vs_fn_pattern.png', dpi=300)
    plt.close()
    print(f"✓ FP vs FN pattern saved")
    
    # ========== CONFUSION MATRIX HEATMAP (Aggregate) ==========
    total_tp = np.sum(tp_list)
    total_fp = np.sum(fp_list)
    total_fn = np.sum(fn_list)
    total_tn = np.sum(tn_list)
    
    confusion_matrix = np.array([
        [total_tn, total_fp],
        [total_fn, total_tp]
    ])
    
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion_matrix, cmap='Blues', aspect='auto')
    
    # Annotations
    ax.text(0, 0, f'{total_tn}', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    ax.text(1, 0, f'{total_fp}', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    ax.text(0, 1, f'{total_fn}', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    ax.text(1, 1, f'{total_tp}', ha='center', va='center', color='white', fontsize=16, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted Negative', 'Predicted Positive'], fontsize=11)
    ax.set_yticklabels(['Actual Negative', 'Actual Positive'], fontsize=11)
    ax.set_title('Aggregated Confusion Matrix (All Bags & Seeds)', fontweight='bold', fontsize=14)
    
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_heatmap.png', dpi=300)
    plt.close()
    print(f"✓ Confusion matrix heatmap saved")
    
    # ========== PRINT SUMMARY ==========
    print(f"\n{'='*60}")
    print(f"CONFUSION MATRIX ANALYSIS")
    print(f"{'='*60}")
    print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
    print(f"Specificity (TN/(TN+FP)): {total_tn/(total_tn+total_fp):.4f}")
    print(f"Sensitivity/Recall (TP/(TP+FN)): {total_tp/(total_tp+total_fn):.4f}")
    print(f"Precision (TP/(TP+FP)): {total_tp/(total_tp+total_fp) if (total_tp+total_fp) > 0 else 0:.4f}")
    print(f"{'='*60}\n")
    
    # CSV speichern
    results_df = pd.DataFrame({
        'bag_id': df['bag_ids'].values,
        'seed': df['seeds'].values,
        'tp': tp_list,
        'fp': fp_list,
        'fn': fn_list,
        'tn': tn_list,
        'bag_size': bag_counts
    })
    results_df.to_csv(f'{output_dir}/confusion_per_bag.csv', index=False)
    print(f"✓ Per-bag confusion results saved to confusion_per_bag.csv")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, required=True)
    parser.add_argument('--h5_path', type=str, default='data/datasets/bags/gwhd_bags.h5')
    parser.add_argument('--dataset_name', type=str, default='gwhd_bags_dense')
    parser.add_argument('--output_dir', type=str, default='./confusion_analysis')
    
    args = parser.parse_args()
    analyze_confusion_distribution(args.json_path, args.h5_path, args.dataset_name, args.output_dir)
    print(f"✓ Analysis complete! Check {args.output_dir}/ for results.")