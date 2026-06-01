# eval/scripts/analyze_attention_distribution.py

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data.data_management.dataset_manager import DatasetReader


def load_attention_data(json_path, h5_path, dataset_name, split='test'):
    """
    Lädt Attention-Gewichte und Instance-Labels aus den geloggten Daten.
    
    Returns:
        attention_weights: Array aller Attention-Gewichte (flach)
        instance_labels: Array aller Ground-Truth Labels (0/1)
        bag_info: Dict mit Info zu welchen Patches zu welcher Bag gehören
    """
    df = pd.read_json(json_path, orient='split')
    
    reader = DatasetReader(h5_path, dataset_name=dataset_name, split=split)

    instanz_labels_all = []
    for  i in range(df.shape[0]):
        # Attention-Gewichte auspacken
        _,_,_,_,instance_label = reader[df['bag_ids'].iloc[i]]
        labels = np.array([int(l.item() if torch.is_tensor(l) else l) for l in instance_label])
        instanz_labels_all.append(labels)

    df['instance_labels'] = instanz_labels_all
    df = df[df['predicted'] == 1]  # Nur positive Bags behalten, da nur dort Attention-Gewichte relevant sind
    return df


def analyze_attention_distribution(df, output_dir='./'):
    """
    Erstellt Visualisierungen und berechnet Diskriminierungs-Metriken.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    all_weights = []
    all_labels = []
    for idx, row in df.iterrows():
        weights = np.array(row['attention_weights']).flatten()
        labels = np.array(row['instance_labels']).flatten()
        all_weights.extend(weights)
        all_labels.extend(labels)

    all_weights = np.array(all_weights)
    all_labels = np.array(all_labels)

    positive_weights = all_weights[all_labels == 1]
    negative_weights = all_weights[all_labels == 0]
    
    print(f"\nAttention Weight Statistics:")
    print(f"Positive Patches (n={len(positive_weights)}):")
    print(f"  Mean: {positive_weights.mean():.6f}, Std: {positive_weights.std():.6f}")
    print(f"  Median: {np.median(positive_weights):.6f}")
    print(f"  Min: {positive_weights.min():.6f}, Max: {positive_weights.max():.6f}")
    print(f"Negative Patches (n={len(negative_weights)}):")
    print(f"  Mean: {negative_weights.mean():.6f}, Std: {negative_weights.std():.6f}")
    print(f"  Median: {np.median(negative_weights):.6f}")
    print(f"  Min: {negative_weights.min():.6f}, Max: {negative_weights.max():.6f}")
    
    # ========== HISTOGRAM ==========
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bins = np.linspace(0, max(positive_weights.max(), negative_weights.max()), 50)
    ax.hist(positive_weights, bins=bins, alpha=0.6, label=f'Positive Patches (n={len(positive_weights)})', 
            color='green', edgecolor='black')
    ax.hist(negative_weights, bins=bins, alpha=0.6, label=f'Negative Patches (n={len(negative_weights)})', 
            color='red', edgecolor='black')
    
    ax.set_xlabel('Attention Weight', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Attention Weights (Positive vs. Negative Patches)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_distribution_histogram.png', dpi=300)
    plt.close()
    print(f"\n✓ Histogram saved to {output_dir}/attention_distribution_histogram.png")
    
    # ========== ROC CURVE ==========
    fpr, tpr, thresholds_roc = roc_curve(all_labels, all_weights)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(fpr, tpr, color='darkblue', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', label='Random Classifier (AUC = 0.5000)')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curve: Can Attention Weights Discriminate Patches?', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/roc_curve.png', dpi=300)
    plt.close()
    print(f"✓ ROC curve saved to {output_dir}/roc_curve.png")
    
    # ========== THRESHOLD ANALYSIS ==========

    precision_scores = []
    recall_scores = []
    f1_scores = []
    all_predictions = []
    results_list = []

    for i in range(df.shape[0]):
        weights_np = np.array(df['attention_weights'].iloc[i]).flatten()
        instance_label = np.array(df['instance_labels'].iloc[i]).flatten()

        threshold = float(df["count_threshold"].iloc[i])

        predicted = (weights_np > threshold).astype(int)
        all_predictions.extend(predicted)  # Flach sammeln statt in List
        
        # TP, FP, FN, TN
        tp = np.sum((predicted == 1) & (instance_label == 1))
        fp = np.sum((predicted == 1) & (instance_label == 0))
        fn = np.sum((predicted == 0) & (instance_label == 1))
        tn = np.sum((predicted == 0) & (instance_label == 0))
        
        # Metriken
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        results_list.append({
        'threshold': threshold,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'tn': tn
    })

    all_predictions = np.array(all_predictions)
        
    
    
    print(f"\n{'='*60}")
    print(f"MEAN RESULTS")
    print(f"{'='*60}")
    print(f"  Threshold: {df['count_threshold'].mean():.6f}")
    print(f"  Precision: {np.mean(precision_scores):.4f}")
    print(f"  Recall:    {np.mean(recall_scores):.4f}")
    print(f"  F1-Score:  {np.mean(f1_scores):.4f}")
    print(f"  AUC (ROC): {roc_auc:.4f}")
    print(f"{'='*60}\n")
    print(f"BEST RESULTS")
    print(f"{'='*60}")
    print(f"  Precision: {np.max(precision_scores):.4f}")
    print(f"  Recall:    {np.max(recall_scores):.4f}")
    print(f"  F1-Score:  {np.max(f1_scores):.4f}")
    print(f"{'='*60}\n")
    
    # ========== PRECISION-RECALL CURVE ==========
    precision_all, recall_all, _ = precision_recall_curve(all_labels, all_predictions)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(recall_all, precision_all, color='darkgreen', lw=2.5, 
            label=f'Precision-Recall Curve (Best F1 = {np.max(f1_scores):.4f})')
    ax.scatter([np.max(recall_scores)], [np.max(precision_scores)], color='red', s=100, zorder=5, label='Best F1-Score')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curve: Patch-Level Discrimination Performance', fontsize=14, fontweight='bold')
    ax.legend(loc="best", fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/precision_recall_curve.png', dpi=300)
    plt.close()
    print(f"✓ Precision-Recall curve saved to {output_dir}/precision_recall_curve.png")
    
    # ========== PRECISION/RECALL vs THRESHOLD ==========
    # fig, ax = plt.subplots(figsize=(12, 6))
    # ax.plot(test_thresholds, precision_scores, label='Precision', color='blue', lw=2, marker='o', markersize=3)
    # ax.plot(test_thresholds, recall_scores, label='Recall', color='green', lw=2, marker='s', markersize=3)
    # ax.plot(test_thresholds, f1_scores, label='F1-Score', color='red', lw=2, marker='^', markersize=3)
    
    # ax.axvline(best_threshold, color='red', linestyle='--', linewidth=2, label=f'Best Threshold = {best_threshold:.4f}')
    
    # ax.set_xlabel('Threshold', fontsize=12)
    # ax.set_ylabel('Score', fontsize=12)
    # ax.set_title('Precision, Recall, and F1-Score vs. Threshold', fontsize=14, fontweight='bold')
    # ax.legend(fontsize=11)
    # ax.grid(alpha=0.3)
    
    # plt.tight_layout()
    # plt.savefig(f'{output_dir}/metrics_vs_threshold.png', dpi=300)
    # plt.close()
    # print(f"✓ Metrics vs. Threshold saved to {output_dir}/metrics_vs_threshold.png")
    
    # ========== SAVE RESULTS TABLE ==========
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'{output_dir}/threshold_analysis_results.csv', index=False)
    print(f"✓ Detailed results saved to {output_dir}/threshold_analysis_results.csv")
    
    return {
        'roc_auc': roc_auc,
        'mean_threshold': df['count_threshold'].mean(),
        'best_f1': np.max(f1_scores) if f1_scores else 0,
        'best_precision': np.max(precision_scores) if precision_scores else 0,
        'best_recall': np.max(recall_scores) if recall_scores else 0,
        'results_df': results_df
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Attention Weight Distribution and Patch-Level Performance')
    parser.add_argument('--json_path', type=str, required=True, 
                        help='Path to MLflow aggregated_run_results.json')
    parser.add_argument('--h5_path', type=str, default='data/datasets/bags/gwhd_bags.h5',
                        help='Path to H5 dataset')
    parser.add_argument('--dataset_name', type=str, default='gwhd_bags_dense',
                        help='Dataset name in H5 file')
    parser.add_argument('--output_dir', type=str, default='./attention_analysis',
                        help='Output directory for plots and results')
    
    args = parser.parse_args()
    
    print(f"Loading attention data from {args.json_path}...")
    df = load_attention_data(args.json_path, args.h5_path, args.dataset_name)
    
    all_labels_flat = np.concatenate([np.array(x).flatten() for x in df['instance_labels']])
    print(f"Loaded {len(df)} bags total")
    print(f"Positive patches: {np.sum(all_labels_flat == 1)}, Negative patches: {np.sum(all_labels_flat == 0)}")

    results = analyze_attention_distribution(df, output_dir=args.output_dir)
    
    print(f"\nAnalysis complete! Check {args.output_dir}/ for visualizations.")