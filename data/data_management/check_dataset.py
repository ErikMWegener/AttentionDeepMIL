import numpy as np
import pandas as pd
from dataset_manager import DatasetReader
import sys
import matplotlib.pyplot as plt

def plot_split_histograms(train_stats, val_stats, test_stats):
    """
    Erstellt eine Figur mit drei Subplots für die Count-Verteilungen 
    der Trainings-, Validierungs- und Test-Daten.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 Reihe, 3 Spalten
    
    splits = [
        ("Train Split", train_stats, axes[0]),
        ("Validation Split", val_stats, axes[1]),
        ("Test Split", test_stats, axes[2])
    ]
    
    for title, stats, ax in splits:
        if "error" in stats:
            ax.text(0.5, 0.5, stats["error"], ha='center', va='center')
            ax.set_title(title)
            continue
            
        hist_data = stats["histogram"]
        frequencies = hist_data["frequencies"]
        bin_edges = hist_data["bin_edges"]
        
        # Berechne die Breite der Bins
        widths = np.diff(bin_edges)
        
        ax.bar(bin_edges[:-1], frequencies, width=widths, align='edge', edgecolor='black', alpha=0.7)

        mean_val = stats.get("mean")
        if mean_val is not None:
            ax.text(0.95, 0.95, f'Mean: {mean_val:.2f}', 
                    transform=ax.transAxes, 
                    fontsize=11,
                    ha='right', va='top', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='gray'))
            
        ax.set_title(title)
        ax.set_xlabel('Count Value')
        ax.set_ylabel('Frequency')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()  # Verhindert, dass sich Labels überschneiden
    plt.savefig(f'../../eval/results/{dataset_name}_histogram_splits.png')  # Bei Bedarf als Bilddatei abspeichern
    #plt.show()


args = sys.argv
dataset_path = args[1]
dataset_name = args[2]
# Trainings-Split
train_reader = DatasetReader(dataset_path, dataset_name, split='train')
train_reader.print_count_distribution()
train_stats = train_reader.analyze_count_distribution()

# Validierungs-Split
val_reader = DatasetReader(dataset_path, dataset_name, split='validation')
val_reader.print_count_distribution()
val_stats = val_reader.analyze_count_distribution()

# Test-Split
test_reader = DatasetReader(dataset_path, dataset_name, split='test')
test_reader.print_count_distribution()
test_stats = test_reader.analyze_count_distribution()

# Erstelle die Histogramme für alle drei Splits
plot_split_histograms(train_stats, val_stats, test_stats)