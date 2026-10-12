#!/bin/bash
# ============================================================
# Re-creates the thesis figures and tables that are generated from the logged
# MLflow results (runs/mlflow.db + runs/mlruns/). Run from the repository root:
#
#   bash eval/make_thesis_figures.sh            (PYTHON=... selects the interpreter)
#
# Output goes to eval/results/ under the file names used in the thesis.
# Not covered here:
#   - feature_vis_*.png: MLflow artifacts written by main_mlflow.py (visualize_features.py)
#   - annotation_spectrum.png, diff_ex_bboxes.png: eval/figures/ (crop chosen visually)
#   - tables assembled by hand from MLflow queries (CLAM, drone, threshold/R^2 tables);
#     their values can be looked up with `python eval/mlflow_toolbox.py table ...`
# ============================================================
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

PY="${PYTHON:-python}"
TB="${PY} eval/mlflow_toolbox.py"
mkdir -p eval/results/crash_exclusion

# ── Method validation on MNIST ──────────────────────────────────────────────
# Bag count and bag length are only part of the dataset name (bags500_len100).
DER='num_bags,bag_length=dataset:bags(\d+)_len(\d+)'

# mnist_validation_auc_bag_length{10,50,100}: bag AUC over the number of training
# bags, crashed seeds (all test bags in one class) excluded
${TB} plot -e mnist_bag_sweep_5seeds --level child --derive "${DER}" \
    --center median --kind line --x num_bags --y auc \
    --split bag_length --series model --no-title \
    --ylim 0.45 1.03 --panel-size 4.4 2.7 \
    --name mnist_validation_auc --format png,pdf \
    --exclude-crashed --outdir eval/results/crash_exclusion

# Counting tables of the MNIST and synthetic-drone sections without crashed seeds
${PY} eval/scripts/crash_exclusion.py

# tab:mnist-density-activation (softmax vs. sigmoid on the high-density bags)
${PY} eval/mnist_bag_sweep/make_density_table.py

# mnist_softmax_scatter.png, mnist_sigmoid_scatter.png
${PY} eval/scripts/plot_mnist_activation_scatter.py

# ── CLAM-based model ───────────────────────────────────────────────────────
# clam_k_sweep: patch AUC, MAE, recall and precision over k_sample
${TB} plot -e clam_sweep_v2,clam_sweep --level child \
    --filter "clam_pseudo_threshold=False" --query "experiment_id == '17' or clam_k_sample >= 24" \
    --kind line --x clam_k_sample --y patch_level_auc,count_mae,patch_level_recall,patch_level_precision \
    --series clam_bag_weight --center median --err iqr --ncol 2 --no-title \
    --name clam_k_sweep --format png,pdf

# clam_readouts: counting error of the different counting rules over k_sample
${TB} plot -e clam_sweep_v3 --level child --filter "grayscaling=False" \
    --filter "clam_pseudo_threshold=False" --kind line --x clam_k_sample \
    --y count_mae,count_calibrated_mae,count_otsu_mae,count_soft_mae --series clam_bag_weight \
    --center median --err iqr --ncol 2 --no-title --name clam_readouts --format png,pdf

# ── From patch to wheat ear counts ──────────────────────────────────────────
# drone_fullimage_conversion: patch count -> ear count on the full images
${PY} eval/scripts/plot_fullimage_conversion.py

echo "Done. Figures and tables are in eval/results/."
