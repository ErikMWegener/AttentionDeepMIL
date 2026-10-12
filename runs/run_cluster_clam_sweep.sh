#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --job-name=clam_sweep
#SBATCH --output=%x_%j.out        # Ausgabe: <job-name>_<job-id>.out

# ============================================================
# Thesis: "CLAM-based model" (k_sample / bag_weight sweep and quantile
# pseudo-labels). MLflow: clam_sweep, clam_sweep_v2, clam_sweep_v3 (the same
# script, rerun with a new EXP_NAME). SLURM job on the university GPU cluster:
#   sbatch run_cluster_clam_sweep.sh
# ============================================================

# ============================================================
# Konfiguration – hier anpassen
# ============================================================
USERNAME="amfuk"
PROJECT_ROOT="/home/${USERNAME}/AttentionDeepMIL"
SCRIPT_DIR="${PROJECT_ROOT}/runs"
WORK_DIR="/zpool1/slurm_data/${USERNAME}/AttentionDeepMIL"
# Docker-Image mit PyTorch + MLflow (von DockerHub)
CONTAINER_IMAGE="docker://pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
CONTAINER_NAME="mil_pytorch"

# Basis-Config fuer CLAM (wird pro Sweep-Punkt via CLI ueberschrieben)
BASE_CONFIG="configs/clam_gwhd_bags_dense_lr0.0005_reg0.0001_ep50_attention_activationsoftmax_config.yaml"
EXP_NAME="clam_sweep"

# ── Sweep-Gitter (hier die zu variierenden Werte eintragen) ────────────────
# Als space-separierte Strings, NICHT als Bash-Arrays: die Werte werden in den
# 'bash -c'-String eingebettet, und ${ARRAY[@]} wuerde dort den String in mehrere
# Argumente zerreissen (-> "unexpected end of file").
# Phase 1: Top-k Instance-Clustering (Standard-CLAM)
BAG_WEIGHTS="0.5 0.7 0.9"                  # Gewicht des Bag-Loss im kombinierten Loss
K_SAMPLES="16 20 24 28 32"                 # Anzahl Top-/Bottom-Instanzen fuers Clustering

# Phase 2: Pseudo-Threshold-Modus (adaptive Quantil-Pseudolabels)
#   Paare "pos:neg" -- ab pos-Quantil pseudo-positiv, bis neg-Quantil pseudo-negativ
QUANTILE_PAIRS="0.50:0.25 0.60:0.20 0.75:0.25 0.80:0.10"
PT_BAG_WEIGHT=0.7             # fixer Bag-Weight fuer die Quantil-Phase
# ============================================================

echo "=== Job gestartet: $(date) ==="
echo "=== Knoten: $(hostname) ==="

# ============================================================
# Container starten und Sweep ausführen
# ============================================================
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-name="${CONTAINER_NAME}" \
  --container-mounts="${HOME}:/home/${USERNAME},${WORK_DIR}:${WORK_DIR}" \
  --container-remap-root \
  --container-writable \
  bash -c "
    set -e
    # Abhängigkeiten installieren (beim ersten Mal; danach cached im Container)
    pip install --quiet mlflow pyyaml scikit-learn matplotlib h5py entmax umap-learn

    mkdir -p '${WORK_DIR}'

    # MLflow Tracking URI setzen (wird via --container-mounts weitergegeben)
    export MLFLOW_TRACKING_URI='sqlite:///${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db'

    cd '${SCRIPT_DIR}'

    # ── Phase 1: bag_weight × k_sample (Top-k Clustering) ──────────────────
    for bw in ${BAG_WEIGHTS}; do
      for k in ${K_SAMPLES}; do
        echo \"=== [Phase 1] bag_weight=\${bw}  k_sample=\${k} ===\"
        python main_mlflow.py \
          --config '${BASE_CONFIG}' \
          --exp_name '${EXP_NAME}' \
          --run_name \"clam_topk_bw\${bw}_k\${k}\" \
          --clam_bag_weight \${bw} \
          --clam_k_sample \${k} \
          --count_threshold_eval
      done
    done

    # ── Phase 2: Pseudo-Threshold-Quantile ────────────────────────────────
    for pair in ${QUANTILE_PAIRS}; do
      qpos=\${pair%%:*}
      qneg=\${pair##*:}
      echo \"=== [Phase 2] pseudo_threshold  q_pos=\${qpos}  q_neg=\${qneg} ===\"
      python main_mlflow.py \
        --config '${BASE_CONFIG}' \
        --exp_name '${EXP_NAME}' \
        --run_name \"clam_pt_qpos\${qpos}_qneg\${qneg}\" \
        --clam_bag_weight ${PT_BAG_WEIGHT} \
        --clam_pseudo_threshold \
        --clam_pseudo_quantile_pos \${qpos} \
        --clam_pseudo_quantile_neg \${qneg} \
        --count_threshold_eval
    done

    # Nach dem Training: DB ins Heimverzeichnis kopieren,
    # damit rsync sie vom Login-Knoten aus erreichen kann.
    cp '${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db' '${SCRIPT_DIR}/mlflow_${SLURM_JOB_ID}.db'
  "
echo "=== Job abgeschlossen: $(date) ==="
echo "=== MLflow-Daten liegen unter: ${WORK_DIR}/mlruns ==="
echo "=== Jetzt sync_mlflow.sh lokal ausführen, um Daten zu übertragen. ==="
