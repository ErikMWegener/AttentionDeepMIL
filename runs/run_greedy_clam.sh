#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=48:00:00
#SBATCH --job-name=greedy_clam
#SBATCH --output=%x_%j.out        # Ausgabe: <job-name>_<job-id>.out

# ============================================================
# Greedy-Backbone-Suche fuer CLAM auf dem Drohnen-Datensatz.
#
# Es werden dieselben Backbone-Achsen wie in run_greedy.sh optimiert
# (model_num_maps, model_pool_size, model_M) -- einmal fuer jede der beiden
# besten CLAM-Konfigurationen aus dem Sweep (run_cluster_clam_sweep.sh):
#   Phase A: Top-k-Clustering    k_sample=20,  bag_weight=0.7
#   Phase B: Quantil-Pseudolabel q_pos=0.8, q_neg=0.1, bag_weight=0.7
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
EXP_NAME="clam_greedy_drone"
BASE_CONFIG="configs/clam_drone_128_bags_config.yaml"
DATA_PATH="${PROJECT_ROOT}/data/datasets/bags/drone_bags.h5"
DATASET="drone_128_bags"

# Suchbudget: waehrend der Suche wenige Seeds, final mehr.
SEEDS="1 10"
FINAL_SEEDS="1 10 100"
EPOCHS=50
ROUNDS=2
PARAMS="model_num_maps model_pool_size model_M"
TARGET_METRIC='auc_mean:0.5,patch_level_auc_mean:0.5,auc_std:-0.3,patch_level_auc_std:-0.3'

# ── Beste CLAM-Konfigurationen aus dem Sweep ──────────────────────────────
BAG_WEIGHT=0.7        # in beiden Phasen identisch
K_SAMPLE=20           # Phase A: Top-k-Clustering
Q_POS=0.8             # Phase B: Quantil-Pseudolabels
Q_NEG=0.1
# ============================================================

echo "=== Job gestartet: $(date) ==="
echo "=== Knoten: $(hostname) ==="

# ============================================================
# Container starten und beide Greedy-Suchen ausführen
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

    # ── Phase A: Top-k-Clustering (k_sample=${K_SAMPLE}, bag_weight=${BAG_WEIGHT}) ──
    echo '=== [Phase A] CLAM Top-k: k_sample=${K_SAMPLE}, bag_weight=${BAG_WEIGHT} ==='
    python greedy_search.py \
      --config '${BASE_CONFIG}' \
      --exp_name '${EXP_NAME}' \
      --run_prefix 'greedy_clam_topk' \
      --model clam \
      --attention_activation softmax \
      --seeds ${SEEDS} \
      --epochs ${EPOCHS} \
      --naive_counting \
      --count_threshold_eval \
      --clam_k_sample ${K_SAMPLE} \
      --clam_bag_weight ${BAG_WEIGHT} \
      --params ${PARAMS} \
      --target_metric '${TARGET_METRIC}' \
      --rounds ${ROUNDS} \
      --final_seeds ${FINAL_SEEDS} \
      --path '${DATA_PATH}' \
      --dataset ${DATASET}

    # ── Phase B: Quantil-Pseudolabels (q_pos=${Q_POS}, q_neg=${Q_NEG}) ────────
    echo '=== [Phase B] CLAM Pseudo-Threshold: q_pos=${Q_POS}, q_neg=${Q_NEG}, bag_weight=${BAG_WEIGHT} ==='
    python greedy_search.py \
      --config '${BASE_CONFIG}' \
      --exp_name '${EXP_NAME}' \
      --run_prefix 'greedy_clam_pt' \
      --model clam \
      --attention_activation softmax \
      --seeds ${SEEDS} \
      --epochs ${EPOCHS} \
      --naive_counting \
      --count_threshold_eval \
      --clam_bag_weight ${BAG_WEIGHT} \
      --clam_pseudo_threshold \
      --clam_pseudo_quantile_pos ${Q_POS} \
      --clam_pseudo_quantile_neg ${Q_NEG} \
      --params ${PARAMS} \
      --target_metric '${TARGET_METRIC}' \
      --rounds ${ROUNDS} \
      --final_seeds ${FINAL_SEEDS} \
      --path '${DATA_PATH}' \
      --dataset ${DATASET}

    # Nach dem Training: DB ins Heimverzeichnis kopieren,
    # damit rsync sie vom Login-Knoten aus erreichen kann.
    cp '${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db' '${SCRIPT_DIR}/mlflow_${SLURM_JOB_ID}.db'
  "
echo "=== Job abgeschlossen: $(date) ==="
echo "=== MLflow-Daten liegen unter: ${WORK_DIR}/mlruns ==="
echo "=== Jetzt sync_mlflow.sh lokal ausführen, um Daten zu übertragen. ==="
