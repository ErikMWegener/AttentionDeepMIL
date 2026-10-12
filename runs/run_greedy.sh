#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=20:00:00
#SBATCH --job-name=greedy_mil
#SBATCH --output=%x_%j.out        # Ausgabe: <job-name>_<job-id>.out

# ============================================================
# Greedy search over the AbMIL backbone dimensions (greedy_search.py).
# Thesis: "Backbone dimensions and greedy search". MLflow: greedy_hpo,
# greedy_sigmoid (gwhd_bags_dense). The values below are from the last use of the
# script; the settings of the thesis runs are stored as MLflow parameters.
#   sbatch run_greedy.sh
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
EXP_NAME="drone_runs"
DATA_PATH="${PROJECT_ROOT}/data/datasets/drone_bags.h5"
DATASET="drone_128_bags"
# ============================================================

echo "=== Job gestartet: $(date) ==="
echo "=== Knoten: $(hostname) ==="

# MLflow-Tracking-URI: Logs gehen direkt in den Shared Storage,
# sodass nach dem Job einfach per rsync abgeholt werden kann.

# ============================================================
# Container starten und Python-Script ausführen
#
# --container-mounts ~/:/home/${USERNAME}
#   → Dein Homeverzeichnis (mit dem Projektcode) ist im Container
#     unter /home/${USERNAME} verfügbar, kein sbcast nötig.
# --container-mounts ${WORK_DIR}:${WORK_DIR}
#   → Shared Storage wird eingehängt, damit MLflow dorthin schreiben kann.
# ============================================================
srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-name="${CONTAINER_NAME}" \
  --container-mounts="${HOME}:/home/${USERNAME},${WORK_DIR}:${WORK_DIR}" \
  --container-remap-root \
  --container-writable \
  bash -c "
    # Abhängigkeiten installieren (beim ersten Mal; danach cached im Container)
    pip install --quiet mlflow pyyaml scikit-learn matplotlib h5py entmax umap-learn

    # In das Projektverzeichnis wechseln
    cd /home/${USERNAME}/AttentionDeepMIL/runs
   
    mkdir -p '${WORK_DIR}'
   
    # MLflow Tracking URI setzen (wird via --container-mounts weitergegeben)
    export MLFLOW_TRACKING_URI='sqlite:///${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db'

    # Script ausführen – Argumente hier anpassen
    cd '${SCRIPT_DIR}'

    python greedy_search.py \
      --config configs/gwhd_config.yaml \
      --exp_name ${EXP_NAME} \
      --attention_activation entmax \
      --seeds 1 2 3 \
      --epochs 50 \
      --naive_counting \
      --params model_num_maps model_pool_size model_M \
      --target_metric 'auc_mean:0.5,patch_level_auc_mean:0.5,auc_std:-0.3,patch_level_auc_std:-0.3' \
      --rounds 2 \
      --final_seeds 1 2 3 4 5 \
      --path ${DATA_PATH} \
      --dataset ${DATASET}
  
    # Nach dem Training: DB ins Heimverzeichnis kopieren,
    # damit rsync sie vom Login-Knoten aus erreichen kann.
    cp '${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db' '${SCRIPT_DIR}/mlflow_${SLURM_JOB_ID}.db'
  "
echo "=== Job abgeschlossen: $(date) ==="
echo "=== MLflow-Daten liegen unter: ${WORK_DIR}/mlruns ==="
echo "=== Jetzt sync_mlflow.sh lokal ausführen, um Daten zu übertragen. ==="
