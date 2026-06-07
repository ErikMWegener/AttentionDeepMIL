#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=08:00:00
#SBATCH --job-name=mil_training
#SBATCH --output=%x_%j.out        # Ausgabe: <job-name>_<job-id>.out

# ============================================================
# Konfiguration – hier anpassen
# ============================================================
USERNAME="<5-Steller>"
PROJECT_NAME="mil_project"
WORK_DIR="/zpool1/slurm_data/${USERNAME}/${PROJECT_NAME}"
# Docker-Image mit PyTorch + MLflow (von DockerHub)
CONTAINER_IMAGE="docker://pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
CONTAINER_NAME="mil_pytorch"
# ============================================================

echo "=== Job gestartet: $(date) ==="
echo "=== Knoten: $(hostname) ==="

# Arbeitsverzeichnis auf Shared Storage anlegen
mkdir -p "${WORK_DIR}"

# MLflow-Tracking-URI: Logs gehen direkt in den Shared Storage,
# sodass nach dem Job einfach per rsync abgeholt werden kann.
export MLFLOW_TRACKING_URI="file://${WORK_DIR}/mlruns"
echo "=== MLflow Tracking URI: ${MLFLOW_TRACKING_URI} ==="

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
    pip install --quiet mlflow pyyaml scikit-learn matplotlib h5py

    # In das Projektverzeichnis wechseln
    cd /home/${USERNAME}/<PFAD-ZUM-PROJEKT>

    # MLflow Tracking URI setzen (wird via --container-mounts weitergegeben)
    export MLFLOW_TRACKING_URI='file://${WORK_DIR}/mlruns'

    # Script ausführen – Argumente hier anpassen
    python main_mlflow.py \
      --config configs/mein_config.yaml \
      --exp_name 'cluster_experiment' \
      --run_name 'run_$(date +%Y%m%d_%H%M%S)' \
      --epochs 50 \
      --seeds 1 2 3
  "

echo "=== Job abgeschlossen: $(date) ==="
echo "=== MLflow-Daten liegen unter: ${WORK_DIR}/mlruns ==="
echo "=== Jetzt sync_mlflow.sh lokal ausführen, um Daten zu übertragen. ==="
