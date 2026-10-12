#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
#SBATCH --job-name=color_models
#SBATCH --output=%x_%j.out        # Ausgabe: <job-name>_<job-id>.out

# ============================================================
# Thesis: "Colour images and learnt grayscale conversion". MLflow: color_models.
# AbMIL and CLAM on the colour GWHD bags with the learnt grayscale layer.
#   sbatch run_color.sh
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

# Dataset: GWHD-Bags in Farbe (RGB) -----------------------------------------
# Modelle konvertieren via --grayscale (LearnedGrayscale) selbst auf 1 Kanal.
DATASET="gwhd_bags_color"
DATA_PATH="../data/datasets/gwhd_bags_color.h5"
EXP_NAME="color_models"
EPOCHS=50
SEEDS="1 10 100"

# Modelle, die mit LearnedGrayscale auf dem Farb-Dataset laufen sollen -------
# Als space-separierter String (kein Bash-Array): ${MODELS[@]} wuerde im
# 'bash -c'-String den String in mehrere Argumente zerreissen.
MODELS="attention clam"
# ============================================================

echo "=== Job gestartet: $(date) ==="
echo "=== Knoten: $(hostname) ==="

# ============================================================
# Container starten und pro Modell einen Run ausführen
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

    for model in ${MODELS}; do
      echo \"=== Modell: \${model} auf ${DATASET} (LearnedGrayscale) ===\"

      # CLAM bekommt zusaetzlich die Counting-Threshold-Auswertung
      extra_args=''
      if [ \"\${model}\" = 'clam' ]; then
        extra_args='--count_threshold_eval'
      fi

      python main_mlflow.py \
        --model \${model} \
        --dataset '${DATASET}' \
        --path '${DATA_PATH}' \
        --epochs ${EPOCHS} \
        --seeds ${SEEDS} \
        --exp_name '${EXP_NAME}' \
        --run_name \"\${model}_${DATASET}_grayscale\" \
        --grayscale \
        --naive_counting \
        --visualize_features \
        \${extra_args}
    done

    # Nach dem Training: DB ins Heimverzeichnis kopieren,
    # damit rsync sie vom Login-Knoten aus erreichen kann.
    cp '${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db' '${SCRIPT_DIR}/mlflow_${SLURM_JOB_ID}.db'
  "
echo "=== Job abgeschlossen: $(date) ==="
echo "=== MLflow-Daten liegen unter: ${WORK_DIR}/mlruns ==="
echo "=== Jetzt sync_mlflow.sh lokal ausführen, um Daten zu übertragen. ==="
