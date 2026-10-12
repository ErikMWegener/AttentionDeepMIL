#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00
#SBATCH --job-name=clam_instscores
#SBATCH --output=%x_%j.out        # Ausgabe: <job-name>_<job-id>.out

# ============================================================
# Gezielte Wiederholung ausgewaehlter CLAM-Sweep-Punkte MIT Instanz-Scores.
#
# Zweck: Die Sweeps clam_sweep / _v2 / _v3 haben keine Instanz-Scores geloggt.
# Diese Laeufe reproduzieren 8 Konfigurationen und schreiben zusaetzlich die
# Spalten instance_scores/instance_labels in aggregated_run_results.json.
#
# Reproduzierbarkeit: --count_threshold_eval MUSS gesetzt bleiben. Das Flag macht
# vor dem Training einen Durchlauf ueber den gemischten Train-Loader (p_train) und
# verschiebt damit den RNG-Strom; nur mit dem Flag entstehen dieselben Modelle wie
# in clam_sweep_v2. Kontrolle nach dem Lauf: patch_level_auc und accuracy je Seed
# muessen exakt den v2-Werten entsprechen.
#
# Abweichung zum alten Code: Das naive Zaehlen nutzt jetzt model.count_threshold
# (0.5) statt 1/K -> count_mae ist NICHT mit v2 vergleichbar. Die 1/K-Zaehlung
# wird aus den Instanz-Scores nachgerechnet.
# ============================================================
USERNAME="amfuk"

PROJECT_ROOT="/home/${USERNAME}/AttentionDeepMIL"
SCRIPT_DIR="${PROJECT_ROOT}/runs"
WORK_DIR="/zpool1/slurm_data/${USERNAME}/AttentionDeepMIL"
CONTAINER_IMAGE="docker://pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime"
CONTAINER_NAME="mil_pytorch"

BASE_CONFIG="configs/clam_gwhd_bags_dense_lr0.0005_reg0.0001_ep50_attention_activationsoftmax_config.yaml"
EXP_NAME="clam_instscores"

# Die Base-Config wurde seit den Sweeps geaendert (num_maps 64, Pfad unter bags/).
# Beides hier explizit auf den Sweep-Stand setzen. DATA_PATH muss auf DIESELBE
# Datei zeigen wie damals ../data/datasets/gwhd_bags_dense.h5 (lokal 794723113 Bytes).
DATA_PATH="../data/datasets/gwhd_bags_dense.h5"
NUM_MAPS=50
SEEDS="1 10 100"

# Top-k-Punkte als "bag_weight:k_sample"
TOPK_POINTS="0.7:4 0.7:8 0.7:20 0.7:32 0.9:8 0.9:20"
# Pseudo-Threshold-Punkte als "q_pos:q_neg" (bag_weight 0.7, k_sample 8 wie im Sweep)
QUANTILE_PAIRS="0.80:0.10 0.50:0.25"
PT_BAG_WEIGHT=0.7
# ============================================================

echo "=== Job gestartet: $(date) ==="
echo "=== Knoten: $(hostname) ==="

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-name="${CONTAINER_NAME}" \
  --container-mounts="${HOME}:/home/${USERNAME},${WORK_DIR}:${WORK_DIR}" \
  --container-remap-root \
  --container-writable \
  bash -c "
    set -e
    pip install --quiet mlflow pyyaml scikit-learn matplotlib h5py entmax umap-learn

    mkdir -p '${WORK_DIR}'
    export MLFLOW_TRACKING_URI='sqlite:///${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db'

    cd '${SCRIPT_DIR}'
    test -f '${DATA_PATH}' || { echo 'Datensatz nicht gefunden: ${DATA_PATH}'; exit 1; }
    ls -l '${DATA_PATH}'

    COMMON=\"--config ${BASE_CONFIG} --exp_name ${EXP_NAME} --path ${DATA_PATH} \
             --model_num_maps ${NUM_MAPS} --seeds ${SEEDS} \
             --naive_counting --count_threshold_eval --soft_counting --log_instance_scores\"

    # ── Top-k ───────────────────────────────────────────────────────────────
    for pt in ${TOPK_POINTS}; do
      bw=\${pt%%:*}
      k=\${pt##*:}
      echo \"=== [Top-k] bag_weight=\${bw}  k_sample=\${k} ===\"
      python main_mlflow.py \${COMMON} \
        --run_name \"clam_topk_bw\${bw}_k\${k}\" \
        --clam_bag_weight \${bw} \
        --clam_k_sample \${k}
    done

    # ── Pseudo-Threshold ────────────────────────────────────────────────────
    for pair in ${QUANTILE_PAIRS}; do
      qpos=\${pair%%:*}
      qneg=\${pair##*:}
      echo \"=== [Pseudo-Threshold] q_pos=\${qpos}  q_neg=\${qneg} ===\"
      python main_mlflow.py \${COMMON} \
        --run_name \"clam_pt_qpos\${qpos}_qneg\${qneg}\" \
        --clam_bag_weight ${PT_BAG_WEIGHT} \
        --clam_k_sample 8 \
        --clam_pseudo_threshold \
        --clam_pseudo_quantile_pos \${qpos} \
        --clam_pseudo_quantile_neg \${qneg}
    done

    cp '${WORK_DIR}/mlflow_${SLURM_JOB_ID}.db' '${SCRIPT_DIR}/mlflow_${SLURM_JOB_ID}.db'
  "
echo "=== Job abgeschlossen: $(date) ==="
echo "=== Danach sync_mlflow.sh lokal ausfuehren. ==="
