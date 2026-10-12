#!/bin/bash
# ============================================================================
# Kompletter Sweep ueber die mit mnist_bags_loader_h5.py erzeugten
# MNIST-Bag-Datasets.
#
#   Gitter : mean_bag_length x num_bag x {attention, gated_attention}
#   Modelle: Standard-Konfiguration von main_mlflow.py
#            (M=500, L=128, pool_size=4, num_maps=50, kernel_size=5,
#             softmax-Attention, lr=5e-4, reg=1e-4)
#
#   Dataset-Gruppen: bags{num_bag}_len{mean_bag_length}, wie von
#   mnist_bags_loader_h5.py geschrieben (train/validation/test in einer Gruppe).
# ============================================================================
set -u

# ── Konfiguration (alles per Env ueberschreibbar) ────────────────────────────
#   Beispiel: EPOCHS=5 SEEDS=1 BAGS="50 100" ./run_mnist_bags_sweep.sh
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${PROJECT_ROOT}/runs"

PYTHON="${PYTHON:-python}"   # Interpreter des Conda-Envs (environment.yml)

# Quelldatei: data/datasets/bags/mnist_bags_base.h5 enthaelt alle Gruppen
# bags{N}_len{L} (siehe mnist_bags_loader_h5.py --mode sweep). Die Kopie direkt
# unter data/datasets/ ist der aeltere Ablageort.
if [ -z "${SOURCE_H5:-}" ]; then
  for cand in "${PROJECT_ROOT}/data/datasets/bags/mnist_bags_base.h5" \
              "${PROJECT_ROOT}/data/datasets/mnist_bags_base.h5"; do
    [ -f "${cand}" ] && SOURCE_H5="${cand}" && break
  done
fi
SOURCE_H5="${SOURCE_H5:-${PROJECT_ROOT}/data/datasets/bags/mnist_bags_base.h5}"

EXP_NAME="${EXP_NAME:-mnist_bag_sweep_5seeds}"
EPOCHS="${EPOCHS:-20}"
SEEDS="${SEEDS:-1 10 100 1000 10000}"   # space-separierte Liste von Seeds
EXTRA_ARGS="${EXTRA_ARGS:---naive_counting}"   # z.B. "--naive_counting --log_attention_weights"

# Sweep-Gitter (space-separierte Strings, passend zum __main__-Block des Loaders)
LENGTHS="${LENGTHS:-10 50 100}"                       # mean_bag_length
BAGS="${BAGS:-50 100 150 200 300 400 500}"            # num_bag im Train-Split
MODELS="${MODELS:-attention gated_attention}"

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///${SCRIPT_DIR}/mlflow.db}"
# ────────────────────────────────────────────────────────────────────────────

echo "=== Sweep gestartet: $(date) ==="
echo "=== Quelle : ${SOURCE_H5} ==="
echo "=== MLflow : ${MLFLOW_TRACKING_URI} ==="

if [ ! -f "${SOURCE_H5}" ]; then
    echo "FEHLER: Quelldatei ${SOURCE_H5} nicht gefunden." >&2
    echo "        Erst mnist_bags_loader_h5.py laufen lassen, oder SOURCE_H5=... setzen." >&2
    exit 1
fi

# ── Sweep ───────────────────────────────────────────────────────────────────
cd "${SCRIPT_DIR}" || exit 1

total=0
failed=0
for length in ${LENGTHS}; do
  for num_bag in ${BAGS}; do
    dataset="bags${num_bag}_len${length}"
    for model in ${MODELS}; do
      run_name="${model}_${dataset}"
      total=$((total + 1))

      echo "=== [$(date +%H:%M:%S)] ${run_name} ==="
      "${PYTHON}" main_mlflow.py \
        --model "${model}" \
        --dataset "${dataset}" \
        --path "${SOURCE_H5}" \
        --epochs "${EPOCHS}" \
        --seeds ${SEEDS} \
        --exp_name "${EXP_NAME}" \
        --run_name "${run_name}" \
        ${EXTRA_ARGS}

      if [ $? -ne 0 ]; then
        failed=$((failed + 1))
        echo "!!! FEHLGESCHLAGEN: ${run_name} - Sweep laeuft weiter"
      fi
    done
  done
done

echo "=== Sweep abgeschlossen: $(date) ==="
echo "=== Runs: ${total}, davon fehlgeschlagen: ${failed} ==="
