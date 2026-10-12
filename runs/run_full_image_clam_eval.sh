#!/usr/bin/env bash
# ============================================================
# Thesis: "From patch to wheat ear counts". Evaluates every CLAM state dict in
# models/state_dicts/clam/ (clam_{k|qt}_annot{32|64|128}_seed{s}.pth) on the full
# annotated drone images: threshold calibrated on the validation bags, linear /
# isotonic conversion fitted on the calibration images, metrics on the test images.
# MLflow: full_image_evaluation_cal_threshold. The *_calthr runs reported in the
# thesis passed the threshold calibrated during training (MLflow pretrained_test,
# count_calibrated_threshold) via --full_image_threshold instead. Runs locally:
#   bash run_full_image_clam_eval.sh          (DRY_RUN=1 only prints the commands)
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
STATE_DICT_DIR="${PROJECT_ROOT}/models/state_dicts/clam"
CONFIG_DIR="${SCRIPT_DIR}/configs"
MAIN_SCRIPT="${MAIN_SCRIPT:-${SCRIPT_DIR}/main_mlflow.py}"
TEST_DIR="${PROJECT_ROOT}/data/datasets/drone/2025_wheat_heads_datasets/real_data/annotated_drone_images_split_4m/test_dataset_4m"
CAL_DIR="${PROJECT_ROOT}/data/datasets/drone/2025_wheat_heads_datasets/real_data/annotated_drone_images_split_4m/train_dataset_4m_split/val"
# Trainings-Bags der Modelle; ihre Val-Bags dienen --calibrate_count_threshold
# zur Kalibrierung des Zaehl-Thresholds (Gruppe annot_drone_<patch>_bags).
ANNOT_BAGS="${PROJECT_ROOT}/data/datasets/bags/annot_drone_bags.h5"

DRY_RUN="${DRY_RUN:-0}"

if [[ ! -f "${MAIN_SCRIPT}" ]]; then
  printf 'Fehler: Entry-Point nicht gefunden: %s\n' "${MAIN_SCRIPT}" >&2
  printf 'Bitte main_mlflow.py in runs/ bereitstellen oder MAIN_SCRIPT=/pfad/zu/main_mlflow.py setzen.\n' >&2
  exit 1
fi

if [[ ! -d "${STATE_DICT_DIR}" ]]; then
  printf 'Fehler: State-Dict-Verzeichnis nicht gefunden: %s\n' "${STATE_DICT_DIR}" >&2
  exit 1
fi

shopt -s nullglob
state_dicts=("${STATE_DICT_DIR}"/*.pth)

if (( ${#state_dicts[@]} == 0 )); then
  printf 'Fehler: Keine .pth-Dateien gefunden in %s\n' "${STATE_DICT_DIR}" >&2
  exit 1
fi

for state_dict in "${state_dicts[@]}"; do
  filename="$(basename -- "${state_dict}")"

  if [[ "${filename}" =~ ^clam_(k|qt)_annot([0-9]+)_seed([0-9]+)\.pth$ ]]; then
    mode="${BASH_REMATCH[1]}"
    patch_size="${BASH_REMATCH[2]}"
    seed="${BASH_REMATCH[3]}"
  else
    printf 'Fehler: Unerwarteter State-Dict-Name: %s\n' "${filename}" >&2
    exit 1
  fi

  if [[ "${mode}" == "k" ]]; then
    model="clamk"
    config="${CONFIG_DIR}/top_clam_k_config.yaml"
  else
    model="clamqu"
    config="${CONFIG_DIR}/top_clam_qu_config.yaml"
  fi

  if [[ ! -f "${config}" ]]; then
    printf 'Fehler: Config nicht gefunden: %s\n' "${config}" >&2
    exit 1
  fi

  run_name="${model}_${patch_size}_${seed}"
  command=(
    python "${MAIN_SCRIPT}"
    --config "${config}"
    --load_model "${state_dict}"
    --full_image_test "${TEST_DIR}"
    --full_image_cal "${CAL_DIR}"
    --full_image_only
    --full_image_patch_size "${patch_size}"
    --full_image_ext .tif
    --calibrate_count_threshold
    --path "${ANNOT_BAGS}"
    --dataset "annot_drone_${patch_size}_bags"
    --run_name "${run_name}"
    --exp_name full_image_evaluation_cal_threshold
  )

  printf '=== %s: %s ===\n' "${filename}" "${run_name}"
  printf '%q ' "${command[@]}"
  printf '\n'

  if [[ "${DRY_RUN}" != "1" ]]; then
    "${command[@]}"
  fi
done