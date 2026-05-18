#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-${CHECKPOINT:-logs/tir2rgb/best_model.pth}}"
DATASETS_FOLDER="${DATASETS_FOLDER:-../datasets}"
DATASET_NAME="${DATASET_NAME:-satellite_0_thermalmapping_135_100}"
SAVE_DIR="${SAVE_DIR:-DSS-VLAD}"
BACKBONE="${BACKBONE:-resnet18conv4}"
AGGREGATION="${AGGREGATION:-netvlad}"
NETVLAD_CLUSTERS="${NETVLAD_CLUSTERS:-64}"
CONV_OUTPUT_DIM="${CONV_OUTPUT_DIM:-4096}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-16}"
VAL_POSITIVE_DIST_THRESHOLD="${VAL_POSITIVE_DIST_THRESHOLD:-50}"
TEST_METHOD="${TEST_METHOD:-hard_resize}"
ESP_TYPE="${ESP_TYPE:-esp}"
DSR_TYPE="${DSR_TYPE:-dsr}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}" >&2
  echo "Set CHECKPOINT=/path/to/checkpoint.pth or pass it as the first argument." >&2
  exit 2
fi

python eval.py \
  --resume "${CHECKPOINT}" \
  --dataset_name "${DATASET_NAME}" \
  --datasets_folder "${DATASETS_FOLDER}" \
  --save_dir "${SAVE_DIR}" \
  --backbone "${BACKBONE}" \
  --aggregation "${AGGREGATION}" \
  --netvlad_clusters "${NETVLAD_CLUSTERS}" \
  --conv_output_dim "${CONV_OUTPUT_DIM}" \
  --infer_batch_size "${INFER_BATCH_SIZE}" \
  --val_positive_dist_threshold "${VAL_POSITIVE_DIST_THRESHOLD}" \
  --test_method "${TEST_METHOD}" \
  --esp_type "${ESP_TYPE}" \
  --dsr_type "${DSR_TYPE}"
