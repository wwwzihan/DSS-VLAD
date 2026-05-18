#!/usr/bin/env bash
set -euo pipefail

DATASETS_FOLDER="${DATASETS_FOLDER:-../datasets}"
DATASET_NAME="${DATASET_NAME:-satellite_0_thermalmapping_135_100}"
SAVE_DIR="${SAVE_DIR:-DSS-VLAD}"
BACKBONE="${BACKBONE:-resnet18conv4}"
AGGREGATION="${AGGREGATION:-netvlad}"
NETVLAD_CLUSTERS="${NETVLAD_CLUSTERS:-64}"
CONV_OUTPUT_DIM="${CONV_OUTPUT_DIM:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
INFER_BATCH_SIZE="${INFER_BATCH_SIZE:-16}"
EPOCHS_NUM="${EPOCHS_NUM:-100}"
LR="${LR:-0.0001}"
MARGIN="${MARGIN:-0.1}"
TRAIN_POSITIVES_DIST_THRESHOLD="${TRAIN_POSITIVES_DIST_THRESHOLD:-35}"
VAL_POSITIVE_DIST_THRESHOLD="${VAL_POSITIVE_DIST_THRESHOLD:-50}"
NEGS_NUM_PER_QUERY="${NEGS_NUM_PER_QUERY:-10}"
QUERIES_PER_EPOCH="${QUERIES_PER_EPOCH:-5000}"
CACHE_REFRESH_RATE="${CACHE_REFRESH_RATE:-1000}"
DA="${DA:-DANN_after}"
LAMBDA_DA="${LAMBDA_DA:-0.1}"
ESP_TYPE="${ESP_TYPE:-esp}"
DSR_TYPE="${DSR_TYPE:-dsr}"

python train.py \
  --dataset_name "${DATASET_NAME}" \
  --datasets_folder "${DATASETS_FOLDER}" \
  --save_dir "${SAVE_DIR}" \
  --backbone "${BACKBONE}" \
  --aggregation "${AGGREGATION}" \
  --netvlad_clusters "${NETVLAD_CLUSTERS}" \
  --conv_output_dim "${CONV_OUTPUT_DIM}" \
  --train_batch_size "${TRAIN_BATCH_SIZE}" \
  --infer_batch_size "${INFER_BATCH_SIZE}" \
  --epochs_num "${EPOCHS_NUM}" \
  --lr "${LR}" \
  --margin "${MARGIN}" \
  --train_positives_dist_threshold "${TRAIN_POSITIVES_DIST_THRESHOLD}" \
  --val_positive_dist_threshold "${VAL_POSITIVE_DIST_THRESHOLD}" \
  --negs_num_per_query "${NEGS_NUM_PER_QUERY}" \
  --queries_per_epoch "${QUERIES_PER_EPOCH}" \
  --cache_refresh_rate "${CACHE_REFRESH_RATE}" \
  --DA "${DA}" \
  --lambda_DA "${LAMBDA_DA}" \
  --esp_type "${ESP_TYPE}" \
  --dsr_type "${DSR_TYPE}"
