#!/usr/bin/env bash
set -euo pipefail

# Quick OCR smoke test (sample-only) for task2.
# Example:
#   bash task2/run_ocr_smoke_test.sh
#   DATA_PATH=/data/mocheg SAMPLE_LIMIT=4 EPOCH=1 BATCH_SIZE=2 bash task2/run_ocr_smoke_test.sh

DATA_PATH=${DATA_PATH:-/media/duy/01DB32AF5EDFC330/thesis_master_download/mocheg}
SAMPLE_LIMIT=${SAMPLE_LIMIT:-4}
EPOCH=${EPOCH:-1}
BATCH_SIZE=${BATCH_SIZE:-2}
VISION_PT=${VISION_PT:-ocr_easyocr}

echo "[smoke] Running train.sh sample flow..."
SAMPLE_LIMIT="$SAMPLE_LIMIT" \
EPOCH="$EPOCH" \
BATCH_SIZE="$BATCH_SIZE" \
DATA_PATH="$DATA_PATH" \
VISION_PT="$VISION_PT" \
bash task2/train.sh

echo "[smoke] Running memory leak smoke test..."
cd task2
python test_ocr_memory_leak.py \
  --path "$DATA_PATH" \
  --sample_limit "$SAMPLE_LIMIT" \
  --batch_size "$BATCH_SIZE" \
  --steps 6

echo "[smoke] Done."
