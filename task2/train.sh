#!/usr/bin/env bash
set -euo pipefail

BATCH_SIZE=${BATCH_SIZE:-2}
EPOCH=${EPOCH:-30}
DATA_PATH=${DATA_PATH:-/media/duy/01DB32AF5EDFC330/thesis_master_download/mocheg}
VISION_PT=${VISION_PT:-ocr_easyocr}
SAMPLE_LIMIT=${SAMPLE_LIMIT:-}
OCR_CACHE_PATH=${OCR_CACHE_PATH:-}

CMD=(
    python task2/main.py
    --batch_size "$BATCH_SIZE"
    --epoch "$EPOCH"
    --val
    --path "$DATA_PATH"
    --vision_pt "$VISION_PT"
)

if [ -n "$SAMPLE_LIMIT" ]; then
    CMD+=(--sample_limit "$SAMPLE_LIMIT")
fi

if [ -n "$OCR_CACHE_PATH" ]; then
    CMD+=(--ocr_cache_path "$OCR_CACHE_PATH")
fi

"${CMD[@]}"
