#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash task2/setup_ocr_env.sh
# Optional envs:
#   INSTALL_PADDLE=1   # also install paddleocr
#   FORCE_TORCH_CPU=1  # force reinstall of cpu torch/torchvision pair

PYTHON_BIN=${PYTHON_BIN:-python}
INSTALL_PADDLE=${INSTALL_PADDLE:-0}
FORCE_TORCH_CPU=${FORCE_TORCH_CPU:-0}

repair_torch_stack() {
  echo "[setup] Repairing torch/torchvision compatibility (CPU wheels)..."
  "$PYTHON_BIN" -m pip install --index-url https://download.pytorch.org/whl/cpu --force-reinstall \
    torch==2.6.0+cpu torchvision==0.21.0+cpu

  # Keep common scientific stack compatible with task2 training dependencies.
  "$PYTHON_BIN" -m pip install --force-reinstall \
    numpy==1.26.4 pillow==10.4.0 fsspec==2024.6.1 typing_extensions==4.14.0 setuptools==79.0.1
}

check_easyocr_import() {
  "$PYTHON_BIN" - <<'PY'
try:
    import easyocr  # noqa: F401
    print('[setup] EasyOCR import check: OK')
except Exception as exc:
    print('[setup] EasyOCR import check failed:', exc)
    raise
PY
}

echo "[setup] Upgrading pip tooling..."
"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel

echo "[setup] Installing OCR training dependencies..."
"$PYTHON_BIN" -m pip install easyocr psutil

if [ "$FORCE_TORCH_CPU" = "1" ]; then
  repair_torch_stack
fi

if ! check_easyocr_import; then
  repair_torch_stack
  check_easyocr_import
fi

if [ "$INSTALL_PADDLE" = "1" ]; then
  echo "[setup] Installing optional PaddleOCR backend..."
  "$PYTHON_BIN" -m pip install paddleocr
fi

echo "[setup] Pre-downloading EasyOCR model weights..."
"$PYTHON_BIN" - <<'PY'
import numpy as np
import easyocr

reader = easyocr.Reader(['en'], gpu=False)
# Trigger first OCR pass so model files are downloaded/cached now.
img = np.ones((32, 128, 3), dtype=np.uint8) * 255
_ = reader.readtext(img, detail=0)
print('[setup] EasyOCR warmup done')
PY

echo "[setup] Verifying Task2 train entrypoint imports..."
"$PYTHON_BIN" - <<'PY'
from task2.model import MultiModalClassification
print('[setup] Import check passed:', MultiModalClassification.__name__)
PY

echo "[setup] Completed successfully."
