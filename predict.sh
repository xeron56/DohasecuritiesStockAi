#!/usr/bin/env bash
set -euo pipefail

PREDICT_PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PREDICT_PROJECT_DIR"

# The CLI backend is google/timesfm-2.0-500m-pytorch. Restrict inference to
# the first visible NVIDIA GPU so TimesFM selects cuda:0.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  "$PREDICT_PROJECT_DIR/timesfm/.venv/bin/tradingagents-predict" "BXPHARMA'PB" \
  --resolution 1d \
  --lookback 1y \
  --split 0.5 \
  --future-steps 12 \
  --open-ui \
  --host 0.0.0.0 \
  --port 8123
