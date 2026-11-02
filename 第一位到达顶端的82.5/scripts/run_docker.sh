#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "$0")/../.." && pwd)
models="$root/AIC-2026-64211460-模型文件与运行环境"
: "${DATA_ROOT:?Set DATA_ROOT to the dataset root containing train and test_3}"
output=${OUTPUT_ROOT:-"$root/运行产物"}
mkdir -p "$output"
docker run --rm --gpus all --ipc=host --network none \
  -v "$DATA_ROOT:/data/uav:ro" -v "$models:/models:ro" -v "$output:/outputs" \
  aic2026-64211460-xl:1.0 "$@"
