#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
: "${AIC_CHECKPOINT:?set AIC_CHECKPOINT to the Stage8 checkpoint}"
: "${AIC_TEST2_ROOT:?set AIC_TEST2_ROOT to the Test2 image directory}"
output="${AIC_OUTPUT_ROOT:-$root/submission_stage8}"
export AIC_PROJECT_ROOT="${AIC_PROJECT_ROOT:-$root}"
export PYTHONPATH="$root:$root/vendor/InternImage/segmentation${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-python}"
command -v "$python_bin" >/dev/null 2>&1 || python_bin=python3
exec "$python_bin" "$root/run_predict.py" \
  --checkpoint "$AIC_CHECKPOINT" \
  --images "$AIC_TEST2_ROOT" \
  --output "$output" "$@"
