#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 PREDICTION_DIR TEST2_DIR OUTPUT_ZIP [CHECKPOINT]" >&2
  exit 2
fi
prediction_dir="$1"
test_dir="$2"
output_zip="$3"
checkpoint="${4:-}"
export AIC_PROJECT_ROOT="${AIC_PROJECT_ROOT:-$root}"
export PYTHONPATH="$root:$root/vendor/InternImage/segmentation${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-python}"
command -v "$python_bin" >/dev/null 2>&1 || python_bin=python3
args=(--prediction-dir "$prediction_dir" --test-dir "$test_dir" \
      --output "$output_zip" --expected-count 1300)
if [[ -n "$checkpoint" ]]; then
  args+=(--checkpoint "$checkpoint")
fi
exec "$python_bin" -m landcover.commands.seal_submission "${args[@]}"
