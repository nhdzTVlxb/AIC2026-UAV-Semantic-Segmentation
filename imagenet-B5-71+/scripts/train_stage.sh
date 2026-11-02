#!/usr/bin/env bash
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
stage="${1:-}"
if [[ -z "$stage" ]]; then
  echo "usage: $0 STAGE [run_train.py arguments...]" >&2
  exit 2
fi
shift
case "$stage" in
  1) recipe="$root/experiments/phase01_decoder.py" ;;
  2) recipe="$root/experiments/phase02_deep_level.py" ;;
  3) recipe="$root/experiments/phase03_upper_levels.py" ;;
  4) recipe="$root/experiments/phase04_consolidation.py" ;;
  5) recipe="$root/experiments/phase05_teacher_ema999.py" ;;
  6) recipe="$root/experiments/phase06_detail_sampling.py" ;;
  7) recipe="$root/experiments/phase07_teacher_refresh.py" ;;
  8) recipe="$root/experiments/phase08_large_crop.py" ;;
  *) echo "stage must be an integer from 1 to 8" >&2; exit 2 ;;
esac
export AIC_PROJECT_ROOT="${AIC_PROJECT_ROOT:-$root}"
export PYTHONPATH="$root:$root/vendor/InternImage/segmentation${PYTHONPATH:+:$PYTHONPATH}"
python_bin="${PYTHON_BIN:-python}"
command -v "$python_bin" >/dev/null 2>&1 || python_bin=python3
exec "$python_bin" "$root/run_train.py" "$recipe" "$@"
