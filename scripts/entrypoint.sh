#!/usr/bin/env bash
set -euo pipefail
cd /opt/apink/src
command=${1:-help}
if [ "$#" -gt 0 ]; then shift; fi
case "$command" in
  train) exec python train.py "$@" ;;
  validate) exec python validate.py "$@" ;;
  predict) exec python predict.py "$@" ;;
  verify) exec python verify.py "$@" ;;
  smoke) exec python train.py --backward-check "$@" ;;
  help) printf '%s
' 'Commands: train | validate | predict | verify | smoke' ;;
  *) exec "$command" "$@" ;;
esac
