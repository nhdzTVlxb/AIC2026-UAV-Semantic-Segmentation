#!/usr/bin/env bash
set -euo pipefail
checkout="$(cd "$(dirname "$0")" && pwd)"
export AIC_PROJECT_ROOT="$checkout"
exec bash "$checkout/environment/build_a100.sh" "$@"
