#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
docker build --platform linux/amd64 --network host -t aic2026-64211460-xl:1.0 .
docker save aic2026-64211460-xl:1.0 | gzip -1 > aic2026-64211460-xl.docker.tar.gz
