#!/usr/bin/env bash
# Thin wrapper — see scripts/bench_mobile.py
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/bench_mobile.py" "$@"
