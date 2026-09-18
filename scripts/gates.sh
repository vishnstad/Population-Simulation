#!/usr/bin/env bash
# Run the full gate suite, including the data-backed Layer 0 and Layer 1 checks.
# Use this locally, where ../data exists. CI runs the subset that needs no microdata.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python}"

echo "== Gate 0: config, budget guard, 429 failover =="
$PY -m pytest -q tests/test_config.py tests/test_gate0.py

echo
echo "== Gate 1: Layer 0 toplines + Layer 1 noise floor =="
$PY -m pytest -q -m "layer0 or layer1"

echo
echo "== no-op run =="
$PY -m popsim.cli noop
