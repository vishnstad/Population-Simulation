#!/usr/bin/env bash
# One window of a chunked elicitation run.
#
# Both shells this project is driven from freeze between tool calls, so nothing
# progresses in the background: a long run is a sequence of bounded windows, and
# this script is one window. It shards the item list across N processes, waits
# for the window to close, and stops them. The store resumes from where it got
# to; the response cache makes anything already spent free.
#
#   scripts/run_elicitation.sh <provider> <profile> <scope> <n_parallel> <seconds> [extra popsim args...]
#
set -uo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-$HOME/venv-b17/bin/python}"
PROVIDER="${1:?provider}"; PROFILE="${2:?profile}"; SCOPE="${3:?scope}"
NPAR="${4:-4}"; SECS="${5:-160}"; shift 5 2>/dev/null || true

pids=()
for i in $(seq 0 $((NPAR-1))); do
  "$PY" -m popsim.cli elicit --scope "$SCOPE" --profile "$PROFILE" \
      --shard "$i" --n-shards "$NPAR" --seconds "$SECS" \
      --set "llm.active_providers=[\"$PROVIDER\"]" "$@" \
      > "/tmp/elicit_${SCOPE}_${i}.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done

echo "=== window done: $SCOPE / $PROVIDER / $PROFILE / ${NPAR}x ==="
grep -h -E "this slice|remaining|failures|no anchor|exceeds|Traceback|Error" /tmp/elicit_${SCOPE}_*.log | sort | uniq -c | sort -rn | head -20
