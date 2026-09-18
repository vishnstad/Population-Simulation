#!/usr/bin/env bash
# Run the full elicitation for one model arm, start to finish, on Gaurav's Mac.
#
#   cd ~/Downloads/fyp/popsim
#   ./scripts/run_on_mac.sh mistral            # the 14B headline arm  (~7 h)
#   ./scripts/run_on_mac.sh mistral_8b         # the throughput arm    (~1 h)
#   ./scripts/run_on_mac.sh mistral_3b         # the bottom tier point (~20 m)
#
# Safe to stop with Ctrl-C and re-run: the store resumes and the response cache
# makes anything already spent free. Safe to run while the session is working —
# it writes one file per item and nothing else touches those files.
#
# It paces itself to the provider's measured req/min, so it is slow on purpose.
# Leave it running; the progress line tells you where it is.
set -uo pipefail
cd "$(dirname "$0")/.."
ARM="${1:?usage: run_on_mac.sh <mistral|mistral_8b|mistral_3b|google|groq> [profile]}"
PROFILE="${2:-permutation}"

VENV="$HOME/venv-b17"
if [ ! -x "$VENV/bin/python" ]; then
  echo "== building the venv (one time) =="
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 "$VENV" && uv pip install --python "$VENV/bin/python" -e ".[dev,pdf]"
  else
    python3 -m venv "$VENV" && "$VENV/bin/pip" install -q -e ".[dev,pdf]"
  fi
fi
PY="$VENV/bin/python"
MODEL=$("$PY" - "$ARM" <<'EOF'
import sys, yaml
arm = sys.argv[1]
cfg = yaml.safe_load(open("configs/gss_main.yaml"))
print(next(p["model"] for p in cfg["llm"]["providers"] if p["name"] == arm))
EOF
)
echo "== arm $ARM -> model $MODEL, profile $PROFILE =="
"$PY" -m popsim.cli doctor --live --set "llm.active_providers=[\"$ARM\"]" || {
  echo "doctor is unhappy — fix that before spending a night on this"; exit 1; }

for SCOPE in population anchors targets; do
  echo; echo "== $SCOPE =="
  for attempt in $(seq 1 400); do
    OUT=$("$PY" -m popsim.cli elicit --scope "$SCOPE" --profile "$PROFILE" \
            --set "llm.active_providers=[\"$ARM\"]" --seconds 900 2>&1)
    echo "$OUT" | grep -E "this slice|remaining|failures" | sed 's/^/   /'
    echo "$OUT" | grep -q -- "-> COMPLETE" && break
    echo "$OUT" | grep -q "calls failed in a row" && {
      echo "   provider is refusing; sleeping 5 min"; sleep 300; }
  done
  "$PY" scripts/store_progress.py "$MODEL" "$PROFILE"
done

echo; echo "== done. Now:"
echo "   $PY -m popsim.cli layer4 --profile $PROFILE --model $MODEL"
