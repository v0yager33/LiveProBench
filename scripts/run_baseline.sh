#!/usr/bin/env bash
# Evaluate one baseline on LiveProBench.
#
#   scripts/run_baseline.sh aura                 # needs an endpoint serving AURA already up
#   scripts/run_baseline.sh joyai --suite first_event
#   scripts/run_baseline.sh moss_vl              # local checkpoint, no server
#   scripts/run_baseline.sh m4 --suite window_response
#   scripts/run_baseline.sh roma                 # local checkpoint, hears audio
#   scripts/run_baseline.sh roma --audio off     # same rows, vision only
#
# Some baselines lack an audio encoder; their adapter applies the normal capability
# fallback, and --audio is a no-op for them. ROMA has one, so it is the only baseline that
# can be run both ways -- see docs/baselines.md.
#
# Any extra argument is passed through to src/run_eval.py.
set -euo pipefail

# A literal '}' inside ${1:?...} closes the expansion early and folds the remainder of the
# message into the value, so the usage text below carries no braces.
MODEL_NAME="${1:?usage: run_baseline.sh aura|joyai|moss_vl|m4|roma [run_eval.py args...]}"
shift || true

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

case "$MODEL_NAME" in
  aura|joyai)
    # Served over HTTP, so several rows can be in flight at once.
    EXTRA=(--workers 4)
    ;;
  moss_vl|m4|roma)
    # One in-process model, one row at a time. --workers is capped to 1 by the registry
    # regardless; setting it here just makes the intent visible in run.json.
    EXTRA=(--workers 1)
    ;;
  *)
    echo "unknown baseline '$MODEL_NAME' (expected aura, joyai, moss_vl, m4 or roma)" >&2
    exit 2
    ;;
esac

# M4 consumes frames rather than chunk files, so bound the frame size: a full-HD frame costs
# roughly 2000 tokens and ~30 polled chunks then overflow a 32k context.
#
# ROMA also consumes frames and also has a 32k context, but is left at the suite's shared
# pixel budget (--frame-max-side defaults to 0, i.e. no extra cap) so that its visual input
# matches everyone else's. The 30-chunk context window keeps every row inside 32k -- a poll's
# prompt saturates near 23k tokens regardless of stream length; docs/baselines.md has the
# arithmetic.
case "$MODEL_NAME" in
  m4) EXTRA+=(--frame-max-side 448) ;;
esac

python3 "$REPO/src/run_eval.py" \
  --model "$MODEL_NAME" \
  --suite all \
  --resume \
  "${EXTRA[@]}" \
  "$@"
