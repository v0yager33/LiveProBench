#!/usr/bin/env bash
# Run every model that is reachable right now, then print the comparison table.
#
#   scripts/run_all.sh                        # ours and any live baseline
#   scripts/run_all.sh --suite first_event
#
# A model whose server is not up, or whose checkpoint is missing, is reported and skipped
# rather than failing the sweep: baselines come and go as GPUs free up, and a partial table
# with honest gaps beats no table. Every run is --resume, so re-running fills the gaps in.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

attempt() {
  local label="$1"; shift
  echo
  echo "################ $label"
  # --check resolves inputs and probes the backend without spending any GPU time.
  if ! python3 "$REPO/src/run_eval.py" "$@" --check >/dev/null 2>&1; then
    echo "[skip] $label is not reachable (server down or checkpoint missing)"
    return 0
  fi
  python3 "$REPO/src/run_eval.py" "$@" || echo "[warn] $label exited non-zero; see above"
}

SUITE=(--suite all)
if [[ "${1:-}" == "--suite" ]]; then SUITE=(--suite "$2"); shift 2; fi

attempt "ours" --model LiveFlow_omni "${SUITE[@]}" \
  --workers 4 --resume --run-name all "$@"

attempt "aura"    --model aura    "${SUITE[@]}" --workers 4 --resume --run-name all "$@"
attempt "joyai"   --model joyai   "${SUITE[@]}" --workers 4 --resume --run-name all "$@"
attempt "moss_vl" --model moss_vl "${SUITE[@]}" --workers 1 --resume --run-name all "$@"
attempt "m4"      --model m4      "${SUITE[@]}" --workers 1 --resume --run-name all \
  --frame-max-side 448 "$@"
# ROMA keeps the shared pixel budget, so no --frame-max-side: the 30-chunk context window
# keeps every row inside its 32k context anyway. docs/baselines.md explains.
attempt "roma"    --model roma    "${SUITE[@]}" --workers 1 --resume --run-name all "$@"

# The audio ablation. Only the two models with an audio encoder change behaviour here; the
# runner files these under '<run>_no_audio' on its own, so nothing merges into the tables
# above. Both sides see the identical rows, so the pair isolates the audio contribution.
attempt "ours (no audio)" --model LiveFlow_omni "${SUITE[@]}" \
  --workers 4 --resume --run-name all --audio off "$@"
attempt "roma (no audio)" --model roma "${SUITE[@]}" \
  --workers 1 --resume --run-name all --audio off "$@"

echo
echo "################ comparison"
python3 "$REPO/src/report.py" --run all

echo
echo "################ audio ablation"
python3 "$REPO/src/report.py" --run all --run all_no_audio \
  --model LiveFlow_omni --model roma
