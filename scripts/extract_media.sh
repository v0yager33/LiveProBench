#!/usr/bin/env bash
# Turn the annotation's timestamps into the frames and audio tracks a run is fed.
#
#   scripts/extract_media.sh                            # every subtask, 1 frame per chunk
#   scripts/extract_media.sh --frames-per-chunk 2       # what --model roma reads
#   scripts/extract_media.sh --suite window_response
#
# Required once per media root, not optional: the published set is annotation only, so a run
# has nothing to look at until this has been done. It writes sha256 for every file it
# produces, which is what lets a later reader show they are looking at the same pixels rather
# than at whatever their own ffmpeg happened to decode.
#
# Pass --compare N instead to measure a handful of chunks against the published 1s slices
# without writing a media root; see docs/data_format.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "$HERE")"

python3 "$REPO/src/extract_media.py" \
  --suite all \
  --workers 16 \
  "$@"
