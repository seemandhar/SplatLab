#!/usr/bin/env bash
# Usage: scripts/poll_training.sh <outputs_dir>
# Returns "running|done|failed|unknown|<PSNR>" based on nerfstudio output folder.
# Used by the master orchestrator to check training progress without reading full logs.
set -euo pipefail

DIR="${1:?outputs_dir is required}"

if [[ ! -d "$DIR" ]]; then
  echo "unknown"; exit 0
fi

# Check for completion marker
if [[ -f "$DIR/results.json" ]]; then
  # Try to extract avg PSNR if present
  if command -v jq >/dev/null 2>&1 && [[ -r "$DIR/results.json" ]]; then
    PSNR=$(jq -r '.psnr // .mean_psnr // "?"' "$DIR/results.json" 2>/dev/null || echo "?")
    echo "done:psnr=$PSNR"
  else
    echo "done"
  fi
  exit 0
fi

# Check for crash markers
if grep -q "Traceback\|Segmentation fault\|CUDA out of memory\|RuntimeError" "$DIR"/*.log 2>/dev/null; then
  echo "failed"
  exit 0
fi

# Check for active training (recent log activity)
LATEST_LOG=$(find "$DIR" -name "*.log" -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
if [[ -n "${LATEST_LOG:-}" ]]; then
  LAST_MOD=$(stat -c %Y "$LATEST_LOG")
  NOW=$(date +%s)
  AGE=$((NOW - LAST_MOD))
  if [[ "$AGE" -lt 300 ]]; then
    echo "running"
    exit 0
  else
    echo "stalled:${AGE}s"
    exit 0
  fi
fi

echo "unknown"
