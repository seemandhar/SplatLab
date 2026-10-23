#!/usr/bin/env bash
# Atomic GPU reservation via flock.
#
# Usage:
#   scripts/gpu_reserve.sh acquire <workspace> <exp_id> [--min-free-gb 20]
#     → stdout: GPU index (e.g. 0-7)    exit 0 on success, 1 if no free GPU
#
#   scripts/gpu_reserve.sh release <workspace> <exp_id> <gpu_idx>
#     → clears the reservation
#
#   scripts/gpu_reserve.sh list <workspace>
#     → prints current reservations as JSON
#
# Reservations live in $WS/.gpu_reservations/<idx>.lock (file locked with flock
# for the entire session). Stale reservations (> 2h old) are auto-cleaned.

set -euo pipefail

CMD="${1:?usage: acquire|release|list <workspace> ...}"
WS="${2:?workspace required}"
RES_DIR="$WS/.gpu_reservations"
mkdir -p "$RES_DIR"

case "$CMD" in
  acquire)
    EXP_ID="${3:?exp_id required}"
    MIN_FREE_GB=20
    if [[ "${4:-}" == "--min-free-gb" ]]; then
      MIN_FREE_GB="$5"
    fi

    # NVML/nvidia-smi is broken on this box (driver mismatch) but CUDA works.
    # Use the torch-based probe (scripts/gpu_free.py) restricted to GPU_ALLOWED (default 3-7).
    TMP=$(mktemp)
    trap 'rm -f "$TMP"' EXIT
    GPU_PROBE_PY="${GPU_PROBE_PY:-python}"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    "$GPU_PROBE_PY" "$SCRIPT_DIR/gpu_free.py" > "$TMP" 2>/dev/null || true

    ACQUIRED=""
    while IFS=, read -r idx free_mb; do
      idx=$(echo "$idx" | tr -d ' ')
      free_mb=$(echo "$free_mb" | tr -d ' ')
      [[ -z "$idx" ]] && continue
      if (( free_mb < MIN_FREE_GB * 1024 )); then
        continue
      fi

      LOCK="$RES_DIR/$idx.lock"
      # Check existing reservation (with stale cleanup at 2h)
      if [[ -s "$LOCK" ]]; then
        owner=$(head -1 "$LOCK" 2>/dev/null || true)
        lock_age=$(( $(date +%s) - $(stat -c %Y "$LOCK") ))
        if [[ "$lock_age" -gt 7200 ]]; then
          : > "$LOCK"   # stale, clear it
        elif [[ "$owner" != "$EXP_ID" ]]; then
          continue      # held by someone else
        fi
      fi

      # Atomic claim via flock. If flock is unavailable, degrade gracefully.
      if command -v flock >/dev/null 2>&1; then
        # flock -x -n will fail if another process holds it
        if ( flock -x -n 9 && echo "$EXP_ID" > "$LOCK" ) 9>>"$LOCK"; then
          ACQUIRED="$idx"
          break
        fi
      else
        echo "$EXP_ID" > "$LOCK"
        ACQUIRED="$idx"
        break
      fi
    done < "$TMP"

    if [[ -n "$ACQUIRED" ]]; then
      echo "$ACQUIRED"
      exit 0
    fi
    echo "no free gpu" >&2
    exit 1
    ;;

  release)
    EXP_ID="${3:?exp_id required}"
    GPU_IDX="${4:?gpu_idx required}"
    LOCK="$RES_DIR/$GPU_IDX.lock"
    if [[ -f "$LOCK" ]]; then
      owner=$(head -1 "$LOCK" 2>/dev/null || true)
      if [[ "$owner" == "$EXP_ID" ]]; then
        rm -f "$LOCK"
        echo "released gpu $GPU_IDX from $EXP_ID"
      else
        echo "warn: $LOCK owned by '$owner' not '$EXP_ID'; not releasing" >&2
        exit 2
      fi
    fi
    ;;

  list)
    echo "{"
    first=1
    for lock in "$RES_DIR"/*.lock; do
      [[ -f "$lock" ]] || continue
      idx=$(basename "$lock" .lock)
      owner=$(head -1 "$lock" 2>/dev/null || true)
      mtime=$(stat -c %Y "$lock")
      age=$(( $(date +%s) - mtime ))
      [[ "$first" -eq 1 ]] || echo ","
      printf '  "%s": {"exp_id": "%s", "age_s": %d}' "$idx" "$owner" "$age"
      first=0
    done
    echo ""
    echo "}"
    ;;

  *)
    echo "unknown command: $CMD" >&2
    exit 2
    ;;
esac
