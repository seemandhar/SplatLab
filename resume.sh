#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# splatlab RESUME SCRIPT
#
#   Picks up the most recent workspace/run_* via /research-resume in a
#   FRESH claude session (new tmux). Use when the original master's
#   context is full or the session was killed.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

TMUX_SESSION="${TMUX_SESSION:-splatlab-resume}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$TMUX_SESSION' already exists." >&2
  echo "       Attach:  tmux attach -t $TMUX_SESSION" >&2
  echo "       Kill:    tmux kill-session -t $TMUX_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" \
  "claude --dangerously-skip-permissions"

sleep 10
tmux send-keys -t "$TMUX_SESSION" "1" Enter
sleep 3
tmux send-keys -t "$TMUX_SESSION" "/research-resume" Enter

cat <<EOF

═══════════════════════════════════════════════════════════════════
  splatlab RESUMED in tmux session: $TMUX_SESSION
═══════════════════════════════════════════════════════════════════
  ATTACH:            tmux attach -t $TMUX_SESSION
  DETACH:            (inside tmux)  Ctrl-b  then  d
  STATUS:            (inside claude) /research-status
  STOP GRACEFULLY:   (inside claude) /research-stop
  KILL SESSION:      tmux kill-session -t $TMUX_SESSION

  ledger pointer:    workspace/run_20260423_010344/ledger.json
  stop condition:    +1.0 dB avg PSNR on >=3 full-iter scenes (extraordinary)
  wall budget:       32h total (14h used, ~18h remaining)
  paper lock:        paper_locked_pending_extraordinary=true (no rewrite unless bar cleared)
═══════════════════════════════════════════════════════════════════
EOF
