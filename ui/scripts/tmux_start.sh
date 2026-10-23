#!/usr/bin/env bash
# Start (or resume into) the single persistent Claude Code research session and
# send one slash command to it. Mirrors scripts/launch.sh, but the command to
# send is supplied by the SplatLab UI backend via GSS_SEND_CMD so the same
# script handles both /research launches and /research-resume.
#
# The backend builds GSS_SEND_CMD from sanitized parts only (numeric hours,
# validated http(s) URLs, scene whitelist). Free-form idea text is NOT here —
# it is seeded into config.yaml beforehand and picked up via the /research
# positional fallback.
#
# Exit codes: 0 ok · 3 session already exists · 1 error
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

SESSION="${GSS_SESSION:-splatlab}"
SEND_CMD="${GSS_SEND_CMD:?GSS_SEND_CMD is required}"

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux not found" >&2; exit 1
fi
if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not on PATH" >&2; exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists"
  exit 3
fi

# Detached session running Claude Code with permissions bypassed (research runs
# 24h unattended — there is no human to answer confirm prompts).
tmux new-session -d -s "$SESSION" -c "$ROOT" "claude --dangerously-skip-permissions"

# Claude takes a few seconds to bootstrap its TUI.
sleep 10
# Accept the trust-folder dialog if it appears (option 1).
tmux send-keys -t "$SESSION" "1" Enter
sleep 3
# Send the slash command and submit.
tmux send-keys -t "$SESSION" "$SEND_CMD" Enter

echo "started"
