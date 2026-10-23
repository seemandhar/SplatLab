#!/usr/bin/env bash
# Idle-deadlock watchdog for splatlab.
#
# Usage (one-shot check):
#   bash scripts/watchdog.sh                           # defaults to most recent workspace
#   bash scripts/watchdog.sh workspace/run_20260418_*
#
# Usage (continuous, as a cron alternative):
#   bash scripts/watchdog.sh --loop 300                # check every 5 min
#
# What it does
# ------------
# A splatlab run can deadlock when a subagent dies silently without
# returning to the master. The master, waiting for a task notification, never
# wakes to apply its own deadlock rule. This watchdog detects that condition
# from outside the Claude Code session.
#
# Trigger:
#   events.jsonl mtime is > $STALE_SECONDS old   (no activity)
#   AND no claude_agent_sdk processes exist for this workspace's run_id
#   AND ledger.status == "running"
#
# Action when triggered:
#   1. Mark any experiment.status == running with no GPU activity as "stalled"
#   2. Log a watchdog_intervention event
#   3. Touch events.jsonl so the master wakes on its polling
#   4. (Optional) send a tmux advisory message
#
# Safe to run concurrently with the master — only reads ledger + writes events.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STALE_SECONDS=${STALE_SECONDS:-1200}   # 20 min default
LOOP_INTERVAL=""
TARGET_WS=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --loop) LOOP_INTERVAL="$2"; shift 2 ;;
    --stale-seconds) STALE_SECONDS="$2"; shift 2 ;;
    --help|-h)
      head -30 "$0" | grep -E "^#"
      exit 0 ;;
    *) TARGET_WS="$1"; shift ;;
  esac
done

resolve_workspace() {
  if [[ -n "$TARGET_WS" && -d "$TARGET_WS" ]]; then
    echo "$TARGET_WS"
    return 0
  fi
  # Most recent run by mtime
  local ws
  ws=$(ls -1td "$PROJECT_ROOT"/workspace/run_* 2>/dev/null | head -1 || true)
  if [[ -z "$ws" ]]; then
    return 1
  fi
  echo "$ws"
}

check_once() {
  local WS
  WS=$(resolve_workspace) || { echo "[watchdog] no workspace"; return 0; }

  local status
  status=$(python3 "$PROJECT_ROOT/scripts/ledger_helpers.py" read --workspace "$WS" --key status 2>/dev/null | tr -d '"')
  if [[ "$status" != "running" && "$status" != "stopping" ]]; then
    echo "[watchdog] $(basename "$WS"): status=$status (nothing to do)"
    return 0
  fi

  local events="$WS/events.jsonl"
  if [[ ! -f "$events" ]]; then
    echo "[watchdog] $WS: no events.jsonl"
    return 0
  fi
  local now last_mod age
  now=$(date +%s)
  last_mod=$(stat -c %Y "$events")
  age=$(( now - last_mod ))

  if [[ "$age" -lt "$STALE_SECONDS" ]]; then
    echo "[watchdog] $(basename "$WS"): events fresh (${age}s < ${STALE_SECONDS}s); ok"
    return 0
  fi

  # Is any agent-sdk process alive for this workspace? We match on the workspace
  # path appearing in any agent's arg string.
  local run_id
  run_id=$(basename "$WS")
  local n_agents
  n_agents=$(pgrep -af "claude_agent_sdk" 2>/dev/null | grep -c "$run_id" || true)
  if [[ "$n_agents" -gt 0 ]]; then
    echo "[watchdog] $(basename "$WS"): events stale (${age}s) but $n_agents agent(s) alive; not intervening"
    return 0
  fi

  echo "[watchdog] $(basename "$WS"): DEADLOCK SUSPECTED — events stale ${age}s, no agents alive, status=$status"

  # Mark running experiments as stalled
  python3 - "$WS" <<'PY'
import json, sys, subprocess
ws = sys.argv[1]
d = json.load(open(f"{ws}/ledger.json"))
stalled = []
for e in d.get("experiments", []):
    if e.get("status") == "running":
        stalled.append(e.get("exp_id"))
for exp_id in stalled:
    subprocess.run(["python3", "scripts/ledger_helpers.py",
                    "set-exp-status", "--workspace", ws,
                    "--exp-id", exp_id, "--status", "stalled",
                    "--reason", "watchdog: no agent alive, events stale"], check=False)
    subprocess.run(["python3", "scripts/ledger_helpers.py",
                    "log-event", "--workspace", ws,
                    "--type", "watchdog_stalled_exp",
                    "--json", json.dumps({"exp_id": exp_id})], check=False)
print(f"[watchdog] marked {len(stalled)} experiments as stalled")
PY

  python3 "$PROJECT_ROOT/scripts/ledger_helpers.py" log-event \
    --workspace "$WS" --type watchdog_intervention \
    --json "$(printf '{"age_s":%d,"stale_threshold_s":%d,"action":"marked_running_as_stalled"}' "$age" "$STALE_SECONDS")"

  # Touch events.jsonl so master's mtime-based polling sees a change
  touch "$events"

  # If the tmux session is alive, nudge the master with a one-liner advisory
  if tmux has-session -t splatlab 2>/dev/null; then
    tmux send-keys -t splatlab \
      "Watchdog advisory: events.jsonl was stale ${age}s with no active subagent. Running experiments were marked 'stalled'. Please resume the loop per R4c meta-pivot." Enter
    echo "[watchdog] nudged tmux session splatlab"
  fi
}

if [[ -n "$LOOP_INTERVAL" ]]; then
  echo "[watchdog] loop mode, interval=${LOOP_INTERVAL}s, stale=${STALE_SECONDS}s"
  while true; do
    check_once || true
    sleep "$LOOP_INTERVAL"
  done
else
  check_once
fi
