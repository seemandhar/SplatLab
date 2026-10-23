#!/usr/bin/env bash
# Launch splatlab inside a persistent tmux session.
# Survives laptop close, ssh disconnect, Claude Code crash — attach later with
#   tmux attach -t splatlab
#
# Usage:
#   scripts/launch.sh                                      # uses config.yaml idea + hours
#   scripts/launch.sh "my idea"                            # override idea
#   scripts/launch.sh "my idea" 48                         # override idea + hours
#   scripts/launch.sh "my idea" 24 "https://arxiv.org/..." # override + paper URL
#   scripts/launch.sh "my idea" 24 "" "https://github.com/..."  # override + git URL

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# defaults from config.yaml (requires pyyaml)
DEFAULT_IDEA=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('idea','') or '')")
DEFAULT_HOURS=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('hours',24))")
DEFAULT_PAPER_URL=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('paper_url','') or '')")
DEFAULT_GIT_URL=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml')).get('git_url','') or '')")

IDEA="${1:-$DEFAULT_IDEA}"
HOURS="${2:-$DEFAULT_HOURS}"
PAPER_URL="${3:-$DEFAULT_PAPER_URL}"
GIT_URL="${4:-$DEFAULT_GIT_URL}"

if [[ -z "$IDEA" ]]; then
  echo "ERROR: no idea provided (and config.yaml idea is empty)." >&2
  echo "Usage: $0 \"<idea>\" [hours] [paper_url] [git_url]" >&2
  exit 1
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "ERROR: tmux is required. Install with: sudo apt install tmux" >&2
  exit 1
fi

if ! command -v claude >/dev/null 2>&1; then
  echo "ERROR: 'claude' CLI not found on PATH. Install Claude Code first." >&2
  exit 1
fi

SESSION="splatlab"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists." >&2
  echo "Attach with:  tmux attach -t $SESSION" >&2
  echo "Or kill it:   tmux kill-session -t $SESSION" >&2
  exit 1
fi

# Build the /research command to send
CMD="/research \"$IDEA\" --hours $HOURS"
if [[ -n "$PAPER_URL" ]]; then CMD="$CMD --paper-url \"$PAPER_URL\""; fi
if [[ -n "$GIT_URL" ]]; then CMD="$CMD --git-url \"$GIT_URL\""; fi

# Start claude in a detached tmux session with ALL permissions bypassed
# (--dangerously-skip-permissions + defaultMode: bypassPermissions in settings.json)
tmux new-session -d -s "$SESSION" -c "$PROJECT_ROOT" "claude --dangerously-skip-permissions"

# Wait for the prompt to appear (claude takes a few sec to bootstrap)
sleep 10

# First accept the trust-folder dialog if it appears, just in case (option 1)
tmux send-keys -t "$SESSION" "1" Enter
sleep 3

# Type the /research slash command and submit
tmux send-keys -t "$SESSION" "$CMD" Enter

MAX_BUDGET=$(python3 -c "import yaml; v=yaml.safe_load(open('config.yaml')).get('max_budget_usd'); print(f'\$'+str(int(v)) if v else '(none — Claude Max / wall-clock only)')")

cat <<EOF

═══════════════════════════════════════════════════════════════════
  splatlab launched
═══════════════════════════════════════════════════════════════════
  session:      $SESSION (tmux)
  idea:         $IDEA
  hours cap:    $HOURS  (authoritative wall-clock limit)
  dollar cap:   $MAX_BUDGET
  paper_url:    ${PAPER_URL:-(none)}
  git_url:      ${GIT_URL:-(none)}

  ATTACH:       tmux attach -t $SESSION
  DETACH:       (inside tmux)  Ctrl-b  then  d
  STATUS:       (inside claude) /research-status
  STOP:         (inside claude) /research-stop
  KILL:         tmux kill-session -t $SESSION

  mode:         --dangerously-skip-permissions (no confirm prompts)

  workspace:    $PROJECT_ROOT/workspace/run_<timestamp>/
  final pdf:    $PROJECT_ROOT/workspace/run_*/paper/main.pdf
═══════════════════════════════════════════════════════════════════
EOF
