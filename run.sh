#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════════════
# splatlab RUN SCRIPT
#
#   Edit the CONFIG section below, save, then:   bash run.sh
#
# Everything above the "DO NOT EDIT BELOW" line is yours. The bottom part
# patches config.yaml with your edits and launches the research team inside
# a detached tmux session. Safe to re-run — it refuses to launch if a
# session is already active.
# ═══════════════════════════════════════════════════════════════════════

set -euo pipefail

# ═══════════════════════════════════════════════════════════════════════
# 🖋  USER CONFIG  — edit these, then `bash run.sh`
# ═══════════════════════════════════════════════════════════════════════

# ── The research question (the only required field) ───────────────────
# Edit idea.txt next to this script (free-form prose, any quotes/dashes OK).
IDEA_FILE="$(dirname "${BASH_SOURCE[0]}")/idea.txt"
if [[ ! -f "$IDEA_FILE" ]]; then
  echo "ERROR: $IDEA_FILE not found — write the research question there." >&2
  exit 1
fi
IDEA="$(<"$IDEA_FILE")"

# ── Run duration ──────────────────────────────────────────────────────
HOURS=24                              # wall-clock cap in hours

# ── Optional targets (leave blank to let the lab pick its own) ────────
PAPER_URL="https://arxiv.org/abs/2408.16760"   # OmniRe (ICLR 2025), the DriveStudio paper
GIT_URL="https://github.com/ziyc/drivestudio"  # baseline github repo to reproduce + improve

# ── Cost control ──────────────────────────────────────────────────────
# Leave blank on Claude Max (no per-token billing). Only set a number if
# you are on pay-as-you-go API billing and want the master to self-restrain.
MAX_BUDGET_USD=""                     # empty = unlimited (Claude Max default)

# ── Experiment setup ──────────────────────────────────────────────────
DATASET="pandaset"                    # DriveStudio urban driving benchmark (self-downloaded)
SCENES="001,011,016,028,053,106,158"  # PandaSet dynamic sequence IDs (DriveStudio convention)
SCENES_QUICK="001"                    # sequence used for smoke-test + single-scene cascade
MAX_ITERS=30000                       # full-training iters for baselines / quality-match runs
CONDA_ENV="nerfstudio"
GPU_INDICES="0,1,2,3,4,5,6,7"   # full 8x A6000 — driver fixed, all GPUs free
CASCADE_ENABLED="true"                # smoke (500) → single-scene (7k) → full (30k)

# ── Agent behaviour ───────────────────────────────────────────────────
MODEL_PRESET="all_opus"               # 14 opus agents bumped 4-7 → 4-8 in .claude/agents/
PAPER_SCOPE="full"                # workshop | full
ALLOW_REPO_REPRODUCTION="true"        # clone + build + run the --git-url repo first

# ── tmux session name (change if running multiple in parallel) ────────
TMUX_SESSION="splatlab"

# ═══════════════════════════════════════════════════════════════════════
# ⚠  DO NOT EDIT BELOW THIS LINE (unless you know why)
# ═══════════════════════════════════════════════════════════════════════

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ── preflight checks ─────────────────────────────────────────────────
if [[ -z "$IDEA" ]]; then
  echo "ERROR: IDEA is empty — edit run.sh and set the research question." >&2
  exit 1
fi

for cmd in tmux claude python3; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "ERROR: required command '$cmd' not on PATH." >&2
    exit 1
  fi
done

if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
  echo "ERROR: tmux session '$TMUX_SESSION' already exists." >&2
  echo "       Attach:  tmux attach -t $TMUX_SESSION" >&2
  echo "       Kill:    tmux kill-session -t $TMUX_SESSION" >&2
  echo "       Or edit TMUX_SESSION in this script for a parallel run." >&2
  exit 1
fi

# ── patch config.yaml with user edits ────────────────────────────────
export IDEA HOURS PAPER_URL GIT_URL MAX_BUDGET_USD DATASET SCENES SCENES_QUICK \
       MAX_ITERS CONDA_ENV GPU_INDICES CASCADE_ENABLED MODEL_PRESET PAPER_SCOPE \
       ALLOW_REPO_REPRODUCTION

python3 <<'PY'
import os, yaml, pathlib

cfg_path = pathlib.Path("config.yaml")
cfg = yaml.safe_load(cfg_path.read_text()) or {}

def to_int_list(s):
    return [int(x.strip()) for x in s.split(",") if x.strip()]
def to_str_list(s):
    return [x.strip() for x in s.split(",") if x.strip()]
def to_bool(s):
    return str(s).lower() in ("1","true","yes","y","on")

cfg["idea"]                = os.environ["IDEA"]
cfg["hours"]               = int(os.environ["HOURS"])
cfg["paper_url"]           = os.environ.get("PAPER_URL", "") or ""
cfg["git_url"]             = os.environ.get("GIT_URL", "") or ""
mb = os.environ.get("MAX_BUDGET_USD", "").strip()
cfg["max_budget_usd"]      = int(mb) if mb else None
cfg["dataset"]             = os.environ["DATASET"]
cfg["scenes"]              = to_str_list(os.environ["SCENES"])
cfg["scenes_quick"]        = to_str_list(os.environ["SCENES_QUICK"])
cfg["max_iters"]           = int(os.environ["MAX_ITERS"])
cfg["conda_env"]           = os.environ["CONDA_ENV"]
cfg["gpu_indices"]         = to_int_list(os.environ["GPU_INDICES"])
cfg["cascade_enabled"]     = to_bool(os.environ["CASCADE_ENABLED"])
cfg["model_preset"]        = os.environ["MODEL_PRESET"]
cfg["paper_scope"]         = os.environ["PAPER_SCOPE"]
cfg["allow_repo_reproduction"] = to_bool(os.environ["ALLOW_REPO_REPRODUCTION"])

# keep stop_condition in sync with hours
sc = cfg.setdefault("stop_condition", {})
sc["wall_hours"] = cfg["hours"]

cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
print("config.yaml patched")
PY

# ── launch inside a detached tmux session ────────────────────────────
# Start Claude Code in the project dir with permissions bypassed (so the
# 24h run never stops at a confirmation prompt). The session stays alive
# after ssh disconnect; reattach with `tmux attach -t $TMUX_SESSION`.

tmux new-session -d -s "$TMUX_SESSION" -c "$PROJECT_ROOT" \
  "claude --dangerously-skip-permissions"

# wait for claude's prompt to initialize
sleep 10

# accept the folder-trust dialog if it shows (harmless if it doesn't)
tmux send-keys -t "$TMUX_SESSION" "1" Enter
sleep 3

# build the /research slash command
CMD="/research \"$IDEA\" --hours $HOURS"
[[ -n "$PAPER_URL" ]] && CMD="$CMD --paper-url \"$PAPER_URL\""
[[ -n "$GIT_URL"   ]] && CMD="$CMD --git-url \"$GIT_URL\""

# Long IDEA strings get treated by tmux as a multi-line paste, which opens
# the editor instead of submitting. Use load-buffer + paste-buffer so it
# arrives as one chunk, then send a final Enter to dispatch.
BUF_FILE="$(mktemp)"
printf '%s' "$CMD" > "$BUF_FILE"
tmux load-buffer -b splatlab-cmd "$BUF_FILE"
tmux paste-buffer -b splatlab-cmd -t "$TMUX_SESSION"
rm -f "$BUF_FILE"
sleep 2
tmux send-keys -t "$TMUX_SESSION" Enter
sleep 2
tmux send-keys -t "$TMUX_SESSION" Enter

# ── summary ──────────────────────────────────────────────────────────
cat <<EOF

═══════════════════════════════════════════════════════════════════
  splatlab launched in tmux session: $TMUX_SESSION
═══════════════════════════════════════════════════════════════════
  idea:              $IDEA
  hours cap:         $HOURS
  dollar cap:        ${MAX_BUDGET_USD:-(none — Claude Max)}
  model preset:      $MODEL_PRESET
  dataset:           $DATASET
  scenes:            $SCENES
  GPUs:              $GPU_INDICES
  paper_url:         ${PAPER_URL:-(none)}
  git_url:           ${GIT_URL:-(none)}

  ATTACH:            tmux attach -t $TMUX_SESSION
  DETACH:            (inside tmux)  Ctrl-b  then  d
  STATUS:            (inside claude) /research-status
  STOP GRACEFULLY:   (inside claude) /research-stop
  KILL SESSION:      tmux kill-session -t $TMUX_SESSION

  workspace:         $PROJECT_ROOT/workspace/run_<ts>/
  final PDF:         $PROJECT_ROOT/workspace/run_*/paper/main.pdf

  quota watch:       python scripts/max_quota.py --since 1d
═══════════════════════════════════════════════════════════════════
EOF
