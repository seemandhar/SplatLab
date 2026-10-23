#!/usr/bin/env bash
# Usage: scripts/init_workspace.sh "<idea>" [paper_url] [git_url]
# Creates workspace/run_<utc_ts>/ with a seeded ledger and symlinks to shared assets.
set -euo pipefail

IDEA="${1:?idea is required}"
PAPER_URL="${2:-}"
GIT_URL="${3:-}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

TS="$(date -u +%Y%m%d_%H%M%S)"
WS="$PROJECT_ROOT/workspace/run_$TS"
mkdir -p "$WS"/{paper,paper/sections,figures,implementation,training_outputs,sota,logs}

# Symlinks to shared assets so subagents have stable paths
ln -sfn "$PROJECT_ROOT/literature" "$WS/literature"
ln -sfn "$PROJECT_ROOT/splatify" "$WS/splatify"
ln -sfn "$PROJECT_ROOT/latex/template.tex" "$WS/latex_template.tex"
ln -sfn "$PROJECT_ROOT/latex" "$WS/latex"
ln -sfn "$PROJECT_ROOT/refs/references.bib" "$WS/references.bib"
ln -sfn "$PROJECT_ROOT/discoveries.txt" "$WS/discoveries.txt"
ln -sfn "$PROJECT_ROOT/novelty.txt" "$WS/novelty.txt"
ln -sfn "$PROJECT_ROOT/initial_ideas.md" "$WS/initial_ideas.md"
ln -sfn "$PROJECT_ROOT/literature_review.md" "$WS/literature_review.md"
ln -sfn "$PROJECT_ROOT/paper_metrics.md" "$WS/paper_metrics.md"

# Seed ledger via helper
python "$PROJECT_ROOT/scripts/ledger_helpers.py" init \
  --workspace "$WS" \
  --user-intent "$IDEA" \
  ${PAPER_URL:+--paper-url "$PAPER_URL"} \
  ${GIT_URL:+--git-url "$GIT_URL"} \
  --config "$PROJECT_ROOT/config.yaml"

echo "$WS"
