#!/usr/bin/env bash
# Usage: scripts/compile_latex.sh <paper_dir>
# Compiles main.tex via pdflatex+bibtex 3x, writes paper.pdf, returns 0 on success.
set -euo pipefail

PAPER_DIR="${1:?paper_dir is required}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LATEX_DIR="$PROJECT_ROOT/latex"

cd "$PAPER_DIR"

if [[ ! -f main.tex ]]; then
  echo "error: $PAPER_DIR/main.tex does not exist" >&2
  exit 2
fi

# Stage CVPR style files + bibstyle + preamble. These files live in
# $PROJECT_ROOT/latex/ (real CVPR 2026 author-kit copies) and must be
# alongside main.tex at compile time.
for f in cvpr.sty ieeenat_fullname.bst preamble.tex; do
  if [[ -f "$LATEX_DIR/$f" ]]; then
    cp -f "$LATEX_DIR/$f" "$PAPER_DIR/$f"
  else
    echo "warn: $LATEX_DIR/$f missing — compile may fail or fall back to basic article style" >&2
  fi
done

LOG=build.log
: > "$LOG"

run() {
  echo ">>> $*" >> "$LOG"
  "$@" >> "$LOG" 2>&1 || { tail -60 "$LOG" >&2; return 1; }
}

# First pass: generate aux for bibtex
run pdflatex -interaction=nonstopmode -halt-on-error main.tex
# Bibtex if references.bib present
if [[ -f references.bib ]]; then
  run bibtex main || true   # bibtex may complain non-fatally
fi
run pdflatex -interaction=nonstopmode -halt-on-error main.tex
run pdflatex -interaction=nonstopmode -halt-on-error main.tex

if [[ ! -f main.pdf ]]; then
  echo "error: main.pdf not produced" >&2
  tail -120 "$LOG" >&2
  exit 3
fi

# Strip agent comments and final clean pass (optional)
if command -v chktex >/dev/null 2>&1; then
  chktex -q main.tex >> "$LOG" 2>&1 || true
fi

SIZE=$(stat -c %s main.pdf)
if [[ "$SIZE" -lt 10240 ]]; then
  echo "error: main.pdf is suspiciously small ($SIZE bytes)" >&2
  exit 4
fi

echo "$PAPER_DIR/main.pdf"
