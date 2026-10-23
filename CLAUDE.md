# SplatLab

Autonomous Gaussian-Splatting research pipeline — **driven by Claude Code**. One slash command dispatches a long-lived `master-orchestrator` subagent that runs for 24+ hours, spawning a team of ~24 specialized research subagents in parallel, all coordinated through a durable `ledger.json`. Produces a real LaTeX PDF.

Port of the original Python + Claude Agent SDK `paper-writer` onto Claude Code's native primitives (subagents, slash commands, Task tool, ScheduleWakeup).

## Quick start

```bash
cd splatlab
# edit config.yaml if desired (hours, model, scenes, budget)
claude
> /research "your research idea here" --hours 24
```

Output lands in `workspace/run_<timestamp>/paper/main.pdf`.

## Architecture

**Master orchestrator = the MAIN Claude Code session.** When you run `/research "idea" --hours 24`, the slash command's prompt BECOMES the main session's instructions: read ledger → decide → dispatch specialist subagents via `Task` → read results → update ledger → sleep/loop → repeat until a stop condition fires. Auto-compact handles the 24-hour context window. State lives in `ledger.json`, never in context.

Why not a Task subagent for the master? Claude Code subagents cannot spawn other subagents (Task tool is unavailable inside Task-dispatched agents). The master NEEDS to dispatch specialists, so it must be the main session.

**Research team** (`.claude/agents/*.md`) — 26 specialized subagents, each a markdown file with YAML frontmatter (name, description, tools, model) plus the full system prompt. The main session dispatches them via the `Task` tool, optionally in parallel and/or background.

**Durable state** — `workspace/<run_id>/ledger.json` (atomic writes via `scripts/ledger_helpers.py`) + `events.jsonl` append-only log. Survives SIGKILL, Claude Code restarts, and context compaction.

**Two execution tracks:**
- **Track A (primary):** clone the target SOTA repo, reproduce numbers, map 9 EVOLVE-BLOCK hooks, iterate via `implementer` + `trainer` on the real repo.
- **Track B (fallback):** use the local `splatify/gs_template` (nerfstudio-compatible) via the same `implementer` + `trainer` chain. Always available, even when Track A is working — good for independent verification.

## Reference corpora (read by the agents)

| File/Dir | Purpose |
|---|---|
| `discoveries.txt` | 994-line catalog of 25 GS methods + innovation taxonomy. Ideator must read. |
| `novelty.txt` | Methodology/contribution details. For method_writer. |
| `initial_ideas.md` | Seed ideas with hypotheses. |
| `literature_review.md` | Existing landscape summary. |
| `paper_metrics.md` | Metric definitions. |
| `reproduction_results.md` | Prior reproduction attempts. |
| `latex/template.tex` | CVPR-style paper template (13KB). |
| `refs/references.bib` | BibTeX corpus. |
| `splatify/` | Full evolutionary system: gs_template (9 EVOLVE-BLOCKs + methods/), eoh.py, elo_gradients.py, evolve_blocks.py, cascade.py, falsify.py, eval.py, invent.py. |
| `literature/` | 145 paper summaries + 11 MineRU extractions + cached PDFs. Shared corpus. Researcher reads this FIRST. |

## The research team

| Subagent | Role | Model tier |
|---|---|---|
| *(main session)* | Senior PI running the lab for 24h; `/research` invokes the full master prompt | opus |
| `researcher` | Dr. Ada — literature review, library-first, fills gaps from web | sonnet |
| `pdf-harvester` | MineRU PDF→MD extraction for new papers | sonnet |
| `repo-reproducer` | **Clones any GitHub baseline, builds env, runs it, reports metrics** | sonnet |
| `evolve-block-mapper` | Annotates a cloned repo with the 9 hook points | sonnet |
| `ideator` | Prof. Kai — Sakana-style archive-aware hypothesis generator | opus |
| `web-idea-scout` | Scans arXiv 2024-2025 for directions the ideator missed | sonnet |
| `implementer` | Dr. Elena — 7-in-one: parse→plan→code→review→validate→integrate→test | opus |
| `debugger` | Dr. Bruno — reads tracebacks, fixes bugs iteratively | sonnet |
| `freeform-mutator` | Track B unified-diff edits on the SOTA worktree | sonnet |
| `smoke-trainer` | 500-iter filter (mandatory before any full training run) | haiku |
| `trainer` | Dr. Mira — runs ns-train / ns-eval / external repos; 2h timeout | sonnet |
| `baseline-runner` | Trains splatfacto reference + computes noise floor | sonnet |
| `visual-comparator` | Side-by-side renders: ours vs baseline on test views | sonnet |
| `ablation-runner` | Leave-one-component-out per sub-innovation | opus |
| `failure-case-analyst` | Finds worst scenes/views, analyzes why | sonnet |
| `figure-generator` | PaperBanana + matplotlib diagrams, pipeline figures, tables | sonnet |
| `teaser-writer` | 1-page teaser image with method name + key visual | sonnet |
| `summary-writer` | Dr. Sam — abstract + introduction + conclusion, consistent voice | opus |
| `body-writer` | Method + experiments sections, cross-consistent with ledger | opus |
| `related-work-writer` | Prof. Noa — ≥40 citations from library, grouped thematically | opus |
| `latex-assembler` | Assembles main.tex, runs pdflatex 3x + bibtex, strips agent comments | sonnet |
| `paper-quality-checker` | Verifies every claim against ledger, kills hallucinations | sonnet |
| `reviewer-a` | Prof. Yuki — senior CVPR methodology + novelty | opus |
| `reviewer-b` | 3DGS domain expert — fair baselines, known trade-offs | opus |
| `devils-advocate` | Dr. Alex — adversarial, finds every weakness | opus |
| `revision-agent` | Addresses merged reviewer feedback | opus |

## Slash commands

- `/research "idea" [--hours N] [--paper-url URL] [--git-url URL]` — init workspace + become the master in this session; runs continuously until stop condition
- `/research-status` — tail last 50 events + show best experiment
- `/research-stop` — write stop signal into ledger; master sees it next iteration, finishes in-flight work, halts
- `/research-resume` — pick up a stopped/crashed run; continues from the ledger

## Launch (tmux-persistent, zero-prompt)

```bash
bash scripts/launch.sh "idea" 24
```

Starts Claude Code in a tmux session with `--dangerously-skip-permissions`, auto-accepts the trust dialog, types the `/research` command. Session survives laptop close / ssh disconnect.

## External dependencies

- `conda env: nerfstudio` (ns-train, ns-eval, ns-render)
- `conda env: mineru` (PDF → MD)
- `pdflatex`, `bibtex` (TeXLive)
- Mip-NeRF 360 at `data/mipnerf360`
- CUDA + at least 1 GPU

## Observability

- `workspace/<run_id>/ledger.json` — full durable state (read with `jq`)
- `workspace/<run_id>/events.jsonl` — append-only event log
- `/research-status` — live view

## Stopping

- Wall-clock hit `hours` in config.yaml → master sets `status=done`
- Cumulative cost hit `max_budget_usd` → same
- Master declares done (3+ reviewer rounds converged, paper verified)
- User: `/research-stop` or `Ctrl-C` in the main session
- Process kill: ledger survives; `/research-resume` picks up where it left off
