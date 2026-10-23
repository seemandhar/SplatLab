---
description: "Resume a previously stopped or crashed run from its last ledger checkpoint"
---

# /research-resume — continue the most recent run

Find the most recent `workspace/run_*/ledger.json`. If none, tell the user and exit.

1. **Read the ledger.** Decide whether the run is resumable:
   - `status in {stopping, stopped, failed}` → resumable; flip to `running`.
   - `status == done` → not resumable; tell the user it's already complete. Paper path is in `paper.tex_path`.
   - `status == running` but last event in `events.jsonl` is older than 30 min → crashed master; resumable, flip to `running`.
   - `status == running` with recent events → already live; tell the user; offer `/research-status`.

2. **Heal experiment state.** Any experiment with `status in {queued, running}` but no heartbeat in 30 min → mark `stalled`. The master will re-queue or skip on the next tick.

3. **Patch status + set resumed_at** in the ledger via `scripts/ledger_helpers.py patch`. Log `run_resumed` event.

4. **Become the master.** The master-orchestrator logic lives in `.claude/commands/research.md`. Read that file and execute it from Phase 1 onward (skip Phase 0 bootstrap — workspace already exists). You ARE the master in this main session; there is no separate master subagent. Auto-compact will handle your 24-hour context.

Do NOT re-initialize the workspace. Do NOT overwrite `ledger.json` (except the atomic `status` flip). The whole point of resume is that the ledger is the source of truth.
