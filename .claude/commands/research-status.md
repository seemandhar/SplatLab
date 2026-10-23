---
description: "Live view of the current (or most recent) autonomous research run"
---

# /research-status — summarize the current run

Find the most recent `workspace/run_*/ledger.json` (by mtime) and summarize it for the user.

Print (in this order, concise — not a wall of text):

1. **Header line:** `run_id`, `status` (running/stopping/done/failed), `wall_elapsed_s` as `HH:MM`, `stop_reason` if set.
2. **Target:** `user_intent` (trimmed to 100 chars), `paper_url` if set, `track` if decided.
3. **Baselines:** reported_metrics vs reproduced_metrics (one line per scene if tight; otherwise 2-3 scenes + "…N more").
4. **Ideas:** `tried=N queued=M rejected=K` with the top-3 tried ideas by PSNR (hypothesis + delta_vs_target).
5. **Experiments:** table of last 5 experiments: exp_id, status, scenes_done/total, avg_psnr, idea_id.
6. **Best:** `exp_id`, `+Δpsnr`, `+Δssim`, `scenes_beaten`.
7. **Budget:** `opus_tokens`, `sonnet_tokens`, `haiku_tokens` (M-scale), `gpu_hours`, `$ spent estimate`.
8. **Paper:** `last_written_at`, `last_verified_at`, `open_issues` count.
9. **Recent events:** tail the last 20 entries of `events.jsonl` as `[time] [agent] [type] summary`.

If there is no run (`workspace/run_*/` is empty), tell the user: "No runs yet. Start one with `/research \"your idea\"`."

If the run is `status=running` but the last event in `events.jsonl` is older than 30 min, add a warning: `⚠ no activity for X min — master may be stuck. Consider /research-stop and /research-resume.`

Keep the output under 40 lines. Use `jq` on the ledger rather than Python where possible (faster).
