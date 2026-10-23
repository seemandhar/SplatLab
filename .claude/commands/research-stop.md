---
description: "Gracefully stop the current research run; in-flight experiments finish"
---

# /research-stop — graceful halt

Find the most recent `workspace/run_*/ledger.json` with `status in {running, stopping}`.

1. **Set the stop signal.** Use `scripts/ledger_helpers.py` to atomically set:
   - `status: "stopping"`
   - `stop_reason: "user_requested"`
2. **Tell the user:** "Stop signal written to ledger. The master will check at its next iteration (≤ 5 min) and begin graceful shutdown. In-flight trainer subagents will finish their current scene; the master will not start new experiments."
3. **Ask the user** if they want to wait: if yes, poll `ledger.status` every 60s and print a dot; when it turns `done` or `stopped`, print the final summary (same as `/research-status`).

If there's no running run, tell the user and exit.

Do NOT hard-kill anything. Trainers are bounded by their own timeouts. If the user wants to hard-kill, they can `Ctrl-C` the main session and then manually `pkill -f ns-train` — but we do not do this from here because it leaves the ledger inconsistent.
