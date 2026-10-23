#!/usr/bin/env python3
"""Atomic read/write helpers for the splatlab ledger.

The ledger is `workspace/<run_id>/ledger.json`. Every mutation goes through
this script so the format is consistent and concurrent writes (from the master
or parallel subagents) cannot corrupt it.

CLI entry points (used by slash commands and subagents):

    init   --workspace PATH --user-intent "..." [--paper-url URL] [--git-url URL] --config config.yaml
    read   --workspace PATH [--key dotted.path]
    patch  --workspace PATH --json '{...}'         # deep-merge into ledger
    set    --workspace PATH --key dotted.path --value VALUE
    add-idea --workspace PATH --hypothesis "..." --rationale "..." [--hook HOOK] [--inspired-by k1,k2]
    add-experiment --workspace PATH --exp-id ID --idea-id ID --worktree PATH --scene NAME
    update-experiment --workspace PATH --exp-id ID --json '{...}'
    set-best --workspace PATH --exp-id ID --json '{metrics:..., delta_vs_target:...}'
    log-event --workspace PATH --type TYPE --json '{...}'    # appends to events.jsonl
    set-status --workspace PATH --status {running|stopping|stopped|done|failed} [--reason TEXT]
    compact --workspace PATH [--keep-last-n 8]
    tail-events --workspace PATH [--n 50]

All writes go through _atomic_update(): file-lock (fcntl.flock) + in-memory lock +
tmp-file + os.replace. Readers get a consistent snapshot.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
_LOCK = threading.Lock()


def _utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _hash_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


@contextlib.contextmanager
def _file_lock(workspace: Path):
    workspace.mkdir(parents=True, exist_ok=True)
    path = workspace / ".ledger.lock"
    with path.open("a+") as fh:
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            yield
        finally:
            with contextlib.suppress(Exception):
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _read_raw(workspace: Path) -> dict:
    return json.loads((workspace / "ledger.json").read_text())


def _write_raw(workspace: Path, state: dict) -> None:
    tmp = workspace / "ledger.json.tmp"
    tmp.write_text(json.dumps(state, indent=2, default=str))
    os.replace(tmp, workspace / "ledger.json")


def _atomic_update(workspace: Path, mutator):
    with _file_lock(workspace), _LOCK:
        state = _read_raw(workspace)
        result = mutator(state)
        _write_raw(workspace, state)
        return result


def _deep_merge(dst: dict, src: dict) -> None:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v


def _set_by_path(state: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    cur = state
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _get_by_path(state: dict, dotted: str) -> Any:
    cur = state
    for k in dotted.split("."):
        cur = cur[k] if isinstance(cur, dict) and k in cur else None
        if cur is None:
            return None
    return cur


# ─────────────────────────────────────────────────────────────────────
# init: build a fresh ledger from config + CLI args
# ─────────────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    if not path or not Path(path).exists():
        return {}
    try:
        import yaml  # type: ignore
        return yaml.safe_load(Path(path).read_text()) or {}
    except ImportError:
        sys.stderr.write("warning: PyYAML not installed; ignoring config.yaml\n")
        return {}


def cmd_init(args: argparse.Namespace) -> None:
    ws = Path(args.workspace).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    cfg = _load_config(args.config)
    run_id = ws.name
    stop = cfg.get("stop_condition") or {}
    initial = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "workspace_path": str(ws),
        "created_at": _utc_iso(),
        "resumed_at": None,
        "target": {
            "user_intent": args.user_intent,
            "paper_url": args.paper_url or cfg.get("paper_url") or None,
            "git_url": args.git_url or cfg.get("git_url") or None,
            "pdf_md_path": None,
            "reported_metrics": {},
            "peer_baselines": {},
            "reproduced_metrics": None,
            "reproduced_at": None,
            "evolve_block_mapping": None,
            "track": None,
        },
        "stop_condition": {
            "avg_psnr_delta": float(stop.get("avg_psnr_delta", 0.5)),
            "avg_ssim_delta": float(stop.get("avg_ssim_delta", 0.005)),
            "min_scenes_beaten": int(stop.get("min_scenes_beaten", 3)),
            "wall_hours": float(cfg.get("hours", stop.get("wall_hours", 24))),
            # Optional API-metered caps. Claude Max users leave these as None
            # and the master skips the corresponding stop checks.
            "opus_token_budget": (
                int(stop["opus_token_budget"])
                if stop.get("opus_token_budget") is not None else None
            ),
            "max_budget_usd": (
                float(cfg["max_budget_usd"])
                if cfg.get("max_budget_usd") is not None else None
            ),
        },
        "config_snapshot": cfg,
        "ideas": {"tried": [], "queued": [], "rejected": []},
        "experiments": [],
        "best": None,
        "budget": {
            "wall_elapsed_s": 0,
            "opus_tokens": 0,
            "sonnet_tokens": 0,
            "haiku_tokens": 0,
            "gpu_hours": 0.0,
            "usd_spent_est": 0.0,
            "started_at": _utc_iso(),
        },
        "paper": {
            "last_written_at": None,
            "last_verified_at": None,
            "tex_path": None,
            "pdf_path": None,
            "open_issues": [],
            "reviewer_notes": None,
        },
        "phase_tokens": {},
        "category_stats": {},
        "ticks": [],
        "status": "running",
        "stop_reason": None,
        "noise_floor": {"psnr_std": 0.1, "measured": False},
    }
    _write_raw(ws, initial)
    # seed events
    events = ws / "events.jsonl"
    events.write_text(
        json.dumps({"ts": _utc_iso(), "type": "run_initialized", "run_id": run_id,
                    "user_intent": args.user_intent}) + "\n"
    )
    print(str(ws))


def cmd_read(args: argparse.Namespace) -> None:
    state = _read_raw(Path(args.workspace))
    if args.key:
        print(json.dumps(_get_by_path(state, args.key), indent=2, default=str))
    else:
        print(json.dumps(state, indent=2, default=str))


def cmd_patch(args: argparse.Namespace) -> None:
    patch = json.loads(args.json)

    def _do(state):
        _deep_merge(state, patch)

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_set(args: argparse.Namespace) -> None:
    try:
        value = json.loads(args.value)
    except json.JSONDecodeError:
        value = args.value

    def _do(state):
        _set_by_path(state, args.key, value)

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_add_idea(args: argparse.Namespace) -> None:
    def _do(state):
        h = _hash_text(args.hypothesis)
        for bucket in ("tried", "queued", "rejected"):
            for e in state["ideas"][bucket]:
                if e.get("hash") == h:
                    return e["id"]
        idea_id = f"idea_{len(state['ideas']['queued']) + len(state['ideas']['tried']):04d}"
        entry = {
            "id": idea_id,
            "hash": h,
            "hypothesis": args.hypothesis,
            "rationale": args.rationale,
            "hook": args.hook or "",
            "inspired_by": (args.inspired_by or "").split(",") if args.inspired_by else [],
            "created_at": _utc_iso(),
        }
        if args.mechanism_class:
            entry["mechanism_class"] = args.mechanism_class
        if args.addresses_bottleneck:
            entry["addresses_bottleneck"] = args.addresses_bottleneck
        if args.addresses_limitation:
            entry["addresses_limitation"] = args.addresses_limitation
        if args.addresses_paper_limitation:
            entry["addresses_paper_limitation"] = args.addresses_paper_limitation
        if args.category:
            entry["category"] = args.category
        if args.expected_psnr_gain is not None:
            entry["expected_psnr_gain"] = args.expected_psnr_gain
        if args.extra_json:
            try:
                extra = json.loads(args.extra_json)
                if isinstance(extra, dict):
                    entry.update({k: v for k, v in extra.items() if k not in entry or k in ("hook", "inspired_by")})
            except json.JSONDecodeError:
                pass
        state["ideas"]["queued"].append(entry)
        return idea_id

    idea_id = _atomic_update(Path(args.workspace), _do)
    print(idea_id)


def cmd_add_experiment(args: argparse.Namespace) -> None:
    def _do(state):
        state["experiments"].append({
            "exp_id": args.exp_id,
            "idea_id": args.idea_id,
            "worktree": args.worktree,
            "scene": args.scene,
            "status": "queued",
            "metrics_per_scene": {},
            "started_at": None,
            "finished_at": None,
            "log_path": None,
        })

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_update_experiment(args: argparse.Namespace) -> None:
    fields = json.loads(args.json)

    def _do(state):
        for exp in state["experiments"]:
            if exp["exp_id"] == args.exp_id:
                exp.update(fields)
                return

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_set_best(args: argparse.Namespace) -> None:
    body = json.loads(args.json)

    def _do(state):
        state["best"] = {
            "exp_id": args.exp_id,
            "metrics": body.get("metrics"),
            "delta_vs_target": body.get("delta_vs_target"),
            "set_at": _utc_iso(),
        }

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_set_status(args: argparse.Namespace) -> None:
    def _do(state):
        state["status"] = args.status
        if args.reason is not None:
            state["stop_reason"] = args.reason

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_log_event(args: argparse.Namespace) -> None:
    body = json.loads(args.json) if args.json else {}
    body.update({"ts": _utc_iso(), "type": args.type})
    events = Path(args.workspace) / "events.jsonl"
    with events.open("a") as fh:
        fh.write(json.dumps(body, default=str) + "\n")
    print("ok")


def cmd_compact(args: argparse.Namespace) -> None:
    keep = int(args.keep_last_n or 8)

    def _do(state):
        exps = state.get("experiments") or []
        if len(exps) <= keep + 2:
            return
        best_id = ((state.get("best") or {}).get("exp_id"))
        active = [e for e in exps if e.get("status") in ("running", "queued", "pending")]
        completed = [e for e in exps if e not in active]
        completed_sorted = sorted(completed, key=lambda e: e.get("started_at") or "", reverse=True)
        kept = list(active)
        seen = {e["exp_id"] for e in kept}
        for e in completed_sorted[:keep]:
            if e["exp_id"] not in seen:
                kept.append(e)
                seen.add(e["exp_id"])
        if best_id and best_id not in seen:
            best = next((e for e in exps if e["exp_id"] == best_id), None)
            if best:
                kept.append(best)
                seen.add(best_id)
        pruned = [e for e in exps if e["exp_id"] not in seen]
        if not pruned:
            return
        summary = state.get("experiments_summary") or {"pruned_count": 0, "ideas_tried": []}
        summary["pruned_count"] = summary.get("pruned_count", 0) + len(pruned)
        ideas = set(summary.get("ideas_tried") or [])
        for e in pruned:
            if e.get("idea_id"):
                ideas.add(e["idea_id"])
        summary["ideas_tried"] = sorted(ideas)
        state["experiments_summary"] = summary
        state["experiments"] = kept

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_tail_events(args: argparse.Namespace) -> None:
    events = Path(args.workspace) / "events.jsonl"
    if not events.exists():
        return
    n = int(args.n or 50)
    lines = events.read_text().splitlines()
    for line in lines[-n:]:
        print(line)


def cmd_tick_wall(args: argparse.Namespace) -> None:
    """Recompute wall_elapsed_s from budget.started_at and the system clock.

    Master must call this at the top of every loop iteration so wall-clock
    discipline (hours_remaining gates) fires on time. Cheap: one clock read.
    """
    def _do(state):
        started = state.get("budget", {}).get("started_at")
        if not started:
            return
        try:
            t0 = datetime.fromisoformat(started.replace("Z", "+00:00")).timestamp()
        except Exception:
            return
        state.setdefault("budget", {})["wall_elapsed_s"] = int(time.time() - t0)

    _atomic_update(Path(args.workspace), _do)
    # Echo elapsed so the caller doesn't need another read
    state = _read_raw(Path(args.workspace))
    print(state.get("budget", {}).get("wall_elapsed_s", 0))


def cmd_set_exp_status(args: argparse.Namespace) -> None:
    """Set experiment.status atomically with an optional reason.

    Use this instead of update-experiment when you just want to flip
    status (smoke_passed → running → done / fail-* / stalled). Overwrites
    any prior status (fixes the stale-state bug where done events landed
    but the experiments[] record stayed at smoke_passed).
    """
    def _do(state):
        for exp in state["experiments"]:
            if exp.get("exp_id") == args.exp_id:
                exp["status"] = args.status
                if args.reason:
                    exp["status_reason"] = args.reason
                if args.status == "done" and not exp.get("finished_at"):
                    exp["finished_at"] = _utc_iso()
                elif args.status == "running" and not exp.get("started_at"):
                    exp["started_at"] = _utc_iso()
                return

    _atomic_update(Path(args.workspace), _do)
    print("ok")


def cmd_record_category_outcome(args: argparse.Namespace) -> None:
    """Update category_stats[hook] with a new experimental outcome.

    Ideator reads this to apply saturation rules (wins==0 AND tried>=5 →
    skip the hook). Master must call this on every experiment_done event.
    """
    hook = args.hook or "HOOK_UNKNOWN"
    delta = float(args.delta_psnr) if args.delta_psnr is not None else None
    beat = str(args.beat_noise_floor).lower() in ("1", "true", "yes")

    def _do(state):
        stats = state.setdefault("category_stats", {}) or {}
        entry = stats.get(hook) or {
            "tried": 0, "wins": 0, "best_delta": None, "avg_delta": 0.0, "saturated": False,
        }
        tried_new = int(entry.get("tried", 0)) + 1
        wins_new = int(entry.get("wins", 0)) + (1 if beat else 0)
        if delta is not None:
            best = entry.get("best_delta")
            entry["best_delta"] = delta if best is None else max(float(best), delta)
            avg_old = float(entry.get("avg_delta", 0.0))
            entry["avg_delta"] = (avg_old * (tried_new - 1) + delta) / tried_new
        entry["tried"] = tried_new
        entry["wins"] = wins_new
        # Category-8 green-field gets a longer runway; others saturate at 5
        sat_thresh = 10 if hook.upper() in ("HOOK_8", "GREEN_FIELD", "NOVEL_INVENTION") else 5
        entry["saturated"] = (tried_new >= sat_thresh and wins_new == 0)
        stats[hook] = entry
        state["category_stats"] = stats

    _atomic_update(Path(args.workspace), _do)
    print("ok")


# ── Cross-run memory ─────────────────────────────────────────────────
# Persistent across runs. Lives at literature/cross_run_memory.json and
# records category saturation, tried mechanisms, and winning patterns per
# regime so a fresh run starts smarter than cold.

def _cross_run_path(workspace: Path) -> Path:
    # literature is a symlink into the project-wide shared library
    lit = workspace / "literature"
    if lit.is_symlink() or lit.is_dir():
        return lit.resolve() / "cross_run_memory.json"
    return workspace.parent / "literature" / "cross_run_memory.json"


def cmd_xrun_read(args: argparse.Namespace) -> None:
    """Read cross-run memory filtered by regime. Prints JSON."""
    path = _cross_run_path(Path(args.workspace))
    if not path.exists():
        print("{}")
        return
    with _file_lock(path.parent):
        data = json.loads(path.read_text() or "{}")
    regime = args.regime
    if regime:
        entry = data.get("regimes", {}).get(regime) or {}
        print(json.dumps(entry, indent=2))
    else:
        print(json.dumps(data, indent=2, default=str))


def cmd_xrun_write(args: argparse.Namespace) -> None:
    """Merge this run's findings into cross-run memory under args.regime."""
    ws = Path(args.workspace)
    state = _read_raw(ws)
    regime = args.regime or state.get("config_snapshot", {}).get("regime_hint") or "unknown"

    # Gather this run's contribution
    cat_stats = state.get("category_stats") or {}
    best = state.get("best") or {}
    tried = [e.get("idea_id") for e in state.get("experiments", []) if e.get("idea_id")]
    winners = [
        {
            "exp_id": e.get("exp_id"),
            "idea_id": e.get("idea_id"),
            "method_name": e.get("method_name"),
            "avg_psnr": (e.get("metrics_per_scene") or {}),
            "delta_vs_target": e.get("delta_vs_target"),
        }
        for e in state.get("experiments", [])
        if e.get("status") == "done"
    ]

    path = _cross_run_path(ws)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock(path.parent):
        if path.exists():
            memory = json.loads(path.read_text() or "{}")
        else:
            memory = {"schema_version": 1, "regimes": {}}
        rm = memory.setdefault("regimes", {}).setdefault(regime, {
            "first_seen": _utc_iso(),
            "run_count": 0,
            "category_stats_aggregate": {},
            "winning_patterns": [],
            "tried_ideas": [],
            "best_ever": None,
        })
        # merge category stats (sum tried/wins, take max best_delta)
        agg = rm["category_stats_aggregate"]
        for hook, entry in cat_stats.items():
            acc = agg.get(hook) or {"tried": 0, "wins": 0, "best_delta": None}
            acc["tried"] += int(entry.get("tried", 0))
            acc["wins"] += int(entry.get("wins", 0))
            bd = entry.get("best_delta")
            if bd is not None:
                acc["best_delta"] = bd if acc["best_delta"] is None else max(acc["best_delta"], bd)
            agg[hook] = acc
        # append tried idea ids (deduped)
        tried_set = set(rm.get("tried_ideas") or []) | set(tried)
        rm["tried_ideas"] = sorted(tried_set)
        # append winners
        rm["winning_patterns"].extend(winners)
        rm["winning_patterns"] = rm["winning_patterns"][-50:]  # keep last 50
        # update best_ever
        best_delta = (best.get("delta_vs_target") or {}).get("avg_psnr")
        if best_delta is not None:
            prior = (rm.get("best_ever") or {}).get("avg_psnr_delta")
            if prior is None or float(best_delta) > float(prior):
                rm["best_ever"] = {
                    "avg_psnr_delta": float(best_delta),
                    "exp_id": best.get("exp_id"),
                    "run_id": state.get("run_id"),
                    "recorded_at": _utc_iso(),
                }
        rm["run_count"] += 1
        rm["last_updated"] = _utc_iso()
        memory["last_updated"] = _utc_iso()

        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(memory, indent=2, default=str))
        os.replace(tmp, path)
    print(str(path))


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("init")
    p.add_argument("--workspace", required=True)
    p.add_argument("--user-intent", required=True)
    p.add_argument("--paper-url", default=None)
    p.add_argument("--git-url", default=None)
    p.add_argument("--config", default="config.yaml")
    p.set_defaults(fn=cmd_init)

    p = sub.add_parser("read")
    p.add_argument("--workspace", required=True)
    p.add_argument("--key", default=None)
    p.set_defaults(fn=cmd_read)

    p = sub.add_parser("patch")
    p.add_argument("--workspace", required=True)
    p.add_argument("--json", required=True)
    p.set_defaults(fn=cmd_patch)

    p = sub.add_parser("set")
    p.add_argument("--workspace", required=True)
    p.add_argument("--key", required=True)
    p.add_argument("--value", required=True)
    p.set_defaults(fn=cmd_set)

    p = sub.add_parser("add-idea")
    p.add_argument("--workspace", required=True)
    p.add_argument("--hypothesis", required=True)
    p.add_argument("--rationale", required=True)
    p.add_argument("--hook", default="")
    p.add_argument("--inspired-by", default="")
    p.add_argument("--mechanism-class", default="",
                   help="Research-class: new_loss_family | new_strategy | new_primitive | new_compositing | new_geometry_coupling | new_graph_topology | new_rendering_pipeline. Tuning-class (auto-reject): grad_rescale | loss_reweight | schedule_tweak | param_gate | init_tweak | threshold_reparam.")
    p.add_argument("--addresses-bottleneck", default="", help="B-ID (e.g. B-03)")
    p.add_argument("--addresses-limitation", default="", help="L-ID (e.g. L-05)")
    p.add_argument("--addresses-paper-limitation", default="", help="PL-ID (e.g. PL-1 or PL-yu2024_gsdf-02)")
    p.add_argument("--category", default="", help="1..8 novelty category")
    p.add_argument("--expected-psnr-gain", type=float, default=None)
    p.add_argument("--extra-json", default="",
                   help="Optional JSON dict of additional fields to merge into the idea entry (e.g. mechanism_derivation, target_file_and_line, novelty_check).")
    p.set_defaults(fn=cmd_add_idea)

    p = sub.add_parser("add-experiment")
    p.add_argument("--workspace", required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--idea-id", required=True)
    p.add_argument("--worktree", required=True)
    p.add_argument("--scene", required=True)
    p.set_defaults(fn=cmd_add_experiment)

    p = sub.add_parser("update-experiment")
    p.add_argument("--workspace", required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--json", required=True)
    p.set_defaults(fn=cmd_update_experiment)

    p = sub.add_parser("set-best")
    p.add_argument("--workspace", required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--json", required=True)
    p.set_defaults(fn=cmd_set_best)

    p = sub.add_parser("set-status")
    p.add_argument("--workspace", required=True)
    p.add_argument("--status", required=True, choices=["running", "stopping", "stopped", "done", "failed"])
    p.add_argument("--reason", default=None)
    p.set_defaults(fn=cmd_set_status)

    p = sub.add_parser("log-event")
    p.add_argument("--workspace", required=True)
    p.add_argument("--type", required=True)
    p.add_argument("--json", default="{}")
    p.set_defaults(fn=cmd_log_event)

    p = sub.add_parser("compact")
    p.add_argument("--workspace", required=True)
    p.add_argument("--keep-last-n", default="8")
    p.set_defaults(fn=cmd_compact)

    p = sub.add_parser("tail-events")
    p.add_argument("--workspace", required=True)
    p.add_argument("--n", default="50")
    p.set_defaults(fn=cmd_tail_events)

    p = sub.add_parser("tick-wall")
    p.add_argument("--workspace", required=True)
    p.set_defaults(fn=cmd_tick_wall)

    p = sub.add_parser("set-exp-status")
    p.add_argument("--workspace", required=True)
    p.add_argument("--exp-id", required=True)
    p.add_argument("--status", required=True,
                   choices=["queued", "smoke_running", "smoke_passed", "smoke_failed",
                            "running", "done", "fail-compile", "fail-train",
                            "fail-compute-mismatch", "stalled", "cancelled"])
    p.add_argument("--reason", default=None)
    p.set_defaults(fn=cmd_set_exp_status)

    p = sub.add_parser("record-category-outcome")
    p.add_argument("--workspace", required=True)
    p.add_argument("--hook", required=True)
    p.add_argument("--delta-psnr", type=float, default=None)
    p.add_argument("--beat-noise-floor", default="false")
    p.set_defaults(fn=cmd_record_category_outcome)

    p = sub.add_parser("xrun-read")
    p.add_argument("--workspace", required=True)
    p.add_argument("--regime", default=None)
    p.set_defaults(fn=cmd_xrun_read)

    p = sub.add_parser("xrun-write")
    p.add_argument("--workspace", required=True)
    p.add_argument("--regime", default=None)
    p.set_defaults(fn=cmd_xrun_write)

    args = ap.parse_args()
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
