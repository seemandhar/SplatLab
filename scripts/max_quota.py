#!/usr/bin/env python3
"""Track Claude Max quota consumption across all Claude Code sessions.

Usage:
    python max_quota.py                   # 7d summary + today + warnings
    python max_quota.py --since 1d        # just last 24h
    python max_quota.py --since 7d        # last week (default)
    python max_quota.py --project splatlab  # filter to one project
    python max_quota.py --json            # machine-readable (for /research-status)

What this measures
------------------
Anthropic doesn't expose a "what's left in my Max quota" API. What we CAN do:
tally every session JSONL that Claude Code has logged and report total tokens.
Compare against rough Max-tier allowances below.

Max tier rough allowances (Anthropic's published docs are vague and change —
treat these as order-of-magnitude guides, not hard numbers):

  Pro          ~45 msgs / 5h   ~216/day    ~1500/week
  Max 5×       ~5× that        ~7500 msgs/week  (heavy user)
  Max 20×      ~20× that       ~30000 msgs/week (very heavy user)

Token-wise, a Claude Code "message" averages 10k-100k tokens (cache-heavy for
tool-calling sessions). A big agentic run like splatlab burns
~50M-200M tokens per week.

Rate-limit errors (429) are the hard signal that quota is exhausted. This
script also greps the JSONLs for those and reports them.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECTS_ROOT = Path.home() / ".claude" / "projects"

# Per-MTok pricing for the "$-equivalent" column (only used as a cross-tier
# effort comparator; NOT a real charge on Claude Max).
RATES = {
    "claude-opus-4-7":          (15.00, 75.00, 18.75, 1.50),
    "claude-opus-4-6":          (15.00, 75.00, 18.75, 1.50),
    "claude-sonnet-4-6":        ( 3.00, 15.00,  3.75, 0.30),
    "claude-haiku-4-5":         ( 1.00,  5.00,  1.25, 0.10),
    "claude-haiku-4-5-20251001":( 1.00,  5.00,  1.25, 0.10),
}


def parse_since(s: str) -> timedelta:
    m = re.match(r"^(\d+)([hdw])$", s)
    if not m:
        raise ValueError(f"bad --since {s}; use e.g. 1h, 24h, 7d, 1w")
    n, u = int(m.group(1)), m.group(2)
    return {"h": timedelta(hours=n), "d": timedelta(days=n), "w": timedelta(weeks=n)}[u]


def iter_messages(since_ts: float, project_filter: str | None):
    """Yield (ts_epoch, project_slug, model, usage_dict, is_rate_limit_err)."""
    for project_dir in PROJECTS_ROOT.iterdir():
        if not project_dir.is_dir():
            continue
        slug = project_dir.name
        if project_filter and project_filter not in slug:
            continue
        for jsonl in list(project_dir.glob("*.jsonl")) + list(project_dir.glob("**/subagents/*.jsonl")):
            try:
                mtime = jsonl.stat().st_mtime
                if mtime < since_ts - 3600:   # skip files obviously too old
                    continue
            except FileNotFoundError:
                continue
            try:
                with jsonl.open(encoding="utf-8", errors="replace") as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        ts = d.get("timestamp") or d.get("ts") or None
                        t_epoch = None
                        if ts:
                            try:
                                t_epoch = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                            except Exception:
                                pass
                        if t_epoch is None:
                            t_epoch = mtime
                        if t_epoch < since_ts:
                            continue
                        msg = d.get("message") or {}
                        usage = msg.get("usage") or {}
                        model = msg.get("model") or "claude-sonnet-4-6"
                        # rate-limit error detection
                        is_rl = False
                        err = d.get("error") or msg.get("error")
                        if err:
                            s = str(err).lower()
                            if "rate_limit" in s or "429" in s or "quota" in s:
                                is_rl = True
                        yield t_epoch, slug, model, usage, is_rl
            except (OSError, PermissionError):
                continue


def cost_for(model: str, u: dict) -> float:
    r = RATES.get(model, RATES["claude-sonnet-4-6"])
    return (
        u.get("input_tokens", 0) / 1e6 * r[0]
        + u.get("output_tokens", 0) / 1e6 * r[1]
        + u.get("cache_creation_input_tokens", 0) / 1e6 * r[2]
        + u.get("cache_read_input_tokens", 0) / 1e6 * r[3]
    )


def fmt_tokens(n: int) -> str:
    if n >= 1_000_000_000: return f"{n/1e9:.2f}B"
    if n >= 1_000_000:     return f"{n/1e6:.2f}M"
    if n >= 1_000:         return f"{n/1e3:.1f}k"
    return str(n)


def run_summary(since_delta: timedelta, project_filter: str | None, as_json: bool):
    now = time.time()
    since_ts = now - since_delta.total_seconds()

    per_project = defaultdict(lambda: {"in": 0, "out": 0, "cc": 0, "cr": 0, "cost": 0.0, "msgs": 0, "rl": 0})
    per_model = defaultdict(lambda: {"in": 0, "out": 0, "cc": 0, "cr": 0, "cost": 0.0, "msgs": 0})
    rate_limit_events = []

    for t_epoch, slug, model, u, is_rl in iter_messages(since_ts, project_filter):
        total_u = any(u.values())
        if not total_u and not is_rl:
            continue
        pp = per_project[slug]
        pp["in"]  += u.get("input_tokens", 0)
        pp["out"] += u.get("output_tokens", 0)
        pp["cc"]  += u.get("cache_creation_input_tokens", 0)
        pp["cr"]  += u.get("cache_read_input_tokens", 0)
        pp["cost"] += cost_for(model, u)
        pp["msgs"] += 1
        if is_rl:
            pp["rl"] += 1
            rate_limit_events.append({"ts": t_epoch, "project": slug, "model": model})
        pm = per_model[model]
        pm["in"]  += u.get("input_tokens", 0)
        pm["out"] += u.get("output_tokens", 0)
        pm["cc"]  += u.get("cache_creation_input_tokens", 0)
        pm["cr"]  += u.get("cache_read_input_tokens", 0)
        pm["cost"] += cost_for(model, u)
        pm["msgs"] += 1

    total = {"in": 0, "out": 0, "cc": 0, "cr": 0, "cost": 0.0, "msgs": 0}
    for pp in per_project.values():
        for k in ("in", "out", "cc", "cr", "msgs"):
            total[k] += pp[k]
        total["cost"] += pp["cost"]

    total_tokens = total["in"] + total["out"] + total["cc"] + total["cr"]

    # Rough Max-tier warning thresholds (tokens/week). Tune to taste.
    MAX_5X_WEEK  = 300_000_000
    MAX_20X_WEEK = 1_200_000_000

    summary = {
        "window": f"last {since_delta.days}d {since_delta.seconds//3600}h",
        "total_tokens": total_tokens,
        "total_msgs": total["msgs"],
        "total_cost_equiv_usd": round(total["cost"], 2),
        "by_category": total,
        "per_project": {k: v for k, v in sorted(per_project.items(), key=lambda x: -x[1]["cost"])},
        "per_model": {k: v for k, v in sorted(per_model.items(), key=lambda x: -x[1]["cost"])},
        "rate_limit_events": rate_limit_events,
        "pct_max_5x_estimate":  round(total_tokens / MAX_5X_WEEK * 100, 1),
        "pct_max_20x_estimate": round(total_tokens / MAX_20X_WEEK * 100, 1),
    }

    if as_json:
        print(json.dumps(summary, indent=2, default=str))
        return

    print(f"╭─ Claude Max quota consumption ({summary['window']}) ─╮")
    print(f"│ total tokens:       {fmt_tokens(total_tokens):>12s}  ({total['msgs']} messages)")
    print(f"│ in:                 {fmt_tokens(total['in']):>12s}")
    print(f"│ out:                {fmt_tokens(total['out']):>12s}")
    print(f"│ cache_create:       {fmt_tokens(total['cc']):>12s}")
    print(f"│ cache_read:         {fmt_tokens(total['cr']):>12s}")
    print(f"│ cost-equiv (API):   ${summary['total_cost_equiv_usd']:>10.2f}")
    print(f"│ est. % Max 5×:      {summary['pct_max_5x_estimate']:>11.1f}%")
    print(f"│ est. % Max 20×:     {summary['pct_max_20x_estimate']:>11.1f}%")
    if rate_limit_events:
        print(f"│ ⚠ RATE-LIMIT events: {len(rate_limit_events)}")
    print(f"╰{'─'*50}╯")

    print("\nTop projects by cost-equivalent:")
    for slug, pp in list(summary["per_project"].items())[:6]:
        print(f"  ${pp['cost']:>7.2f}  {fmt_tokens(pp['in']+pp['out']+pp['cc']+pp['cr']):>10}  {pp['msgs']:>4} msgs  {slug[:60]}")

    print("\nPer model:")
    for model, pm in summary["per_model"].items():
        print(f"  ${pm['cost']:>7.2f}  {fmt_tokens(pm['in']+pm['out']+pm['cc']+pm['cr']):>10}  {pm['msgs']:>4} msgs  {model}")

    if summary["pct_max_5x_estimate"] > 80:
        print(f"\n⚠  ESTIMATE: you're at ~{summary['pct_max_5x_estimate']:.0f}% of a Max 5× weekly allowance.")
        print("   If on Max 5×, consider throttling the lab or running lighter agents.")
    elif summary["pct_max_5x_estimate"] > 50:
        print(f"\nℹ  ESTIMATE: ~{summary['pct_max_5x_estimate']:.0f}% of a Max 5× weekly allowance used.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="7d", help="time window: 1h, 24h, 7d, 1w")
    ap.add_argument("--project", default=None, help="substring filter on project slug")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of pretty output")
    args = ap.parse_args()
    run_summary(parse_since(args.since), args.project, args.json)
