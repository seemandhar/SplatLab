#!/usr/bin/env python3
"""Promote this-run artifacts to the global splatlab directories.

Lifts reusable outputs from `workspace/<run_id>/` to the project-root globals
so the next run of any regime starts with a richer shared state.

Promoted:
  - New baseline depots registered in `ledger.code_depots` → merged into
    `baselines_registry.json` at project root (records env name, git SHA,
    reproduced metrics, build timestamp). Future runs' codebase-selection
    can check this registry first and skip re-cloning + re-building.
  - New bibkeys in `workspace/<run>/paper/references.bib` OR
    `workspace/<run>/references.bib` → merged into `refs/references.bib`
    (dedup by bibkey; this-run entries never overwrite existing globals).
  - New paper-summary markdown in `workspace/<run>/literature/paper_summaries/*.md`
    → copied into `literature/paper_summaries/` if the filename does not
    already exist globally.

Usage:
  python3 scripts/promote_run_artifacts.py --workspace workspace/run_YYYYMMDD_HHMMSS

Idempotent: safe to re-run; only new content is promoted. Logs each action.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha(path: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except Exception:
        return None


def promote_baselines(workspace: Path, project_root: Path) -> dict:
    ledger_path = workspace / "ledger.json"
    if not ledger_path.exists():
        return {"baselines_promoted": 0, "reason": "no ledger"}

    ledger = json.loads(ledger_path.read_text())
    depots = ledger.get("code_depots") or {}
    reproduced = (ledger.get("target") or {}).get("reproduced_metrics") or {}
    preferred_depot = ledger.get("preferred_depot")

    registry_path = project_root / "baselines_registry.json"
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    else:
        registry = {
            "schema_version": 1,
            "description": ("Known-good baseline depots discovered by prior runs. "
                            "Each entry records what was cloned, the env built for it, "
                            "the reproduced metrics, and the originating run. "
                            "Codebase-selection (Phase 0 §4b) can consult this first "
                            "to skip re-cloning + re-building when the same depot is needed again."),
            "entries": {},
        }

    n_promoted = 0
    n_updated = 0
    for depot_key, meta in depots.items():
        if not isinstance(meta, dict):
            continue
        depot_path = Path(meta.get("path", ""))
        env_name = meta.get("env_name")
        built_at = meta.get("built_at")
        import_ok = meta.get("import_verified", False)
        if not env_name or not import_ok:
            continue
        git_sha = _git_sha(depot_path) if depot_path.exists() else None

        entry = {
            "depot_key": depot_key,
            "git_sha": git_sha,
            "env_name": env_name,
            "built_at": built_at,
            "last_used": _utcnow(),
            "originating_run": workspace.name,
            "was_preferred": (depot_key == preferred_depot),
            "reproduced_metrics": reproduced if depot_key == preferred_depot else None,
        }

        prev = registry["entries"].get(depot_key)
        if prev is None:
            registry["entries"][depot_key] = entry
            n_promoted += 1
        else:
            # Preserve earlier history if metrics are already populated; update last_used.
            prev["last_used"] = _utcnow()
            if entry["git_sha"] and entry["git_sha"] != prev.get("git_sha"):
                prev["git_sha"] = entry["git_sha"]
                prev["rebuilt_at"] = built_at
            if entry["reproduced_metrics"] and not prev.get("reproduced_metrics"):
                prev["reproduced_metrics"] = entry["reproduced_metrics"]
            registry["entries"][depot_key] = prev
            n_updated += 1

    registry["last_updated"] = _utcnow()
    registry_path.write_text(json.dumps(registry, indent=2, sort_keys=False))
    return {"baselines_promoted": n_promoted, "baselines_updated": n_updated,
            "registry_path": str(registry_path)}


_BIB_ENTRY = re.compile(r"@[A-Za-z]+\s*{\s*([^,\s]+)\s*,", re.M)


def _bibkeys(text: str) -> set[str]:
    return set(_BIB_ENTRY.findall(text))


def _bib_entries(text: str) -> dict[str, str]:
    """Return {bibkey: entry_text} by walking brace-balanced @... blocks."""
    out = {}
    i = 0
    while i < len(text):
        m = _BIB_ENTRY.search(text, i)
        if not m:
            break
        key = m.group(1)
        start = m.start()
        # Find matching closing brace
        j = text.find("{", m.end() - 1)
        depth = 0
        k = j
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        if depth != 0:
            break
        out[key] = text[start:k + 1]
        i = k + 1
    return out


def promote_bibtex(workspace: Path, project_root: Path) -> dict:
    src_candidates = [
        workspace / "paper" / "references.bib",
        workspace / "references.bib",
    ]
    src = next((p for p in src_candidates if p.exists()), None)
    if src is None:
        return {"bib_promoted": 0, "reason": "no workspace bib"}

    dst = project_root / "refs" / "references.bib"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        dst.write_text("")

    existing = _bib_entries(dst.read_text())
    incoming = _bib_entries(src.read_text())
    added = []
    for key, entry in incoming.items():
        if key not in existing:
            existing[key] = entry
            added.append(key)
    if added:
        # Append new entries to preserve original ordering of globals.
        with dst.open("a") as f:
            for key in added:
                f.write("\n" + incoming[key] + "\n")
    return {"bib_promoted": len(added), "bib_src": str(src), "bib_dst": str(dst),
            "new_keys": added[:20]}


def promote_paper_summaries(workspace: Path, project_root: Path) -> dict:
    src_dirs = [
        workspace / "literature" / "paper_summaries",
        workspace / "paper_summaries",
    ]
    dst = project_root / "literature" / "paper_summaries"
    dst.mkdir(parents=True, exist_ok=True)
    existing = {p.name for p in dst.glob("*.md")}
    copied = []
    for src in src_dirs:
        if not src.exists():
            continue
        for md in src.glob("*.md"):
            if md.name not in existing:
                shutil.copy2(md, dst / md.name)
                copied.append(md.name)
                existing.add(md.name)
    return {"summaries_promoted": len(copied), "summaries_dst": str(dst),
            "new_summaries": copied[:20]}


def main() -> int:
    ap = argparse.ArgumentParser(description="Promote run artifacts to project globals")
    ap.add_argument("--workspace", required=True, help="workspace/run_YYYYMMDD_HHMMSS")
    ap.add_argument("--project-root", default=None,
                    help="Defaults to the parent of --workspace's parent (splatlab root)")
    args = ap.parse_args()

    ws = Path(args.workspace).resolve()
    if not ws.exists():
        print(f"ERROR: workspace not found: {ws}", file=sys.stderr)
        return 1
    project_root = Path(args.project_root).resolve() if args.project_root else ws.parent.parent

    result = {
        "workspace": str(ws),
        "project_root": str(project_root),
        "promoted_at": _utcnow(),
    }
    result.update(promote_baselines(ws, project_root))
    result.update(promote_bibtex(ws, project_root))
    result.update(promote_paper_summaries(ws, project_root))

    # Log under the workspace so it shows up in the run's event stream.
    ev_path = ws / "events.jsonl"
    try:
        with ev_path.open("a") as f:
            f.write(json.dumps({"type": "artifacts_promoted_to_global",
                                "ts": _utcnow(), **result}) + "\n")
    except Exception:
        pass

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
