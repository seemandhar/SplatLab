#!/usr/bin/env python3
"""
CLI wrapper for the discover pipeline.

Allows tournament_runner (or any external caller) to invoke
run_discover_pipeline() as a subprocess and collect JSON results
from stdout and workspace/discover_results.json.

Usage:
    python splatify/run_discover.py \
        --impl-path /path/to/implementation \
        --workspace /path/to/workspace/discovery \
        --gpu 0 --scene garden --max-iters 10000 --max-generations 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repository root
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the iterative discover pipeline from the CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--impl-path",
        required=True,
        help="Path to existing generated implementation directory.",
    )
    p.add_argument(
        "--workspace",
        required=True,
        help="Working directory for this discover run.",
    )
    p.add_argument("--gpu", default="0", help="GPU device IDs, comma-separated.")
    p.add_argument("--scene", default="garden", help="Scene name for training.")
    p.add_argument(
        "--max-iters",
        type=int,
        default=10000,
        help="Training iterations per candidate per generation.",
    )
    p.add_argument(
        "--max-generations",
        type=int,
        default=3,
        help="Number of generations to run.",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Training data path (None = auto-resolve from scene).",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Claude model preset (discoverer agents always use OPUS).",
    )
    p.add_argument(
        "--num-views",
        type=int,
        default=-1,
        help="Number of training views (-1 = all).",
    )
    p.add_argument(
        "--stop-file",
        default=None,
        help="Path to a file whose existence signals graceful stop.",
    )
    return p


def _make_stop_event(stop_file_path: str | None) -> threading.Event | None:
    """
    If --stop-file is given, return a threading.Event that a background
    thread will set once the file appears on disk.  The discover pipeline
    checks this event between generations to exit gracefully.
    """
    if not stop_file_path:
        return None

    event = threading.Event()

    def _watcher():
        while not event.is_set():
            if os.path.exists(stop_file_path):
                event.set()
                return
            time.sleep(2)

    t = threading.Thread(target=_watcher, daemon=True)
    t.start()
    return event


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)

    stop_event = _make_stop_event(args.stop_file)

    # ── Import here so sys.path is already set ───────────────────────────
    from splatify.discover import run_discover_pipeline

    try:
        result = asyncio.run(
            run_discover_pipeline(
                impl_path=args.impl_path,
                workspace=workspace,
                event_queue=None,
                stop_event=stop_event,
                gpu=args.gpu,
                data_path=args.data_path,
                scene=args.scene,
                max_iters=args.max_iters,
                model=args.model,
                num_views=args.num_views,
                max_generations=args.max_generations,
            )
        )
    except KeyboardInterrupt:
        print(
            json.dumps({"error": "Interrupted by user", "status": "interrupted"}),
            flush=True,
        )
        return 130
    except Exception as exc:
        tb = traceback.format_exc()
        error_payload = {
            "error": str(exc),
            "traceback": tb,
            "status": "failed",
        }
        # Best-effort save so the caller can inspect even on failure
        try:
            results_path = workspace / "discover_results.json"
            results_path.write_text(json.dumps(error_payload, indent=2))
        except Exception:
            pass
        print(json.dumps(error_payload), flush=True)
        return 1

    # ── Persist and emit results ─────────────────────────────────────────
    if result is None:
        result = {"status": "completed", "note": "pipeline returned None"}

    results_path = workspace / "discover_results.json"
    results_path.write_text(json.dumps(result, indent=2, default=str))

    # Compact JSON on stdout for easy machine parsing
    print(json.dumps(result, default=str), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
