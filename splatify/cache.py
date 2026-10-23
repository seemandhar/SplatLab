"""Shared caches for splatify: PSNR memoisation + cross-run baseline cache.

Two caches:

  1. PSNR memoisation per run-output-dir. _read_psnr_from_tb() is called
     from 3 places (invent.py, discover.py, pipeline ingester) and reads
     TensorBoard event files each time. We write the parsed PSNR to
     `<output_dir>/metrics_cached.json` the first time and reuse it.

  2. Cross-run baseline cache. Splatfacto baselines for (scene, stage,
     iters) rarely change — pin them at
     `~/.splatlab/baselines-cache/<scene>_<stage>_<iters>.json`
     so every new run skips the warm-up splatfacto train.

Neither cache calls an LLM; they only save disk I/O and GPU time.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional


# --- 1) per-run PSNR memo -------------------------------------------------

def cached_psnr_path(run_dir: Path) -> Path:
    return Path(run_dir) / "metrics_cached.json"


def read_cached_psnr(run_dir: Path) -> Optional[float]:
    p = cached_psnr_path(run_dir)
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text())
        v = data.get("psnr")
        return float(v) if v is not None else None
    except (OSError, ValueError):
        return None


def write_cached_psnr(run_dir: Path, psnr: float, **meta) -> None:
    p = cached_psnr_path(run_dir)
    try:
        payload = {"psnr": float(psnr), "ts": time.time(), **meta}
        p.write_text(json.dumps(payload, indent=2))
    except OSError:
        pass


# --- 2) cross-run baseline cache ------------------------------------------

def baseline_cache_root() -> Path:
    env = os.environ.get("PAPERWRITER_BASELINE_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".splatlab" / "baselines-cache"


def _baseline_key(scene: str, stage: str, iters: int, method: str = "splatfacto") -> str:
    return f"{method}__{scene}__{stage}__{iters}.json"


def read_cached_baseline(
    scene: str, stage: str, iters: int, method: str = "splatfacto",
) -> Optional[dict]:
    p = baseline_cache_root() / _baseline_key(scene, stage, iters, method)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return None


def write_cached_baseline(
    scene: str, stage: str, iters: int, payload: dict, method: str = "splatfacto",
) -> None:
    root = baseline_cache_root(); root.mkdir(parents=True, exist_ok=True)
    p = root / _baseline_key(scene, stage, iters, method)
    try:
        p.write_text(json.dumps({
            "scene": scene, "stage": stage, "iters": iters,
            "method": method, "cached_at": time.time(),
            **payload,
        }, indent=2))
    except OSError:
        pass
