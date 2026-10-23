"""
AlphaEvolve-style evaluation cascade for 3DGS candidates.

Stages:
    S1  compile/import check    (no GPU, ~5s)
    S2  100-iter smoke train    (1 GPU, ~30-60s) — liveness check
    S3  single-scene short eval (1 GPU, ~5-10m) — baseline-Δ gate (cheap filter)
    S4  full benchmark          (1 GPU, hours)  — winner only, reporting

Gating:
    S1 fail: drop
    S2 fail: drop
    S3 fail: drop (psnr < baseline - Δ)
    S4: no gating — it is the final report on the champion.

Parallelism:
    GPUPool hands out one GPU per candidate via a thread-safe queue.
    run_generation fans all N candidates of a generation across the pool;
    with N candidates and G GPUs, G run concurrently and (N-G) wait.
    Generation barrier is implicit: the outer helper awaits all futures.
    Each candidate holds one GPU for the whole stage; S4 runs its scenes
    sequentially on that single GPU.

Baselines:
    splatfacto is run once per (scene, stage) and cached to
    <workspace>/baselines/<scene>_<stage>.json. Subsequent candidates
    reuse the cached value for gating.

This module is intentionally independent of the evolutionary discover.py
loop — it just takes already-installed candidates and runs the cascade
against them. The caller decides what "candidate" means.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════
# DATA SHAPES
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CascadeCandidate:
    """A candidate method already installed into the nerfstudio env.

    Minimal interface — cascade only needs the CLI name, an on-disk
    implementation path (for S1 import checks), and an opaque id used
    for logging. Everything else is stage/config.
    """
    id: str
    method_cli: str          # e.g. "my-method-island-3"
    impl_path: Path          # directory containing the installed pkg
    package_name: str | None = None   # python import name (for S1)
    display_name: str = ""   # human-readable label for logs


@dataclass
class StageResult:
    candidate_id: str
    stage: str                         # s1 | s2 | s3 | s4
    passed: bool = False
    gated: bool = False                # S3/S4: cleared baseline - delta?
    gpu: str | None = None
    metrics: dict = field(default_factory=dict)
    per_scene: dict = field(default_factory=dict)
    wall_time_s: float = 0.0
    error: str | None = None
    log_path: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["log_path"] = str(self.log_path) if self.log_path else None
        return d


@dataclass
class CascadeConfig:
    """Knobs for the eval cascade. Defaults lifted from PipelineConfig."""
    smoke_iters: int = 100
    single_scene: str = "garden"
    single_scene_iters: int = 7000
    full_iters: int = 30000
    baseline_delta_db: float = 1.0
    full_scenes: list[str] = field(default_factory=lambda: [
        "bicycle", "bonsai", "counter", "garden",
        "kitchen", "room", "stump", "flowers", "treehill",
    ])
    conda_env: str = "nerfstudio"
    stage_timeout_s: int = 7200
    baseline_method_cli: str = "splatfacto"
    baseline_package_name: str | None = None

    @classmethod
    def from_pipeline_config(cls, pc) -> "CascadeConfig":
        scenes = list(pc.scenes) if getattr(pc, "scenes", None) else ["garden"]
        return cls(
            smoke_iters=pc.cascade_smoke_iters,
            single_scene=pc.cascade_single_scene,
            single_scene_iters=pc.cascade_single_scene_iters,
            full_iters=pc.cascade_full_iters,
            baseline_delta_db=pc.cascade_baseline_delta_db,
            full_scenes=scenes if len(scenes) > 1 else cls.__dataclass_fields__["full_scenes"].default_factory(),
            conda_env=pc.conda_env,
            stage_timeout_s=pc.cascade_stage_timeout_s,
        )


# ═══════════════════════════════════════════════════════════════════
# GPU POOL
# ═══════════════════════════════════════════════════════════════════

class GPUPool:
    """Thread-safe FIFO pool of GPU ids.

    Same pattern as the existing `gpu_queue` in discover.py, lifted
    into a reusable object with a context manager so callers can't
    forget to release on exceptions.
    """

    def __init__(self, gpu_ids: list[str]):
        self._q: queue.Queue[str] = queue.Queue()
        for g in gpu_ids:
            self._q.put(str(g))
        self._total = len(gpu_ids)

    @property
    def total(self) -> int:
        return self._total

    def acquire(self, timeout: float | None = None) -> str:
        return self._q.get(timeout=timeout) if timeout else self._q.get()

    def release(self, gpu: str) -> None:
        self._q.put(gpu)

    class _Ctx:
        def __init__(self, pool, gpu): self.pool, self.gpu = pool, gpu
        def __enter__(self): return self.gpu
        def __exit__(self, *a): self.pool.release(self.gpu)

    def acquire_ctx(self):
        """`with pool.acquire_ctx() as gpu:` — releases on exit always."""
        return GPUPool._Ctx(self, self.acquire())


# ═══════════════════════════════════════════════════════════════════
# SUBPROCESS HELPERS
# ═══════════════════════════════════════════════════════════════════

def _build_env(gpu: str | None, base: dict | None = None) -> dict[str, str]:
    env = os.environ.copy() if base is None else dict(base)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    env.setdefault("CC", "/usr/bin/gcc-11")
    env.setdefault("CXX", "/usr/bin/g++-11")
    env.setdefault("CUDAHOSTCXX", "/usr/bin/g++-11")
    return env


def _run_shell(cmd: str, env: dict, timeout: int, log_path: Path | None = None) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True,
            timeout=timeout, env=env,
        )
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text((proc.stdout or "") + "\n---STDERR---\n" + (proc.stderr or ""))
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as e:
        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(f"TIMEOUT after {timeout}s\n{e}")
        return 124, "", f"Timeout after {timeout}s"


def _conda_prefix(env_name: str) -> str:
    return f'eval "$(conda shell.bash hook 2>/dev/null)" && conda activate {env_name} && '


def _read_psnr_from_run_dir(run_dir: Path, read_tb_script: Path | None = None) -> float | None:
    """Best-effort PSNR parse from an ns-train output directory.

    Mirrors the logic in invent.py's _read_psnr_from_tb: try read_tb.py
    first, then fall back to scanning eval JSONs.
    """
    if read_tb_script and read_tb_script.exists():
        try:
            proc = subprocess.run(
                ["python", str(read_tb_script), str(run_dir), "--json"],
                capture_output=True, text=True, timeout=30,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                psnr = data.get("final_psnr")
                if psnr:
                    return float(psnr)
        except Exception:
            pass

    for json_file in run_dir.rglob("*.json"):
        try:
            data = json.loads(json_file.read_text())
            if isinstance(data, dict):
                if "results" in data and isinstance(data["results"], dict) and "psnr" in data["results"]:
                    return float(data["results"]["psnr"])
                if "psnr" in data:
                    return float(data["psnr"])
        except Exception:
            continue
    return None


def _latest_run_subdir(output_dir: Path, scene: str, method_cli: str) -> Path | None:
    scene_method = output_dir / scene / method_cli
    if not scene_method.exists():
        return None
    runs = sorted(scene_method.iterdir(), key=lambda p: p.name, reverse=True)
    return runs[0] if runs else None


# ═══════════════════════════════════════════════════════════════════
# STAGE RUNNERS
# ═══════════════════════════════════════════════════════════════════

def run_s1_compile(
    candidate: CascadeCandidate,
    cfg: CascadeConfig,
    workspace: Path,
) -> StageResult:
    """S1: import the candidate's package. No GPU, fast-fail for broken code."""
    t0 = time.time()
    pkg = candidate.package_name or _guess_package_name(candidate.impl_path)
    result = StageResult(candidate_id=candidate.id, stage="s1")
    if not pkg:
        result.error = "Could not infer package name"
        result.wall_time_s = time.time() - t0
        return result

    log_path = workspace / "cascade_logs" / candidate.id / "s1.log"
    cmd = _conda_prefix(cfg.conda_env) + f'python -c "import {pkg}; print(\'OK\')"'
    env = _build_env(gpu=None)
    rc, out, err = _run_shell(cmd, env, timeout=60, log_path=log_path)
    result.wall_time_s = time.time() - t0
    result.log_path = str(log_path)
    if rc == 0 and "OK" in out:
        result.passed = True
    else:
        result.error = (err or out or "import failed").strip()[-500:]
    return result


def run_s2_smoke(
    candidate: CascadeCandidate,
    cfg: CascadeConfig,
    workspace: Path,
    data_path: str,
    gpu: str,
) -> StageResult:
    """S2: 100-iter smoke train. Catches NaN/OOM/shape bugs."""
    t0 = time.time()
    result = StageResult(candidate_id=candidate.id, stage="s2", gpu=gpu)
    output_dir = workspace / "cascade_outputs" / "s2" / candidate.id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "cascade_logs" / candidate.id / "s2.log"

    cmd = (
        _conda_prefix(cfg.conda_env)
        + f'ns-train {candidate.method_cli} '
        + f'--data "{data_path}" '
        + f'--vis none '
        + f'--max-num-iterations {cfg.smoke_iters} '
        + f'--output-dir "{output_dir}" '
        + f'--viewer.quit-on-train-completion True '
        + f'colmap --eval-mode interval '
    )
    env = _build_env(gpu)
    rc, out, err = _run_shell(cmd, env, timeout=900, log_path=log_path)
    result.wall_time_s = time.time() - t0
    result.log_path = str(log_path)

    if rc != 0:
        result.error = (err or out or "smoke train failed").strip()[-500:]
        return result

    # Liveness: "NaN" / "inf" in stdout is a fail even with rc=0
    lowered = (out + err).lower()
    if "nan" in lowered and "loss" in lowered:
        result.error = "NaN loss detected"
        return result
    result.passed = True
    return result


def run_s3_single_scene(
    candidate: CascadeCandidate,
    cfg: CascadeConfig,
    workspace: Path,
    data_root: str,
    gpu: str,
    baseline_psnr: float | None,
    read_tb_script: Path | None = None,
) -> StageResult:
    """S3: full short training on one scene, PSNR eval, baseline-Δ gate."""
    t0 = time.time()
    scene = cfg.single_scene
    result = StageResult(candidate_id=candidate.id, stage="s3", gpu=gpu)
    output_dir = workspace / "cascade_outputs" / "s3" / candidate.id
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "cascade_logs" / candidate.id / "s3.log"
    data_path = f"{data_root}/{scene}"

    cmd = (
        _conda_prefix(cfg.conda_env)
        + f'ns-train {candidate.method_cli} '
        + f'--data "{data_path}" '
        + f'--vis tensorboard '
        + f'--steps-per-eval-batch 10000 '
        + f'--max-num-iterations {cfg.single_scene_iters} '
        + f'--output-dir "{output_dir}" '
        + f'--viewer.quit-on-train-completion True '
        + f'colmap --eval-mode interval '
    )
    env = _build_env(gpu)
    rc, out, err = _run_shell(cmd, env, timeout=cfg.stage_timeout_s, log_path=log_path)
    result.wall_time_s = time.time() - t0
    result.log_path = str(log_path)

    if rc != 0:
        result.error = (err or out or "s3 train failed").strip()[-500:]
        return result

    run_dir = _latest_run_subdir(output_dir, scene, candidate.method_cli)
    psnr = _read_psnr_from_run_dir(run_dir, read_tb_script) if run_dir else None
    if psnr is None:
        result.error = "Could not parse PSNR from s3 run"
        return result

    result.metrics = {"psnr": psnr, "scene": scene}
    result.passed = True
    if baseline_psnr is not None:
        result.gated = psnr >= (baseline_psnr - cfg.baseline_delta_db)
    else:
        result.gated = True  # no baseline cached → skip gate
    return result


def run_s4_full_benchmark(
    candidate: CascadeCandidate,
    cfg: CascadeConfig,
    workspace: Path,
    data_root: str,
    gpu: str,
    scene_baselines: dict[str, float] | None = None,
    read_tb_script: Path | None = None,
    on_scene_done: Callable[[str, StageResult], None] | None = None,
) -> StageResult:
    """S4: full benchmark, scenes sequential on the winner's single GPU."""
    t0 = time.time()
    result = StageResult(candidate_id=candidate.id, stage="s4", gpu=gpu)
    per_scene: dict[str, dict] = {}
    output_base = workspace / "cascade_outputs" / "s4" / candidate.id
    output_base.mkdir(parents=True, exist_ok=True)

    for scene in cfg.full_scenes:
        scene_out = output_base / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        log_path = workspace / "cascade_logs" / candidate.id / f"s4_{scene}.log"
        data_path = f"{data_root}/{scene}"
        cmd = (
            _conda_prefix(cfg.conda_env)
            + f'ns-train {candidate.method_cli} '
            + f'--data "{data_path}" '
            + f'--vis tensorboard '
            + f'--steps-per-eval-batch 10000 '
            + f'--max-num-iterations {cfg.full_iters} '
            + f'--output-dir "{scene_out}" '
            + f'--viewer.quit-on-train-completion True '
            + f'colmap --eval-mode interval '
        )
        env = _build_env(gpu)
        rc, out, err = _run_shell(cmd, env, timeout=cfg.stage_timeout_s, log_path=log_path)
        psnr = None
        run_dir = _latest_run_subdir(scene_out, scene, candidate.method_cli) if rc == 0 else None
        if run_dir:
            psnr = _read_psnr_from_run_dir(run_dir, read_tb_script)
        per_scene[scene] = {
            "psnr": psnr,
            "baseline": (scene_baselines or {}).get(scene),
            "ok": rc == 0 and psnr is not None,
            "error": None if rc == 0 else (err or out or "").strip()[-300:],
        }
        if on_scene_done:
            partial = StageResult(
                candidate_id=candidate.id, stage="s4", gpu=gpu,
                per_scene={scene: per_scene[scene]},
            )
            on_scene_done(scene, partial)

    psnrs = [v["psnr"] for v in per_scene.values() if v.get("psnr") is not None]
    avg_psnr = sum(psnrs) / len(psnrs) if psnrs else 0.0
    result.per_scene = per_scene
    result.metrics = {"avg_psnr": avg_psnr, "scenes_ok": len(psnrs), "scenes_total": len(cfg.full_scenes)}
    result.passed = len(psnrs) > 0
    result.wall_time_s = time.time() - t0
    return result


# ═══════════════════════════════════════════════════════════════════
# BASELINE CACHE
# ═══════════════════════════════════════════════════════════════════

def _baseline_cache_path(workspace: Path, scene: str, stage: str) -> Path:
    return workspace / "baselines" / f"{scene}_{stage}.json"


def ensure_baseline(
    workspace: Path,
    scene: str,
    stage: str,
    cfg: CascadeConfig,
    data_root: str,
    pool: GPUPool,
    event_sink=None,
    read_tb_script: Path | None = None,
) -> float | None:
    """Return cached baseline PSNR for (scene, stage), running splatfacto once if missing.

    Consults both the per-workspace cache AND the cross-run cache at
    ~/.splatlab/baselines-cache/ so every new run skips the warm-up
    splatfacto train when the same (scene, stage, iters) was done before.
    """
    iters = cfg.single_scene_iters if stage == "s3" else cfg.full_iters
    # 1) per-workspace cache.
    cache = _baseline_cache_path(workspace, scene, stage)
    if cache.exists():
        try:
            return float(json.loads(cache.read_text()).get("psnr"))
        except Exception:
            pass
    # 2) cross-run baseline cache.
    try:
        from splatify.cache import read_cached_baseline, write_cached_baseline
        cross = read_cached_baseline(scene, stage, iters, cfg.baseline_method_cli)
        if cross and cross.get("psnr") is not None:
            # Mirror into the per-workspace cache for consistency.
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps({"psnr": cross["psnr"], "source": "cross_run"}, indent=2))
            return float(cross["psnr"])
    except ImportError:
        read_cached_baseline = write_cached_baseline = None  # type: ignore
    if event_sink is not None:
        event_sink.put({
            "type": "cascade_baseline_running",
            "scene": scene, "stage": stage, "iters": iters,
            "message": f"Running splatfacto baseline for {scene} @ {iters} iters ({stage})...",
        })

    output_dir = workspace / "baselines_runs" / f"{scene}_{stage}"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "cascade_logs" / "baselines" / f"{scene}_{stage}.log"
    data_path = f"{data_root}/{scene}"

    cmd = (
        _conda_prefix(cfg.conda_env)
        + f'ns-train {cfg.baseline_method_cli} '
        + f'--data "{data_path}" '
        + f'--vis tensorboard '
        + f'--max-num-iterations {iters} '
        + f'--output-dir "{output_dir}" '
        + f'--viewer.quit-on-train-completion True '
        + f'colmap --eval-mode interval '
    )

    with pool.acquire_ctx() as gpu:
        env = _build_env(gpu)
        rc, out, err = _run_shell(cmd, env, timeout=cfg.stage_timeout_s, log_path=log_path)

    if rc != 0:
        if event_sink is not None:
            event_sink.put({
                "type": "cascade_baseline_failed",
                "scene": scene, "stage": stage,
                "error": (err or out or "")[-300:],
                "message": f"Baseline run failed for {scene} ({stage})",
            })
        return None

    run_dir = _latest_run_subdir(output_dir, scene, cfg.baseline_method_cli)
    psnr = _read_psnr_from_run_dir(run_dir, read_tb_script) if run_dir else None
    if psnr is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({
            "scene": scene, "stage": stage, "iters": iters, "psnr": psnr,
        }, indent=2))
        # Also persist to the cross-run cache so future runs skip this train.
        try:
            from splatify.cache import write_cached_baseline
            write_cached_baseline(scene, stage, iters,
                                  {"psnr": psnr}, cfg.baseline_method_cli)
        except ImportError:
            pass
        if event_sink is not None:
            event_sink.put({
                "type": "cascade_baseline_done",
                "scene": scene, "stage": stage, "psnr": psnr,
                "message": f"Baseline {scene} ({stage}) = {psnr:.2f} dB",
            })
    return psnr


# ═══════════════════════════════════════════════════════════════════
# GENERATION-LEVEL DRIVERS
# ═══════════════════════════════════════════════════════════════════

def run_generation(
    candidates: list[CascadeCandidate],
    stage: str,
    cfg: CascadeConfig,
    workspace: Path,
    pool: GPUPool,
    data_root: str,
    event_sink=None,
    baseline_psnr: float | None = None,
    read_tb_script: Path | None = None,
) -> list[StageResult]:
    """Run one stage across all candidates in a generation, GPU-pooled.

    Generation-synchronous: returns only when every candidate has
    finished the stage (or errored out). Order of returned results
    matches the input order.
    """
    if event_sink is not None:
        event_sink.put({
            "type": "cascade_generation_start",
            "stage": stage,
            "num_candidates": len(candidates),
            "message": f"Cascade {stage.upper()}: {len(candidates)} candidates on {pool.total} GPU(s)",
        })

    results: list[StageResult | None] = [None] * len(candidates)

    def _worker(idx: int, cand: CascadeCandidate) -> StageResult:
        if stage == "s1":
            if event_sink is not None:
                event_sink.put({
                    "type": "cascade_candidate_start",
                    "candidate_id": cand.id, "stage": "s1",
                    "message": f"S1 import check: {cand.display_name or cand.id}",
                })
            res = run_s1_compile(cand, cfg, workspace)
        else:
            gpu = pool.acquire()
            try:
                if event_sink is not None:
                    event_sink.put({
                        "type": "cascade_candidate_start",
                        "candidate_id": cand.id, "stage": stage, "gpu": gpu,
                        "message": f"{stage.upper()} on GPU {gpu}: {cand.display_name or cand.id}",
                    })
                if stage == "s2":
                    data_path = f"{data_root}/{cfg.single_scene}"
                    res = run_s2_smoke(cand, cfg, workspace, data_path, gpu)
                elif stage == "s3":
                    res = run_s3_single_scene(
                        cand, cfg, workspace, data_root, gpu,
                        baseline_psnr=baseline_psnr, read_tb_script=read_tb_script,
                    )
                else:
                    raise ValueError(f"run_generation does not handle stage {stage}")
            finally:
                pool.release(gpu)

        if event_sink is not None:
            event_sink.put({
                "type": "cascade_candidate_done",
                "candidate_id": cand.id,
                "stage": stage,
                "result": res.to_dict(),
                "message": (
                    f"{stage.upper()} {cand.display_name or cand.id}: "
                    + ("OK" if res.passed else f"FAIL — {res.error}")
                ),
            })
        return res

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(candidates))) as tp:
        fut_to_idx = {tp.submit(_worker, i, c): i for i, c in enumerate(candidates)}
        for fut in concurrent.futures.as_completed(fut_to_idx):
            idx = fut_to_idx[fut]
            try:
                results[idx] = fut.result()
            except Exception as e:
                results[idx] = StageResult(
                    candidate_id=candidates[idx].id, stage=stage,
                    error=f"worker exception: {e}",
                )

    finalized: list[StageResult] = [r for r in results if r is not None]

    if event_sink is not None:
        passed = sum(1 for r in finalized if r.passed)
        gated = sum(1 for r in finalized if r.gated)
        ranked = sorted(
            [r for r in finalized if r.passed],
            key=lambda r: r.metrics.get("psnr", 0.0),
            reverse=True,
        )
        event_sink.put({
            "type": "cascade_generation_done",
            "stage": stage,
            "passed": passed,
            "gated": gated,
            "ranked": [r.candidate_id for r in ranked],
            "message": f"{stage.upper()} done — {passed}/{len(finalized)} passed, {gated} gated",
        })

    return finalized


def promote_to_full_benchmark(
    winner: CascadeCandidate,
    cfg: CascadeConfig,
    workspace: Path,
    pool: GPUPool,
    data_root: str,
    event_sink=None,
    scene_baselines: dict[str, float] | None = None,
    read_tb_script: Path | None = None,
) -> StageResult:
    """Run S4 on the generation winner. Scenes sequential on one GPU."""
    if event_sink is not None:
        event_sink.put({
            "type": "cascade_promoted",
            "candidate_id": winner.id,
            "message": f"Promoting {winner.display_name or winner.id} to full benchmark",
        })

    with pool.acquire_ctx() as gpu:
        if event_sink is not None:
            event_sink.put({
                "type": "cascade_candidate_start",
                "candidate_id": winner.id, "stage": "s4", "gpu": gpu,
                "message": f"S4 full benchmark on GPU {gpu}",
            })

        def _on_scene(scene: str, partial: StageResult):
            if event_sink is not None:
                event_sink.put({
                    "type": "cascade_scene_done",
                    "candidate_id": winner.id,
                    "scene": scene,
                    "result": partial.per_scene.get(scene, {}),
                })

        result = run_s4_full_benchmark(
            winner, cfg, workspace, data_root, gpu,
            scene_baselines=scene_baselines,
            read_tb_script=read_tb_script,
            on_scene_done=_on_scene,
        )

    if event_sink is not None:
        event_sink.put({
            "type": "cascade_benchmark_done",
            "candidate_id": winner.id,
            "result": result.to_dict(),
            "message": (
                f"S4 done: avg PSNR {result.metrics.get('avg_psnr', 0):.2f} dB "
                f"across {result.metrics.get('scenes_ok', 0)} scenes"
            ),
        })
    return result


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _guess_package_name(impl_path: Path) -> str | None:
    """Best-effort package name inference from an implementation directory."""
    if not impl_path.exists():
        return None
    for child in impl_path.iterdir():
        if (not child.is_dir()
                or child.name.startswith("__")
                or child.name.endswith(".egg-info")):
            continue
        if (child / "__init__.py").exists():
            return child.name
    return None
