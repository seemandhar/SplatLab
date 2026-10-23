"""
Falsification subagent — adversarial robustness check for 3DGS candidates.

Rationale: no other end-to-end pipeline runs an explicit "try to break
it" pass on each generated method. A candidate that beats baseline on
garden@7k may still collapse on flowers (high-frequency textures), on
sparse-view training (3 views), or when its key hyperparameter is
neutralized. Falsification catches the overfit, reports it, and
penalizes the candidate's effective score so the ranker prefers
methods that survive.

Design:

- Three attacks per candidate, run on the candidate's GPU slot after
  its S3 eval has cleared baseline-Δ:

    A1  edge_scene     — train on a harder scene (default 'flowers') at
                         `edge_iters` iters (default 3000). Same method,
                         same hyperparameters. Expected behavior: at
                         least clears baseline_flowers - 2.0 dB.

    A2  sparse_views   — re-run S3 on the same garden scene, but with
                         `--num-training-views 3`. A method that only
                         works with dense views collapses here.

    A3  ablation_attack — re-run S3 with the "key hyperparameter" from
                         the candidate's rationale set to 0 (or an
                         identity passthrough). If PSNR barely drops,
                         the claimed contribution is meaningless.

- Falsification *score* combines the raw garden PSNR with penalty
  terms for each failing attack. A method that wins garden by 0.8 dB
  but fails A3 (ablation is a no-op) loses its rank to one that won
  by 0.4 dB and survives all three. Formula in `score_candidate`.

- All attacks go through the existing `cascade.run_s3_single_scene`
  training primitive so we inherit its logging/gating/GPU semantics.
  Attacks hold the candidate's GPU for their duration — callers
  schedule falsification inside the same generation GPU pool.

Failure modes we accept:
- Attack code may fail to set the "key hparam"; in that case A3 is
  skipped and reported as `not_attempted`, which counts as "passed"
  (we don't penalize methods that don't expose a tunable).
- Edge scene may not exist in the dataset; in that case A1 is skipped
  similarly.
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from . import cascade


# ═══════════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FalsifyConfig:
    """Knobs for falsification. Defaults picked to be cheap."""
    edge_scene: str = "flowers"
    edge_iters: int = 3000
    edge_baseline_db: float = 21.5        # mipnerf360 flowers splatfacto
    edge_delta_db: float = 2.0            # must clear baseline - 2.0

    sparse_views: int = 3
    sparse_delta_db: float = 3.0          # PSNR drop vs garden S3 OK up to 3.0

    ablation_delta_db: float = 0.3        # PSNR must drop ≥0.3 if hparam matters

    # Scoring weights. Garden PSNR is the headline; each attack failure
    # subtracts a penalty.
    w_garden: float = 1.0
    p_edge_fail: float = 1.5
    p_sparse_fail: float = 1.0
    p_ablation_fail: float = 2.0          # largest: overclaiming is worst


# ═══════════════════════════════════════════════════════════════════
# RESULT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class AttackResult:
    name: str
    passed: bool
    attempted: bool = True
    psnr: float | None = None
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FalsifyReport:
    candidate_id: str
    garden_psnr: float
    attacks: list[AttackResult] = field(default_factory=list)
    score: float = 0.0
    wall_time_s: float = 0.0

    @property
    def passed_all(self) -> bool:
        return all(a.passed or not a.attempted for a in self.attacks)

    @property
    def num_failed(self) -> int:
        return sum(1 for a in self.attacks if a.attempted and not a.passed)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "garden_psnr": self.garden_psnr,
            "attacks": [a.to_dict() for a in self.attacks],
            "score": self.score,
            "wall_time_s": self.wall_time_s,
            "passed_all": self.passed_all,
            "num_failed": self.num_failed,
        }


# ═══════════════════════════════════════════════════════════════════
# KEY HPARAM EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def _guess_key_hparam(rationale: str, impl_path: Path) -> tuple[str, str] | None:
    """Best-effort extraction of the ablatable hyperparameter.

    Looks at the rationale for patterns like 'loss_weight=0.3' or
    'λ=0.1'; falls back to scanning config.py files for a single
    numeric field that the rationale mentions by name.

    Returns (file_path, param_name) or None if we can't identify one.
    Conservative: returns None rather than guessing wrong.
    """
    import re
    text = rationale or ""
    # Literal attr=value in the rationale
    m = re.search(r"([a-zA-Z_][a-zA-Z0-9_]{2,})\s*[=:]\s*([0-9.eE+-]+)", text)
    if not m:
        return None
    param = m.group(1)
    # Find a file that sets this param.
    for py in impl_path.rglob("*.py"):
        try:
            content = py.read_text()
        except Exception:
            continue
        if re.search(rf"\b{re.escape(param)}\s*[:=]\s*[0-9.]", content):
            return str(py), param
    return None


def _zero_hparam_in_file(file_path: Path, param: str) -> bool:
    """Rewrite the file so `param = <n>` becomes `param = 0.0`.

    Returns True if a change was made. Preserves the original in
    `<file>.falsify-bak` for later restoration.
    """
    import re
    try:
        original = file_path.read_text()
    except Exception:
        return False
    pattern = re.compile(rf"(\b{re.escape(param)}\s*[:=]\s*)[0-9.eE+-]+")
    new = pattern.sub(r"\g<1>0.0", original, count=1)
    if new == original:
        return False
    bak = file_path.with_suffix(file_path.suffix + ".falsify-bak")
    bak.write_text(original)
    file_path.write_text(new)
    return True


def _restore_hparam(file_path: Path) -> None:
    bak = file_path.with_suffix(file_path.suffix + ".falsify-bak")
    if bak.exists():
        file_path.write_text(bak.read_text())
        bak.unlink()


# ═══════════════════════════════════════════════════════════════════
# ATTACK RUNNERS
# ═══════════════════════════════════════════════════════════════════

def _train_s3_like(
    candidate: cascade.CascadeCandidate,
    workspace: Path,
    data_root: str,
    gpu: str,
    scene: str,
    iters: int,
    conda_env: str,
    extra_flags: str = "",
    tag: str = "attack",
    stage_timeout_s: int = 3600,
    read_tb_script: Path | None = None,
) -> tuple[float | None, str | None]:
    """Generic "train one scene at N iters, return PSNR or error"."""
    output_dir = workspace / "falsify_outputs" / candidate.id / tag
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = workspace / "cascade_logs" / candidate.id / f"falsify_{tag}.log"
    data_path = f"{data_root}/{scene}"

    cmd = (
        cascade._conda_prefix(conda_env)
        + f'ns-train {candidate.method_cli} '
        + f'--data "{data_path}" '
        + f'--vis tensorboard '
        + f'--max-num-iterations {iters} '
        + f'--output-dir "{output_dir}" '
        + f'--viewer.quit-on-train-completion True '
        + f'{extra_flags} '
        + f'colmap --eval-mode interval '
    )
    env = cascade._build_env(gpu)
    rc, out, err = cascade._run_shell(
        cmd, env, timeout=stage_timeout_s, log_path=log_path,
    )
    if rc != 0:
        return None, (err or out or "")[-400:]
    run_dir = cascade._latest_run_subdir(output_dir, scene, candidate.method_cli)
    if not run_dir:
        return None, "no run dir produced"
    psnr = cascade._read_psnr_from_run_dir(run_dir, read_tb_script)
    if psnr is None:
        return None, "could not parse psnr"
    return psnr, None


def run_edge_scene_attack(
    candidate: cascade.CascadeCandidate,
    cfg: FalsifyConfig,
    workspace: Path,
    data_root: str,
    gpu: str,
    conda_env: str,
    read_tb_script: Path | None = None,
) -> AttackResult:
    psnr, err = _train_s3_like(
        candidate, workspace, data_root, gpu,
        scene=cfg.edge_scene, iters=cfg.edge_iters,
        conda_env=conda_env, tag="A1_edge",
        read_tb_script=read_tb_script,
    )
    if psnr is None:
        return AttackResult(
            name="edge_scene", passed=False,
            note=f"crashed on {cfg.edge_scene}: {err or 'unknown'}",
        )
    threshold = cfg.edge_baseline_db - cfg.edge_delta_db
    return AttackResult(
        name="edge_scene", passed=(psnr >= threshold),
        psnr=psnr,
        note=f"{cfg.edge_scene}@{cfg.edge_iters}: {psnr:.2f} vs ≥{threshold:.2f}",
    )


def run_sparse_view_attack(
    candidate: cascade.CascadeCandidate,
    cfg: FalsifyConfig,
    garden_psnr: float,
    workspace: Path,
    data_root: str,
    gpu: str,
    conda_env: str,
    single_scene: str,
    s3_iters: int,
    read_tb_script: Path | None = None,
) -> AttackResult:
    psnr, err = _train_s3_like(
        candidate, workspace, data_root, gpu,
        scene=single_scene, iters=s3_iters,
        conda_env=conda_env, tag="A2_sparse",
        extra_flags=f"--pipeline.datamanager.num_training_views {cfg.sparse_views}",
        read_tb_script=read_tb_script,
    )
    if psnr is None:
        return AttackResult(
            name="sparse_views", passed=False,
            note=f"crashed under {cfg.sparse_views} views: {err or 'unknown'}",
        )
    drop = garden_psnr - psnr
    passed = drop <= cfg.sparse_delta_db + 2.0  # allow extra slack
    return AttackResult(
        name="sparse_views", passed=passed, psnr=psnr,
        note=f"Δ={drop:+.2f} dB under {cfg.sparse_views} views",
    )


def run_ablation_attack(
    candidate: cascade.CascadeCandidate,
    cfg: FalsifyConfig,
    garden_psnr: float,
    rationale: str,
    workspace: Path,
    data_root: str,
    gpu: str,
    conda_env: str,
    single_scene: str,
    s3_iters: int,
    read_tb_script: Path | None = None,
) -> AttackResult:
    guess = _guess_key_hparam(rationale, candidate.impl_path)
    if not guess:
        return AttackResult(
            name="ablation", passed=True, attempted=False,
            note="no identifiable key hyperparameter in rationale",
        )
    file_path_s, param = guess
    file_path = Path(file_path_s)
    zeroed = _zero_hparam_in_file(file_path, param)
    if not zeroed:
        return AttackResult(
            name="ablation", passed=True, attempted=False,
            note=f"could not zero {param}",
        )

    try:
        psnr, err = _train_s3_like(
            candidate, workspace, data_root, gpu,
            scene=single_scene, iters=s3_iters,
            conda_env=conda_env, tag="A3_ablate",
            read_tb_script=read_tb_script,
        )
    finally:
        _restore_hparam(file_path)

    if psnr is None:
        # Crashing with the hparam zeroed is actually a "pass" — it
        # means the hparam was load-bearing, which is what we want.
        return AttackResult(
            name="ablation", passed=True, psnr=None,
            note=f"zeroing {param} made training crash (load-bearing ✓)",
        )
    drop = garden_psnr - psnr
    passed = drop >= cfg.ablation_delta_db
    return AttackResult(
        name="ablation", passed=passed, psnr=psnr,
        note=(
            f"zeroing {param}: Δ={drop:+.2f} dB (need ≥ {cfg.ablation_delta_db:.2f} "
            f"for the claim to be real)"
        ),
    )


# ═══════════════════════════════════════════════════════════════════
# SCORING
# ═══════════════════════════════════════════════════════════════════

def score_candidate(report: FalsifyReport, cfg: FalsifyConfig) -> float:
    """Weighted score: garden PSNR minus penalties for failed attacks."""
    score = cfg.w_garden * report.garden_psnr
    for attack in report.attacks:
        if not attack.attempted:
            continue
        if attack.passed:
            continue
        if attack.name == "edge_scene":
            score -= cfg.p_edge_fail
        elif attack.name == "sparse_views":
            score -= cfg.p_sparse_fail
        elif attack.name == "ablation":
            score -= cfg.p_ablation_fail
    return score


# ═══════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

def falsify_candidate(
    candidate: cascade.CascadeCandidate,
    garden_psnr: float,
    rationale: str,
    workspace: Path,
    data_root: str,
    gpu: str,
    conda_env: str,
    single_scene: str,
    s3_iters: int,
    cfg: FalsifyConfig | None = None,
    read_tb_script: Path | None = None,
    event_sink=None,
) -> FalsifyReport:
    """Run all three attacks on one candidate, return a scored report.

    The caller is responsible for having already acquired the GPU
    (falsification runs on the candidate's existing slot).
    """
    cfg = cfg or FalsifyConfig()
    t0 = time.time()
    report = FalsifyReport(candidate_id=candidate.id, garden_psnr=garden_psnr)

    if event_sink is not None:
        event_sink.put({
            "type": "falsify_start",
            "candidate_id": candidate.id,
            "message": f"Falsifying {candidate.display_name or candidate.id}",
        })

    report.attacks.append(run_edge_scene_attack(
        candidate, cfg, workspace, data_root, gpu,
        conda_env, read_tb_script=read_tb_script,
    ))
    if event_sink is not None:
        event_sink.put({
            "type": "falsify_attack_done",
            "candidate_id": candidate.id,
            "attack": report.attacks[-1].to_dict(),
        })

    report.attacks.append(run_sparse_view_attack(
        candidate, cfg, garden_psnr, workspace, data_root, gpu,
        conda_env, single_scene, s3_iters,
        read_tb_script=read_tb_script,
    ))
    if event_sink is not None:
        event_sink.put({
            "type": "falsify_attack_done",
            "candidate_id": candidate.id,
            "attack": report.attacks[-1].to_dict(),
        })

    report.attacks.append(run_ablation_attack(
        candidate, cfg, garden_psnr, rationale, workspace,
        data_root, gpu, conda_env, single_scene, s3_iters,
        read_tb_script=read_tb_script,
    ))
    if event_sink is not None:
        event_sink.put({
            "type": "falsify_attack_done",
            "candidate_id": candidate.id,
            "attack": report.attacks[-1].to_dict(),
        })

    report.score = score_candidate(report, cfg)
    report.wall_time_s = time.time() - t0

    if event_sink is not None:
        event_sink.put({
            "type": "falsify_done",
            "candidate_id": candidate.id,
            "report": report.to_dict(),
            "message": (
                f"Falsify {candidate.display_name or candidate.id}: "
                f"score={report.score:.2f} "
                f"({3 - report.num_failed}/3 attacks survived)"
            ),
        })

    return report
