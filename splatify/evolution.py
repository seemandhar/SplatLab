"""
Sub-project B orchestrator — wires EVOLVE-BLOCK, EoH, Elo + verbal
gradients, falsification, and the re-ideation feedback loop into a
single entrypoint.

Execution model:

    for cycle in 0..max_re_ideations:
        annotate EVOLVE-BLOCKs in impl_path (idempotent)
        run_discover_pipeline(
            impl_path,
            pre_generation_hook  = EoH-driven strategy planner,
            post_generation_hook = Elo update + falsify + gradients,
        )
        if best_psnr >= baseline_psnr:        # we succeeded
            return result
        # feedback loop: pull in negative examples and re-ideate
        impl_path = run_negative_ideator(result, workspace)

State that persists across generations (and across re-ideations):
    - EloState                  (workspace/elo/state.json)
    - OperatorScheduler         (workspace/eoh/state.json)
    - lessons.jsonl             (workspace/elo/lessons.jsonl)
    - ri_pairs.jsonl            (workspace/eoh/ri_pairs.jsonl)
    - falsify_reports/*.json    (workspace/falsify/)

Everything is resumable: if the job is killed mid-run and restarted
(through the web UI's rehydration), the new EvolutionContext loads
the same state files and continues from where it left off.
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import cascade
from . import eoh as eoh_mod
from . import evolve_blocks
from . import elo_gradients as elo_mod
from . import falsify as falsify_mod
from .job_store import EventSink


# ═══════════════════════════════════════════════════════════════════
# CONTEXT
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EvolutionContext:
    """Carries all cross-generation state for one evolutionary run."""
    workspace: Path
    elo: elo_mod.EloState
    scheduler: eoh_mod.OperatorScheduler
    ri_pool: list[eoh_mod.RIPair] = field(default_factory=list)
    cycle: int = 0
    re_ideation_count: int = 0
    # Seeded RNG so re-runs are reproducible given the same workspace.
    rng: random.Random = field(default_factory=lambda: random.Random(0xEB01))

    @classmethod
    def load_or_new(cls, workspace: Path) -> "EvolutionContext":
        workspace = Path(workspace)
        elo = elo_mod.EloState.load(workspace)
        sched = eoh_mod.OperatorScheduler.load(workspace)
        ctx = cls(workspace=workspace, elo=elo, scheduler=sched)
        ctx._load_ri_pool()
        return ctx

    def save(self) -> None:
        self.elo.save(self.workspace)
        self.scheduler.save(self.workspace)
        self._save_ri_pool()

    # ── RI pair persistence ────────────────────────────────────

    def _ri_path(self) -> Path:
        return self.workspace / "eoh" / "ri_pairs.jsonl"

    def _load_ri_pool(self) -> None:
        path = self._ri_path()
        if not path.exists():
            return
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                self.ri_pool.append(eoh_mod.RIPair(
                    key=d["key"],
                    rationale=d.get("rationale", ""),
                    impl_path=Path(d.get("impl_path", "")),
                    elo=float(d.get("elo", 1500.0)),
                    psnr=float(d.get("psnr", 0.0)),
                    operator_used=d.get("operator_used", ""),
                    parents=list(d.get("parents", [])),
                    hooks_touched=list(d.get("hooks_touched", [])),
                ))
            except Exception:
                continue

    def _save_ri_pool(self) -> None:
        path = self._ri_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            for p in self.ri_pool:
                f.write(json.dumps(p.to_dict()) + "\n")

    # ── pool maintenance ───────────────────────────────────────

    def add_ri_pair(self, pair: eoh_mod.RIPair) -> None:
        # Refresh Elo from the live ladder.
        pair.elo = self.elo.get(pair.key)
        self.ri_pool.append(pair)
        # Trim pool to top-24 by Elo to bound memory/context.
        if len(self.ri_pool) > 24:
            self.ri_pool.sort(key=lambda p: p.elo, reverse=True)
            self.ri_pool = self.ri_pool[:24]


# ═══════════════════════════════════════════════════════════════════
# HOOK IMPLEMENTATIONS (called from discover.run_discover_pipeline)
# ═══════════════════════════════════════════════════════════════════

def _candidate_key(cycle: int, gen_index: int, island_id: int) -> str:
    """Stable Elo key. Includes cycle so re-ideated runs don't collide."""
    return f"c{cycle}_g{gen_index}_i{island_id}"


def make_pre_generation_hook(
    ctx: EvolutionContext,
    event_sink: EventSink | None,
) -> Callable:
    """Build a pre_generation_hook bound to this context.

    The hook returns the list of strategies discover.py should use
    for the next generation. We build it from the EoH operator
    scheduler; each "strategy" carries the operator name + a
    lessons-augmented prompt that tells the discoverer agent how to
    mutate.
    """
    def hook(gen_index: int, default_strategies: list[dict]) -> list[dict]:
        n_slots = len(default_strategies)
        plan = ctx.scheduler.plan_generation(n_slots)
        lessons = elo_mod.build_lessons_digest(ctx.workspace)

        # Prepare strategy dicts in the shape discover.py expects.
        strategies: list[dict] = []
        for i, op in enumerate(plan):
            parents = eoh_mod.pick_parents(ctx.ri_pool, op, ctx.rng)
            parent_summaries = " | ".join(
                f"{p.key} (psnr={p.psnr:.2f}, elo={p.elo:.0f})"
                for p in parents
            ) or "(no parents yet — use baseline)"
            description = (
                f"[EoH-{op}] {eoh_mod.operator_prompt(op).splitlines()[0]}\n"
                f"  parents: {parent_summaries}\n"
                f"  hooks in scope: any EVOLVE-BLOCK"
            )
            strategies.append({
                "id": f"eoh_{op}_{i}",
                "name": f"EoH-{op}",
                "emoji": {"E1": "⨯", "E2": "⨯", "M1": "~", "M2": "π", "M3": "§"}[op],
                "description": description,
                # Carry hidden state that post_generation_hook uses to
                # reconstruct parent linkage.
                "_eoh_operator": op,
                "_eoh_parents": [p.key for p in parents],
                "_eoh_lessons": lessons,
            })

        if event_sink is not None:
            event_sink.put({
                "type": "evolution_plan",
                "generation": gen_index,
                "plan": [s["name"] for s in strategies],
                "pool_size": len(ctx.ri_pool),
                "message": (
                    f"Gen {gen_index} EoH plan: "
                    + " / ".join(s["name"] for s in strategies)
                    + f"  (pool={len(ctx.ri_pool)})"
                ),
            })
        return strategies
    return hook


def make_post_generation_hook(
    ctx: EvolutionContext,
    event_sink: EventSink | None,
    cfg,
    config,
    gpu_queue,
    data_root: str,
    run_verbal_gradient: Callable | None = None,
) -> Callable:
    """Build a post_generation_hook.

    After a generation's candidates have finished training and had
    their PSNR measured, this hook:
      1. Seeds Elo entries for each alive candidate.
      2. Runs all-pairs Elo updates.
      3. Records per-operator credit assignment from Elo delta.
      4. Fires off verbal gradient agents for surprising-loss pairs.
      5. Runs falsification on the top-K survivors.
      6. Saves state to disk.
    """
    def hook(gen_index: int, gen_data, candidates, gen_workspace):
        # Build ScoredCandidate views + RIPair upserts.
        scored: list[elo_mod.ScoredCandidate] = []
        for c in candidates:
            key = _candidate_key(ctx.cycle, gen_index, c.id)
            scored.append(elo_mod.ScoredCandidate(
                key=key,
                display_name=f"gen{gen_index}-i{c.id}",
                psnr=float(getattr(c, "psnr", 0.0) or 0.0),
                alive=bool(getattr(c, "alive", False)),
                impl_path=Path(getattr(c, "path", gen_workspace)),
                rationale=getattr(c, "improvement_summary", "") or "",
                hooks_touched=[],
            ))
            if c.alive and c.psnr > 0:
                # Extract operator/parents from strategy dict if present.
                strat = getattr(c, "strategy", {}) or {}
                op = strat.get("_eoh_operator", "")
                parents = strat.get("_eoh_parents", []) or []
                pair = eoh_mod.RIPair(
                    key=key,
                    rationale=getattr(c, "improvement_summary", "") or "",
                    impl_path=Path(getattr(c, "path", gen_workspace)),
                    elo=ctx.elo.get(key),
                    psnr=float(c.psnr),
                    operator_used=op,
                    parents=list(parents),
                    hooks_touched=[],
                )
                ctx.add_ri_pair(pair)

        # Snapshot pre-update Elo so we can compute per-operator gain.
        pre_elo = {s.key: ctx.elo.get(s.key) for s in scored}

        pairs = elo_mod.update_from_generation(ctx.elo, scored)

        # Credit assignment for EoH operators:
        for c in candidates:
            if not (c.alive and c.psnr > 0):
                continue
            strat = getattr(c, "strategy", {}) or {}
            op = strat.get("_eoh_operator")
            if not op:
                continue
            key = _candidate_key(ctx.cycle, gen_index, c.id)
            delta = ctx.elo.get(key) - pre_elo.get(key, 1500.0)
            ctx.scheduler.record_outcome(op, delta)

        # Emit the current Elo ladder for the UI.
        if event_sink is not None:
            ladder = [
                {"key": k, "rating": round(v, 1)}
                for k, v in ctx.elo.ranking()[:12]
            ]
            event_sink.put({
                "type": "elo_ladder",
                "generation": gen_index,
                "ladder": ladder,
                "message": f"Elo ladder updated (top-{len(ladder)})",
            })

        # Verbal gradients on surprising pairs.
        # run_verbal_gradient is async; we're called synchronously from
        # inside discover.py's async loop, so the standard approach is
        # to run the coroutine in a fresh thread with its own event
        # loop (mirrors discover.py's _debug_agent_in_thread pattern).
        def _run_gradient_sync(w, l, wp, lp):
            import threading as _t
            result_box: dict[str, Any] = {"gradient": None, "error": None}
            def _worker():
                try:
                    result_box["gradient"] = asyncio.run(
                        run_verbal_gradient(w, l, wp, lp, event_sink=event_sink)
                    )
                except Exception as _e:
                    result_box["error"] = _e
            t = _t.Thread(target=_worker, daemon=True)
            t.start(); t.join(timeout=150.0)
            if result_box["error"]:
                raise result_box["error"]
            return result_box["gradient"]

        if run_verbal_gradient is not None and pairs:
            for winner_k, loser_k, margin in elo_mod.surprising_pairs(ctx.elo, pairs):
                winner = next((s for s in scored if s.key == winner_k), None)
                loser = next((s for s in scored if s.key == loser_k), None)
                if not winner or not loser or not winner.impl_path or not loser.impl_path:
                    continue
                try:
                    gradient = _run_gradient_sync(
                        winner, loser, winner.impl_path, loser.impl_path,
                    )
                except Exception as e:
                    gradient = None
                    if event_sink is not None:
                        event_sink.put({
                            "type": "gradient_error",
                            "error": str(e),
                            "message": f"Gradient agent failed: {e}",
                        })
                if gradient:
                    gradient.winner_key = winner_k
                    gradient.loser_key = loser_k
                    gradient.margin_db = margin
                    gradient.generation = gen_index
                    elo_mod.append_lesson(ctx.workspace, gradient)
                    if event_sink is not None:
                        event_sink.put({
                            "type": "verbal_gradient",
                            "winner": winner_k,
                            "loser": loser_k,
                            "margin_db": round(margin, 2),
                            "lesson": gradient.lesson,
                            "hooks": gradient.hooks,
                        })

        # Falsification on the top-2 alive candidates (by PSNR within
        # this generation). We cap at 2 because falsify_candidate runs
        # three training jobs each — cheap per candidate but expensive
        # at fleet scale.
        alive = sorted(
            [s for s in scored if s.alive and s.psnr > 0],
            key=lambda s: s.psnr, reverse=True,
        )[:2]
        for s in alive:
            cand = cascade.CascadeCandidate(
                id=s.key,
                method_cli=_method_cli_from_scored(candidates, s),
                impl_path=s.impl_path or gen_workspace,
                display_name=s.display_name,
            )
            # Acquire a GPU from the shared queue and release on exit.
            gpu = gpu_queue.get()
            try:
                report = falsify_mod.falsify_candidate(
                    candidate=cand,
                    garden_psnr=s.psnr,
                    rationale=s.rationale,
                    workspace=ctx.workspace,
                    data_root=data_root,
                    gpu=str(gpu),
                    conda_env=config.conda_env,
                    single_scene=getattr(config, "cascade_single_scene", "garden"),
                    s3_iters=getattr(config, "cascade_single_scene_iters", 7000),
                    cfg=falsify_mod.FalsifyConfig(),
                    event_sink=event_sink,
                )
                # Persist report.
                reports_dir = ctx.workspace / "falsify"
                reports_dir.mkdir(parents=True, exist_ok=True)
                (reports_dir / f"{s.key}.json").write_text(
                    json.dumps(report.to_dict(), indent=2)
                )
                # Demote the candidate's Elo if it failed attacks —
                # this is the penalty that makes robustness matter.
                if report.num_failed > 0:
                    penalty = 30.0 * report.num_failed
                    ctx.elo.ratings[s.key] = ctx.elo.get(s.key) - penalty
                    if event_sink is not None:
                        event_sink.put({
                            "type": "falsify_penalty",
                            "candidate_id": s.key,
                            "penalty": penalty,
                            "message": (
                                f"Falsify penalty: {s.key} -{penalty:.0f} Elo "
                                f"({report.num_failed} attack(s) failed)"
                            ),
                        })
            finally:
                gpu_queue.put(gpu)

        ctx.save()
        return None  # discover.py interprets None = use its own default logic
    return hook


def _method_cli_from_scored(candidates, scored) -> str:
    """Best-effort method_cli lookup — matches scored.key back to Island."""
    for c in candidates:
        if f"i{c.id}" in scored.key:
            return getattr(c, "method_name_cli", "")
    return ""


# ═══════════════════════════════════════════════════════════════════
# OUTER FEEDBACK LOOP
# ═══════════════════════════════════════════════════════════════════

async def run_evolution_pipeline(
    impl_path: str,
    workspace: Path,
    event_queue,
    stop_event: threading.Event | None,
    gpu: str,
    data_path: str | None,
    scene: str,
    max_iters: int,
    model: str | None,
    num_views: int,
    max_generations: int,
    max_re_ideations: int = 2,
    run_verbal_gradient: Callable | None = None,
    run_negative_ideator: Callable | None = None,
) -> dict:
    """Outer driver: up to max_re_ideations+1 discover cycles.

    A cycle runs the (EoH + Elo + falsify) inner loop via
    discover.run_discover_pipeline. If the winner doesn't beat the
    baseline, and we have re-ideations left, we invoke the negative
    ideator to produce a fresh seed implementation and run again.
    """
    # Imports are local because discover.py imports us too in some
    # paths — avoid circular import at module load time.
    from .discover import run_discover_pipeline
    from splatify.splatify_config import PipelineConfig

    config = PipelineConfig()
    workspace = Path(workspace)
    ctx = EvolutionContext.load_or_new(workspace)
    data_root = (data_path or config.default_dataset or "").rstrip("/")

    if event_queue is not None:
        event_queue.put({
            "type": "evolution_start",
            "message": (
                f"Evolution v2: up to {max_re_ideations + 1} cycles × "
                f"{max_generations} gens, EoH + Elo + falsify"
            ),
            "max_re_ideations": max_re_ideations,
            "max_generations": max_generations,
        })

    # One-time: annotate EVOLVE-BLOCK markers in the seed implementation.
    marker_counts = evolve_blocks.annotate_implementation(Path(impl_path))
    if event_queue is not None:
        event_queue.put({
            "type": "evolve_blocks_annotated",
            "files": marker_counts,
            "message": (
                f"EVOLVE-BLOCK markers added to {len(marker_counts)} file(s) "
                f"in baseline implementation"
                if marker_counts else
                "EVOLVE-BLOCK markers already present"
            ),
        })

    final_result: dict = {}
    current_impl = impl_path

    for cycle in range(max_re_ideations + 1):
        ctx.cycle = cycle
        if event_queue is not None:
            event_queue.put({
                "type": "evolution_cycle_start",
                "cycle": cycle + 1,
                "total_cycles": max_re_ideations + 1,
                "impl_path": current_impl,
                "message": f"Cycle {cycle + 1}/{max_re_ideations + 1} starting",
            })

        # Build hooks bound to this cycle.
        pre_hook = make_pre_generation_hook(ctx, event_sink=event_queue)
        # post_hook needs gpu_queue, which discover creates internally —
        # we pass a factory and discover.py wires it at generation time.
        def post_hook_factory(gpu_queue):
            return make_post_generation_hook(
                ctx, event_sink=event_queue, cfg=None, config=config,
                gpu_queue=gpu_queue, data_root=data_root,
                run_verbal_gradient=run_verbal_gradient,
            )

        result = await run_discover_pipeline(
            impl_path=current_impl,
            workspace=workspace,
            event_queue=event_queue,
            stop_event=stop_event,
            gpu=gpu,
            data_path=data_path,
            scene=scene,
            max_iters=max_iters,
            model=model,
            num_views=num_views,
            max_generations=max_generations,
            pre_generation_hook=pre_hook,
            post_generation_hook_factory=post_hook_factory,
        )
        final_result = result

        baseline = float(result.get("baseline_psnr", 0.0) or 0.0)
        best = float(result.get("final_best_psnr", 0.0) or 0.0)

        cleared = best > 0 and (baseline == 0 or best >= baseline)
        if event_queue is not None:
            event_queue.put({
                "type": "evolution_cycle_done",
                "cycle": cycle + 1,
                "best_psnr": best,
                "baseline_psnr": baseline,
                "cleared_baseline": cleared,
                "message": (
                    f"Cycle {cycle + 1} done — best {best:.2f} vs baseline {baseline:.2f} "
                    + ("✓ cleared" if cleared else "✗ below baseline")
                ),
            })

        if cleared or cycle >= max_re_ideations:
            break

        if stop_event is not None and stop_event.is_set():
            break

        # Re-ideate. The negative ideator reads this cycle's failed
        # rationales + final PSNRs and emits a fresh seed implementation.
        if run_negative_ideator is None:
            if event_queue is not None:
                event_queue.put({
                    "type": "evolution_skip_reideation",
                    "message": "No negative ideator provided — stopping.",
                })
            break

        ctx.re_ideation_count += 1
        if event_queue is not None:
            event_queue.put({
                "type": "evolution_reideation_start",
                "attempt": ctx.re_ideation_count,
                "message": (
                    f"Re-ideation #{ctx.re_ideation_count}: "
                    f"best {best:.2f} < baseline {baseline:.2f}, "
                    f"asking negative ideator for a fresh seed"
                ),
            })

        try:
            new_impl = await run_negative_ideator(
                prior_result=result,
                workspace=workspace,
                event_queue=event_queue,
            )
        except Exception as e:
            if event_queue is not None:
                event_queue.put({
                    "type": "evolution_reideation_failed",
                    "error": str(e),
                    "message": f"Re-ideation failed: {e}",
                })
            break

        if not new_impl:
            if event_queue is not None:
                event_queue.put({
                    "type": "evolution_reideation_failed",
                    "message": "Negative ideator returned no new implementation",
                })
            break

        # Re-annotate the new implementation.
        evolve_blocks.annotate_implementation(Path(new_impl))
        current_impl = str(new_impl)

    ctx.save()
    if event_queue is not None:
        event_queue.put({
            "type": "evolution_done",
            "cycles_run": ctx.cycle + 1,
            "re_ideations": ctx.re_ideation_count,
            "result": final_result,
            "message": (
                f"Evolution v2 done after {ctx.cycle + 1} cycle(s) "
                f"({ctx.re_ideation_count} re-ideation(s))"
            ),
        })
    return final_result
