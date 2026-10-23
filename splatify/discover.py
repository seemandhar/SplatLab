#!/usr/bin/env python3
"""
Discovery Pipeline — Iterative Evolutionary Improvement for Gaussian Splatting.

Unlike the tournament-style /invent pipeline which has fixed rounds with elimination,
/discover runs indefinitely, each generation building on the previous best:

  Gen 1: baseline + 7 variants → train N iters → pick best
  Gen 2: 8 variants of best   → train N iters → pick best
  Gen 3: 8 variants of best   → train N iters → pick best
  ... until user clicks STOP

Accumulates knowledge across generations — each agent knows what worked and what didn't.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import queue
import re
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    ToolResultBlock,
)

from splatify.splatify_config import PipelineConfig, get_claude_auth_env
from splatify.invent import (
    Island,
    STRATEGIES,
    push_event,
    _find_method_pkg_dir,
    _find_method_name_from_impl,
    _read_psnr_from_tb,
    _uniquify_island_package,
    _install_method,
    _train_method,
    _uninstall_all_islands,
    _run_debug_agent,
    _island_summary,
    MAX_DEBUG_RETRIES,
)


# ═══════════════════════════════════════════════════════════════════
# DISCOVERER AGENT PROMPT
# ═══════════════════════════════════════════════════════════════════

DISCOVERER_PROMPT = r"""You are a Discovery Agent in an iterative evolutionary improvement pipeline
for Gaussian Splatting. You are competing against {num_competitors} other agents.
The winner of each generation becomes the base for the next generation.

## Generation {generation}

## Your Starting Point / Suggestion
Your assigned category is "{strategy_name}": {strategy_description}

This is a SUGGESTION, not a constraint. You are free to pursue ANY approach you
believe will maximize PSNR — combine multiple strategies, invent new ones, or
ignore the suggestion entirely if you have a better idea.

## Current State
- Baseline PSNR (original, unmodified): {baseline_psnr} dB
- Current best PSNR to beat: {current_best_psnr} dB
- Your goal: BEAT {current_best_psnr} dB

## Previous Generations — What Worked and What Didn't
{history}

## Your Task
1. Read the current best implementation at: {candidate_dir}
   - Read ALL files in {method_pkg}/ (model.py, configs.py, losses.py, rendering.py, datamanager.py, __init__.py)
   - Read pyproject.toml
2. Read the gs_template framework for reference:
   - {gs_root}/model.py — Base class with 9 extension points
   - {gs_root}/rendering.py — Rendering helpers
   - {gs_root}/losses.py — 30+ reusable loss functions
   - {gs_root}/shading.py — SH, normals, cubemaps
   - {gs_root}/configs.py — MethodSpecification examples
3. Read 2-3 example methods from {methods_dir}/ for inspiration
4. **Search the web** (WebSearch / WebFetch) for the latest Gaussian Splatting
   improvements, tricks, and techniques from 2024-2026. Look for things like:
   - Recent high-PSNR methods and what makes them work
   - Novel loss functions, densification strategies, or rendering tricks
   - Hyperparameter tuning insights from papers and blog posts
   - Techniques from Mini-Splatting, Pixel-GS, GOF, AbsGS, Spec-Gaussian, etc.
   You are encouraged to search the web EVERY generation — the field moves fast
   and new ideas can compound on top of previous improvements.
5. Apply whatever improvements you think will BEAT {current_best_psnr} dB.
   You can combine ideas from the web, from example methods, and from your own
   reasoning. There are no constraints on what you change as long as the method
   still installs and trains.

## Critical Rules
- You MUST rename the method to "{method_name}" in ALL files:
  - {method_pkg}/configs.py: method_name="{method_name_cli}"
  - pyproject.toml: entry point must be "{method_name_cli}" = "{method_pkg}.configs:{method_name_var}"
  - {method_pkg}/__init__.py: update variable names if needed
- All internal imports must use "{method_pkg}" as the package name
- Keep the method installable: `pip install -e .` must work
- Keep the method trainable: `ns-train {method_name_cli} --data <path> --max-num-iterations 2000` must work
- Do NOT break existing functionality — enhance it
- Write complete files (no patches), use the Write tool
- Focus on changes that MAXIMIZE PSNR — you are being judged on this metric
- LEARN from previous generations: don't repeat approaches that failed, build on what worked
- The current code already incorporates all winning strategies from previous generations
- Be bold but practical — untested theoretical changes may crash training
- Do NOT run pip install or ns-train yourself — the pipeline handles that

NOTE: The code you're reading was modified by previous agents. Variable names and method
names may not follow standard patterns. When renaming, make sure to update ALL references.

## Output
After making changes, write a brief summary to {candidate_dir}/improvement_summary.txt explaining:
- What you changed and why
- Expected impact on PSNR
- Any risks or tradeoffs
"""


# ═══════════════════════════════════════════════════════════════════
# IDEA REFINER PROMPT (OPUS) — generates 8 strategies for next gen
# ═══════════════════════════════════════════════════════════════════

IDEA_REFINER_PROMPT = r"""You are an elite Gaussian Splatting research strategist. Your task is to
analyze the results of the latest discovery generation and produce exactly 8 focused
improvement strategies for the next generation of agents.

## Current State
- Baseline PSNR (original, unmodified): {baseline_psnr} dB
- Current global best PSNR: {global_best_psnr} dB
- Latest generation best PSNR: {gen_best_psnr} dB

## Full History of All Generations
{history}

## Best Implementation So Far
The best implementation is at: {best_impl_path}
Read the key files ({method_pkg}/model.py, configs.py, losses.py, rendering.py) to understand
what has already been tried and what the current code does.

## GS Framework Reference
- {gs_root}/model.py — Base class with 9 extension points
- {gs_root}/losses.py — 30+ reusable loss functions
- {gs_root}/rendering.py — Rendering helpers
- {gs_root}/shading.py — SH, normals, cubemaps

## Your Task

Analyze what worked, what failed, and WHY. Then produce exactly 8 targeted improvement
strategies for the next generation. Each strategy should be:
- **Specific**: Not "improve losses" but "add frequency-domain L1 loss on DCT coefficients
  with weight 0.05, targeting high-frequency detail recovery"
- **Informed**: Based on what the history shows works/fails for THIS implementation
- **Diverse**: Cover different aspects (don't put all eggs in one basket)
- **Actionable**: An agent should be able to implement it in ~30 minutes of coding

Search the web for the latest Gaussian Splatting papers and techniques (2024-2026) to inform
your strategies. Look for novel ideas that haven't been tried yet.

## Output Format

You MUST write your output to {output_path} as a JSON file with exactly this structure:
```json
[
  {{
    "id": "strategy_1",
    "name": "Short Name (2-4 words)",
    "emoji": "S",
    "description": "Detailed description of what to change, why, and expected impact. Be specific about code changes, hyperparameters, and techniques."
  }},
  ... (exactly 8 strategies)
]
```

Write the file using the Write tool. Do NOT include any other text in the JSON file.
"""


async def _run_idea_refiner(
    workspace: Path,
    gen_index: int,
    baseline_psnr: float,
    global_best_psnr: float,
    gen_best_psnr: float,
    history: str,
    best_impl_path: Path,
    base_pkg: str,
    config: PipelineConfig,
    event_queue: queue.Queue | None = None,
) -> list[dict]:
    """Run an OPUS idea refiner agent to generate 8 strategies for the next generation.

    Returns a list of 8 strategy dicts, or falls back to default STRATEGIES.
    """
    output_path = workspace / f"strategies_gen_{gen_index + 1}.json"

    prompt = IDEA_REFINER_PROMPT.format(
        baseline_psnr=f"{baseline_psnr:.2f}" if baseline_psnr > 0 else "N/A",
        global_best_psnr=f"{global_best_psnr:.2f}" if global_best_psnr > 0 else "N/A",
        gen_best_psnr=f"{gen_best_psnr:.2f}" if gen_best_psnr > 0 else "N/A",
        history=history,
        best_impl_path=str(best_impl_path),
        method_pkg=base_pkg,
        gs_root=str(config.gs_template_root),
        output_path=str(output_path),
    )

    push_event(event_queue, "stage", {
        "stage": "refining_ideas",
        "generation": gen_index,
        "message": f"Gen {gen_index}: OPUS refining ideas for next generation...",
    })

    agent_opts = ClaudeAgentOptions(
        cwd=str(best_impl_path),
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
        max_turns=40,
        model="claude-opus-4-6",
        env=get_claude_auth_env(),
    )

    try:
        async for message in query(prompt=prompt, options=agent_opts):
            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(event_queue, "refiner_log", {
                                "generation": gen_index,
                                "message": text[:200],
                            })
            elif isinstance(message, ResultMessage):
                result_usage = getattr(message, "usage", None)
                cost = getattr(message, "total_cost_usd", None)
                push_event(event_queue, "refiner_done", {
                    "generation": gen_index,
                    "cost_usd": cost,
                    "message": f"Idea refiner complete for gen {gen_index + 1}",
                })
    except Exception as e:
        push_event(event_queue, "refiner_error", {
            "generation": gen_index,
            "error": str(e),
            "message": f"Idea refiner failed: {e} — falling back to default strategies",
        })
        return STRATEGIES[:8]

    # Parse the output
    if output_path.exists():
        try:
            strategies = json.loads(output_path.read_text())
            if isinstance(strategies, list) and len(strategies) == 8:
                # Validate each has required fields
                for s in strategies:
                    for key in ("id", "name", "description"):
                        if key not in s:
                            raise ValueError(f"Strategy missing '{key}' field")
                    s.setdefault("emoji", s["name"][0].upper())
                push_event(event_queue, "strategies_generated", {
                    "generation": gen_index + 1,
                    "strategies": [{"name": s["name"], "id": s["id"]} for s in strategies],
                    "message": f"8 new strategies generated for gen {gen_index + 1}",
                })
                return strategies
            else:
                push_event(event_queue, "refiner_error", {
                    "generation": gen_index,
                    "message": f"Expected 8 strategies, got {len(strategies) if isinstance(strategies, list) else 'non-list'}",
                })
        except (json.JSONDecodeError, ValueError) as e:
            push_event(event_queue, "refiner_error", {
                "generation": gen_index,
                "error": str(e),
                "message": f"Failed to parse strategies: {e}",
            })

    return STRATEGIES[:8]


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class DiscoverGeneration:
    """Tracks results for one generation in the discover pipeline."""
    index: int
    candidates: list[dict] = field(default_factory=list)   # _island_summary dicts
    best_id: int | None = None
    best_psnr: float = 0.0
    best_strategy: str = ""
    best_summary: str = ""
    input_psnr: float = 0.0   # PSNR of the base going into this generation


def _rename_baseline_method(island_dir: Path, base_pkg: str,
                            old_cli: str, new_cli: str,
                            old_var: str, new_var: str):
    """Rename method identifiers for the baseline candidate (no agent modifications).

    This handles renaming the method_name, variable name, and entry point
    so the baseline can be installed alongside other island packages.
    """
    # configs.py: method_name and MethodSpecification variable
    configs = island_dir / base_pkg / "configs.py"
    if configs.exists():
        content = configs.read_text()
        content = re.sub(
            r'method_name\s*=\s*["\']' + re.escape(old_cli) + r'["\']',
            f'method_name="{new_cli}"',
            content,
        )
        # Rename the MethodSpecification variable
        if old_var != new_var:
            content = content.replace(f'{old_var} =', f'{new_var} =')
        configs.write_text(content)

    # pyproject.toml: entry point
    pyproject = island_dir / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        # Replace the entry point line
        content = re.sub(
            r'^' + re.escape(old_cli) + r'(\s*=\s*"[^"]*\.configs:)' + re.escape(old_var) + r'"',
            f'{new_cli}\\1{new_var}"',
            content,
            flags=re.MULTILINE,
        )
        pyproject.write_text(content)

    # __init__.py: update imports
    init = island_dir / base_pkg / "__init__.py"
    if init.exists():
        content = init.read_text()
        if old_var != new_var:
            content = content.replace(old_var, new_var)
        init.write_text(content)


def _build_history(generations: list[DiscoverGeneration], baseline_psnr: float) -> str:
    """Build the history section for the discoverer prompt."""
    if not generations:
        return "No previous generations — this is the first attempt."

    lines = []
    for gen in generations:
        base_label = f"Base PSNR: {gen.input_psnr:.2f} dB" if gen.input_psnr > 0 else "Base: original"
        lines.append(f"### Generation {gen.index} ({base_label})")

        for cand in gen.candidates:
            psnr_str = f"{cand['psnr']:.2f} dB" if cand.get("psnr", 0) > 0 else "FAILED"
            winner = " ★ WINNER" if cand["id"] == gen.best_id else ""
            strategy = cand.get("strategy", "Unknown")
            summary = cand.get("improvement_summary", "")
            if summary:
                # First line of summary only
                summary = summary.split("\n")[0][:120]
                lines.append(f"  - {strategy}: {psnr_str}{winner} — {summary}")
            else:
                status = cand.get("status", "")
                if status in ("error", "eliminated"):
                    lines.append(f"  - {strategy}: {psnr_str} ({status})")
                else:
                    lines.append(f"  - {strategy}: {psnr_str}{winner}")

        if gen.best_psnr > 0:
            delta = gen.best_psnr - gen.input_psnr if gen.input_psnr > 0 else 0
            delta_str = f" (+{delta:.2f})" if delta > 0 else f" ({delta:.2f})" if delta < 0 else ""
            lines.append(f"  → Best: {gen.best_strategy} = {gen.best_psnr:.2f} dB{delta_str}")
        else:
            lines.append("  → All candidates failed this generation")
        lines.append("")

    total_improvement = generations[-1].best_psnr - baseline_psnr if baseline_psnr > 0 else 0
    if total_improvement != 0:
        lines.append(f"**Total improvement so far: {baseline_psnr:.2f} → {generations[-1].best_psnr:.2f} dB "
                     f"({'+'if total_improvement > 0 else ''}{total_improvement:.2f} dB)**")

    return "\n".join(lines)


def _copy_implementation(src_path: Path, dst_path: Path, base_pkg: str):
    """Copy an implementation's package dir and config files to a new directory."""
    dst_path.mkdir(parents=True, exist_ok=True)

    # Copy package directory
    src_pkg = src_path / base_pkg
    dst_pkg = dst_path / base_pkg
    if dst_pkg.exists():
        shutil.rmtree(dst_pkg)
    if src_pkg.exists():
        shutil.copytree(src_pkg, dst_pkg)

    # Copy config files
    for f in ["pyproject.toml", "README.md"]:
        src = src_path / f
        if src.exists():
            shutil.copy2(src, dst_path / f)


# ═══════════════════════════════════════════════════════════════════
# SAFE INSTALL + TRAIN
# ═══════════════════════════════════════════════════════════════════
#
# Problem: concurrent `pip install -e .` into the same conda env can
# corrupt metadata. And claude_agent_sdk query() uses anyio cancel
# scopes that crash when called across asyncio.run() boundaries.
#
# Solution: Use a threading.Lock to serialize pip installs while
# keeping training parallel. Debug agents run in their own thread
# via asyncio.run() to get a fresh event loop (same pattern as
# invent.py's _run_agent_in_thread).
# ═══════════════════════════════════════════════════════════════════

# Module-level lock: only one pip install at a time
_pip_lock = threading.Lock()


def _install_with_lock(island: Island, conda_env: str,
                       event_queue: queue.Queue | None) -> tuple[bool, str]:
    """Install with a lock so concurrent installs don't race."""
    with _pip_lock:
        return _install_method(island, conda_env, event_queue)


def _debug_agent_in_thread(island: Island, phase: str, error_output: str,
                           config: PipelineConfig,
                           event_queue: queue.Queue | None,
                           model: str | None) -> bool:
    """Run debug agent in its own thread + event loop to avoid anyio cross-task errors."""
    return asyncio.run(
        _run_debug_agent(island, phase, error_output, config, event_queue, model)
    )


def _run_candidate_full_pipeline(
    island: Island,
    needs_agent: bool,
    gen_index: int,
    current_best_psnr: float,
    baseline_psnr: float,
    history: str,
    base_pkg: str,
    gpu_queue: queue.Queue,
    max_iters: int,
    data_path: str,
    output_dir: Path,
    conda_env: str,
    scene: str,
    gen_workspace: Path,
    event_queue: queue.Queue | None,
    config: PipelineConfig | None = None,
    model: str | None = None,
) -> Path | None:
    """Full pipeline for one candidate, runs in its own thread.

    agent improve → save copy → uniquify → install → train → PSNR.
    Each step starts immediately after the previous — no waiting for
    other candidates.  GPU is acquired from a shared queue only when
    training starts, so agents + install can overlap freely.

    Returns the saved pre-uniquification path (for next gen base), or None.
    """
    _config = config or PipelineConfig()

    # ── 1. Run discoverer agent (skip for baseline) ──────────────
    if needs_agent:
        try:
            success = asyncio.run(
                _run_discoverer_agent(
                    island, gen_index, current_best_psnr, baseline_psnr,
                    history, _config, event_queue, model,
                )
            )
        except Exception as e:
            island.alive = False
            island.status = "error"
            island.error = str(e)
            push_event(event_queue, "island_error", {
                "island_id": island.id,
                "generation": gen_index,
                "error": str(e),
                "message": f"Gen {gen_index} · C{island.id}: Agent failed — {e}",
            })
            return None

        if not success:
            island.alive = False
            island.status = "error"
            return None

    if not island.alive:
        return None

    # ── 2. Save pre-uniquification copy ──────────────────────────
    saved = gen_workspace / f"saved_{island.id}"
    if saved.exists():
        shutil.rmtree(saved)
    shutil.copytree(island.path, saved)

    # ── 3. Uniquify package names ────────────────────────────────
    actual_pkg = _find_method_pkg_dir(island.path) or base_pkg
    ok = _uniquify_island_package(island, actual_pkg)
    if not ok:
        island.alive = False
        island.status = "error"
        island.error = "Package uniquification failed"
        push_event(event_queue, "island_error", {
            "island_id": island.id,
            "generation": gen_index,
            "error": "Package directory not found after agent improvement",
            "message": f"Gen {gen_index} · C{island.id}: Package rename failed",
        })
        return None

    push_event(event_queue, "island_log", {
        "island_id": island.id,
        "generation": gen_index,
        "message": f"Package → {island.pkg_name}",
    })

    # ── 4. Install + Train (with debug retries) ─────────────────
    for attempt in range(MAX_DEBUG_RETRIES + 1):
        retry_label = f" (retry {attempt})" if attempt > 0 else ""

        # Install (serialized via pip lock)
        island.status = "installing"
        push_event(event_queue, "island_status", {
            "island_id": island.id,
            "generation": gen_index,
            "status": "installing",
            "message": f"Gen {gen_index} · C{island.id}: Installing{retry_label}...",
        })

        install_ok, install_err = _install_with_lock(island, conda_env, event_queue)
        if not install_ok:
            if attempt < MAX_DEBUG_RETRIES:
                push_event(event_queue, "island_log", {
                    "island_id": island.id,
                    "generation": gen_index,
                    "message": f"Install failed, debugging{retry_label}...",
                })
                island.status = "debugging"
                push_event(event_queue, "island_status", {
                    "island_id": island.id,
                    "generation": gen_index,
                    "status": "debugging",
                })
                _debug_agent_in_thread(
                    island, "installation", install_err,
                    _config, event_queue, model,
                )
                continue
            else:
                island.alive = False
                island.status = "error"
                island.error = f"Install failed: {install_err[:200]}"
                island.psnr = 0.0
                push_event(event_queue, "island_error", {
                    "island_id": island.id,
                    "generation": gen_index,
                    "error": island.error,
                })
                return None

        # Acquire GPU from shared pool (blocks until one is free)
        gpu = gpu_queue.get()
        try:
            island.status = "training"
            push_event(event_queue, "island_status", {
                "island_id": island.id,
                "generation": gen_index,
                "status": "training",
                "message": f"Gen {gen_index} · C{island.id}: Training {max_iters} iters on GPU {gpu}{retry_label}...",
            })

            train_ok, train_err = _train_method(
                island, max_iters, data_path, output_dir,
                conda_env, gpu, event_queue, _config.viewer_port,
                _config.num_views,
            )
        finally:
            gpu_queue.put(gpu)  # always release GPU back to pool

        if not train_ok:
            if attempt < MAX_DEBUG_RETRIES:
                push_event(event_queue, "island_log", {
                    "island_id": island.id,
                    "generation": gen_index,
                    "message": f"Training failed, debugging{retry_label}...",
                })
                island.status = "debugging"
                push_event(event_queue, "island_status", {
                    "island_id": island.id,
                    "generation": gen_index,
                    "status": "debugging",
                })
                _debug_agent_in_thread(
                    island, "training", train_err,
                    _config, event_queue, model,
                )
                continue
            else:
                island.alive = False
                island.status = "error"
                island.error = f"Training failed: {train_err[:200]}"
                island.psnr = 0.0
                push_event(event_queue, "island_error", {
                    "island_id": island.id,
                    "generation": gen_index,
                    "error": island.error,
                })
                return None

        # ── Success: read PSNR ──
        psnr = _read_psnr_from_tb(output_dir, scene, island.method_name_cli)
        island.psnr = psnr or 0.0
        island.psnrs_by_round[gen_index] = island.psnr
        island.status = "trained"

        push_event(event_queue, "island_psnr", {
            "island_id": island.id,
            "psnr": island.psnr,
            "round": gen_index,
            "message": f"C{island.id} ({island.strategy['name']}): PSNR = {island.psnr:.2f} dB",
        })
        return saved  # done — return saved copy path for potential next-gen base

    return None  # exhausted all retries


# ═══════════════════════════════════════════════════════════════════
# DISCOVERER AGENT
# ═══════════════════════════════════════════════════════════════════

async def _run_discoverer_agent(
    island: Island,
    generation: int,
    current_best_psnr: float,
    baseline_psnr: float,
    history: str,
    config: PipelineConfig,
    event_queue: queue.Queue | None = None,
    model: str | None = None,
) -> bool:
    """Run a Claude discoverer agent to improve a candidate's code."""
    gs_root = str(config.gs_template_root)
    methods_dir = str(config.methods_dir)

    method_pkg = _find_method_pkg_dir(island.path) or "gs_method"

    num_competitors = 7 if generation == 1 else 7  # Other agents competing

    prompt = DISCOVERER_PROMPT.format(
        generation=generation,
        num_competitors=num_competitors,
        strategy_name=island.strategy["name"],
        strategy_description=island.strategy["description"],
        baseline_psnr=f"{baseline_psnr:.2f}" if baseline_psnr > 0 else "N/A (being measured)",
        current_best_psnr=f"{current_best_psnr:.2f}" if current_best_psnr > 0 else "N/A (being measured)",
        history=history,
        candidate_dir=str(island.path),
        method_pkg=method_pkg,
        gs_root=gs_root,
        methods_dir=methods_dir,
        method_name=island.method_name,
        method_name_cli=island.method_name_cli,
        method_name_var=island.method_name_var,
    )

    push_event(event_queue, "island_status", {
        "island_id": island.id,
        "status": "improving",
        "generation": generation,
        "strategy": island.strategy["name"],
        "message": f"Gen {generation} · Candidate {island.id} ({island.strategy['name']}): Agent improving...",
    })

    agent_opts = ClaudeAgentOptions(
        cwd=str(island.path),
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
        permission_mode="acceptEdits",
        max_turns=80,
        env=get_claude_auth_env(),
    )
    if model:
        agent_opts.model = model

    success = False
    try:
        # IMPORTANT: Do NOT return/break inside async for — that forcefully
        # closes the query() async generator, triggering anyio cancel-scope
        # cleanup in a different task and causing RuntimeError.
        # Instead, set a flag and let the generator finish naturally.
        async for message in query(
            prompt=prompt,
            options=agent_opts,
        ):
            if success:
                continue  # Already got ResultMessage; drain remaining messages

            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(event_queue, "island_log", {
                                "island_id": island.id,
                                "generation": generation,
                                "message": text[:200],
                            })
                    elif isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "?")
                        tool_input = getattr(block, "input", {})
                        summary = ""
                        for key in ("file_path", "command", "pattern", "query"):
                            if key in tool_input:
                                summary = str(tool_input[key])[:100]
                                break
                        push_event(event_queue, "island_tool", {
                            "island_id": island.id,
                            "generation": generation,
                            "tool": name,
                            "summary": summary,
                        })

            elif isinstance(message, ResultMessage):
                island.status = "improved"
                summary_file = island.path / "improvement_summary.txt"
                if summary_file.exists():
                    island.improvement_summary = summary_file.read_text()[:500]

                result_usage = getattr(message, "usage", None)
                if isinstance(result_usage, dict):
                    island.input_tokens += result_usage.get("input_tokens", 0) or 0
                    island.output_tokens += result_usage.get("output_tokens", 0) or 0
                cost = getattr(message, "total_cost_usd", None)

                push_event(event_queue, "island_tokens", {
                    "island_id": island.id,
                    "generation": generation,
                    "input_tokens": island.input_tokens,
                    "output_tokens": island.output_tokens,
                    "cost_usd": cost,
                })
                push_event(event_queue, "island_status", {
                    "island_id": island.id,
                    "status": "improved",
                    "generation": generation,
                    "summary": island.improvement_summary,
                    "message": f"Gen {generation} · Candidate {island.id} ({island.strategy['name']}): Improved",
                })
                success = True

    except Exception as e:
        island.error = str(e)
        island.status = "error"
        push_event(event_queue, "island_error", {
            "island_id": island.id,
            "generation": generation,
            "error": str(e),
            "message": f"Gen {generation} · Candidate {island.id} ({island.strategy['name']}): Error — {e}",
        })
        return False

    return success


# ═══════════════════════════════════════════════════════════════════
# MAIN DISCOVER PIPELINE
# ═══════════════════════════════════════════════════════════════════

async def run_evolution_entry(
    impl_path: str,
    workspace: Path,
    event_queue=None,
    stop_event: threading.Event | None = None,
    gpu: str = "0",
    data_path: str | None = None,
    scene: str = "garden",
    max_iters: int = 10000,
    model: str | None = None,
    num_views: int = -1,
    max_generations: int = 3,
    max_re_ideations: int = 2,
) -> dict:
    """Sub-project B entrypoint: wraps run_discover_pipeline with the
    full evolution stack (EVOLVE-BLOCK + EoH + Elo + verbal gradients
    + falsification + re-ideation feedback loop).

    This is the function web.py should call when the user wants the
    v2 pipeline. Parameters mirror run_discover_pipeline so it's a
    drop-in replacement.
    """
    # Local imports to avoid a hard cycle at module load time.
    from .evolution import run_evolution_pipeline
    from .b_agents import run_verbal_gradient_agent, run_negative_ideator_agent

    return await run_evolution_pipeline(
        impl_path=impl_path,
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
        max_re_ideations=max_re_ideations,
        run_verbal_gradient=run_verbal_gradient_agent,
        run_negative_ideator=run_negative_ideator_agent,
    )


async def run_discover_pipeline(
    impl_path: str,
    workspace: Path,
    event_queue: queue.Queue | None = None,
    stop_event: threading.Event | None = None,
    gpu: str = "0",
    data_path: str | None = None,
    scene: str = "garden",
    max_iters: int = 10000,
    model: str | None = None,
    num_views: int = -1,
    max_generations: int = 3,
    # ── Sub-project B hooks (optional; None = legacy behavior) ──
    # pre_generation_hook(gen_index, default_strategies) -> strategies
    #     Lets evolution.py replace the default STRATEGIES/refiner
    #     with an EoH-driven plan for each generation.
    # post_generation_hook_factory(gpu_queue) -> hook
    #     Called once per generation with the GPU queue; returns a
    #     callable hook(gen_index, gen_data, candidates, gen_workspace)
    #     that runs AFTER PSNR is known (Elo + verbal gradients +
    #     falsification). Return value is ignored.
    pre_generation_hook=None,
    post_generation_hook_factory=None,
) -> dict:
    """
    Run the iterative discovery pipeline.

    Each generation creates 8 candidate implementations, trains them,
    and the best PSNR winner becomes the base for the next generation.
    Runs for exactly max_generations (default 3). After each generation,
    an OPUS idea refiner agent analyzes results and generates 8 targeted
    strategies for the next generation. The global best across all 8n
    candidates is tracked.

    Args:
        impl_path: Path to existing generated implementation
        workspace: Working directory for this discover run
        event_queue: SSE event queue for UI updates
        stop_event: Threading event to signal graceful stop
        gpu: GPU device IDs, comma-separated
        data_path: Training data path
        scene: Scene name for training
        max_iters: Training iterations per candidate per generation
        model: Claude model preset (discoverer agents always use OPUS)
        num_views: Number of training views (-1 = all)
        max_generations: Number of generations to run (default 3)
    """
    config = PipelineConfig()
    config.num_views = num_views
    impl = Path(impl_path)
    gpus = [int(g.strip()) for g in gpu.split(",")]

    if not data_path:
        data_path = config.default_dataset

    base_snake, base_cli, base_var, base_pkg = _find_method_name_from_impl(impl)

    output_dir = workspace / "training_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    generations: list[DiscoverGeneration] = []
    current_best_path = impl      # Implementation to use as base for next gen
    current_best_psnr = 0.0
    global_best_psnr = 0.0        # Best PSNR across ALL candidates in ALL generations
    global_best_path: Path | None = None
    global_best_gen = 0
    global_best_strategy = ""
    baseline_psnr = 0.0
    gen_index = 0
    total_input_tokens = 0
    total_output_tokens = 0
    # Strategies for the current generation — starts with defaults, refined by OPUS after each gen
    current_strategies = STRATEGIES[:8]

    # Force OPUS for discoverer agents (idea generation)
    discoverer_model = "claude-opus-4-6"

    push_event(event_queue, "stage", {
        "stage": "setup",
        "message": (
            f"Discovery pipeline starting: method={base_cli}, "
            f"GPUs={gpus}, iters={max_iters}/gen, scene={scene}, "
            f"generations={max_generations}, discoverer=OPUS"
        ),
    })

    # ─── Main Generation Loop ────────────────────────────────────
    for gen_index in range(1, max_generations + 1):
        is_first_gen = (gen_index == 1)

        # Check stop event before starting new generation
        if stop_event and stop_event.is_set():
            push_event(event_queue, "stage", {
                "stage": "stopping",
                "message": "Stop requested — finishing up...",
            })
            break

        gen_workspace = workspace / f"gen_{gen_index}"
        gen_workspace.mkdir(parents=True, exist_ok=True)
        gen_output_dir = output_dir / f"gen_{gen_index}"
        gen_output_dir.mkdir(parents=True, exist_ok=True)

        # Determine strategies for this generation
        if is_first_gen:
            # 7 agent strategies + 1 baseline = 8 candidates
            agent_strategies = current_strategies[:7]
        else:
            # All 8 strategies (refined by OPUS idea refiner from previous gen)
            agent_strategies = current_strategies[:8]

        # Sub-project B: let the evolution orchestrator replace the
        # default strategies with an EoH-driven plan (with parents +
        # lessons). If pre_generation_hook is None, keep legacy behavior.
        if pre_generation_hook is not None:
            try:
                agent_strategies = pre_generation_hook(gen_index, agent_strategies)
            except Exception as _hook_err:
                push_event(event_queue, "island_log", {
                    "island_id": -1,
                    "generation": gen_index,
                    "message": f"pre_generation_hook failed: {_hook_err}",
                })

        gen_data = DiscoverGeneration(
            index=gen_index,
            input_psnr=current_best_psnr,
        )

        push_event(event_queue, "gen_start", {
            "generation": gen_index,
            "num_candidates": len(agent_strategies) + (1 if is_first_gen else 0),
            "current_best_psnr": current_best_psnr,
            "base_method": base_cli,
            "message": (
                f"Generation {gen_index}: "
                + (f"Baseline + {len(agent_strategies)} variants" if is_first_gen
                   else f"8 variants (base PSNR: {current_best_psnr:.2f} dB)")
            ),
        })

        # ─── Create candidates ───────────────────────────────────
        candidates: list[Island] = []
        agents_to_run: list[Island] = []

        if is_first_gen:
            # Candidate 0: baseline (unmodified copy of original)
            baseline_dir = gen_workspace / "island_0"
            _copy_implementation(impl, baseline_dir, base_pkg)

            baseline_island = Island(
                id=0,
                strategy={"id": "baseline", "name": "Baseline", "emoji": "B",
                          "description": "Unmodified baseline implementation"},
                path=baseline_dir,
                method_name=f"{base_snake}_island_0",
                method_name_cli=f"{base_cli}-island-0",
                method_name_var=f"{base_var}_island_0",
                train_output_dir=gen_output_dir,
            )
            # Rename entry points so baseline can coexist with other islands
            _rename_baseline_method(
                baseline_dir, base_pkg,
                old_cli=base_cli, new_cli=baseline_island.method_name_cli,
                old_var=base_var, new_var=baseline_island.method_name_var,
            )
            candidates.append(baseline_island)

            push_event(event_queue, "island_created", {
                "island_id": 0,
                "generation": gen_index,
                "strategy": "Baseline",
                "strategy_id": "baseline",
                "method_name": baseline_island.method_name_cli,
                "message": f"Gen {gen_index} · Candidate 0: Baseline (unmodified)",
            })

            # Candidates 1-7: agent-modified variants
            for i, strategy in enumerate(agent_strategies):
                cand_id = i + 1
                island_dir = gen_workspace / f"island_{cand_id}"
                _copy_implementation(impl, island_dir, base_pkg)

                island = Island(
                    id=cand_id,
                    strategy=strategy,
                    path=island_dir,
                    method_name=f"{base_snake}_island_{cand_id}",
                    method_name_cli=f"{base_cli}-island-{cand_id}",
                    method_name_var=f"{base_var}_island_{cand_id}",
                    train_output_dir=gen_output_dir,
                )
                candidates.append(island)
                agents_to_run.append(island)

                push_event(event_queue, "island_created", {
                    "island_id": cand_id,
                    "generation": gen_index,
                    "strategy": strategy["name"],
                    "strategy_id": strategy["id"],
                    "method_name": island.method_name_cli,
                    "message": f"Gen {gen_index} · Candidate {cand_id}: {strategy['name']}",
                })

        else:
            # Gen 2+: all 8 candidates modified by agents
            for i, strategy in enumerate(agent_strategies):
                island_dir = gen_workspace / f"island_{i}"
                _copy_implementation(current_best_path, island_dir, base_pkg)

                island = Island(
                    id=i,
                    strategy=strategy,
                    path=island_dir,
                    method_name=f"{base_snake}_island_{i}",
                    method_name_cli=f"{base_cli}-island-{i}",
                    method_name_var=f"{base_var}_island_{i}",
                    train_output_dir=gen_output_dir,
                )
                candidates.append(island)
                agents_to_run.append(island)

                push_event(event_queue, "island_created", {
                    "island_id": i,
                    "generation": gen_index,
                    "strategy": strategy["name"],
                    "strategy_id": strategy["id"],
                    "method_name": island.method_name_cli,
                    "message": f"Gen {gen_index} · Candidate {i}: {strategy['name']}",
                })

        # ─── Uninstall old packages before this generation ────────
        _uninstall_all_islands(config.conda_env, base_pkg=base_pkg)

        # ─── Build history for agent prompts ──────────────────────
        history = _build_history(generations, baseline_psnr) if agents_to_run else ""

        # ─── GPU pool: training slots shared across candidates ────
        gpu_queue: queue.Queue[int] = queue.Queue()
        for g in gpus:
            gpu_queue.put(g)

        push_event(event_queue, "stage", {
            "stage": "running",
            "generation": gen_index,
            "message": (
                f"Gen {gen_index}: Launching {len(candidates)} candidates "
                f"(each: agent → install → train {max_iters} iters) on {len(gpus)} GPU(s)..."
            ),
        })

        # ─── Run all candidates fully in parallel ─────────────────
        # Each candidate independently: agent → save → uniquify → install → train
        # Pip installs are serialized via _pip_lock; GPUs are allocated via gpu_queue.
        saved_copies: dict[int, Path] = {}

        def _run_one(cand, needs_agent):
            """Full pipeline for one candidate in its own thread."""
            return _run_candidate_full_pipeline(
                island=cand,
                needs_agent=needs_agent,
                gen_index=gen_index,
                current_best_psnr=current_best_psnr,
                baseline_psnr=baseline_psnr,
                history=history,
                base_pkg=base_pkg,
                gpu_queue=gpu_queue,
                max_iters=max_iters,
                data_path=data_path,
                output_dir=gen_output_dir,
                conda_env=config.conda_env,
                scene=scene,
                gen_workspace=gen_workspace,
                event_queue=event_queue,
                config=config,
                model=discoverer_model,
            )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            future_map = {}
            for cand in candidates:
                needs_agent = cand in agents_to_run
                fut = loop.run_in_executor(pool, _run_one, cand, needs_agent)
                future_map[fut] = cand

            results = await asyncio.gather(*future_map.keys(), return_exceptions=True)

        for fut_result, cand in zip(results, future_map.values()):
            if isinstance(fut_result, Exception):
                cand.alive = False
                cand.status = "error"
                cand.error = str(fut_result)
                push_event(event_queue, "island_error", {
                    "island_id": cand.id,
                    "generation": gen_index,
                    "error": str(fut_result),
                    "message": f"Gen {gen_index} · C{cand.id}: Pipeline failed — {fut_result}",
                })
            elif isinstance(fut_result, Path):
                saved_copies[cand.id] = fut_result

        # ─── Find best PSNR ─────────────────────────────────────
        alive_candidates = [c for c in candidates if c.alive and c.psnr > 0]

        if alive_candidates:
            best = max(alive_candidates, key=lambda c: c.psnr)

            if is_first_gen:
                # Record baseline PSNR
                baseline_cand = next(
                    (c for c in candidates if c.strategy["id"] == "baseline" and c.psnr > 0),
                    None,
                )
                if baseline_cand:
                    baseline_psnr = baseline_cand.psnr

            gen_data.best_id = best.id
            gen_data.best_psnr = best.psnr
            gen_data.best_strategy = best.strategy["name"]
            gen_data.best_summary = best.improvement_summary

            # Update current best if improved (or if first gen)
            if best.psnr > current_best_psnr or current_best_psnr == 0:
                current_best_psnr = best.psnr
                # Use saved pre-uniquification copy as next gen's base
                if best.id in saved_copies:
                    current_best_path = saved_copies[best.id]

            # Track global best across ALL candidates in ALL generations
            if best.psnr > global_best_psnr:
                global_best_psnr = best.psnr
                global_best_gen = gen_index
                global_best_strategy = best.strategy["name"]
                if best.id in saved_copies:
                    global_best_path = saved_copies[best.id]

            gen_data.candidates = [_island_summary(c) for c in candidates]
            generations.append(gen_data)

            delta = best.psnr - gen_data.input_psnr if gen_data.input_psnr > 0 else 0
            delta_str = f" ({'+' if delta >= 0 else ''}{delta:.2f})" if gen_data.input_psnr > 0 else ""

            total_candidates = sum(len(g.candidates) for g in generations)
            push_event(event_queue, "gen_result", {
                "generation": gen_index,
                "best_id": best.id,
                "best_psnr": best.psnr,
                "best_strategy": best.strategy["name"],
                "best_summary": best.improvement_summary,
                "baseline_psnr": baseline_psnr,
                "current_best_psnr": current_best_psnr,
                "global_best_psnr": global_best_psnr,
                "global_best_gen": global_best_gen,
                "global_best_strategy": global_best_strategy,
                "total_candidates": total_candidates,
                "delta": delta_str,
                "candidates": [
                    {
                        "id": c.id, "psnr": c.psnr, "strategy": c.strategy["name"],
                        "alive": c.alive, "status": c.status,
                        "summary": c.improvement_summary[:100] if c.improvement_summary else "",
                    }
                    for c in candidates
                ],
                "message": (
                    f"Gen {gen_index}/{max_generations}: Best = {best.strategy['name']} "
                    f"({best.psnr:.2f} dB{delta_str}) | "
                    f"Global best: {global_best_psnr:.2f} dB [{total_candidates} total runs]"
                ),
            })

            push_event(event_queue, "new_best", {
                "generation": gen_index,
                "psnr": current_best_psnr,
                "global_best_psnr": global_best_psnr,
                "baseline_psnr": baseline_psnr,
                "improvement": global_best_psnr - baseline_psnr if baseline_psnr > 0 else 0,
                "strategy": best.strategy["name"],
                "total_candidates": total_candidates,
                "message": (
                    f"Global best: {global_best_psnr:.2f} dB"
                    + (f" (+{global_best_psnr - baseline_psnr:.2f} over baseline)"
                       if baseline_psnr > 0 else "")
                    + f" | {total_candidates}/{8 * max_generations} runs complete"
                ),
            })

        else:
            # All candidates failed training
            gen_data.candidates = [_island_summary(c) for c in candidates]
            generations.append(gen_data)

            push_event(event_queue, "gen_result", {
                "generation": gen_index,
                "best_id": None,
                "best_psnr": 0,
                "message": f"Gen {gen_index}: All candidates failed training — continuing with previous best",
            })

        # Sub-project B: post-generation hook (Elo + gradients + falsify).
        # Runs AFTER gen_result but BEFORE the next idea refiner call,
        # so falsification penalties can influence the next generation.
        if post_generation_hook_factory is not None:
            try:
                _post_hook = post_generation_hook_factory(gpu_queue)
                _post_hook(gen_index, gen_data, candidates, gen_workspace)
            except Exception as _post_err:
                push_event(event_queue, "island_log", {
                    "island_id": -1,
                    "generation": gen_index,
                    "message": f"post_generation_hook failed: {_post_err}",
                })

        # Update token totals
        gen_in = sum(c.input_tokens for c in candidates)
        gen_out = sum(c.output_tokens for c in candidates)
        total_input_tokens += gen_in
        total_output_tokens += gen_out

        # Cost estimate from token counts. Discover's mutator agents
        # run on Opus (forced), so price at Opus rates.
        try:
            from config import estimate_cost as _est
            _cost = _est(total_input_tokens, total_output_tokens, "claude-opus-4-6")
        except Exception:
            _cost = 0.0
        push_event(event_queue, "tokens_update", {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "cost_usd": round(_cost, 4),
            "generation": gen_index,
        })

        # Clean up installed packages for next generation
        _uninstall_all_islands(config.conda_env, base_pkg=base_pkg)

        # Check stop event after generation completes
        if stop_event and stop_event.is_set():
            push_event(event_queue, "stage", {
                "stage": "stopping",
                "message": "Stop requested after generation complete — wrapping up...",
            })
            break

        # ─── OPUS Idea Refiner: generate strategies for next gen ─
        # Skip when pre_generation_hook owns strategy generation.
        if gen_index < max_generations and pre_generation_hook is None:
            history = _build_history(generations, baseline_psnr)
            gen_best_psnr = gen_data.best_psnr if gen_data.best_psnr > 0 else current_best_psnr
            current_strategies = await _run_idea_refiner(
                workspace=workspace,
                gen_index=gen_index,
                baseline_psnr=baseline_psnr,
                global_best_psnr=global_best_psnr,
                gen_best_psnr=gen_best_psnr,
                history=history,
                best_impl_path=current_best_path,
                base_pkg=base_pkg,
                config=config,
                event_queue=event_queue,
            )

    # ─── Compile Final Results ───────────────────────────────────
    improvement = global_best_psnr - baseline_psnr if baseline_psnr > 0 else 0

    # Build evolution path
    evolution_path = []
    if baseline_psnr > 0:
        evolution_path.append({
            "label": "Baseline",
            "psnr": baseline_psnr,
            "generation": 0,
        })
    for gen in generations:
        if gen.best_psnr > 0:
            evolution_path.append({
                "label": f"Gen {gen.index}: {gen.best_strategy}",
                "psnr": gen.best_psnr,
                "generation": gen.index,
                "summary": gen.best_summary[:200] if gen.best_summary else "",
            })

    total_candidates = sum(len(g.candidates) for g in generations)
    final_result = {
        "baseline_psnr": baseline_psnr,
        "final_best_psnr": global_best_psnr,
        "improvement": round(improvement, 2),
        "num_generations": len(generations),
        "total_candidates": total_candidates,
        "global_best_gen": global_best_gen,
        "global_best_strategy": global_best_strategy,
        "evolution_path": evolution_path,
        "generations": [
            {
                "index": g.index,
                "input_psnr": g.input_psnr,
                "best_id": g.best_id,
                "best_psnr": g.best_psnr,
                "best_strategy": g.best_strategy,
                "best_summary": g.best_summary,
                "candidates": g.candidates,
            }
            for g in generations
        ],
        "best_implementation_path": str(global_best_path or current_best_path),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }

    # Save results
    results_file = workspace / "discover_results.json"
    results_file.write_text(json.dumps(final_result, indent=2, default=str))

    push_event(event_queue, "discover_done", {
        "message": (
            f"Discovery complete! {len(generations)} generations, {total_candidates} total runs. "
            f"Baseline: {baseline_psnr:.2f} → Global best: {global_best_psnr:.2f} dB "
            f"(+{improvement:.2f} dB) from gen {global_best_gen} ({global_best_strategy})"
        ),
        "result": final_result,
    })

    push_event(event_queue, "done", {
        "message": (
            f"Discovery complete! {len(generations)} generations, {total_candidates} total runs. "
            f"Baseline: {baseline_psnr:.2f} → Global best: {global_best_psnr:.2f} dB "
            f"(+{improvement:.2f})"
        ),
        "result": final_result,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    })

    return final_result
