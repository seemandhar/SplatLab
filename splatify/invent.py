#!/usr/bin/env python3
"""
Evolutionary Tournament Pipeline for Gaussian Splatting Method Improvement.

Spawns 8 "island" agents, each given a different improvement strategy.
They independently modify a base implementation, then compete through
training rounds with PSNR-based elimination:

  Round 1: 8 islands -> train 2k iters -> drop 4 lowest PSNR -> 4 survive
  Round 2: 4 islands -> train 7k iters -> drop 2 lowest PSNR -> 2 survive
  Round 3: 2 islands -> train 30k iters -> drop 1 lowest PSNR -> 1 winner
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import os
import queue
import re
import shutil
import subprocess
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

# ═══════════════════════════════════════════════════════════════════
# IMPROVEMENT STRATEGIES
# ═══════════════════════════════════════════════════════════════════

STRATEGIES = [
    {
        "id": "loss_engineering",
        "name": "Loss Engineering",
        "emoji": "L",
        "description": (
            "Improve quality by adding or refining loss functions. "
            "Consider adding perceptual losses (LPIPS/VGG), frequency-domain losses, "
            "edge-aware losses, or multi-scale supervision. Look at gs_template/losses.py "
            "for reusable losses. Tune loss weights for optimal PSNR."
        ),
    },
    {
        "id": "densification",
        "name": "Densification Strategy",
        "emoji": "D",
        "description": (
            "Improve quality by optimizing the Gaussian densification/pruning strategy. "
            "Consider adjusting split/clone thresholds, adding opacity-based pruning, "
            "scale-based pruning, gradient-based densification triggers, or progressive "
            "densification schedules. Focus on placing Gaussians where they matter most."
        ),
    },
    {
        "id": "lr_schedule",
        "name": "Learning Rate & Optimization",
        "emoji": "O",
        "description": (
            "Improve quality by tuning optimizer hyperparameters and learning rate schedules. "
            "Consider warmup phases, cosine annealing, per-parameter LR tuning, "
            "different LR for position vs appearance, exponential decay schedules. "
            "Reference the paper's reported hyperparameters and recent best practices."
        ),
    },
    {
        "id": "rendering",
        "name": "Rendering Enhancement",
        "emoji": "R",
        "description": (
            "Improve quality by enhancing the rendering pipeline. Consider anti-aliasing "
            "techniques, better opacity handling, improved color representation (higher SH "
            "degree), view-dependent effects, near/far plane handling, or background modeling. "
            "Use gs_template/rendering.py helpers and gs_template/shading.py utilities."
        ),
    },
    {
        "id": "new_params",
        "name": "New Gaussian Parameters",
        "emoji": "P",
        "description": (
            "Improve quality by adding new learnable per-Gaussian attributes. Consider "
            "per-Gaussian feature vectors, learned normal vectors, anisotropy parameters, "
            "or material properties. Each new parameter needs: populate_modules(), "
            "split_gaussians(), get_gaussian_param_groups(), and optimizer config entries."
        ),
    },
    {
        "id": "regularization",
        "name": "Regularization",
        "emoji": "G",
        "description": (
            "Improve quality by adding regularization terms. Consider opacity entropy "
            "regularization, scale regularization (flatten or limit), normal consistency, "
            "depth smoothness, distortion loss, total variation, or compact/sparse "
            "representations. Check gs_template/losses.py for available regularizers."
        ),
    },
    {
        "id": "training_schedule",
        "name": "Training Schedule",
        "emoji": "T",
        "description": (
            "Improve quality by optimizing the training schedule. Consider progressive "
            "resolution training, coarse-to-fine strategies, curriculum learning, "
            "delayed loss activation, warm-up periods for certain losses, or "
            "step-dependent hyperparameter annealing. Focus on training stability."
        ),
    },
    {
        "id": "web_inspired",
        "name": "Web-Inspired Innovation",
        "emoji": "W",
        "description": (
            "Search the web for the latest Gaussian Splatting improvements, tricks, and "
            "best practices published in 2024-2025. Look for techniques from recent papers "
            "like Mini-Splatting, Pixel-GS, GOF, AbsGS, Spec-Gaussian, etc. Apply the most "
            "promising technique that is compatible with this implementation."
        ),
    },
]

# ═══════════════════════════════════════════════════════════════════
# TOURNAMENT CONFIG
# ═══════════════════════════════════════════════════════════════════

TOURNAMENT_ROUNDS = [
    {"max_iters": 2000, "survivors": 4, "label": "Round 1 (2k iters)"},
    {"max_iters": 7000, "survivors": 2, "label": "Round 2 (7k iters)"},
    {"max_iters": 30000, "survivors": 1, "label": "Round 3 (30k iters)"},
]

MAX_DEBUG_RETRIES = 4  # retries per island per training round


# ═══════════════════════════════════════════════════════════════════
# INVENTOR AGENT PROMPT
# ═══════════════════════════════════════════════════════════════════

INVENTOR_PROMPT = r"""You are an Inventor Agent for the Splatify evolutionary pipeline.
You are competing against 7 other agents to produce the BEST Gaussian Splatting
implementation measured by PSNR on the garden scene.

## Your Improvement Strategy: {strategy_name}
{strategy_description}

## Your Task
1. Read the existing implementation at: {island_dir}
   - Read ALL files in {method_pkg}/ (model.py, configs.py, losses.py, rendering.py, datamanager.py, __init__.py)
   - Read pyproject.toml
2. Read the research paper at: {paper_path}
3. Read the gs_template framework for reference:
   - {gs_root}/model.py — Base class with extension points
   - {gs_root}/rendering.py — Rendering helpers
   - {gs_root}/losses.py — 30+ reusable loss functions
   - {gs_root}/shading.py — SH, normals, cubemaps
   - {gs_root}/configs.py — MethodSpecification examples
4. Read 2-3 example methods from {methods_dir}/ for inspiration
5. {web_search_instruction}
6. Apply your improvement strategy to enhance the method's PSNR quality

## Critical Rules
- You MUST rename the method to "{method_name}" in ALL files:
  - {method_pkg}/configs.py: method_name="{method_name_cli}"
  - pyproject.toml: entry point must be "{method_name_cli}" = "{method_pkg}.configs:{method_name_var}"
  - {method_pkg}/__init__.py: update variable names if needed
- All internal imports must use "{method_pkg}" as the package name (e.g., `from {method_pkg}.model import ...`)
- Keep the method installable: `pip install -e .` must work
- Keep the method trainable: `ns-train {method_name_cli} --data <path> --max-num-iterations 2000` must work
- Do NOT break existing functionality — enhance it
- Write complete files (no patches), use the Write tool
- Focus on changes that MAXIMIZE PSNR — you are being judged on this metric
- Be bold but practical — untested theoretical changes may crash training
- Do NOT run pip install or ns-train yourself — the tournament pipeline handles that

## Output
After making changes, write a brief summary to {island_dir}/improvement_summary.txt explaining:
- What you changed and why
- Expected impact on PSNR
- Any risks or tradeoffs
"""


# ═══════════════════════════════════════════════════════════════════
# DEBUGGER AGENT PROMPT
# ═══════════════════════════════════════════════════════════════════

DEBUGGER_PROMPT_INVENT = r"""You are a Debugger Agent for the Splatify evolutionary pipeline.
Your job is to diagnose and fix errors in a Gaussian Splatting method implementation.

## Error Phase: {phase}
## Error Output:
```
{error_output}
```

## Implementation Directory: {island_dir}
## Package Directory: {pkg_name}/

## Your Task
1. Read the error carefully and identify the root cause
2. Read the relevant source files in {island_dir}/{pkg_name}/
3. Read pyproject.toml and setup.cfg for package configuration
4. Fix the code to resolve the error
5. Write corrected files using the Write tool

## Critical Rules
- Do NOT change the method name ({method_name_cli}) or entry points
- Write complete files, not patches
- Focus ONLY on fixing the error — do not add new features or improvements
- Do NOT run pip install or ns-train — the pipeline handles that
- If the error is in an import, check that all dependencies are in pyproject.toml
- If it's a shape/dimension error, trace the tensor shapes through the code
- If it's a missing attribute, check the base class API (SplatfactoModel / GSTemplateModel)

## Common Error Categories
1. **Import errors**: Missing module, wrong import path (should use {pkg_name}.xxx not gs_method.xxx)
2. **Type/shape errors**: Tensor dimension mismatches, wrong dtypes
3. **Config errors**: Missing fields in ModelConfig or MethodSpecification
4. **Runtime errors**: Missing self.xys/self.radii in get_outputs(), wrong param groups
5. **Densification errors**: New params not handled in split_gaussians()

## gs_template reference
The base framework is at: {gs_root}
Read {gs_root}/model.py for the base class API if needed.
"""


# ═══════════════════════════════════════════════════════════════════
# ISLAND DATACLASS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Island:
    id: int
    strategy: dict
    path: Path
    method_name: str          # snake_case: original_method_island_0
    method_name_cli: str      # kebab-case: original-method-island-0
    method_name_var: str      # variable name for configs.py
    pkg_name: str = ""        # unique package dir: gs_method_island_0
    psnr: float = 0.0
    psnrs_by_round: dict = field(default_factory=dict)
    alive: bool = True
    improvement_summary: str = ""
    status: str = "pending"   # pending, improving, debugging, training, eliminated, winner
    error: str = ""
    train_output_dir: Path = None
    input_tokens: int = 0
    output_tokens: int = 0


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def push_event(q: queue.Queue | None, event_type: str, data: dict):
    """Push event to SSE queue if available."""
    if q:
        q.put({"type": event_type, **data})


def _find_method_pkg_dir(impl_path: Path) -> str | None:
    """Find the method package directory name in an implementation.

    The package dir is named after the method (e.g., "dashgaussian/"),
    or might be the legacy "gs_method/" for older generations.
    """
    # Look for any directory with __init__.py (excluding __pycache__, .egg-info)
    for d in sorted(impl_path.iterdir()):
        if (d.is_dir() and (d / "__init__.py").exists()
                and not d.name.startswith("__") and not d.name.endswith(".egg-info")):
            return d.name
    return None


def _find_method_name_from_impl(impl_path: Path) -> tuple[str, str, str, str]:
    """Extract method name and package dir from an existing implementation.

    Returns (snake_name, cli_name, var_name, pkg_dir_name).
    """
    pkg_dir = _find_method_pkg_dir(impl_path)

    pyproject = impl_path / "pyproject.toml"
    if pyproject.exists():
        text = pyproject.read_text()
        # Match entry points like: dashgaussian = "dashgaussian.configs:dashgaussian_method"
        # or legacy: dashgaussian = "gs_method.configs:dashgaussian_method"
        m = re.search(r'^([a-z0-9_-]+)\s*=\s*"(\w+)\.configs:(\w+)"', text, re.MULTILINE)
        if m:
            cli_name = m.group(1)
            entry_pkg = m.group(2)
            var_name = m.group(3)
            snake_name = cli_name.replace("-", "_")
            # Use the actual directory name if found, otherwise fall back to entry point pkg
            actual_pkg = pkg_dir or entry_pkg
            return snake_name, cli_name, var_name, actual_pkg

    # Try finding configs.py in the package directory
    if pkg_dir:
        configs_py = impl_path / pkg_dir / "configs.py"
        if configs_py.exists():
            text = configs_py.read_text()
            m = re.search(r'method_name\s*=\s*["\']([^"\']+)["\']', text)
            if m:
                cli_name = m.group(1)
                snake_name = cli_name.replace("-", "_")
                m2 = re.search(r'^(\w+)\s*=\s*MethodSpecification', text, re.MULTILINE)
                var_name = m2.group(1) if m2 else snake_name + "_method"
                return snake_name, cli_name, var_name, pkg_dir

    return "method", "method", "method_spec", pkg_dir or "gs_method"


def _find_paper_path(impl_path: Path, workspace_dir: Path) -> str | None:
    """Try to find the cleaned paper markdown for this implementation."""
    run_id = impl_path.name
    workspace_run = workspace_dir / run_id
    cleaned = workspace_run / "cleaned_paper.md"
    if cleaned.exists():
        return str(cleaned)
    raw = workspace_run / "raw_paper.md"
    if raw.exists():
        return str(raw)
    return None


def _run_shell(cmd: str, cwd: str | None = None, env: dict | None = None,
               timeout: int = 300) -> tuple[bool, str, str]:
    """Run a shell command, return (success, stdout, stderr)."""
    result = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True, text=True,
        timeout=timeout,
        cwd=cwd,
        env=env or os.environ.copy(),
    )
    return result.returncode == 0, result.stdout, result.stderr


def _read_psnr_from_tb(output_dir: Path, scene: str, method_cli: str) -> float | None:
    """Read PSNR from TensorBoard logs after training, with a per-run memo.

    The first successful read is cached to `<run_dir>/metrics_cached.json`.
    Subsequent calls short-circuit on that file, which avoids re-scanning
    TensorBoard event files across invent.py + discover.py + pipeline ingester.
    """
    scene_method_dir = output_dir / scene / method_cli
    if not scene_method_dir.exists():
        return None

    runs = sorted(scene_method_dir.iterdir(), key=lambda p: p.name, reverse=True)
    if not runs:
        return None

    run_dir = runs[0]

    # Memoised fast path.
    try:
        from splatify.cache import read_cached_psnr, write_cached_psnr
    except ImportError:  # pragma: no cover
        read_cached_psnr = write_cached_psnr = None  # type: ignore
    if read_cached_psnr is not None:
        cached = read_cached_psnr(run_dir)
        if cached is not None:
            return cached

    # Try read_tb.py
    psnr: float | None = None
    try:
        config = PipelineConfig()
        ok, stdout, _ = _run_shell(
            f'python "{config.read_tb_script}" "{run_dir}" --json',
            timeout=30,
        )
        if ok and stdout.strip():
            data = json.loads(stdout)
            psnr = data.get("final_psnr")
    except Exception:
        pass

    # Fallback: read eval JSON files
    if psnr is None:
        for json_file in run_dir.rglob("*.json"):
            try:
                data = json.loads(json_file.read_text())
                if "results" in data and "psnr" in data["results"]:
                    psnr = data["results"]["psnr"]; break
                if "psnr" in data:
                    psnr = data["psnr"]; break
            except Exception:
                continue

    if psnr is not None and write_cached_psnr is not None:
        write_cached_psnr(run_dir, psnr, scene=scene, method_cli=method_cli)
    return psnr


# ═══════════════════════════════════════════════════════════════════
# PACKAGE UNIQUIFICATION
# ═══════════════════════════════════════════════════════════════════

def _uniquify_island_package(island: Island, base_pkg: str) -> bool:
    """
    Rename <base_pkg>/ -> <base_pkg>_island_N/ and rewrite all internal imports.

    This is critical: all 8 islands share the same Python module name.
    Without renaming, pip install of one island overwrites the others.
    After renaming, all 8 can be installed simultaneously, enabling parallel training.

    Args:
        island: The island to uniquify.
        base_pkg: The original package directory name (e.g., "dashgaussian" or "gs_method").
    """
    src = island.path / base_pkg
    pkg = f"{base_pkg}_island_{island.id}"
    dst = island.path / pkg
    island.pkg_name = pkg

    if not src.exists():
        return False

    # Remove destination if it exists (re-run safety)
    if dst.exists():
        shutil.rmtree(dst)

    # Rename directory
    shutil.copytree(src, dst)
    shutil.rmtree(src)

    # Rewrite all Python files: base_pkg -> base_pkg_island_N
    for py_file in dst.glob("*.py"):
        content = py_file.read_text()
        content = content.replace(f"from {base_pkg}.", f"from {pkg}.")
        content = content.replace(f"from {base_pkg} import", f"from {pkg} import")
        content = content.replace(f"import {base_pkg}.", f"import {pkg}.")
        content = content.replace(f"import {base_pkg}", f"import {pkg}")
        content = content.replace(f'"{base_pkg}.', f'"{pkg}.')
        py_file.write_text(content)

    # Rewrite pyproject.toml
    pyproject = island.path / "pyproject.toml"
    if pyproject.exists():
        content = pyproject.read_text()
        content = content.replace(f'"{base_pkg}.', f'"{pkg}.')
        content = content.replace(f"{base_pkg}.", f"{pkg}.")
        # Also fix the package name to be unique
        content = re.sub(
            r'^name\s*=\s*"[^"]+"',
            f'name = "{island.method_name_cli}"',
            content,
            flags=re.MULTILINE,
        )
        pyproject.write_text(content)

    # Create a setup.cfg or update pyproject to find the renamed package
    # setuptools needs to know about the new package directory name
    setup_cfg = island.path / "setup.cfg"
    setup_cfg.write_text(
        f"[options]\n"
        f"packages = {pkg}\n"
    )

    return True


# ═══════════════════════════════════════════════════════════════════
# INSTALL / TRAIN / CLEANUP
# ═══════════════════════════════════════════════════════════════════

def _install_method(island: Island, conda_env: str,
                    event_queue: queue.Queue | None = None) -> tuple[bool, str]:
    """pip install an island's method. Returns (success, error_message)."""
    cmd = (
        f'eval "$(conda shell.bash hook 2>/dev/null)" && conda activate {conda_env} && '
        f'cd "{island.path}" && pip install -e . 2>&1 && ns-install-cli 2>&1'
    )
    try:
        ok, stdout, stderr = _run_shell(cmd, timeout=180)
        output = stdout + "\n" + stderr
        if not ok:
            # Extract the last meaningful error
            lines = [l for l in output.splitlines() if l.strip()]
            error_tail = "\n".join(lines[-10:]) if lines else "Unknown error"
            push_event(event_queue, "island_log", {
                "island_id": island.id,
                "message": f"Install error:\n{error_tail[-300:]}",
            })
            return False, error_tail
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Install timed out (180s)"
    except Exception as e:
        return False, str(e)


def _train_method(
    island: Island,
    max_iters: int,
    data_path: str,
    output_dir: Path,
    conda_env: str,
    gpu: int = 0,
    event_queue: queue.Queue | None = None,
    viewer_port: int | None = None,
    num_views: int = -1,
) -> tuple[bool, str]:
    """Train an island's method. Returns (success, error_message)."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CC"] = "/usr/bin/gcc-11"
    env["CXX"] = "/usr/bin/g++-11"
    env["CUDAHOSTCXX"] = "/usr/bin/g++-11"

    vis_mode = "viewer+tensorboard" if viewer_port else "tensorboard"
    viewer_flag = f"--viewer.websocket-port {viewer_port} " if viewer_port else ""
    views_flag = f"--num-training-views {num_views} " if num_views > 0 else ""

    cmd = (
        f'eval "$(conda shell.bash hook 2>/dev/null)" && conda activate {conda_env} && '
        f'ns-train {island.method_name_cli} '
        f'--data "{data_path}" '
        f'--vis {vis_mode} '
        f'--steps-per-eval-batch 10000 '
        f'--max-num-iterations {max_iters} '
        f'--output-dir "{output_dir}" '
        f'--viewer.quit-on-train-completion True '
        f'{viewer_flag}'
        f'colmap --eval-mode interval '
        f'{views_flag}'
    )

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True,
            timeout=7200,
            env=env,
        )
        if result.returncode != 0:
            output = result.stdout + "\n" + result.stderr
            lines = [l for l in output.splitlines() if l.strip()]
            error_tail = "\n".join(lines[-15:]) if lines else "Unknown training error"
            push_event(event_queue, "island_log", {
                "island_id": island.id,
                "message": f"Train error:\n{error_tail[-400:]}",
            })
            return False, error_tail
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Training timed out (2h)"
    except Exception as e:
        return False, str(e)


def _uninstall_all_islands(conda_env: str, base_pkg: str = "", num_islands: int = 8):
    """Uninstall all island packages to clean the environment."""
    # Collect all possible package names
    pkgs = []
    base_kebab = base_pkg.replace("_", "-")
    for i in range(num_islands):
        if base_pkg:
            pkgs.append(f"{base_kebab}-island-{i}")
            pkgs.append(f"{base_pkg}_island_{i}")
        # Also handle legacy gs_method naming
        pkgs.append(f"gs-method-island-{i}")
        pkgs.append(f"gs_method_island_{i}")
    # Also uninstall any leftover base package
    if base_pkg:
        pkgs.append(base_kebab)
    pkgs.append("gs-method")

    pkg_list = " ".join(set(pkgs))  # deduplicate
    cmd = (
        f'eval "$(conda shell.bash hook 2>/dev/null)" && conda activate {conda_env} && '
        f'pip uninstall -y {pkg_list} 2>/dev/null; true'
    )
    _run_shell(cmd, timeout=60)


# ═══════════════════════════════════════════════════════════════════
# INVENTOR AGENT
# ═══════════════════════════════════════════════════════════════════

async def _run_inventor_agent(
    island: Island,
    paper_path: str | None,
    arxiv: str | None,
    config: PipelineConfig,
    event_queue: queue.Queue | None = None,
    model: str | None = None,
) -> bool:
    """Run a Claude inventor agent to improve an island's code."""
    gs_root = str(config.gs_template_root)
    methods_dir = str(config.methods_dir)

    if paper_path:
        paper_instruction = f"Read the paper at: {paper_path}"
    elif arxiv:
        paper_instruction = (
            f"Download and read the paper from arXiv: {arxiv}. "
            f"Use WebFetch to get it or Bash to wget the PDF."
        )
    else:
        paper_instruction = "No paper available — focus on the existing code and web research."

    web_instruction = (
        "Use WebSearch to find recent improvements and best practices for Gaussian Splatting"
        if island.strategy["id"] == "web_inspired"
        else "Optionally use WebSearch if you need to look up specific techniques"
    )

    # Find the method package directory in this island
    method_pkg = _find_method_pkg_dir(island.path) or "gs_method"

    prompt = INVENTOR_PROMPT.format(
        strategy_name=island.strategy["name"],
        strategy_description=island.strategy["description"],
        island_dir=str(island.path),
        paper_path=paper_instruction,
        gs_root=gs_root,
        methods_dir=methods_dir,
        method_name=island.method_name,
        method_name_cli=island.method_name_cli,
        method_name_var=island.method_name_var,
        method_pkg=method_pkg,
        web_search_instruction=web_instruction,
    )

    push_event(event_queue, "island_status", {
        "island_id": island.id,
        "status": "improving",
        "strategy": island.strategy["name"],
        "message": f"Island {island.id} ({island.strategy['name']}): Agent improving code...",
    })

    inventor_opts = ClaudeAgentOptions(
            cwd=str(island.path),
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
            permission_mode="acceptEdits",
            max_turns=80,
            env=get_claude_auth_env(),
        )
    if model:
        inventor_opts.model = model

    success = False
    try:
        # Do NOT return/break inside async for — that forcefully closes the
        # query() async generator, triggering anyio cancel-scope cleanup in
        # a different task context → RuntimeError.  Set flag, let loop drain.
        async for message in query(
            prompt=prompt,
            options=inventor_opts,
        ):
            if success:
                continue

            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(event_queue, "island_log", {
                                "island_id": island.id,
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
                            "tool": name,
                            "summary": summary,
                        })

            elif isinstance(message, ResultMessage):
                island.status = "improved"
                summary_file = island.path / "improvement_summary.txt"
                if summary_file.exists():
                    island.improvement_summary = summary_file.read_text()[:500]

                # Extract token usage
                result_usage = getattr(message, "usage", None)
                if isinstance(result_usage, dict):
                    island.input_tokens += result_usage.get("input_tokens", 0) or 0
                    island.output_tokens += result_usage.get("output_tokens", 0) or 0
                cost = getattr(message, "total_cost_usd", None)

                push_event(event_queue, "island_tokens", {
                    "island_id": island.id,
                    "input_tokens": island.input_tokens,
                    "output_tokens": island.output_tokens,
                    "cost_usd": cost,
                })
                push_event(event_queue, "island_status", {
                    "island_id": island.id,
                    "status": "improved",
                    "summary": island.improvement_summary,
                    "message": f"Island {island.id} ({island.strategy['name']}): Code improved",
                })
                success = True

    except Exception as e:
        island.error = str(e)
        island.status = "error"
        push_event(event_queue, "island_error", {
            "island_id": island.id,
            "error": str(e),
            "message": f"Island {island.id} ({island.strategy['name']}): Agent error — {e}",
        })
        return False

    return success


# ═══════════════════════════════════════════════════════════════════
# DEBUG AGENT
# ═══════════════════════════════════════════════════════════════════

async def _run_debug_agent(
    island: Island,
    phase: str,
    error_output: str,
    config: PipelineConfig,
    event_queue: queue.Queue | None = None,
    model: str | None = None,
) -> bool:
    """Run a Claude debug agent to fix an island's code after a failure."""
    # Determine the actual package directory name
    actual_pkg = island.pkg_name or _find_method_pkg_dir(island.path) or "gs_method"
    prompt = DEBUGGER_PROMPT_INVENT.format(
        phase=phase,
        error_output=error_output[-2000:],
        island_dir=str(island.path),
        pkg_name=actual_pkg,
        method_name_cli=island.method_name_cli,
        gs_root=str(config.gs_template_root),
    )

    push_event(event_queue, "island_status", {
        "island_id": island.id,
        "status": "debugging",
        "message": f"Island {island.id}: Debugging {phase} error...",
    })

    debug_opts = ClaudeAgentOptions(
            cwd=str(island.path),
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
            permission_mode="acceptEdits",
            max_turns=30,
            env=get_claude_auth_env(),
        )
    if model:
        debug_opts.model = model

    success = False
    try:
        async for message in query(
            prompt=prompt,
            options=debug_opts,
        ):
            if success:
                continue

            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(event_queue, "island_log", {
                                "island_id": island.id,
                                "message": f"[DEBUG] {text[:200]}",
                            })
                    elif isinstance(block, ToolUseBlock):
                        name = getattr(block, "name", "?")
                        tool_input = getattr(block, "input", {})
                        summary = ""
                        for key in ("file_path", "command", "pattern"):
                            if key in tool_input:
                                summary = str(tool_input[key])[:100]
                                break
                        push_event(event_queue, "island_tool", {
                            "island_id": island.id,
                            "tool": name,
                            "summary": f"[DEBUG] {summary}",
                        })

            elif isinstance(message, ResultMessage):
                result_usage = getattr(message, "usage", None)
                if isinstance(result_usage, dict):
                    island.input_tokens += result_usage.get("input_tokens", 0) or 0
                    island.output_tokens += result_usage.get("output_tokens", 0) or 0

                push_event(event_queue, "island_tokens", {
                    "island_id": island.id,
                    "input_tokens": island.input_tokens,
                    "output_tokens": island.output_tokens,
                })
                push_event(event_queue, "island_log", {
                    "island_id": island.id,
                    "message": f"Debug agent finished for {phase} error",
                })
                success = True

    except Exception as e:
        push_event(event_queue, "island_log", {
            "island_id": island.id,
            "message": f"Debug agent error: {e}",
        })
        return False

    return success


# ═══════════════════════════════════════════════════════════════════
# PARALLEL TRAIN HELPER
# ═══════════════════════════════════════════════════════════════════

async def _train_island_async(
    island: Island,
    max_iters: int,
    data_path: str,
    output_dir: Path,
    conda_env: str,
    gpu: int,
    scene: str,
    round_idx: int,
    event_queue: queue.Queue | None,
    config: PipelineConfig | None = None,
    model: str | None = None,
):
    """Install + train a single island, with debug retries on failure."""
    _config = config or PipelineConfig()

    for attempt in range(MAX_DEBUG_RETRIES + 1):
        is_retry = attempt > 0
        retry_label = f" (retry {attempt}/{MAX_DEBUG_RETRIES})" if is_retry else ""

        # ── Install ──
        island.status = "training"
        push_event(event_queue, "island_status", {
            "island_id": island.id,
            "status": "installing",
            "message": f"Island {island.id}{retry_label}: Installing on GPU {gpu}...",
        })

        install_ok, install_err = await asyncio.to_thread(
            _install_method, island, conda_env, event_queue,
        )
        if not install_ok:
            if attempt < MAX_DEBUG_RETRIES:
                push_event(event_queue, "island_log", {
                    "island_id": island.id,
                    "message": f"Install failed, launching debug agent{retry_label}...",
                })
                await _run_debug_agent(island, "installation", install_err, _config, event_queue, model)
                continue
            else:
                island.alive = False
                island.status = "error"
                island.error = f"Install failed after {MAX_DEBUG_RETRIES} debug retries: {install_err[:200]}"
                island.psnr = 0.0
                push_event(event_queue, "island_error", {
                    "island_id": island.id,
                    "error": island.error,
                    "message": f"Island {island.id}: Install FAILED after {MAX_DEBUG_RETRIES} retries",
                })
                return

        # ── Train ──
        push_event(event_queue, "island_status", {
            "island_id": island.id,
            "status": "training",
            "message": f"Island {island.id}{retry_label}: Training {max_iters} iters on GPU {gpu}...",
        })
        if _config.viewer_port:
            push_event(event_queue, "viewer_started", {
                "island_id": island.id,
                "port": _config.viewer_port,
                "url": f"https://viewer.nerfstudio.com/versions/24-05/?websocket_url=ws://localhost:{_config.viewer_port}",
            })

        train_ok, train_err = await asyncio.to_thread(
            _train_method, island, max_iters, data_path, output_dir,
            conda_env, gpu, event_queue, _config.viewer_port, _config.num_views,
        )
        if not train_ok:
            if attempt < MAX_DEBUG_RETRIES:
                push_event(event_queue, "island_log", {
                    "island_id": island.id,
                    "message": f"Training failed, launching debug agent{retry_label}...",
                })
                await _run_debug_agent(island, "training", train_err, _config, event_queue, model)
                continue
            else:
                island.alive = False
                island.status = "error"
                island.error = f"Training failed after {MAX_DEBUG_RETRIES} debug retries: {train_err[:200]}"
                island.psnr = 0.0
                push_event(event_queue, "island_error", {
                    "island_id": island.id,
                    "error": island.error,
                    "message": f"Island {island.id}: Training FAILED after {MAX_DEBUG_RETRIES} retries",
                })
                return

        # Both install and train succeeded — break retry loop
        break

    # Read PSNR
    psnr = _read_psnr_from_tb(output_dir, scene, island.method_name_cli)
    island.psnr = psnr or 0.0
    island.psnrs_by_round[round_idx + 1] = island.psnr
    island.status = "trained"

    push_event(event_queue, "island_psnr", {
        "island_id": island.id,
        "psnr": island.psnr,
        "round": round_idx + 1,
        "message": f"Island {island.id} ({island.strategy['name']}): PSNR = {island.psnr:.2f} dB",
    })


# ═══════════════════════════════════════════════════════════════════
# BATCHED PARALLEL TRAINING
# ═══════════════════════════════════════════════════════════════════

async def _train_islands_batched(
    alive_islands: list[Island],
    gpus: list[int],
    max_iters: int,
    data_path: str,
    output_dir: Path,
    conda_env: str,
    scene: str,
    round_idx: int,
    event_queue: queue.Queue | None,
    config: PipelineConfig | None = None,
    model: str | None = None,
):
    """Train islands in batches of len(gpus), one island per GPU.

    With 4 GPUs and 8 islands: trains islands 0-3 first (one per GPU),
    waits for all to finish, then trains islands 4-7.
    """
    batch_size = len(gpus)

    for batch_start in range(0, len(alive_islands), batch_size):
        batch = alive_islands[batch_start:batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(alive_islands) + batch_size - 1) // batch_size

        push_event(event_queue, "train_batch", {
            "batch": batch_num,
            "total_batches": total_batches,
            "island_ids": [isl.id for isl in batch],
            "message": (
                f"Training batch {batch_num}/{total_batches}: "
                f"islands {[isl.id for isl in batch]} on GPUs {gpus[:len(batch)]}"
            ),
        })

        def _run_train_in_thread(island_t, gpu_id_t):
            return asyncio.run(
                _train_island_async(
                    island_t, max_iters, data_path, output_dir,
                    conda_env, gpu_id_t, scene, round_idx, event_queue,
                    config=config, model=model,
                )
            )

        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = [
                loop.run_in_executor(pool, _run_train_in_thread, island, gpus[i])
                for i, island in enumerate(batch)
            ]
            await asyncio.gather(*futures, return_exceptions=True)


# ═══════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════

async def run_invent_pipeline(
    arxiv: str | None,
    impl_path: str,
    workspace: Path,
    event_queue: queue.Queue | None = None,
    gpu: str = "0",
    data_path: str | None = None,
    scene: str = "garden",
    viewer_port: int | None = None,
    model: str | None = None,
):
    """
    Run the full evolutionary tournament pipeline.

    Args:
        arxiv: arXiv URL/ID for the paper
        impl_path: Path to existing generated implementation
        workspace: Working directory for this invent run
        event_queue: SSE event queue for UI updates
        gpu: GPU device ID(s), comma-separated for multi-GPU
        data_path: Training data path (defaults to garden scene)
        scene: Scene name for training
        viewer_port: Nerfstudio viewer websocket port (None = no viewer)
    """
    config = PipelineConfig()
    config.viewer_port = viewer_port
    impl = Path(impl_path)
    gpus = [int(g.strip()) for g in gpu.split(",")]

    if not data_path:
        data_path = config.default_dataset

    paper_path = _find_paper_path(impl, config.workspace_dir)
    base_snake, base_cli, base_var, base_pkg = _find_method_name_from_impl(impl)

    output_dir = workspace / "training_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    push_event(event_queue, "stage", {
        "stage": "setup",
        "message": f"Setting up tournament: base method={base_cli}, {len(STRATEGIES)} islands, GPUs={gpus}",
    })

    # ─── Clean environment ────────────────────────────────────────
    push_event(event_queue, "stage", {
        "stage": "cleanup",
        "message": "Cleaning up any previously installed island packages...",
    })
    _uninstall_all_islands(config.conda_env, base_pkg=base_pkg)

    # ─── Create Islands ──────────────────────────────────────────
    islands: list[Island] = []
    for i, strategy in enumerate(STRATEGIES):
        island_dir = workspace / f"island_{i}"
        island_dir.mkdir(parents=True, exist_ok=True)

        # Copy base implementation (package dir is named after the method, e.g. "dashgaussian")
        if (island_dir / base_pkg).exists():
            shutil.rmtree(island_dir / base_pkg)
        shutil.copytree(impl / base_pkg, island_dir / base_pkg)
        for f in ["pyproject.toml", "README.md"]:
            src = impl / f
            if src.exists():
                shutil.copy2(src, island_dir / f)

        method_snake = f"{base_snake}_island_{i}"
        method_cli = f"{base_cli}-island-{i}"
        method_var = f"{base_var}_island_{i}"

        island = Island(
            id=i,
            strategy=strategy,
            path=island_dir,
            method_name=method_snake,
            method_name_cli=method_cli,
            method_name_var=method_var,
            train_output_dir=output_dir,
        )
        islands.append(island)

        push_event(event_queue, "island_created", {
            "island_id": i,
            "strategy": strategy["name"],
            "strategy_id": strategy["id"],
            "method_name": method_cli,
            "message": f"Island {i}: {strategy['name']} ({method_cli})",
        })

    # ─── Phase 1: Inventor Agents Improve Code (ALL PARALLEL) ────
    push_event(event_queue, "stage", {
        "stage": "improving",
        "message": "Phase 1: Spawning 8 inventor agents in PARALLEL...",
    })

    # Run each agent in its own thread with its own event loop to avoid
    # anyio cancel-scope cross-task contamination.
    def _run_agent_in_thread(island):
        return asyncio.run(
            _run_inventor_agent(island, paper_path, arxiv, config, event_queue, model)
        )

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(islands)) as pool:
        futures = [
            loop.run_in_executor(pool, _run_agent_in_thread, island)
            for island in islands
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

    for island, result in zip(islands, results):
        if isinstance(result, Exception):
            island.alive = False
            island.status = "error"
            island.error = str(result)
            push_event(event_queue, "island_error", {
                "island_id": island.id,
                "error": str(result),
                "message": f"Island {island.id} failed: {result}",
            })
        elif not result:
            island.alive = False
            island.status = "error"

    alive_count = sum(1 for isl in islands if isl.alive)
    total_in = sum(isl.input_tokens for isl in islands)
    total_out = sum(isl.output_tokens for isl in islands)
    push_event(event_queue, "tokens_update", {
        "input_tokens": total_in,
        "output_tokens": total_out,
    })
    push_event(event_queue, "stage", {
        "stage": "improving_done",
        "message": f"Code improvement done. {alive_count}/{len(islands)} islands successful.",
    })

    if alive_count == 0:
        push_event(event_queue, "error", {"message": "All islands failed during code improvement."})
        push_event(event_queue, "done", {"message": "Tournament aborted — no islands survived."})
        return {"winner": None, "islands": [_island_summary(isl) for isl in islands]}

    # ─── Phase 1.5: Uniquify package names ────────────────────────
    push_event(event_queue, "stage", {
        "stage": "uniquify",
        "message": f"Renaming packages to unique names ({base_pkg} → {base_pkg}_island_N)...",
    })

    for island in islands:
        if not island.alive:
            continue
        # Find the actual package dir (agent might have kept or renamed it)
        actual_pkg = _find_method_pkg_dir(island.path) or base_pkg
        ok = _uniquify_island_package(island, actual_pkg)
        if not ok:
            island.alive = False
            island.status = "error"
            island.error = "Package uniquification failed"
            push_event(event_queue, "island_error", {
                "island_id": island.id,
                "error": f"{actual_pkg} directory not found after agent improvement",
                "message": f"Island {island.id}: Package rename failed — eliminated",
            })
        else:
            push_event(event_queue, "island_log", {
                "island_id": island.id,
                "message": f"Renamed to {island.pkg_name}",
            })

    alive_count = sum(1 for isl in islands if isl.alive)
    if alive_count == 0:
        push_event(event_queue, "error", {"message": "All islands failed during package rename."})
        push_event(event_queue, "done", {"message": "Tournament aborted."})
        return {"winner": None, "islands": [_island_summary(isl) for isl in islands]}

    # ─── Phase 2: Tournament Rounds (PARALLEL across GPUs) ────────
    for round_idx, round_cfg in enumerate(TOURNAMENT_ROUNDS):
        max_iters = round_cfg["max_iters"]
        target_survivors = round_cfg["survivors"]
        label = round_cfg["label"]

        alive_islands = [isl for isl in islands if isl.alive]
        if len(alive_islands) <= target_survivors:
            push_event(event_queue, "round_skip", {
                "round": round_idx + 1,
                "message": f"Skipping {label} — only {len(alive_islands)} alive (need >{target_survivors})",
            })
            continue

        push_event(event_queue, "round_start", {
            "round": round_idx + 1,
            "label": label,
            "alive_count": len(alive_islands),
            "max_iters": max_iters,
            "message": f"{label}: Training {len(alive_islands)} islands for {max_iters} iters across {len(gpus)} GPU(s)...",
        })

        # Train islands in GPU-sized batches (e.g. 4 GPUs → 4 at a time)
        await _train_islands_batched(
            alive_islands, gpus, max_iters, data_path, output_dir,
            config.conda_env, scene, round_idx, event_queue,
            config=config, model=model,
        )

        # Rank and eliminate
        alive_islands = [isl for isl in islands if isl.alive]
        alive_islands.sort(key=lambda isl: isl.psnr, reverse=True)

        survivors = alive_islands[:target_survivors]
        eliminated = alive_islands[target_survivors:]

        for isl in eliminated:
            isl.alive = False
            isl.status = "eliminated"
            push_event(event_queue, "island_eliminated", {
                "island_id": isl.id,
                "psnr": isl.psnr,
                "round": round_idx + 1,
                "message": f"Island {isl.id} ({isl.strategy['name']}): ELIMINATED (PSNR={isl.psnr:.2f})",
            })

        ranking_msg = " > ".join(
            f"#{isl.id}({isl.strategy['emoji']}){isl.psnr:.2f}"
            for isl in alive_islands
        )
        push_event(event_queue, "round_result", {
            "round": round_idx + 1,
            "ranking": [
                {"island_id": isl.id, "psnr": isl.psnr, "strategy": isl.strategy["name"], "alive": isl.alive}
                for isl in alive_islands
            ],
            "survivors": [isl.id for isl in survivors],
            "eliminated": [isl.id for isl in eliminated],
            "message": f"{label} results: {ranking_msg}",
        })

    # ─── Declare Winner ──────────────────────────────────────────
    alive_islands = [isl for isl in islands if isl.alive]
    winner = None
    if alive_islands:
        winner = max(alive_islands, key=lambda isl: isl.psnr)
        winner.status = "winner"
        push_event(event_queue, "winner", {
            "island_id": winner.id,
            "strategy": winner.strategy["name"],
            "psnr": winner.psnr,
            "improvement_summary": winner.improvement_summary,
            "method_path": str(winner.path),
            "message": (
                f"WINNER: Island {winner.id} ({winner.strategy['name']}) "
                f"with PSNR = {winner.psnr:.2f} dB"
            ),
        })
    else:
        push_event(event_queue, "error", {
            "message": "No islands survived the tournament.",
        })

    result = {
        "winner": _island_summary(winner) if winner else None,
        "islands": [_island_summary(isl) for isl in islands],
        "rounds": [{"label": r["label"], "max_iters": r["max_iters"]} for r in TOURNAMENT_ROUNDS],
    }

    total_in = sum(isl.input_tokens for isl in islands)
    total_out = sum(isl.output_tokens for isl in islands)
    push_event(event_queue, "tokens_update", {
        "input_tokens": total_in,
        "output_tokens": total_out,
    })
    push_event(event_queue, "done", {
        "message": "Evolutionary tournament complete!",
        "result": result,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    })

    # Save results
    results_file = workspace / "tournament_results.json"
    results_file.write_text(json.dumps(result, indent=2))

    return result


def _island_summary(island: Island) -> dict:
    """Serialize an island to a JSON-friendly dict."""
    return {
        "id": island.id,
        "strategy": island.strategy["name"],
        "strategy_id": island.strategy["id"],
        "method_name_cli": island.method_name_cli,
        "pkg_name": island.pkg_name,
        "psnr": island.psnr,
        "psnrs_by_round": island.psnrs_by_round,
        "alive": island.alive,
        "status": island.status,
        "improvement_summary": island.improvement_summary,
        "error": island.error,
        "path": str(island.path),
        "input_tokens": island.input_tokens,
        "output_tokens": island.output_tokens,
    }


# ═══════════════════════════════════════════════════════════════════
# SCENE-DRIVEN INVENTION
# ═══════════════════════════════════════════════════════════════════

# Colors for scene prior islands (used by the UI)
SCENE_PRIOR_COLORS = [
    "#f0883e", "#58a6ff", "#d29922", "#79c0ff",
    "#bc8cff", "#3fb950", "#f85149", "#e6edf3",
]

SCENE_RESEARCHER_PROMPT = r"""You are a Physics Research Agent for Gaussian Splatting scene reconstruction.

## Scene Description
{scene_description}

## Your Task
Research the physical phenomena that affect visual appearance and rendering of this type of scene.
Search the web for:
1. Physical equations governing light transport in this scene (scattering, absorption, refraction, etc.)
2. Material/surface properties relevant to objects in the scene
3. Atmospheric/environmental effects (fog, underwater caustics, smoke, etc.)
4. Any recent computer graphics or neural rendering papers that address this scene type
5. Mathematical models that could be encoded as loss functions, rendering modifications, or learnable parameters

## Output Format
You MUST output EXACTLY a JSON block with 8 distinct knowledge priors. Each prior should represent
a DIFFERENT physical phenomenon or rendering technique relevant to this scene.

```json
[
  {{
    "name": "Short Name (2-4 words)",
    "emoji": "X",
    "physical_phenomenon": "What physical effect this models",
    "equations": "Key equations in LaTeX or plain math notation",
    "gs_implementation_hint": "How to implement this in Gaussian Splatting: which extension points to use (compute_custom_losses, get_outputs, populate_modules, etc.), what new parameters to add, what loss functions to create",
    "web_sources": "URLs or paper titles where you found this information"
  }},
  ...8 total entries...
]
```

## Guidelines
- Each prior must be DISTINCT — cover different physical aspects of the scene
- Prioritize priors that can be practically implemented as:
  - New loss functions in compute_custom_losses()
  - New per-Gaussian parameters in populate_modules()
  - Modified rendering in get_outputs()
  - Custom regularization terms
- Include the actual mathematical equations when possible
- Be specific about HOW each prior maps to Gaussian Splatting modifications
- One prior should focus on general scene-type-specific optimizations (e.g., LR tuning for this scene type)
- One prior should focus on regularization specific to this scene type
- Use single-letter emojis: L, S, R, A, M, D, P, W (for Loss, Scattering, Refraction, Absorption, Material, Density, Parameters, Wavelength — adapt to scene)

Output ONLY the JSON block — no other text before or after.
"""

SCENE_INVENTOR_PROMPT = r"""You are a Scene-Specialized Inventor Agent for the Splatify evolutionary pipeline.
You are competing against 7 other agents to produce the BEST Gaussian Splatting
implementation for a specific scene type, measured by PSNR.

## Scene Description
{scene_description}

## Your Knowledge Prior: {prior_name}
{prior_description}

### Physical Phenomenon
{physical_phenomenon}

### Key Equations
{equations}

### Implementation Guidance
{gs_implementation_hint}

### Reference Sources
{web_sources}

## Your Task
1. Read the base implementation at: {island_dir}
   - Read ALL files in {method_pkg}/ (model.py, configs.py, losses.py, rendering.py, datamanager.py, __init__.py)
   - Read pyproject.toml
2. Read the gs_template framework for reference:
   - {gs_root}/model.py — Base class with 9 extension points
   - {gs_root}/rendering.py — Rendering helpers (prepare_camera, rasterize_gaussians, assemble_outputs)
   - {gs_root}/losses.py — 30+ reusable loss functions
   - {gs_root}/shading.py — SH, normals, cubemaps
   - {gs_root}/configs.py — MethodSpecification examples
3. Read 2-3 example methods from {methods_dir}/ for implementation patterns
4. Use WebSearch to find additional details about implementing your knowledge prior
5. Implement your physics-based improvement:
   - Add new loss functions based on the equations above
   - Add new learnable parameters if the prior requires them
   - Modify rendering if needed for scene-specific effects
   - Tune hyperparameters for this scene type

## Critical Rules
- You MUST rename the method to "{method_name}" in ALL files:
  - {method_pkg}/configs.py: method_name="{method_name_cli}"
  - pyproject.toml: entry point must be "{method_name_cli}" = "{method_pkg}.configs:{method_name_var}"
  - {method_pkg}/__init__.py: update variable names if needed
- All internal imports must use "{method_pkg}" as the package name
- Keep the method installable: `pip install -e .` must work
- Keep the method trainable: `ns-train {method_name_cli} --data <path> --max-num-iterations 2000` must work
- Do NOT break existing functionality — enhance it with your physics prior
- Write complete files (no patches), use the Write tool
- Focus on changes that MAXIMIZE PSNR — be grounded in physics but practical
- Be bold but cautious — untested theoretical changes may crash training
- Do NOT run pip install or ns-train yourself — the tournament pipeline handles that

## Output
After making changes, write a brief summary to {island_dir}/improvement_summary.txt explaining:
- What physical prior you implemented and the key equations
- What code changes you made (new losses, params, rendering modifications)
- Expected impact on PSNR for this scene type
- Any risks or tradeoffs
"""


def _create_base_gs_method(output_dir: Path, method_name: str = "scene_gs") -> Path:
    """Create a minimal base Gaussian Splatting method package from gs_template.

    This generates a simple splatfacto-equivalent method that serves as the
    starting point for scene-driven invention. Each island will modify this base.

    Returns the path to the created implementation directory.
    """
    impl_dir = output_dir / "base_impl"
    impl_dir.mkdir(parents=True, exist_ok=True)
    pkg_dir = impl_dir / method_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    method_cls = method_name.replace("_", " ").title().replace(" ", "")
    method_cli = method_name.replace("_", "-")
    method_var = f"{method_name}_method"

    # __init__.py
    (pkg_dir / "__init__.py").write_text(
        f'"""Scene-specialized Gaussian Splatting method: {method_name}."""\n'
        f"from {method_name}.model import {method_cls}Model, {method_cls}ModelConfig\n"
        f"from {method_name}.configs import {method_var}\n\n"
        f'__all__ = ["{method_cls}Model", "{method_cls}ModelConfig", "{method_var}"]\n'
    )

    # model.py
    (pkg_dir / "model.py").write_text(
        f'"""{method_cls} Model — Scene-specialized Gaussian Splatting."""\n'
        f"from __future__ import annotations\n\n"
        f"from dataclasses import dataclass, field\n"
        f"from typing import Type, Dict, Any\n\n"
        f"import torch\n"
        f"from gs_template.model import GSTemplateModel, GSTemplateModelConfig\n"
        f"from {method_name}.losses import compute_scene_losses\n\n\n"
        f"@dataclass\n"
        f"class {method_cls}ModelConfig(GSTemplateModelConfig):\n"
        f'    """Config for {method_cls} model."""\n\n'
        f"    _target: Type = field(default_factory=lambda: {method_cls}Model)\n\n\n"
        f"class {method_cls}Model(GSTemplateModel):\n"
        f'    """Scene-specialized GS model. Override extension points to add physics priors."""\n\n'
        f"    config: {method_cls}ModelConfig\n\n"
        f"    def populate_modules(self):\n"
        f"        super().populate_modules()\n"
        f"        # Add scene-specific parameters or modules here\n\n"
        f"    def compute_custom_losses(\n"
        f"        self,\n"
        f"        outputs: Dict[str, Any],\n"
        f"        batch: Dict[str, Any],\n"
        f"        metrics_dict: Dict[str, Any],\n"
        f"    ) -> Dict[str, Any]:\n"
        f"        loss_dict = super().compute_custom_losses(outputs, batch, metrics_dict)\n"
        f"        scene_losses = compute_scene_losses(self, outputs, batch)\n"
        f"        loss_dict.update(scene_losses)\n"
        f"        return loss_dict\n"
    )

    # losses.py
    (pkg_dir / "losses.py").write_text(
        f'"""{method_cls} losses — Scene-specific loss functions."""\n'
        f"from __future__ import annotations\n\n"
        f"from typing import Dict, Any\n\n"
        f"import torch\n\n\n"
        f"def compute_scene_losses(\n"
        f"    model: Any,\n"
        f"    outputs: Dict[str, Any],\n"
        f"    batch: Dict[str, Any],\n"
        f") -> Dict[str, torch.Tensor]:\n"
        f'    """Compute scene-specific loss terms. Override this with physics-based losses."""\n'
        f"    return {{}}\n"
    )

    # rendering.py
    (pkg_dir / "rendering.py").write_text(
        f'"""{method_cls} rendering — Scene-specific rendering helpers."""\n'
        f"from __future__ import annotations\n\n"
        f"# Re-export base rendering helpers\n"
        f"from gs_template.rendering import (\n"
        f"    prepare_camera,\n"
        f"    gather_base_params,\n"
        f"    rasterize_gaussians,\n"
        f"    assemble_outputs,\n"
        f")\n\n"
        f"__all__ = [\n"
        f'    "prepare_camera",\n'
        f'    "gather_base_params",\n'
        f'    "rasterize_gaussians",\n'
        f'    "assemble_outputs",\n'
        f"]\n"
    )

    # datamanager.py
    (pkg_dir / "datamanager.py").write_text(
        f'"""{method_cls} DataManager — Scene-specific data handling."""\n'
        f"from __future__ import annotations\n\n"
        f"from nerfstudio.data.datamanagers.full_images_datamanager import (\n"
        f"    FullImageDatamanagerConfig,\n"
        f")\n\n"
        f"# Use standard full-image datamanager by default\n"
        f"# Override with custom DataManager if scene requires special preprocessing\n"
    )

    # configs.py
    (pkg_dir / "configs.py").write_text(
        f'"""{method_cls} — Method registration for nerfstudio."""\n'
        f"from __future__ import annotations\n\n"
        f"from nerfstudio.cameras.camera_optimizers import CameraOptimizerConfig\n"
        f"from nerfstudio.configs.base_config import ViewerConfig\n"
        f"from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig\n"
        f"from nerfstudio.data.dataparsers.nerfstudio_dataparser import NerfstudioDataParserConfig\n"
        f"from nerfstudio.engine.optimizers import AdamOptimizerConfig\n"
        f"from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig\n"
        f"from nerfstudio.engine.trainer import TrainerConfig\n"
        f"from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig\n"
        f"from nerfstudio.plugins.types import MethodSpecification\n\n"
        f"from {method_name}.model import {method_cls}ModelConfig\n\n\n"
        f"{method_var} = MethodSpecification(\n"
        f"    config=TrainerConfig(\n"
        f'        method_name="{method_cli}",\n'
        f"        steps_per_eval_image=100,\n"
        f"        steps_per_eval_batch=0,\n"
        f"        steps_per_save=2000,\n"
        f"        steps_per_eval_all_images=1000,\n"
        f"        max_num_iterations=30000,\n"
        f"        mixed_precision=False,\n"
        f"        pipeline=VanillaPipelineConfig(\n"
        f"            datamanager=FullImageDatamanagerConfig(\n"
        f"                dataparser=NerfstudioDataParserConfig(load_3D_points=True),\n"
        f'                cache_images_type="uint8",\n'
        f"            ),\n"
        f"            model={method_cls}ModelConfig(),\n"
        f"        ),\n"
        f"        optimizers={{\n"
        f'            "means": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=1.6e-4, eps=1e-15),\n"
        f"                \"scheduler\": ExponentialDecaySchedulerConfig(\n"
        f"                    lr_final=1.6e-6, max_steps=30000,\n"
        f"                ),\n"
        f"            }},\n"
        f'            "features_dc": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=0.0025, eps=1e-15),\n"
        f"                \"scheduler\": None,\n"
        f"            }},\n"
        f'            "features_rest": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=0.0025 / 20, eps=1e-15),\n"
        f"                \"scheduler\": None,\n"
        f"            }},\n"
        f'            "opacities": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=0.05, eps=1e-15),\n"
        f"                \"scheduler\": None,\n"
        f"            }},\n"
        f'            "scales": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=0.005, eps=1e-15),\n"
        f"                \"scheduler\": None,\n"
        f"            }},\n"
        f'            "quats": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=0.001, eps=1e-15),\n"
        f"                \"scheduler\": None,\n"
        f"            }},\n"
        f'            "camera_opt": {{\n'
        f"                \"optimizer\": AdamOptimizerConfig(lr=1e-4, eps=1e-15),\n"
        f"                \"scheduler\": ExponentialDecaySchedulerConfig(\n"
        f"                    lr_final=5e-7, max_steps=30000,\n"
        f"                    warmup_steps=1000, lr_pre_warmup=0,\n"
        f"                ),\n"
        f"            }},\n"
        f"        }},\n"
        f"        viewer=ViewerConfig(num_rays_per_chunk=1 << 15),\n"
        f"    ),\n"
        f'    description="{method_cls}: Scene-specialized Gaussian Splatting method.",\n'
        f")\n"
    )

    # pyproject.toml
    (impl_dir / "pyproject.toml").write_text(
        f"[build-system]\n"
        f'requires = ["setuptools>=61.0"]\n'
        f'build-backend = "setuptools.build_meta"\n\n'
        f"[project]\n"
        f'name = "{method_cli}"\n'
        f'version = "0.1.0"\n'
        f'requires-python = ">=3.10"\n'
        f"dependencies = [\n"
        f'    "nerfstudio>=1.1.0",\n'
        f"]\n\n"
        f'[project.entry-points."nerfstudio.method_configs"]\n'
        f'{method_cli} = "{method_name}.configs:{method_var}"\n'
    )

    # README.md
    (impl_dir / "README.md").write_text(
        f"# {method_cls}\n\n"
        f"Scene-specialized Gaussian Splatting method.\n\n"
        f"## Usage\n\n"
        f"```bash\n"
        f"pip install -e .\n"
        f"ns-install-cli\n"
        f"ns-train {method_cli} --data <path>\n"
        f"```\n"
    )

    return impl_dir


async def _run_researcher_agent(
    scene_description: str,
    workspace: Path,
    event_queue: queue.Queue | None = None,
    model: str | None = None,
) -> list[dict]:
    """Run a Claude researcher agent to discover 8 physics priors for a scene.

    Returns a list of 8 prior dicts with keys:
    name, emoji, physical_phenomenon, equations, gs_implementation_hint, web_sources
    """
    prompt = SCENE_RESEARCHER_PROMPT.format(scene_description=scene_description)

    push_event(event_queue, "stage", {
        "stage": "researching",
        "message": f"Researching physics priors for scene: {scene_description[:100]}...",
    })

    result_text = ""
    researcher_opts = ClaudeAgentOptions(
        cwd=str(workspace),
        allowed_tools=["WebSearch", "WebFetch", "Read", "Write", "Bash"],
        permission_mode="acceptEdits",
        max_turns=40,
        env=get_claude_auth_env(),
    )
    if model:
        researcher_opts.model = model

    try:
        async for message in query(
            prompt=prompt,
            options=researcher_opts,
        ):
            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            result_text = text
                            push_event(event_queue, "island_log", {
                                "island_id": -1,
                                "message": f"[Researcher] {text[:200]}",
                            })

            elif isinstance(message, ResultMessage):
                # The result should contain the JSON
                final = getattr(message, "result", "") or result_text
                if final:
                    result_text = final
                break

    except Exception as e:
        push_event(event_queue, "error", {
            "message": f"Researcher agent failed: {e}",
        })
        return []

    # Parse the JSON from the result
    priors = _parse_priors_json(result_text)

    if not priors or len(priors) < 8:
        push_event(event_queue, "island_log", {
            "island_id": -1,
            "message": f"Researcher returned {len(priors) if priors else 0} priors, padding to 8...",
        })
        priors = _pad_priors(priors or [], scene_description)

    # Save priors to workspace
    (workspace / "scene_priors.json").write_text(json.dumps(priors, indent=2))

    for i, prior in enumerate(priors[:8]):
        push_event(event_queue, "island_created", {
            "island_id": i,
            "strategy": prior["name"],
            "strategy_id": f"scene_prior_{i}",
            "message": f"Prior {i}: {prior['name']} — {prior.get('physical_phenomenon', '')[:80]}",
        })

    push_event(event_queue, "stage", {
        "stage": "research_done",
        "message": f"Discovered {len(priors[:8])} physics priors for scene.",
    })

    return priors[:8]


def _parse_priors_json(text: str) -> list[dict] | None:
    """Extract the JSON array of priors from researcher agent output."""
    # Try to find a JSON array in the text
    # Look for ```json ... ``` blocks first
    json_match = re.search(r'```(?:json)?\s*\n(\[[\s\S]*?\])\s*\n```', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find a bare JSON array
    json_match = re.search(r'(\[\s*\{[\s\S]*?\}\s*\])', text)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Last resort: try the whole text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    return None


def _pad_priors(priors: list[dict], scene_description: str) -> list[dict]:
    """Pad priors list to 8 with generic fallback strategies."""
    fallbacks = [
        {
            "name": "Depth Regularization",
            "emoji": "D",
            "physical_phenomenon": "Scene depth consistency and smoothness",
            "equations": "L_depth = pearson_correlation(predicted_depth, mono_depth)",
            "gs_implementation_hint": "Add pearson_depth_loss from gs_template/losses.py in compute_custom_losses",
            "web_sources": "Standard depth regularization for Gaussian Splatting",
        },
        {
            "name": "Opacity Control",
            "emoji": "O",
            "physical_phenomenon": "Gaussian opacity distribution for clean geometry",
            "equations": "L_entropy = -sum(alpha * log(alpha) + (1-alpha) * log(1-alpha))",
            "gs_implementation_hint": "Add opacity_entropy_loss from gs_template/losses.py to encourage binary opacities",
            "web_sources": "Opacity entropy regularization (SuGaR, 2DGS)",
        },
        {
            "name": "Scale Regularization",
            "emoji": "S",
            "physical_phenomenon": "Preventing degenerate Gaussian shapes",
            "equations": "L_scale = max(scales, dim=-1) / (min(scales, dim=-1) + eps)",
            "gs_implementation_hint": "Add scale_flatten_loss from gs_template/losses.py to prevent needle-like Gaussians",
            "web_sources": "Scale regularization (2DGS, SuGaR)",
        },
        {
            "name": "Color Enhancement",
            "emoji": "C",
            "physical_phenomenon": "Better color representation via higher SH degree",
            "equations": "color = sum_l sum_m c_lm * Y_lm(view_dir), l=0..3",
            "gs_implementation_hint": "Increase SH degree in model config, tune learning rates for features_rest",
            "web_sources": "Spherical harmonics for view-dependent appearance",
        },
        {
            "name": "Perceptual Quality",
            "emoji": "P",
            "physical_phenomenon": "Human visual perception alignment",
            "equations": "L_perceptual = LPIPS(rendered, gt) + lambda * SSIM(rendered, gt)",
            "gs_implementation_hint": "Add LPIPS loss alongside L1/SSIM, tune the loss weight for optimal PSNR",
            "web_sources": "Perceptual losses in neural rendering",
        },
        {
            "name": "Densification Tuning",
            "emoji": "G",
            "physical_phenomenon": f"Optimal Gaussian placement for {scene_description[:30]} scenes",
            "equations": "grad_threshold adaptive based on training progress",
            "gs_implementation_hint": "Tune densify_grad_thresh, densify_size_thresh, cull_alpha_thresh in config",
            "web_sources": "Densification strategies (AbsGS, Mini-Splatting)",
        },
        {
            "name": "Training Schedule",
            "emoji": "T",
            "physical_phenomenon": "Coarse-to-fine optimization dynamics",
            "equations": "LR(t) = LR_init * (LR_final / LR_init)^(t / T)",
            "gs_implementation_hint": "Tune learning rate schedules, warmup periods, and loss weight annealing",
            "web_sources": "Training schedule optimization for 3DGS",
        },
        {
            "name": "Web-Inspired Innovation",
            "emoji": "W",
            "physical_phenomenon": f"Latest techniques for {scene_description[:30]} reconstruction",
            "equations": "Search for latest 2024-2025 papers on similar scenes",
            "gs_implementation_hint": "Use WebSearch to find and apply the latest relevant technique",
            "web_sources": "Web search during implementation",
        },
    ]

    result = list(priors)
    for fallback in fallbacks:
        if len(result) >= 8:
            break
        # Don't duplicate existing prior names
        existing_names = {p.get("name", "").lower() for p in result}
        if fallback["name"].lower() not in existing_names:
            result.append(fallback)

    return result[:8]


async def _run_scene_inventor_agent(
    island: Island,
    scene_description: str,
    prior: dict,
    config: PipelineConfig,
    event_queue: queue.Queue | None = None,
    model: str | None = None,
) -> bool:
    """Run a Claude inventor agent that implements a physics prior for a scene."""
    gs_root = str(config.gs_template_root)
    methods_dir = str(config.methods_dir)
    method_pkg = _find_method_pkg_dir(island.path) or "scene_gs"

    prompt = SCENE_INVENTOR_PROMPT.format(
        scene_description=scene_description,
        prior_name=prior.get("name", "Unknown"),
        prior_description=f"Apply the physical knowledge of '{prior.get('name', '')}' to improve "
                          f"Gaussian Splatting reconstruction quality for this scene type.",
        physical_phenomenon=prior.get("physical_phenomenon", "N/A"),
        equations=prior.get("equations", "N/A"),
        gs_implementation_hint=prior.get("gs_implementation_hint", "N/A"),
        web_sources=prior.get("web_sources", "N/A"),
        island_dir=str(island.path),
        gs_root=gs_root,
        methods_dir=methods_dir,
        method_name=island.method_name,
        method_name_cli=island.method_name_cli,
        method_name_var=island.method_name_var,
        method_pkg=method_pkg,
    )

    push_event(event_queue, "island_status", {
        "island_id": island.id,
        "status": "improving",
        "strategy": prior.get("name", "Unknown"),
        "message": f"Island {island.id} ({prior['name']}): Implementing physics prior...",
    })

    scene_inventor_opts = ClaudeAgentOptions(
            cwd=str(island.path),
            allowed_tools=["Read", "Write", "Bash", "Glob", "Grep", "WebSearch", "WebFetch"],
            permission_mode="acceptEdits",
            max_turns=80,
            env=get_claude_auth_env(),
        )
    if model:
        scene_inventor_opts.model = model

    success = False
    try:
        async for message in query(
            prompt=prompt,
            options=scene_inventor_opts,
        ):
            if success:
                continue

            if isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        text = getattr(block, "text", "")
                        if text:
                            push_event(event_queue, "island_log", {
                                "island_id": island.id,
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
                    "input_tokens": island.input_tokens,
                    "output_tokens": island.output_tokens,
                    "cost_usd": cost,
                })
                push_event(event_queue, "island_status", {
                    "island_id": island.id,
                    "status": "improved",
                    "summary": island.improvement_summary,
                    "message": f"Island {island.id} ({prior['name']}): Physics prior implemented",
                })
                success = True

    except Exception as e:
        island.error = str(e)
        island.status = "error"
        push_event(event_queue, "island_error", {
            "island_id": island.id,
            "error": str(e),
            "message": f"Island {island.id} ({prior['name']}): Agent error — {e}",
        })
        return False

    return success


async def run_scene_invent_pipeline(
    scene_description: str,
    workspace: Path,
    event_queue: queue.Queue | None = None,
    gpu: str = "0",
    data_path: str | None = None,
    scene: str = "garden",
    viewer_port: int | None = None,
    method_name: str | None = None,
    model: str | None = None,
):
    """
    Run the scene-driven evolutionary tournament pipeline.

    Instead of improving an existing implementation, this mode:
    1. Researches physics priors relevant to the scene description
    2. Creates a base GS method from gs_template
    3. Spawns 8 agents, each implementing a different physics prior
    4. Runs the same elimination tournament (8 -> 4 -> 2 -> 1)

    Args:
        scene_description: Natural language description of the scene type
        workspace: Working directory for this run
        event_queue: SSE event queue for UI updates
        gpu: GPU device ID(s), comma-separated for multi-GPU
        data_path: Training data path (defaults to garden scene)
        scene: Scene name for training
        viewer_port: Nerfstudio viewer websocket port
        method_name: Optional method name override (default: derived from scene)
    """
    config = PipelineConfig()
    config.viewer_port = viewer_port
    gpus = [int(g.strip()) for g in gpu.split(",")]

    if not data_path:
        data_path = config.default_dataset

    # Derive method name from scene description
    if not method_name:
        # Convert scene description to a safe snake_case name
        safe = re.sub(r'[^a-z0-9]+', '_', scene_description.lower().strip())
        safe = re.sub(r'_+', '_', safe).strip('_')[:30]
        method_name = f"scene_{safe}" if safe else "scene_gs"

    output_dir = workspace / "training_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    push_event(event_queue, "stage", {
        "stage": "setup",
        "message": f"Scene-driven tournament: \"{scene_description[:80]}\" | method={method_name}",
    })

    # ─── Phase 0: Research Physics Priors ─────────────────────────
    # Run researcher in its own thread/event-loop to isolate anyio cancel
    # scopes — same pattern used for the inventor agents below.
    def _run_researcher_in_thread():
        return asyncio.run(
            _run_researcher_agent(scene_description, workspace, event_queue, model)
        )

    loop_r = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        priors = await loop_r.run_in_executor(pool, _run_researcher_in_thread)
    if not priors:
        push_event(event_queue, "error", {"message": "Failed to discover physics priors."})
        push_event(event_queue, "done", {"message": "Tournament aborted — no priors found."})
        return {"winner": None, "islands": [], "priors": []}

    # ─── Phase 1: Create Base Implementation ─────────────────────
    push_event(event_queue, "stage", {
        "stage": "scaffolding",
        "message": f"Creating base GS method: {method_name}",
    })
    base_impl = _create_base_gs_method(workspace, method_name=method_name)
    base_pkg = method_name
    base_snake = method_name
    base_cli = method_name.replace("_", "-")
    base_var = f"{method_name}_method"

    # ─── Clean environment ────────────────────────────────────────
    push_event(event_queue, "stage", {
        "stage": "cleanup",
        "message": "Cleaning up any previously installed island packages...",
    })
    _uninstall_all_islands(config.conda_env, base_pkg=base_pkg)

    # ─── Create Islands ──────────────────────────────────────────
    islands: list[Island] = []
    for i, prior in enumerate(priors):
        island_dir = workspace / f"island_{i}"
        island_dir.mkdir(parents=True, exist_ok=True)

        # Copy base implementation
        if (island_dir / base_pkg).exists():
            shutil.rmtree(island_dir / base_pkg)
        shutil.copytree(base_impl / base_pkg, island_dir / base_pkg)
        for f in ["pyproject.toml", "README.md"]:
            src = base_impl / f
            if src.exists():
                shutil.copy2(src, island_dir / f)

        method_snake = f"{base_snake}_island_{i}"
        method_cli = f"{base_cli}-island-{i}"
        method_var_i = f"{base_var}_island_{i}"

        # Create a strategy dict compatible with the existing Island dataclass
        strategy = {
            "id": f"scene_prior_{i}",
            "name": prior.get("name", f"Prior {i}"),
            "emoji": prior.get("emoji", str(i)),
            "description": prior.get("physical_phenomenon", ""),
        }

        island = Island(
            id=i,
            strategy=strategy,
            path=island_dir,
            method_name=method_snake,
            method_name_cli=method_cli,
            method_name_var=method_var_i,
            train_output_dir=output_dir,
        )
        islands.append(island)

    # ─── Phase 2: Scene Inventor Agents (ALL PARALLEL) ───────────
    push_event(event_queue, "stage", {
        "stage": "improving",
        "message": "Phase 2: Spawning 8 scene inventor agents with physics priors in PARALLEL...",
    })

    # Run each agent in its own thread with its own event loop to avoid
    # anyio cancel-scope cross-task contamination (cancel scopes leak
    # between asyncio.Tasks sharing the same loop).
    def _run_agent_in_thread(island, prior):
        return asyncio.run(
            _run_scene_inventor_agent(island, scene_description, prior, config, event_queue, model)
        )

    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(priors)) as pool:
        futures = [
            loop.run_in_executor(pool, _run_agent_in_thread, island, prior)
            for island, prior in zip(islands, priors)
        ]
        results = await asyncio.gather(*futures, return_exceptions=True)

    for island, result in zip(islands, results):
        if isinstance(result, Exception):
            island.alive = False
            island.status = "error"
            island.error = str(result)
            push_event(event_queue, "island_error", {
                "island_id": island.id,
                "error": str(result),
                "message": f"Island {island.id} failed: {result}",
            })
        elif not result:
            island.alive = False
            island.status = "error"

    alive_count = sum(1 for isl in islands if isl.alive)
    total_in = sum(isl.input_tokens for isl in islands)
    total_out = sum(isl.output_tokens for isl in islands)
    push_event(event_queue, "tokens_update", {
        "input_tokens": total_in,
        "output_tokens": total_out,
    })
    push_event(event_queue, "stage", {
        "stage": "improving_done",
        "message": f"Physics prior implementation done. {alive_count}/{len(islands)} islands successful.",
    })

    if alive_count == 0:
        push_event(event_queue, "error", {"message": "All islands failed during implementation."})
        push_event(event_queue, "done", {"message": "Tournament aborted — no islands survived."})
        return {"winner": None, "islands": [_island_summary(isl) for isl in islands], "priors": priors}

    # ─── Phase 2.5: Uniquify package names ────────────────────────
    push_event(event_queue, "stage", {
        "stage": "uniquify",
        "message": f"Renaming packages to unique names ({base_pkg} -> {base_pkg}_island_N)...",
    })

    for island in islands:
        if not island.alive:
            continue
        actual_pkg = _find_method_pkg_dir(island.path) or base_pkg
        ok = _uniquify_island_package(island, actual_pkg)
        if not ok:
            island.alive = False
            island.status = "error"
            island.error = "Package uniquification failed"
            push_event(event_queue, "island_error", {
                "island_id": island.id,
                "error": f"{actual_pkg} directory not found after agent improvement",
                "message": f"Island {island.id}: Package rename failed — eliminated",
            })
        else:
            push_event(event_queue, "island_log", {
                "island_id": island.id,
                "message": f"Renamed to {island.pkg_name}",
            })

    alive_count = sum(1 for isl in islands if isl.alive)
    if alive_count == 0:
        push_event(event_queue, "error", {"message": "All islands failed during package rename."})
        push_event(event_queue, "done", {"message": "Tournament aborted."})
        return {"winner": None, "islands": [_island_summary(isl) for isl in islands], "priors": priors}

    # ─── Phase 3: Tournament Rounds (same as standard invent) ─────
    for round_idx, round_cfg in enumerate(TOURNAMENT_ROUNDS):
        max_iters = round_cfg["max_iters"]
        target_survivors = round_cfg["survivors"]
        label = round_cfg["label"]

        alive_islands = [isl for isl in islands if isl.alive]
        if len(alive_islands) <= target_survivors:
            push_event(event_queue, "round_skip", {
                "round": round_idx + 1,
                "message": f"Skipping {label} — only {len(alive_islands)} alive (need >{target_survivors})",
            })
            continue

        push_event(event_queue, "round_start", {
            "round": round_idx + 1,
            "label": label,
            "alive_count": len(alive_islands),
            "max_iters": max_iters,
            "message": f"{label}: Training {len(alive_islands)} islands for {max_iters} iters across {len(gpus)} GPU(s)...",
        })

        # Train islands in GPU-sized batches (e.g. 4 GPUs → 4 at a time)
        await _train_islands_batched(
            alive_islands, gpus, max_iters, data_path, output_dir,
            config.conda_env, scene, round_idx, event_queue,
            config=config, model=model,
        )

        # Rank and eliminate
        alive_islands = [isl for isl in islands if isl.alive]
        alive_islands.sort(key=lambda isl: isl.psnr, reverse=True)

        survivors = alive_islands[:target_survivors]
        eliminated = alive_islands[target_survivors:]

        for isl in eliminated:
            isl.alive = False
            isl.status = "eliminated"
            push_event(event_queue, "island_eliminated", {
                "island_id": isl.id,
                "psnr": isl.psnr,
                "round": round_idx + 1,
                "message": f"Island {isl.id} ({isl.strategy['name']}): ELIMINATED (PSNR={isl.psnr:.2f})",
            })

        ranking_msg = " > ".join(
            f"#{isl.id}({isl.strategy.get('emoji', '?')}){isl.psnr:.2f}"
            for isl in alive_islands
        )
        push_event(event_queue, "round_result", {
            "round": round_idx + 1,
            "ranking": [
                {"island_id": isl.id, "psnr": isl.psnr, "strategy": isl.strategy["name"], "alive": isl.alive}
                for isl in alive_islands
            ],
            "survivors": [isl.id for isl in survivors],
            "eliminated": [isl.id for isl in eliminated],
            "message": f"{label} results: {ranking_msg}",
        })

    # ─── Declare Winner ──────────────────────────────────────────
    alive_islands = [isl for isl in islands if isl.alive]
    winner = None
    if alive_islands:
        winner = max(alive_islands, key=lambda isl: isl.psnr)
        winner.status = "winner"
        push_event(event_queue, "winner", {
            "island_id": winner.id,
            "strategy": winner.strategy["name"],
            "psnr": winner.psnr,
            "improvement_summary": winner.improvement_summary,
            "method_path": str(winner.path),
            "message": (
                f"WINNER: Island {winner.id} ({winner.strategy['name']}) "
                f"with PSNR = {winner.psnr:.2f} dB"
            ),
        })
    else:
        push_event(event_queue, "error", {
            "message": "No islands survived the tournament.",
        })

    result = {
        "winner": _island_summary(winner) if winner else None,
        "islands": [_island_summary(isl) for isl in islands],
        "priors": priors,
        "scene_description": scene_description,
        "rounds": [{"label": r["label"], "max_iters": r["max_iters"]} for r in TOURNAMENT_ROUNDS],
    }

    total_in = sum(isl.input_tokens for isl in islands)
    total_out = sum(isl.output_tokens for isl in islands)
    push_event(event_queue, "tokens_update", {
        "input_tokens": total_in,
        "output_tokens": total_out,
    })
    push_event(event_queue, "done", {
        "message": "Scene-driven evolutionary tournament complete!",
        "result": result,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
    })

    # Save results
    results_file = workspace / "tournament_results.json"
    results_file.write_text(json.dumps(result, indent=2))

    return result
