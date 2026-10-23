"""
Sub-project B agent adapters.

Two small Claude agents invoked inline by evolution.py:

    run_verbal_gradient_agent  — Sonnet. Given (winner_blocks, loser_blocks,
        PSNRs, rationales), returns a 2-3 sentence lesson with a JSON tail.
        Cheap: one Sonnet call per "surprising" loss per generation (≤4).

    run_negative_ideator_agent — Opus. Given the final result of a failed
        cycle (best PSNR < baseline, plus all rationales), writes a fresh
        seed implementation into a new directory and returns its path.
        Called only on re-ideation, so ≤ max_re_ideations times per run.

Both adapters follow the discover.py pattern: run claude_agent_sdk.query()
in the current event loop, stream assistant messages to the event sink,
and parse a structured output at the end.
"""
from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
)

from splatify.splatify_config import get_claude_auth_env
from splatify.elo_gradients import (
    ScoredCandidate, VerbalGradient,
    build_gradient_prompt, parse_gradient_output,
)
from splatify.evolve_blocks import extract_all_blocks, find_marked_files


# ═══════════════════════════════════════════════════════════════════
# VERBAL GRADIENT AGENT
# ═══════════════════════════════════════════════════════════════════

VERBAL_GRADIENT_SYSTEM = (
    "You are a 3D Gaussian Splatting research assistant performing "
    "post-hoc analysis. Your job is to read two candidate method diffs "
    "and explain in 2-3 sentences WHY the winner beat the loser. Focus on "
    "the mechanism (e.g. 'the winner's loss includes a depth regularizer "
    "that prevents floater blobs in underconstrained views'), not on the "
    "PSNR number. End every response with exactly one JSON line: "
    '{"hooks": [...], "lesson": "..."}'
)


def _read_blocks_from_dir(impl_path: Path) -> dict[str, str]:
    """Read all EVOLVE blocks present under an implementation directory."""
    merged: dict[str, str] = {}
    for py in find_marked_files(impl_path):
        try:
            blocks = extract_all_blocks(py.read_text())
        except Exception:
            continue
        for hook, body in blocks.items():
            # If multiple files contribute a hook, keep the first.
            merged.setdefault(hook, body)
    return merged


async def run_verbal_gradient_agent(
    winner: ScoredCandidate,
    loser: ScoredCandidate,
    winner_path: Path,
    loser_path: Path,
    event_sink=None,
    timeout_s: float = 120.0,
) -> VerbalGradient | None:
    """Invoke the verbal gradient agent; returns a parsed VerbalGradient."""
    winner_blocks = _read_blocks_from_dir(winner_path)
    loser_blocks = _read_blocks_from_dir(loser_path)

    prompt = (
        VERBAL_GRADIENT_SYSTEM + "\n\n"
        + build_gradient_prompt(winner, loser, winner_blocks, loser_blocks)
    )

    opts = ClaudeAgentOptions(
        allowed_tools=[],  # pure reasoning, no tool use needed
        permission_mode="acceptEdits",
        max_turns=2,
        model="claude-sonnet-4-6",
        env=get_claude_auth_env(),
    )

    text = ""
    try:
        async def _run():
            nonlocal text
            async for message in query(prompt=prompt, options=opts):
                if isinstance(message, AssistantMessage):
                    for block in getattr(message, "content", []):
                        if isinstance(block, TextBlock):
                            t = getattr(block, "text", "")
                            if t:
                                text += t
                                if event_sink is not None:
                                    event_sink.put({
                                        "type": "gradient_log",
                                        "message": t[:200],
                                    })
                elif isinstance(message, ResultMessage):
                    pass
        await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError:
        if event_sink is not None:
            event_sink.put({
                "type": "gradient_error",
                "message": "Verbal gradient agent timed out",
            })
        return None
    except Exception as e:
        if event_sink is not None:
            event_sink.put({
                "type": "gradient_error",
                "message": f"Verbal gradient agent error: {e}",
            })
        return None

    return parse_gradient_output(text)


# ═══════════════════════════════════════════════════════════════════
# NEGATIVE IDEATOR AGENT
# ═══════════════════════════════════════════════════════════════════

NEGATIVE_IDEATOR_SYSTEM = (
    "You are the negative-ideator for a 3D Gaussian Splatting research "
    "pipeline. A full evolutionary cycle just finished BELOW baseline. "
    "Your job is to read the failed rationales, identify the common "
    "failure pattern, and propose a FRESH seed implementation that "
    "avoids those specific pitfalls. "
    "Do NOT repeat ideas from the failed cycle. "
    "Use Read/Glob/Grep to inspect the best implementation, then use "
    "Write to create a new method package in the target directory. "
    "When done, emit a single JSON line on its own: "
    '{"new_impl_path": "<absolute path>"}'
)


def _build_negative_ideator_prompt(prior_result: dict, seed_dir: Path) -> str:
    failed_rationales: list[str] = []
    for g in prior_result.get("generations", []) or []:
        for c in g.get("candidates", []) or []:
            summary = c.get("summary") or c.get("best_summary") or ""
            if summary:
                failed_rationales.append(
                    f"- gen{g.get('index','?')}-i{c.get('id','?')} "
                    f"(psnr={c.get('psnr','?')}): {summary}"
                )

    bp = prior_result.get("baseline_psnr", 0)
    bb = prior_result.get("final_best_psnr", 0)
    path_hint = prior_result.get("best_implementation_path", "<unknown>")

    return (
        NEGATIVE_IDEATOR_SYSTEM + "\n\n"
        + f"BASELINE PSNR: {bp:.2f}\n"
        + f"BEST ACHIEVED:  {bb:.2f}  (deficit: {bb - bp:+.2f} dB)\n"
        + f"BEST IMPL PATH: {path_hint}\n"
        + f"NEW SEED TARGET: {seed_dir}\n\n"
        + "FAILED RATIONALES (avoid these directions):\n"
        + "\n".join(failed_rationales[:30])
        + "\n\nTASK:\n"
        + "1. Read the best-so-far implementation to understand its structure.\n"
        + "2. Diagnose what the failed rationales have in common (e.g. all\n"
        + "   touched the wrong hook, all introduced numerical instability,\n"
        + "   all required hyperparameters we couldn't tune).\n"
        + "3. Copy the baseline (splatfacto structure) into NEW SEED TARGET\n"
        + "   and implement ONE new idea that addresses that diagnosis —\n"
        + "   structurally different from the failed directions.\n"
        + "4. End your message with the JSON line described above.\n"
    )


async def run_negative_ideator_agent(
    prior_result: dict,
    workspace: Path,
    event_queue=None,
    timeout_s: float = 900.0,
) -> str | None:
    """Run the negative ideator; returns the path of the new seed impl."""
    workspace = Path(workspace)
    seed_dir = workspace / "reideations" / f"cycle_{int(prior_result.get('num_generations', 0)) + 1}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    # Bootstrap the new seed by copying the best impl (if any) as a
    # starting point the agent can rewrite.
    best_path = prior_result.get("best_implementation_path")
    if best_path and Path(best_path).exists():
        try:
            if not any(seed_dir.iterdir()):
                shutil.copytree(best_path, seed_dir, dirs_exist_ok=True)
        except Exception:
            pass

    prompt = _build_negative_ideator_prompt(prior_result, seed_dir)

    opts = ClaudeAgentOptions(
        cwd=str(seed_dir),
        allowed_tools=["Read", "Write", "Bash", "Glob", "Grep"],
        permission_mode="acceptEdits",
        max_turns=40,
        model="claude-opus-4-6",
        env=get_claude_auth_env(),
    )

    if event_queue is not None:
        event_queue.put({
            "type": "negative_ideator_start",
            "seed_dir": str(seed_dir),
            "message": "Negative ideator running (Opus)",
        })

    result_path: str | None = None
    text_buf = ""

    try:
        async def _run():
            nonlocal text_buf
            async for message in query(prompt=prompt, options=opts):
                if isinstance(message, AssistantMessage):
                    for block in getattr(message, "content", []):
                        if isinstance(block, TextBlock):
                            t = getattr(block, "text", "")
                            if t:
                                text_buf += t
                                if event_queue is not None:
                                    event_queue.put({
                                        "type": "negative_ideator_log",
                                        "message": t[:300],
                                    })
                elif isinstance(message, ResultMessage):
                    pass
        await asyncio.wait_for(_run(), timeout=timeout_s)
    except asyncio.TimeoutError:
        if event_queue is not None:
            event_queue.put({
                "type": "negative_ideator_error",
                "message": "Negative ideator timed out",
            })
        return None
    except Exception as e:
        if event_queue is not None:
            event_queue.put({
                "type": "negative_ideator_error",
                "error": str(e),
                "message": f"Negative ideator failed: {e}",
            })
        return None

    # Parse the trailing JSON line.
    for line in reversed(text_buf.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                p = data.get("new_impl_path")
                if p and Path(p).exists():
                    result_path = p
                    break
            except json.JSONDecodeError:
                continue

    # Fallback: if the agent wrote into seed_dir but didn't emit JSON,
    # use seed_dir if it now contains a method package.
    if result_path is None:
        pkg_dirs = [
            d for d in seed_dir.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
            and not d.name.startswith("__")
        ]
        if pkg_dirs:
            result_path = str(seed_dir)

    if event_queue is not None:
        event_queue.put({
            "type": "negative_ideator_done",
            "new_impl_path": result_path,
            "message": (
                f"Negative ideator produced new seed: {result_path}"
                if result_path else
                "Negative ideator produced no usable seed"
            ),
        })
    return result_path
