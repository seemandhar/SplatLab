#!/usr/bin/env python3
"""
Read TensorBoard event files and extract PSNR training curve.

Usage:
    python read_tb.py <logdir>
    python read_tb.py /path/to/outputs/garden/method-name/2026-03-09_152119/

Outputs a JSON summary of the PSNR curve with sampled points across training,
plus a text summary for quick review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def extract_psnr(logdir: str, max_samples: int = 50) -> dict:
    """Extract PSNR curve from TensorBoard event files.

    Returns dict with:
        - tag: the scalar tag name
        - total_steps: total number of logged steps
        - final_psnr: last PSNR value
        - max_psnr: peak PSNR value and step
        - samples: list of (step, psnr) sampled across training
        - summary: human-readable text summary
    """
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})  # 0 = load all
    ea.Reload()

    scalars = ea.Tags().get("scalars", [])
    psnr_tags = [t for t in scalars if "psnr" in t.lower()]

    if not psnr_tags:
        return {
            "error": f"No PSNR scalar found. Available tags: {scalars[:20]}",
            "all_tags": scalars,
        }

    # Prefer eval PSNR over train PSNR for quality assessment
    tag = psnr_tags[0]
    for t in psnr_tags:
        if "eval" in t.lower() and "all" not in t.lower():
            tag = t
            break
    events = ea.Scalars(tag)
    n = len(events)

    if n == 0:
        return {"error": f"Tag '{tag}' has no data points."}

    # Sample evenly across training
    if n <= max_samples:
        indices = list(range(n))
    else:
        step = n / max_samples
        indices = [int(i * step) for i in range(max_samples)]
        if indices[-1] != n - 1:
            indices.append(n - 1)

    samples = [(events[i].step, round(events[i].value, 4)) for i in indices]

    # Find max and final
    max_event = max(events, key=lambda e: e.value)
    final_event = events[-1]

    # Detect issues in the curve
    issues = []

    # Check if PSNR is still rising at end (might need more training)
    if n >= 10:
        last_10_pct = events[int(n * 0.9):]
        start_val = last_10_pct[0].value
        end_val = last_10_pct[-1].value
        if end_val - start_val > 0.5:
            issues.append(f"PSNR still rising in last 10% of training (+{end_val - start_val:.2f}dB)")

    # Check if PSNR dropped significantly from peak
    if max_event.value - final_event.value > 1.0:
        issues.append(
            f"PSNR dropped {max_event.value - final_event.value:.2f}dB from peak "
            f"(peak={max_event.value:.2f} at step {max_event.step}, final={final_event.value:.2f})"
        )

    # Check if final PSNR is unreasonably low for garden scene
    if final_event.value < 20.0:
        issues.append(f"Final PSNR is very low ({final_event.value:.2f}dB) — possible training failure")

    # Check for NaN or zero
    nan_count = sum(1 for e in events if e.value != e.value or e.value == 0)
    if nan_count > 0:
        issues.append(f"{nan_count} NaN/zero values detected in PSNR curve")

    # Build text summary
    lines = [
        f"PSNR Training Curve Summary (using: {tag})",
        f"  All PSNR tags found: {psnr_tags}",
        f"  Log dir: {logdir}",
        f"  Total logged steps: {n}",
        f"  Step range: {events[0].step} → {final_event.step}",
        f"  Final PSNR: {final_event.value:.2f} dB (step {final_event.step})",
        f"  Peak PSNR:  {max_event.value:.2f} dB (step {max_event.step})",
        "",
        "PSNR curve (sampled):",
    ]

    # Print a compact text curve
    for step, psnr in samples[::max(1, len(samples) // 20)]:
        bar_len = max(0, int((psnr - 15) * 2))  # scale bar relative to 15dB
        bar = "█" * bar_len
        lines.append(f"  step {step:>6d}: {psnr:>7.2f} dB {bar}")

    if issues:
        lines.append("")
        lines.append("ISSUES DETECTED:")
        for issue in issues:
            lines.append(f"  ⚠ {issue}")
    else:
        lines.append("")
        lines.append("No issues detected — training curve looks healthy.")

    summary = "\n".join(lines)

    return {
        "tag": tag,
        "total_steps": n,
        "step_range": [events[0].step, final_event.step],
        "final_psnr": round(final_event.value, 4),
        "final_step": final_event.step,
        "max_psnr": round(max_event.value, 4),
        "max_psnr_step": max_event.step,
        "issues": issues,
        "samples": samples,
        "summary": summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract PSNR from TensorBoard logs")
    parser.add_argument("logdir", help="Path to the training output directory containing event files")
    parser.add_argument("--json", action="store_true", help="Output as JSON instead of text")
    parser.add_argument("--max-samples", type=int, default=50, help="Max sampled points")
    args = parser.parse_args()

    result = extract_psnr(args.logdir, max_samples=args.max_samples)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if "error" in result:
            print(f"ERROR: {result['error']}")
            sys.exit(1)
        print(result["summary"])


if __name__ == "__main__":
    main()
