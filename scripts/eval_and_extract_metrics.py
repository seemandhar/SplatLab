#!/usr/bin/env python3
"""
Post-training evaluation and metrics extraction for FastGS models.
Handles render.py + metrics.py + results extraction for the trainer.
"""

import json
import sys
import os
from pathlib import Path
import subprocess

def run_evaluation(output_dir, worktree):
    """
    Run render.py and metrics.py on a trained model.
    """
    output_dir = Path(output_dir)
    worktree = Path(worktree)

    # Step 1: Render test views
    print(f"[EVAL] Rendering test views to {output_dir}/test...")
    render_cmd = [
        "python", str(worktree / "render.py"),
        "--model_path", str(output_dir),
        "--skip_train",
        "--quiet"
    ]
    result = subprocess.run(render_cmd, cwd=str(worktree), capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[WARN] render.py exited with code {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")

    # Step 2: Compute metrics
    print(f"[EVAL] Computing metrics...")
    metrics_cmd = [
        "python", str(worktree / "metrics.py"),
        "-m", str(output_dir)
    ]
    result = subprocess.run(metrics_cmd, cwd=str(worktree), capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"[WARN] metrics.py exited with code {result.returncode}")
        print(result.stderr[-500:] if result.stderr else "")

    # Step 3: Extract results from results.json
    results_file = output_dir / "results.json"
    psnr = ssim = lpips = 0

    if results_file.exists():
        try:
            with open(results_file) as f:
                results = json.load(f)
            # Results structure: {"ours_<iters>": {"PSNR": X, "SSIM": Y, "LPIPS": Z}}
            # or just direct keys if merged
            for key, val in results.items():
                if isinstance(val, dict) and "PSNR" in val:
                    psnr = float(val.get("PSNR", 0))
                    ssim = float(val.get("SSIM", 0))
                    lpips = float(val.get("LPIPS", 0))
                    break
            print(f"[EVAL] Extracted metrics: PSNR={psnr:.3f}, SSIM={ssim:.3f}, LPIPS={lpips:.3f}")
        except Exception as e:
            print(f"[WARN] Failed to parse results.json: {e}")

    # Step 4: Count gaussians
    n_gaussians = 0
    ply_path = output_dir / "point_cloud" / "iteration_30000" / "point_cloud.ply"
    if ply_path.exists():
        try:
            with open(ply_path) as f:
                for line in f:
                    if "element vertex" in line:
                        n_gaussians = int(line.split()[-1])
                        break
            print(f"[EVAL] Counted {n_gaussians} gaussians")
        except:
            pass

    return psnr, ssim, lpips, n_gaussians

def main():
    if len(sys.argv) < 4:
        print("Usage: eval_and_extract_metrics.py <output_dir> <worktree> <exp_id> <scene>")
        sys.exit(1)

    output_dir = sys.argv[1]
    worktree = sys.argv[2]
    exp_id = sys.argv[3]
    scene = sys.argv[4]

    output_dir = Path(output_dir)

    # Check if wall_time exists
    wall_time_file = output_dir / "wall_time_s.txt"
    wall_time = 0
    if wall_time_file.exists():
        try:
            wall_time = int(wall_time_file.read_text().strip())
        except:
            pass

    # Run evaluation
    psnr, ssim, lpips, n_gaussians = run_evaluation(str(output_dir), worktree)

    # Determine status
    status = "timed_out" if wall_time >= 4498 else "done"

    # Create scene_metrics.json
    metrics = {
        "exp_id": exp_id,
        "scene": scene,
        "psnr": psnr,
        "ssim": ssim,
        "lpips": lpips,
        "n_gaussians_final": n_gaussians,
        "wall_time_s": wall_time,
        "status": status
    }

    metrics_file = output_dir / "scene_metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"[RESULT] Metrics written to {metrics_file}")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
