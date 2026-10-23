#!/usr/bin/env python3
"""
Train and evaluate 3DGS methods on Mip-NeRF 360 or LLFF scenes.

Usage:
    python run_all_scenes.py                        # train + eval (mipnerf360)
    python run_all_scenes.py --dataset llff          # use LLFF dataset
    python run_all_scenes.py --eval-only             # skip training, just eval
    python run_all_scenes.py --gpu 1                 # use GPU 1
    python run_all_scenes.py --method fsgs           # specify method
    python run_all_scenes.py --num-views 12          # few-shot with 12 views
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed


# ─── Configuration ───────────────────────────────────────────────
DEFAULT_OUTPUT_ROOT = Path("outputs")

DATASETS = {
    "mipnerf360": {
        "data_root": Path("data/mipnerf360/"),
        "scenes": [
            "bicycle", "bonsai", "counter", "garden", "kitchen",
            "room", "stump", "flowers", "treehill",
        ],
    },
    "llff": {
        "data_root": Path("data/nerf_llff_data/"),
        "scenes": [
            "fern", "flower", "fortress", "horns",
            "leaves", "orchids", "room", "trex",
        ],
    },
}

def run_cmd(cmd: list[str], gpu: int = 0) -> int:
    """Run a command with the specified GPU."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["CC"] = "/usr/bin/gcc-11"
    env["CXX"] = "/usr/bin/g++-11"
    env["CUDAHOSTCXX"] = "/usr/bin/g++-11"
    print(f"\n$ {' '.join(cmd)}\n")
    result = subprocess.run(cmd, env=env)
    return result.returncode

def get_latest_config(scene: str, method: str, output_root: Path = DEFAULT_OUTPUT_ROOT) -> Path | None:
    """Find the most recent config.yml for a scene/method combo."""
    run_dir = output_root / scene / method
    if not run_dir.exists():
        return None
    # Get all timestamp directories sorted by name (newest last)
    runs = sorted(run_dir.iterdir(), key=lambda p: p.name, reverse=True)
    for run in runs:
        config = run / "config.yml"
        if config.exists():
            return config
    return None


def train_scene(scene: str, method: str, gpu: int, max_iters: int = 30000,
                num_views: int = -1, output_root: Path = DEFAULT_OUTPUT_ROOT,
                data_root: Path = None, viewer_port: int | None = None) -> bool:
    """Train a single scene. Returns True on success."""
    data_path = data_root / scene
    if not data_path.exists():
        print(f"  WARNING: Data not found at {data_path}, skipping.")
        return False

    cmd = [
        "ns-train", method,
        "--data", str(data_path),
        "--vis", "viewer+tensorboard",
        "--steps-per-eval-batch", "10000",
        "--max-num-iterations", str(max_iters),
        "--output-dir", str(output_root),
        "--viewer.quit-on-train-completion", "True",
    ]

    if viewer_port is not None:
        cmd += ["--viewer.websocket-port", str(viewer_port)]

    cmd += [
        "colmap",
        "--eval-mode", "interval",
    ]

    # Few-shot: must come after colmap subcommand
    if num_views > 0:
        cmd += ["--num-training-views", str(num_views)]

    ret = run_cmd(cmd, gpu=gpu)
    if ret != 0:
        print(f"  ERROR: Training failed for {scene} (exit code {ret})")
        return False
    return True


def count_gaussians(scene, method, output_root: Path = DEFAULT_OUTPUT_ROOT):
    """Count number of Gaussians from the latest checkpoint."""
    config = get_latest_config(scene, method, output_root)
    if config is None:
        return None
    ckpt_dir = config.parent / "nerfstudio_models"
    if not ckpt_dir.exists():
        return None
    ckpts = sorted(ckpt_dir.glob("step-*.ckpt"), reverse=True)
    if not ckpts:
        return None
    try:
        import torch
        ckpt = torch.load(str(ckpts[0]), map_location="cpu")
        pipeline = ckpt.get("pipeline", {})
        for key in pipeline:
            if "means" in key:
                return pipeline[key].shape[0]
    except Exception as e:
        print(f"  WARNING: Could not count gaussians for {scene}: {e}")
    return None


def eval_scene(scene: str, method: str, gpu: int, results_dir: Path,
               output_root: Path = DEFAULT_OUTPUT_ROOT) -> dict | None:
    """Evaluate a single scene. Returns the results dict or None."""
    config = get_latest_config(scene, method, output_root)
    if config is None:
        print(f"  WARNING: No config found for {scene}/{method}, skipping eval.")
        return None

    output_path = results_dir / f"{scene}.json"
    print(f"  Config: {config}")

    cmd = [
        "ns-eval",
        "--load-config", str(config),
        "--output-path", str(output_path),
    ]
    ret = run_cmd(cmd, gpu=gpu)
    if ret != 0:
        print(f"  ERROR: Eval failed for {scene} (exit code {ret})")
        return None

    # Read and return results
    if output_path.exists():
        with open(output_path) as f:
            data = json.load(f)
        # Add gaussian count from checkpoint
        num_gs = count_gaussians(scene, method, output_root)
        if num_gs is not None:
            if "results" in data:
                data["results"]["num_gaussians"] = num_gs
            else:
                data["num_gaussians"] = num_gs
        return data
    return None


def print_results_table(all_results: dict[str, dict], method: str, num_views: int):
    """Print a formatted results table and save summary."""
    view_str = f" ({num_views} views)" if num_views > 0 else " (all views)"
    print("\n" + "=" * 85)
    print(f"  Results: {method}{view_str}")
    print("=" * 85)
    print(f"  {'Scene':<12s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}  {'FPS':>8s}  {'#Gaussians':>12s}")
    print("-" * 85)

    metrics_agg = {"psnr": [], "ssim": [], "lpips": [], "fps": [], "num_rays_per_sec": [], "num_gaussians": []}

    for scene in sorted(all_results.keys()):
        if scene not in all_results or all_results[scene] is None:
            print(f"  {scene:<12s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>12s}")
            continue

        data = all_results[scene]
        r = data.get("results", data)

        psnr = r.get("psnr", 0)
        ssim = r.get("ssim", 0)
        lpips = r.get("lpips", 0)
        fps = r.get("fps", 0)
        num_gs = r.get("num_gaussians", r.get("num_points", 0))

        gs_str = f"{int(num_gs):,}" if num_gs else "N/A"
        print(f"  {scene:<12s}  {psnr:>8.2f}  {ssim:>8.4f}  {lpips:>8.4f}  {fps:>8.1f}  {gs_str:>12s}")

        for k in metrics_agg:
            if k == "num_gaussians":
                val = r.get("num_gaussians", r.get("num_points", None))
                if val is not None:
                    metrics_agg[k].append(val)
            elif k in r:
                metrics_agg[k].append(r[k])

    print("-" * 85)

    # Averages
    n = len(metrics_agg["psnr"])
    if n > 0:
        avg_psnr = sum(metrics_agg["psnr"]) / n
        avg_ssim = sum(metrics_agg["ssim"]) / n
        avg_lpips = sum(metrics_agg["lpips"]) / n if metrics_agg["lpips"] else 0
        avg_fps = sum(metrics_agg["fps"]) / n if metrics_agg["fps"] else 0
        avg_gs = sum(metrics_agg["num_gaussians"]) / n if metrics_agg["num_gaussians"] else 0
        gs_avg_str = f"{int(avg_gs):,}" if avg_gs else "N/A"
        print(f"  {'AVERAGE':<12s}  {avg_psnr:>8.2f}  {avg_ssim:>8.4f}  {avg_lpips:>8.4f}  {avg_fps:>8.1f}  {gs_avg_str:>12s}")
    print("=" * 85)

    return {k: sum(v) / len(v) for k, v in metrics_agg.items() if v}


def train_and_eval(scene, method, gpu, max_iters, num_views, eval_only, results_dir,
                   output_root: Path = DEFAULT_OUTPUT_ROOT, data_root: Path = None,
                   viewer_port: int | None = None):
    """Train and eval a single scene on a given GPU. Returns (scene, result_dict)."""
    view_tag = f" [{num_views}v]" if num_views > 0 else ""
    status = f"[GPU {gpu}] {scene}{view_tag}"
    if not eval_only:
        print(f"\n>>> {status}: TRAINING <<<")
        success = train_scene(scene, method, gpu, max_iters, num_views=num_views,
                              output_root=output_root, data_root=data_root,
                              viewer_port=viewer_port)
        if not success:
            print(f"  {status}: Training FAILED, skipping eval.")
            return (scene, None)

    print(f"\n>>> {status}: EVALUATING <<<")
    data = eval_scene(scene, method, gpu, results_dir, output_root=output_root)
    return (scene, data)


def main():
    parser = argparse.ArgumentParser(description="Train & eval 3DGS methods on Mip-NeRF 360 / LLFF")
    parser.add_argument("--eval-only", action="store_true", help="Skip training, only evaluate")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device IDs, comma-separated (e.g. 0,1,2)")
    parser.add_argument("--method", type=str, default="vod3dgs-l", help="Method name")
    parser.add_argument("--dataset", type=str, default="mipnerf360", choices=list(DATASETS.keys()),
                        help="Dataset to use (default: mipnerf360)")
    parser.add_argument("--scenes", nargs="+", default=None, help="Subset of scenes to run")
    parser.add_argument("--max-num-iterations", type=int, default=30000, help="Max training iterations")
    parser.add_argument("--num-views", type=int, default=-1,
                        help="Number of training views for few-shot (-1 = all views). "
                             "E.g. --num-views 3 for LLFF, --num-views 12, --num-views 24 for Mip-NeRF360")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for training runs (default: outputs)")
    parser.add_argument("--viewer-port", type=int, default=None,
                        help="Nerfstudio viewer websocket port (default: auto)")
    args = parser.parse_args()

    dataset_cfg = DATASETS[args.dataset]
    data_root = dataset_cfg["data_root"]
    method = args.method
    gpus = [int(g.strip()) for g in args.gpu.split(",")]
    scenes = args.scenes if args.scenes else dataset_cfg["scenes"]
    num_views = args.num_views
    OUTPUT_ROOT = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_ROOT

    # Include view count in results directory for few-shot runs
    if num_views > 0:
        results_dir = Path(f"results/{method}_{num_views}v")
    else:
        results_dir = Path(f"results/{method}")
    results_dir.mkdir(parents=True, exist_ok=True)
    max_iters = args.max_num_iterations

    print(f"Dataset:    {args.dataset}")
    print(f"Data root:  {data_root}")
    print(f"Method:     {method}")
    print(f"GPUs:       {gpus}")
    print(f"Scenes:     {scenes}")
    print(f"Views:      {num_views if num_views > 0 else 'all'}")
    print(f"Max iters:  {max_iters}")
    print(f"Results:    {results_dir}")
    print(f"Parallel:   {len(gpus)} workers")
    print(f"Time:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # ─── Train + Eval in parallel across GPUs ────────────────────
    all_results = {}

    with ProcessPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {}
        for i, scene in enumerate(scenes):
            gpu = gpus[i % len(gpus)]
            future = executor.submit(
                train_and_eval, scene, method, gpu, max_iters,
                num_views, args.eval_only, results_dir, OUTPUT_ROOT, data_root,
                viewer_port=args.viewer_port,
            )
            futures[future] = scene

        for future in as_completed(futures):
            scene = futures[future]
            try:
                _, data = future.result()
                all_results[scene] = data
            except Exception as e:
                print(f"  ERROR: {scene} failed with: {e}")
                all_results[scene] = None

    # ─── Summary ─────────────────────────────────────────────────
    averages = print_results_table(all_results, method, num_views)

    # Save summary JSON
    summary = {
        "method": method,
        "num_training_views": num_views if num_views > 0 else "all",
        "timestamp": datetime.now().isoformat(),
        "num_scenes": len([s for s in all_results if all_results[s] is not None]),
        "averages": averages,
        "per_scene": all_results,
    }
    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_path}")


if __name__ == "__main__":
    main()