#!/usr/bin/env python3
"""Extract DriveStudio eval metrics from a run's log_dir into a clean per-region dict.

DriveStudio's tools/eval.py writes:
  {log_dir}/metrics/images_full_<time>.json   keys: image_metrics/full/<metric>
  {log_dir}/metrics/images_test_<time>.json   keys: image_metrics/test/<metric>
where <metric> in {psnr,ssim,lpips,occupied_psnr,occupied_ssim,masked_psnr,
masked_ssim,human_psnr,human_ssim,vehicle_psnr,vehicle_ssim}.

full = reconstruction (all frames); test = held-out NVS (test_image_stride>0).

Usage:
  python extract_ds_metrics.py <log_dir> [--out out.json]
Prints JSON: {"full": {...}, "test": {...}, "log_dir": ...}. Uses the LATEST
json of each split (by mtime). Exit 2 if no metrics files found.
"""
import sys, os, json, glob, argparse

METRICS = ["psnr", "ssim", "lpips", "occupied_psnr", "occupied_ssim",
           "masked_psnr", "masked_ssim", "human_psnr", "human_ssim",
           "vehicle_psnr", "vehicle_ssim"]


def latest(pattern):
    fs = glob.glob(pattern)
    return max(fs, key=os.path.getmtime) if fs else None


def parse_split(path):
    if not path or not os.path.exists(path):
        return {}
    with open(path) as f:
        d = json.load(f)
    out = {}
    for k, v in d.items():
        name = k.split("/")[-1]  # strip image_metrics/full|test/
        if name in METRICS:
            try:
                out[name] = float(v)
            except (TypeError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log_dir")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    mdir = os.path.join(a.log_dir, "metrics")
    res = {
        "log_dir": a.log_dir,
        "full": parse_split(latest(os.path.join(mdir, "images_full_*.json"))),
        "test": parse_split(latest(os.path.join(mdir, "images_test_*.json"))),
    }
    if not res["full"] and not res["test"]:
        sys.stderr.write(f"no metrics json under {mdir}\n")
        sys.exit(2)
    s = json.dumps(res, indent=2)
    if a.out:
        with open(a.out, "w") as f:
            f.write(s)
    print(s)


if __name__ == "__main__":
    main()
