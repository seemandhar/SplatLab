#!/usr/bin/env python3
"""Torch-based free-GPU probe — replaces nvidia-smi (NVML is broken on this box).

Outputs one line per ALLOWED gpu:  "<idx>, <free_mb>"  (comma-sep to match the
gpu_reserve.sh `IFS=,` parser).

Allowed set is controlled by env GPU_ALLOWED (comma/space separated indices).
Default "3,4,5,6,7" per operator: GPUs 0-2 are left for other users on this
shared box. Set GPU_ALLOWED="all" to probe every visible device.

Usage:
  python gpu_free.py            # CSV "idx, free_mb" lines for allowed GPUs
  python gpu_free.py --count <min_gb>   # print integer count of GPUs with >= min_gb free
"""
import os
import sys

try:
    import torch
except Exception as e:  # pragma: no cover
    sys.stderr.write(f"gpu_free: torch import failed: {e}\n")
    sys.exit(3)


def allowed_indices(ndev):
    raw = os.environ.get("GPU_ALLOWED", "3,4,5,6,7").strip()
    if raw.lower() == "all":
        return list(range(ndev))
    out = []
    for tok in raw.replace(",", " ").split():
        try:
            i = int(tok)
        except ValueError:
            continue
        if 0 <= i < ndev:
            out.append(i)
    return out or list(range(ndev))


def main():
    if not torch.cuda.is_available():
        sys.stderr.write("gpu_free: cuda not available\n")
        sys.exit(2)
    ndev = torch.cuda.device_count()
    idxs = allowed_indices(ndev)

    if len(sys.argv) >= 3 and sys.argv[1] == "--count":
        min_gb = float(sys.argv[2])
        c = 0
        for i in idxs:
            free, _ = torch.cuda.mem_get_info(i)
            if free / 2**30 >= min_gb:
                c += 1
        print(c)
        return

    for i in idxs:
        free, total = torch.cuda.mem_get_info(i)
        print(f"{i}, {int(free / 2**20)}")


if __name__ == "__main__":
    main()
