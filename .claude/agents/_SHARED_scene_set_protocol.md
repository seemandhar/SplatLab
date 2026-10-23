# MipNeRF360 scene-set protocol — READ BEFORE REPORTING ANY AVERAGE

This file is referenced by every agent that reports, compares, or verifies a PSNR/SSIM/LPIPS number on MipNeRF360. Violating it produces unfair comparisons that fail review.

## The two scene sets in the wild

- **9-scene (full)** — `bicycle, bonsai, counter, flowers, garden, kitchen, room, stump, treehill`
  - Used by: 3DGS-MCMC, Mip-Splatting, Scaffold-GS (most modern papers).
- **7-scene (INRIA / 3DGS original)** — drops `flowers` + `treehill`
  - `bicycle, bonsai, counter, garden, kitchen, room, stump`
  - Used by: vanilla 3DGS (Kerbl 2023), early follow-ups, and some efficiency-focused papers.

The two extra scenes (flowers, treehill) are the *hardest outdoor scenes*. A 7-scene average is **systematically higher** than a 9-scene average for the same method — typically by **+0.3 to +0.7 dB**. Comparing a 7-scene method number against a 9-scene baseline number (or vice versa) is an unfair comparison that will be caught in review.

## The hard rules

1. **Every reported average must be tagged with its scene set.** Use the exact phrase "9-scene avg" or "7-scene avg" in tables, captions, prose, and event payloads. Never just "avg PSNR".
2. **Apples-to-apples or nothing.** When comparing method A to method B, both must be reported on the **same scene set**. If the published number for B is 9-scene and we only ran A on 7, EITHER:
   - (a) run A on the missing scenes (`flowers`, `treehill`) and report 9-scene, OR
   - (b) recompute B's average on the same 7-scene subset (only if per-scene B numbers are available in the source paper / repo); cite the per-scene source explicitly, OR
   - (c) refuse to make the head-to-head claim and frame it as "on our 7-scene subset".
3. **`config.scenes` is authoritative for what was actually run.** Read it. Do not assume.
4. **No silent imputation.** If a scene was not trained for an idea, that scene's cell in the table is `—`, not the baseline value, not zero, not the per-method average.
5. **Capacity, protocol, and seed must match too.** Scene-set match is necessary but not sufficient. Also verify Gaussian budget (cap_max), eval protocol (every-8th vs nerfbaselines split), and seed count.

## Violations to flag

When you spot any of the following in your own output OR another agent's output, halt and report:
- An average labeled "MipNeRF360 avg" without "7-" or "9-" prefix.
- A method-vs-baseline delta where the two averages are computed over different scene sets.
- A claim of "+X dB over [published method]" without verifying the published number's scene set in the source paper / repo, AND verifying our number is on the same set.
- A table mixing 7-scene and 9-scene rows in the same column.
- Use of the unqualified phrase "all scenes" — say "all 7" or "all 9" explicitly.

## What to write into events.jsonl

When emitting baseline_complete, scene_complete, best_updated, paper_verified — always include:
```json
{"scene_set": "7-scene"|"9-scene", "scenes_listed": ["bicycle","..."], ...}
```
This makes downstream ledger consumers and the paper_quality_checker able to cross-check.
