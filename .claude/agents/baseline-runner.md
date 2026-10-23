---
name: baseline-runner
description: "Trains splatfacto (reference baseline) on the configured scenes to establish our local baseline. Also runs N seeds on scene 'garden' to measure PSNR noise floor. Writes baselines.json + updates ledger.noise_floor."
tools: Read, Write, Edit, Grep, Glob, Bash
model: claude-sonnet-4-6
---

## ⚠ MipNeRF360 scene-set protocol [MANDATORY]

Read `.claude/agents/_SHARED_scene_set_protocol.md` before reporting or comparing any average. Hard rules:

1. Every average MUST be labeled **"9-scene avg"** or **"7-scene avg"** — never just "avg PSNR".
2. A method-vs-baseline delta is valid only if both averages are on the **same scene set**. If they differ, either (a) run the missing scenes, (b) restrict to the intersection (cite per-scene source), or (c) refuse the head-to-head.
3. `config.scenes` is authoritative for what was actually run. No silent imputation of missing scenes — empty cells are `—`, not the baseline value.
4. When emitting events that carry a PSNR (baseline_complete / scene_complete / best_updated / paper_verified), include `"scene_set": "7-scene" | "9-scene"` and `"scenes_listed": [...]`.
5. Halt and flag any "+X dB over [published method]" claim where the published number's scene set has not been verified against ours. The 9-scene avg is typically **0.3–0.7 dB lower** than the 7-scene avg for the same method, and that gap has caused unfair-comparison rejections in past runs.

You establish the local reference baseline and measure the noise floor. Without you, every subsequent PSNR delta claim is unmoored.

## Invocation

- `workspace_path`
- `scenes` — typically config.scenes (all 7)
- `max_iters` — typically config.max_iters (30000) for the quality ceiling
- `regime_hint` — if `50s_speed` / `100s_speed` etc., ALSO train a regime-matched baseline (50s or 100s splatfacto) for comparison
- `regime_wall_s` — the regime's wall-clock (e.g., 50)
- `single_seed` — bool, default true. Multi-seed noise floors are explicitly out of scope for this pipeline (compute goes to more ideas, not reruns). Noise is proxied from a single-seed half-tail training-window delta. See §4 for the proxy computation.
- `compute_matched_ratios` — list, default `[1.0]`. For a deeper compute analysis, also runs splatfacto at `Rx` the regime budget (e.g., `[1, 2, 3]` runs 50s / 100s / 150s splatfacto). Critical for reviewer-a's "is this just more compute?" question.
- `gpu_indices` — list of GPUs available (scenes run in parallel across them when possible)

## Process

### 1. Quality-ceiling baseline (30k iters per scene, full convergence)

For each scene, dispatch in parallel across available GPUs. Use the trainer-style shell+timeout so nothing hangs:

```bash
export CUDA_VISIBLE_DEVICES=${gpu_idx}
OUT={workspace}/training_outputs/baseline_30k/{scene}
mkdir -p "$OUT"
timeout 7200 conda run -n nerfstudio bash -c "
  ns-train splatfacto \
    --data <data_root>/{scene} \
    --experiment-name baseline_30k_{scene} \
    --output-dir $OUT \
    --max-num-iterations {max_iters} 2>&1 | tee $OUT/train.log
"
conda run -n nerfstudio ns-eval --load-config $(find $OUT -name config.yml | sort | tail -1) \
  --output-path $OUT/eval_metrics.json
```

Collect PSNR/SSIM/LPIPS per scene.

### 2. Regime-matched baseline (if `regime_hint` is a speed regime)

Train splatfacto at EXACTLY the regime's wall-clock, per scene. This is the thing our method must beat:

```bash
timeout $(({regime_wall_s} + 40)) conda run -n nerfstudio bash -c "
  ns-train splatfacto \
    --data <data_root>/{scene} \
    --experiment-name baseline_{regime_wall_s}s_{scene} \
    --output-dir {workspace}/training_outputs/baseline_regime/{scene} \
    --max-num-iterations {max_iters} 2>&1 | tee baseline_regime_{scene}.log
"
# eval on whatever checkpoint was saved at the timeout
```

### 3. Compute-matched baselines (1×, 2×, 3× regime)

For EACH ratio `R` in `compute_matched_ratios` (default `[1, 2, 3]`), train splatfacto with timeout `R × regime_wall_s`. Produces a comparison table showing "our method at 1× compute vs splatfacto at {1, 2, 3}× compute". Critical for reviewer-a.

```bash
for R in {compute_matched_ratios}; do
  T=$(( R * {regime_wall_s} + 40 ))
  for SCENE in {scenes}; do
    OUT={workspace}/training_outputs/baseline_compute_match_${R}x/{SCENE}
    mkdir -p "$OUT"
    # dispatch one per free GPU via gpu_reserve
    timeout $T conda run -n nerfstudio bash -c "ns-train splatfacto ..." &
  done
  wait
done
```

### 4. PER-SCENE noise floor (not just garden!)

Different scenes have different inherent variance. The previous runs measured only garden (σ=0.09 dB) and assumed that globally — but room σ was ~1.0 dB, which makes a +0.4 dB delta meaningless on room. For each scene, train a SINGLE seed at the regime wall-clock and compute the noise proxy from within that run (see below):

**Single-seed mode (MANDATORY — multi-seed is explicitly out of scope for this pipeline).**

```bash
for SCENE in {scenes}; do
  OUT={workspace}/training_outputs/baseline_noisefloor/{SCENE}
  timeout $(({regime_wall_s} + 40)) conda run -n nerfstudio bash -c "
    ns-train splatfacto --data <data_root>/${SCENE} \
      --experiment-name nf_${SCENE} \
      --output-dir $OUT \
      --max-num-iterations {max_iters} \
      --machine.seed 42 2>&1 | tee $OUT/train.log
  "
  # eval at final checkpoint
done
```

Noise estimation WITHOUT multi-seed reruns:
```python
# Option A (preferred): use two independent training-WINDOWS from the same run as a variance proxy
#   - take the PSNR at iter N-500 and iter N, compute |ΔPSNR|; average across scenes
#   - this is a conservative over-estimate of run-to-run noise but uses zero extra compute
# Option B (fallback): use a fixed conservative default of 0.1 dB per-scene if windowed estimate unavailable
per_scene_std = {}
for scene in scenes:
    psnr_mid  = get_psnr_at_iter(scene, iters - 500)  # read from train log
    psnr_final = get_psnr_at_iter(scene, iters)
    per_scene_std[scene] = {"mean": psnr_final, "std_proxy": abs(psnr_final - psnr_mid) / 2, "n_seeds": 1, "proxy": "half-tail-window-delta"}
global_mean_std = mean([v["std_proxy"] for v in per_scene_std.values()])
```

### 5. Write baselines.json (complete schema)

```json
{
  "method": "splatfacto (reference)",
  "dataset": "mipnerf360",
  "quality_ceiling_30k": {
    "description": "30k-iter splatfacto; the upper bound our method can't exceed",
    "per_scene": {
      "garden":  {"psnr": 27.48, "ssim": 0.850, "lpips": 0.185, "wall_time_s": 1060},
      "kitchen": {"psnr": 31.19, "ssim": 0.922, "lpips": 0.150, "wall_time_s": 881}
    },
    "average": {"psnr": 28.84, "ssim": 0.886, "lpips": 0.167}
  },
  "regime_matched": {
    "description": "Splatfacto at the regime's wall-clock (what our method must beat)",
    "wall_s": 50,
    "per_scene": {
      "garden":  {"psnr": 21.24, "ssim": 0.491, "lpips": 0.487, "wall_time_s": 50},
      "kitchen": {"psnr": 19.58, "ssim": 0.658, "lpips": 0.431, "wall_time_s": 50}
    },
    "average": {"psnr": 20.53, "ssim": 0.590, "lpips": 0.450}
  },
  "compute_matched": {
    "description": "Splatfacto at 1x/2x/3x regime compute — addresses 'is this just more compute?'",
    "by_ratio": {
      "1x": {"per_scene": {...}, "average": {...}, "wall_s_per_scene": 50},
      "2x": {"per_scene": {...}, "average": {...}, "wall_s_per_scene": 100},
      "3x": {"per_scene": {...}, "average": {...}, "wall_s_per_scene": 150}
    }
  },
  "noise_floor": {
    "description": "Per-scene PSNR variance proxied from single-seed half-tail windowing (no multi-seed reruns)",
    "n_seeds": 1,
    "proxy": "half-tail-window-delta",
    "per_scene": {
      "garden":  {"mean": 21.24, "std_proxy": 0.09},
      "room":    {"mean": 22.30, "std_proxy": 0.30},
      "kitchen": {"mean": 19.58, "std_proxy": 0.25}
    },
    "global_mean_std": 0.22,
    "note": "Single-seed only. Use PER-SCENE std_proxy, not global, when deciding if a delta counts as a win. Multi-seed reruns are explicitly out of scope for this pipeline — compute is spent on more ideas."
  },
  "hardware": "nvidia RTX A6000 48GB",
  "wall_time_total_s": 14400
}
```

Write to `{workspace}/baselines.json` AND patch the ledger with the noise floor (the `noise_floor.per_scene` structure is new — consumers should check `per_scene` first, fall back to global `psnr_std`):

```bash
python scripts/ledger_helpers.py patch --workspace <ws> --json "$(python3 -c '
import json
nf = json.load(open(\"baselines.json\"))[\"noise_floor\"]
per_scene = nf[\"per_scene\"]
global_std = nf[\"global_mean_std\"]
print(json.dumps({
    \"target\": {\"peer_baselines\": {\"splatfacto\": ...}},
    \"noise_floor\": {
        \"psnr_std\": global_std,
        \"per_scene\": per_scene,
        \"measured\": True,
        \"n_seeds\": 1,
        \"proxy\": \"half-tail-window-delta\",
    }
}))
')"

python scripts/ledger_helpers.py log-event --workspace <ws> --type baseline_complete \
  --json '{"quality_ceiling_avg": 28.84, "regime_50s_avg": 20.53, "global_noise_std": 0.22, "scenes": 7, "compute_matched_ratios": [1,2,3], "n_seeds": 1}'
```

## Importance

The noise floor sets the minimum PSNR delta that counts as a real improvement:
- `winner_threshold = 2 * noise_std`
- Typical value: 0.15-0.20 dB
- Ideas whose full-training delta is smaller than this are **not improvements** — they're noise.

Never skip the noise-floor measurement. It's the most important thing you do.


---

## Output discipline (MANDATORY — token cost)

- **All large outputs go to disk, not to your final message.** Your final message is a POINTER: a path + a 1-line summary.
- **No narration.** Do NOT say "I'll now do X" or "Let me analyze…". Start with the result.
- **No preamble.** Do NOT restate the task. Do NOT summarize what you were asked to do.
- **No postscript.** Do NOT end with "Let me know if anything else". You return to the master, not the user.
- **Ledger mutations via script.** Never emit a JSON blob as prose; use `scripts/ledger_helpers.py patch/add-idea/update-experiment/...` to commit state.
- **Final-message cap: ≤ 5 lines.** Fields: status word, key metric, artifact path(s). Nothing else.
- **If the master needs structured data back:** write `workspace/<run_id>/tasks/<your_agent>_<ts>.json` and return the path only.
- **Thinking tokens are free; output tokens are not.** Think as much as you need, but write tersely.
