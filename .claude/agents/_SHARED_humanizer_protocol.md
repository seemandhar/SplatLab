# Humanizer protocol — READ BEFORE WRITING OR FINALIZING ANY SECTION

This file is referenced by every agent that writes or revises a `.tex` section
of the paper. The full skill lives at `.claude/skills/humanizer/SKILL.md`; this
file is the condensed actionable checklist.

The reader who matters is a CVPR program committee member. They read 8-12
papers a weekend. The first thing they do with a suspect paper is feed the
abstract to a detector. The second is read it aloud. If your section trips
either of these, the paper is dead before the figures are looked at.

## Read this file every time you write a section

Apply the gate AFTER drafting the prose, BEFORE writing the file to disk.
Two passes:

1. **Pattern pass** — scan for each entry in the catalog below; rewrite hits.
2. **Read-aloud pass** — mentally read the section. If a sentence sounds like
   a press release, a tutorial, a chatbot turn, or a results dump, rewrite it.

The full pattern catalog with worked examples is in
`.claude/skills/humanizer/SKILL.md`. This file is the cheat-sheet.

## The hard bans (paper-quality-checker greps these)

### Banned vocabulary
delve, tapestry, landscape (abstract sense), pivotal, crucial, foster /
fostering, showcase, testament, navigate (abstract), leverage (verb), realm,
embark, underscore (verb), multifaceted, nuanced, comprehensive, robust
(throwaway), intricate / intricacies, cornerstone, paradigm, synergy, holistic,
streamline, cutting-edge, groundbreaking, vibrant, breathtaking, nestled,
bustling, moreover, furthermore, notably, importantly, interestingly,
"it is worth noting that", "it is important to note that", actually,
additionally, align with, enduring, enhance, garner, highlight (verb),
interplay, key (adjective), valuable.

### Banned openers
- "In the realm of…"
- "In today's rapidly evolving…"
- "In order to…" → "To…"
- "This section will discuss…" → just discuss it
- "As we can see from…" → cite the figure directly
- "Recent advances in X have shown…"
- "With the rise of…"

### Banned sentence patterns
- **Copula avoidance.** "X serves as Y" / "X stands as Y" / "X represents Y"
  / "X marks Y" / "X functions as Y" — use is / are.
- **Negative parallelism.** "Not only X, but also Y" / "It's not just X;
  it's Y" / "X without Y" — make the positive claim directly.
- **Superficial -ing tails.** Sentences ending with a present-participle
  clause that adds no information ("…, reflecting the broader trend toward
  perceptual losses", "…, ensuring that the model remains tractable",
  "…, contributing to improved performance"). Cut the tail.
- **Rule-of-three padding.** Lists of exactly three near-synonyms
  ("streamlining processes, enhancing collaboration, and fostering
  alignment"). Pick the one that's true.
- **False ranges.** "from X to Y" where X and Y are not on a meaningful
  scale ("from per-scene retraining to feed-forward inference, from dense
  supervision to unsupervised priors"). Pick one half.
- **Vague attributions.** "researchers have shown" / "experts argue" /
  "recent work" / "the community has noted". Cite specifically with
  `\citet{}` or remove the claim.
- **Significance inflation.** "marks a pivotal moment", "underscores its
  importance", "represents a paradigm shift", "is a testament to". Drop.
- **Promotional adjectives.** breathtaking, vibrant, stunning, elegant,
  beautifully, seamlessly, powerful, groundbreaking. Replace with the
  specific property or cut.
- **Persuasive-authority tropes.** "The real question is…", "At its core…",
  "What really matters is…", "Fundamentally…". Drop.
- **Filler hedging.** "It could potentially possibly be argued that…", "may
  perhaps in some cases…". One hedge or none.
- **Generic positive conclusions.** "the future looks bright", "exciting
  times lie ahead", "a step in the right direction". Cut.
- **Chatbot artifacts.** "Of course!", "I hope this helps", "Let me know
  if…", "As an AI…". Cut.
- **Knowledge-cutoff disclaimers.** "as of the time of writing", "based on
  available information". Cut or commit to a fact.
- **Fragmented headers.** A heading followed by a one-sentence paragraph
  that restates the heading. Cut the warm-up.
- **Signposting.** "Let's dive in…", "Here's what you need to know…", "We
  will now discuss…". Just discuss it.

### Style limits
- **Em dashes (`---`, `--`, `—`):** abstract / conclusion: max **1** each.
  Other sections (introduction, method, experiments, related work): max **3**
  each. Convert excess to commas, parentheses, or a sentence break.
  paper-quality-checker greps these.
- **Boldface in prose:** none. Bold the "Ours" row of the table; don't bold
  inline phrases for emphasis.
- **Curly quotes:** use straight `"` (LaTeX: `` `` … '' ``).
- **Emojis:** none anywhere.
- **Title-case section headings:** sentence-case only ("Method" not "The
  Proposed Method").

## Number-density gate (the most-violated rule in this lab)

Apply on every abstract and conclusion:

1. **List every number** in the section to a scratch buffer. Include dB
   deltas, absolute PSNR, scene counts, parameter counts, percentages,
   ablation rows, σ values, reproduction caveats, control-experiment
   numbers, count-of-negatives, grid sizes — everything.
2. **Rank by load-bearing weight.** A load-bearing number is one that, if
   removed, would weaken the paper's central claim. Decoration is everything
   else.
3. **Cut to the cap.** Abstract: at most **3 distinct numeric quantities,
   hard max 4**. Conclusion: at most **3**.
4. **Default keepers (priority order):**
   1. Headline delta vs the closest baseline ("+0.57 dB over splatfacto").
   2. Scene set / dataset scope ("on the 7 Mip-NeRF 360 scenes").
   3. One secondary metric (LPIPS, NLL) IF it adds to the claim.
   4. One ablation takeaway IF the ablation IS the contribution.
5. **Default cuts:**
   - Per-scene numbers (already in the table).
   - Grid / resolution / tile sizes.
   - Wall-clock or memory unless the contribution is efficiency.
   - σ, noise floor, p-values.
   - "Reproduced X dB vs published Y dB" caveats.
   - Counts of negatives, control numbers, side-experiment deltas.
   - Counts of components: "$24$ controlled experiments", "$13$ negatives".

If after the cut the abstract still has more than 4 numbers, you have not
finished cutting. Most lab papers can land at 2-3 numbers and read better
for it.

## Engineering-report language (separate hard ban — don't let it back in)

NEVER in abstract / introduction / method / conclusion:
- Line counts ("65 lines of Python", "~150 LOC", "single-file change").
- File names (`train.py`, `gaussian_model.py`, `scene/dataset_readers.py`).
- Function names (`def densify_and_prune_fastgs()`).
- Language names ("Python", "Pythonic", "CUDA-free", "pure-Python").
- "Drop-in replacement / patch / modification" as headline framing.
- Wall-clock as a headline claim ("+2.5% training overhead" in abstract).

Substitutes:
- Express simplicity through math: "one additional loss term, no new learnable
  parameters", "complexity is $O(HW)$ per iteration".
- Express overhead in the experiments compute-cost subsection only, with
  full hardware/protocol annotation.
- Describe the interface, not the file: "modifies only the photometric
  objective; densification and rasterization are unchanged".

See `summary-writer.md` and `body-writer.md` "Academic voice" sections for
the full table of academic substitutes.

## The two-question audit (do this before writing the file)

After drafting and applying the patterns, ask yourself:

1. **"What about the below would still tag as obviously AI-generated?"**
   Answer briefly. Common residuals: rhythm too uniform, transitions too tidy,
   adjectives too consistent, every sentence about the same length. Note them.
2. **"Now make it not obviously AI-generated."** Revise once more. Vary
   sentence length deliberately. Cut a sentence that sounds like filler.
   Replace one abstract phrase with a concrete one.

This second pass is mandatory. The first pass removes the obvious tells; the
second pass removes the residual rhythm.

## When you're done

Write the cleaned section back. Don't announce the cleanup. The
paper-quality-checker greps the file afterward and either passes or flags.

If you're working through `paper-quality-checker`'s flagged issues, prefer
fixing in this order: critical hallucinated metrics first, then
engineering-report language, then number overload, then AI-tell vocabulary,
then style limits.
