---
name: humanizer
version: 2.5.1-academic
description: |
  Remove signs of AI-generated writing from paper text. Use when editing
  any LaTeX section (abstract, introduction, method, experiments, related
  work, conclusion, supplementary, rebuttal) so the prose reads like a
  human researcher wrote it. Based on Wikipedia's "Signs of AI writing"
  guide, adapted for academic ML / 3D-vision papers. Detects and fixes:
  inflated significance claims, promotional language, superficial -ing
  phrases, vague attributions, em-dash overuse, rule-of-three padding,
  AI vocabulary, copula avoidance, negative parallelism, false ranges,
  fragmented headers, filler hedging, and number-overload in abstracts.
license: MIT
source: https://github.com/blader/humanizer/blob/main/SKILL.md
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
---

# Humanizer (academic edition): remove AI writing patterns from paper sections

Use this skill when editing or finalizing any section of `paper/sections/*.tex`,
`paper/supplementary*.tex`, or `paper/response.tex`. This is the canonical guide;
`.claude/agents/_SHARED_humanizer_protocol.md` is the condensed version each
writer-agent reads on every dispatch.

## When this skill applies

- Just finished writing or revising a section (abstract, intro, method, etc.).
- Reviewing the paper before `paper-quality-checker` runs.
- Addressing a `revision-agent` flag with `kind=ai_tell_word`,
  `kind=ai_tell_pattern`, `kind=number_overload`, or
  `kind=engineering_report_language`.

## Process (every section)

1. Read the section.
2. Scan for each pattern in the catalog below.
3. Rewrite each problematic span. Preserve all numeric claims, citations,
   equations, and labels — only the surrounding prose changes.
4. Final audit: read the section aloud (mentally). If a sentence sounds like
   a press release, a tutorial intro, or a chatbot turn, rewrite it.
5. Self-check: ask "what about the below would still tag as obviously AI?"
   Answer briefly, then revise.

## Academic voice rules (specific to this lab)

- **Numbers are accents, not the prose.** The abstract holds at most 3-4
  distinct numeric quantities; the conclusion at most 3. Every other number
  belongs in the experiments section's tables. See "Number-density gate"
  below for the cut protocol.
- **No engineering-report language.** Never cite line counts, file names
  (`train.py`), function names (`def densify_and_prune()`), language names
  ("Python"), or "drop-in patch" framings in abstract / intro / method /
  conclusion. See `body-writer.md` and `summary-writer.md` "Academic voice"
  sections for the full ban list and academic substitutes.
- **No first-person voice in standard sections.** "We show that…" is fine;
  "I keep coming back to…" is not. The standard humanizer skill recommends
  more first-person; ignore that for paper sections. Apply it cautiously in
  rebuttals and supplementary discussion only.
- **Soul lives in clarity, not personality.** A human-written paper conveys
  a researcher with opinions through honest reporting (negative results,
  failure cases, hedged claims with the actual hedge), not through
  conversational asides.

## Pattern catalog

### Content patterns

#### 1. Significance / legacy inflation
**Watch:** stands as / serves as, is a testament, marking a pivotal moment,
underscores its importance, reflects broader trends, contributing to the,
setting the stage for, key turning point, evolving landscape, focal point,
indelible mark, deeply rooted, paradigm shift.

**Fix:** Drop the inflation. State the fact.

> Before: This work marks a pivotal moment in the evolution of feed-forward 3DGS calibration, underscoring the importance of post-hoc uncertainty estimation in the broader landscape of neural rendering.
>
> After: We add a post-hoc calibration head to a frozen feed-forward 3DGS backbone. The head produces per-pixel uncertainty without retraining.

#### 2. Notability / coverage padding
**Watch:** "has been featured in", "widely adopted", "active community", "leading benchmark".

**Fix:** Cite specifically, or cut.

#### 3. Superficial -ing analyses
**Watch:** highlighting…, underscoring…, ensuring…, reflecting…, symbolizing…,
contributing to…, fostering…, encompassing…, showcasing…, marking…, bridging…

**Fix:** Cut the participial tail. The previous clause stands on its own.

> Before: We add a per-pixel residual buffer, ensuring that high-frequency errors are penalized appropriately, reflecting the broader trend toward perceptual losses in 3D reconstruction.
>
> After: We add a per-pixel residual buffer that penalizes high-frequency errors.

#### 4. Promotional language
**Watch:** vibrant, rich, profound, breathtaking, must-have, groundbreaking,
seamless, cutting-edge, state-of-the-art (without a number), powerful, elegant
(as adjective), beautifully (as adverb), nestled, in the heart of.

**Fix:** Replace with the specific property.

#### 5. Vague attributions / weasel words
**Watch:** "researchers have shown", "experts argue", "industry observers",
"recent work", "the community has noted", "it is widely believed", "many studies",
"prior work has demonstrated".

**Fix:** Cite the specific paper. `\citet{kerbl2023}` or nothing.

#### 6. "Challenges and Future Prospects" template
**Watch:** sections that read "Despite these advances, challenges remain…".

**Fix:** Either cut or replace with concrete, named limitations and one
specific future-work direction (not "extending to other domains").

### Language and grammar patterns

#### 7. AI vocabulary words (HARD BAN — paper-quality-checker greps these)

actually, additionally, align with, crucial, delve, emphasizing, enduring,
enhance, fostering, garner, highlight (as verb), interplay, intricate /
intricacies, key (as adjective), landscape (abstract noun), pivotal, showcase,
tapestry, testament, underscore (as verb), valuable, vibrant, multifaceted,
nuanced, comprehensive, robust (as throwaway adjective), cornerstone, paradigm,
synergy, holistic, streamline, breathtaking, nestled, bustling, moreover,
furthermore, notably, importantly, interestingly, "it is worth noting that",
"it is important to note that", navigate, leverage, realm, embark.

#### 8. Copula avoidance
**Watch:** "X serves as Y", "X stands as Y", "X represents Y", "X marks Y",
"X functions as Y", "X acts as Y", "X plays the role of Y".

**Fix:** Use is/are.

> Before: The cross-attention entropy serves as a per-pixel uncertainty signal.
>
> After: Cross-attention entropy is a per-pixel uncertainty signal.

#### 9. Negative parallelism / tailing negation
**Watch:** "Not only X, but also Y", "It's not just X; it's Y", "X without Y, no Z".

**Fix:** Make the positive claim directly.

> Before: This is not merely a calibration head; it is a window into the model's internal uncertainty.
>
> After: This calibration head exposes the model's internal uncertainty.

#### 10. Rule-of-three padding
**Watch:** any list of exactly three near-synonyms used to feel comprehensive.
"streamlining processes, enhancing collaboration, and fostering alignment".

**Fix:** Pick the one that's actually true and keep it.

#### 11. Elegant variation (synonym cycling)
**Watch:** the same noun referred to as "the model / the architecture / the
network / the system / the framework" within one paragraph.

**Fix:** Pick one term per concept per section.

#### 12. False ranges
**Watch:** "from X to Y" where X and Y are not on a meaningful scale.
"from per-scene retraining to feed-forward inference, from dense supervision to
unsupervised priors".

**Fix:** Drop one half, or list as a short list.

#### 13. Passive-voice and subjectless fragments (use judgment)
**Watch:** "No retraining required.", "Improvements are observed across all metrics."

**Fix:** Use active voice unless the actor is irrelevant. (Methods sections
often use passive correctly — "primitives are projected onto the image plane".)

### Style patterns

#### 14. Em-dash overuse (HARD: ≤ 3 em dashes per section)
**Watch:** `---`, `--`, `—`, `–` used as in-line interrupters.

**Fix:** Convert to commas, parentheses, or two sentences.

> Before: HeadVarAttDis --- a chain extension reading attention-entropy variance --- reduces NLL to 0.345.
>
> After: HeadVarAttDis (a chain extension reading attention-entropy variance) reduces NLL to 0.345.

#### 15. Boldface overuse
Use boldface for the table's "Ours" row and not much else. Do not bold
inline phrases for emphasis.

#### 16. Inline-header vertical lists
**Watch:** `\item \textbf{Speed:} …`, `\item \textbf{Quality:} …`

**Fix:** Use a short paragraph or keep the list but drop the bolded header
when the item is one sentence.

#### 17. Title-case in headings
**Fix:** Sentence-case section / subsection titles. "Method" not "The Proposed
Method"; "Experiments on Mip-NeRF 360" not "Experiments On Mip-NeRF 360".

#### 18. Emojis
None. Anywhere.

#### 19. Curly quotes
Replace `"` and `"` with straight `"` (LaTeX: `` `` … '' ``).

### Communication patterns

#### 20. Chatbot artifacts
**Watch:** "Of course!", "I hope this helps", "Let me know if…", "As an AI…",
"Here is an overview of…".

**Fix:** Delete. The reader doesn't want to be addressed.

#### 21. Knowledge-cutoff disclaimers
**Watch:** "as of the time of writing", "based on available information",
"while specific details remain limited".

**Fix:** Either commit to a fact and cite it, or omit the claim entirely.

#### 22. Sycophantic / servile tone
Not relevant in academic prose; flag in rebuttals only.

### Filler and hedging

#### 23. Filler phrases
- "in order to" → "to"
- "due to the fact that" → "because"
- "at this point in time" → "now"
- "in the event that" → "if"
- "has the ability to" → "can"
- "it is important to note that" → cut
- "a wide range of" → "many", or specific count

#### 24. Excessive hedging
**Watch:** "could potentially possibly", "may perhaps", "might in some cases".
**Fix:** One hedge or none.

#### 25. Generic positive conclusions
**Watch:** "the future looks bright", "exciting times lie ahead", "a step in the
right direction", "this opens new horizons".
**Fix:** Cut. End on a concrete limitation or direction.

#### 26. Hyphenated word-pair overuse (use judgment in academic writing)
"high-quality", "real-time", "long-term", "data-driven", "end-to-end",
"cross-functional" — humans hyphenate inconsistently; LLMs do it uniformly.
In a paper, prefer the field's existing convention (e.g. "real-time" with
hyphen is standard in graphics; "high quality" without is fine when adjectival
sense is unambiguous).

#### 27. Persuasive-authority tropes
**Watch:** "The real question is…", "At its core…", "What really matters is…",
"Fundamentally…".
**Fix:** Drop the framing. Make the claim.

#### 28. Signposting / announcements
**Watch:** "Let's dive in…", "Here's what you need to know…", "We will now
discuss…", "In what follows we…".
**Fix:** Just discuss it. Or use a section heading.

#### 29. Fragmented headers
**Watch:** a section heading followed by a one-sentence paragraph that restates
the heading.
**Fix:** Cut the warm-up sentence.

## Number-density gate (the abstract / conclusion fix)

The most common AI tell in this lab's papers is the abstract that reads as
a results dump. Apply this gate every time you finalize an abstract or
conclusion:

1. **Enumerate every number** in the section in a scratch list. Include dB
   deltas, scene counts, parameter counts, percentages, ablation rows,
   reproduction caveats, σ values — everything.
2. **Rank by load-bearing weight.** Which numbers, if removed, would weaken
   the paper's claim? Which are decoration?
3. **Cut to the cap.** Abstract: keep at most 3, hard max 4. Conclusion: keep
   at most 3.
4. **Default keepers (in order of priority):**
   - The headline delta vs the closest baseline (e.g. "+0.57 dB PSNR over splatfacto").
   - The dataset / scene-set scope ("on the 7 Mip-NeRF 360 scenes").
   - One secondary metric IF it materially adds to the story (LPIPS, NLL).
   - One ablation takeaway IF the ablation is the contribution.
5. **Default cuts:**
   - Per-scene numbers (already in the table).
   - Grid / resolution / tile sizes.
   - Wall-clock or memory, unless the paper is about efficiency.
   - σ values, noise floors, p-values.
   - "Reproduced X dB vs published Y dB" caveats.
   - Counts of negatives, control-experiment numbers, negative-result deltas.

If the abstract has more than 4 numbers after this pass, you have not
finished cutting.

## Worked example (real failure mode from this lab)

**Before (run_20260501_212753 abstract — 12 distinct numbers, 6 em dashes):**

> The nebula scene is an orbital fly-around of a rendered astronomical nebula:
> an emissive volumetric medium with sparse high-luminance stars on a
> near-black background. This content violates the surface-BRDF assumptions
> that vanilla 3D Gaussian Splatting inherits from Mip-NeRF~360-class
> benchmarks. We characterise the scene's light transport (volume-emission
> integral with weak extinction, view-independent emission, sparse-Dirac
> point sources) and use that physics to guide $24$ controlled experiments
> against a reproduced \texttt{splatfacto} baseline at $33.44$ dB.
>
> Two findings emerge. First, on the heuristic-densification substrate the
> two scene-physics priors we propose lift PSNR independently --- image-space
> star seeding (\textsc{StarSeedGS}, $+0.53$ dB) and chained isotropy
> regularisation (\textsc{IsoPriorChain}, $+0.33$ dB) --- but their
> composition on splatfacto regresses ($-1.10$ dB), exposing a
> substrate-prior interaction we report in full. Second, on the
> MCMC-relocation substrate the same priors compose constructively:
> \textsc{Nebula3DGS-v2} reaches $36.36$ dB / SSIM $0.945$ / LPIPS $0.201$,
> the best LPIPS of any variant we trained, and the full three-layer
> composition reaches $36.28$ dB / LPIPS $0.205$. A random-seed control
> isolates StarSeedGS's mechanism: seeding $4699$ random Gaussians regresses
> by $1.87$ dB while \textsc{StarSeedGS} (same count) gains $0.53$ dB --- a
> $2.39$ dB falsifier that the win comes from \emph{where} the Gaussians are
> placed. We report all $13$ negatives in the same detail as the positives.

Numbers in this draft: 24, 33.44, +0.53, +0.33, −1.10, 36.36, 0.945, 0.201,
36.28, 0.205, 4699, 1.87, +0.53, 2.39, 13 → **15 distinct numbers**.

**After (3 numbers, ≤ 3 em dashes, no participial tails, no false ranges):**

> Vanilla 3D Gaussian Splatting assumes opaque surfaces and fails on emissive
> volumetric content like rendered astronomical nebulae. We propose two
> physics-motivated priors: image-space seeding for sparse point sources
> (\textsc{StarSeedGS}) and an isotropy regularizer for diffuse emission
> (\textsc{IsoPriorChain}). On the MCMC relocation substrate \cite{kheradmand2024mcmc}
> the combined method, \textsc{Nebula3DGS}, reaches $36.36$ dB on the nebula
> scene, $+2.92$ dB above a reproduced \texttt{splatfacto} baseline. A random-seed
> control places the same number of Gaussians without the image-space prior and
> regresses by $1.87$ dB, isolating the mechanism to placement rather than
> capacity. Both priors compose constructively on MCMC and destructively on
> heuristic densification; we report this substrate dependence and all
> negative results in the experiments.

Cuts made: removed 24 controlled experiments (engineering bookkeeping), 33.44
dB baseline (implied by the +2.92 delta), 0.945 SSIM and the two LPIPS
values (secondary metrics, in main table), the second composition's 36.28 dB
(repeat of the same headline number with one fewer component), the random-seed
count 4699 (implementation detail), the +0.53 / +0.33 individual contributions
(belong in ablation table), the +0.53 win-restate, and the 13-negatives count.
Em dashes converted to parentheses or two sentences.

## Output

Write the cleaned section back to its original path. Do not announce the
edit — the change is the change. The paper-quality-checker reads the file
afterward and either passes the gate or flags remaining issues.

## Reference

- Wikipedia: Signs of AI writing, https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing
- humanizer skill source: https://github.com/blader/humanizer/blob/main/SKILL.md
