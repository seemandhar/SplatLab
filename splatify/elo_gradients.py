"""
Elo rating + ReEvo-style verbal gradients.

Combines two ideas into one module:

1. **Elo** — instead of `max(candidates, key=psnr)`, every pair of
   candidates within a generation contributes to a transitive rating
   that persists across generations. A candidate that was second in
   gen 1 but beat everyone in gen 2 ends up with a higher rating than
   a candidate that was first in gen 1 alone. Ratings are seeded at
   1500 for newcomers (including across re-ideations).

2. **Verbal gradients** — when a winner beats a "surprisingly good"
   loser (high prior Elo), we ask a small Sonnet agent to read both
   diffs and write 2-3 sentences explaining *why*. These sentences
   accumulate into a per-hook "lessons" file that the next generation's
   mutator agent reads. This is the only place in the pipeline where a
   failed candidate's code informs the next round, and it's cheap —
   one extra Sonnet call per surprising-loss pair.

Design notes:
- Elo math is bog-standard K=32, all pairs per generation.
- "Surprisingly good" = loser's prior rating was within SURPRISE_WINDOW
  of the winner's. This avoids wasting gradient calls on obvious losses.
- The verbal gradient agent is defined in agents/definitions.py; this
  module just builds its input prompt and parses its output.
- Lessons file is append-only JSONL at
  <workspace>/elo/lessons.jsonl, with per-hook rollups materialized on
  demand via `build_lessons_digest`.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

K_FACTOR: float = 32.0
INITIAL_RATING: float = 1500.0
SURPRISE_WINDOW: float = 60.0  # Elo diff below which a loss is "surprising"
MAX_GRADIENTS_PER_GEN: int = 4


# ═══════════════════════════════════════════════════════════════════
# ELO RATING
# ═══════════════════════════════════════════════════════════════════

@dataclass
class EloState:
    """Persistent ladder — keyed by stable candidate identifier.

    We use `candidate_key` (e.g. "gen3_island2") rather than the
    integer `id` from discover.py because ids are recycled across
    generations. A single JSON file at workspace/elo/state.json holds
    the ladder across the whole run.
    """
    ratings: dict[str, float] = field(default_factory=dict)
    pair_history: list[dict] = field(default_factory=list)  # audit trail

    def get(self, key: str) -> float:
        return self.ratings.get(key, INITIAL_RATING)

    def ensure(self, key: str) -> float:
        if key not in self.ratings:
            self.ratings[key] = INITIAL_RATING
        return self.ratings[key]

    def apply_match(self, winner_key: str, loser_key: str, margin_db: float = 0.0):
        """Update ratings after one winner/loser match.

        `margin_db` (PSNR gap) scales K slightly so a blowout moves
        ratings more than a photo-finish. Bounded to [0.5, 1.5] of K.
        """
        rw = self.ensure(winner_key)
        rl = self.ensure(loser_key)
        expected_w = 1.0 / (1.0 + math.pow(10.0, (rl - rw) / 400.0))
        scale = max(0.5, min(1.5, 0.75 + margin_db * 0.25))
        delta = K_FACTOR * scale * (1.0 - expected_w)
        self.ratings[winner_key] = rw + delta
        self.ratings[loser_key] = rl - delta
        self.pair_history.append({
            "winner": winner_key, "loser": loser_key,
            "margin_db": round(margin_db, 3),
            "delta": round(delta, 2),
            "rw_before": round(rw, 1), "rl_before": round(rl, 1),
        })

    def ranking(self) -> list[tuple[str, float]]:
        return sorted(self.ratings.items(), key=lambda kv: kv[1], reverse=True)

    def to_dict(self) -> dict:
        return {"ratings": self.ratings, "pair_history": self.pair_history[-200:]}

    @classmethod
    def from_dict(cls, data: dict) -> "EloState":
        obj = cls()
        obj.ratings = dict(data.get("ratings", {}))
        obj.pair_history = list(data.get("pair_history", []))
        return obj

    @classmethod
    def load(cls, workspace: Path) -> "EloState":
        path = Path(workspace) / "elo" / "state.json"
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text()))
            except Exception:
                pass
        return cls()

    def save(self, workspace: Path) -> None:
        path = Path(workspace) / "elo" / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


@dataclass
class ScoredCandidate:
    """Minimal shape the Elo + gradient layer needs from a candidate."""
    key: str             # stable across generations (e.g. "gen2_island5")
    display_name: str
    psnr: float
    alive: bool
    impl_path: Path | None = None  # needed for verbal gradient diffs
    rationale: str = ""           # EoH "why" from the mutator agent
    hooks_touched: list[str] = field(default_factory=list)


def update_from_generation(
    state: EloState,
    candidates: list[ScoredCandidate],
) -> list[tuple[str, str, float]]:
    """Run all-pairs Elo updates within one generation.

    Returns the list of pair results (winner_key, loser_key, margin_db)
    so the caller can pick out "surprising" pairs for verbal gradients.
    """
    alive = [c for c in candidates if c.alive and c.psnr > 0]
    pairs: list[tuple[str, str, float]] = []
    for i, a in enumerate(alive):
        for b in alive[i + 1:]:
            if a.psnr == b.psnr:
                # Draw: tiny symmetric nudge toward mean.
                state.ensure(a.key); state.ensure(b.key)
                continue
            winner, loser = (a, b) if a.psnr > b.psnr else (b, a)
            margin = winner.psnr - loser.psnr
            state.apply_match(winner.key, loser.key, margin_db=margin)
            pairs.append((winner.key, loser.key, margin))
    return pairs


def surprising_pairs(
    state: EloState,
    pairs: list[tuple[str, str, float]],
    limit: int = MAX_GRADIENTS_PER_GEN,
) -> list[tuple[str, str, float]]:
    """Pick the N pairs where the loser was a priori close to the winner.

    Pre-match ratings are pulled from `pair_history`; if a pair isn't
    in the history yet we fall back to current rating as a proxy. We
    favor small pre-match Elo diffs (strong opponents, upset-flavored
    losses) over blowouts.
    """
    ranked = sorted(
        pairs,
        key=lambda p: abs(state.get(p[0]) - state.get(p[1])),
    )
    picked: list[tuple[str, str, float]] = []
    for p in ranked:
        diff = abs(state.get(p[0]) - state.get(p[1]))
        if diff <= SURPRISE_WINDOW or len(picked) < 2:
            picked.append(p)
        if len(picked) >= limit:
            break
    return picked


# ═══════════════════════════════════════════════════════════════════
# VERBAL GRADIENTS
# ═══════════════════════════════════════════════════════════════════

@dataclass
class VerbalGradient:
    winner_key: str
    loser_key: str
    margin_db: float
    hooks: list[str]             # union of hooks_touched across the pair
    lesson: str                  # 2-3 sentence natural-language takeaway
    generation: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def build_gradient_prompt(
    winner: ScoredCandidate,
    loser: ScoredCandidate,
    winner_blocks: dict[str, str],
    loser_blocks: dict[str, str],
) -> str:
    """Build the input for the verbal_gradient agent.

    The agent receives per-hook diffs, the PSNR delta, and each
    candidate's EoH rationale, and is asked to produce a short
    explanation of *why* the winning change helped. We intentionally
    give it only the marked blocks (not the whole file) so its output
    focuses on the evolved regions.
    """
    diff_sections: list[str] = []
    for hook in sorted(set(winner_blocks) | set(loser_blocks)):
        w = winner_blocks.get(hook, "<no block>").strip()
        l = loser_blocks.get(hook, "<no block>").strip()
        if w == l:
            continue
        diff_sections.append(
            f"### HOOK {hook}\n"
            f"--- LOSER ({loser.display_name}, PSNR={loser.psnr:.2f}):\n"
            f"```python\n{l}\n```\n"
            f"+++ WINNER ({winner.display_name}, PSNR={winner.psnr:.2f}):\n"
            f"```python\n{w}\n```\n"
        )

    return (
        "Two 3DGS candidates competed on the same scene.\n"
        f"WINNER: {winner.display_name} @ {winner.psnr:.2f} dB\n"
        f"  rationale: {winner.rationale or '(none)'}\n"
        f"LOSER:  {loser.display_name} @ {loser.psnr:.2f} dB\n"
        f"  rationale: {loser.rationale or '(none)'}\n"
        f"Margin: {winner.psnr - loser.psnr:+.2f} dB\n\n"
        "Below are the EVOLVE-BLOCK diffs (only blocks that differ):\n\n"
        + "\n".join(diff_sections)
        + "\n\nWrite a concise 2-3 sentence takeaway explaining WHY the\n"
          "winning change helped. Focus on the mechanism, not the result.\n"
          "End your message with a single JSON line:\n"
          '{"hooks": ["<hook1>", ...], "lesson": "<your 2-3 sentences>"}\n'
    )


def parse_gradient_output(text: str) -> VerbalGradient | None:
    """Extract the JSON line from the gradient agent's response."""
    if not text:
        return None
    # Find the last JSON object on its own line.
    for line in reversed(text.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                data = json.loads(line)
                hooks = data.get("hooks") or []
                lesson = (data.get("lesson") or "").strip()
                if lesson:
                    return VerbalGradient(
                        winner_key="", loser_key="", margin_db=0.0,
                        hooks=list(hooks), lesson=lesson,
                    )
            except json.JSONDecodeError:
                continue
    return None


def append_lesson(workspace: Path, gradient: VerbalGradient) -> None:
    path = Path(workspace) / "elo" / "lessons.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(gradient.to_dict()) + "\n")


def load_lessons(workspace: Path) -> list[VerbalGradient]:
    path = Path(workspace) / "elo" / "lessons.jsonl"
    if not path.exists():
        return []
    out: list[VerbalGradient] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            out.append(VerbalGradient(
                winner_key=d.get("winner_key", ""),
                loser_key=d.get("loser_key", ""),
                margin_db=float(d.get("margin_db", 0.0)),
                hooks=list(d.get("hooks", [])),
                lesson=d.get("lesson", ""),
                generation=int(d.get("generation", 0)),
            ))
        except Exception:
            continue
    return out


def build_lessons_digest(
    workspace: Path,
    hooks_filter: list[str] | None = None,
    limit: int = 12,
) -> str:
    """Markdown-ish summary of accumulated gradients for the mutator.

    Groups by hook so the mutator sees "lessons for get_loss_dict"
    together. Drops hooks with no lessons. Caps at `limit` lessons
    total (most recent wins).
    """
    lessons = load_lessons(workspace)
    if not lessons:
        return ""
    if hooks_filter:
        filt = set(hooks_filter)
        lessons = [l for l in lessons if not l.hooks or (set(l.hooks) & filt)]
    lessons = lessons[-limit:]
    if not lessons:
        return ""

    by_hook: dict[str, list[str]] = {}
    unattributed: list[str] = []
    for g in lessons:
        if not g.hooks:
            unattributed.append(f"- {g.lesson}")
            continue
        for h in g.hooks:
            by_hook.setdefault(h, []).append(f"- {g.lesson}")

    lines = ["## Lessons from prior generations (Elo gradients)"]
    for hook in sorted(by_hook):
        lines.append(f"\n### {hook}")
        lines.extend(by_hook[hook])
    if unattributed:
        lines.append("\n### general")
        lines.extend(unattributed)
    return "\n".join(lines) + "\n"
