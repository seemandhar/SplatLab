"""
Evolution of Heuristics (EoH) — (rationale, implementation) pairs with
five mutation operators, plus epsilon-greedy operator selection.

Motivation: FunSearch-style evolution mutates code only, so every LLM
call has to re-derive the "why" from scratch. EoH instead carries a
short natural-language rationale alongside every implementation and
gives the mutator operator-specific instructions, which dramatically
reduces redundant reasoning and empirically cuts LLM queries by
orders of magnitude at similar performance.

Operators (from the EoH paper):

  E1 — Crossover.            Pick good pieces from ≥2 parents, produce a
                             child that combines them. Conservative mix.
  E2 — Exploratory crossover. Same inputs, but the agent is explicitly
                             told to experiment: mix parents in
                             unexpected ways or add a novel twist.
  M1 — Refine.               Single-parent edit that improves clarity or
                             corrects a suspected weakness. Small diff.
  M2 — Parameter tweak.      Single-parent edit that only touches
                             numerical constants / hyperparameters.
  M3 — Structural rewrite.   Single-parent edit that may restructure
                             control flow or add/remove a sub-block.

Scheduling: epsilon-greedy per-operator credit. Each operator tracks
the average Elo gain of children produced via that operator. With
prob (1-ε) we sample the operator with the highest running average;
with prob ε we sample uniformly. This is deliberately simple —
the operator prompts do the heavy lifting, not the bandit math.

State lives at `<workspace>/eoh/state.json` so it persists across
generations and across a Flask restart.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ═══════════════════════════════════════════════════════════════════
# OPERATOR CATALOG
# ═══════════════════════════════════════════════════════════════════

OPERATORS: tuple[str, ...] = ("E1", "E2", "M1", "M2", "M3")

# Number of parent candidates each operator consumes.
OPERATOR_ARITY: dict[str, int] = {
    "E1": 2, "E2": 2, "M1": 1, "M2": 1, "M3": 1,
}


def operator_prompt(op: str, lessons: str = "") -> str:
    """Agent-facing instructions for one mutation operator.

    `lessons` is the digest produced by elo_gradients.build_lessons_digest;
    passing an empty string is fine.
    """
    header = {
        "E1": (
            "You are applying EoH operator E1 (Crossover).\n"
            "You will receive TWO parent implementations. Pick the most\n"
            "effective code from each parent and COMBINE them into a single\n"
            "child that inherits the strengths of both. Prefer conservative\n"
            "mixing — do not invent entirely new ideas here; that's E2's job.\n"
        ),
        "E2": (
            "You are applying EoH operator E2 (Exploratory Crossover).\n"
            "You will receive TWO parent implementations. Mix them in an\n"
            "UNEXPECTED way, or add one novel twist on top of the mix.\n"
            "Embrace exploration — this operator is allowed to try ideas\n"
            "that may not pan out. But the result must still compile and\n"
            "train without NaNs.\n"
        ),
        "M1": (
            "You are applying EoH operator M1 (Refine).\n"
            "You will receive ONE parent implementation. Improve it with a\n"
            "SMALL targeted edit that fixes a suspected weakness, improves\n"
            "numerical stability, or tightens a loss formulation. Small\n"
            "diffs only — do not restructure.\n"
        ),
        "M2": (
            "You are applying EoH operator M2 (Parameter Tweak).\n"
            "You will receive ONE parent implementation. Change ONLY\n"
            "numerical constants / hyperparameters (loss weights, learning\n"
            "rates, thresholds, coefficients). Do not change the control\n"
            "flow, structure, or any non-numeric code.\n"
        ),
        "M3": (
            "You are applying EoH operator M3 (Structural Rewrite).\n"
            "You will receive ONE parent implementation. You may restructure\n"
            "control flow inside the EVOLVE-BLOCK(s), add a helper step, or\n"
            "remove a sub-block. The overall method signature must remain\n"
            "unchanged and the method must still train.\n"
        ),
    }[op]

    footer = (
        "\nProduce TWO things in order:\n"
        "1. A one-line 'Rationale:' (≤25 words) stating the single idea\n"
        "   behind your edit.\n"
        "2. The mutated code, using the EVOLVE-BLOCK protocol exactly.\n"
    )

    return header + (lessons + "\n" if lessons else "") + footer


# ═══════════════════════════════════════════════════════════════════
# (RATIONALE, IMPLEMENTATION) PAIR + SCHEDULER
# ═══════════════════════════════════════════════════════════════════

@dataclass
class RIPair:
    """Rationale + implementation pointer. Carried through the pipeline.

    `impl_path` is a directory on disk (an island's copy of the method
    package). `rationale` is the one-liner the mutator agent produced.
    `elo` is the candidate's current rating — used to pick parents.
    """
    key: str                          # stable id e.g. gen3_island2
    rationale: str
    impl_path: Path
    elo: float = 1500.0
    psnr: float = 0.0
    operator_used: str = ""           # the operator that spawned THIS pair
    parents: list[str] = field(default_factory=list)
    hooks_touched: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["impl_path"] = str(self.impl_path)
        return d


@dataclass
class OperatorStats:
    """Running average of Elo gain per child for one operator."""
    n: int = 0
    total_gain: float = 0.0

    @property
    def avg_gain(self) -> float:
        return self.total_gain / self.n if self.n > 0 else 0.0

    def record(self, gain: float) -> None:
        self.n += 1
        self.total_gain += gain


@dataclass
class OperatorScheduler:
    """Epsilon-greedy selector over the 5 EoH operators.

    The scheduler's job is to pick WHICH operator spawns each slot in
    the next generation. Caller then builds the prompt via
    `operator_prompt` and hands parents to the mutator agent.
    """
    stats: dict[str, OperatorStats] = field(default_factory=lambda: {op: OperatorStats() for op in OPERATORS})
    epsilon: float = 0.25
    rng_seed: int | None = None

    def __post_init__(self):
        self._rng = random.Random(self.rng_seed)
        for op in OPERATORS:
            self.stats.setdefault(op, OperatorStats())

    # ── selection ───────────────────────────────────────────────

    def choose(self) -> str:
        """Sample one operator. ε-greedy over running gain averages."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(OPERATORS)
        # Exploit: best avg gain. Ties broken randomly.
        ranked = sorted(
            OPERATORS,
            key=lambda op: (self.stats[op].avg_gain, self._rng.random()),
            reverse=True,
        )
        return ranked[0]

    def plan_generation(self, n_slots: int) -> list[str]:
        """Pick an operator for each of `n_slots` children.

        The first slot always runs M1 (conservative refine) to keep a
        safe anchor in the generation; the rest are ε-greedy.
        """
        plan: list[str] = []
        if n_slots > 0:
            plan.append("M1")
        for _ in range(max(0, n_slots - 1)):
            plan.append(self.choose())
        return plan

    # ── credit assignment ──────────────────────────────────────

    def record_outcome(self, op: str, elo_delta: float) -> None:
        if op in self.stats:
            self.stats[op].record(elo_delta)

    # ── persistence ────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "stats": {op: asdict(s) for op, s in self.stats.items()},
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OperatorScheduler":
        obj = cls(epsilon=float(data.get("epsilon", 0.25)))
        for op, s in (data.get("stats") or {}).items():
            if op in obj.stats:
                obj.stats[op] = OperatorStats(
                    n=int(s.get("n", 0)),
                    total_gain=float(s.get("total_gain", 0.0)),
                )
        return obj

    @classmethod
    def load(cls, workspace: Path) -> "OperatorScheduler":
        path = Path(workspace) / "eoh" / "state.json"
        if path.exists():
            try:
                return cls.from_dict(json.loads(path.read_text()))
            except Exception:
                pass
        return cls()

    def save(self, workspace: Path) -> None:
        path = Path(workspace) / "eoh" / "state.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


# ═══════════════════════════════════════════════════════════════════
# PARENT SELECTION
# ═══════════════════════════════════════════════════════════════════

def pick_parents(
    pool: list[RIPair],
    op: str,
    rng: random.Random | None = None,
) -> list[RIPair]:
    """Choose parents for an operator invocation.

    Arity-1 operators (M1/M2/M3) weight by Elo (softmax, τ=200).
    Arity-2 operators (E1/E2) pick one Elo-weighted parent plus one
    uniformly random parent to inject diversity — prevents the best
    pair from dominating every crossover.
    """
    if rng is None:
        rng = random.Random()
    if not pool:
        return []

    def _softmax_pick(tau: float = 200.0) -> RIPair:
        import math
        elos = [c.elo for c in pool]
        mx = max(elos)
        weights = [math.exp((e - mx) / tau) for e in elos]
        total = sum(weights) or 1.0
        r = rng.random() * total
        acc = 0.0
        for c, w in zip(pool, weights):
            acc += w
            if r <= acc:
                return c
        return pool[-1]

    arity = OPERATOR_ARITY[op]
    if arity == 1:
        return [_softmax_pick()]
    # arity == 2: one strong + one random
    p1 = _softmax_pick()
    remaining = [c for c in pool if c.key != p1.key] or pool
    p2 = rng.choice(remaining)
    return [p1, p2]


# ═══════════════════════════════════════════════════════════════════
# PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════

def build_mutation_prompt(
    op: str,
    parents: list[RIPair],
    lessons: str,
    hooks_in_scope: list[str],
    target_path: Path,
) -> str:
    """Assemble the full mutator input: op instructions + parents + lessons.

    The target_path is the directory the child should write its
    mutated code to; the agent is told to read parents, write child.
    """
    from .evolve_blocks import guarded_diff_prompt
    lines = [operator_prompt(op, lessons=lessons)]
    for i, p in enumerate(parents, 1):
        lines.append(
            f"\nPARENT {i}: {p.key} (PSNR={p.psnr:.2f}, Elo={p.elo:.0f})\n"
            f"  rationale: {p.rationale or '(none)'}\n"
            f"  code: {p.impl_path}"
        )
    lines.append(f"\nWRITE THE CHILD TO: {target_path}")
    lines.append(guarded_diff_prompt(hooks_in_scope))
    return "\n".join(lines)
