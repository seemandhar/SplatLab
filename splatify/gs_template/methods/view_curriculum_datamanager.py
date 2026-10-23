"""
View-Curriculum DataManager — pose-similarity progressive view expansion
=========================================================================
Idea: idea_0007 (HOOK_9_DATA)

Hypothesis: concentrating early gradient on a central, coherent subset
of training views yields well-formed coarse geometry 2-3x faster than
uniform random view sampling. After `phase1_secs` of wall-clock time,
expand to the full view set to eliminate seen-view bias.

Precedent: DashGaussian (chen2025dashgaussian) — progressive resolution.
Mini-Splatting2 (fang2024minisplatting2) — consistent multi-view before
expansion. DNGaussian (li2024dngaussian) — sparse-view regularization.

Implementation:
  - __init__: O(N^2) pose-similarity sort once, producing a list of
    training view indices ordered most-central-first.
  - next_train: wall-clock-gated restriction of the candidate view set.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type

import torch
from torch import Tensor

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.datamanagers.full_images_datamanager import (
    FullImageDatamanager,
    FullImageDatamanagerConfig,
)
from nerfstudio.utils.rich_utils import CONSOLE


@dataclass
class ViewCurriculumDataManagerConfig(FullImageDatamanagerConfig):
    """DataManager that progressively expands the training view set by wall-clock."""

    _target: Type = field(default_factory=lambda: ViewCurriculumDataManager)

    phase1_frac: float = 0.30
    """Fraction of training views used during phase 1 (most-central subset)."""

    phase1_secs: float = 15.0
    """Wall-clock seconds of phase-1 restricted sampling before expanding to all views."""

    lambda_rot: float = 0.5
    """Weight on rotation similarity relative to normalized camera-center proximity."""

    ablation_random_order: bool = False
    """ABLATION A1: if True, replace pose-similarity centrality sort with a random
    permutation (disables pose-similarity clustering; phase-1 subset is random)."""

    ablation_progressive_expand: bool = True
    """ABLATION A4: if True (default / full method), progressively grow the active
    pool between phase1_secs*0.5 and phase1_secs by linearly increasing k from
    phase1_frac*N to N. If False, a hard switch occurs exactly at phase1_secs."""


class ViewCurriculumDataManager(FullImageDatamanager):
    """Wall-clock-gated pose-similarity view curriculum.

    Phase 1 (t < phase1_secs): draws only from the `phase1_frac` most-central
        training views, where centrality = sum of pairwise pose similarity.
    Phase 2 (t >= phase1_secs): uses all training views (standard behavior).

    Pose similarity between views i,j is:
        sim(i,j) = -||c_i - c_j||_2 / d_max  +  lambda_rot * tr(R_i^T R_j) / 3
    where c_* are camera centers and R_* are camera rotations from
    `cameras.camera_to_worlds`.
    """

    config: ViewCurriculumDataManagerConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._sorted_idx: List[int] = self._compute_centrality_order()
        self._t0: Optional[float] = None
        self._phase2_announced: bool = False

        N = len(self._sorted_idx)
        k1 = max(1, int(round(self.config.phase1_frac * N)))
        CONSOLE.print(
            f"[bold green]ViewCurriculum[/]: N={N} train views; "
            f"phase1 uses top-{k1} most-central for first "
            f"{self.config.phase1_secs:.1f}s"
        )
        CONSOLE.print(f"  phase1 indices (first 10): {self._sorted_idx[:k1][:10]}")

    # ═══════════════════════════════════════════════════════════════════
    # Centrality ordering (O(N^2), computed once)
    # ═══════════════════════════════════════════════════════════════════

    def _compute_centrality_order(self) -> List[int]:
        """Return training view indices sorted by descending pose centrality."""
        c2w = self.train_dataset.cameras.camera_to_worlds  # (N, 3, 4)
        if not isinstance(c2w, Tensor):
            c2w = torch.as_tensor(c2w)
        c2w = c2w.detach().cpu().float()
        N = int(c2w.shape[0])

        if N <= 1:
            return list(range(N))

        # ABLATION A1: random permutation instead of pose-similarity clustering.
        if getattr(self.config, "ablation_random_order", False):
            g = torch.Generator().manual_seed(0)
            return torch.randperm(N, generator=g).tolist()

        centers = c2w[..., 3]            # (N, 3)
        rots = c2w[..., :3]              # (N, 3, 3)

        dist = torch.cdist(centers, centers)  # (N, N)
        d_max = dist.max().clamp(min=1e-8)
        prox = -dist / d_max                  # in [-1, 0]

        # tr(R_i^T R_j) = sum_{a,b} R_i[a,b] * R_j[a,b]
        rot_sim = torch.einsum("nab,mab->nm", rots, rots) / 3.0  # in [-1, 1]

        sim = prox + self.config.lambda_rot * rot_sim
        sim.fill_diagonal_(0.0)
        score = sim.sum(dim=1)                # (N,)
        order = torch.argsort(score, descending=True).tolist()
        return order

    # ═══════════════════════════════════════════════════════════════════
    # Wall-clock-gated view pool
    # ═══════════════════════════════════════════════════════════════════

    def _active_pool(self) -> List[int]:
        N = len(self._sorted_idx)
        if self._t0 is None:
            self._t0 = time.time()
        elapsed = time.time() - self._t0
        P = float(self.config.phase1_secs)
        k_min = max(1, int(round(self.config.phase1_frac * N)))

        # Hard switch (A4 ablation path or phase1_secs==0).
        if not getattr(self.config, "ablation_progressive_expand", True):
            if P > 0 and elapsed < P:
                return self._sorted_idx[:k_min]
            if not self._phase2_announced:
                CONSOLE.print(
                    f"[bold green]ViewCurriculum[/]: phase 2 (hard switch) at "
                    f"t={elapsed:.1f}s"
                )
                self._phase2_announced = True
            return list(range(N))

        # Progressive expansion (full method): k grows linearly from k_min to N
        # over [P/2, P]. Before P/2: phase-1 subset. After P: all views.
        if P <= 0.0:
            return list(range(N))
        if elapsed < 0.5 * P:
            return self._sorted_idx[:k_min]
        if elapsed < P:
            frac = (elapsed - 0.5 * P) / (0.5 * P)  # in [0,1]
            k = int(round(k_min + frac * (N - k_min)))
            k = max(k_min, min(N, k))
            return self._sorted_idx[:k]
        if not self._phase2_announced:
            CONSOLE.print(
                f"[bold green]ViewCurriculum[/]: phase 2 (progressive) complete "
                f"at t={elapsed:.1f}s"
            )
            self._phase2_announced = True
        return list(range(N))

    def next_train(self, step: int) -> Tuple[Cameras, Dict]:
        """Restrict next_train's candidate view pool to the currently-active subset."""
        active = set(self._active_pool())

        # Drop any queued views not in the active pool.
        self.train_unseen_cameras = [
            i for i in self.train_unseen_cameras if i in active
        ]
        # If we emptied the queue (or were already empty), refill from active pool.
        if len(self.train_unseen_cameras) == 0:
            pool = list(active)
            random.shuffle(pool)
            self.train_unseen_cameras = pool

        return super().next_train(step)
