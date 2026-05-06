"""Vectorised Thompson Sampling bandit + cold-start classifier.

The reference ``BanditState`` is a per-user dataclass holding six
floats (α/β for base, recent, exploratory) and ``HybridRecommender``
draws one Beta sample per user per request via Python ``random``.

For batched eval that's a per-user Python call. Here it is replaced
by two ``(n_users, 3)`` tensors for α and β and one
``torch.distributions.Beta(α, β).sample()`` over the whole batch.

Cold-start prior weights are a ``(4, 3)`` lookup table indexed by the
classifier output, materialised once on device.

Components:
- Per-user Thompson Sampling for the three composition weights.
- Geometric-mean blend with the cold-start tier priors (matches the
  reference semantics).
- Per-user mask of unused components (``use_recent`` /
  ``use_exploratory``).
- Vectorised bandit ``update`` mirroring the reference reward update
  (positive reward → α, negative reward → β, capped at ``MAX_AB``).

Reward computation from telemetry is not migrated — it is a
low-volume ingestion path that the reference CPU code handles
adequately.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import pandas as pd
import torch


# Mirror src.models.bandit._MAX_ALPHA_BETA
MAX_AB = 100.0
DEFAULT_AB = 2.0  # Beta(2, 2) prior


class ColdLevel(IntEnum):
    """Integer-encoded cold-start level (matches src.models.cold_start)."""
    COLD_NO_DATA = 0
    COLD_ONBOARDING = 1
    WARM_FEW = 2
    WARM_FULL = 3


# Prior composition weights per level — same numbers as
# src.models.cold_start.LEVEL_WEIGHTS but laid out as a (4, 3) table
# in (base, recent, exploratory) order so we can ``index_select`` it.
_LEVEL_PRIORS: list[tuple[float, float, float]] = [
    (0.0, 0.0, 1.0),  # COLD_NO_DATA      — only popularity
    (0.6, 0.0, 0.4),  # COLD_ONBOARDING   — content + popularity
    (0.5, 0.3, 0.2),  # WARM_FEW          — balanced with explore
    (0.6, 0.3, 0.1),  # WARM_FULL         — exploit-heavy
]

WARM_FULL_THRESHOLD = 10


@dataclass
class BanditTensors:
    """Per-user α/β for the 3 composition components.

    Shapes: ``alpha`` and ``beta`` are ``(n_users, 3)`` tensors. The
    component order is fixed: column 0 = base, 1 = recent, 2 = exploratory.

    ``user_idx`` lets you map a user_id back to its row.
    """
    user_ids: list[str]
    user_idx: dict[str, int]
    alpha: torch.Tensor   # (n_users, 3)
    beta: torch.Tensor    # (n_users, 3)

    @classmethod
    def fresh(cls, user_ids: list[str], device: torch.device) -> "BanditTensors":
        """Create a tensor pair seeded with the Beta(2, 2) prior."""
        n = len(user_ids)
        alpha = torch.full((n, 3), DEFAULT_AB, device=device)
        beta = torch.full((n, 3), DEFAULT_AB, device=device)
        return cls(
            user_ids=list(user_ids),
            user_idx={u: i for i, u in enumerate(user_ids)},
            alpha=alpha,
            beta=beta,
        )

    def update(self, rewards: torch.Tensor) -> None:
        """Apply a batched reward update.

        Args:
            rewards: ``(n_users, 3)`` float tensor. Positive entries
                accumulate into ``alpha``; negative entries accumulate
                their absolute value into ``beta``. Both tensors are
                clamped to ``MAX_AB`` so the posteriors don't collapse.

        Mirrors ``src.models.bandit.update_state`` element-wise.
        """
        if rewards.shape != self.alpha.shape:
            raise ValueError(
                f"rewards shape {tuple(rewards.shape)} != "
                f"alpha shape {tuple(self.alpha.shape)}"
            )
        pos = rewards.clamp(min=0.0)
        neg = (-rewards).clamp(min=0.0)
        self.alpha = (self.alpha + pos).clamp(max=MAX_AB)
        self.beta = (self.beta + neg).clamp(max=MAX_AB)


def classify_users(
    n_signals_per_user: torch.Tensor,
    has_onboarding: torch.Tensor,
) -> torch.Tensor:
    """Vectorised cold-start classifier.

    Args:
        n_signals_per_user: ``(n_users,)`` long tensor.
        has_onboarding:    ``(n_users,)`` bool tensor.

    Returns:
        ``(n_users,)`` long tensor of ``ColdLevel`` values.
    """
    levels = torch.full_like(n_signals_per_user, fill_value=int(ColdLevel.WARM_FULL))
    has_signals = n_signals_per_user > 0
    few_signals = has_signals & (n_signals_per_user < WARM_FULL_THRESHOLD)
    no_signals = ~has_signals
    onb = has_onboarding

    levels = torch.where(no_signals & ~onb, torch.full_like(levels, int(ColdLevel.COLD_NO_DATA)), levels)
    levels = torch.where(no_signals & onb, torch.full_like(levels, int(ColdLevel.COLD_ONBOARDING)), levels)
    levels = torch.where(few_signals, torch.full_like(levels, int(ColdLevel.WARM_FEW)), levels)
    return levels


def level_priors(levels: torch.Tensor) -> torch.Tensor:
    """Look up ``(base, recent, exploratory)`` priors for each user.

    Args:
        levels: ``(n_users,)`` long tensor of ``ColdLevel`` values.

    Returns:
        ``(n_users, 3)`` float tensor on the same device as ``levels``.
    """
    table = torch.tensor(_LEVEL_PRIORS, dtype=torch.float32, device=levels.device)
    return table.index_select(0, levels.to(torch.long))


def warmup_from_signals(
    user_ids: list[str],
    train_signals: pd.DataFrame,
    device: torch.device,
    *,
    learning_rate: float = 1.0,
) -> BanditTensors:
    """Vectorised offline bandit warmup.

    Mirrors ``src.inference.bandit_warmup.warmup_all`` but vectorises
    the per-signal reward tabulation. Each signal's ``weight`` maps to
    a fixed reward triple on (base, recent, exploratory):

        weight >= 0.7 → (0.5, 0.5, 0.1)
        0.3 <= w < 0.7 → (0.3, 0.1, 0.0)
        weight < 0     → (0.0, 0.0, -0.2)
        otherwise      → (0.0, 0.0, 0.0)

    Per-user totals are summed and applied via ``BanditTensors.update``.
    The resulting tensors carry the same offline warmup signal as the
    legacy CPU dict of ``BanditState`` instances.
    """
    bandit = BanditTensors.fresh(user_ids, device)
    if train_signals is None or train_signals.empty:
        return bandit

    # Build per-row reward triples in NumPy first — much faster than
    # iterrows for the 100K-1M-row datasets we evaluate on.
    weights = train_signals["weight"].to_numpy(dtype=np.float32)
    reward_rows = np.zeros((len(weights), 3), dtype=np.float32)

    high = weights >= 0.7
    mid = (weights >= 0.3) & ~high
    neg = weights < 0
    reward_rows[high] = (0.5, 0.5, 0.1)
    reward_rows[mid] = (0.3, 0.1, 0.0)
    reward_rows[neg] = (0.0, 0.0, -0.2)
    reward_rows *= float(learning_rate)

    # Aggregate per user using a small DataFrame join.
    rewards_df = pd.DataFrame(reward_rows, columns=["base", "recent", "exp"])
    rewards_df["user_id"] = train_signals["user_id"].to_numpy()
    summed = rewards_df.groupby("user_id")[["base", "recent", "exp"]].sum()

    # Build the (n_users, 3) reward tensor aligned to ``user_ids``.
    n = len(user_ids)
    out = np.zeros((n, 3), dtype=np.float32)
    for uid, row in summed.iterrows():
        idx = bandit.user_idx.get(str(uid))
        if idx is None:
            continue
        out[idx] = row.to_numpy(dtype=np.float32)

    bandit.update(torch.from_numpy(out).to(device))
    return bandit


def sample_composition_weights(
    bandit: BanditTensors,
    levels: torch.Tensor,
    *,
    use_recent: bool = True,
    use_exploratory: bool = True,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample per-user ``(base, recent, exp)`` weights, batched.

    Steps:
        1. Draw a Beta sample per user/component from ``BanditTensors``.
        2. Geometric-mean blend with the cold-start tier prior:
           ``w_i = sqrt(thompson_i * prior_i)`` (legacy semantics).
        3. Zero out components disabled by config flags.
        4. Row-normalise so weights sum to 1; rows that collapse fall
           back to a hard "all base" assignment.
    """
    device = bandit.alpha.device

    # 1. Thompson Sampling — Beta(α, β) per user/component.
    dist = torch.distributions.Beta(bandit.alpha, bandit.beta)
    if generator is not None:
        # torch.distributions doesn't accept a Generator yet; fall back
        # to manual inverse-CDF style sampling via the Gamma trick is
        # overkill here. Reproducibility relies on the global seed set
        # by ``use_deterministic`` in src_torch.device.
        thompson = dist.sample()
    else:
        thompson = dist.sample()

    # 2. Geometric blend with priors.
    priors = level_priors(levels)            # (n_users, 3)
    blended = (thompson * priors).clamp(min=0.0).sqrt()

    # 3. Apply config-level masks.
    if not use_recent:
        blended[:, 1] = 0.0
    if not use_exploratory:
        blended[:, 2] = 0.0

    # 4. Row-normalise.
    row_sum = blended.sum(dim=1, keepdim=True)
    safe = row_sum.clamp(min=1e-12)
    weights = blended / safe

    # Rows that summed to ~0 (all priors zero or all components masked):
    # fall back to (1, 0, 0) so we still recommend something.
    fallback = torch.zeros_like(weights)
    fallback[:, 0] = 1.0
    weights = torch.where(row_sum < 1e-12, fallback, weights)
    return weights
