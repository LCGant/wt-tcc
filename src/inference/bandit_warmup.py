"""Offline warmup of bandit state from historical signals.

In online operation, the bandit_state accumulates Beta(α,β) updates as the
system observes user reactions to past recommendations. In offline eval,
this loop is skipped and Thompson Sampling always draws from the Beta(2,2)
prior — making the bandit effectively inert.

This module simulates a warmup pass: for each user, it splits their training
signals into K folds, simulates "the system would have recommended these
items", and applies pseudo-rewards to the bandit state. The result is
populated bandit states that diverge per-user, allowing Thompson Sampling
to actually differentiate.

The simulation is conservative — it only uses signals already in the train
set (no leakage), and applies bounded pseudo-rewards based on signal weight.
"""
from __future__ import annotations

import pandas as pd

from src.models.bandit import BanditState, update_state


def warmup_user_bandit(
    user_signals: pd.DataFrame,
    learning_rate: float = 1.0,
    initial_state: BanditState | None = None,
) -> BanditState:
    """Simulate bandit updates from a user's signal history.

    Maps each signal to a synthetic reward on the three components:
      - High positive weight (>= 0.7): treats as success on base + recent
      - Moderate weight (0.3 - 0.7): success on base only
      - Negative weight (< 0): failure on exploratory (skip-equivalent)

    Returns the resulting BanditState (may be the same instance as input,
    mutated, or a fresh one if initial_state is None).
    """
    state = initial_state or BanditState()
    if user_signals is None or user_signals.empty:
        return state

    for _, row in user_signals.iterrows():
        w = float(row.get("weight", 0))
        if w >= 0.7:
            rewards = {"base": 0.5, "recent": 0.5, "exploratory": 0.1}
        elif w >= 0.3:
            rewards = {"base": 0.3, "recent": 0.1, "exploratory": 0.0}
        elif w < 0:
            rewards = {"base": 0.0, "recent": 0.0, "exploratory": -0.2}
        else:
            continue
        update_state(state, rewards, learning_rate=learning_rate)

    return state


def warmup_all(
    train_signals: pd.DataFrame,
    learning_rate: float = 1.0,
) -> dict[str, BanditState]:
    """Build bandit_states dict by warming up each user's state from their signals.

    Returns: {user_id: BanditState}
    """
    if train_signals is None or train_signals.empty:
        return {}
    out: dict[str, BanditState] = {}
    for uid, group in train_signals.groupby("user_id"):
        out[str(uid)] = warmup_user_bandit(group, learning_rate=learning_rate)
    return out
