"""Thompson Sampling contextual bandit for feed composition weights."""

from __future__ import annotations

import random as _random
from dataclasses import dataclass


@dataclass
class BanditState:
    """Beta distribution parameters for each composition weight component."""
    base_alpha: float = 2.0
    base_beta: float = 2.0
    recent_alpha: float = 2.0
    recent_beta: float = 2.0
    exploratory_alpha: float = 2.0
    exploratory_beta: float = 2.0


def sample_weights(state: BanditState, rng: _random.Random | None = None) -> tuple[float, float, float]:
    """
    Sample composition weights via Thompson Sampling.

    Returns (base_weight, recent_weight, exploratory_weight) normalized to sum=1.
    """
    r = rng or _random.Random()
    base = r.betavariate(state.base_alpha, state.base_beta)
    recent = r.betavariate(state.recent_alpha, state.recent_beta)
    exploratory = r.betavariate(state.exploratory_alpha, state.exploratory_beta)

    total = base + recent + exploratory
    if total == 0:
        return 0.4, 0.4, 0.2  # fallback defaults

    return base / total, recent / total, exploratory / total


# Reward values for different user actions
REWARD_CLICK = 0.5
REWARD_DETAIL_VIEW_LONG = 1.0    # detail_view + view_time > 10s
REWARD_CHECKIN = 2.0
REWARD_IMPRESSION_SKIP = -0.1


def compute_rewards(telemetry_events: list[dict]) -> dict[str, float]:
    """
    Compute reward per composition component from telemetry events.

    Returns {"base": total, "recent": total, "exploratory": total}
    weighted by how much each component contributed to the recommendation.

    Simplified: all positive rewards boost all components equally,
    negative rewards (skipped impressions) penalize exploratory more.
    """
    total_positive = 0.0
    total_negative = 0.0

    for evt in telemetry_events:
        event_type = evt.get("event_type", "")
        payload = evt.get("payload", {})

        if event_type == "feed_position_click":
            total_positive += REWARD_CLICK
        elif event_type == "place_detail_view":
            duration = payload.get("duration_ms", 0) if isinstance(payload, dict) else 0
            if duration > 10000:
                total_positive += REWARD_DETAIL_VIEW_LONG
            else:
                total_positive += REWARD_CLICK * 0.5
        elif event_type == "impression":
            total_negative += abs(REWARD_IMPRESSION_SKIP)

    # Distribute rewards: positive boosts all, negative penalizes exploratory more
    return {
        "base": total_positive * 0.4,
        "recent": total_positive * 0.4,
        "exploratory": total_positive * 0.2 - total_negative * 0.5,
    }


_MAX_ALPHA_BETA = 100.0


def update_state(state: BanditState, rewards: dict[str, float], learning_rate: float = 1.0) -> BanditState:
    """
    Update bandit state after observing rewards.

    Positive reward → increment alpha (success).
    Negative reward → increment beta (failure).
    Parameters are capped at _MAX_ALPHA_BETA to prevent distribution collapse.

    learning_rate > 1.0 accelerates learning for new users (asymmetric learning).
    Recommended: 3.0 for cold users, 1.5 for warm_few, 1.0 for warm_full.
    """
    for component in ("base", "recent", "exploratory"):
        reward = rewards.get(component, 0.0) * learning_rate
        alpha_key = f"{component}_alpha"
        beta_key = f"{component}_beta"

        if reward > 0:
            setattr(state, alpha_key, min(_MAX_ALPHA_BETA, getattr(state, alpha_key) + reward))
        elif reward < 0:
            setattr(state, beta_key, min(_MAX_ALPHA_BETA, getattr(state, beta_key) + abs(reward)))

    return state
