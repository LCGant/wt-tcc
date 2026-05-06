"""Cold start strategy — classifies users and selects recommendation approach."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ColdStartLevel(Enum):
    COLD_NO_DATA = "cold_no_data"           # 0 interactions, no onboarding
    COLD_ONBOARDING = "cold_onboarding"     # 0 interactions, has onboarding
    WARM_FEW = "warm_few"                   # 1-9 interactions
    WARM_FULL = "warm_full"                 # 10+ interactions


@dataclass(frozen=True)
class CompositionWeights:
    exploratory: float
    base: float
    recent: float


# Composition weights per cold start level
# Composition weights per cold start level.
LEVEL_WEIGHTS: dict[ColdStartLevel, CompositionWeights] = {
    ColdStartLevel.COLD_NO_DATA: CompositionWeights(
        exploratory=1.0, base=0.0, recent=0.0,
    ),
    ColdStartLevel.COLD_ONBOARDING: CompositionWeights(
        exploratory=0.4, base=0.6, recent=0.0,
    ),
    ColdStartLevel.WARM_FEW: CompositionWeights(
        exploratory=0.2, base=0.5, recent=0.3,
    ),
    ColdStartLevel.WARM_FULL: CompositionWeights(
        exploratory=0.1, base=0.6, recent=0.3,
    ),
}

WARM_FULL_THRESHOLD = 10


def classify_user(interaction_count: int, has_onboarding: bool) -> ColdStartLevel:
    """Classify a user's cold start level based on their data availability."""
    if interaction_count == 0 and not has_onboarding:
        return ColdStartLevel.COLD_NO_DATA
    if interaction_count == 0 and has_onboarding:
        return ColdStartLevel.COLD_ONBOARDING
    if interaction_count < WARM_FULL_THRESHOLD:
        return ColdStartLevel.WARM_FEW
    return ColdStartLevel.WARM_FULL


def get_composition_weights(level: ColdStartLevel) -> CompositionWeights:
    """Return the composition weights for a cold start level."""
    return LEVEL_WEIGHTS[level]
