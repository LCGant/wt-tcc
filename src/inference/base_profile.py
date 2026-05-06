"""Generate base preference vectors from all-time signals + onboarding."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd

from src.constants import BASE_HALF_LIFE_DAYS
from src.features.temporal import apply_decay
from src.inference.normalize import normalize_weights


# Dimension mapping from onboarding question keys
_QUESTION_TO_DIMENSION = {
    "preferred_categories": "category",
    "preferred_price_levels": "price_level",
    "preferred_vibes": "highlight",
    "preferred_for": "recommended_for",
}


def generate_base_preferences(
    user_id: str,
    signals: pd.DataFrame,
    onboarding: pd.DataFrame,
    places_df: pd.DataFrame,
    half_life_days: float = BASE_HALF_LIFE_DAYS,
) -> list[dict]:
    """
    Generate base preference vector (all-time, stable).

    Returns list of dicts matching user_preference_profiles schema:
      {profile_type, dimension, dimension_value, weight}
    """
    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    # 1. Onboarding answers (weight 1.0 each)
    user_onboarding = onboarding[onboarding["user_id"] == user_id] if not onboarding.empty else pd.DataFrame()
    for _, row in user_onboarding.iterrows():
        dim = _QUESTION_TO_DIMENSION.get(row["question_key"])
        if not dim:
            continue
        for val in row["answer_values"] or []:
            weights[dim][val] += 1.0

    # 2. Interaction signals (decayed)
    user_signals = signals[signals["user_id"] == user_id] if not signals.empty else pd.DataFrame()
    if not user_signals.empty:
        user_signals = apply_decay(user_signals, half_life_days)

        # Build place_id → category lookup
        place_categories = {}
        place_prices = {}
        if not places_df.empty:
            place_categories = dict(zip(places_df["public_id"], places_df["category"]))
            place_prices = dict(zip(places_df["public_id"], places_df["price_level"]))

        for _, sig in user_signals.iterrows():
            pid = sig["place_id"]
            w = sig.get("weight_decayed", sig["weight"])

            cat = place_categories.get(pid, "")
            if cat:
                weights["category"][cat] += w

            price = place_prices.get(pid, 0)
            if price and price > 0:
                weights["price_level"][str(price)] += w * 0.3

    # 3. Normalize to [0, 1]
    prefs = []
    for dimension, values in weights.items():
        for value, w in normalize_weights(dict(values)).items():
            prefs.append({
                "profile_type": "base",
                "dimension": dimension,
                "dimension_value": value,
                "weight": round(w, 4),
            })
    return prefs
