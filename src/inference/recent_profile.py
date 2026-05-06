"""Generate recent preference vectors from telemetry (last 14 days)."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.constants import RECENT_HALF_LIFE_DAYS, RECENT_WINDOW_DAYS
from src.etl.signals import SIGNAL_WEIGHTS
from src.features.temporal import apply_decay
from src.inference.normalize import normalize_weights


def generate_recent_preferences(
    user_id: str,
    telemetry: pd.DataFrame,
    places_df: pd.DataFrame,
    half_life_days: float = RECENT_HALF_LIFE_DAYS,
    window_days: int = RECENT_WINDOW_DAYS,
) -> list[dict]:
    """
    Generate recent preference vector (short-term, fast-changing).

    Returns list of dicts matching user_preference_profiles schema:
      {profile_type, dimension, dimension_value, weight}
    """
    if telemetry.empty or places_df.empty:
        return []

    # Filter to user + recent window
    cutoff = datetime.now(timezone.utc) - timedelta(days=window_days)
    user_telemetry = telemetry[
        (telemetry["user_id"] == user_id)
        & (pd.to_datetime(telemetry["created_at"], utc=True) >= cutoff)
    ]
    if user_telemetry.empty:
        return []

    # Add weight column based on event type (telemetry doesn't have it).
    # Derive from the canonical SIGNAL_WEIGHTS via the same event->signal mapping
    # used in ETL so there is a single source of truth.
    _event_to_signal = {
        "feed_position_click": "telemetry_click",
        "place_detail_view": "telemetry_detail_view",
        "impression": "telemetry_impression",
        "search_query": "telemetry_search",
        "content_view": "telemetry_click",
        "content_like": "telemetry_click",
        "route_click": "telemetry_click",
        "visit_feedback": "telemetry_detail_view",
    }
    _event_weights = {evt: SIGNAL_WEIGHTS.get(sig, 0.1) for evt, sig in _event_to_signal.items()}
    if "weight" not in user_telemetry.columns:
        user_telemetry = user_telemetry.copy()
        user_telemetry["weight"] = user_telemetry["event_type"].map(_event_weights).fillna(0.1)

    # Apply fast decay
    user_telemetry = apply_decay(user_telemetry, half_life_days)

    # Build lookups
    place_categories = dict(zip(places_df["public_id"], places_df["category"]))

    weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for _, evt in user_telemetry.iterrows():
        pid = evt.get("entity_id") or evt.get("place_id", "")
        cat = place_categories.get(pid, "")
        if not cat:
            continue

        w = evt.get("weight_decayed", evt.get("weight", 0.1))
        weights["category"][cat] += w

    # Normalize
    prefs = []
    for dimension, values in weights.items():
        for value, w in normalize_weights(dict(values)).items():
            prefs.append({
                "profile_type": "recent",
                "dimension": dimension,
                "dimension_value": value,
                "weight": round(w, 4),
            })
    return prefs
