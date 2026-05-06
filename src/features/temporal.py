"""Temporal decay for signal weighting."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd


def apply_decay(
    signals: pd.DataFrame,
    half_life_days: float,
    time_col: str | None = None,
    reference_time: datetime | None = None,
) -> pd.DataFrame:
    """
    Apply exponential decay to signal weights based on age.

    weight_decayed = weight * exp(-0.693 * age_days / half_life_days)

    A signal from half_life_days ago retains 50% of its weight.

    Parameters
    ----------
    reference_time : datetime | None
        Time used as "now" when computing age. Default: datetime.now(UTC).
        Pass max(signals[time_col]) to use the most recent observation as the
        reference — required for offline evaluation on historical datasets
        (e.g., MovieLens-100K with timestamps from 1997-1998), where decay
        relative to today's date would zero out all weights.
    """
    if signals.empty:
        return signals

    # Auto-detect time column
    if time_col is None:
        if "timestamp" in signals.columns:
            time_col = "timestamp"
        elif "created_at" in signals.columns:
            time_col = "created_at"
        else:
            return signals

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    age_seconds = (reference_time - pd.to_datetime(signals[time_col], utc=True)).dt.total_seconds()
    age_days = age_seconds / 86400.0

    decay = np.exp(-0.693 * age_days / half_life_days)
    signals = signals.copy()
    signals["weight_decayed"] = signals["weight"] * decay
    return signals
