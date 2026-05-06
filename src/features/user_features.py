"""User feature vectors from interaction signals."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.place_features import PlaceFeatureIndex


def build_user_profile(
    user_signals: pd.DataFrame,
    place_index: PlaceFeatureIndex,
) -> np.ndarray | None:
    """
    Build a user profile as the weighted average of TF-IDF vectors
    of places the user interacted with.

    Returns None if no valid interactions.
    """
    if user_signals.empty or place_index.matrix.shape[0] == 0:
        return None

    weight_col = "weight_decayed" if "weight_decayed" in user_signals.columns else "weight"

    indices = []
    weights = []
    for _, row in user_signals.iterrows():
        idx = place_index.get_index(row["place_id"])
        if idx is not None:
            indices.append(idx)
            weights.append(row[weight_col])

    if not indices:
        return None

    weights_arr = np.array(weights, dtype=np.float64)
    total = np.abs(weights_arr).sum()
    if total == 0:
        return None

    weights_arr /= total
    vectors = place_index.matrix[indices].toarray()
    # np.average requires non-negative weights summing > 0; fall back to mean
    # when normalization produces a zero-sum (e.g. all-negative signals).
    if weights_arr.sum() == 0:
        profile = vectors.mean(axis=0)
    else:
        profile = np.average(vectors, weights=weights_arr, axis=0)
    return profile
