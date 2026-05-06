"""Content-based recommendation using TF-IDF + cosine similarity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from src.features.place_features import PlaceFeatureIndex
from src.features.user_features import build_user_profile
from src.features.temporal import apply_decay


class ContentBasedModel:
    """
    Recommends places similar to those the user already liked.

    1. fit() — builds TF-IDF index of all places
    2. recommend() — computes user profile, ranks candidates by cosine similarity
    """

    def __init__(self, half_life_days: float = 90.0):
        self._place_index = PlaceFeatureIndex()
        self._half_life = half_life_days

    @property
    def place_index(self) -> PlaceFeatureIndex:
        return self._place_index

    def fit(self, places_df: pd.DataFrame) -> None:
        """Build TF-IDF index from places."""
        self._place_index.fit(places_df)

    def recommend(
        self,
        user_signals: pd.DataFrame,
        n: int = 50,
        reference_time=None,
    ) -> list[tuple[str, float]]:
        """
        Return top-N places ranked by cosine similarity to user profile.

        Returns: [(place_public_id, score), ...]

        reference_time: optional datetime; passed through to apply_decay so
        offline evaluation on historical datasets (e.g., MovieLens-100K) can
        use the dataset's max timestamp instead of datetime.now().
        """
        if user_signals.empty or self._place_index.matrix.shape[0] == 0:
            return []

        signals = apply_decay(user_signals, self._half_life, reference_time=reference_time)
        profile = build_user_profile(signals, self._place_index)
        if profile is None:
            return []

        scores = cosine_similarity([profile], self._place_index.matrix)[0]

        # Exclude places the user already interacted with heavily
        interacted = set(user_signals["place_id"].unique())

        top_indices = np.argsort(scores)[::-1]
        results = []
        for idx in top_indices:
            if len(results) >= n:
                break
            pid = self._place_index._place_ids[idx]
            if pid not in interacted:
                results.append((pid, float(scores[idx])))

        return results
