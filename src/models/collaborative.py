"""Collaborative filtering using ALS (Alternating Least Squares).

Uses the `implicit` library when available, falls back to a simple
co-occurrence matrix when it's not installed (e.g., local dev on Windows).
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

log = logging.getLogger(__name__)

try:
    from implicit.als import AlternatingLeastSquares as _ALS

    _HAS_IMPLICIT = True
except ImportError:
    _HAS_IMPLICIT = False


class CollaborativeModel:
    """
    Learns latent embeddings for users and places from interaction data.

    When `implicit` is installed: uses ALS with configurable factors.
    When `implicit` is missing: falls back to item-item co-occurrence.
    """

    def __init__(self, factors: int = 64, regularization: float = 0.01, iterations: int = 50):
        self._factors = factors
        self._reg = regularization
        self._iterations = iterations
        self._model = None
        self._user_map: dict[str, int] = {}
        self._place_map: dict[str, int] = {}
        self._reverse_place_map: dict[int, str] = {}
        self._interaction_matrix: csr_matrix | None = None
        # Fallback co-occurrence
        self._cooccurrence: dict[str, dict[str, float]] | None = None

    def fit(self, interactions: pd.DataFrame) -> None:
        """
        Train from interactions DataFrame with columns: user_id, place_id, weight.
        """
        if interactions.empty:
            log.warning("No interactions to train collaborative model")
            return

        # Build ID mappings
        users = interactions["user_id"].unique()
        places = interactions["place_id"].unique()
        self._user_map = {uid: i for i, uid in enumerate(users)}
        self._place_map = {pid: i for i, pid in enumerate(places)}
        self._reverse_place_map = {i: pid for pid, i in self._place_map.items()}

        # Build sparse matrix (users × places)
        rows = interactions["user_id"].map(self._user_map).values
        cols = interactions["place_id"].map(self._place_map).values
        weights = interactions["weight"].clip(lower=0).values.astype(np.float32)

        self._interaction_matrix = csr_matrix(
            (weights, (rows, cols)),
            shape=(len(users), len(places)),
        )

        if _HAS_IMPLICIT:
            log.info("Training ALS model (factors=%d, iterations=%d)", self._factors, self._iterations)
            self._model = _ALS(
                factors=self._factors,
                regularization=self._reg,
                iterations=self._iterations,
                use_gpu=False,
            )
            self._model.fit(self._interaction_matrix)
        else:
            log.warning("implicit not installed — using co-occurrence fallback")
            self._build_cooccurrence(interactions)

    def recommend(self, user_id: str, n: int = 50) -> list[tuple[str, float]]:
        """Return top-N recommended places for a user."""
        if self._model is not None and _HAS_IMPLICIT:
            return self._recommend_als(user_id, n)
        if self._cooccurrence is not None:
            return self._recommend_cooccurrence(user_id, n)
        return []

    def _recommend_als(self, user_id: str, n: int) -> list[tuple[str, float]]:
        if user_id not in self._user_map:
            return []
        user_idx = self._user_map[user_id]
        item_indices, scores = self._model.recommend(
            user_idx,
            self._interaction_matrix[user_idx],
            N=n,
            filter_already_liked_items=False,
        )
        return [
            (self._reverse_place_map[idx], float(score))
            for idx, score in zip(item_indices, scores)
            if idx in self._reverse_place_map
        ]

    def _build_cooccurrence(self, interactions: pd.DataFrame) -> None:
        """Build item-item co-occurrence from shared users."""
        positive = interactions[interactions["weight"] > 0]
        user_places: dict[str, set[str]] = positive.groupby("user_id")["place_id"].apply(set).to_dict()

        cooc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for places in user_places.values():
            place_list = list(places)
            for i, a in enumerate(place_list):
                for b in place_list[i + 1:]:
                    cooc[a][b] += 1.0
                    cooc[b][a] += 1.0

        self._cooccurrence = dict(cooc)

    def _recommend_cooccurrence(self, user_id: str, n: int) -> list[tuple[str, float]]:
        if self._interaction_matrix is None or user_id not in self._user_map:
            return []

        user_idx = self._user_map[user_id]
        user_row = self._interaction_matrix[user_idx].toarray().flatten()
        liked_indices = np.where(user_row > 0)[0]
        liked_ids = {self._reverse_place_map[i] for i in liked_indices if i in self._reverse_place_map}

        scores: dict[str, float] = defaultdict(float)
        for pid in liked_ids:
            neighbors = self._cooccurrence.get(pid, {})
            for neighbor, count in neighbors.items():
                if neighbor not in liked_ids:
                    scores[neighbor] += count

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n]
