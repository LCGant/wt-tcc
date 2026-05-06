"""Place feature vectors via TF-IDF."""

from __future__ import annotations

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix


class PlaceFeatureIndex:
    """
    Builds TF-IDF vectors from place attributes.

    Each place is represented as a "document" composed of:
      category + highlights + recommended_for + price_{N}

    Example: "restaurant live_music outdoor_seating craft_drinks price_2"
    """

    def __init__(self):
        self._vectorizer = TfidfVectorizer()
        self._matrix: csr_matrix | None = None
        self._place_ids: list[str] = []
        self._id_to_index: dict[str, int] = {}

    @property
    def matrix(self) -> csr_matrix:
        if self._matrix is None:
            raise RuntimeError("PlaceFeatureIndex not fitted")
        return self._matrix

    def fit(self, places_df: pd.DataFrame) -> None:
        """Build TF-IDF index from places DataFrame."""
        if places_df.empty:
            self._matrix = csr_matrix((0, 0))
            return

        documents = places_df.apply(self._to_document, axis=1)
        self._matrix = self._vectorizer.fit_transform(documents)
        self._place_ids = places_df["public_id"].tolist()
        self._id_to_index = {pid: i for i, pid in enumerate(self._place_ids)}

    def get_vector(self, place_public_id: str) -> csr_matrix | None:
        """Return TF-IDF vector for a place."""
        idx = self._id_to_index.get(place_public_id)
        if idx is None:
            return None
        return self._matrix[idx]

    def get_index(self, place_public_id: str) -> int | None:
        """Return matrix row index for a place."""
        return self._id_to_index.get(place_public_id)

    def get_all_indices(self, place_ids: list[str]) -> list[int]:
        """Return matrix row indices for multiple places, skipping unknowns."""
        return [self._id_to_index[pid] for pid in place_ids if pid in self._id_to_index]

    @staticmethod
    def _to_document(row) -> str:
        parts = [str(row.get("category", ""))]
        for h in row.get("highlights") or []:
            parts.append(str(h))
        for r in row.get("recommended_for") or []:
            parts.append(str(r))
        price = row.get("price_level", 0)
        if price and price > 0:
            parts.append(f"price_{price}")
        return " ".join(p for p in parts if p)
