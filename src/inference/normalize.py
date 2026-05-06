"""Shared weight normalization for preference profiles."""

from __future__ import annotations


def normalize_weights(weights: dict[str, float], min_threshold: float = 0.01) -> dict[str, float]:
    """Normalize weight dict to [0, 1] range, dropping values below threshold."""
    if not weights:
        return {}
    max_w = max(weights.values(), default=0)
    if max_w < 1e-9:
        return {}
    return {
        k: normalized
        for k, v in weights.items()
        if (normalized := max(0.0, min(1.0, v / max(max_w, 1e-9)))) >= min_threshold
    }
