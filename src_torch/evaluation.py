"""Vectorised evaluation kernel.

The legacy pipeline computes Precision@K / Recall@K / NDCG@K / Coverage
inside a per-user Python loop — that loop is the dominant cost at scale
(e.g., 53K users × 2K items on goodbooks-10k makes the per-user version
intractable). This module replaces the loop with two batched torch
operations: ``torch.topk`` on the score matrix and a hits-vs-truth mask
multiplication.

All functions accept a dense score tensor of shape ``(n_users, n_items)``
and a sparse "truth" tensor of the same shape carrying 1.0 on positives
held out for testing. The score tensor is expected to have ``-inf`` on
items the user has already seen in train so they don't contaminate the
top-K. Callers (``hybrid.py``) are responsible for that masking.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EvalMetrics:
    """Aggregated metrics returned by ``evaluate_batch``.

    All values are population means across the user batch supplied to
    the evaluator. ``coverage`` reports the fraction of the catalogue
    that appeared in any user's top-K.
    """

    precision_at_5: float
    precision_at_10: float
    precision_at_20: float
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    ndcg_at_5: float
    ndcg_at_10: float
    ndcg_at_20: float
    coverage_at_10: float

    def to_dict(self) -> dict[str, float]:
        return {
            "precision@5": self.precision_at_5,
            "precision@10": self.precision_at_10,
            "precision@20": self.precision_at_20,
            "recall@5": self.recall_at_5,
            "recall@10": self.recall_at_10,
            "recall@20": self.recall_at_20,
            "ndcg@5": self.ndcg_at_5,
            "ndcg@10": self.ndcg_at_10,
            "ndcg@20": self.ndcg_at_20,
            "coverage": self.coverage_at_10,
        }


def evaluate_batch(
    scores: torch.Tensor,
    truth: torch.Tensor,
    n_items_total: int,
    k_values: tuple[int, ...] = (5, 10, 20),
) -> EvalMetrics:
    """Compute ranking metrics on a (n_users, n_items) score matrix.

    Args:
        scores: dense float tensor, ``-inf`` where the item is "seen".
        truth:  same shape, 1.0 where the held-out positive lives.
        n_items_total: catalogue size (used for ``coverage``).
        k_values: cutoffs to report (defaults to 5/10/20).

    Returns:
        ``EvalMetrics`` populated for the requested cutoffs.
    """
    n_items_actual = scores.shape[1]
    requested_max_k = max(k_values)
    # ``torch.topk`` requires k ≤ tensor.shape[dim]. Clamp so tiny test
    # catalogues (or any caller asking for top-K beyond the catalogue
    # size) just get all items in the top-N — the metric formulas still
    # hold, the trailing positions are simply empty.
    max_k = min(requested_max_k, n_items_actual)
    _, topk_idx = scores.topk(max_k, dim=1)            # (n_users, max_k)

    # Gather truth at the recommended positions: 1.0 if relevant, 0.0 else.
    hits = truth.gather(1, topk_idx)                   # (n_users, max_k)

    metrics: dict[str, float] = {}
    rel_per_user = truth.sum(dim=1).clamp(min=1.0)     # |relevant items| per user

    # Pre-compute the inverse log gains used by NDCG over positions 1..max_k.
    # gain_per_pos[r] = 1 / log2(r + 2) for r in [0, max_k)
    pos = torch.arange(max_k, device=scores.device, dtype=scores.dtype)
    gain_per_pos = 1.0 / torch.log2(pos + 2.0)

    for k in k_values:
        # When the requested K exceeds the catalogue, fall back to
        # the actual catalogue size — Precision@K is still the hit
        # count divided by the requested K, since the user *asked* for
        # K recommendations and we returned all we had.
        k_eff = min(k, max_k)
        hits_k = hits[:, :k_eff]
        # Precision@K: hits in top-K / K (use the *requested* K).
        precision = hits_k.sum(dim=1) / float(k)
        # Recall@K: hits in top-K / total relevant
        recall = hits_k.sum(dim=1) / rel_per_user
        # NDCG@K: discounted cumulative gain normalised by ideal DCG
        dcg = (hits_k * gain_per_pos[:k_eff]).sum(dim=1)
        # Ideal DCG: top-min(K, |relevant|) hits all rank perfectly.
        ideal_hits = torch.minimum(
            rel_per_user, torch.tensor(k_eff, dtype=scores.dtype, device=scores.device)
        )
        # Build a mask for positions r < ideal_hits[user] then sum gains.
        ideal_dcg = _ideal_dcg(ideal_hits, gain_per_pos[:k_eff])
        ndcg = (dcg / ideal_dcg.clamp(min=1e-12))

        metrics[f"p{k}"] = float(precision.mean())
        metrics[f"r{k}"] = float(recall.mean())
        metrics[f"n{k}"] = float(ndcg.mean())

    # Coverage@10: distinct items in any user's top-10 / catalogue size
    top10 = topk_idx[:, : min(10, max_k)]
    distinct = torch.unique(top10).numel()
    coverage = distinct / float(n_items_total) if n_items_total > 0 else 0.0

    # The dataclass exposes fixed fields for K ∈ {5, 10, 20}; if the
    # caller skipped any of those, fall back to 0.0 to avoid a noisy
    # KeyError. ``to_dict()`` still reflects whatever was measured.
    return EvalMetrics(
        precision_at_5=metrics.get("p5", 0.0),
        precision_at_10=metrics.get("p10", 0.0),
        precision_at_20=metrics.get("p20", 0.0),
        recall_at_5=metrics.get("r5", 0.0),
        recall_at_10=metrics.get("r10", 0.0),
        recall_at_20=metrics.get("r20", 0.0),
        ndcg_at_5=metrics.get("n5", 0.0),
        ndcg_at_10=metrics.get("n10", 0.0),
        ndcg_at_20=metrics.get("n20", 0.0),
        coverage_at_10=coverage,
    )


def _ideal_dcg(ideal_hits: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
    """Compute IDCG per-user given the count of relevant items they have.

    For ``ideal_hits[u] = m``, the IDCG sums the first ``m`` discount
    gains. We do this without a Python loop by expanding gains into a
    matrix and masking the tail.
    """
    k = gains.numel()
    pos = torch.arange(k, device=gains.device).unsqueeze(0)         # (1, k)
    mask = pos < ideal_hits.unsqueeze(1)                            # (n_users, k)
    return (mask.to(gains.dtype) * gains).sum(dim=1)
