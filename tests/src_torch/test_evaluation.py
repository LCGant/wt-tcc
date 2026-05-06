"""Tests for ``src_torch.evaluation``.

We hand-build small score/truth tensors with predictable rankings so
the expected metric values are computable on paper. That way the
test catches both vectorisation bugs (wrong axis, off-by-one in
gather, etc.) and the IDCG / coverage formulas.

The dataclass exposes fixed slots for K ∈ {5, 10, 20}, so most tests
pin ``k_values=(5, 10, 20)`` and use catalogues big enough that those
cuts are meaningful.
"""
from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from src_torch.evaluation import EvalMetrics, _ideal_dcg, evaluate_batch


def _zeros(n_users: int, n_items: int, device) -> torch.Tensor:
    return torch.zeros((n_users, n_items), device=device)


def test_evaluate_batch_perfect_ranking(cpu_device):
    """Two users, 20 items. Each user has exactly one relevant item,
    and that item scores highest. With ``k_values=(5, 10, 20)`` we
    expect:

      Recall@K = 1.0 for every K (the lone positive is in top-1)
      Precision@K = 1/K
      NDCG@K = 1.0 (the relevant item lands at rank 1, optimal)
    """
    scores = _zeros(2, 20, cpu_device)
    scores[0, 0] = 10.0   # u1's relevant lives at index 0
    scores[1, 7] = 10.0   # u2's relevant lives at index 7

    truth = _zeros(2, 20, cpu_device)
    truth[0, 0] = 1.0
    truth[1, 7] = 1.0

    m = evaluate_batch(scores, truth, n_items_total=20)

    assert m.recall_at_5 == pytest.approx(1.0)
    assert m.recall_at_10 == pytest.approx(1.0)
    assert m.recall_at_20 == pytest.approx(1.0)

    assert m.precision_at_5 == pytest.approx(1 / 5)
    assert m.precision_at_10 == pytest.approx(1 / 10)
    assert m.precision_at_20 == pytest.approx(1 / 20)

    assert m.ndcg_at_5 == pytest.approx(1.0)
    assert m.ndcg_at_10 == pytest.approx(1.0)
    assert m.ndcg_at_20 == pytest.approx(1.0)


def test_evaluate_batch_metric_keys_match_dataclass(cpu_device):
    """Sanity: ``to_dict`` exposes every fixed K + coverage."""
    scores = _zeros(2, 20, cpu_device)
    truth = _zeros(2, 20, cpu_device)
    m = evaluate_batch(scores, truth, n_items_total=20)
    d = m.to_dict()
    expected = {
        "precision@5", "precision@10", "precision@20",
        "recall@5", "recall@10", "recall@20",
        "ndcg@5", "ndcg@10", "ndcg@20",
        "coverage",
    }
    assert set(d.keys()) == expected


def test_evaluate_batch_handles_zero_relevant(cpu_device):
    """A user with zero held-out positives must not blow up Recall (we
    clamp the divisor to 1) — Precision/NDCG just stay at 0.
    """
    scores = torch.linspace(1.0, 20.0, steps=20, device=cpu_device).unsqueeze(0)
    truth = _zeros(1, 20, cpu_device)
    m = evaluate_batch(scores, truth, n_items_total=20)
    assert m.precision_at_5 == 0.0
    assert m.recall_at_5 == 0.0
    assert m.ndcg_at_5 == 0.0


def test_evaluate_batch_coverage_counts_distinct(cpu_device):
    """Coverage = unique items in any top-10 / catalogue size.

    A single user with strictly descending scores over 20 items
    surfaces exactly the first 10 items in their top-10. The catalogue
    size is also 20, so coverage = 10/20 = 0.5.
    """
    scores = torch.arange(20, 0, -1, dtype=torch.float32, device=cpu_device).unsqueeze(0)
    truth = _zeros(1, 20, cpu_device)
    m = evaluate_batch(scores, truth, n_items_total=20)
    assert m.coverage_at_10 == pytest.approx(0.5)


def test_evaluate_batch_ndcg_against_known_dcg(cpu_device):
    """Single user, 20 items. Two relevant items: one at rank 1
    (highest score) and one at rank 3.

      DCG@5  = 1/log2(2) + 0/log2(3) + 1/log2(4) + 0 + 0 = 1 + 0.5 = 1.5
      IDCG@5 = 1/log2(2) + 1/log2(3) ≈ 1 + 0.6309 ≈ 1.6309
      NDCG@5 ≈ 0.9197
    """
    scores = torch.zeros(20, device=cpu_device)
    scores[0] = 10.0    # rank 1
    scores[5] = 5.0     # rank 2
    scores[1] = 3.0     # rank 3
    scores = scores.unsqueeze(0)
    truth = torch.zeros(20, device=cpu_device)
    truth[0] = 1.0      # in top-1 → contributes at rank 1
    truth[1] = 1.0      # in top-3 → contributes at rank 3
    truth = truth.unsqueeze(0)

    m = evaluate_batch(scores, truth, n_items_total=20)

    expected_dcg = 1.0 + 1.0 / math.log2(4)
    expected_idcg = 1.0 + 1.0 / math.log2(3)
    expected_ndcg = expected_dcg / expected_idcg
    assert m.ndcg_at_5 == pytest.approx(expected_ndcg, rel=1e-4)


def test_evaluate_batch_clamps_max_k_to_catalogue(cpu_device):
    """Tiny catalogues (here 3 items) must not crash ``torch.topk``
    when the requested K exceeds the catalogue size — the kernel
    clamps internally and reports Precision@K against the *requested*
    K so the metric semantics stay intact (1 hit / 5 = 0.2, etc.).
    """
    scores = torch.tensor([[5.0, 1.0, 3.0]], device=cpu_device)
    truth = torch.tensor([[1.0, 0.0, 0.0]], device=cpu_device)
    m = evaluate_batch(scores, truth, n_items_total=3)
    # 1 hit out of 5 = 0.2 (the *requested* K); the catalogue clamp is
    # invisible to the caller other than via Coverage.
    assert m.precision_at_5 == pytest.approx(1 / 5)
    assert m.recall_at_5 == pytest.approx(1.0)
    # Coverage uses min(10, max_k=3) = 3 unique items / 3 catalogue = 1.0
    assert m.coverage_at_10 == pytest.approx(1.0)


def test_ideal_dcg_grows_with_relevant_count(cpu_device):
    gains = 1.0 / torch.log2(torch.arange(3, dtype=torch.float32) + 2.0)
    # Users with 1, 2, 3 relevant items respectively.
    rel = torch.tensor([1.0, 2.0, 3.0])
    idcg = _ideal_dcg(rel, gains)
    assert idcg.shape == (3,)
    # Strictly monotonic — more relevants ⇒ larger ideal DCG.
    assert idcg[0] < idcg[1] < idcg[2]
    # First user gets exactly the rank-1 gain.
    assert idcg[0].item() == pytest.approx(gains[0].item())
