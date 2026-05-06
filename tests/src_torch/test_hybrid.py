"""Tests for ``src_torch.hybrid``.

Covers:
- ``_flatten_user_signals`` correctness (concat order, unknown-item
  filtering, empty users skipped).
- ``freeze_models`` against lightweight stand-ins for the legacy
  ``ContentBasedModel`` and ``CollaborativeModel``.
- ``score_users`` end-to-end on a 2-user × 3-item synthetic case
  whose top-K is computable by hand.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src_torch.bandit import BanditTensors
from src_torch.hybrid import (
    HybridArtifacts,
    _flatten_user_signals,
    freeze_models,
    score_users,
)


# ─────────────── _flatten_user_signals ───────────────

def test_flatten_concatenates_in_user_order():
    item_idx = {"p1": 0, "p2": 1, "p3": 2}
    sigs = {
        "u1": pd.DataFrame({
            "place_id": ["p1", "p2"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "weight": [1.0, 1.0],
        }),
        "u2": pd.DataFrame({
            "place_id": ["p3"],
            "timestamp": pd.to_datetime(["2024-01-03"], utc=True),
            "weight": [1.0],
        }),
    }
    rows, items, ts = _flatten_user_signals(["u1", "u2"], sigs, item_idx)

    assert rows.tolist() == [0, 0, 1]            # row index = position in user_ids
    assert items.tolist() == [0, 1, 2]
    assert ts.shape == (3,)
    assert ts[0] < ts[1] < ts[2]                 # monotone in timestamp


def test_flatten_drops_unknown_items_and_empty_users():
    item_idx = {"p1": 0}
    sigs = {
        "u1": pd.DataFrame({
            "place_id": ["p1", "p_missing"],
            "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"], utc=True),
            "weight": [1.0, 1.0],
        }),
        "u2": pd.DataFrame(columns=["place_id", "timestamp", "weight"]),  # empty
        "u3": pd.DataFrame({
            "place_id": ["p_unknown"],
            "timestamp": pd.to_datetime(["2024-01-03"], utc=True),
            "weight": [1.0],
        }),
    }
    rows, items, ts = _flatten_user_signals(["u1", "u2", "u3"], sigs, item_idx)
    # Only u1's first signal survives the catalogue filter.
    assert rows.tolist() == [0]
    assert items.tolist() == [0]
    assert ts.shape == (1,)


def test_flatten_returns_empty_when_nothing_matches():
    rows, items, ts = _flatten_user_signals(
        ["u1"],
        {"u1": pd.DataFrame({
            "place_id": ["unknown"],
            "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
            "weight": [1.0],
        })},
        item_idx={"p1": 0},
    )
    assert rows.size == 0 and items.size == 0 and ts.size == 0


# ─────────────── freeze_models ───────────────

def _fake_content_model():
    """Stub mirroring the surface ``freeze_models`` reads."""
    matrix = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float32)
    place_index = SimpleNamespace(
        matrix=matrix,
        _place_ids=["p1", "p2", "p3"],
    )
    return SimpleNamespace(place_index=place_index)


def _fake_collab_model():
    """Stub mirroring the surface ``freeze_models`` reads from
    ``CollaborativeModel`` after a successful ``fit`` against
    ``implicit``.
    """
    user_factors = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    item_factors = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    inner = SimpleNamespace(user_factors=user_factors, item_factors=item_factors)
    return SimpleNamespace(
        _model=inner,
        _user_map={"u1": 0, "u2": 1},
        _place_map={"p1": 0, "p2": 1, "p3": 2},
        _reverse_place_map={0: "p1", 1: "p2", 2: "p3"},
    )


def test_freeze_models_normalises_content_features(cpu_device):
    art = freeze_models(
        content_model=_fake_content_model(),
        collab_model=None,
        popularity={},
        half_life_days=30.0,
        reference_time=None,
        device=cpu_device,
    )
    norms = art.item_features.norm(dim=1)
    assert torch.allclose(norms, torch.ones(3), atol=1e-5)
    assert art.item_ids == ["p1", "p2", "p3"]
    assert art.als_user_factors is None  # no collab model


def test_freeze_models_attaches_collab_factors(cpu_device):
    art = freeze_models(
        content_model=_fake_content_model(),
        collab_model=_fake_collab_model(),
        popularity={"p1": 1.0, "p2": 0.5},
        half_life_days=30.0,
        reference_time=None,
        device=cpu_device,
    )
    assert art.als_user_factors is not None
    assert art.als_item_factors.shape == (3, 2)
    assert art.item_popularity[0].item() == pytest.approx(1.0)
    assert art.item_popularity[1].item() == pytest.approx(0.5)
    assert art.item_popularity[2].item() == pytest.approx(0.0)


# ─────────────── score_users (end-to-end on tiny case) ───────────────

def _build_artefacts(cpu_device) -> HybridArtifacts:
    """Three orthonormal items; popularity (1.0, 0.5, 0.0); no collab."""
    item_features = torch.tensor([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], device=cpu_device)
    return HybridArtifacts(
        item_ids=["p1", "p2", "p3"],
        item_idx={"p1": 0, "p2": 1, "p3": 2},
        item_features=item_features,
        item_popularity=torch.tensor([1.0, 0.5, 0.0], device=cpu_device),
        als_user_ids=None,
        als_user_idx=None,
        als_user_factors=None,
        als_item_factors=None,
        half_life_seconds=30.0 * 86400.0,
        reference_time_unix=pd.Timestamp("2024-01-15", tz="UTC").timestamp(),
    )


def test_score_users_seen_items_get_minus_inf(cpu_device):
    art = _build_artefacts(cpu_device)
    user_signals = {
        "u1": pd.DataFrame({
            "place_id": ["p1"],
            "timestamp": pd.to_datetime(["2024-01-10"], utc=True),
            "weight": [1.0],
        }),
    }
    scores, _ = score_users(
        art,
        user_signals=user_signals,
        user_ids=["u1"],
        bandit=None,
        composition_weights=(1.0, 0.0, 0.0),  # base only — pure content
        exploratory_noise=0.0,
    )
    # p1 is seen → must be masked out of the top-K.
    assert scores[0, 0].item() == float("-inf")
    # p2/p3 should be finite.
    assert torch.isfinite(scores[0, 1])
    assert torch.isfinite(scores[0, 2])


def test_score_users_content_profile_picks_aligned_item(cpu_device):
    """A user whose only signal is on item 0 ([1, 0, 0]) should score
    the orthogonal item 1 ([0, 1, 0]) at 0 and item 2 ([0, 0, 1]) at 0
    — only item 0 itself aligns, and that's masked. The top-1 then
    falls back to whichever the popularity column promotes (item 0,
    but masked → item 1 wins).
    """
    art = _build_artefacts(cpu_device)
    user_signals = {
        "u1": pd.DataFrame({
            "place_id": ["p1"],
            "timestamp": pd.to_datetime(["2024-01-10"], utc=True),
            "weight": [1.0],
        }),
    }
    # 50/50 base+exploratory so popularity decides between equal-content rows.
    scores, _ = score_users(
        art,
        user_signals=user_signals,
        user_ids=["u1"],
        bandit=None,
        composition_weights=(0.5, 0.0, 0.5),
        exploratory_noise=0.0,
    )
    # p1 is masked to -inf by score_users; p2 (pop 0.5) wins over p3 (pop 0.0).
    assert int(scores[0].argmax().item()) == 1


def test_score_users_with_bandit_runs_without_error(cpu_device):
    """Smoke test: passing a fresh bandit + onboarding map should
    produce per-user weights and not blow up. We don't pin exact
    metric values because Beta sampling is stochastic.
    """
    torch.manual_seed(0)
    art = _build_artefacts(cpu_device)
    bandit = BanditTensors.fresh(["u1", "u2"], cpu_device)
    user_signals = {
        "u1": pd.DataFrame({
            "place_id": ["p1"],
            "timestamp": pd.to_datetime(["2024-01-10"], utc=True),
            "weight": [1.0],
        }),
        "u2": pd.DataFrame(columns=["place_id", "timestamp", "weight"]),
    }
    scores, ids = score_users(
        art,
        user_signals=user_signals,
        user_ids=["u1", "u2"],
        bandit=bandit,
        has_onboarding={"u2": True},
        exploratory_noise=0.0,
    )
    assert ids == ["u1", "u2"]
    assert scores.shape == (2, 3)
    # u1 has p1 seen; u2 has nothing seen.
    assert scores[0, 0].item() == float("-inf")
    assert torch.isfinite(scores[1]).all()
