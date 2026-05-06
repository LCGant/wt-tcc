"""Tests for ``src_torch.bandit``.

Cover the four pieces independently: tensor construction, vectorised
classification, prior lookup, Beta-sample composition, reward updates,
and the offline warmup wrapper. Every test pins ``device=cpu`` so it
runs on any CI runner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from src_torch.bandit import (
    BanditTensors,
    ColdLevel,
    DEFAULT_AB,
    MAX_AB,
    classify_users,
    level_priors,
    sample_composition_weights,
    warmup_from_signals,
)


# ─────────────── BanditTensors ───────────────

def test_fresh_returns_prior_shaped_tensors(cpu_device):
    bt = BanditTensors.fresh(["u1", "u2", "u3"], cpu_device)
    assert bt.alpha.shape == (3, 3)
    assert bt.beta.shape == (3, 3)
    assert torch.allclose(bt.alpha, torch.full((3, 3), DEFAULT_AB))
    assert torch.allclose(bt.beta, torch.full((3, 3), DEFAULT_AB))
    assert bt.user_idx == {"u1": 0, "u2": 1, "u3": 2}


def test_update_splits_positive_into_alpha_negative_into_beta(cpu_device):
    bt = BanditTensors.fresh(["u1"], cpu_device)
    rewards = torch.tensor([[0.5, -0.3, 0.0]], device=cpu_device)
    bt.update(rewards)
    assert torch.allclose(bt.alpha, torch.tensor([[2.5, 2.0, 2.0]]))
    assert torch.allclose(bt.beta, torch.tensor([[2.0, 2.3, 2.0]]))


def test_update_caps_at_max(cpu_device):
    bt = BanditTensors.fresh(["u1"], cpu_device)
    rewards = torch.full((1, 3), 1e6, device=cpu_device)
    bt.update(rewards)
    assert torch.allclose(bt.alpha, torch.full((1, 3), MAX_AB))


def test_update_rejects_shape_mismatch(cpu_device):
    bt = BanditTensors.fresh(["u1", "u2"], cpu_device)
    with pytest.raises(ValueError, match="rewards shape"):
        bt.update(torch.zeros((3, 3)))


# ─────────────── classify_users ───────────────

@pytest.mark.parametrize("n_signals,has_onb,expected", [
    (0, False, ColdLevel.COLD_NO_DATA),
    (0, True, ColdLevel.COLD_ONBOARDING),
    (5, False, ColdLevel.WARM_FEW),
    (9, True, ColdLevel.WARM_FEW),
    (10, False, ColdLevel.WARM_FULL),
    (50, True, ColdLevel.WARM_FULL),
])
def test_classify_users_table(n_signals, has_onb, expected, cpu_device):
    counts = torch.tensor([n_signals], device=cpu_device, dtype=torch.long)
    onb = torch.tensor([has_onb], device=cpu_device, dtype=torch.bool)
    levels = classify_users(counts, onb)
    assert int(levels.item()) == int(expected)


def test_classify_users_batch_dimensions_preserved(cpu_device):
    counts = torch.tensor([0, 0, 5, 50], device=cpu_device, dtype=torch.long)
    onb = torch.tensor([False, True, False, True], device=cpu_device, dtype=torch.bool)
    levels = classify_users(counts, onb)
    assert levels.shape == (4,)
    assert levels.tolist() == [
        int(ColdLevel.COLD_NO_DATA),
        int(ColdLevel.COLD_ONBOARDING),
        int(ColdLevel.WARM_FEW),
        int(ColdLevel.WARM_FULL),
    ]


# ─────────────── level_priors ───────────────

def test_level_priors_returns_table_rows(cpu_device):
    levels = torch.tensor(
        [int(ColdLevel.COLD_NO_DATA), int(ColdLevel.WARM_FULL)],
        device=cpu_device, dtype=torch.long,
    )
    priors = level_priors(levels)
    # COLD_NO_DATA row → only popularity
    assert torch.allclose(priors[0], torch.tensor([0.0, 0.0, 1.0]))
    # WARM_FULL row → exploit-heavy
    assert torch.allclose(priors[1], torch.tensor([0.6, 0.3, 0.1]))


# ─────────────── sample_composition_weights ───────────────

def test_sample_weights_rows_are_stochastic(cpu_device):
    torch.manual_seed(0)
    bt = BanditTensors.fresh(["a", "b", "c", "d"], cpu_device)
    levels = torch.tensor([
        int(ColdLevel.WARM_FULL),
        int(ColdLevel.WARM_FEW),
        int(ColdLevel.COLD_ONBOARDING),
        int(ColdLevel.COLD_NO_DATA),
    ], device=cpu_device, dtype=torch.long)

    weights = sample_composition_weights(bt, levels)
    assert weights.shape == (4, 3)
    sums = weights.sum(dim=1)
    assert torch.allclose(sums, torch.ones(4), atol=1e-5)


def test_sample_weights_zero_priors_force_fallback(cpu_device):
    """COLD_NO_DATA users have priors (0, 0, 1) — only the exploratory
    column survives the geometric blend. With ``use_exploratory=False``
    every component is masked, so the function should fall back to
    ``(1, 0, 0)`` rather than emit NaNs.
    """
    torch.manual_seed(0)
    bt = BanditTensors.fresh(["only-cold"], cpu_device)
    levels = torch.tensor([int(ColdLevel.COLD_NO_DATA)], device=cpu_device, dtype=torch.long)

    weights = sample_composition_weights(bt, levels, use_exploratory=False)
    assert torch.allclose(weights[0], torch.tensor([1.0, 0.0, 0.0]))


def test_sample_weights_use_recent_flag(cpu_device):
    torch.manual_seed(0)
    bt = BanditTensors.fresh(["x"], cpu_device)
    levels = torch.tensor([int(ColdLevel.WARM_FULL)], device=cpu_device, dtype=torch.long)
    weights = sample_composition_weights(bt, levels, use_recent=False)
    assert weights[0, 1].item() == pytest.approx(0.0, abs=1e-6)


# ─────────────── warmup_from_signals ───────────────

def test_warmup_aggregates_per_user(cpu_device):
    """Two users with two signals each. weight=1.0 hits the top band
    (rewards 0.5/0.5/0.1 each). Aggregating should give α[0] - prior =
    (1.0, 1.0, 0.2) for each user.
    """
    signals = pd.DataFrame({
        "user_id": ["u1", "u1", "u2", "u2"],
        "place_id": ["p1", "p2", "p1", "p3"],
        "weight": [1.0, 1.0, 1.0, 1.0],
        "timestamp": pd.to_datetime(["2024-01-01"] * 4, utc=True),
    })
    bt = warmup_from_signals(["u1", "u2"], signals, cpu_device)
    delta_alpha = bt.alpha - DEFAULT_AB
    assert torch.allclose(delta_alpha, torch.tensor([
        [1.0, 1.0, 0.2],
        [1.0, 1.0, 0.2],
    ]), atol=1e-5)


def test_warmup_negative_weights_increment_beta(cpu_device):
    """Negative weights map to (0, 0, -0.2) → only β[exp] grows."""
    signals = pd.DataFrame({
        "user_id": ["u1"],
        "place_id": ["p1"],
        "weight": [-0.5],
        "timestamp": pd.to_datetime(["2024-01-01"], utc=True),
    })
    bt = warmup_from_signals(["u1"], signals, cpu_device)
    assert torch.allclose(bt.alpha, torch.full((1, 3), DEFAULT_AB))
    assert bt.beta[0, 2].item() == pytest.approx(DEFAULT_AB + 0.2, abs=1e-5)
    # base/recent β untouched
    assert bt.beta[0, 0].item() == pytest.approx(DEFAULT_AB)
    assert bt.beta[0, 1].item() == pytest.approx(DEFAULT_AB)


def test_warmup_empty_signals_returns_prior(cpu_device):
    bt = warmup_from_signals(["u1"], pd.DataFrame(), cpu_device)
    assert torch.allclose(bt.alpha, torch.full((1, 3), DEFAULT_AB))
    assert torch.allclose(bt.beta, torch.full((1, 3), DEFAULT_AB))


def test_warmup_unknown_user_id_skipped(cpu_device):
    """A signal whose user_id isn't in the supplied list should not
    crash the aggregation; the returned tensors only carry rows for
    known users.
    """
    signals = pd.DataFrame({
        "user_id": ["ghost", "u1"],
        "place_id": ["p1", "p1"],
        "weight": [1.0, 1.0],
        "timestamp": pd.to_datetime(["2024-01-01"] * 2, utc=True),
    })
    bt = warmup_from_signals(["u1"], signals, cpu_device)
    delta_alpha = bt.alpha - DEFAULT_AB
    # Only u1's reward should have landed in row 0.
    assert torch.allclose(delta_alpha[0], torch.tensor([0.5, 0.5, 0.1]), atol=1e-5)
