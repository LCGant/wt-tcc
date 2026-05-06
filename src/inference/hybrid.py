"""Hybrid recommender — the system described in the article.

Composes three components per user, weighted by Thompson Sampling
sampled from cold-start-level priors:

    score(u, i) = w_base * base(u, i) + w_recent * recent(u, i) + w_exp * exploratory(i)

Where:
  - base(u, i)        = α * content_score + (1-α) * collaborative_score
                        (long-term preference, half-life 90d)
  - recent(u, i)      = content_score on signals decayed with half-life 3d
                        (sensitive to recent behavior)
  - exploratory(i)    = popularity-based with controlled noise
                        (catalog discovery, anti-saturation)

Weights w_* come from:
  - Per-level priors (LEVEL_WEIGHTS in cold_start.py) when bandit state is fresh
  - Thompson Sampling from BanditState when state has accumulated rewards

The level classifier (cold_no_data / cold_onboarding / warm_few / warm_full)
determines the prior weights and the learning rate of the bandit update.
"""
from __future__ import annotations

import random as _random
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.models.bandit import BanditState, sample_weights
from src.models.cold_start import (
    ColdStartLevel,
    classify_user,
    get_composition_weights,
)
from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel
from src.constants import RECENT_HALF_LIFE_DAYS


@dataclass
class HybridConfig:
    base_content_weight: float = 0.6  # α inside base component
    use_thompson: bool = True
    use_recent: bool = True
    use_exploratory: bool = True
    use_classifier: bool = True
    exploratory_noise: float = 0.2  # additive noise on popularity scores


class HybridRecommender:
    """Composition of base + recent + exploratory components.

    All three subsystems must be `fit` independently first:
      - content_model.fit(places_df)
      - collab_model.fit(train_signals)
      - popularity is computed from train_signals on demand

    Per-user state:
      - signals (full history with weight, timestamp)
      - onboarding (set of declared categories, may be empty)
      - bandit_state (BanditState; defaults to Beta(2,2) prior)
    """

    def __init__(
        self,
        content: ContentBasedModel,
        collab: CollaborativeModel | None,
        train_signals: pd.DataFrame,
        config: HybridConfig | None = None,
        rng_seed: int = 42,
        reference_time=None,
    ) -> None:
        self.content = content
        self.collab = collab
        self.train_signals = train_signals
        self.config = config or HybridConfig()
        self._rng = _random.Random(rng_seed)
        self._np_rng = np.random.default_rng(rng_seed)
        self._popularity = self._compute_popularity(train_signals)
        # Reference time: defaults to max(timestamp) in train_signals when not provided.
        # This is critical for offline evaluation on historical datasets (e.g. MovieLens-100K),
        # where decay relative to datetime.now() would zero out all weights.
        if reference_time is None and not train_signals.empty and "timestamp" in train_signals.columns:
            try:
                reference_time = pd.to_datetime(train_signals["timestamp"], utc=True).max()
            except Exception:
                reference_time = None
        self._reference_time = reference_time

    @staticmethod
    def _compute_popularity(signals: pd.DataFrame) -> dict[str, float]:
        if signals.empty:
            return {}
        pos = signals[signals["weight"] > 0]
        if pos.empty:
            pos = signals
        agg = pos.groupby("place_id")["weight"].sum()
        if len(agg) == 0:
            return {}
        # Min-max normalize to [0, 1]
        mx = agg.max()
        if mx <= 0:
            return {}
        return (agg / mx).to_dict()

    # ── Component 1: base (content + collaborative blend) ──────
    def _base_scores(
        self,
        user_id: str,
        user_signals: pd.DataFrame,
        n: int,
    ) -> dict[str, float]:
        if user_signals.empty:
            return {}
        # Content scores from full-decay user profile
        cb_recs = self.content.recommend(user_signals, n=n * 3, reference_time=self._reference_time)
        cb_scores = {pid: s for pid, s in cb_recs}

        # Collaborative scores
        col_scores: dict[str, float] = {}
        if self.collab is not None:
            try:
                col_recs = self.collab.recommend(user_id, n=n * 3)
                col_scores = {pid: s for pid, s in col_recs}
            except Exception:
                col_scores = {}

        # Normalize each to [0,1] and blend
        def _normalize(d: dict[str, float]) -> dict[str, float]:
            if not d:
                return {}
            mn = min(d.values())
            mx = max(d.values())
            if mx - mn < 1e-9:
                return {k: 1.0 for k in d}
            return {k: (v - mn) / (mx - mn) for k, v in d.items()}

        cb_n = _normalize(cb_scores)
        col_n = _normalize(col_scores)

        alpha = self.config.base_content_weight
        all_pids = set(cb_n) | set(col_n)
        return {
            pid: alpha * cb_n.get(pid, 0.0) + (1 - alpha) * col_n.get(pid, 0.0)
            for pid in all_pids
        }

    # ── Component 2: recent (short half-life) ─────────────────
    def _recent_scores(
        self,
        user_signals: pd.DataFrame,
        n: int,
    ) -> dict[str, float]:
        if user_signals.empty:
            return {}
        # Build a temporary content model with short half-life on recent signals
        tmp = ContentBasedModel(half_life_days=RECENT_HALF_LIFE_DAYS)
        tmp._place_index = self.content._place_index  # reuse fitted index
        recs = tmp.recommend(user_signals, n=n * 3, reference_time=self._reference_time)
        if not recs:
            return {}
        scores = {pid: s for pid, s in recs}
        mx = max(scores.values()) if scores else 0
        if mx <= 0:
            return {}
        return {pid: v / mx for pid, v in scores.items()}

    # ── Component 3: exploratory (popularity + noise) ─────────
    def _exploratory_scores(
        self,
        user_seen: set[str],
        n: int,
    ) -> dict[str, float]:
        if not self._popularity:
            return {}
        candidates = [(pid, score) for pid, score in self._popularity.items() if pid not in user_seen]
        if not candidates:
            return {}
        # Add controlled gaussian noise to surface less-popular items occasionally
        noisy = {}
        for pid, score in candidates:
            noise = self._np_rng.normal(0, self.config.exploratory_noise)
            noisy[pid] = max(0.0, score + noise)
        return noisy

    # ── Main: recommend ───────────────────────────────────────
    def recommend(
        self,
        user_id: str,
        user_signals: pd.DataFrame,
        user_onboarding: pd.DataFrame | None = None,
        bandit_state: BanditState | None = None,
        n: int = 10,
    ) -> list[tuple[str, float]]:
        # 1. Classify user level
        if self.config.use_classifier:
            n_signals = len(user_signals) if user_signals is not None else 0
            has_onb = user_onboarding is not None and not user_onboarding.empty
            level = classify_user(n_signals, has_onb)
        else:
            level = ColdStartLevel.WARM_FULL

        # 2. Sample composition weights
        if self.config.use_thompson and bandit_state is not None:
            w_base, w_recent, w_exp = sample_weights(bandit_state, rng=self._rng)
            # Blend with level priors (geometric mean to stabilize)
            level_w = get_composition_weights(level)
            w_base = (w_base * level_w.base) ** 0.5 if level_w.base > 0 else 0.0
            w_recent = (w_recent * level_w.recent) ** 0.5 if level_w.recent > 0 else 0.0
            w_exp = (w_exp * level_w.exploratory) ** 0.5 if level_w.exploratory > 0 else 0.0
            tot = w_base + w_recent + w_exp
            if tot > 0:
                w_base, w_recent, w_exp = w_base / tot, w_recent / tot, w_exp / tot
        else:
            level_w = get_composition_weights(level)
            w_base, w_recent, w_exp = level_w.base, level_w.recent, level_w.exploratory

        if not self.config.use_recent:
            w_recent = 0.0
        if not self.config.use_exploratory:
            w_exp = 0.0
        # Re-normalize after possible zeros
        tot = w_base + w_recent + w_exp
        if tot > 0:
            w_base, w_recent, w_exp = w_base / tot, w_recent / tot, w_exp / tot
        else:
            w_base = 1.0

        # 3. Compute each component
        base_s = self._base_scores(user_id, user_signals, n) if w_base > 0 else {}
        recent_s = self._recent_scores(user_signals, n) if w_recent > 0 else {}
        seen = set(user_signals["place_id"].unique()) if not user_signals.empty else set()
        exp_s = self._exploratory_scores(seen, n) if w_exp > 0 else {}

        # 4. Combine
        all_pids = set(base_s) | set(recent_s) | set(exp_s)
        combined: dict[str, float] = {}
        for pid in all_pids:
            if pid in seen:
                continue
            combined[pid] = (
                w_base * base_s.get(pid, 0.0) +
                w_recent * recent_s.get(pid, 0.0) +
                w_exp * exp_s.get(pid, 0.0)
            )

        ranked = sorted(combined.items(), key=lambda x: -x[1])[:n]
        return ranked
