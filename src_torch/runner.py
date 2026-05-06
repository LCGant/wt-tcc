"""End-to-end evaluation runner for the torch backend.

Glues the legacy CPU loaders + content/ALS fitters to the torch
hybrid scorer + vectorised metrics. Produces the same log line format
as the legacy ``evaluate-movielens`` command so existing aggregators
(``aggregate_movielens.py`` etc.) can parse the output unchanged.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from src.config import log as core_log  # reuse the logger configured by src/config
from src.etl.movielens import load_lastfm_2k, load_movielens_100k, load_movielens_1m
from src.models.collaborative import CollaborativeModel
from src.models.content_based import ContentBasedModel

from .bandit import BanditTensors, warmup_from_signals
from .device import select_device, use_deterministic
from .evaluation import EvalMetrics, evaluate_batch
from .hybrid import HybridArtifacts, freeze_models, score_users

log = logging.getLogger("src_torch.runner")


def evaluate_movielens_torch(
    root: str | Path,
    variant: str = "100k",
    test_days: int = 30,
    seed_value: int = 42,
    cold_fraction: float = 0.0,
    cold_keep_k: int = 5,
    user_batch_size: int = 1024,
    use_bandit: bool = True,
    use_bandit_warmup: bool = True,
    exploratory_noise: float = 0.2,
) -> EvalMetrics:
    """Evaluate the hybrid recommender on a public dataset using torch.

    Mirrors the surface of ``src.cli.evaluate_movielens`` but runs the
    per-user loop as one batched matmul on the configured device.

    Args:
        use_bandit: when ``True`` (default), composes per-user weights via
            the vectorised Thompson Sampling bandit + cold-start tier
            priors. When ``False``, falls back to fixed
            ``(0.6, 0.3, 0.1)`` weights — useful for ablation runs.
        use_bandit_warmup: when ``True`` (default), seeds bandit α/β from
            train signals via ``warmup_from_signals`` so Thompson
            Sampling differentiates per user. When ``False``, every user
            starts with the Beta(2, 2) prior (effectively inert).
        exploratory_noise: stddev of additive Gaussian noise on the
            popularity score, mirroring the legacy
            ``HybridConfig.exploratory_noise``. Set to ``0.0`` for a
            deterministic eval.
    """
    use_deterministic()
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    device = select_device()

    data = _load_dataset(variant, Path(root))
    places_df = data["places_df"]
    signals = data["signals_df"]

    train_signals, test_signals, ref_time = _temporal_split(signals, test_days)
    if cold_fraction > 0:
        train_signals = _truncate_cold(train_signals, cold_fraction, cold_keep_k, seed_value)

    core_log.info(
        "torch backend loaded: %d places, %d train signals, %d test signals",
        len(places_df), len(train_signals), len(test_signals),
    )

    # Fit legacy CPU models — fast for the dataset sizes we care about.
    t0 = time.perf_counter()
    content = ContentBasedModel()
    content.fit(places_df)

    collab = CollaborativeModel(factors=64, iterations=50)
    collab.fit(train_signals)
    fit_secs = time.perf_counter() - t0
    core_log.info("torch backend: legacy fit took %.2fs", fit_secs)

    # Popularity from train signals (positive weights only).
    pos = train_signals[train_signals["weight"] > 0]
    if pos.empty:
        popularity: dict[str, float] = {}
    else:
        counts = pos.groupby("place_id")["weight"].sum()
        max_count = float(counts.max()) if len(counts) else 1.0
        popularity = (counts / max_count).to_dict()

    # Freeze artefacts on device.
    artefacts = freeze_models(
        content_model=content,
        collab_model=collab,
        popularity=popularity,
        half_life_days=30.0,
        reference_time=ref_time,
        device=device,
    )

    # Build per-user training history dict (consumed by score_users).
    user_signals_map: dict[str, pd.DataFrame] = {
        uid: g for uid, g in train_signals.groupby("user_id")
    }

    # Determine the test-user universe and the truth tensor.
    test_pos = test_signals[test_signals["weight"] > 0]
    test_users = sorted(test_pos["user_id"].unique())
    if not test_users:
        raise RuntimeError("No positive test signals — empty evaluation set")

    truth = _build_truth_tensor(test_pos, test_users, artefacts.item_idx, device)

    # Per-user bandit state. Warmup seeds α/β from train signals so the
    # Thompson Sampling actually differentiates between users — without
    # it, all draws come from the Beta(2, 2) prior and the bandit is a
    # no-op. ``use_bandit_warmup=False`` opts out for ablations.
    if use_bandit and use_bandit_warmup:
        bandit_t0 = time.perf_counter()
        bandit = warmup_from_signals(test_users, train_signals, device)
        core_log.info(
            "torch backend: bandit warmup took %.2fs (%d users, %d signals)",
            time.perf_counter() - bandit_t0, len(test_users), len(train_signals),
        )
    elif use_bandit:
        bandit = BanditTensors.fresh(test_users, device)
    else:
        bandit = None

    # Run the eval in user-batches and aggregate metrics.
    t1 = time.perf_counter()
    aggregated: dict[str, list[float]] = {}
    seen_topk: set[int] = set()
    total_users = 0

    for batch in _batched(test_users, user_batch_size):
        scores, _ = score_users(
            artefacts,
            user_signals=user_signals_map,
            user_ids=batch,
            bandit=bandit,
            exploratory_noise=exploratory_noise,
        )
        # Slice the truth rows for this batch.
        row_idx = torch.tensor(
            [test_users.index(u) for u in batch], device=device, dtype=torch.long
        )
        truth_batch = truth.index_select(0, row_idx)
        m = evaluate_batch(scores, truth_batch, n_items_total=len(artefacts.item_ids))

        # Accumulate; weight by batch size.
        for k, v in m.to_dict().items():
            aggregated.setdefault(k, []).append(v * len(batch))
        # Track distinct top-10 items for true coverage.
        _, top10_idx = scores.topk(10, dim=1)
        for j in top10_idx.flatten().tolist():
            seen_topk.add(int(j))
        total_users += len(batch)

    eval_secs = time.perf_counter() - t1

    final = {k: sum(v) / total_users for k, v in aggregated.items()}
    final["coverage"] = len(seen_topk) / float(len(artefacts.item_ids))

    core_log.info("torch backend: eval took %.2fs (%d users)", eval_secs, total_users)
    _log_metrics(final)

    return EvalMetrics(
        precision_at_5=final["precision@5"],
        precision_at_10=final["precision@10"],
        precision_at_20=final["precision@20"],
        recall_at_5=final["recall@5"],
        recall_at_10=final["recall@10"],
        recall_at_20=final["recall@20"],
        ndcg_at_5=final["ndcg@5"],
        ndcg_at_10=final["ndcg@10"],
        ndcg_at_20=final["ndcg@20"],
        coverage_at_10=final["coverage"],
    )


# ──────────────────── helpers ────────────────────

def _load_dataset(variant: str, root: Path):
    if variant == "1m":
        return load_movielens_1m(root)
    if variant == "lastfm":
        return load_lastfm_2k(root)
    return load_movielens_100k(root)


def _temporal_split(
    signals: pd.DataFrame, test_days: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Split signals into train/test by holding out the last ``test_days``.

    Returns (train, test, reference_time). ``reference_time`` is the
    test-set boundary, used as the temporal anchor for decay weights so
    the legacy + torch paths stay aligned.
    """
    s = signals.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
    s = s.sort_values("timestamp")
    cutoff = s["timestamp"].max() - pd.Timedelta(days=test_days)
    train = s[s["timestamp"] <= cutoff].copy()
    test = s[s["timestamp"] > cutoff].copy()
    return train, test, cutoff


def _truncate_cold(
    train: pd.DataFrame, cold_fraction: float, keep_k: int, seed: int
) -> pd.DataFrame:
    """For ``cold_fraction`` of users, keep only their earliest ``keep_k`` signals."""
    rng = np.random.default_rng(seed)
    users = train["user_id"].drop_duplicates().to_numpy()
    n_cold = int(len(users) * cold_fraction)
    if n_cold <= 0:
        return train
    cold = set(rng.choice(users, size=n_cold, replace=False).tolist())
    keep_mask = np.ones(len(train), dtype=bool)
    train_sorted = train.sort_values("timestamp")
    for uid, g in train_sorted.groupby("user_id"):
        if uid in cold and len(g) > keep_k:
            drop_idx = g.iloc[keep_k:].index
            keep_mask[train_sorted.index.get_indexer(drop_idx)] = False
    return train_sorted[keep_mask].copy()


def _build_truth_tensor(
    test_pos: pd.DataFrame,
    user_order: list[str],
    item_idx: dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Materialise the ``(n_users, n_items)`` truth tensor as a dense float32.

    For our dataset sizes (~1k users × ~1.5k items = 1.5M entries × 4 bytes
    = 6 MB) the dense layout is cheaper than torch's sparse-COO machinery
    and lets ``evaluate_batch`` use a simple ``gather``.
    """
    n_users = len(user_order)
    n_items = len(item_idx)
    user_to_row = {u: i for i, u in enumerate(user_order)}

    out = torch.zeros((n_users, n_items), dtype=torch.float32, device=device)
    for uid, g in test_pos.groupby("user_id"):
        if uid not in user_to_row:
            continue
        r = user_to_row[uid]
        cols = [item_idx[p] for p in g["place_id"] if p in item_idx]
        if cols:
            out[r, torch.tensor(cols, device=device, dtype=torch.long)] = 1.0
    return out


def _batched(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _log_metrics(metrics: dict[str, float]) -> None:
    for k, v in metrics.items():
        core_log.info("metric %s = %.6f", k, v)
