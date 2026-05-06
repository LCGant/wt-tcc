"""Smoke test: torch backend vs reference CPU backend.

Runs both pipelines on MovieLens-100K with the same seed and prints a
side-by-side metric comparison plus a wall-clock comparison. Metrics
are accepted within a loose tolerance (``TOL = 0.10``); the goal is
not bit-for-bit reproduction (Thompson Sampling is stochastic) but to
confirm the torch backend produces ranking quality consistent with
the reference implementation.

Usage:
    python experiments/parity_torch_vs_cpu.py --root datasets/ml-100k
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path


TOL = 0.10  # absolute tolerance per metric (sanity-check, not exact parity)


def _run_torch(root: Path, seed: int):
    from src_torch.runner import evaluate_movielens_torch

    t0 = time.perf_counter()
    metrics = evaluate_movielens_torch(
        root=str(root), variant="100k", test_days=30, seed_value=seed,
    )
    secs = time.perf_counter() - t0
    return metrics.to_dict(), secs


def _run_cpu(root: Path, seed: int):
    """Replicate the CPU evaluate-movielens hot path in-process.

    Mirrors what ``src.cli.evaluate_movielens`` does for ranking metrics
    so we can compare against the torch backend without spawning a CLI
    subprocess.
    """
    import numpy as np
    import pandas as pd
    from datetime import timedelta

    from src.etl.movielens import load_movielens_100k
    from src.models.content_based import ContentBasedModel
    from src.models.collaborative import CollaborativeModel
    from src.models.bandit import BanditState
    from src.inference.hybrid import HybridRecommender, HybridConfig
    from src.evaluation.offline_metrics import evaluate_model, coverage

    np.random.seed(seed)

    data = load_movielens_100k(root)
    places_df = data["places_df"]
    signals = data["signals_df"]
    signals["timestamp"] = pd.to_datetime(signals["timestamp"], utc=True)

    cutoff = signals["timestamp"].max() - timedelta(days=30)
    train = signals[signals["timestamp"] <= cutoff].copy()
    test = signals[signals["timestamp"] > cutoff].copy()

    content = ContentBasedModel()
    content.fit(places_df)
    collab = CollaborativeModel(factors=64, iterations=50)
    collab.fit(train)

    bandit_states: dict[str, BanditState] = {}
    hybrid = HybridRecommender(
        content=content,
        collab=collab,
        train_signals=train,
        config=HybridConfig(),
        rng_seed=seed,
        reference_time=cutoff,
    )

    test_pos = test[test["weight"] > 0]
    test_users = sorted(test_pos["user_id"].unique())

    # Cache top-10 lists so we can compute coverage without re-running.
    all_top10: list[str] = []

    def recommend_fn(uid, n):
        user_sigs = train[train["user_id"] == uid]
        state = bandit_states.setdefault(uid, BanditState())
        recs = hybrid.recommend(uid, user_sigs, None, state, n)
        if n >= 10:
            all_top10.extend(pid for pid, _ in recs[:10])
        return recs

    t0 = time.perf_counter()
    ranking = evaluate_model(recommend_fn, test_pos, k_values=[5, 10, 20])
    secs = time.perf_counter() - t0

    metrics = {
        "precision@5":  ranking["precision@5"],
        "precision@10": ranking["precision@10"],
        "precision@20": ranking["precision@20"],
        "recall@5":     ranking["recall@5"],
        "recall@10":    ranking["recall@10"],
        "recall@20":    ranking["recall@20"],
        "ndcg@5":       ranking["ndcg@5"],
        "ndcg@10":      ranking["ndcg@10"],
        "ndcg@20":      ranking["ndcg@20"],
        "coverage":     coverage(all_top10, len(places_df)),
    }
    return metrics, secs


def _print_table(cpu, torch_, cpu_t, torch_t):
    print()
    print(f"{'metric':<14} | {'cpu':>10} | {'torch':>10} | {'Δ':>10}")
    print("-" * 52)
    worst = 0.0
    for k in cpu.keys():
        a = cpu[k]; b = torch_[k]
        d = abs(a - b)
        worst = max(worst, d)
        flag = "" if d <= TOL else "  ✗"
        print(f"{k:<14} | {a:>10.6f} | {b:>10.6f} | {d:>10.6f}{flag}")
    print("-" * 52)
    print(f"{'time (s)':<14} | {cpu_t:>10.2f} | {torch_t:>10.2f} | "
          f"{cpu_t/torch_t if torch_t else float('nan'):>9.2f}× speedup")
    print()
    return worst


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/ml-100k", type=Path)
    p.add_argument("--seed", default=42, type=int)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print(f"Running CPU backend (seed={args.seed})…")
    cpu_metrics, cpu_t = _run_cpu(args.root, args.seed)

    print(f"Running torch backend (seed={args.seed})…")
    torch_metrics, torch_t = _run_torch(args.root, args.seed)

    worst = _print_table(cpu_metrics, torch_metrics, cpu_t, torch_t)
    if worst > TOL:
        print(f"FAIL — worst delta {worst:.6f} > tolerance {TOL}")
        return 1
    print(f"OK — worst delta {worst:.6f} ≤ tolerance {TOL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
