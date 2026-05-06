"""Offline evaluation metrics for recommendation quality."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def precision_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Of the top K recommended, how many are relevant?"""
    top_k = recommended[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for pid in top_k if pid in relevant)
    return hits / k


def recall_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Of all relevant items, how many appear in top K?"""
    if not relevant:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k & relevant)
    return hits / len(relevant)


def ndcg_at_k(recommended: list[str], relevant: set[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at K."""
    top_k = recommended[:k]
    dcg = 0.0
    for i, pid in enumerate(top_k):
        if pid in relevant:
            dcg += 1.0 / math.log2(i + 2)  # +2 because i is 0-indexed

    # Ideal DCG: all relevant items at the top
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    if idcg == 0:
        return 0.0
    return dcg / idcg


def evaluate_model(
    recommend_fn,
    test_interactions: pd.DataFrame,
    k_values: list[int] | None = None,
    return_per_user: bool = False,
) -> dict[str, float] | tuple[dict[str, float], dict[str, dict[str, float]]]:
    """
    Evaluate a recommendation model using temporal hold-out test data.

    Args:
        recommend_fn: Callable(user_id, n) -> list[(place_id, score)]
        test_interactions: DataFrame with columns: user_id, place_id, weight
                          (only positive interactions as ground truth)
        k_values: List of K values to evaluate (default: [5, 10, 20])
        return_per_user: if True, also return {user_id: {metric: value}} for
                        statistical tests (Wilcoxon, paired t-test).

    Returns:
        - If return_per_user=False: dict of metric_name → mean value
        - If return_per_user=True: (mean dict, per-user dict)
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # Filter to positive interactions as ground truth
    positive = test_interactions[test_interactions["weight"] > 0]
    if positive.empty:
        empty = {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}
        return (empty, {}) if return_per_user else empty

    users = positive["user_id"].unique()
    results: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]
    }
    per_user: dict[str, dict[str, float]] = {}

    max_k = max(k_values)
    for uid in users:
        truth = set(positive[positive["user_id"] == uid]["place_id"])
        if not truth:
            continue

        recommendations = recommend_fn(uid, max_k)
        rec_ids = [pid for pid, _ in recommendations]

        user_metrics: dict[str, float] = {}
        for k in k_values:
            p = precision_at_k(rec_ids, truth, k)
            r = recall_at_k(rec_ids, truth, k)
            n = ndcg_at_k(rec_ids, truth, k)
            results[f"precision@{k}"].append(p)
            results[f"recall@{k}"].append(r)
            results[f"ndcg@{k}"].append(n)
            user_metrics[f"precision@{k}"] = p
            user_metrics[f"recall@{k}"] = r
            user_metrics[f"ndcg@{k}"] = n
        per_user[uid] = user_metrics

    aggregated = {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in results.items()}
    return (aggregated, per_user) if return_per_user else aggregated


# --- Diversity Metrics ---

def intra_list_diversity(recommended_categories: list[list[str]]) -> float:
    """
    Average category diversity within each recommendation list.
    1.0 = all different categories, 0.0 = all same category.
    """
    if not recommended_categories:
        return 0.0
    scores = []
    for cats in recommended_categories:
        if len(cats) <= 1:
            scores.append(0.0)
            continue
        unique = len(set(cats))
        scores.append((unique - 1) / (len(cats) - 1))
    return float(np.mean(scores))


def coverage(all_recommended: list[str], catalog_size: int) -> float:
    """Fraction of catalog that appears in at least one recommendation."""
    if catalog_size == 0:
        return 0.0
    return len(set(all_recommended)) / catalog_size


def serendipity(recommended: list[str], user_history: set[str], relevant: set[str]) -> float:
    """
    Fraction of relevant recommendations that are NOT in user's history.
    Measures unexpected-but-good recommendations.
    """
    if not recommended:
        return 0.0
    unexpected_hits = sum(1 for pid in recommended if pid in relevant and pid not in user_history)
    return unexpected_hits / len(recommended)


# --- Learning Quality Metrics ---

def onboarding_correction_rate(
    onboarding_categories: set[str],
    learned_top_categories: set[str],
) -> float:
    """
    Measures how much learned preferences diverge from onboarding.
    0.0 = no correction (identical), 1.0 = completely different.
    """
    if not onboarding_categories and not learned_top_categories:
        return 0.0
    union = onboarding_categories | learned_top_categories
    if not union:
        return 0.0
    intersection = onboarding_categories & learned_top_categories
    return 1.0 - (len(intersection) / len(union))


def confidence_evolution(alpha_history: list[float]) -> float:
    """
    Measures trend of bandit alpha parameters over time.
    Positive = growing confidence, negative = shrinking.
    Returns slope of linear regression.
    """
    if len(alpha_history) < 2:
        return 0.0
    x = np.arange(len(alpha_history), dtype=np.float64)
    y = np.array(alpha_history, dtype=np.float64)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


# --- Baselines ---

def relevance_density(test_interactions: pd.DataFrame, catalog_size: int) -> dict[str, float]:
    """
    Measure |R_u| / |P|: average number of relevant items per user, divided by catalog.
    This is the theoretical baseline for random recommendation.

    Returns dict with:
      - avg_relevant_per_user: |R_u| averaged across users
      - density: avg_relevant / catalog_size (fraction)
      - random_baseline_pct: same as density but as percentage
    """
    if catalog_size == 0 or test_interactions.empty:
        return {"avg_relevant_per_user": 0.0, "density": 0.0, "random_baseline_pct": 0.0}

    positive = test_interactions[test_interactions["weight"] > 0]
    if positive.empty:
        return {"avg_relevant_per_user": 0.0, "density": 0.0, "random_baseline_pct": 0.0}

    relevant_per_user = positive.groupby("user_id")["place_id"].nunique()
    avg_rel = float(relevant_per_user.mean())
    density = avg_rel / catalog_size
    return {
        "avg_relevant_per_user": avg_rel,
        "density": density,
        "random_baseline_pct": density * 100.0,
    }


def wilcoxon_paired(
    system_per_user: dict[str, dict[str, float]],
    baseline_per_user: dict[str, dict[str, float]],
    metric: str = "precision@10",
) -> dict[str, float]:
    """Wilcoxon signed-rank test on paired per-user metrics.

    Returns:
        statistic: W statistic
        pvalue: two-sided p-value
        n: number of paired observations (after dropping zero-difference)
        median_diff: median of (system - baseline)
        win_rate: fraction of users where system > baseline
    """
    common = set(system_per_user.keys()) & set(baseline_per_user.keys())
    if not common:
        return {"statistic": 0.0, "pvalue": 1.0, "n": 0, "median_diff": 0.0, "win_rate": 0.0}

    diffs = []
    sys_better = 0
    for uid in common:
        s = system_per_user[uid].get(metric, 0.0)
        b = baseline_per_user[uid].get(metric, 0.0)
        diffs.append(s - b)
        if s > b:
            sys_better += 1

    diffs_arr = np.array(diffs)
    nonzero = diffs_arr[diffs_arr != 0]
    if len(nonzero) < 2:
        return {
            "statistic": 0.0, "pvalue": 1.0, "n": int(len(nonzero)),
            "median_diff": float(np.median(diffs_arr)),
            "win_rate": sys_better / len(common),
        }

    try:
        from scipy.stats import wilcoxon
        result = wilcoxon(nonzero, alternative="two-sided", zero_method="zsplit")
        return {
            "statistic": float(result.statistic),
            "pvalue": float(result.pvalue),
            "n": int(len(nonzero)),
            "median_diff": float(np.median(diffs_arr)),
            "win_rate": sys_better / len(common),
        }
    except Exception:
        return {
            "statistic": 0.0, "pvalue": 1.0, "n": int(len(nonzero)),
            "median_diff": float(np.median(diffs_arr)),
            "win_rate": sys_better / len(common),
        }


def popularity_baseline(
    train_interactions: pd.DataFrame,
    test_interactions: pd.DataFrame,
    catalog_size: int,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """
    Compute P@K, NDCG@K of a popularity-only recommender.

    Recommends the top-K most popular items (by interaction count in train)
    to every user. Excludes items the user already interacted with in train.

    Returns dict {precision@K: ..., ndcg@K: ..., coverage: ...}
    """
    if k_values is None:
        k_values = [5, 10, 20]

    if train_interactions.empty or test_interactions.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "ndcg"]}

    # Global popularity ranking (most positively-interacted items first)
    train_pos = train_interactions[train_interactions["weight"] > 0]
    if train_pos.empty:
        train_pos = train_interactions
    popularity = train_pos.groupby("place_id")["weight"].sum().sort_values(ascending=False)
    popular_items = list(popularity.index)

    test_pos = test_interactions[test_interactions["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "ndcg"]}

    users = test_pos["user_id"].unique()
    max_k = max(k_values)
    results: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "ndcg"]}
    all_recs: list[str] = []

    for uid in users:
        truth = set(test_pos[test_pos["user_id"] == uid]["place_id"])
        if not truth:
            continue
        # Filter out items already seen in train
        seen = set(train_interactions[train_interactions["user_id"] == uid]["place_id"])
        recs = [p for p in popular_items if p not in seen][:max_k]
        all_recs.extend(recs)

        for k in k_values:
            results[f"precision@{k}"].append(precision_at_k(recs, truth, k))
            results[f"ndcg@{k}"].append(ndcg_at_k(recs, truth, k))

    out = {metric: float(np.mean(vals)) if vals else 0.0 for metric, vals in results.items()}
    out["coverage"] = coverage(all_recs, catalog_size)
    return out
