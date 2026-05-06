"""Reference baselines for honest comparison.

All baselines follow the same interface:
    baseline_fn(train_df, test_df, n_items, k_values=[5,10,20]) -> dict

Returns metrics: precision@K, ndcg@K, recall@K, coverage.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from src.evaluation.offline_metrics import (
    precision_at_k, recall_at_k, ndcg_at_k, coverage,
)


def _als_use_gpu() -> bool:
    """Mirror of CollaborativeModel._gpu_requested — honour AI_USE_GPU=1
    only if implicit.gpu is actually importable."""
    if os.environ.get("AI_USE_GPU", "").strip().lower() not in {"1", "true", "yes"}:
        return False
    try:
        import implicit.gpu  # noqa: F401
        return True
    except Exception:
        return False


def _temporal_train_val_split(train_df: pd.DataFrame, val_frac: float = 0.1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split train into sub-train + validation by holding out the most recent val_frac of signals.

    Used for hyperparameter tuning without test-set leakage.
    """
    if "timestamp" not in train_df.columns or train_df.empty:
        return train_df, pd.DataFrame()
    sorted_df = train_df.sort_values("timestamp")
    n = len(sorted_df)
    cut = int(n * (1.0 - val_frac))
    return sorted_df.iloc[:cut].copy(), sorted_df.iloc[cut:].copy()


def _user_truth(test_pos: pd.DataFrame) -> dict[str, set[str]]:
    return {uid: set(g["place_id"]) for uid, g in test_pos.groupby("user_id")}


def _user_seen(train_df: pd.DataFrame) -> dict[str, set[str]]:
    return {uid: set(g["place_id"]) for uid, g in train_df.groupby("user_id")}


def _aggregate(per_user_metrics: dict[str, list[float]], k_values: list[int]) -> dict[str, float]:
    return {m: float(np.mean(v)) if v else 0.0 for m, v in per_user_metrics.items()}


# ───────────────────────── Random ─────────────────────────

def random_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
    rng_seed: int = 42,
) -> dict[str, float]:
    """Recommend K random items per user (excluding seen)."""
    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    truth = _user_truth(test_pos)
    seen = _user_seen(train_df)
    catalog = list(set(train_df["place_id"]).union(set(test_df["place_id"])))

    rng = np.random.default_rng(rng_seed)
    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}

    max_k = max(k_values)
    for uid, rels in truth.items():
        candidates = [p for p in catalog if p not in seen.get(uid, set())]
        if len(candidates) == 0:
            continue
        rec = list(rng.choice(candidates, size=min(max_k, len(candidates)), replace=False))
        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    return out


# ──────────────────────── Popularity ──────────────────────

def popularity_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Recommend top-K most popular items per user (excluding seen)."""
    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        train_pos = train_df
    pop = train_pos.groupby("place_id")["weight"].sum().sort_values(ascending=False)
    popular = list(pop.index)

    truth = _user_truth(test_pos)
    seen = _user_seen(train_df)

    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}

    max_k = max(k_values)
    for uid, rels in truth.items():
        rec = [p for p in popular if p not in seen.get(uid, set())][:max_k]
        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    return out


# ─────────────────── Item-based kNN ───────────────────────

def itemknn_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
    knn_k: int = 50,
) -> dict[str, float]:
    """Item-based kNN with cosine similarity.

    Score(u, i) = sum over j in user_history(u) of cos_sim(i, j).
    """
    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    # Build user-item matrix (users as rows, items as cols)
    users = sorted(train_pos["user_id"].unique())
    items = sorted(train_pos["place_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: j for j, it in enumerate(items)}

    from scipy.sparse import csr_matrix
    rows = train_pos["user_id"].map(u_idx).to_numpy()
    cols = train_pos["place_id"].map(i_idx).to_numpy()
    data = np.ones(len(train_pos), dtype=np.float32)
    M = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    # Item-item cosine similarity
    M_T = M.T  # items x users
    norms = np.sqrt(M_T.multiply(M_T).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    M_T_norm = csr_matrix(M_T.multiply(1.0 / norms[:, None]))
    sim = (M_T_norm @ M_T_norm.T).toarray()  # items x items
    np.fill_diagonal(sim, 0)

    truth = _user_truth(test_pos)
    seen = _user_seen(train_df)

    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}

    max_k = max(k_values)
    for uid, rels in truth.items():
        user_seen = seen.get(uid, set())
        # Score every catalog item by sum of similarities to seen items
        seen_idx = [i_idx[p] for p in user_seen if p in i_idx]
        if not seen_idx:
            rec = []
        else:
            scores = sim[:, seen_idx].sum(axis=1)
            ranked = np.argsort(-scores)
            rec = []
            for j in ranked:
                pid = items[j]
                if pid in user_seen:
                    continue
                rec.append(pid)
                if len(rec) >= max_k:
                    break

        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    return out


# ──────────────────────── iALS ─────────────────────────────

# Hyperparameter grid for iALS tuning. Values follow Rendle (2022)
# "Revisiting the Performance of iALS" recommendations adapted for ML-100K.
IALS_GRID = [
    {"factors": 64, "regularization": 0.01, "alpha": 40.0, "iterations": 30},
    {"factors": 128, "regularization": 0.01, "alpha": 10.0, "iterations": 30},
    {"factors": 128, "regularization": 0.1, "alpha": 40.0, "iterations": 30},
]


def _ndcg_on_holdout(
    model, M_train, holdout: pd.DataFrame, u_idx: dict, i_idx: dict, items: list, k: int = 10
) -> float:
    """Score model on a held-out signal frame; returns mean NDCG@k."""
    pos = holdout[holdout["weight"] > 0]
    truth = {uid: set(g["place_id"]) for uid, g in pos.groupby("user_id")}
    scores = []
    for uid, rels in truth.items():
        if uid not in u_idx:
            continue
        try:
            recs_idx, _ = model.recommend(
                u_idx[uid], M_train[u_idx[uid]], N=k, filter_already_liked_items=True,
            )
            rec = [items[j] for j in recs_idx if j < len(items)]
            scores.append(ndcg_at_k(rec, rels, k))
        except Exception:
            continue
    return float(np.mean(scores)) if scores else 0.0


def _build_user_item_matrix(train_pos: pd.DataFrame, alpha: float = 1.0):
    """Helper: build sparse user-item matrix + index maps."""
    from scipy.sparse import csr_matrix
    users = sorted(train_pos["user_id"].unique())
    items = sorted(train_pos["place_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: j for j, it in enumerate(items)}
    rows = train_pos["user_id"].map(u_idx).to_numpy()
    cols = train_pos["place_id"].map(i_idx).to_numpy()
    weights = train_pos["weight"].to_numpy(dtype=np.float32) * alpha
    M = csr_matrix((weights, (rows, cols)), shape=(len(users), len(items)))
    return M, u_idx, i_idx, users, items


def ials_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
    factors: int = 64,
    iterations: int = 30,
    regularization: float = 0.01,
    alpha: float = 40.0,
    rng_seed: int = 42,
    tune: bool = False,
) -> dict[str, float]:
    """iALS (Alternating Least Squares for implicit feedback).

    Hu, Koren, Volinsky 2008. Uses the `implicit` library.

    tune=True enables a small grid search using temporal hold-out (last 10% of
    train signals as validation). The best (factors, regularization, alpha) by
    NDCG@10 is then retrained on the full train.
    """
    try:
        from implicit.als import AlternatingLeastSquares
    except ImportError:
        return {"error": "implicit not installed"}

    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    # ── Hyperparameter tuning via temporal hold-out ──
    chosen = {"factors": factors, "regularization": regularization, "alpha": alpha, "iterations": iterations}
    if tune:
        sub_train, val = _temporal_train_val_split(train_pos, val_frac=0.1)
        if not val.empty and len(sub_train) > 100:
            best_ndcg = -1.0
            for params in IALS_GRID:
                M_sub, u_sub, i_sub, _users_sub, items_sub = _build_user_item_matrix(sub_train, alpha=params["alpha"])
                try:
                    m = AlternatingLeastSquares(
                        factors=params["factors"], iterations=params["iterations"],
                        regularization=params["regularization"], random_state=rng_seed, use_gpu=_als_use_gpu(),
                    )
                    m.fit(M_sub, show_progress=False)
                    score = _ndcg_on_holdout(m, M_sub, val, u_sub, i_sub, items_sub, k=10)
                    if score > best_ndcg:
                        best_ndcg = score
                        chosen = params
                except Exception:
                    continue

    # ── Retrain on full train with chosen hyperparams ──
    M, u_idx, i_idx, users, items = _build_user_item_matrix(train_pos, alpha=chosen["alpha"])

    model = AlternatingLeastSquares(
        factors=chosen["factors"],
        iterations=chosen["iterations"],
        regularization=chosen["regularization"],
        random_state=rng_seed,
        use_gpu=_als_use_gpu(),
    )
    model.fit(M, show_progress=False)

    truth = _user_truth(test_pos)
    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}
    max_k = max(k_values)

    for uid, rels in truth.items():
        if uid not in u_idx:
            continue  # cold user, can't recommend
        recs_idx, _scores = model.recommend(
            u_idx[uid], M[u_idx[uid]], N=max_k, filter_already_liked_items=True,
        )
        rec = [items[j] for j in recs_idx if j < len(items)]
        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    out["_chosen_params"] = chosen  # for debugging / reporting
    return out


# ──────────────────────── BPR ─────────────────────────────

# Hyperparameter grid for BPR tuning. Values follow Rendle 2009 +
# common defaults from `implicit` benchmarks.
BPR_GRID = [
    {"factors": 64, "regularization": 0.01, "learning_rate": 0.05, "iterations": 100},
    {"factors": 128, "regularization": 0.01, "learning_rate": 0.01, "iterations": 200},
    {"factors": 64, "regularization": 0.001, "learning_rate": 0.1, "iterations": 100},
]


def bpr_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
    factors: int = 64,
    iterations: int = 100,
    learning_rate: float = 0.05,
    regularization: float = 0.01,
    rng_seed: int = 42,
    tune: bool = False,
) -> dict[str, float]:
    """BPR (Bayesian Personalized Ranking).

    Rendle et al. 2009. Uses the `implicit` library.

    tune=True enables a small grid search using temporal hold-out (last 10% of
    train signals as validation).
    """
    try:
        from implicit.bpr import BayesianPersonalizedRanking
    except ImportError:
        return {"error": "implicit not installed"}

    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    chosen = {
        "factors": factors, "regularization": regularization,
        "learning_rate": learning_rate, "iterations": iterations,
    }
    if tune:
        sub_train, val = _temporal_train_val_split(train_pos, val_frac=0.1)
        if not val.empty and len(sub_train) > 100:
            best_ndcg = -1.0
            for params in BPR_GRID:
                M_sub, u_sub, i_sub, _users_sub, items_sub = _build_user_item_matrix(sub_train, alpha=1.0)
                # BPR uses implicit binary signals (data=1)
                M_sub.data = np.ones_like(M_sub.data, dtype=np.float32)
                try:
                    m = BayesianPersonalizedRanking(
                        factors=params["factors"], iterations=params["iterations"],
                        learning_rate=params["learning_rate"],
                        regularization=params["regularization"],
                        random_state=rng_seed, use_gpu=_als_use_gpu(),
                    )
                    m.fit(M_sub, show_progress=False)
                    score = _ndcg_on_holdout(m, M_sub, val, u_sub, i_sub, items_sub, k=10)
                    if score > best_ndcg:
                        best_ndcg = score
                        chosen = params
                except Exception:
                    continue

    # ── Retrain on full train ──
    M, u_idx, i_idx, users, items = _build_user_item_matrix(train_pos, alpha=1.0)
    M.data = np.ones_like(M.data, dtype=np.float32)

    model = BayesianPersonalizedRanking(
        factors=chosen["factors"],
        iterations=chosen["iterations"],
        learning_rate=chosen["learning_rate"],
        regularization=chosen["regularization"],
        random_state=rng_seed,
        use_gpu=_als_use_gpu(),
    )
    model.fit(M, show_progress=False)

    truth = _user_truth(test_pos)
    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}
    max_k = max(k_values)

    for uid, rels in truth.items():
        if uid not in u_idx:
            continue
        recs_idx, _scores = model.recommend(
            u_idx[uid], M[u_idx[uid]], N=max_k, filter_already_liked_items=True,
        )
        rec = [items[j] for j in recs_idx if j < len(items)]
        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    out["_chosen_params"] = chosen
    return out


# ──────────────────────── EASE-R ─────────────────────────────

def ease_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
    regularization: float = 200.0,
) -> dict[str, float]:
    """EASE-R: Embarrassingly Shallow Autoencoder for sparse data.

    Steck 2019 (https://arxiv.org/abs/1905.03375).
    Closed-form, very strong on sparse implicit feedback. No iteration.

    Algorithm:
        G = X^T X        # item-item Gram matrix
        diag(G) += λ     # regularize
        P = G^-1
        B = -P / diag(P) # off-diagonal weights
        diag(B) = 0      # forbid self-similarity
        scores = X B
    """
    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    from scipy.sparse import csr_matrix

    users = sorted(train_pos["user_id"].unique())
    items = sorted(train_pos["place_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: j for j, it in enumerate(items)}
    rows = train_pos["user_id"].map(u_idx).to_numpy()
    cols = train_pos["place_id"].map(i_idx).to_numpy()
    data = np.ones(len(train_pos), dtype=np.float32)
    X = csr_matrix((data, (rows, cols)), shape=(len(users), len(items)))

    # G = X^T X (item-item co-occurrence)
    G = (X.T @ X).toarray().astype(np.float64)
    diag_idx = np.diag_indices_from(G)
    G[diag_idx] += regularization

    P = np.linalg.inv(G)
    B = -P / np.diag(P)[None, :]
    B[diag_idx] = 0.0
    B = B.astype(np.float32)

    truth = _user_truth(test_pos)
    seen = _user_seen(train_df)
    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}
    max_k = max(k_values)

    for uid, rels in truth.items():
        if uid not in u_idx:
            continue
        user_row = X[u_idx[uid]].toarray().ravel()
        scores = user_row @ B  # (n_items,)
        # Mask seen items
        for pid in seen.get(uid, set()):
            if pid in i_idx:
                scores[i_idx[pid]] = -np.inf
        ranked = np.argsort(-scores)
        rec = [items[j] for j in ranked[:max_k] if scores[j] > -np.inf]
        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    return out


# ──────────────────────── LinUCB ─────────────────────────────

def linucb_baseline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    k_values: list[int] | None = None,
    alpha: float = 0.1,
    n_components: int = 32,
    rng_seed: int = 42,
) -> dict[str, float]:
    """LinUCB contextual bandit (Li et al. 2010).

    Treats each item as an arm with a linear payoff model in the user's context.
    Context vectors are derived from low-rank embedding of the train interaction
    matrix (truncated SVD with `n_components` factors), avoiding raw IDs.

    For arm a, payoff θ_a^T x_user is estimated by ridge regression on observed
    rewards. Selection adds a confidence bonus α·sqrt(x^T A_a^{-1} x). With α=0
    this reduces to greedy contextual recommendation; α>0 explores.

    Note on alpha: with α=1.0 and SVD-init contexts, the exploration bonus
    dominates the mean reward, biasing the bandit toward items that have no
    training updates (their A_a stays at identity, giving the largest bonus).
    Batch evaluation has no online interleaving to recover from this. We use
    α=0.1 as the default, which keeps exploration without overwhelming the
    learned θ_a.

    This is a strong RL/bandit baseline aligned with the proposed system's
    Thompson-Sampling-over-bandit formulation. We use n_components=32 by default.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        return {f"{m}@{k}": 0.0 for k in k_values for m in ["precision", "recall", "ndcg"]}

    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    users = sorted(train_pos["user_id"].unique())
    items = sorted(train_pos["place_id"].unique())
    u_idx = {u: i for i, u in enumerate(users)}
    i_idx = {it: j for j, it in enumerate(items)}
    nU, nI = len(users), len(items)

    rows = train_pos["user_id"].map(u_idx).to_numpy()
    cols = train_pos["place_id"].map(i_idx).to_numpy()
    data = np.ones(len(train_pos), dtype=np.float32)
    M = csr_matrix((data, (rows, cols)), shape=(nU, nI))

    # User context via truncated SVD on M (M ≈ U Σ V^T). Take U Σ as user context.
    # Cap n_components by min(nU, nI) - 1 (svds requirement).
    k_svd = min(n_components, nU - 1, nI - 1)
    if k_svd < 2:
        return {"error": "dataset too small for SVD-based LinUCB"}
    try:
        U_svd, s_svd, _ = svds(M.astype(np.float32), k=k_svd, random_state=rng_seed)
        # Re-order by decreasing singular value
        order = np.argsort(-s_svd)
        U_svd = U_svd[:, order]
        s_svd = s_svd[order]
        user_context = U_svd * s_svd  # (nU, k_svd)
    except Exception as exc:
        return {"error": f"LinUCB SVD failed: {exc}"}

    d = user_context.shape[1]

    # Ridge regression per item: A_a = I + Σ x x^T over positive interactions
    #                            b_a = Σ r·x  (r=1 for positives we trained on)
    # We restrict training to top-N most-popular items to keep memory bounded.
    # Items with no positive train signal cannot be evaluated.
    A = np.tile(np.eye(d, dtype=np.float32), (nI, 1, 1))  # (nI, d, d)
    b = np.zeros((nI, d), dtype=np.float32)

    for u_id, item_idx_for_u in zip(rows, cols):
        x = user_context[u_id]
        A[item_idx_for_u] += np.outer(x, x).astype(np.float32)
        b[item_idx_for_u] += x  # reward = 1

    # Pre-compute A^{-1} and theta = A^{-1} b for each item
    theta = np.zeros((nI, d), dtype=np.float32)
    A_inv = np.zeros_like(A)
    for j in range(nI):
        try:
            A_inv[j] = np.linalg.inv(A[j])
            theta[j] = A_inv[j] @ b[j]
        except np.linalg.LinAlgError:
            A_inv[j] = np.eye(d, dtype=np.float32)
            theta[j] = np.zeros(d, dtype=np.float32)

    truth = _user_truth(test_pos)
    seen = _user_seen(train_df)
    all_recs: list[str] = []
    metrics: dict[str, list[float]] = {f"{m}@{k}": [] for k in k_values for m in ["precision", "recall", "ndcg"]}
    max_k = max(k_values)

    for uid, rels in truth.items():
        if uid not in u_idx:
            continue
        x = user_context[u_idx[uid]]
        # Mean reward
        mu = theta @ x  # (nI,)
        # Confidence bonus
        bonus = alpha * np.sqrt(np.einsum("ijk,k->ij", A_inv, x) @ x)  # equivalent to sqrt(x^T A^{-1} x) per arm
        score = mu + bonus
        # Mask seen
        for pid in seen.get(uid, set()):
            if pid in i_idx:
                score[i_idx[pid]] = -np.inf
        ranked = np.argsort(-score)
        rec = []
        for j in ranked:
            if score[j] == -np.inf:
                continue
            rec.append(items[j])
            if len(rec) >= max_k:
                break
        all_recs.extend(rec)
        for k in k_values:
            metrics[f"precision@{k}"].append(precision_at_k(rec, rels, k))
            metrics[f"recall@{k}"].append(recall_at_k(rec, rels, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(rec, rels, k))

    out = _aggregate(metrics, k_values)
    out["coverage"] = coverage(all_recs, n_items)
    out["_chosen_params"] = {"alpha": alpha, "n_components": d}
    return out


def all_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_items: int,
    rng_seed: int = 42,
    tune: bool = False,
) -> dict[str, dict[str, float]]:
    """Run all baselines and return nested dict {baseline_name: metrics}.

    tune=True triggers a small grid search inside iALS and BPR using temporal
    hold-out (last 10% of train as validation). EASE-R has a fixed closed-form
    solution and does not need tuning at this scale.
    """
    return {
        "random": random_baseline(train_df, test_df, n_items, rng_seed=rng_seed),
        "popularity": popularity_baseline(train_df, test_df, n_items),
        "itemknn": itemknn_baseline(train_df, test_df, n_items),
        "ials": ials_baseline(train_df, test_df, n_items, rng_seed=rng_seed, tune=tune),
        "bpr": bpr_baseline(train_df, test_df, n_items, rng_seed=rng_seed, tune=tune),
        "ease": ease_baseline(train_df, test_df, n_items),
        "linucb": linucb_baseline(train_df, test_df, n_items, rng_seed=rng_seed),
    }


def baselines_per_user(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    rng_seed: int = 42,
    k_values: list[int] | None = None,
    tune: bool = False,
    ials_params: dict | None = None,
    bpr_params: dict | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Same as all_baselines but returns per-user metrics for paired stats.

    Returns: {baseline_name: {user_id: {metric: value}}}

    Each baseline produces recommendations as a function:
        recommend_fn(uid, k) -> list[(place_id, score)]

    We then call evaluate_model with return_per_user=True for Wilcoxon-ready
    paired data against the system's per-user metrics.

    To avoid duplicate grid search, the caller can pass `ials_params` and
    `bpr_params` (returned in `all_baselines()`'s `_chosen_params`) so the
    per-user evaluation uses the already-tuned hyperparameters.
    """
    if k_values is None:
        k_values = [5, 10, 20]

    test_pos = test_df[test_df["weight"] > 0]
    if test_pos.empty:
        return {}

    seen = _user_seen(train_df)

    # ── Random ──
    catalog = list(set(train_df["place_id"]).union(set(test_df["place_id"])))
    rng = np.random.default_rng(rng_seed)

    def random_fn(uid: str, n: int):
        candidates = [p for p in catalog if p not in seen.get(uid, set())]
        if not candidates:
            return []
        chosen = rng.choice(candidates, size=min(n, len(candidates)), replace=False)
        return [(p, 1.0) for p in chosen]

    # ── Popularity ──
    train_pos = train_df[train_df["weight"] > 0]
    if train_pos.empty:
        train_pos = train_df
    pop_rank = train_pos.groupby("place_id")["weight"].sum().sort_values(ascending=False)
    popular = list(pop_rank.index)

    def popularity_fn(uid: str, n: int):
        rec = [p for p in popular if p not in seen.get(uid, set())][:n]
        return [(p, 1.0 / (i + 1)) for i, p in enumerate(rec)]

    # ── itemKNN ──
    knn_users = sorted(train_pos["user_id"].unique())
    knn_items = sorted(train_pos["place_id"].unique())
    knn_u_idx = {u: i for i, u in enumerate(knn_users)}
    knn_i_idx = {it: j for j, it in enumerate(knn_items)}
    knn_sim = None
    if knn_users and knn_items:
        from scipy.sparse import csr_matrix
        rows = train_pos["user_id"].map(knn_u_idx).to_numpy()
        cols = train_pos["place_id"].map(knn_i_idx).to_numpy()
        data = np.ones(len(train_pos), dtype=np.float32)
        M = csr_matrix((data, (rows, cols)), shape=(len(knn_users), len(knn_items)))
        M_T = M.T
        norms = np.sqrt(M_T.multiply(M_T).sum(axis=1)).A1
        norms[norms == 0] = 1.0
        M_T_norm = csr_matrix(M_T.multiply(1.0 / norms[:, None]))
        knn_sim = (M_T_norm @ M_T_norm.T).toarray()
        np.fill_diagonal(knn_sim, 0)

    def itemknn_fn(uid: str, n: int):
        if knn_sim is None:
            return []
        user_seen = seen.get(uid, set())
        seen_idx = [knn_i_idx[p] for p in user_seen if p in knn_i_idx]
        if not seen_idx:
            return []
        scores = knn_sim[:, seen_idx].sum(axis=1)
        ranked = np.argsort(-scores)
        rec = []
        for j in ranked:
            pid = knn_items[j]
            if pid in user_seen:
                continue
            rec.append((pid, float(scores[j])))
            if len(rec) >= n:
                break
        return rec

    # ── iALS ──
    ials_model = None
    ials_users = []
    ials_items = []
    ials_u_idx: dict = {}
    ials_M = None
    # Use caller-provided tuned params if available; else default or full grid search.
    ials_chosen = ials_params or {"factors": 64, "regularization": 0.01, "alpha": 40.0, "iterations": 30}
    try:
        from implicit.als import AlternatingLeastSquares
        from scipy.sparse import csr_matrix
        ials_users = sorted(train_pos["user_id"].unique())
        ials_items = sorted(train_pos["place_id"].unique())
        ials_u_idx = {u: i for i, u in enumerate(ials_users)}
        i_idx_2 = {it: j for j, it in enumerate(ials_items)}
        rows = train_pos["user_id"].map(ials_u_idx).to_numpy()
        cols = train_pos["place_id"].map(i_idx_2).to_numpy()
        weights = train_pos["weight"].to_numpy(dtype=np.float32) * float(ials_chosen["alpha"])
        ials_M = csr_matrix((weights, (rows, cols)), shape=(len(ials_users), len(ials_items)))
        ials_model = AlternatingLeastSquares(
            factors=int(ials_chosen["factors"]),
            iterations=int(ials_chosen["iterations"]),
            regularization=float(ials_chosen["regularization"]),
            random_state=rng_seed, use_gpu=_als_use_gpu(),
        )
        ials_model.fit(ials_M, show_progress=False)
    except Exception:
        ials_model = None

    def ials_fn(uid: str, n: int):
        if ials_model is None or uid not in ials_u_idx:
            return []
        idx = ials_u_idx[uid]
        try:
            recs_idx, scores = ials_model.recommend(
                idx, ials_M[idx], N=n, filter_already_liked_items=True,
            )
            return [(ials_items[j], float(s)) for j, s in zip(recs_idx, scores) if j < len(ials_items)]
        except Exception:
            return []

    # ── BPR ──
    bpr_model = None
    bpr_M = None
    bpr_chosen = bpr_params or {"factors": 64, "regularization": 0.01, "learning_rate": 0.05, "iterations": 100}
    try:
        from implicit.bpr import BayesianPersonalizedRanking
        from scipy.sparse import csr_matrix
        # Reuse iALS user/item maps (same train data)
        rows = train_pos["user_id"].map(ials_u_idx).to_numpy()
        cols = train_pos["place_id"].map({it: j for j, it in enumerate(ials_items)}).to_numpy()
        data = np.ones(len(train_pos), dtype=np.float32)
        bpr_M = csr_matrix((data, (rows, cols)), shape=(len(ials_users), len(ials_items)))
        bpr_model = BayesianPersonalizedRanking(
            factors=int(bpr_chosen["factors"]),
            iterations=int(bpr_chosen["iterations"]),
            learning_rate=float(bpr_chosen["learning_rate"]),
            regularization=float(bpr_chosen["regularization"]),
            random_state=rng_seed, use_gpu=_als_use_gpu(),
        )
        bpr_model.fit(bpr_M, show_progress=False)
    except Exception:
        bpr_model = None

    def bpr_fn(uid: str, n: int):
        if bpr_model is None or uid not in ials_u_idx:
            return []
        idx = ials_u_idx[uid]
        try:
            recs_idx, scores = bpr_model.recommend(
                idx, bpr_M[idx], N=n, filter_already_liked_items=True,
            )
            return [(ials_items[j], float(s)) for j, s in zip(recs_idx, scores) if j < len(ials_items)]
        except Exception:
            return []

    # ── EASE-R ──
    ease_B = None
    ease_users = ials_users
    ease_items = ials_items
    ease_u_idx = ials_u_idx
    ease_X = None
    try:
        from scipy.sparse import csr_matrix as _csr
        e_i_idx = {it: j for j, it in enumerate(ease_items)}
        rows = train_pos["user_id"].map(ease_u_idx).to_numpy()
        cols = train_pos["place_id"].map(e_i_idx).to_numpy()
        data = np.ones(len(train_pos), dtype=np.float32)
        ease_X = _csr((data, (rows, cols)), shape=(len(ease_users), len(ease_items)))
        G = (ease_X.T @ ease_X).toarray().astype(np.float64)
        diag = np.diag_indices_from(G)
        G[diag] += 200.0  # regularization
        P = np.linalg.inv(G)
        B = -P / np.diag(P)[None, :]
        B[diag] = 0.0
        ease_B = B.astype(np.float32)
    except Exception:
        ease_B = None

    def ease_fn(uid: str, n: int):
        if ease_B is None or uid not in ease_u_idx:
            return []
        e_i_idx = {it: j for j, it in enumerate(ease_items)}
        user_row = ease_X[ease_u_idx[uid]].toarray().ravel()
        scores = user_row @ ease_B
        for pid in seen.get(uid, set()):
            if pid in e_i_idx:
                scores[e_i_idx[pid]] = -np.inf
        ranked = np.argsort(-scores)
        out = []
        for j in ranked[:n]:
            if scores[j] == -np.inf:
                continue
            out.append((ease_items[j], float(scores[j])))
        return out

    # ── LinUCB (contextual bandit, Li et al. 2010) ──
    linucb_user_ctx = None
    linucb_theta = None
    linucb_A_inv = None
    linucb_users = ials_users
    linucb_items = ials_items
    linucb_u_idx = ials_u_idx
    try:
        from scipy.sparse.linalg import svds as _svds
        from scipy.sparse import csr_matrix as _csr
        nU_l, nI_l = len(linucb_users), len(linucb_items)
        e_i_idx_l = {it: j for j, it in enumerate(linucb_items)}
        rows_l = train_pos["user_id"].map(linucb_u_idx).to_numpy()
        cols_l = train_pos["place_id"].map(e_i_idx_l).to_numpy()
        data_l = np.ones(len(train_pos), dtype=np.float32)
        M_l = _csr((data_l, (rows_l, cols_l)), shape=(nU_l, nI_l))
        k_svd = min(32, nU_l - 1, nI_l - 1)
        if k_svd >= 2:
            U_svd, s_svd, _ = _svds(M_l.astype(np.float32), k=k_svd, random_state=rng_seed)
            order = np.argsort(-s_svd)
            linucb_user_ctx = U_svd[:, order] * s_svd[order]
            d_ctx = linucb_user_ctx.shape[1]
            A = np.tile(np.eye(d_ctx, dtype=np.float32), (nI_l, 1, 1))
            b_vec = np.zeros((nI_l, d_ctx), dtype=np.float32)
            for u_id, item_idx in zip(rows_l, cols_l):
                x_v = linucb_user_ctx[u_id]
                A[item_idx] += np.outer(x_v, x_v).astype(np.float32)
                b_vec[item_idx] += x_v
            linucb_theta = np.zeros((nI_l, d_ctx), dtype=np.float32)
            linucb_A_inv = np.zeros_like(A)
            for j in range(nI_l):
                try:
                    linucb_A_inv[j] = np.linalg.inv(A[j])
                    linucb_theta[j] = linucb_A_inv[j] @ b_vec[j]
                except np.linalg.LinAlgError:
                    linucb_A_inv[j] = np.eye(d_ctx, dtype=np.float32)
    except Exception:
        linucb_theta = None

    def linucb_fn(uid: str, n: int):
        if linucb_theta is None or uid not in linucb_u_idx:
            return []
        x_v = linucb_user_ctx[linucb_u_idx[uid]]
        mu = linucb_theta @ x_v
        bonus = 1.0 * np.sqrt(np.einsum("ijk,k->ij", linucb_A_inv, x_v) @ x_v)
        score = mu + bonus
        e_i_idx_l = {it: j for j, it in enumerate(linucb_items)}
        for pid in seen.get(uid, set()):
            if pid in e_i_idx_l:
                score[e_i_idx_l[pid]] = -np.inf
        ranked = np.argsort(-score)
        out = []
        for j in ranked[:n]:
            if score[j] == -np.inf:
                continue
            out.append((linucb_items[j], float(score[j])))
        return out

    # ── Evaluate each baseline with per-user output ──
    from src.evaluation.offline_metrics import evaluate_model

    fns = {
        "random": random_fn,
        "popularity": popularity_fn,
        "itemknn": itemknn_fn,
        "ials": ials_fn,
        "bpr": bpr_fn,
        "ease": ease_fn,
        "linucb": linucb_fn,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for name, fn in fns.items():
        try:
            _agg, per_user = evaluate_model(fn, test_df, k_values, return_per_user=True)
            out[name] = per_user
        except Exception:
            out[name] = {}
    return out
