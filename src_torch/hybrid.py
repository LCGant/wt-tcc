"""Vectorised hybrid scorer.

Reuses the CPU-fitted ``ContentBasedModel`` and ``CollaborativeModel``
— TF-IDF over a few thousand items and ALS via the Cython ``implicit``
library, both fast on CPU — and uploads their fitted artefacts to
torch tensors for the hot path.

The hot path is the per-user scoring loop, which the reference Python
implementation runs as one iteration per user. Here it is rewritten as
a single ``(n_users, n_items)`` matmul on the device.

Composition can be either fixed weights or sampled per-user via the
vectorised Thompson Sampling bandit + cold-start classifier (see
``src_torch.bandit``). When a ``BanditTensors`` is supplied weights
are drawn per-user; otherwise the global ``composition_weights``
argument is used. Per-user min-max normalisation is taken over the
full score vector (matches the reference semantics in the limit), and
the exploratory component adds optional Gaussian noise per
(user, item) — same formula as the reference
``HybridRecommender._exploratory_scores``.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from .bandit import BanditTensors, classify_users, sample_composition_weights

log = logging.getLogger("src_torch.hybrid")


@dataclass
class HybridArtifacts:
    """Frozen view of the legacy models, packaged as torch tensors.

    Built once after the legacy CPU pipeline has fitted ``ContentBased``
    and ``Collaborative``. From here on every operation is a torch op
    on the chosen device.
    """

    # Catalogue
    item_ids: list[str]
    item_idx: dict[str, int]
    item_features: torch.Tensor            # (n_items, d_content), L2-normalised
    item_popularity: torch.Tensor          # (n_items,) in [0, 1]

    # Collaborative factors (None if implicit-ALS not fitted)
    als_user_ids: list[str] | None
    als_user_idx: dict[str, int] | None
    als_user_factors: torch.Tensor | None  # (n_users_als, d_collab)
    als_item_factors: torch.Tensor | None  # (n_items, d_collab)

    # Decay
    half_life_seconds: float
    reference_time_unix: float


def freeze_models(
    content_model,
    collab_model,
    popularity: dict[str, float],
    half_life_days: float,
    reference_time,
    device: torch.device,
) -> HybridArtifacts:
    """Materialise legacy models as torch tensors on the target device.

    Args:
        content_model: a fitted ``src.models.content_based.ContentBasedModel``.
        collab_model:  optional fitted ``src.models.collaborative.CollaborativeModel``.
        popularity:    mapping ``place_id → score`` in [0, 1].
        half_life_days: temporal decay constant for the base profile.
        reference_time: anchor for decay (e.g. last test-set timestamp).
        device:        torch device for all tensors below.

    Returns:
        A ``HybridArtifacts`` with all components on ``device``.
    """
    # Content: l2-normalise rows so cosine sim collapses to a matmul.
    place_index = content_model.place_index
    raw = place_index.matrix
    if hasattr(raw, "toarray"):
        dense = raw.toarray().astype(np.float32)
    else:
        dense = np.asarray(raw, dtype=np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    item_features = torch.from_numpy(dense / norms).to(device)

    item_ids = list(place_index._place_ids)
    item_idx = {pid: i for i, pid in enumerate(item_ids)}

    # Popularity vector aligned to item_idx; default 0 for unknown ids.
    pop_arr = np.zeros(len(item_ids), dtype=np.float32)
    for pid, score in popularity.items():
        if pid in item_idx:
            pop_arr[item_idx[pid]] = float(score)
    if pop_arr.max() > 0:
        pop_arr = pop_arr / pop_arr.max()
    item_popularity = torch.from_numpy(pop_arr).to(device)

    # Collaborative — optional. We translate `implicit`'s factor matrices.
    als_user_ids = als_user_factors = als_item_factors = None
    als_user_idx = None
    if (
        collab_model is not None
        and getattr(collab_model, "_model", None) is not None
    ):
        try:
            uf = collab_model._model.user_factors
            itf = collab_model._model.item_factors
            uf_np = _as_numpy(uf)
            itf_np = _as_numpy(itf)

            # Reorder ALS item factors so row i matches item_idx[i].
            als_item_to_global = collab_model._reverse_place_map  # idx → place_id
            global_to_local = {pid: i for i, pid in enumerate(item_ids)}
            n_items = len(item_ids)
            d = itf_np.shape[1]
            reordered = np.zeros((n_items, d), dtype=np.float32)
            for als_idx, pid in als_item_to_global.items():
                if pid in global_to_local:
                    reordered[global_to_local[pid]] = itf_np[als_idx]

            als_user_ids = list(collab_model._user_map.keys())
            als_user_idx = dict(collab_model._user_map)
            als_user_factors = torch.from_numpy(uf_np.astype(np.float32)).to(device)
            als_item_factors = torch.from_numpy(reordered).to(device)
        except Exception as exc:  # pragma: no cover — defensive
            log.warning("Could not import ALS factors into torch: %s", exc)

    return HybridArtifacts(
        item_ids=item_ids,
        item_idx=item_idx,
        item_features=item_features,
        item_popularity=item_popularity,
        als_user_ids=als_user_ids,
        als_user_idx=als_user_idx,
        als_user_factors=als_user_factors,
        als_item_factors=als_item_factors,
        half_life_seconds=half_life_days * 86400.0,
        reference_time_unix=_to_unix(reference_time),
    )


def score_users(
    art: HybridArtifacts,
    user_signals: dict[str, pd.DataFrame],
    user_ids: list[str],
    composition_weights: tuple[float, float, float] | None = (0.6, 0.3, 0.1),
    base_content_weight: float = 0.6,
    seen_mask_value: float = float("-inf"),
    bandit: BanditTensors | None = None,
    has_onboarding: dict[str, bool] | None = None,
    exploratory_noise: float = 0.0,
    use_recent: bool = True,
    use_exploratory: bool = True,
) -> tuple[torch.Tensor, list[str]]:
    """Score a batch of users against the full catalogue.

    Args:
        art: frozen artefacts.
        user_signals: per-user training history (the same dict the
            legacy ``HybridRecommender.recommend`` consumes).
        user_ids: which users to score (subset of ``user_signals``).
        composition_weights: ``(base, recent, exploratory)`` fallback
            used when ``bandit`` is ``None``. If both ``bandit`` and
            ``composition_weights`` are provided, the bandit wins.
        base_content_weight: α in ``α·content + (1-α)·collab`` inside
            the base component.
        seen_mask_value: value to write into score positions the user
            already interacted with — ``-inf`` keeps them off the top-K
            without polluting metric computations.
        bandit: optional ``BanditTensors`` to sample per-user weights
            via Thompson Sampling. When supplied, the function blends
            its samples with cold-start tier priors (legacy semantics).
        has_onboarding: mapping ``user_id → bool`` for the cold-start
            classifier. Defaults to all-False when missing.
        exploratory_noise: stddev of additive Gaussian noise on the
            popularity score (mirrors the legacy
            ``HybridConfig.exploratory_noise``). 0 = deterministic.
        use_recent / use_exploratory: gate flags mirroring
            ``HybridConfig``; when False the corresponding column is
            zeroed before normalisation.

    Returns:
        ``(scores, user_ids)`` where ``scores`` has shape
        ``(len(user_ids), n_items)`` on ``art.item_features.device``.
    """
    device = art.item_features.device
    n_items = art.item_features.shape[0]
    d_content = art.item_features.shape[1]
    n_batch = len(user_ids)

    seen = torch.zeros((n_batch, n_items), dtype=torch.bool, device=device)

    half_life_full = art.half_life_seconds
    half_life_recent = max(half_life_full * 0.05, 86400.0)  # ~5% of base, min 1 day
    ref_t = art.reference_time_unix

    # Flatten every batch user's signal history into three parallel
    # numpy arrays (user_row, item_idx, ts_unix). This keeps the
    # per-user Python work to a single dict lookup + numpy concat,
    # avoiding the previous nested loop.
    flat_rows, flat_items, flat_ts = _flatten_user_signals(
        user_ids, user_signals, art.item_idx,
    )

    if flat_rows.size > 0:
        # Upload once; from here on every op runs on the device.
        user_row_t = torch.from_numpy(flat_rows).to(device)
        item_idx_t = torch.from_numpy(flat_items).to(device)
        ts_t = torch.from_numpy(flat_ts).to(device)

        # Decay weights — broadcast across the whole flat signal table.
        age = (ref_t - ts_t).clamp(min=0.0)
        decay_full = torch.exp(-age * (math.log(2.0) / half_life_full))
        decay_recent = torch.exp(-age * (math.log(2.0) / half_life_recent))

        # Mark seen items via advanced indexing.
        seen[user_row_t, item_idx_t] = True

        # Per-user weight totals via index_add_ on a (n_batch,) accumulator.
        w_total_full = torch.zeros(n_batch, device=device)
        w_total_recent = torch.zeros(n_batch, device=device)
        w_total_full.index_add_(0, user_row_t, decay_full)
        w_total_recent.index_add_(0, user_row_t, decay_recent)

        # Per-user weighted feature sums via index_add_ on (n_batch, d).
        feats = art.item_features.index_select(0, item_idx_t)        # (N, d)
        profiles_full = torch.zeros((n_batch, d_content), device=device)
        profiles_recent = torch.zeros((n_batch, d_content), device=device)
        profiles_full.index_add_(0, user_row_t, decay_full.unsqueeze(1) * feats)
        profiles_recent.index_add_(0, user_row_t, decay_recent.unsqueeze(1) * feats)

        # Mean profile per user; rows with zero weight stay zero.
        profiles_full = profiles_full / w_total_full.unsqueeze(1).clamp(min=1e-12)
        profiles_recent = profiles_recent / w_total_recent.unsqueeze(1).clamp(min=1e-12)
    else:
        profiles_full = torch.zeros((n_batch, d_content), device=device)
        profiles_recent = torch.zeros((n_batch, d_content), device=device)

    # Cosine sim = profile · item_features.T (re-normalise to be safe).
    p_full = _l2_normalise(profiles_full)
    p_recent = _l2_normalise(profiles_recent)

    content_scores_full = p_full @ art.item_features.T          # (n_batch, n_items)
    content_scores_recent = p_recent @ art.item_features.T      # (n_batch, n_items)

    # Collaborative component, when ALS was fitted.
    collab_scores = torch.zeros_like(content_scores_full)
    if art.als_user_factors is not None and art.als_user_idx is not None:
        # Map batch users to ALS row indices when possible.
        als_rows = []
        target_rows = []
        for row, uid in enumerate(user_ids):
            j = art.als_user_idx.get(uid)
            if j is not None:
                als_rows.append(j)
                target_rows.append(row)
        if als_rows:
            uf = art.als_user_factors[torch.tensor(als_rows, device=device)]
            partial = uf @ art.als_item_factors.T               # (n_known, n_items)
            collab_scores[torch.tensor(target_rows, device=device)] = partial

    # Base = α·content_full + (1-α)·collab, normalised per row.
    base = (
        base_content_weight * _row_minmax(content_scores_full)
        + (1.0 - base_content_weight) * _row_minmax(collab_scores)
    )

    # Recent = content with shorter decay, row-normalised.
    recent = _row_minmax(content_scores_recent)

    # Exploratory = popularity broadcast + optional per-(user,item) noise.
    # Noise stddev mirrors HybridConfig.exploratory_noise; 0 disables it
    # for deterministic offline runs.
    exp_scores = art.item_popularity.unsqueeze(0).expand(n_batch, -1).clone()
    if exploratory_noise > 0:
        noise = torch.randn_like(exp_scores) * float(exploratory_noise)
        exp_scores = (exp_scores + noise).clamp(min=0.0)

    # Per-user composition weights — bandit when supplied, otherwise the
    # fixed fallback tuple. The bandit path also pulls in the cold-start
    # classifier so each user gets the right tier prior.
    if bandit is not None:
        levels = _classify_batch(user_ids, user_signals, has_onboarding, device)
        # Align bandit rows to this batch (some test users may not have
        # a row yet → seed with the prior on the fly).
        batch_alpha, batch_beta = _gather_bandit_rows(bandit, user_ids, device)
        scratch = BanditTensors(
            user_ids=list(user_ids),
            user_idx={u: i for i, u in enumerate(user_ids)},
            alpha=batch_alpha,
            beta=batch_beta,
        )
        weights = sample_composition_weights(
            scratch, levels,
            use_recent=use_recent,
            use_exploratory=use_exploratory,
        )                                                  # (n_batch, 3)
    else:
        cw = composition_weights or (1.0, 0.0, 0.0)
        weights = torch.tensor(cw, device=device, dtype=base.dtype).unsqueeze(0).expand(n_batch, -1)

    w_base = weights[:, 0:1]
    w_recent = weights[:, 1:2]
    w_exp = weights[:, 2:3]
    scores = w_base * base + w_recent * recent + w_exp * exp_scores

    # Mask seen items
    scores = scores.masked_fill(seen, seen_mask_value)
    return scores, list(user_ids)


def _classify_batch(
    user_ids: list[str],
    user_signals: dict[str, pd.DataFrame],
    has_onboarding: dict[str, bool] | None,
    device: torch.device,
) -> torch.Tensor:
    """Build the cold-start ``levels`` tensor for a batch of users."""
    n = len(user_ids)
    counts = torch.zeros(n, dtype=torch.long, device=device)
    onb = torch.zeros(n, dtype=torch.bool, device=device)
    for i, uid in enumerate(user_ids):
        sigs = user_signals.get(uid)
        counts[i] = 0 if sigs is None else int(len(sigs))
        if has_onboarding is not None:
            onb[i] = bool(has_onboarding.get(uid, False))
    return classify_users(counts, onb)


def _gather_bandit_rows(
    bandit: BanditTensors,
    user_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(alpha, beta)`` rows aligned to ``user_ids``.

    Users not present in the bandit (e.g. test-only users) get fresh
    Beta(2, 2) rows so they still receive a sensible Thompson sample.
    """
    from .bandit import DEFAULT_AB
    n = len(user_ids)
    alpha = torch.full((n, 3), DEFAULT_AB, device=device)
    beta = torch.full((n, 3), DEFAULT_AB, device=device)
    rows_known: list[int] = []
    rows_target: list[int] = []
    for i, uid in enumerate(user_ids):
        j = bandit.user_idx.get(uid)
        if j is not None:
            rows_known.append(j)
            rows_target.append(i)
    if rows_known:
        src_idx = torch.tensor(rows_known, device=device, dtype=torch.long)
        dst_idx = torch.tensor(rows_target, device=device, dtype=torch.long)
        alpha.index_copy_(0, dst_idx, bandit.alpha.index_select(0, src_idx).to(device))
        beta.index_copy_(0, dst_idx, bandit.beta.index_select(0, src_idx).to(device))
    return alpha, beta


# ──────────────────── helpers ────────────────────

def _flatten_user_signals(
    user_ids: list[str],
    user_signals: dict[str, pd.DataFrame],
    item_idx: dict[str, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten per-user histories into three parallel numpy arrays.

    Returns ``(user_rows, item_idxs, ts_unix)`` covering every (user,
    item) pair the catalogue still knows about. Empty users contribute
    nothing. The caller can upload the arrays to the device in one
    transfer and use ``index_add_`` for the per-user aggregations.
    """
    row_chunks: list[np.ndarray] = []
    pid_chunks: list[np.ndarray] = []
    ts_chunks: list[np.ndarray] = []
    for row, uid in enumerate(user_ids):
        sigs = user_signals.get(uid)
        if sigs is None or sigs.empty:
            continue
        n = len(sigs)
        row_chunks.append(np.full(n, row, dtype=np.int64))
        pid_chunks.append(sigs["place_id"].to_numpy())
        ts_chunks.append(
            pd.to_datetime(sigs["timestamp"], utc=True).astype("int64").to_numpy() / 1e9
        )

    if not row_chunks:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float32)
        return empty_i, empty_i, empty_f

    rows = np.concatenate(row_chunks)
    pids = np.concatenate(pid_chunks)
    ts = np.concatenate(ts_chunks).astype(np.float32)

    # Map place_id → item_idx with a single pandas .map call (C-level
    # loop, faster than a Python comprehension for big batches).
    mapped = pd.Series(pids).map(item_idx).to_numpy()
    keep = ~pd.isna(mapped)
    if not keep.any():
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float32)
        return empty_i, empty_i, empty_f

    return (
        rows[keep].astype(np.int64),
        mapped[keep].astype(np.int64),
        ts[keep],
    )


def _row_minmax(x: torch.Tensor) -> torch.Tensor:
    """Min-max normalise each row to [0, 1]. Empty/constant rows → zeros."""
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    rng = (mx - mn).clamp(min=1e-12)
    out = (x - mn) / rng
    # Guard against rows that are uniformly zero (no signal): keep them zero.
    constant = (mx - mn).abs() < 1e-12
    out = out.masked_fill(constant, 0.0)
    return out


def _l2_normalise(x: torch.Tensor) -> torch.Tensor:
    n = x.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return x / n


def _as_numpy(arr) -> np.ndarray:
    """Pull the underlying NumPy array out of an ``implicit`` matrix."""
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "to_numpy"):  # implicit GPU returns CuPy-like; convert
        return arr.to_numpy()
    if hasattr(arr, "get"):       # CuPy fallback
        return arr.get()
    return np.asarray(arr)


def _to_unix(reference_time) -> float:
    if reference_time is None:
        import time
        return float(time.time())
    if hasattr(reference_time, "timestamp"):
        return float(reference_time.timestamp())
    return float(reference_time)
