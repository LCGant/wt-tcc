"""CLI entry point for the recommendation engine."""

from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone

import click

from src.constants import (
    LEARNING_RATE_WARM_FEW,
    LEARNING_RATE_WARM_FULL,
    RECENT_WINDOW_DAYS,
)


@click.group()
def cli():
    """Role AI — Recommendation Engine CLI."""
    pass


@cli.command()
@click.option("--tenant", default="default", help="Tenant ID")
def train(tenant):
    """Train recommendation models from current data."""
    from src.config import DB_URL, log
    from src.db import connect
    from src.etl.signals import extract_all_signals
    from src.etl.places import extract_place_features
    from src.models.content_based import ContentBasedModel
    from src.models.collaborative import CollaborativeModel

    log.info("Training models for tenant=%s", tenant)
    conn = connect(DB_URL)
    try:
        places_df = extract_place_features(conn, tenant)
        signals = extract_all_signals(conn, tenant)

        if places_df.empty:
            log.warning("No places found — skipping training")
            return

        # Content-based
        cb = ContentBasedModel()
        cb.fit(places_df)
        log.info("Content-based model fitted: %d places, %d features", places_df.shape[0], cb.place_index.matrix.shape[1])

        # Collaborative
        collab = CollaborativeModel()
        if not signals.empty:
            collab.fit(signals[["user_id", "place_id", "weight"]])
            log.info("Collaborative model fitted: %d interactions", len(signals))
        else:
            log.warning("No signals — skipping collaborative model")

        # Save
        from src.models.storage import TrainedModels, save_models
        path = save_models(TrainedModels(content=cb, collaborative=collab))
        log.info("Training complete — models saved to %s", path)
    finally:
        conn.close()


@cli.command()
@click.option("--tenant", default="default", help="Tenant ID")
@click.option("--user", default=None, help="Recompute for a specific user (or all)")
def recompute(tenant, user):
    """Generate preference vectors for users."""
    from src.config import DB_URL, log
    from src.db import connect
    from src.etl.signals import extract_all_signals
    from src.etl.places import extract_place_features
    from src.etl.users import extract_user_ids, extract_onboarding
    from src.etl.telemetry import extract_telemetry
    from src.inference.base_profile import generate_base_preferences
    from src.inference.recent_profile import generate_recent_preferences
    from src.inference.writer import write_preferences

    from src.models.cold_start import classify_user, get_composition_weights, ColdStartLevel
    from src.inference.bandit_writer import run_bandit_update
    import pandas as pd

    log.info("Recomputing preferences for tenant=%s user=%s", tenant, user or "all")
    conn = connect(DB_URL)
    try:
        places_df = extract_place_features(conn, tenant)
        onboarding = extract_onboarding(conn, tenant)
        signals = extract_all_signals(conn, tenant)

        since_recent = datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)
        telemetry = extract_telemetry(conn, tenant, since_recent)

        if user:
            users = [user]
        else:
            users = extract_user_ids(conn, tenant)
            if not users and not signals.empty:
                users = sorted(signals["user_id"].unique())

        # Pre-group to avoid O(N) filter per user
        signals_by_user = signals.groupby("user_id") if not signals.empty else {}
        onboarding_by_user = onboarding.groupby("user_id") if not onboarding.empty else {}
        telemetry_users = set(telemetry["user_id"]) if not telemetry.empty else set()
        telemetry_by_user = telemetry.groupby("user_id") if not telemetry.empty else {}

        count = 0
        for uid in users:
            # Cold start classification
            user_signals = signals_by_user.get_group(uid) if uid in getattr(signals_by_user, 'groups', {}) else pd.DataFrame()
            user_onboarding = onboarding_by_user.get_group(uid) if uid in getattr(onboarding_by_user, 'groups', {}) else pd.DataFrame()
            has_onboarding = len(user_onboarding) > 0
            interaction_count = len(user_signals)

            level = classify_user(interaction_count, has_onboarding)
            log.debug("User %s: level=%s interactions=%d onboarding=%s", uid, level.value, interaction_count, has_onboarding)

            if level == ColdStartLevel.COLD_NO_DATA:
                # L1: no preferences to write — Go uses 100% exploratory
                count += 1
                continue

            # Base preferences (all-time)
            base = generate_base_preferences(uid, signals, onboarding, places_df)
            write_preferences(conn, tenant, uid, "base", base)

            # Recent preferences — only if telemetry exists for this user
            has_telemetry = uid in telemetry_users
            if has_telemetry and level in (ColdStartLevel.WARM_FEW, ColdStartLevel.WARM_FULL):
                recent = generate_recent_preferences(uid, telemetry, places_df)
                write_preferences(conn, tenant, uid, "recent", recent)

            # Composition weights: adapt based on available signals
            if has_telemetry and level in (ColdStartLevel.WARM_FEW, ColdStartLevel.WARM_FULL):
                lr = {ColdStartLevel.WARM_FEW: LEARNING_RATE_WARM_FEW, ColdStartLevel.WARM_FULL: LEARNING_RATE_WARM_FULL}.get(level, LEARNING_RATE_WARM_FULL)
                uid_telemetry = telemetry_by_user.get_group(uid) if uid in getattr(telemetry_by_user, 'groups', {}) else pd.DataFrame()
                user_telemetry = [
                    {"event_type": r.get("event_type", ""), "payload": r.get("payload", {})}
                    for _, r in uid_telemetry.iterrows()
                ]
                run_bandit_update(conn, tenant, uid, user_telemetry, learning_rate=lr)
            elif not has_telemetry and level in (ColdStartLevel.WARM_FEW, ColdStartLevel.WARM_FULL):
                # User has explicit signals but no telemetry — increase base weight
                from src.inference.bandit_writer import write_composition_weights
                write_composition_weights(conn, tenant, uid, 0.7, 0.0, 0.3)
            else:
                # Static composition weights based on cold start level
                from src.inference.bandit_writer import write_composition_weights
                cw = get_composition_weights(level)
                write_composition_weights(conn, tenant, uid, cw.base, cw.recent, cw.exploratory)

            count += 1

        # Cleanup orphan preferences (deleted users, stale data)
        from src.inference.writer import cleanup_orphan_preferences
        orphans = cleanup_orphan_preferences(conn, tenant)
        if orphans > 0:
            log.info("Cleaned %d orphan preference rows", orphans)

        # Record run
        from src.inference.incremental import record_run
        record_run(conn, tenant, "full_recompute", datetime.now(timezone.utc), count)
        log.info("Recomputed preferences for %d users", count)
    finally:
        conn.close()


@cli.command()
@click.option("--tenant", default="default", help="Tenant ID")
def update(tenant):
    """Incremental preference update — process only new signals since last run."""
    from src.config import DB_URL, log
    from src.db import connect
    from src.etl.signals import extract_all_signals
    from src.etl.places import extract_place_features
    from src.inference.incremental import get_last_run_timestamp, record_run, incremental_update
    from src.inference.bandit_writer import run_bandit_update

    log.info("Incremental update for tenant=%s", tenant)
    conn = connect(DB_URL)
    try:
        last_run = get_last_run_timestamp(conn, tenant, "incremental")
        if last_run is None:
            last_run = get_last_run_timestamp(conn, tenant, "full_recompute")
        if last_run is None:
            log.warning("No previous run found — run 'recompute' first")
            return

        log.info("Processing signals since %s", last_run.isoformat())
        places_df = extract_place_features(conn, tenant)
        new_signals = extract_all_signals(conn, tenant, since=last_run)

        if new_signals.empty:
            log.info("No new signals since last run")
            return

        # Pre-group by user
        signals_by_user = new_signals.groupby("user_id")
        count = 0

        # Bandit update — map signal_type back to original event_type
        _signal_to_event = {
            "telemetry_click": "feed_position_click",
            "telemetry_detail_view": "place_detail_view",
            "telemetry_impression": "impression",
            "telemetry_search": "search_query",
        }

        for uid, user_new in signals_by_user:
            incremental_update(conn, tenant, uid, user_new, places_df, profile_type="recent")

            user_telemetry = [
                {"event_type": _signal_to_event.get(r.get("signal_type", ""), r.get("signal_type", "")), "payload": {}}
                for _, r in user_new.iterrows()
            ]
            run_bandit_update(conn, tenant, uid, user_telemetry)
            count += 1

        now = datetime.now(timezone.utc)
        record_run(conn, tenant, "incremental", now, count)
        log.info("Incremental update: %d users updated from %d new signals", count, len(new_signals))
    finally:
        conn.close()


@cli.command()
@click.option("--tenant", default="default", help="Tenant ID")
@click.option("--verbose", is_flag=True, help="Show exact dataset counts")
@click.option("--seed-value", default=42, type=int, help="RNG seed for hybrid + baselines (reproducibility)")
def evaluate(tenant, verbose, seed_value):
    """Evaluate model with full offline metrics report."""
    from src.config import DB_URL, log
    from src.db import connect, read_cursor
    from src.etl.signals import extract_all_signals
    from src.etl.places import extract_place_features
    from src.etl.users import extract_onboarding
    from src.models.content_based import ContentBasedModel
    from src.models.cold_start import classify_user, ColdStartLevel
    from src.evaluation.offline_metrics import (
        evaluate_model,
        intra_list_diversity,
        coverage,
        serendipity,
        onboarding_correction_rate,
        confidence_evolution,
        relevance_density,
        wilcoxon_paired,
    )
    from src.evaluation.baselines import all_baselines, baselines_per_user
    from src.models.collaborative import CollaborativeModel
    from src.models.bandit import BanditState
    from src.inference.hybrid import HybridRecommender, HybridConfig
    from src.inference.bandit_writer import load_bandit_state
    import pandas as pd

    log.info("Evaluating for tenant=%s", tenant)
    conn = connect(DB_URL)
    try:
        places_df = extract_place_features(conn, tenant)
        if verbose:
            all_signals, signal_breakdown = extract_all_signals(conn, tenant, return_breakdown=True)
        else:
            all_signals = extract_all_signals(conn, tenant)
            signal_breakdown = None
        onboarding = extract_onboarding(conn, tenant)

        if all_signals.empty:
            log.warning("No signals — cannot evaluate")
            return

        # ── Temporal split ──
        cutoff = datetime.now(timezone.utc) - timedelta(days=RECENT_WINDOW_DAYS)
        all_signals["timestamp"] = pd.to_datetime(all_signals["timestamp"], utc=True)
        train_signals = all_signals[all_signals["timestamp"] < cutoff]
        test_signals = all_signals[all_signals["timestamp"] >= cutoff]

        if train_signals.empty or test_signals.empty:
            log.warning("Not enough data for temporal split")
            return

        # ── Train content-based ──
        cb = ContentBasedModel()
        cb.fit(places_df)

        # ── Train collaborative (ALS) ──
        col = CollaborativeModel()
        try:
            col.fit(train_signals)
        except Exception as exc:
            log.warning("Collaborative model fit failed: %s — using content-only base", exc)
            col = None

        # ── Hybrid recommender (the actual system from the article) ──
        hybrid = HybridRecommender(
            content=cb,
            collab=col,
            train_signals=train_signals,
            config=HybridConfig(),
            rng_seed=seed_value,
        )

        # Load bandit state from DB (populated by recompute step).
        # If empty (first eval before recompute), defaults to Beta(2,2) prior.
        bandit_states: dict[str, BanditState] = {}

        def _get_state(uid: str) -> BanditState:
            if uid not in bandit_states:
                try:
                    bandit_states[uid] = load_bandit_state(conn, tenant, uid)
                except Exception:
                    bandit_states[uid] = BanditState()
            return bandit_states[uid]

        # Onboarding per user
        if not onboarding.empty:
            onb_by_user = {uid: g for uid, g in onboarding.groupby("user_id")}
        else:
            onb_by_user = {}

        def recommend_fn(uid, n):
            user_sigs = train_signals[train_signals["user_id"] == uid]
            user_onb = onb_by_user.get(uid)
            state = _get_state(uid)
            return hybrid.recommend(uid, user_sigs, user_onb, state, n)

        click.echo("")
        click.echo("=" * 60)
        click.echo("  AI EVALUATION REPORT")
        click.echo("=" * 60)

        # ── Dataset summary ──
        users = sorted(all_signals["user_id"].unique())
        n_places = len(places_df)
        n_signals = len(all_signals)
        def _bucket(n):
            if n < 100: return "<100"
            if n < 1000: return f"~{(n // 100) * 100}"
            return f"~{(n // 1000)}k"

        click.echo("")
        click.echo("  Dataset:")
        if verbose:
            click.echo(f"    users:        {len(users)}")
            click.echo(f"    places:       {n_places}")
            click.echo(f"    signals:      {n_signals}")
            click.echo(f"    train split:  {len(train_signals)} (before {cutoff.date()})")
            click.echo(f"    test split:   {len(test_signals)} (last {RECENT_WINDOW_DAYS} days)")
            if signal_breakdown:
                click.echo("")
                click.echo("  Signal breakdown (raw counts before filters):")
                for source, count in signal_breakdown["raw_counts"].items():
                    click.echo(f"    {source:22s}: {count}")
                click.echo(f"    {'raw_total':22s}: {signal_breakdown['raw_total']}")
                click.echo(f"    {'after MAX_SIGNALS cap':22s}: {signal_breakdown['after_cap']}")
                click.echo(f"    {'dropped by per-user cap':22s}: {signal_breakdown['dropped_by_cap']}")
                click.echo(f"    note: {signal_breakdown['note_reviews']}")
        else:
            click.echo(f"    users:        {_bucket(len(users))}")
            click.echo(f"    places:       {_bucket(n_places)}")
            click.echo(f"    signals:      {_bucket(n_signals)}")
            click.echo(f"    train/test:   {RECENT_WINDOW_DAYS}-day temporal split")

        # ── Cold start distribution ──
        signals_by_user = all_signals.groupby("user_id")
        onboarding_by_user = onboarding.groupby("user_id") if not onboarding.empty else {}
        level_counts: dict[ColdStartLevel, int] = {lv: 0 for lv in ColdStartLevel}
        level_users: dict[ColdStartLevel, list[str]] = {lv: [] for lv in ColdStartLevel}

        for uid in users:
            user_sigs = signals_by_user.get_group(uid) if uid in getattr(signals_by_user, 'groups', {}) else pd.DataFrame()
            user_onb = onboarding_by_user.get_group(uid) if uid in getattr(onboarding_by_user, 'groups', {}) else pd.DataFrame()
            level = classify_user(len(user_sigs), len(user_onb) > 0)
            level_counts[level] += 1
            level_users[level].append(uid)

        click.echo("")
        click.echo("  Cold Start Distribution:")
        for lv in ColdStartLevel:
            click.echo(f"    {lv.value:20s}: {level_counts[lv]:4d} users")

        # ── Ranking metrics (global) — also collect per-user for Wilcoxon ──
        ranking, ranking_per_user = evaluate_model(recommend_fn, test_signals, return_per_user=True)
        click.echo("")
        click.echo("  Ranking Metrics (global):")
        for k, v in sorted(ranking.items()):
            click.echo(f"    {k:15s}: {v:.4f}")
            log.info("metric %s = %.4f", k, v)

        # ── Density (theoretical baseline) ──
        density_info = relevance_density(test_signals, n_places)

        # ── Full baseline suite ──
        click.echo("")
        click.echo("  Computing baselines (random / popularity / itemknn / iALS-tuned / BPR-tuned / EASE-R)...")
        baselines = all_baselines(train_signals, test_signals, n_places, rng_seed=seed_value, tune=True)

        click.echo("")
        click.echo("  Baselines (P@10 | NDCG@10 | Coverage):")
        click.echo(f"    avg relevant items/user:  {density_info['avg_relevant_per_user']:.2f}")
        click.echo(f"    relevance density (δ):    {density_info['density']*100:.3f}%  (= |R_u|/|P|)")
        click.echo("")
        for bname in ("random", "popularity", "itemknn", "ials", "bpr", "ease", "linucb"):
            b = baselines.get(bname, {})
            if "error" in b:
                click.echo(f"    {bname:<12s}  SKIPPED: {b['error']}")
                continue
            p10 = b.get("precision@10", 0) * 100
            n10 = b.get("ndcg@10", 0) * 100
            cv = b.get("coverage", 0) * 100
            click.echo(f"    {bname:<12s}  P@10={p10:6.2f}%  NDCG@10={n10:6.2f}%  Cov={cv:6.2f}%")
            for k in (5, 10, 20):
                log.info("metric %s_p%d = %.6f", bname, k, b.get(f"precision@{k}", 0))
                log.info("metric %s_ndcg%d = %.6f", bname, k, b.get(f"ndcg@{k}", 0))
                log.info("metric %s_recall%d = %.6f", bname, k, b.get(f"recall@{k}", 0))
            log.info("metric %s_coverage = %.6f", bname, b.get("coverage", 0))
        log.info("metric density = %.6f", density_info["density"])

        # ── Multipliers (system vs each baseline) ──
        sys_p10 = ranking.get("precision@10", 0.0)
        click.echo("")
        click.echo("  System multipliers (vs each baseline, P@10):")
        for bname in ("random", "popularity", "itemknn", "ials", "bpr", "ease", "linucb"):
            b = baselines.get(bname, {})
            bp10 = b.get("precision@10", 0)
            if bp10 > 0:
                mult = sys_p10 / bp10
                click.echo(f"    vs {bname:<12s} {mult:.2f}x")
                log.info("metric mult_%s = %.4f", bname, mult)
        # Also report theoretical random (= density)
        if density_info["density"] > 0:
            mult_th = sys_p10 / density_info["density"]
            click.echo(f"    vs theoretical-random (= δ):  {mult_th:.2f}x")
            log.info("metric mult_theoretical = %.4f", mult_th)

        # ── Wilcoxon signed-rank tests (paired per-user) ──
        click.echo("")
        click.echo("  Wilcoxon signed-rank (system vs each baseline, P@10):")
        click.echo("  H0: system and baseline have the same median P@10 per user")
        # Reuse tuned hyperparameters from all_baselines() to avoid re-running grid search.
        ials_tuned = baselines.get("ials", {}).get("_chosen_params")
        bpr_tuned = baselines.get("bpr", {}).get("_chosen_params")
        baseline_per_user = baselines_per_user(
            train_signals, test_signals, rng_seed=seed_value,
            ials_params=ials_tuned, bpr_params=bpr_tuned,
        )
        for bname in ("random", "popularity", "itemknn", "ials", "bpr", "ease", "linucb"):
            bpu = baseline_per_user.get(bname, {})
            if not bpu:
                click.echo(f"    {bname:<12s}  SKIPPED (no per-user data)")
                continue
            wx = wilcoxon_paired(ranking_per_user, bpu, "precision@10")
            sig = "***" if wx["pvalue"] < 0.001 else ("**" if wx["pvalue"] < 0.01 else ("*" if wx["pvalue"] < 0.05 else " "))
            click.echo(
                f"    vs {bname:<10s}  W={wx['statistic']:>8.0f}  "
                f"p={wx['pvalue']:.4f} {sig}  "
                f"median_diff={wx['median_diff']*100:+.2f}%  "
                f"win_rate={wx['win_rate']*100:.1f}%  n={wx['n']}"
            )
            log.info("metric wilcoxon_%s_pvalue = %.6f", bname, wx["pvalue"])
            log.info("metric wilcoxon_%s_winrate = %.4f", bname, wx["win_rate"])
            log.info("metric wilcoxon_%s_mediandiff = %.6f", bname, wx["median_diff"])

        # ── Ranking metrics per cold start level ──
        click.echo("")
        click.echo("  Ranking Metrics by Cold Start Level:")
        for lv in ColdStartLevel:
            lv_uids = set(level_users[lv])
            if not lv_uids:
                continue
            lv_test = test_signals[test_signals["user_id"].isin(lv_uids)]
            if lv_test.empty:
                click.echo(f"    {lv.value}: (no test data)")
                continue
            lv_metrics = evaluate_model(recommend_fn, lv_test)
            click.echo(f"    {lv.value}:")
            for mk, mv in sorted(lv_metrics.items()):
                click.echo(f"      {mk:15s}: {mv:.4f}")

        # ── Behavioral metrics ──
        click.echo("")
        click.echo("  Behavioral Metrics:")

        # Compute recommendations for all test users for behavioral metrics
        test_users = test_signals["user_id"].unique()
        all_rec_ids: list[str] = []
        all_rec_cats: list[list[str]] = []
        serendipity_scores: list[float] = []

        # Build place_id → category map
        cat_map = {}
        if "public_id" in places_df.columns and "category" in places_df.columns:
            cat_map = dict(zip(places_df["public_id"].astype(str), places_df["category"]))

        for uid in test_users:
            recs = recommend_fn(uid, 20)
            rec_ids = [pid for pid, _ in recs]
            all_rec_ids.extend(rec_ids)
            all_rec_cats.append([cat_map.get(str(pid), "unknown") for pid in rec_ids])

            user_history = set(train_signals[train_signals["user_id"] == uid]["place_id"])
            user_relevant = set(test_signals[(test_signals["user_id"] == uid) & (test_signals["weight"] > 0)]["place_id"])
            serendipity_scores.append(serendipity(rec_ids, user_history, user_relevant))

        ild = intra_list_diversity(all_rec_cats)
        cov = coverage(all_rec_ids, n_places)
        avg_serendipity = float(sum(serendipity_scores) / len(serendipity_scores)) if serendipity_scores else 0.0

        click.echo(f"    ILD (diversity):  {ild:.4f}")
        click.echo(f"    Coverage:         {cov:.4f}")
        click.echo(f"    Serendipity:      {avg_serendipity:.4f}")

        # ── Learning quality metrics ──
        click.echo("")
        click.echo("  Learning Quality:")

        # OCR: compare onboarding declared categories vs learned top categories from signals
        ocr_scores: list[float] = []
        for uid in test_users:
            uid_onb = onboarding_by_user.get_group(uid) if uid in getattr(onboarding_by_user, 'groups', {}) else pd.DataFrame()
            if uid_onb.empty:
                continue
            # Onboarding categories
            onb_cats: set[str] = set()
            for _, row in uid_onb.iterrows():
                if row.get("question_key") == "preferred_categories":
                    vals = row.get("answer_values", [])
                    if isinstance(vals, list):
                        onb_cats.update(vals)
            if not onb_cats:
                continue
            # Learned top categories from signal interactions
            uid_sigs = signals_by_user.get_group(uid) if uid in getattr(signals_by_user, 'groups', {}) else pd.DataFrame()
            if uid_sigs.empty:
                continue
            # Map place_id → category from signals
            learned_cats: set[str] = set()
            for pid in uid_sigs[uid_sigs["weight"] > 0]["place_id"].unique():
                cat = cat_map.get(str(pid), "")
                if cat:
                    learned_cats.add(cat)
            ocr_scores.append(onboarding_correction_rate(onb_cats, learned_cats))

        avg_ocr = float(sum(ocr_scores) / len(ocr_scores)) if ocr_scores else 0.0
        click.echo(f"    OCR (correction): {avg_ocr:.4f}")

        # Confidence Evolution: read bandit alpha history from DB
        with read_cursor(conn, tenant) as cur:
            cur.execute(
                "SELECT user_id, base_alpha, recent_alpha, exploratory_alpha FROM user_bandit_state WHERE tenant_id = %s",
                (tenant,),
            )
            bandit_rows = cur.fetchall()

        if bandit_rows:
            alphas = [float(r["base_alpha"]) + float(r["recent_alpha"]) + float(r["exploratory_alpha"]) for r in bandit_rows]
            conf_evo = confidence_evolution(alphas)
            avg_alpha = sum(alphas) / len(alphas)
            click.echo(f"    Confidence evo:   {conf_evo:.4f} (slope)")
            click.echo(f"    Avg alpha sum:    {avg_alpha:.2f}")
        else:
            click.echo("    Confidence evo:   N/A (no bandit state)")

        # ── Summary ──
        click.echo("")
        click.echo("-" * 60)
        click.echo("  Summary:")
        p10 = ranking.get("precision@10", 0.0)
        ndcg10 = ranking.get("ndcg@10", 0.0)
        click.echo(f"    P@10={p10:.4f}  NDCG@10={ndcg10:.4f}  ILD={ild:.4f}  Cov={cov:.4f}")
        click.echo("=" * 60)
        click.echo("")

    finally:
        conn.close()


@cli.command(name="ablation")
@click.option("--root", type=click.Path(exists=True, file_okay=False), required=True,
              help="Directory containing MovieLens-100K files (u.data, u.item)")
@click.option("--test-days", default=30, type=int)
@click.option("--seed-value", default=42, type=int)
@click.option("--cold-fraction", default=0.0, type=float,
              help="Fraction of users to truncate to simulate cold start (0.0 = no truncation)")
@click.option("--cold-keep-k", default=5, type=int,
              help="Number of earliest interactions to keep for cold-truncated users")
def ablation(root, test_days, seed_value, cold_fraction, cold_keep_k):
    """Run ablation study on MovieLens: full system vs each component disabled.

    Variants tested:
        - full         : all components on
        - no_thompson  : fixed weights from level priors (no sampling)
        - no_recent    : no recent-decay component (only base + exploratory)
        - no_classifier: no cold-start classifier (everyone WARM_FULL)
        - no_exploratory: no exploratory popularity component
    """
    from src.config import log
    from src.etl.movielens import load_movielens_100k
    from src.models.content_based import ContentBasedModel
    from src.models.collaborative import CollaborativeModel
    from src.models.bandit import BanditState
    from src.inference.hybrid import HybridRecommender, HybridConfig
    from src.evaluation.offline_metrics import evaluate_model
    from pathlib import Path
    from datetime import timedelta

    data = load_movielens_100k(Path(root))
    places_df = data["places_df"]
    all_signals = data["signals_df"]

    max_ts = all_signals["timestamp"].max()
    cutoff = max_ts - timedelta(days=test_days)
    train_signals = all_signals[all_signals["timestamp"] < cutoff]
    test_signals = all_signals[all_signals["timestamp"] >= cutoff]

    # Cold-start simulation: truncate `cold_fraction` of users to first `cold_keep_k`
    # interactions. Same logic as evaluate-movielens — required for the classifier
    # and Thompson sampling to actually differentiate user populations.
    if cold_fraction > 0:
        import random as _r
        _rng = _r.Random(seed_value)
        all_train_users = sorted(train_signals["user_id"].unique())
        n_cold = int(len(all_train_users) * cold_fraction)
        cold_users = set(_rng.sample(all_train_users, n_cold)) if n_cold else set()
        if cold_users:
            train_signals = train_signals.sort_values("timestamp")
            keep_idx = []
            cold_count: dict = {}
            for idx, row in train_signals.iterrows():
                uid = row["user_id"]
                if uid in cold_users:
                    cold_count[uid] = cold_count.get(uid, 0) + 1
                    if cold_count[uid] > cold_keep_k:
                        continue
                keep_idx.append(idx)
            train_signals = train_signals.loc[keep_idx]
            log.info("Cold simulation: truncated %d/%d users to first %d interactions",
                     len(cold_users), len(all_train_users), cold_keep_k)

    cb = ContentBasedModel()
    cb.fit(places_df)
    col = CollaborativeModel()
    try:
        col.fit(train_signals)
    except Exception:
        col = None

    variants = {
        "full": HybridConfig(),
        "no_thompson": HybridConfig(use_thompson=False),
        "no_recent": HybridConfig(use_recent=False),
        "no_classifier": HybridConfig(use_classifier=False),
        "no_exploratory": HybridConfig(use_exploratory=False),
    }

    click.echo("")
    click.echo("=" * 60)
    click.echo("  ABLATION STUDY (MovieLens-100K, seed={})".format(seed_value))
    if cold_fraction > 0:
        click.echo(f"  Cold simulation: {cold_fraction*100:.0f}% users truncated to first {cold_keep_k} interactions")
    click.echo("=" * 60)
    click.echo("")
    click.echo(f"  {'variant':<18s}  {'P@5':>8s}  {'P@10':>8s}  {'NDCG@10':>8s}  {'NDCG@20':>8s}")
    click.echo(f"  {'-'*18}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}")

    for name, cfg in variants.items():
        hybrid = HybridRecommender(
            content=cb, collab=col, train_signals=train_signals,
            config=cfg, rng_seed=seed_value,
        )
        from src.inference.bandit_warmup import warmup_all
        bandit_states = warmup_all(train_signals)

        def recommend_fn(uid, n):
            user_sigs = train_signals[train_signals["user_id"] == uid]
            state = bandit_states.get(uid) or bandit_states.setdefault(uid, BanditState())
            return hybrid.recommend(uid, user_sigs, None, state, n)

        ranking = evaluate_model(recommend_fn, test_signals)
        click.echo(
            f"  {name:<18s}  "
            f"{ranking.get('precision@5', 0)*100:>7.2f}%  "
            f"{ranking.get('precision@10', 0)*100:>7.2f}%  "
            f"{ranking.get('ndcg@10', 0)*100:>7.2f}%  "
            f"{ranking.get('ndcg@20', 0)*100:>7.2f}%"
        )
        for k_metric in ranking:
            log.info("metric ablation_%s_%s = %.6f", name, k_metric, ranking[k_metric])

    click.echo("")
    click.echo("=" * 60)


@cli.command(name="sensitivity")
@click.option("--root", type=click.Path(exists=True, file_okay=False), required=True,
              help="Directory containing MovieLens-100K files")
@click.option("--test-days", default=30, type=int)
@click.option("--seed-value", default=42, type=int)
@click.option("--cold-fraction", default=0.5, type=float)
@click.option("--cold-keep-k", default=3, type=int)
@click.option("--param", type=click.Choice([
    "exploratory_noise", "half_life_recent", "base_content_weight",
    "cold_fraction", "cold_keep_k",
]), required=True, help="Hyperparameter to vary")
def sensitivity(root, test_days, seed_value, cold_fraction, cold_keep_k, param):
    """Sensitivity analysis: vary one system hyperparameter, fix others.

    Reports P@10 and NDCG@10 across a grid of values to assess robustness.
    """
    from src.config import log
    from src.etl.movielens import load_movielens_100k
    from src.models.content_based import ContentBasedModel
    from src.models.collaborative import CollaborativeModel
    from src.models.bandit import BanditState
    from src.inference.hybrid import HybridRecommender, HybridConfig
    from src.inference.bandit_warmup import warmup_all
    from src.evaluation.offline_metrics import evaluate_model
    from src import constants
    from pathlib import Path
    from datetime import timedelta
    import time

    data = load_movielens_100k(Path(root))
    places_df = data["places_df"]
    all_signals = data["signals_df"]

    max_ts = all_signals["timestamp"].max()
    cutoff = max_ts - timedelta(days=test_days)
    full_train_signals = all_signals[all_signals["timestamp"] < cutoff]
    test_signals = all_signals[all_signals["timestamp"] >= cutoff]

    def _apply_cold_sim(df, frac, keep_k, seed):
        """Truncate `frac` of users to first `keep_k` interactions. Returns truncated df."""
        if frac <= 0:
            return df
        import random as _r
        _rng = _r.Random(seed)
        users = sorted(df["user_id"].unique())
        n_cold = int(len(users) * frac)
        cold_users = set(_rng.sample(users, n_cold)) if n_cold else set()
        if not cold_users:
            return df
        sorted_df = df.sort_values("timestamp")
        keep_idx = []
        counts: dict = {}
        for idx, row in sorted_df.iterrows():
            uid = row["user_id"]
            if uid in cold_users:
                counts[uid] = counts.get(uid, 0) + 1
                if counts[uid] > keep_k:
                    continue
            keep_idx.append(idx)
        return sorted_df.loc[keep_idx]

    # For non-cold-sim params, apply baseline cold sim once.
    # For cold-sim params, applied per iteration inside loop.
    is_cold_sim_param = param in ("cold_fraction", "cold_keep_k")
    if not is_cold_sim_param:
        train_signals = _apply_cold_sim(full_train_signals, cold_fraction, cold_keep_k, seed_value)
    else:
        train_signals = full_train_signals  # placeholder, recomputed in loop

    cb = ContentBasedModel()
    cb.fit(places_df)
    col_default = None
    if not is_cold_sim_param:
        col_default = CollaborativeModel()
        try:
            col_default.fit(train_signals)
        except Exception:
            col_default = None

    # Define grid per parameter
    if param == "exploratory_noise":
        values = [0.0, 0.1, 0.2, 0.3, 0.5]
        unit = ""
    elif param == "half_life_recent":
        values = [1.0, 3.0, 7.0, 14.0, 30.0]
        unit = " days"
    elif param == "base_content_weight":
        values = [0.0, 0.3, 0.6, 0.8, 1.0]
        unit = ""
    elif param == "cold_fraction":
        values = [0.25, 0.5, 0.75]
        unit = ""
    elif param == "cold_keep_k":
        values = [1, 3, 5, 10]
        unit = " interactions"
    else:
        values = []
        unit = ""

    click.echo("")
    click.echo("=" * 60)
    click.echo(f"  SENSITIVITY ANALYSIS — {param} (seed={seed_value})")
    if cold_fraction > 0:
        click.echo(f"  Cold simulation: {cold_fraction*100:.0f}% users truncated to {cold_keep_k} interactions")
    click.echo("=" * 60)
    click.echo("")
    click.echo(f"  {'value':<12s}  {'P@5':>8s}  {'P@10':>8s}  {'NDCG@10':>8s}  {'Coverage':>8s}  {'time/user':>10s}")
    click.echo(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*8}  {'-'*10}")

    for v in values:
        # Re-derive train_signals + collab for cold-sim params
        if param == "cold_fraction":
            iter_train = _apply_cold_sim(full_train_signals, v, cold_keep_k, seed_value)
        elif param == "cold_keep_k":
            iter_train = _apply_cold_sim(full_train_signals, cold_fraction, int(v), seed_value)
        else:
            iter_train = train_signals

        if is_cold_sim_param:
            col = CollaborativeModel()
            try:
                col.fit(iter_train)
            except Exception:
                col = None
        else:
            col = col_default

        # Configure hybrid
        if param == "exploratory_noise":
            cfg = HybridConfig(exploratory_noise=v)
        elif param == "base_content_weight":
            cfg = HybridConfig(base_content_weight=v)
        else:
            cfg = HybridConfig()

        # half_life monkey-patch only for that param
        old_hl = None
        if param == "half_life_recent":
            old_hl = constants.RECENT_HALF_LIFE_DAYS
            constants.RECENT_HALF_LIFE_DAYS = v
            from src.inference import hybrid as _hyb_mod
            _hyb_mod.RECENT_HALF_LIFE_DAYS = v

        hybrid = HybridRecommender(
            content=cb, collab=col, train_signals=iter_train,
            config=cfg, rng_seed=seed_value,
        )

        bandit_states = warmup_all(iter_train)

        def recommend_fn(uid, n, _train=iter_train, _bs=bandit_states, _hyb=hybrid):
            user_sigs = _train[_train["user_id"] == uid]
            state = _bs.get(uid) or _bs.setdefault(uid, BanditState())
            return _hyb.recommend(uid, user_sigs, None, state, n)

        # Time inference
        test_users = sorted(test_signals["user_id"].unique())[:50]
        t0 = time.perf_counter()
        for uid in test_users:
            recommend_fn(uid, 20)
        elapsed = time.perf_counter() - t0
        time_per_user_ms = (elapsed / max(1, len(test_users))) * 1000.0

        ranking = evaluate_model(recommend_fn, test_signals)

        # Coverage estimation on first 100 users
        all_recs = []
        for uid in sorted(test_signals["user_id"].unique())[:100]:
            recs = recommend_fn(uid, 20)
            all_recs.extend([pid for pid, _ in recs])
        n_unique = len(set(all_recs))
        cov = n_unique / max(1, len(places_df))

        click.echo(
            f"  {f'{v:.2f}{unit}':<12s}  "
            f"{ranking.get('precision@5', 0)*100:>7.2f}%  "
            f"{ranking.get('precision@10', 0)*100:>7.2f}%  "
            f"{ranking.get('ndcg@10', 0)*100:>7.2f}%  "
            f"{cov*100:>7.2f}%  "
            f"{time_per_user_ms:>8.2f}ms"
        )

        log.info("metric sensitivity_%s_%.4f_p5 = %.6f", param, v, ranking.get("precision@5", 0))
        log.info("metric sensitivity_%s_%.4f_p10 = %.6f", param, v, ranking.get("precision@10", 0))
        log.info("metric sensitivity_%s_%.4f_ndcg10 = %.6f", param, v, ranking.get("ndcg@10", 0))
        log.info("metric sensitivity_%s_%.4f_coverage = %.6f", param, v, cov)
        log.info("metric sensitivity_%s_%.4f_time_ms = %.4f", param, v, time_per_user_ms)

        # Restore half_life if changed
        if old_hl is not None:
            constants.RECENT_HALF_LIFE_DAYS = old_hl
            from src.inference import hybrid as _hyb_mod
            _hyb_mod.RECENT_HALF_LIFE_DAYS = old_hl

    click.echo("")
    click.echo("=" * 60)


@cli.command(name="evaluate-movielens")
@click.option("--root", type=click.Path(exists=True, file_okay=False), required=True,
              help="Directory containing MovieLens files (u.data/u.item for 100K, ratings.dat/movies.dat for 1M)")
@click.option("--variant", type=click.Choice(["100k", "1m", "lastfm"]), default="100k",
              help="Dataset variant: MovieLens-100K, MovieLens-1M, or Last.fm-2k")
@click.option("--test-days", default=30, type=int,
              help="Days from end of timestamp range to use as test set")
@click.option("--seed-value", default=42, type=int,
              help="RNG seed for hybrid + baselines (reproducibility)")
@click.option("--cold-fraction", default=0.0, type=float,
              help="Fraction of users to truncate to simulate cold start (0.0 = no truncation)")
@click.option("--cold-keep-k", default=5, type=int,
              help="Number of earliest interactions to keep for cold-truncated users")
@click.option("--backend", type=click.Choice(["cpu", "torch"]), default="cpu",
              help="Evaluation backend. 'cpu' = legacy per-user loop; 'torch' = vectorised "
                   "GPU/CPU pipeline (set AI_USE_GPU=1 for CUDA).")
def evaluate_movielens(root, variant, test_days, seed_value, cold_fraction, cold_keep_k, backend):
    """Evaluate hybrid recommender against MovieLens-100K or 1M dataset.

    Uses the most recent `--test-days` of timestamps as test set.
    No DB access required — loads everything from filesystem.
    """
    if backend == "torch":
        # Vectorised path — tensors on the configured device, no per-user
        # Python loop. Produces the same metric log lines as the CPU path.
        from src_torch.runner import evaluate_movielens_torch
        evaluate_movielens_torch(
            root=root,
            variant=variant,
            test_days=test_days,
            seed_value=seed_value,
            cold_fraction=cold_fraction,
            cold_keep_k=cold_keep_k,
        )
        return

    from src.config import log
    from src.etl.movielens import load_movielens_100k, load_movielens_1m, load_lastfm_2k
    from src.models.content_based import ContentBasedModel
    from src.models.collaborative import CollaborativeModel
    from src.models.bandit import BanditState
    from src.inference.hybrid import HybridRecommender, HybridConfig
    from src.evaluation.offline_metrics import (
        evaluate_model, intra_list_diversity, coverage, serendipity,
        relevance_density, wilcoxon_paired,
    )
    from src.evaluation.baselines import all_baselines, baselines_per_user
    from pathlib import Path
    from datetime import timedelta

    if variant == "1m":
        log.info("Loading MovieLens-1M from %s", root)
        data = load_movielens_1m(Path(root))
    elif variant == "lastfm":
        log.info("Loading Last.fm-2k from %s (top-1500 artists subsample)", root)
        data = load_lastfm_2k(Path(root))
    else:
        log.info("Loading MovieLens-100K from %s", root)
        data = load_movielens_100k(Path(root))
    places_df = data["places_df"]
    all_signals = data["signals_df"]
    onboarding = data["onboarding_df"]

    log.info("MovieLens loaded: %d places, %d signals, %d users",
             len(places_df), len(all_signals), all_signals["user_id"].nunique())

    # Temporal split based on actual timestamp range
    max_ts = all_signals["timestamp"].max()
    cutoff = max_ts - timedelta(days=test_days)
    train_signals = all_signals[all_signals["timestamp"] < cutoff]
    test_signals = all_signals[all_signals["timestamp"] >= cutoff]

    if train_signals.empty or test_signals.empty:
        log.warning("Not enough data for split — adjust --test-days")
        return

    # Cold-start simulation: truncate `cold_fraction` of users to their first
    # `cold_keep_k` interactions in train. Test set untouched. This forces the
    # cold-start classifier to actually see warm_few / cold_onboarding users.
    if cold_fraction > 0:
        import random as _r
        _rng = _r.Random(seed_value)
        all_train_users = sorted(train_signals["user_id"].unique())
        n_cold = int(len(all_train_users) * cold_fraction)
        cold_users = set(_rng.sample(all_train_users, n_cold)) if n_cold else set()
        if cold_users:
            train_signals = train_signals.sort_values("timestamp")
            keep_idx = []
            cold_count: dict = {}
            for idx, row in train_signals.iterrows():
                uid = row["user_id"]
                if uid in cold_users:
                    cold_count[uid] = cold_count.get(uid, 0) + 1
                    if cold_count[uid] > cold_keep_k:
                        continue
                keep_idx.append(idx)
            train_signals = train_signals.loc[keep_idx]
            log.info("Cold simulation: truncated %d/%d users to first %d interactions",
                     len(cold_users), len(all_train_users), cold_keep_k)

    n_places = len(places_df)
    users = sorted(all_signals["user_id"].unique())

    click.echo("")
    click.echo("=" * 60)
    click.echo("  MOVIELENS-100K EVALUATION REPORT")
    click.echo("=" * 60)
    click.echo("")
    click.echo("  Dataset:")
    click.echo(f"    users:        {len(users)}")
    click.echo(f"    places:       {n_places}")
    click.echo(f"    signals:      {len(all_signals)}")
    click.echo(f"    train split:  {len(train_signals)} (before {cutoff.date()})")
    click.echo(f"    test split:   {len(test_signals)} (last {test_days} days)")

    # Train system components
    cb = ContentBasedModel()
    cb.fit(places_df)

    col = CollaborativeModel()
    try:
        col.fit(train_signals)
    except Exception as exc:
        log.warning("Collaborative fit failed: %s", exc)
        col = None

    hybrid = HybridRecommender(
        content=cb, collab=col, train_signals=train_signals, config=HybridConfig(),
        rng_seed=seed_value,
    )

    # Warmup bandit states from train signals (offline pseudo-rewards).
    # Without this, Thompson Sampling always draws from Beta(2,2) prior and
    # contributes nothing — see ablation study.
    from src.inference.bandit_warmup import warmup_all
    bandit_states = warmup_all(train_signals)
    log.info("Warmed up bandit states for %d users", len(bandit_states))

    def recommend_fn(uid, n):
        user_sigs = train_signals[train_signals["user_id"] == uid]
        state = bandit_states.get(uid) or bandit_states.setdefault(uid, BanditState())
        return hybrid.recommend(uid, user_sigs, None, state, n)

    # Ranking metrics + per-user
    ranking, ranking_per_user = evaluate_model(recommend_fn, test_signals, return_per_user=True)
    click.echo("")
    click.echo("  Ranking Metrics (global):")
    for k, v in sorted(ranking.items()):
        click.echo(f"    {k:15s}: {v:.4f}")
        log.info("metric %s = %.4f", k, v)

    # Density
    density_info = relevance_density(test_signals, n_places)
    click.echo("")
    click.echo(f"  Relevance density: {density_info['density']*100:.3f}%  (avg {density_info['avg_relevant_per_user']:.2f} relevant/user)")

    # Baselines
    click.echo("")
    click.echo("  Computing baselines (random / popularity / itemknn / iALS-tuned / BPR-tuned / EASE-R)...")
    baselines = all_baselines(train_signals, test_signals, n_places, rng_seed=seed_value, tune=True)
    click.echo("")
    click.echo("  Baselines (P@10 | NDCG@10 | Coverage):")
    for bname in ("random", "popularity", "itemknn", "ials", "bpr", "ease", "linucb"):
        b = baselines.get(bname, {})
        if "error" in b:
            click.echo(f"    {bname:<12s}  SKIPPED: {b['error']}")
            continue
        click.echo(f"    {bname:<12s}  P@10={b.get('precision@10', 0)*100:6.2f}%  "
                   f"NDCG@10={b.get('ndcg@10', 0)*100:6.2f}%  "
                   f"Cov={b.get('coverage', 0)*100:6.2f}%")
        for k in (5, 10, 20):
            log.info("metric %s_p%d = %.6f", bname, k, b.get(f"precision@{k}", 0))
            log.info("metric %s_ndcg%d = %.6f", bname, k, b.get(f"ndcg@{k}", 0))
        log.info("metric %s_coverage = %.6f", bname, b.get("coverage", 0))

    # Log chosen hyperparameters for tuned models (transparency in paper)
    for bname in ("ials", "bpr"):
        cp = baselines.get(bname, {}).get("_chosen_params")
        if cp:
            log.info("tuned_%s_params = %s", bname, cp)

    # Multipliers
    sys_p10 = ranking.get("precision@10", 0.0)
    click.echo("")
    click.echo("  System multipliers (vs each baseline, P@10):")
    for bname in ("random", "popularity", "itemknn", "ials", "bpr", "ease", "linucb"):
        b = baselines.get(bname, {})
        bp10 = b.get("precision@10", 0)
        if bp10 > 0:
            mult = sys_p10 / bp10
            click.echo(f"    vs {bname:<12s} {mult:.2f}x")
            log.info("metric mult_%s = %.4f", bname, mult)

    # Wilcoxon (reuse tuned hyperparameters to skip duplicate grid search)
    click.echo("")
    click.echo("  Wilcoxon signed-rank (system vs each baseline, P@10):")
    ials_tuned = baselines.get("ials", {}).get("_chosen_params")
    bpr_tuned = baselines.get("bpr", {}).get("_chosen_params")
    baseline_per_user = baselines_per_user(
        train_signals, test_signals, rng_seed=seed_value,
        ials_params=ials_tuned, bpr_params=bpr_tuned,
    )
    for bname in ("random", "popularity", "itemknn", "ials", "bpr", "ease", "linucb"):
        bpu = baseline_per_user.get(bname, {})
        if not bpu:
            continue
        wx = wilcoxon_paired(ranking_per_user, bpu, "precision@10")
        sig = "***" if wx["pvalue"] < 0.001 else ("**" if wx["pvalue"] < 0.01 else ("*" if wx["pvalue"] < 0.05 else " "))
        click.echo(
            f"    vs {bname:<10s}  W={wx['statistic']:>8.0f}  p={wx['pvalue']:.4f} {sig}  "
            f"win_rate={wx['win_rate']*100:.1f}%  n={wx['n']}"
        )
        log.info("metric wilcoxon_%s_pvalue = %.6f", bname, wx["pvalue"])
        log.info("metric wilcoxon_%s_winrate = %.4f", bname, wx["win_rate"])

    # Behavioral
    test_users = test_signals["user_id"].unique()
    cat_map = dict(zip(places_df["public_id"].astype(str), places_df["category"]))
    all_rec_ids: list = []
    all_rec_cats: list = []
    serendipity_scores: list = []
    for uid in test_users:
        recs = recommend_fn(uid, 20)
        rec_ids = [pid for pid, _ in recs]
        all_rec_ids.extend(rec_ids)
        all_rec_cats.append([cat_map.get(str(pid), "unknown") for pid in rec_ids])
        user_history = set(train_signals[train_signals["user_id"] == uid]["place_id"])
        user_relevant = set(test_signals[(test_signals["user_id"] == uid) & (test_signals["weight"] > 0)]["place_id"])
        serendipity_scores.append(serendipity(rec_ids, user_history, user_relevant))

    ild = intra_list_diversity(all_rec_cats)
    cov = coverage(all_rec_ids, n_places)
    avg_serendip = float(sum(serendipity_scores) / len(serendipity_scores)) if serendipity_scores else 0.0
    click.echo("")
    click.echo("  Behavioral Metrics:")
    click.echo(f"    ILD (diversity):  {ild:.4f}")
    click.echo(f"    Coverage:         {cov:.4f}")
    click.echo(f"    Serendipity:      {avg_serendip:.4f}")
    log.info("metric ild = %.6f", ild)
    log.info("metric coverage = %.6f", cov)
    log.info("metric serendipity = %.6f", avg_serendip)

    click.echo("")
    click.echo("=" * 60)


@cli.command()
@click.option("--tenant", default="default")
@click.option("--users", default=50, type=int)
@click.option("--places", "n_places", default=30, type=int)
@click.option("--interactions-per-user", default=20, type=int)
@click.option("--days", default=30, type=int)
@click.option("--center-lat", default=-23.55, type=float)
@click.option("--center-lng", default=-46.63, type=float)
@click.option("--radius-km", default=15.0, type=float)
@click.option("--seed-value", default=42, type=int, help="Random seed for reproducibility")
@click.option("--clean", is_flag=True, help="Remove all seeded data first")
def seed(tenant, users, n_places, interactions_per_user, days, center_lat, center_lng, radius_km, seed_value, clean):
    """Generate synthetic training data."""
    from src.config import ADMIN_DB_URL, log
    from src.db import connect
    from src.seed.places import generate_places
    from src.seed.interactions import generate_interactions
    from src.seed.personas import pick_persona
    from src.seed.writer import write_seed, clean_seed

    # Seed requires owner role (writes to places/telemetry tables)
    if not ADMIN_DB_URL:
        log.error("SOCIAL_DB_URL is required for seeding (admin/owner role needed)")
        sys.exit(1)
    db_url = ADMIN_DB_URL
    log.info("Seeding: %d users, %d places, %d interactions/user, %d days", users, n_places, interactions_per_user, days)
    rng = random.Random(seed_value)

    conn = connect(db_url)
    try:
        if clean:
            clean_seed(conn, tenant, force=False)
            click.echo("Cleaned seed data.")

        # Generate places
        places = generate_places(n_places, center_lat, center_lng, radius_km, rng)
        click.echo(f"Generated {len(places)} places")

        # Generate users with personas
        user_list = []
        for i in range(users):
            persona = pick_persona(rng)
            user_list.append({
                "user_id": str(i + 1),
                "persona": persona,
            })
        click.echo(f"Generated {len(user_list)} users")

        # Generate interactions
        interactions = generate_interactions(user_list, places, interactions_per_user, days, rng)
        click.echo(f"Generated interactions: {sum(len(v) for v in interactions.values())} total")

        # Write to database
        summary = write_seed(conn, tenant, places, interactions)
        for k, v in summary.items():
            click.echo(f"  {k}: {v}")

        click.echo("Seed complete.")
    finally:
        conn.close()


@cli.command()
@click.option("--tenant", default="default", help="Tenant ID")
@click.option("--users", default=200, type=int)
@click.option("--places", "n_places", default=500, type=int)
@click.option("--interactions-per-user", default=80, type=int)
@click.option("--days", default=90, type=int)
@click.option("--seed-value", default=42, type=int)
@click.pass_context
def pipeline(ctx, tenant, users, n_places, interactions_per_user, days, seed_value):
    """Run full pipeline: seed → train → recompute → evaluate."""
    from src.config import ADMIN_DB_URL

    click.echo("=" * 60)
    click.echo("  FULL AI PIPELINE")
    click.echo("=" * 60)
    click.echo("")

    # Step 1: Seed
    if ADMIN_DB_URL:
        click.echo("[1/4] Seeding synthetic data...")
        try:
            ctx.invoke(seed, tenant=tenant, users=users, n_places=n_places,
                       interactions_per_user=interactions_per_user, days=days,
                       center_lat=-23.55, center_lng=-46.63, radius_km=15.0,
                       seed_value=seed_value, clean=True)
        except (SystemExit, Exception) as e:
            click.echo(f"[FATAL] Seed failed: {e}", err=True)
            raise SystemExit(1)
        click.echo("")
    else:
        click.echo("[1/4] Skipping seed (no SOCIAL_DB_URL)")
        click.echo("")

    # Step 2: Train
    click.echo("[2/4] Training models...")
    try:
        ctx.invoke(train, tenant=tenant)
    except (SystemExit, Exception) as e:
        click.echo(f"[FATAL] Train failed: {e}", err=True)
        raise SystemExit(1)
    click.echo("")

    # Step 3: Recompute
    click.echo("[3/4] Recomputing preferences...")
    try:
        ctx.invoke(recompute, tenant=tenant, user=None)
    except (SystemExit, Exception) as e:
        click.echo(f"[FATAL] Recompute failed: {e}", err=True)
        raise SystemExit(1)
    click.echo("")

    # Step 4: Evaluate
    click.echo("[4/4] Running evaluation...")
    try:
        ctx.invoke(evaluate, tenant=tenant, verbose=True, seed_value=seed_value)
    except (SystemExit, Exception) as e:
        click.echo(f"[FATAL] Evaluate failed: {e}", err=True)
        raise SystemExit(1)


if __name__ == "__main__":
    cli()
