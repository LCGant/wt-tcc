"""Incremental preference updates — micro-batch delta processing."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime

import pandas as pd
import psycopg2.extensions

from src.db import read_cursor, transaction
from src.constants import RECENT_HALF_LIFE_DAYS
from src.features.temporal import apply_decay
from src.inference.normalize import normalize_weights

log = logging.getLogger(__name__)


def get_last_run_timestamp(conn: psycopg2.extensions.connection, tenant_id: str, run_type: str) -> datetime | None:
    """Get timestamp of the last run for a given type."""
    with read_cursor(conn) as cur:
        cur.execute(
            "SELECT last_signal_at FROM ai_run_metadata WHERE tenant_id = %s AND run_type = %s ORDER BY created_at DESC LIMIT 1",
            (tenant_id, run_type),
        )
        row = cur.fetchone()
    return row["last_signal_at"] if row else None


def record_run(conn: psycopg2.extensions.connection, tenant_id: str, run_type: str, last_signal_at: datetime, users_processed: int) -> None:
    """Record a run in ai_run_metadata."""
    with transaction(conn, tenant_id) as cur:
        cur.execute(
            "INSERT INTO ai_run_metadata (tenant_id, run_type, last_signal_at, users_processed) VALUES (%s, %s, %s, %s)",
            (tenant_id, run_type, last_signal_at, users_processed),
        )


def load_current_preferences(conn: psycopg2.extensions.connection, tenant_id: str, user_id: str, profile_type: str) -> dict[str, dict[str, float]]:
    """Load existing preference weights as {dimension: {value: weight}}."""
    with read_cursor(conn) as cur:
        cur.execute(
            "SELECT dimension, dimension_value, weight FROM user_preference_profiles WHERE tenant_id = %s AND user_id = %s AND profile_type = %s",
            (tenant_id, user_id, profile_type),
        )
        rows = cur.fetchall()
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        out[r["dimension"]][r["dimension_value"]] = r["weight"]
    return dict(out)


def incremental_update(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    user_id: str,
    new_signals: pd.DataFrame,
    places_df: pd.DataFrame,
    profile_type: str = "recent",
    half_life_days: float = RECENT_HALF_LIFE_DAYS,
    decay_existing: float = 0.9,
) -> int:
    """
    Incrementally update preference vector for one user.

    Instead of deleting and recomputing from scratch:
    1. Load existing preferences
    2. Decay existing weights by factor (simulates time passing)
    3. Compute delta from new signals
    4. Merge: existing_decayed + delta
    5. Re-normalize and write back

    Returns number of preferences written.
    """
    if new_signals.empty:
        return 0

    # Apply temporal decay to new signals
    new_signals = apply_decay(new_signals, half_life_days)

    # Build place lookups
    place_categories = {}
    if not places_df.empty:
        place_categories = dict(zip(places_df["public_id"], places_df["category"]))

    # Compute delta from new signals
    delta: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for _, sig in new_signals.iterrows():
        pid = sig.get("place_id", "")
        cat = place_categories.get(pid, "")
        if cat:
            w = sig.get("weight_decayed", sig.get("weight", 0.1))
            delta["category"][cat] += w

    if not delta:
        return 0

    # Load existing preferences
    existing = load_current_preferences(conn, tenant_id, user_id, profile_type)

    # Merge: decay existing + add delta
    merged: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for dim, values in existing.items():
        for val, w in values.items():
            merged[dim][val] = w * decay_existing

    for dim, values in delta.items():
        for val, w in values.items():
            merged[dim][val] += w

    # Normalize per dimension to [0, 1]
    prefs = []
    for dim, values in merged.items():
        for val, w in normalize_weights(dict(values)).items():
            prefs.append({
                "profile_type": profile_type,
                "dimension": dim,
                "dimension_value": val,
                "weight": round(w, 4),
            })

    # Write back
    from src.inference.writer import write_preferences
    return write_preferences(conn, tenant_id, user_id, profile_type, prefs)
