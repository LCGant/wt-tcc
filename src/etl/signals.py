"""Extract and unify all interaction signals into a single DataFrame."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import psycopg2.extensions

from src.constants import MAX_SIGNALS_PER_USER
from src.db import read_cursor


# Signal weights matching the Go service preferences.go
SIGNAL_WEIGHTS = {
    "onboarding": 1.0,
    "favorite": 1.5,
    "want_to_go": 1.0,
    "been_there": 2.0,
    "review_positive": 2.0,
    "review_negative": -1.0,
    "place_like": 1.0,
    "checkin": 2.5,
    "checkin_return": 3.75,
    "post_about_place": 1.0,
    "post_like": 0.3,
    "post_save": 0.5,
    "post_view": 0.1,
    "telemetry_click": 0.5,
    "telemetry_detail_view": 0.3,
    "telemetry_impression": -0.1,
    "telemetry_search": 0.8,
}


def extract_all_signals(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    since: datetime | None = None,
    return_breakdown: bool = False,
) -> pd.DataFrame:
    """
    Extract all user×place interaction signals into a unified DataFrame.

    Columns: user_id, place_id (public_id), signal_type, weight, timestamp

    If return_breakdown is True, returns a tuple (combined_df, breakdown_dict)
    where breakdown_dict reports raw counts per source plus filter losses.
    """
    library = _extract_library(conn, tenant_id, since)
    reviews = _extract_reviews(conn, tenant_id, since)
    likes = _extract_place_likes(conn, tenant_id, since)
    checkins = _extract_checkins(conn, tenant_id, since)
    telemetry = _extract_telemetry_signals(conn, tenant_id, since)
    posts = _extract_post_place_signals(conn, tenant_id, since)

    raw_counts = {
        "library": len(library),
        "reviews": len(reviews),
        "place_likes": len(likes),
        "checkins": len(checkins),
        "telemetry": len(telemetry),
        "post_place_signals": len(posts),
    }
    raw_total = sum(raw_counts.values())

    frames = [f for f in (library, reviews, likes, checkins, telemetry, posts) if not f.empty]
    if not frames:
        empty = pd.DataFrame(columns=["user_id", "place_id", "signal_type", "weight", "timestamp"])
        if return_breakdown:
            return empty, {"raw_counts": raw_counts, "raw_total": 0, "after_cap": 0,
                           "dropped_by_cap": 0, "final_total": 0}
        return empty

    combined = pd.concat(frames, ignore_index=True)
    before_cap = len(combined)

    # Cap signals per user to prevent bot/anomalous users from dominating the model.
    if len(combined) > 0:
        combined = combined.groupby("user_id", group_keys=False).head(MAX_SIGNALS_PER_USER)

    after_cap = len(combined)

    if return_breakdown:
        breakdown = {
            "raw_counts": raw_counts,
            "raw_total": raw_total,
            "before_cap": before_cap,
            "after_cap": after_cap,
            "dropped_by_cap": before_cap - after_cap,
            "final_total": after_cap,
            "note_reviews": (
                "Reviews with rating=3 are filtered upstream (weight=0); "
                "reviews count above already excludes them."
            ),
        }
        return combined, breakdown

    return combined


def _since_clause(since: datetime | None) -> tuple[str, tuple]:
    if since is None:
        return " AND TRUE", ()
    return " AND created_at >= %s", (since,)


def _extract_library(conn: psycopg2.extensions.connection, tenant_id: str, since: datetime | None) -> pd.DataFrame:
    extra_sql, extra_args = _since_clause(since)
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            f"""
            SELECT e.owner_user_id AS user_id,
                   COALESCE(e.place_id, '') AS place_id,
                   e.entry_type AS signal_type,
                   e.created_at AS timestamp
            FROM profile_place_entries e
            WHERE e.tenant_id = %s AND e.place_id != ''
            {extra_sql}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["weight"] = df["signal_type"].map(SIGNAL_WEIGHTS).fillna(1.0)
    return df[["user_id", "place_id", "signal_type", "weight", "timestamp"]]


def _extract_reviews(conn: psycopg2.extensions.connection, tenant_id: str, since: datetime | None) -> pd.DataFrame:
    extra_sql, extra_args = _since_clause(since)
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            f"""
            SELECT r.owner_user_id AS user_id,
                   COALESCE(r.place_id, '') AS place_id,
                   r.rating,
                   r.created_at AS timestamp
            FROM profile_reviews r
            WHERE r.tenant_id = %s AND r.place_id != ''
            {extra_sql}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["signal_type"] = df["rating"].apply(lambda r: "review_positive" if r >= 4 else "review_negative" if r <= 2 else "review_neutral")
    df["weight"] = df["signal_type"].map(SIGNAL_WEIGHTS).fillna(0.0)
    # Neutral reviews (3 stars) get zero weight
    df = df[df["weight"] != 0.0]
    return df[["user_id", "place_id", "signal_type", "weight", "timestamp"]]


def _extract_place_likes(conn: psycopg2.extensions.connection, tenant_id: str, since: datetime | None) -> pd.DataFrame:
    extra_sql, extra_args = _since_clause(since)
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            f"""
            SELECT pl.actor_id AS user_id,
                   p.public_id AS place_id,
                   pl.created_at AS timestamp
            FROM place_likes pl
            JOIN places p ON p.id = pl.place_id AND p.tenant_id = pl.tenant_id
            WHERE pl.tenant_id = %s
            {extra_sql}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["signal_type"] = "place_like"
    df["weight"] = SIGNAL_WEIGHTS["place_like"]
    return df[["user_id", "place_id", "signal_type", "weight", "timestamp"]]


def _extract_checkins(conn: psycopg2.extensions.connection, tenant_id: str, since: datetime | None) -> pd.DataFrame:
    extra_sql, extra_args = _since_clause(since)
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            f"""
            SELECT c.user_id,
                   p.public_id AS place_id,
                   c.verified,
                   c.created_at AS timestamp
            FROM checkins c
            JOIN places p ON p.id = c.place_id
            WHERE c.tenant_id = %s
            {extra_sql}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["signal_type"] = "checkin"
    df["weight"] = df["verified"].apply(lambda v: SIGNAL_WEIGHTS["checkin"] if v is True else SIGNAL_WEIGHTS["checkin"] * 0.5)
    return df[["user_id", "place_id", "signal_type", "weight", "timestamp"]]


def _extract_telemetry_signals(conn: psycopg2.extensions.connection, tenant_id: str, since: datetime | None) -> pd.DataFrame:
    extra_sql, extra_args = _since_clause(since)
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            f"""
            SELECT t.user_id,
                   t.entity_id AS place_id,
                   t.event_type,
                   t.payload,
                   t.created_at AS timestamp
            FROM telemetry_events t
            WHERE t.tenant_id = %s
              AND t.entity_type = 'place'
              AND t.entity_id != ''
            {extra_sql}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)

    type_map = {
        "feed_position_click": "telemetry_click",
        "place_detail_view": "telemetry_detail_view",
        "impression": "telemetry_impression",
        "search_query": "telemetry_search",
        "content_view": "telemetry_click",
        "content_like": "telemetry_click",
        "content_create": "post_about_place",
        "route_click": "telemetry_click",
        "visit_feedback": "telemetry_detail_view",
    }
    df["signal_type"] = df["event_type"].map(type_map).fillna("telemetry_other")
    df["weight"] = df["signal_type"].map(SIGNAL_WEIGHTS).fillna(0.1)

    # Anti-fraud: mirror Go's 6 protection layers in ETL
    # 1. Reduce weight for suspicious events (anomaly flag from Go) — strict bool check
    suspicious_mask = df["payload"].apply(lambda p: isinstance(p, dict) and p.get("_suspicious") is True)
    df.loc[suspicious_mask, "weight"] *= 0.1

    # 2. Reduce weight for clamped view_time events (AFK detection from Go) — strict bool check
    clamped_mask = df["payload"].apply(lambda p: isinstance(p, dict) and p.get("_clamped") is True)
    df.loc[clamped_mask, "weight"] *= 0.5

    # 3. Reduce weight for truncated payloads (oversized events from Go) — strict bool check
    truncated_mask = df["payload"].apply(lambda p: isinstance(p, dict) and p.get("_truncated") is True)
    df.loc[truncated_mask, "weight"] *= 0.3

    # 4. Dedup: collapse identical (user, place, event_type) within 1s window
    df = df.sort_values("timestamp")
    df["_dedup_key"] = df["user_id"] + ":" + df["place_id"] + ":" + df["signal_type"]
    df["_prev_ts"] = df.groupby("_dedup_key")["timestamp"].shift(1)
    df["_gap"] = (pd.to_datetime(df["timestamp"]) - pd.to_datetime(df["_prev_ts"])).dt.total_seconds()
    dedup_mask = df["_gap"].notna() & (df["_gap"] < 1.0)
    df = df[~dedup_mask].drop(columns=["_dedup_key", "_prev_ts", "_gap"])

    return df[["user_id", "place_id", "signal_type", "weight", "timestamp"]]


def _extract_post_place_signals(conn: psycopg2.extensions.connection, tenant_id: str, since: datetime | None) -> pd.DataFrame:
    """Extract signals from posts that reference places (posts, likes, saves)."""
    extra_sql, extra_args = _since_clause(since)
    frames = []

    with read_cursor(conn, tenant_id) as cur:
        # Posts about places
        cur.execute(
            f"""
            SELECT p.owner_user_id AS user_id, p.place_id, p.created_at AS timestamp
            FROM posts p
            WHERE p.tenant_id = %s AND p.place_id != ''
            {extra_sql}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows)
            df["signal_type"] = "post_about_place"
            df["weight"] = SIGNAL_WEIGHTS["post_about_place"]
            frames.append(df[["user_id", "place_id", "signal_type", "weight", "timestamp"]])

    like_since = "" if since is None else " AND pl.created_at >= %s"
    save_since = "" if since is None else " AND ps.created_at >= %s"

    with read_cursor(conn, tenant_id) as cur:
        # Likes on place-related posts
        cur.execute(
            f"""
            SELECT pl.actor_id AS user_id, p.place_id, pl.created_at AS timestamp
            FROM post_likes pl
            JOIN posts p ON p.id = pl.post_id
            WHERE p.tenant_id = %s AND p.place_id != ''
            {like_since}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows)
            df["signal_type"] = "post_like"
            df["weight"] = SIGNAL_WEIGHTS["post_like"]
            frames.append(df[["user_id", "place_id", "signal_type", "weight", "timestamp"]])

    with read_cursor(conn, tenant_id) as cur:
        # Saves on place-related posts
        cur.execute(
            f"""
            SELECT ps.actor_id AS user_id, p.place_id, ps.created_at AS timestamp
            FROM post_saves ps
            JOIN posts p ON p.id = ps.post_id
            WHERE p.tenant_id = %s AND p.place_id != ''
            {save_since}
            """,
            (tenant_id, *extra_args),
        )
        rows = cur.fetchall()
        if rows:
            df = pd.DataFrame(rows)
            df["signal_type"] = "post_save"
            df["weight"] = SIGNAL_WEIGHTS["post_save"]
            frames.append(df[["user_id", "place_id", "signal_type", "weight", "timestamp"]])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)
