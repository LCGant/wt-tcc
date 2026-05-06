"""Extract place features from PostgreSQL."""

from __future__ import annotations

import pandas as pd
import psycopg2.extensions

from src.db import read_cursor


def extract_places(conn: psycopg2.extensions.connection, tenant_id: str) -> pd.DataFrame:
    """Load all active places with their attributes."""
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            """
            SELECT id, public_id, category, price_level, latitude, longitude,
                   rating_average, ratings_count, verified, status,
                   is_accessible, is_outdoor, is_family_friendly, is_pet_friendly,
                   has_parking, has_wifi, serves_alcohol, accepts_reservations,
                   created_at
            FROM places
            WHERE tenant_id = %s AND status = 'active'
            """,
            (tenant_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def extract_place_highlights(conn: psycopg2.extensions.connection, tenant_id: str) -> dict[int, list[str]]:
    """Load highlights grouped by place_id."""
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            "SELECT place_id, slug FROM place_highlights WHERE tenant_id = %s",
            (tenant_id,),
        )
        rows = cur.fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        out.setdefault(r["place_id"], []).append(r["slug"])
    return out


def extract_place_recommended_for(conn: psycopg2.extensions.connection, tenant_id: str) -> dict[int, list[str]]:
    """Load recommended_for tags grouped by place_id."""
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            "SELECT place_id, slug FROM place_recommended_for WHERE tenant_id = %s",
            (tenant_id,),
        )
        rows = cur.fetchall()
    out: dict[int, list[str]] = {}
    for r in rows:
        out.setdefault(r["place_id"], []).append(r["slug"])
    return out


def extract_place_features(conn: psycopg2.extensions.connection, tenant_id: str) -> pd.DataFrame:
    """Load places enriched with highlights and recommended_for."""
    df = extract_places(conn, tenant_id)
    if df.empty:
        return df

    highlights = extract_place_highlights(conn, tenant_id)
    recommended = extract_place_recommended_for(conn, tenant_id)

    df["highlights"] = df["id"].map(lambda pid: highlights.get(pid, []))
    df["recommended_for"] = df["id"].map(lambda pid: recommended.get(pid, []))

    return df
