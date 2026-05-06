"""Extract user data from PostgreSQL."""

from __future__ import annotations

import pandas as pd
import psycopg2.extensions

from src.db import read_cursor


def extract_user_ids(conn: psycopg2.extensions.connection, tenant_id: str) -> list[str]:
    """Return all user IDs in the tenant that have a profile."""
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            "SELECT DISTINCT user_id FROM profiles WHERE tenant_id = %s",
            (tenant_id,),
        )
        return [r["user_id"] for r in cur.fetchall()]


def extract_onboarding(conn: psycopg2.extensions.connection, tenant_id: str) -> pd.DataFrame:
    """Load onboarding responses for all users."""
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            """
            SELECT user_id, question_key, answer_values
            FROM onboarding_responses
            WHERE tenant_id = %s
            """,
            (tenant_id,),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["user_id", "question_key", "answer_values"])
    return pd.DataFrame(rows)
