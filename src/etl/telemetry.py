"""Extract telemetry events from PostgreSQL."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
import psycopg2.extensions

from src.db import read_cursor


def extract_telemetry(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    since: datetime,
    entity_type: str = "place",
) -> pd.DataFrame:
    """Load telemetry events, optionally filtered by entity type and time."""
    with read_cursor(conn, tenant_id) as cur:
        cur.execute(
            """
            SELECT t.user_id, t.event_type, t.entity_type, t.entity_id,
                   t.payload, t.session_id, t.created_at,
                   p.id AS place_db_id, p.category, p.price_level
            FROM telemetry_events t
            LEFT JOIN places p ON p.public_id = t.entity_id AND p.tenant_id = t.tenant_id
            WHERE t.tenant_id = %s
              AND t.created_at >= %s
              AND t.entity_type = %s
            ORDER BY t.created_at DESC
            """,
            (tenant_id, since, entity_type),
        )
        rows = cur.fetchall()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)
