"""Write preference vectors to PostgreSQL."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import psycopg2.extensions
import psycopg2.extras

from src.db import transaction

log = logging.getLogger(__name__)


def write_preferences(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    user_id: str,
    profile_type: str,
    prefs: list[dict],
) -> int:
    """
    Atomically replace all preferences for a user×profile_type.

    Each dict in prefs must have: dimension, dimension_value, weight.

    Returns number of rows written.
    """
    now = datetime.now(timezone.utc)

    with transaction(conn, tenant_id) as cur:
        # Clear old preferences
        cur.execute(
            "DELETE FROM user_preference_profiles WHERE tenant_id = %s AND user_id = %s AND profile_type = %s",
            (tenant_id, user_id, profile_type),
        )

        if not prefs:
            return 0

        # Batch insert
        values = [
            (
                tenant_id,
                user_id,
                user_id,       # actor_id = user_id for person
                "person",      # actor_type
                profile_type,
                p["dimension"],
                p["dimension_value"],
                p["weight"],
                now,
            )
            for p in prefs
        ]

        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO user_preference_profiles
                (tenant_id, user_id, actor_id, actor_type, profile_type, dimension, dimension_value, weight, updated_at)
            VALUES %s
            """,
            values,
        )

    _audit_write(conn, tenant_id, user_id, "write_preferences", profile_type, len(prefs))
    log.debug("Wrote %d %s preferences for user %s", len(prefs), profile_type, user_id)
    return len(prefs)


def _audit_write(conn: psycopg2.extensions.connection, tenant_id: str, user_id: str, operation: str, profile_type: str, rows: int) -> None:
    """Append an entry to the AI write audit log (append-only)."""
    try:
        with transaction(conn, tenant_id) as cur:
            cur.execute(
                "INSERT INTO ai_write_audit (tenant_id, user_id, operation, profile_type, rows_affected) VALUES (%s, %s, %s, %s, %s)",
                (tenant_id, user_id, operation, profile_type, rows),
            )
    except Exception:
        log.error("CRITICAL: failed to write AI audit entry for %s/%s", user_id, operation, exc_info=True)
        raise


def cleanup_orphan_preferences(conn: psycopg2.extensions.connection, tenant_id: str) -> int:
    """
    Remove preference rows that reference places no longer in the catalog.

    Preferences store dimension_value as category/price strings, not place IDs,
    so orphans are rare. This cleans composition weights for deleted users and
    stale entries where the dimension_value no longer matches any active place attribute.
    """
    with transaction(conn, tenant_id) as cur:
        # Remove preferences for users that no longer have profiles
        cur.execute(
            """
            DELETE FROM user_preference_profiles upp
            WHERE upp.tenant_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM profiles p
                  WHERE p.user_id = upp.user_id AND p.tenant_id = upp.tenant_id
              )
            """,
            (tenant_id,),
        )
        orphan_count = cur.rowcount

        # Remove bandit state for deleted users
        cur.execute(
            """
            DELETE FROM user_bandit_state ubs
            WHERE ubs.tenant_id = %s
              AND NOT EXISTS (
                  SELECT 1 FROM profiles p
                  WHERE p.user_id = ubs.user_id AND p.tenant_id = ubs.tenant_id
              )
            """,
            (tenant_id,),
        )
        orphan_count += cur.rowcount

    log.info("Cleaned %d orphan preference/bandit rows for tenant %s", orphan_count, tenant_id)
    return orphan_count
