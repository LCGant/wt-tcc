"""Write synthetic seed data to PostgreSQL."""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timezone

import psycopg2.extensions
import psycopg2.extras
from psycopg2 import sql

from src.db import transaction

log = logging.getLogger(__name__)


def write_seed(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    places: list[dict],
    interactions: dict,
) -> dict:
    """
    Write all seed data to PostgreSQL.

    Returns summary: {places_created, onboarding_created, ...}
    """
    summary = {}
    now = datetime.now(timezone.utc)

    # 1. Places
    place_db_ids = _write_places(conn, tenant_id, places)
    summary["places"] = len(place_db_ids)

    # Map public_id → db_id for FK references
    pid_map = {p["public_id"]: place_db_ids[i] for i, p in enumerate(places) if i < len(place_db_ids)}

    # 2. Place highlights + recommended_for
    _write_place_metadata(conn, tenant_id, places, pid_map)

    # 3. Onboarding
    summary["onboarding"] = _write_onboarding(conn, tenant_id, interactions.get("onboarding", []))

    # 4. Library entries
    summary["library_entries"] = _write_library(conn, tenant_id, interactions.get("library_entries", []))

    # 5. Reviews
    summary["reviews"] = _write_reviews(conn, tenant_id, interactions.get("reviews", []))

    # 6. Place likes/dislikes
    summary["place_likes"] = _write_place_likes(conn, tenant_id, interactions.get("place_likes", []), pid_map)
    summary["place_dislikes"] = _write_place_dislikes(conn, tenant_id, interactions.get("place_dislikes", []), pid_map)

    # 7. Checkins
    summary["checkins"] = _write_checkins(conn, tenant_id, interactions.get("checkins", []), pid_map)

    # 8. Telemetry
    summary["telemetry"] = _write_telemetry(conn, tenant_id, interactions.get("telemetry", []))

    return summary


def clean_seed(conn: psycopg2.extensions.connection, tenant_id: str, force: bool = False) -> None:
    """Remove all data for a tenant (dangerous — use only for seed cleanup)."""
    if os.environ.get("AI_ENV", "").lower() == "production":
        raise RuntimeError("refusing destructive operation: AI_ENV=production")
    if not force and tenant_id not in ("default", "test", "dev"):
        raise ValueError(f"refusing to clean tenant '{tenant_id}' without force=True")
    with transaction(conn, tenant_id) as cur:
        for table in [
            "telemetry_events", "checkins", "place_likes", "place_dislikes",
            "profile_reviews", "profile_place_entries", "onboarding_responses",
            "place_highlights", "place_recommended_for", "place_hours", "place_media",
            "user_preference_profiles", "analytics_snapshots",
        ]:
            cur.execute(sql.SQL("DELETE FROM {} WHERE tenant_id = %s").format(sql.Identifier(table)), (tenant_id,))
        cur.execute("DELETE FROM places WHERE tenant_id = %s", (tenant_id,))
    log.info("Cleaned all seed data for tenant %s", tenant_id)


def _write_places(conn, tenant_id, places):
    ids = []
    with transaction(conn, tenant_id) as cur:
        for p in places:
            cur.execute(
                """
                INSERT INTO places (public_id, owner_user_id, tenant_id, owner_actor_id, owner_actor_type,
                    name, category, latitude, longitude, rating_average, ratings_count,
                    price_level, verified, status, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'active', %s, %s)
                RETURNING id
                """,
                (p["public_id"], "seed", tenant_id, "seed", "person",
                 p["name"], p["category"], p["latitude"], p["longitude"],
                 p["rating_average"], p["ratings_count"], p["price_level"],
                 p["verified"], p["created_at"], p["created_at"]),
            )
            ids.append(cur.fetchone()["id"])
    return ids


def _write_place_metadata(conn, tenant_id, places, pid_map):
    with transaction(conn, tenant_id) as cur:
        for p in places:
            db_id = pid_map.get(p["public_id"])
            if not db_id:
                continue
            for h in p.get("highlights", []):
                cur.execute(
                    "INSERT INTO place_highlights (place_id, tenant_id, slug) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (db_id, tenant_id, h),
                )
            for r in p.get("recommended_for", []):
                cur.execute(
                    "INSERT INTO place_recommended_for (place_id, tenant_id, slug) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (db_id, tenant_id, r),
                )


def _write_onboarding(conn, tenant_id, entries):
    if not entries:
        return 0
    with transaction(conn, tenant_id) as cur:
        for e in entries:
            cur.execute(
                """
                INSERT INTO onboarding_responses (tenant_id, user_id, actor_id, actor_type, question_key, answer_values)
                VALUES (%s, %s, %s, 'person', %s, %s)
                ON CONFLICT (tenant_id, user_id, question_key) DO UPDATE SET answer_values = EXCLUDED.answer_values
                """,
                (tenant_id, e["user_id"], e["user_id"], e["question_key"], e["answer_values"]),
            )
    return len(entries)


def _write_library(conn, tenant_id, entries):
    if not entries:
        return 0
    count = 0
    with transaction(conn, tenant_id) as cur:
        for e in entries:
            cur.execute(
                """
                INSERT INTO profile_place_entries
                    (public_id, owner_user_id, tenant_id, owner_actor_id, owner_actor_type,
                     entry_type, place_ref, place_id, name_snapshot, visibility, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'person', %s, %s, %s, '', 'public', %s, %s)
                ON CONFLICT (owner_actor_id, owner_actor_type, entry_type, place_ref) DO NOTHING
                """,
                (str(uuid.uuid4()), e["user_id"], tenant_id, e["user_id"],
                 e["entry_type"], e["place_id"], e["place_id"],
                 e["created_at"], e["created_at"]),
            )
            count += 1
    return count


def _write_reviews(conn, tenant_id, entries):
    if not entries:
        return 0
    count = 0
    with transaction(conn, tenant_id) as cur:
        for e in entries:
            cur.execute(
                """
                INSERT INTO profile_reviews
                    (public_id, owner_user_id, tenant_id, owner_actor_id, owner_actor_type,
                     place_ref, place_id, name_snapshot, rating, visibility, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'person', %s, %s, '', %s, 'public', %s, %s)
                ON CONFLICT (owner_actor_id, owner_actor_type, place_ref) DO NOTHING
                """,
                (str(uuid.uuid4()), e["user_id"], tenant_id, e["user_id"],
                 e["place_id"], e["place_id"], e["rating"],
                 e["created_at"], e["created_at"]),
            )
            count += 1
    return count


def _write_place_likes(conn, tenant_id, entries, pid_map):
    if not entries:
        return 0
    count = 0
    with transaction(conn, tenant_id) as cur:
        for e in entries:
            db_id = pid_map.get(e["place_id"])
            if not db_id:
                continue
            cur.execute(
                """
                INSERT INTO place_likes (tenant_id, place_id, actor_id, actor_type, created_at)
                VALUES (%s, %s, %s, 'person', %s)
                ON CONFLICT DO NOTHING
                """,
                (tenant_id, db_id, e["user_id"], e["created_at"]),
            )
            count += 1
    return count


def _write_place_dislikes(conn, tenant_id, entries, pid_map):
    if not entries:
        return 0
    count = 0
    with transaction(conn, tenant_id) as cur:
        for e in entries:
            db_id = pid_map.get(e["place_id"])
            if not db_id:
                continue
            cur.execute(
                """
                INSERT INTO place_dislikes (tenant_id, place_id, actor_id, actor_type, created_at)
                VALUES (%s, %s, %s, 'person', %s)
                ON CONFLICT DO NOTHING
                """,
                (tenant_id, db_id, e["user_id"], e["created_at"]),
            )
            count += 1
    return count


def _write_checkins(conn, tenant_id, entries, pid_map):
    if not entries:
        return 0
    count = 0
    with transaction(conn, tenant_id) as cur:
        for e in entries:
            db_id = pid_map.get(str(e.get("place_db_id", "")))
            if not db_id:
                # Try to find by iterating pid_map values
                continue
            cur.execute(
                """
                INSERT INTO checkins
                    (public_id, tenant_id, user_id, actor_id, actor_type, place_id,
                     latitude, longitude, distance_meters, verified, created_at)
                VALUES (%s, %s, %s, %s, 'person', %s, %s, %s, %s, %s, %s)
                """,
                (str(uuid.uuid4()), tenant_id, e["user_id"], e["user_id"],
                 db_id, e["latitude"], e["longitude"],
                 e["distance_meters"], e["verified"], e["created_at"]),
            )
            count += 1
    return count


def _write_telemetry(conn, tenant_id, entries):
    if not entries:
        return 0
    with transaction(conn, tenant_id) as cur:
        values = [
            (
                tenant_id,
                e["user_id"],
                e["user_id"],
                "person",
                e["event_type"],
                "place",
                e["entity_id"],
                json.dumps(e.get("payload", {})),
                "",
                e["created_at"],
            )
            for e in entries
        ]
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO telemetry_events
                (tenant_id, user_id, actor_id, actor_type, event_type, entity_type, entity_id, payload, session_id, created_at)
            VALUES %s
            """,
            values,
        )
    return len(entries)
