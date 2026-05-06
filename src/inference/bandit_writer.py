"""Read/write bandit state and composition weights."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import psycopg2.extensions

from src.db import transaction, read_cursor
from src.models.bandit import BanditState, sample_weights, compute_rewards, update_state

log = logging.getLogger(__name__)


def load_bandit_state(conn: psycopg2.extensions.connection, tenant_id: str, user_id: str) -> BanditState:
    """Load bandit state from DB, returning defaults if not found."""
    with read_cursor(conn) as cur:
        cur.execute(
            "SELECT base_alpha, base_beta, recent_alpha, recent_beta, exploratory_alpha, exploratory_beta "
            "FROM user_bandit_state WHERE tenant_id = %s AND user_id = %s",
            (tenant_id, user_id),
        )
        row = cur.fetchone()
    if not row:
        return BanditState()
    return BanditState(
        base_alpha=row["base_alpha"], base_beta=row["base_beta"],
        recent_alpha=row["recent_alpha"], recent_beta=row["recent_beta"],
        exploratory_alpha=row["exploratory_alpha"], exploratory_beta=row["exploratory_beta"],
    )


def save_bandit_state(conn: psycopg2.extensions.connection, tenant_id: str, user_id: str, state: BanditState) -> None:
    """Upsert bandit state to DB."""
    now = datetime.now(timezone.utc)
    with transaction(conn, tenant_id) as cur:
        cur.execute(
            """
            INSERT INTO user_bandit_state (tenant_id, user_id, base_alpha, base_beta,
                recent_alpha, recent_beta, exploratory_alpha, exploratory_beta, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (tenant_id, user_id) DO UPDATE SET
                base_alpha = EXCLUDED.base_alpha, base_beta = EXCLUDED.base_beta,
                recent_alpha = EXCLUDED.recent_alpha, recent_beta = EXCLUDED.recent_beta,
                exploratory_alpha = EXCLUDED.exploratory_alpha, exploratory_beta = EXCLUDED.exploratory_beta,
                updated_at = EXCLUDED.updated_at
            """,
            (tenant_id, user_id, state.base_alpha, state.base_beta,
             state.recent_alpha, state.recent_beta,
             state.exploratory_alpha, state.exploratory_beta, now),
        )


def write_composition_weights(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    user_id: str,
    base_w: float,
    recent_w: float,
    exploratory_w: float,
) -> None:
    """Write sampled composition weights to user_preference_profiles."""
    now = datetime.now(timezone.utc)
    with transaction(conn, tenant_id) as cur:
        # Clear old composition weights
        cur.execute(
            "DELETE FROM user_preference_profiles WHERE tenant_id = %s AND user_id = %s AND profile_type = 'composition'",
            (tenant_id, user_id),
        )
        # Insert new weights
        for dim_val, weight in [("base", base_w), ("recent", recent_w), ("exploratory", exploratory_w)]:
            cur.execute(
                """
                INSERT INTO user_preference_profiles
                    (tenant_id, user_id, actor_id, actor_type, profile_type, dimension, dimension_value, weight, updated_at)
                VALUES (%s, %s, %s, 'person', 'composition', 'weight', %s, %s, %s)
                """,
                (tenant_id, user_id, user_id, dim_val, weight, now),
            )


def run_bandit_update(
    conn: psycopg2.extensions.connection,
    tenant_id: str,
    user_id: str,
    recent_telemetry: list[dict],
    learning_rate: float = 1.0,
) -> tuple[float, float, float]:
    """
    Full bandit cycle for one user:
    1. Load state
    2. Compute rewards from recent telemetry
    3. Update state with rewards (asymmetric learning for new users)
    4. Sample new weights
    5. Save state + write composition weights
    Returns (base_w, recent_w, exploratory_w)
    """
    state = load_bandit_state(conn, tenant_id, user_id)

    # Update from rewards if telemetry exists
    if recent_telemetry:
        rewards = compute_rewards(recent_telemetry)
        state = update_state(state, rewards, learning_rate=learning_rate)

    # Sample new weights
    base_w, recent_w, exploratory_w = sample_weights(state)

    # Persist
    save_bandit_state(conn, tenant_id, user_id, state)
    write_composition_weights(conn, tenant_id, user_id, base_w, recent_w, exploratory_w)

    # Audit trail
    from src.inference.writer import _audit_write
    _audit_write(conn, tenant_id, user_id, "bandit_update", "composition", 3)

    return base_w, recent_w, exploratory_w
