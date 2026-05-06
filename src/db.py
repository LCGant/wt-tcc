"""PostgreSQL connection management."""

from __future__ import annotations

import contextlib
import os
import re
from typing import Generator

import psycopg2
import psycopg2.extras

_TENANT_ID = os.environ.get("TENANT_ID", "default").strip()

# Strict allowlist: tenant IDs must be alphanumeric/dash/underscore only.
_VALID_TENANT_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_tenant_id(tid: str) -> str:
    """Validate tenant ID against strict allowlist. Raises ValueError on bad input."""
    if not tid or not _VALID_TENANT_RE.match(tid):
        raise ValueError(f"invalid tenant_id: must match [a-zA-Z0-9_-]+, got {tid!r}")
    return tid


def _set_tenant(cur, tenant_id: str | None = None) -> None:
    """Set the RLS tenant context for the current transaction."""
    tid = _validate_tenant_id(tenant_id or _TENANT_ID)
    cur.execute("SET LOCAL app.current_tenant = %s", (tid,))


def connect(db_url: str) -> psycopg2.extensions.connection:
    """Open a new database connection."""
    conn = psycopg2.connect(db_url)
    conn.autocommit = False
    return conn


@contextlib.contextmanager
def transaction(conn: psycopg2.extensions.connection, tenant_id: str | None = None) -> Generator[psycopg2.extensions.cursor, None, None]:
    """Context manager that commits on success or rolls back on error."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        _set_tenant(cur, tenant_id)
        yield cur
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()


@contextlib.contextmanager
def read_cursor(conn: psycopg2.extensions.connection, tenant_id: str | None = None) -> Generator[psycopg2.extras.RealDictCursor, None, None]:
    """Context manager for read-only queries. Sets RLS tenant, rolls back at end."""
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        _set_tenant(cur, tenant_id)
        yield cur
    finally:
        conn.rollback()
        cur.close()
