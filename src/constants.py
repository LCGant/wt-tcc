"""Shared constants for the recommendation engine."""

from __future__ import annotations

# ── Temporal windows ──────────────────────────────────────────────────
RECENT_WINDOW_DAYS = 14
BASE_HALF_LIFE_DAYS = 90.0
RECENT_HALF_LIFE_DAYS = 3.0

# ── Cold-start thresholds ────────────────────────────────────────────
WARM_FULL_THRESHOLD = 10

# ── Bandit learning rates per cold-start level ───────────────────────
LEARNING_RATE_WARM_FEW = 1.5
LEARNING_RATE_WARM_FULL = 1.0

# ── ETL caps ─────────────────────────────────────────────────────────
MAX_SIGNALS_PER_USER = 500
