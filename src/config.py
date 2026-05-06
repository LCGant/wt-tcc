"""Configuration loaded from environment variables."""

import logging
import os
import sys


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        print(f"FATAL: required environment variable {name} is not set", file=sys.stderr)
        sys.exit(1)
    return value


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip()


# Database — AI_DB_URL is mandatory (restricted ai_worker role)
DB_URL: str = _require_env("AI_DB_URL")

# Admin DB URL — owner role for seed operations (write to all tables)
ADMIN_DB_URL: str = _env("SOCIAL_DB_URL", "")

# Tenant
TENANT_ID: str = _env("TENANT_ID", "default")

# Logging
LOG_LEVEL: str = _env("LOG_LEVEL", "info").upper()

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)

log = logging.getLogger("role-ai")
