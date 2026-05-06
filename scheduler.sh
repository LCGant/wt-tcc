#!/bin/sh
set -eu
trap 'kill 0; exit 0' TERM INT

# AI Scheduler — runs incremental updates, periodic recomputes, and weekly training.
# Designed to run as a long-lived container alongside the AI service.
#
# Schedule:
#   - update (incremental):  every hour
#   - recompute (full):      every 6 hours
#   - train (models):        every Sunday at 03:00
#
# All commands are idempotent and safe to restart at any time.

TENANT="${TENANT_ID:-default}"
LOG_PREFIX="[ai-scheduler]"

log() { echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') $LOG_PREFIX $*"; }

run_update() {
    log "Starting incremental update..."
    if python -m src.cli update --tenant "$TENANT"; then
        log "Incremental update completed."
    else
        log "ERROR: incremental update failed (exit $?)."
    fi
}

run_recompute() {
    log "Starting full recompute..."
    if python -m src.cli recompute --tenant "$TENANT"; then
        log "Full recompute completed."
    else
        log "ERROR: full recompute failed (exit $?)."
    fi
}

run_train() {
    log "Starting model training..."
    if python -m src.cli train --tenant "$TENANT"; then
        log "Model training completed."
    else
        log "ERROR: model training failed (exit $?)."
    fi
}

# Initial run: recompute on startup to ensure preferences are fresh.
log "Scheduler started. Tenant=$TENANT"

# Ensure only one instance runs at a time.
LOCKFILE="/app/data/ai-scheduler.lock"
exec 200>"$LOCKFILE"
if ! flock -n 200; then
    log "Another scheduler instance running. Exiting."
    exit 0
fi

log "Running initial recompute..."
run_recompute

# Track hours since last recompute and day of week for training.
hours_since_recompute=0

while true; do
    sleep 3600  # 1 hour

    hours_since_recompute=$((hours_since_recompute + 1))

    # Every hour: incremental update
    run_update

    # Every 6 hours: full recompute
    if [ "$hours_since_recompute" -ge 6 ]; then
        run_recompute
        hours_since_recompute=0
    fi

    # Sunday at ~03:00 UTC: weekly training
    current_day=$(date -u '+%u')   # 7 = Sunday
    current_hour=$(date -u '+%H')
    if [ "$current_day" = "7" ] && [ "$current_hour" = "03" ]; then
        run_train
    fi
done
