#!/usr/bin/env bash
# Synthetic-data pipeline: 10 seeds × 6 volumes (interactions/user) = 60 runs.
# Reproduces the volume-vs-quality curves (Tables 6-8 + Figure 3 of the paper).
# Each run executes the full pipeline (seed → train → recompute → evaluate)
# inside the AI container against a fresh postgres database.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/experiments/results/pipeline"
mkdir -p "${LOG_DIR}"

SEEDS=(42 123 7 999 31 4242 314159 271828 161803 1729)
VOLUMES=(2 5 10 20 40 80)

START=$(date +%s)
TOTAL=$(( ${#SEEDS[@]} * ${#VOLUMES[@]} ))
COUNT=0

cd "${ROOT}"
for seed in "${SEEDS[@]}"; do
    for vol in "${VOLUMES[@]}"; do
        COUNT=$(( COUNT + 1 ))
        ELAPSED=$(( $(date +%s) - START ))
        echo "[${COUNT}/${TOTAL}]  seed=${seed} volume=${vol}  (${ELAPSED}s elapsed)"
        log="${LOG_DIR}/seed${seed}_vol${vol}.log"

        if ! MSYS_NO_PATHCONV=1 docker compose --profile ai run --rm \
            ai pipeline \
                --users 200 --places 500 \
                --interactions-per-user "${vol}" \
                --days 90 --seed-value "${seed}" \
            > "${log}" 2>&1; then
            echo "  FAILED — see ${log}"
        fi
    done
done

END=$(date +%s)
echo ""
echo "Done in $(( (END - START) / 60 ))m $(( (END - START) % 60 ))s"
echo "Logs: ${LOG_DIR}"
echo "Aggregate: python experiments/aggregate_experiments.py"
