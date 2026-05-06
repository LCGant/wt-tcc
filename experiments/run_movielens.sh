#!/usr/bin/env bash
# MovieLens-100K — 5 seeds, ~30-60s each. No DB required (in-memory).
#
# Dataset must be unpacked at: datasets/ml-100k/
# Get it at https://files.grouplens.org/datasets/movielens/ml-100k.zip

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/experiments/results/movielens"
DATASET="${ROOT}/datasets/ml-100k"

mkdir -p "${LOG_DIR}"

if [ ! -d "${DATASET}" ]; then
    echo "Dataset not found: ${DATASET}"
    echo "Run: cd datasets && unzip ml-100k.zip"
    exit 1
fi

SEEDS=(42 123 7 999 31)

START=$(date +%s)
TOTAL=${#SEEDS[@]}
COUNT=0

cd "${ROOT}"
for seed in "${SEEDS[@]}"; do
    COUNT=$(( COUNT + 1 ))
    ELAPSED=$(( $(date +%s) - START ))
    echo "[${COUNT}/${TOTAL}]  seed=${seed}  (${ELAPSED}s elapsed)"
    log="${LOG_DIR}/seed${seed}.log"

    # The MovieLens evaluator runs entirely in-memory; no postgres needed.
    if ! MSYS_NO_PATHCONV=1 docker compose --profile ai run --rm \
        -v "${DATASET}:/data/ml-100k:ro" \
        -e PYTHONHASHSEED="${seed}" \
        --entrypoint python \
        ai -m src.cli evaluate-movielens \
            --root /data/ml-100k --test-days 30 --seed-value "${seed}" \
        > "${log}" 2>&1; then
        echo "  FAILED — see ${log}"
    fi
done

END=$(date +%s)
echo ""
echo "Done in $(( (END - START) / 60 ))m $(( (END - START) % 60 ))s"
echo "Logs: ${LOG_DIR}"
echo "Aggregate: python experiments/aggregate_movielens.py"
