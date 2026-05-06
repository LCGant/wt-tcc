#!/usr/bin/env bash
# Last.fm-2k evaluation across 5 seeds (no DB required).
#
# Dataset must be unpacked at: datasets/lastfm-2k/
# Get it at https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/experiments/results/lastfm"
DATASET="${ROOT}/datasets/lastfm-2k"

mkdir -p "${LOG_DIR}"
[ -d "${DATASET}" ] || { echo "Dataset not found: ${DATASET}"; exit 1; }

SEEDS=(42 123 7 999 31)

START=$(date +%s)
cd "${ROOT}"
for seed in "${SEEDS[@]}"; do
    log="${LOG_DIR}/seed${seed}.log"
    echo "  Last.fm seed=${seed}"
    if ! MSYS_NO_PATHCONV=1 docker compose --profile ai run --rm \
        -v "${DATASET}:/data/lastfm:ro" \
        --entrypoint python \
        ai -m src.cli evaluate-movielens \
            --root /data/lastfm --variant lastfm \
            --test-days 90 --seed-value "${seed}" \
            --cold-fraction 0.5 --cold-keep-k 3 \
        > "${log}" 2>&1; then
        echo "    FAILED — see ${log}"
    fi
done

END=$(date +%s)
echo ""
echo "Done in $(( (END - START) / 60 ))m $(( (END - START) % 60 ))s"
echo "Logs: ${LOG_DIR}"
echo "Aggregate: python experiments/aggregate_lastfm.py"
