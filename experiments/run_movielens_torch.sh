#!/usr/bin/env bash
# MovieLens-100K — torch backend, 5 seeds.
#
# Same protocol as run_movielens.sh but selects the vectorised pipeline
# via --backend=torch. Set AI_USE_GPU=1 in the environment to opt into
# CUDA (otherwise the kernel runs on CPU and is still ~10× faster than
# the per-user Python loop).
#
# Dataset must be unpacked at: datasets/ml-100k/

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="${ROOT}/experiments/results/movielens_torch"
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

    if ! MSYS_NO_PATHCONV=1 docker compose --profile ai run --rm \
        -v "${DATASET}:/data/ml-100k:ro" \
        -e PYTHONHASHSEED="${seed}" \
        -e AI_USE_GPU="${AI_USE_GPU:-0}" \
        --entrypoint python \
        ai -m src.cli evaluate-movielens \
            --backend torch \
            --root /data/ml-100k --test-days 30 --seed-value "${seed}" \
        > "${log}" 2>&1; then
        echo "  FAILED — see ${log}"
    fi
done

END=$(date +%s)
echo ""
echo "Done in $(( (END - START) / 60 ))m $(( (END - START) % 60 ))s"
echo "Logs: ${LOG_DIR}"
echo "Aggregate: python experiments/aggregate_movielens.py ${LOG_DIR}"
