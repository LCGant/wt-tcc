# Hybrid Recommender — TCC reproducibility package

> [Versão em português](README.pt-BR.md)

This repository contains the recommendation engine and the experiment harness
used in the undergraduate thesis. It is the **academic deliverable** —
self-contained, reproducible, and decoupled from the surrounding application.

## What's inside

```
.
├── src/                       Recommendation engine (Python)
│   ├── cli.py                 CLI entrypoint (`role-ai`)
│   ├── models/                ALS collaborative + TF-IDF content + Thompson bandit
│   ├── inference/             Hybrid composition, cold-start, normalization
│   ├── evaluation/            Offline metrics (P@K, Recall@K, NDCG@K, ILD,
│   │                          Coverage, Serendipity, OBC)
│   ├── etl/                   Postgres → DataFrame extraction
│   └── seed/                  Synthetic-data generators
├── experiments/               Reproducibility scripts
│   ├── run_movielens.sh       MovieLens-100K · 5 seeds · in-memory
│   ├── run_lastfm.sh          Last.fm-2k · 5 seeds · in-memory
│   ├── run_synthetic_pipeline.sh   Volume × seed grid (60 runs · postgres)
│   └── aggregate_*.py         Parse logs → tables.md + results.csv
├── db/migrations/             Schema for the synthetic-pipeline experiments
├── docker-compose.yml         Postgres + AI container (self-contained)
├── Dockerfile                 AI service image
├── pyproject.toml             Python deps
└── requirements.lock          Pinned dep versions
```

## Algorithm overview

The engine is a **hybrid recommender** that combines three signals through a
per-user Thompson Sampling bandit:

- **Content-based**: TF-IDF feature index over place attributes with cosine
  similarity ranking and temporal decay.
- **Collaborative filtering**: implicit-feedback ALS (`implicit` library), with
  graceful fallback to item-item co-occurrence when the binary is unavailable.
- **Thompson Sampling bandit**: per-user Beta-distributed weights for three
  feed components — base preferences, recent preferences, exploration.

Cold-start is handled by classifying users into four levels (no_data,
onboarding_only, warm_few, warm_full) with asymmetric learning rates.

## Quick start

### 1. Prerequisites

- Docker Desktop (or any compose v2 runtime)
- 4 GB RAM available for the containers
- ~1 GB disk for postgres + Python deps

### 2. Public-dataset experiments (no database needed)

These run entirely in-memory. Download the datasets first:

```bash
mkdir -p datasets && cd datasets
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip ml-100k.zip
curl -O https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
unzip hetrec2011-lastfm-2k.zip -d lastfm-2k
cd ..
```

Build the image and run the evaluators:

```bash
docker compose --profile ai build
bash experiments/run_movielens.sh        # ~5 minutes
bash experiments/run_lastfm.sh           # ~5 minutes
python experiments/aggregate_movielens.py
python experiments/aggregate_lastfm.py
```

Aggregated tables land in `experiments/results/` (`movielens_table.md`,
`lastfm_table.md`).

### 3. Synthetic-pipeline experiments (volume × seed grid)

These reproduce Tables 6-8 and Figure 3 of the paper. They require postgres
with the social schema (compose handles that) and take ~30-45 minutes for the
full 10 × 6 grid.

```bash
docker compose up -d postgres                 # boots postgres + applies schema
bash experiments/run_synthetic_pipeline.sh    # 60 runs, ~30-45 min
python experiments/aggregate_experiments.py   # writes results.csv + tables.md
python experiments/plot_figure3.py            # writes Figure 3
```

Outputs:

| File | Description |
|------|-------------|
| `experiments/results/results.csv` | Per-run metrics (one row per seed × volume) |
| `experiments/results/tables.md`   | Aggregated Markdown tables (mean ± SD) |
| `experiments/results/summary.json` | Mean / SD / 95% CI per (volume, metric) |

### 4. Tear down

```bash
docker compose down -v          # also removes the postgres volume
```

## Metrics

| Metric | Range | Meaning |
|--------|-------|---------|
| **P@K** | [0, 1] | Precision: relevant items in top-K |
| **Recall@K** | [0, 1] | Coverage of held-out positives |
| **NDCG@K** | [0, 1] | Rank-aware quality |
| **ILD** | [0, 1] | Intra-list diversity (1 = max diversity) |
| **Coverage** | [0, 1] | Catalog fraction touched by recommendations |
| **Serendipity** | [0, 1] | Unexpected-but-relevant rate |
| **OBC** | [0, 1] | Onboarding–Behavior Correction: 0 = identical to declared, 1 = fully diverged |

## Citing this work

```
@thesis{wt-tcc,
  title  = {Hybrid Recommender System with Thompson Sampling and Onboarding-Behavior Correction},
  author = {Antonio, Lucas Guilherme},
  year   = {2026},
  type   = {Undergraduate Thesis},
}
```

## License

See [LICENSE](LICENSE) (Apache 2.0). The AI engine and experiment harness are
released for academic reproducibility; the surrounding production platform is
out of scope and remains private.
