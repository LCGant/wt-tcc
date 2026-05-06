# Modeling Dynamic Preferences — Hybrid Recommender

> [Versão em português](README.pt-BR.md)

Reproducibility package for the undergraduate thesis **"Modelando Preferências
Dinâmicas: Aplicações de Aprendizado por Reforço em Sistemas de Recomendação"**
(Universidade Cidade de São Paulo · Computer Science, 2026).

**Authors:** Lucas Guilherme Antonio · Thalles Portal Lopes de Miranda ·
Leonardo Marques Pereira Bouzan · Danielle Longati · Matheus Silva Braga ·
Felipe Peçanha Pereira · Daniely Silva de Miranda

The repository contains the recommendation engine and the experiment harness —
self-contained, reproducible, and decoupled from any surrounding application.

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

Two public benchmarks across distinct domains, both run entirely in-memory
through the same Docker image:

| Dataset | Domain | Size | Source |
|---------|--------|------|--------|
| MovieLens-100K | movies | 100K ratings | grouplens.org |
| Last.fm-2k     | music  | ~187K plays  | grouplens.org (HetRec 2011) |

Download whichever you want to run:

```bash
mkdir -p datasets && cd datasets
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
curl -O https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
unzip hetrec2011-lastfm-2k.zip -d lastfm-2k
cd ..
```

Build the image and run the evaluators (independent — run either or both):

```bash
docker compose --profile ai build
bash experiments/run_movielens.sh        # ~7 minutes (5 seeds)
bash experiments/run_lastfm.sh           # ~7 minutes (5 seeds)
python experiments/aggregate_movielens.py
python experiments/aggregate_lastfm.py
```

Aggregated tables land in `experiments/results/` as
`{movielens,lastfm}_table.md`.

### Public-dataset results (5 seeds each)

#### MovieLens-100K — System vs 7 baselines

| Model | P@10 | NDCG@10 | Coverage |
|---|---|---|---|
| Popularity | 25.11% ± 0.00% | 27.28% ± 0.00% | 5.29% |
| **Hybrid system** | **18.47% ± 0.75%** | **20.17% ± 0.64%** | **54.03% ± 0.51%** |
| EASE-R | 10.75% ± 0.00% | 13.62% ± 0.00% | 15.58% |
| iALS (tuned) | 8.48% ± 1.32% | 10.70% ± 2.13% | 27.06% |
| BPR (tuned) | 4.81% ± 0.22% | 6.28% ± 0.88% | 42.35% |
| Item-kNN | 3.36% ± 0.00% | 3.64% ± 0.00% | 8.92% |
| Random | 2.63% ± 0.20% | 2.51% ± 0.25% | — |
| LinUCB (α=1) | 0.00% | 0.00% | 6.90% |

Wilcoxon signed-rank (P@10) is significant (p < 0.05) against every baseline.

#### Last.fm-2k — Cold-start, top-1500 artists, 90-day test split

| Model | P@10 | NDCG@10 | Coverage |
|---|---|---|---|
| EASE-R | 3.56% ± 0.27% | 9.01% ± 0.39% | 45.28% |
| BPR (tuned) | 3.16% ± 0.50% | 8.58% ± 2.07% | 68.72% |
| Popularity | 2.89% ± 0.07% | 8.43% ± 0.19% | 3.67% |
| iALS (tuned) | 2.78% ± 0.46% | 6.76% ± 1.26% | 63.11% |
| Item-kNN | 2.62% ± 0.25% | 5.25% ± 0.47% | 57.97% |
| **Hybrid system** | **2.37% ± 0.25%** | **5.81% ± 0.34%** | **64.05% ± 1.24%** |
| LinUCB (α=1) | 0.31% ± 0.08% | 0.82% ± 0.38% | 41.35% |
| Random | 0.31% ± 0.04% | — | — |

In this cold-start regime the system is *statistically equivalent* to the
strong baselines (Wilcoxon p > 0.05 against item-kNN/iALS/BPR/EASE-R) while
keeping much higher catalog coverage than Popularity.

#### How to read the numbers

The headline takeaway across both benchmarks is the **coverage / precision
trade-off**: even when Popularity wins on P@10 (MovieLens), the proposed
hybrid recommender keeps its 54% catalog coverage versus Popularity's ~5%,
avoiding popularity-bubble collapse while remaining competitive on ranking
quality.

Known limitation in this batch evaluation: **LinUCB with α=1.0** behaves
pathologically because the exploration bonus dominates the mean reward,
biasing the bandit toward arms with no training updates. The default has
been lowered to α=0.1 in the code; the table above reflects the original
α=1.0 run and is reported as-is for transparency.

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

### 4. Optional: vectorised backend (experimental)

The default `evaluate-movielens` command runs the per-user scoring loop
in pure Python. The repository also ships a vectorised PyTorch backend
that replaces that loop with a single `(n_users, n_items)` matmul and
adds opt-in CUDA acceleration:

```bash
# CPU torch path (still vectorised; ~5-10× faster than the default loop)
docker compose --profile ai run --rm ai \
    python -m src.cli evaluate-movielens \
    --root /data/ml-100k --backend torch --seed-value 42

# GPU torch path (CUDA-capable host required)
AI_USE_GPU=1 bash experiments/run_movielens_torch.sh
```

The torch backend mirrors the reference hybrid (Thompson Sampling
bandit + cold-start tier priors + exploratory noise). The CPU
pipeline remains the reference implementation for the reproducibility
tables above; the torch backend is provided as a faster alternative
for batched evaluation.

The vectorised package ships with a pytest suite under
`tests/src_torch/`. Run it inside the container:

```bash
docker compose --profile ai run --rm --entrypoint pytest ai tests
```

(Tests are skipped automatically when the image is built with
`--build-arg INSTALL_TORCH=0`.)

### 5. Tear down

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
  title  = {Modelando Preferências Dinâmicas: Aplicações de Aprendizado por
            Reforço em Sistemas de Recomendação},
  author = {Antonio, Lucas Guilherme and Lopes de Miranda, Thalles Portal and
            Pereira Bouzan, Leonardo Marques and Longati, Danielle and
            Braga, Matheus Silva and Pereira, Felipe Pe{\c c}anha and
            Silva de Miranda, Daniely},
  year         = {2026},
  type         = {Undergraduate Thesis},
  institution  = {Universidade Cidade de S{\~a}o Paulo (UNICID)},
  note         = {Computer Science},
}
```

## License

See [LICENSE](LICENSE) (Apache 2.0). The AI engine and experiment harness are
released for academic reproducibility; the surrounding production platform is
out of scope and remains private.
