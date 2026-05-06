# Recomendador Híbrido — pacote de reprodutibilidade do TCC

> [English version](README.md)

Este repositório contém o motor de recomendação e o conjunto de experimentos
usados no Trabalho de Conclusão de Curso. É a **entrega acadêmica** —
auto-contido, reproduzível, e separado do produto comercial em volta.

## O que tem aqui

```
.
├── src/                       Motor de recomendação (Python)
│   ├── cli.py                 Entrypoint da CLI (`role-ai`)
│   ├── models/                ALS colaborativo + TF-IDF de conteúdo + bandit Thompson
│   ├── inference/             Composição híbrida, cold-start, normalização
│   ├── evaluation/            Métricas offline (P@K, Recall@K, NDCG@K, ILD,
│   │                          Coverage, Serendipity, OBC)
│   ├── etl/                   Extração Postgres → DataFrame
│   └── seed/                  Geradores de dados sintéticos
├── experiments/               Scripts de reprodutibilidade
│   ├── run_movielens.sh       MovieLens-100K · 5 seeds · em memória
│   ├── run_lastfm.sh          Last.fm-2k · 5 seeds · em memória
│   ├── run_synthetic_pipeline.sh   Grid volume × seed (60 runs · postgres)
│   └── aggregate_*.py         Parsing dos logs → tables.md + results.csv
├── db/migrations/             Schema usado pelo experimento de pipeline
├── docker-compose.yml         Postgres + AI (auto-contido)
├── Dockerfile                 Imagem do serviço AI
├── pyproject.toml             Dependências Python
└── requirements.lock          Versões pinadas
```

## Visão geral do algoritmo

O motor é um **recomendador híbrido** que combina três sinais via bandit
Thompson Sampling por usuário:

- **Conteúdo (content-based)**: índice TF-IDF sobre features dos places, com
  ranqueamento por similaridade do cosseno e decaimento temporal.
- **Colaborativo**: ALS para feedback implícito (biblioteca `implicit`), com
  fallback para coocorrência item-item quando o binário não está disponível.
- **Bandit Thompson Sampling**: pesos Beta por usuário para três componentes
  do feed — preferências base, preferências recentes, exploração.

Cold-start é tratado classificando usuários em quatro níveis (sem_dados,
só_onboarding, warm_few, warm_full) com taxas de aprendizado assimétricas.

## Como rodar

### 1. Pré-requisitos

- Docker Desktop (ou qualquer runtime compose v2)
- 4 GB de RAM disponíveis
- ~1 GB de disco para postgres + dependências Python

### 2. Experimentos com datasets públicos (sem banco)

Rodam tudo em memória. Baixe os datasets primeiro:

```bash
mkdir -p datasets && cd datasets
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip && unzip ml-100k.zip
curl -O https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
unzip hetrec2011-lastfm-2k.zip -d lastfm-2k
cd ..
```

Builde a imagem e rode os avaliadores:

```bash
docker compose --profile ai build
bash experiments/run_movielens.sh        # ~5 minutos
bash experiments/run_lastfm.sh           # ~5 minutos
python experiments/aggregate_movielens.py
python experiments/aggregate_lastfm.py
```

As tabelas agregadas vão pra `experiments/results/` (`movielens_table.md`,
`lastfm_table.md`).

### 3. Experimentos do pipeline sintético (grid volume × seed)

Reproduz as Tabelas 6-8 e a Figura 3 do artigo. Precisa do postgres com o
schema social (o compose cuida disso) e leva ~30-45 minutos pro grid completo
de 10 × 6.

```bash
docker compose up -d postgres                 # sobe postgres + aplica schema
bash experiments/run_synthetic_pipeline.sh    # 60 runs, ~30-45 min
python experiments/aggregate_experiments.py   # gera results.csv + tables.md
python experiments/plot_figure3.py            # gera a Figura 3
```

Saída:

| Arquivo | Descrição |
|---------|-----------|
| `experiments/results/results.csv` | Métricas por run (uma linha por seed × volume) |
| `experiments/results/tables.md`   | Tabelas Markdown agregadas (média ± DP) |
| `experiments/results/summary.json` | Média / DP / IC 95% por (volume, métrica) |

### 4. Encerrar

```bash
docker compose down -v          # também remove o volume do postgres
```

## Métricas

| Métrica | Faixa | Significado |
|---------|-------|-------------|
| **P@K** | [0, 1] | Precisão: itens relevantes no top-K |
| **Recall@K** | [0, 1] | Cobertura dos positivos retidos |
| **NDCG@K** | [0, 1] | Qualidade sensível ao ranqueamento |
| **ILD** | [0, 1] | Diversidade intralista (1 = máxima) |
| **Coverage** | [0, 1] | Fração do catálogo recomendada |
| **Serendipity** | [0, 1] | Taxa de "inesperado mas relevante" |
| **OBC** | [0, 1] | Onboarding–Behavior Correction: 0 = igual ao declarado, 1 = totalmente divergente |

## Citando este trabalho

```
@thesis{wt-tcc,
  title  = {Sistema de recomendação híbrido com Thompson Sampling e
            Onboarding-Behavior Correction},
  author = {Antonio, Lucas Guilherme},
  year   = {2026},
  type   = {Trabalho de Conclusão de Curso},
}
```

## Licença

Ver [LICENSE](LICENSE) (Apache 2.0). O motor de IA e o conjunto de
experimentos são liberados para reprodutibilidade acadêmica; a plataforma de
produção em volta está fora do escopo e permanece privada.
