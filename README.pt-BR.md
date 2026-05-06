# Modelando Preferências Dinâmicas — Recomendador Híbrido

> [English version](README.md)

Pacote de reprodutibilidade do Trabalho de Conclusão de Curso **"Modelando
Preferências Dinâmicas: Aplicações de Aprendizado por Reforço em Sistemas de
Recomendação"** (Universidade Cidade de São Paulo · Ciência da Computação,
2026).

**Autores:** Lucas Guilherme Antonio · Thalles Portal Lopes de Miranda ·
Leonardo Marques Pereira Bouzan · Danielle Longati · Matheus Silva Braga ·
Felipe Peçanha Pereira · Daniely Silva de Miranda

O repositório contém o motor de recomendação e o conjunto de experimentos —
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

Dois benchmarks públicos em domínios distintos, ambos rodam em memória pela
mesma imagem Docker:

| Dataset | Domínio | Tamanho | Origem |
|---------|---------|---------|--------|
| MovieLens-100K | filmes | 100K ratings | grouplens.org |
| Last.fm-2k     | música | ~187K plays  | grouplens.org (HetRec 2011) |

Baixar:

```bash
mkdir -p datasets && cd datasets
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
curl -O https://files.grouplens.org/datasets/hetrec2011/hetrec2011-lastfm-2k.zip
unzip hetrec2011-lastfm-2k.zip -d lastfm-2k
cd ..
```

Build da imagem + execução (independentes — pode rodar um, outro, ou ambos):

```bash
docker compose --profile ai build
bash experiments/run_movielens.sh        # ~7 minutos (5 seeds)
bash experiments/run_lastfm.sh           # ~7 minutos (5 seeds)
python experiments/aggregate_movielens.py
python experiments/aggregate_lastfm.py
```

Tabelas agregadas em `experiments/results/` como
`{movielens,lastfm}_table.md`.

### Resultados em datasets públicos (5 seeds cada)

#### MovieLens-100K — Sistema vs 7 baselines

| Modelo | P@10 | NDCG@10 | Coverage |
|---|---|---|---|
| Popularidade | 25,11% ± 0,00% | 27,28% ± 0,00% | 5,29% |
| **Sistema híbrido** | **18,47% ± 0,75%** | **20,17% ± 0,64%** | **54,03% ± 0,51%** |
| EASE-R | 10,75% ± 0,00% | 13,62% ± 0,00% | 15,58% |
| iALS (tuned) | 8,48% ± 1,32% | 10,70% ± 2,13% | 27,06% |
| BPR (tuned) | 4,81% ± 0,22% | 6,28% ± 0,88% | 42,35% |
| Item-kNN | 3,36% ± 0,00% | 3,64% ± 0,00% | 8,92% |
| Aleatório | 2,63% ± 0,20% | 2,51% ± 0,25% | — |
| LinUCB (α=1) | 0,00% | 0,00% | 6,90% |

Wilcoxon signed-rank (P@10) é significativo (p < 0,05) contra todas as
baselines.

#### Last.fm-2k — Cold-start, top-1500 artistas, split de 90 dias

| Modelo | P@10 | NDCG@10 | Coverage |
|---|---|---|---|
| EASE-R | 3,56% ± 0,27% | 9,01% ± 0,39% | 45,28% |
| BPR (tuned) | 3,16% ± 0,50% | 8,58% ± 2,07% | 68,72% |
| Popularidade | 2,89% ± 0,07% | 8,43% ± 0,19% | 3,67% |
| iALS (tuned) | 2,78% ± 0,46% | 6,76% ± 1,26% | 63,11% |
| Item-kNN | 2,62% ± 0,25% | 5,25% ± 0,47% | 57,97% |
| **Sistema híbrido** | **2,37% ± 0,25%** | **5,81% ± 0,34%** | **64,05% ± 1,24%** |
| LinUCB (α=1) | 0,31% ± 0,08% | 0,82% ± 0,38% | 41,35% |
| Aleatório | 0,31% ± 0,04% | — | — |

Em regime cold-start o sistema é *estatisticamente equivalente* às baselines
fortes (Wilcoxon p > 0,05 contra item-kNN/iALS/BPR/EASE-R), mantendo
coverage muito superior à Popularidade.

#### Como ler

A leitura central nos dois benchmarks é o **trade-off coverage / precisão**:
mesmo quando Popularidade vence em P@10 (MovieLens), o sistema mantém 54%
de coverage de catálogo contra ~5% da Popularidade — evita o colapso em
bolha de popularidade enquanto mantém qualidade competitiva no
ranqueamento.

Limitação conhecida na avaliação em batch: **LinUCB com α=1,0** apresenta
comportamento patológico porque o bônus de exploração domina o mean reward,
enviesando o bandit para arms sem updates de treino. O default foi reduzido
para α=0,1 no código; a tabela acima reflete a execução original com α=1,0
e é reportada como tal por transparência.

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

### 4. Opcional: backend vetorizado (experimental)

O comando padrão `evaluate-movielens` roda o loop de scoring por
usuário em Python puro. O repositório também traz um backend em
PyTorch que substitui esse loop por uma única matmul
`(n_users, n_items)` e habilita aceleração via CUDA:

```bash
# Caminho torch em CPU (ainda vetorizado; ~5-10× mais rápido que o loop padrão)
docker compose --profile ai run --rm ai \
    python -m src.cli evaluate-movielens \
    --root /data/ml-100k --backend torch --seed-value 42

# Caminho torch em GPU (host com CUDA necessário)
AI_USE_GPU=1 bash experiments/run_movielens_torch.sh
```

O backend torch espelha a composição de referência (Thompson Sampling
bandit + priors do classificador de cold-start + ruído exploratório).
O pipeline CPU continua sendo a implementação de referência para as
tabelas de reprodução acima; o backend torch é uma alternativa mais
rápida para avaliação em batch.

O pacote vetorizado vem com uma suíte pytest em `tests/src_torch/`.
Para rodar dentro do container:

```bash
docker compose --profile ai run --rm --entrypoint pytest ai tests
```

(Os testes são pulados automaticamente quando a imagem é construída
com `--build-arg INSTALL_TORCH=0`.)

### 5. Encerrar

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
  title  = {Modelando Preferências Dinâmicas: Aplicações de Aprendizado por
            Reforço em Sistemas de Recomendação},
  author = {Antonio, Lucas Guilherme and Lopes de Miranda, Thalles Portal and
            Pereira Bouzan, Leonardo Marques and Longati, Danielle and
            Braga, Matheus Silva and Pereira, Felipe Pe{\c c}anha and
            Silva de Miranda, Daniely},
  year         = {2026},
  type         = {Trabalho de Conclus{\~a}o de Curso},
  institution  = {Universidade Cidade de S{\~a}o Paulo (UNICID)},
  note         = {Ci{\^e}ncia da Computa{\c c}{\~a}o},
}
```

## Licença

Ver [LICENSE](LICENSE) (Apache 2.0). O motor de IA e o conjunto de
experimentos são liberados para reprodutibilidade acadêmica; a plataforma de
produção em volta está fora do escopo e permanece privada.
