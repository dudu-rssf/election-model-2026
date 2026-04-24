# Modelo Eleitoral 2026

Modelo bottom-up de previsão eleitoral presidencial para 2026. Prevê percentual de voto por candidato em cada município brasileiro, agrega para estado e nacional, e entrega um dashboard interativo.

## Hipóteses de pesquisa

- **H1** — Prefeitos de partido X favorecem o presidenciável de X.
- **H2** — Municípios com mesmo grupo político dominante há muitos anos entregam mais voto ao candidato apoiado, com efeito heterogêneo por região.
- **H3** — Coligações estaduais do partido têm peso maior que a filiação à base federal.

## Stack

Python 3.11+, dados via Base dos Dados (BigQuery) → Parquet local, LightGBM e PyMC para modelagem, Streamlit + Folium/GeoPandas no dashboard. Versionamento em Git, empacotamento com `venv` + `requirements.txt`.

## Estrutura

```
.
├── config.yaml          # todos os parâmetros, modo dev/prod
├── requirements.txt     # versões travadas
├── Makefile             # atalhos (install, dev, test, lint, clean)
├── run_prod.sh          # pipeline completo (máquina pesada)
├── colab_setup.ipynb    # receita pra rodar em Colab Pro
│
├── src/
│   ├── config.py            # carrega config.yaml
│   ├── ingestion/           # wrappers Base dos Dados
│   ├── features/            # panel, historical, continuity, local_power, vertical, structural
│   ├── models/              # baseline, lgbm, bayesian
│   ├── aggregation/         # município → UF → nacional
│   └── dashboard/           # app Streamlit
│
├── scripts/             # CLI entrypoints (01_ingest ... 07_dashboard)
├── notebooks/           # exploração, SEM lógica de produção
├── tests/               # pytest
├── reports/             # status, validações, análise de hipóteses
└── data/                # NÃO COMITADO (raw / interim / processed)
```

## Desenvolvendo (modo dev)

Requer Python 3.11+.

### Primeira vez (bootstrap)

Após copiar/clonar o projeto:

```bash
# Inicializa git local + (opcional) aponta pro GitHub.
# Se existir uma pasta .git herdada sem objects/ (artefato de sandbox), o script limpa.
bash bootstrap_git.sh https://github.com/SEU_USUARIO/SEU_REPO.git
```

### Rodar o pipeline

```bash
make install                   # cria .venv e instala deps
# editar config.yaml se quiser ajustar amostra; mantém mode: dev
make dev                       # roda pipeline completo em amostra
make test                      # roda pytest
```

Em modo dev o pipeline processa apenas São Paulo, três eleições presidenciais (2014, 2018, 2022) e 100 municípios amostrados. Deve completar em poucos minutos em qualquer laptop moderno.

## Rodando em produção (modo prod)

Modo prod processa todos os 5.570 municípios e todas as eleições desde 1998. Componentes pesados:

- Ingestão Base dos Dados: depende da rede; minutos a dezenas de minutos.
- Treino LightGBM: minutos.
- Treino PyMC hierárquico: **horas** (potencialmente > 10h dependendo da máquina e do número de níveis hierárquicos).

### Colab Pro (padrão)

Abrir `colab_setup.ipynb` no Colab Pro e seguir as células. O notebook clona o repo, instala dependências, monta o Google Drive para persistir `data/` e `models/`, troca `config.yaml` para `mode: prod` e dispara `run_prod.sh`.

Atenção ao limite de 12 horas de sessão. Se o PyMC exceder, considere reduzir o pooling hierárquico ou mover o treino para uma VM persistente.

### Servidor próprio / VM na nuvem

```bash
git clone <repo-url>
cd eleicao2026
make install
# preencher config.yaml: gcp.billing_project_id
# autenticar: gcloud auth application-default login
bash run_prod.sh
```

## Configuração

Toda configuração vive em `config.yaml`. Módulos acessam via `src.config`:

```python
from src.config import MODE, SEED, PATHS, MODE_CFG
```

Para trocar amostra dev, editar a seção `dev:` em `config.yaml`.

## Regras de portabilidade

- Nenhum path absoluto; tudo relativo via `src.config.PATHS`.
- Seeds fixos (`SEED = 42`) em todos os módulos com aleatoriedade.
- Dados, modelos e `.venv` não entram no repositório.
- Toda dependência nova precisa ser adicionada em `requirements.txt`.
