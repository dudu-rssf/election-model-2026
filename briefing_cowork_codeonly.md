# Briefing Cowork — Modelo Eleitoral 2026 (Modo Code-Only)

> **Para o Cowork:** sua função neste projeto é **escrever e validar código**, não executar o pipeline completo. Execução pesada (treino de modelo bayesiano, processamento dos 5.570 municípios) será feita em outra máquina. Você trabalha sempre em **modo dev**: amostra pequena, execução rápida, foco em correção lógica e reprodutibilidade.

---

## PARTE 0 — REGRAS DE OURO (leia primeiro e sempre respeite)

### 1. Você NUNCA roda o pipeline completo

- Toda vez que for executar código, use a flag `--mode dev` (definida no `config.yaml`)
- Modo `dev` = 1 estado (São Paulo), 3 eleições presidenciais (2014, 2018, 2022), amostra de 100 municípios do estado
- Modo `prod` = Brasil inteiro, todas as eleições. **Isso é responsabilidade do usuário em outra máquina.**
- Se algum script não tiver o toggle dev/prod, **adicione antes de rodar**

### 2. Código precisa ser portável

O código vai rodar em pelo menos 2 máquinas diferentes. Portanto:
- **NUNCA** hardcode paths absolutos (nada de `/Users/fulano/...`). Use `Path(__file__).parent` ou paths relativos a partir de `config.yaml`
- **NUNCA** assuma que bibliotecas estão instaladas — tudo no `requirements.txt`
- **SEMPRE** fixe seeds (`SEED = 42` em config)
- **SEMPRE** salve versões de bibliotecas no `requirements.txt` com `==` (pin exato)

### 3. Reprodutibilidade acima de tudo

Cada script deve poder ser rodado isoladamente dado o estado do diretório anterior. Um comando único (`make all` ou `bash run_all.sh`) deve reproduzir tudo do zero. Se não puder, o código está errado.

### 4. Git é a fonte da verdade

- Commit no fim de cada fase, com mensagem descritiva
- Dados NUNCA entram no repo (`.gitignore` cuida disso)
- Modelos treinados NUNCA entram no repo (são gerados em outra máquina)
- Apenas código, configs, documentação, relatórios markdown

### 5. Teste tudo com amostra

Antes de declarar qualquer fase "pronta", o código deve rodar em modo `dev` de ponta a ponta sem erro. Se travar em modo dev, vai travar em prod.

---

## PARTE 1 — CONTEXTO DO PROJETO

### Objetivo

Modelo bottom-up de previsão eleitoral presidencial 2026. Prever % voto por candidato em cada município brasileiro, agregar para estado e nacional, entregar dashboard interativo.

### Hipóteses de pesquisa

- **H1:** Prefeitos de partido X favorecem presidenciável de X
- **H2:** Municípios com mesmo grupo político dominante há muitos anos entregam mais voto ao candidato apoiado, com efeito heterogêneo por região
- **H3:** Coligações estaduais do partido têm peso maior que filiação à base federal

### Stack técnica

- Python 3.11+ (lock específico no `requirements.txt`)
- Dados: Base dos Dados (BigQuery) → Parquet local
- Modelagem: LightGBM (rápido) + PyMC (hierárquico bayesiano)
- Dashboard: Streamlit + Folium/GeoPandas
- Empacotamento: venv + `requirements.txt` (sem Docker por ora)
- Versionamento: Git

### Decisões de infraestrutura

- **Desenvolvimento (Cowork):** laptop do usuário, modo `dev` sempre
- **Execução pesada:** Google Colab Pro é o padrão assumido. Código deve funcionar lá sem mudança. Se o usuário escolher outra máquina depois, o código já está portável.
- **Transferência:** GitHub (repo privado que o usuário vai criar)

---

## PARTE 2 — ESTRUTURA DO PROJETO

### Layout de diretórios

```
eleicao2026/
├── .gitignore              # data/, *.parquet, models/, .venv/
├── README.md               # overview, como rodar em dev e prod
├── requirements.txt        # versões travadas
├── config.yaml             # TODOS os parâmetros, paths, modo dev/prod
├── Makefile                # atalhos: make install, make dev, make test
├── run_prod.sh             # script para rodar tudo em modo prod (pra máquina pesada)
├── colab_setup.ipynb       # notebook pra setup automático no Colab
│
├── src/
│   ├── __init__.py
│   ├── config.py           # carrega config.yaml, expõe constants
│   ├── ingestion/
│   ├── features/
│   ├── models/
│   ├── aggregation/
│   └── dashboard/
│
├── scripts/                # CLI entrypoints
│   ├── 01_ingest.py
│   ├── 02_build_panel.py
│   ├── 03_features.py
│   ├── 04_hypotheses.py
│   ├── 05_train.py
│   ├── 06_predict.py
│   └── 07_dashboard.py
│
├── notebooks/              # exploração, SEM lógica de produção
├── tests/                  # pytest, testes unitários
├── reports/                # markdown gerado + plots
└── data/                   # NÃO COMITAR
    ├── raw/
    ├── interim/
    └── processed/
```

### `config.yaml` — template obrigatório

```yaml
# Modo de execução: dev (amostra) ou prod (completo)
mode: dev

# Semente global
seed: 42

# Paths (relativos à raiz do projeto)
paths:
  data_raw: data/raw
  data_interim: data/interim
  data_processed: data/processed
  models: models
  reports: reports

# Google Cloud (o usuário preenche)
gcp:
  billing_project_id: null   # PREENCHER ANTES DE RODAR

# Modo dev — amostra reduzida
dev:
  ufs: ["SP"]
  anos_presidencial: [2014, 2018, 2022]
  max_municipios: 100

# Modo prod — tudo
prod:
  ufs: "all"
  anos_presidencial: [1998, 2002, 2006, 2010, 2014, 2018, 2022]
  max_municipios: null

# Modelo
model:
  lgbm:
    n_estimators: 500
    learning_rate: 0.05
  bayesian:
    chains: 4
    warmup: 2000
    samples: 2000
    target_accept: 0.9
```

### `src/config.py` — ponto único de acesso

```python
from pathlib import Path
import yaml

ROOT = Path(__file__).parent.parent
CONFIG = yaml.safe_load((ROOT / "config.yaml").read_text())

MODE = CONFIG["mode"]
SEED = CONFIG["seed"]
MODE_CFG = CONFIG[MODE]  # expõe "dev" ou "prod" automaticamente

# Paths absolutos derivados
PATHS = {k: ROOT / v for k, v in CONFIG["paths"].items()}
```

Todo módulo importa de `src.config`. Ninguém lê YAML diretamente.

### `.gitignore` mínimo

```
data/
models/
.venv/
__pycache__/
*.pyc
.ipynb_checkpoints/
.env
reports/**/*.png
reports/**/*.html
!reports/.gitkeep
```

### `requirements.txt` — travado

Ao criar, rode `pip freeze > requirements.txt` depois da instalação inicial, assim todas as versões ficam pinadas. Revise pra remover libs do sistema (não precisam aparecer).

Pacotes essenciais a incluir:
```
basedosdados
pandas
polars
pyarrow
numpy
scipy
scikit-learn
lightgbm
pymc
arviz
pytensor
geopandas
geobr
folium
streamlit
plotly
shap
optuna
pyyaml
pytest
jupyter
```

### `Makefile`

```makefile
.PHONY: install dev test clean lint

install:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

dev:
	.venv/bin/python scripts/01_ingest.py
	.venv/bin/python scripts/02_build_panel.py
	.venv/bin/python scripts/03_features.py
	.venv/bin/python scripts/04_hypotheses.py
	.venv/bin/python scripts/05_train.py
	.venv/bin/python scripts/06_predict.py

test:
	.venv/bin/pytest tests/ -v

lint:
	.venv/bin/python -m ruff check src/ scripts/

clean:
	rm -rf data/interim data/processed models
```

### `run_prod.sh` — pra máquina pesada

```bash
#!/bin/bash
set -e
# Muda modo pra prod temporariamente
python -c "import yaml; c=yaml.safe_load(open('config.yaml')); c['mode']='prod'; yaml.safe_dump(c, open('config.yaml','w'))"
python scripts/01_ingest.py
python scripts/02_build_panel.py
python scripts/03_features.py
python scripts/05_train.py
python scripts/06_predict.py
echo "Pipeline prod concluído. Artefatos em models/ e data/processed/"
```

### `colab_setup.ipynb` — receita pro usuário rodar no Colab

Notebook que:
1. Clona o repo do GitHub
2. Instala dependências
3. Monta o Google Drive (pra persistir dados/modelos)
4. Altera config pra `mode: prod`
5. Roda o pipeline

Cada célula documentada. Usuário só precisa editar uma célula com o URL do repo dele.

---

## PARTE 3 — FASES EXECUTÁVEIS

Cada fase produz código testado em modo dev. Você **não** executa em prod.

### FASE 0 — Bootstrap

**Tarefas:**
1. Criar toda a estrutura de pastas + arquivos base (`.gitignore`, `README.md` esqueleto, `Makefile`, `config.yaml`, `src/config.py`, `run_prod.sh`, `colab_setup.ipynb`)
2. Criar venv, instalar dependências, gerar `requirements.txt` final com versões travadas
3. Criar repo git local (`git init`), fazer primeiro commit
4. **PARAR E PERGUNTAR:** usuário precisa (a) criar repo privado no GitHub e fornecer URL pra `git remote add origin`, (b) fornecer `billing_project_id` do Google Cloud pra `config.yaml`

**Critério de aceite:** `make install` funciona; `python -c "from src.config import MODE; print(MODE)"` imprime "dev"

**Entregável:** estrutura completa + primeiro commit

---

### FASE 1 — Ingestão

**Tarefas:**
1. Escrever `src/ingestion/bd_client.py`: wrapper fino do `basedosdados` com cache local em Parquet
2. Escrever `scripts/01_ingest.py` que baixa:
   - Resultados presidenciais por município (filtro respeitando `MODE_CFG`)
   - Resultados de prefeito por município
   - Candidatos (com coligações)
   - Diretório de municípios (IBGE)
   - Geometrias municipais (via `geobr`)
3. Implementar validação: `src/ingestion/validate.py` — soma dos votos do vencedor bate com oficial (tolerância 0.1%)
4. **Rodar em modo dev:** deve completar em < 5 minutos baixando só dados de SP

**Em modo dev:**
- Filtro SQL adiciona `AND sigla_uf IN ('SP')`
- Filtro de anos aplicado
- Amostra de 100 municípios aplicada **após** download (pra manter estrutura realista)

**Entregável:** ingestão funciona em dev; parquets em `data/raw/`; relatório de validação

**PARAR E PERGUNTAR se:** schemas das tabelas mudarem ou validação falhar em >1%

---

### FASE 2 — Painel mestre

**Tarefas:** construir painel `municipio × ano_eleicao_presidencial` com prefeito vigente anexado. Mesma lógica do briefing anterior, mas:
- Código em `src/features/panel.py`
- Script `scripts/02_build_panel.py`
- Roda em < 1 min em dev

**Regra importante:** "prefeito vigente em ano presidencial X" = prefeito eleito na última eleição municipal antes de X. Documentar no código.

**Entregável:** `data/interim/painel_mestre.parquet` (em dev, tamanho reduzido)

---

### FASE 3 — Feature engineering

**Tarefas:**
1. `src/features/historical.py` — features de voto histórico (lag, swing, volatilidade)
2. `src/features/local_power.py` — alinhamento prefeito, margem, primeiro mandato
3. `src/features/continuity.py` — **índice de continuidade política** (a feature crítica do projeto):
   - Transições eleitorais classificadas (continuidade total/forte/parcial/ruptura)
   - `anos_consecutivos_mesmo_grupo` calculado
   - Validação obrigatória: imprimir top 20 municípios em `reports/top_continuidade_dev.md` pra revisão humana
4. `src/features/vertical.py` — alinhamento com governador, deputados federais, base federal
5. `src/features/structural.py` — região, UF, porte, log_eleitorado
6. `scripts/03_features.py` consolida tudo em `data/processed/features.parquet`

**Testes obrigatórios em `tests/test_features.py`:**
- Índice de continuidade para município fictício com transições conhecidas
- Sanity checks: sem NaN onde não deveria, ranges corretos

**Entregável:** features calculadas em dev + testes passando + `reports/top_continuidade_dev.md` pra revisão

**PARAR E PERGUNTAR se:** o top 20 de continuidade não fizer sentido (indica lógica errada)

---

### FASE 4 — Análise exploratória de hipóteses

**Tarefas:**
- `scripts/04_hypotheses.py` gera `reports/hipoteses_validacao_dev.md`
- Testa H1, H2, H3 na amostra dev (sabendo que os resultados podem não ser conclusivos com só SP)
- **Ressalva clara no relatório:** "Resultados de hipóteses foram gerados em modo dev; conclusões definitivas requerem rodar em modo prod"

**Entregável:** código das análises pronto pra rodar em prod depois

---

### FASE 5 — Modelagem

**Aqui a lógica code-only é mais crítica.** Modelos pesados NUNCA rodam em dev além de sanidade.

**Tarefas:**
1. `src/models/baseline.py` — regressão linear. Roda em qualquer máquina.
2. `src/models/lgbm.py` — LightGBM. Em dev: n_estimators reduzido (50), sem tuning. Apenas verifica que o código roda e salva modelo.
3. `src/models/bayesian.py` — PyMC. Em dev: chains=2, warmup=100, samples=100 apenas pra verificar que amostragem inicia e termina sem erro. **Os valores reais vêm do config em modo prod** (4 chains, 2000 warmup, 2000 samples).
4. `scripts/05_train.py` treina todos os modelos; salva em `models/`
5. `scripts/06_predict.py` carrega modelos, gera predições para todas as células do painel; salva em `data/processed/predictions.parquet`

**Em modo dev, o objetivo é apenas:** "todos os modelos rodam sem erro, salvam artefato, carregam de volta". Performance não importa.

**Testes obrigatórios em `tests/test_models.py`:**
- Baseline treina e prevê sem erro
- LGBM roda com n_estimators=10 sem crashar
- PyMC roda com 50 samples sem divergência
- Predições têm shape correto

**Entregável:** código completo de modelagem, passando em dev

**Observação crítica pro usuário:** quando rodar em prod, **modelo bayesiano pode demorar horas**. Colab Pro tem limite de 12h de sessão. Se exceder, precisa migrar pra cloud ou simplificar (pooling só em região, menos features).

---

### FASE 6 — Dashboard

**Tarefas:**
- `src/dashboard/app.py` com Streamlit
- Mesma especificação do briefing anterior (mapa clicável, detalhe por estado, detalhe por município)
- **Em dev, o app abre com os dados de SP apenas** — garante que visualmente funciona

**Cuidado com performance:**
- `@st.cache_data` em leitura de parquets
- Geometria simplificada (`geopandas.simplify(tolerance=0.01)`)
- Dashboard deve abrir em < 3s em dev

**Entregável:** `streamlit run src/dashboard/app.py` abre UI funcional com dados de SP

---

### FASE 7 — Handoff para máquina potente

**Tarefas:**
1. Escrever `README.md` completo com duas seções claras:
   - **"Desenvolvendo":** como clonar, setup de venv, rodar em modo dev
   - **"Rodando em produção":** instruções pro Colab (com link pro `colab_setup.ipynb`), pra servidor próprio, pra cloud. Incluir estimativas de tempo/memória.
2. Escrever `colab_setup.ipynb` completo e testar (abrir no Colab manualmente se possível)
3. Gerar `reports/handoff.md` listando:
   - Estado atual do código
   - O que o usuário precisa fazer pra rodar em prod
   - Quanto tempo cada fase prod deve levar (estimativa)
   - Sinais de que algo deu errado na execução prod
4. Commit final, push pro GitHub

**Entregável:** projeto pronto pra rodar em qualquer máquina com Python 3.11+

---

## PARTE 4 — REGRAS DE OPERAÇÃO

### Ciclo de trabalho por fase

Para cada fase:
1. Ler especificação da fase acima
2. Escrever código necessário em `src/` e `scripts/`
3. Escrever testes em `tests/`
4. Rodar `make test` — tudo passa
5. Rodar o script da fase em modo dev — funciona sem erro
6. `git add -A && git commit -m "Fase N: descrição"`
7. Gerar `reports/status_fase_N.md` com:
   - O que foi feito
   - Decisões autônomas tomadas
   - Tempo de execução em dev
   - **Estimativa de tempo em prod** (fundamental pro usuário planejar)
   - Problemas encontrados
   - Próximos passos
8. Parar e aguardar usuário confirmar antes de seguir pra próxima fase

### O que decidir sozinho

- Qualquer escolha de implementação (estrutura de funções, bibliotecas auxiliares)
- Refactor pra clareza
- Otimizações razoáveis
- Mensagens de commit

### O que perguntar

- URL do GitHub, billing project id do GCP (Fase 0)
- Mudanças de schema em dados (Fase 1)
- Se top de continuidade parecer errado (Fase 3)
- Se hipótese não se sustentar (Fase 4)
- Se modelo bayesiano crashar mesmo em dev (Fase 5)
- Qualquer ambiguidade com impacto metodológico

### O que NUNCA fazer

- Rodar em modo prod
- Processar dados de todos os estados
- Treinar modelo bayesiano com configurações de produção
- Hardcodar paths absolutos
- Commitar dados ou modelos
- Instalar dependência sem adicionar no `requirements.txt`
- Alterar `config.yaml` > `mode: prod` (isso é do usuário, em outra máquina)

---

## APÊNDICE — Checklist de portabilidade

Antes de dizer "fase concluída", verificar:

- [ ] Todos os paths usam `src/config.py`, nenhum path absoluto
- [ ] Todas as deps novas estão no `requirements.txt` com versão travada
- [ ] Seeds fixos (numpy, random, torch se usar, PyMC `random_seed`)
- [ ] `make test` passa
- [ ] Código rodou em modo dev sem erro
- [ ] Commit feito com mensagem descritiva
- [ ] Status report gerado com estimativa de tempo em prod

Se algum item falhar, a fase não está pronta.
