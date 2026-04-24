# Status — Fase 0 (Bootstrap)

Data: 2026-04-18
Modo: dev (config.yaml)

## O que foi feito

Estrutura completa do projeto criada em `Modelo2026/`:

- Layout de diretórios: `src/{ingestion,features,models,aggregation,dashboard}`, `scripts/`, `notebooks/`, `tests/`, `reports/`, `data/{raw,interim,processed}`, `models/`. Pastas vazias preservadas no git com `.gitkeep`.
- `config.yaml` central com `mode: dev`, `seed: 42`, paths relativos, amostra dev (UF=SP, anos 2014/2018/2022, max 100 municípios), config prod (todas UFs, 7 eleições), hiperparâmetros LightGBM e PyMC.
- `src/config.py` como ponto único de acesso: expõe `MODE`, `SEED`, `MODE_CFG`, `PATHS`, `GCP_BILLING_PROJECT_ID`, `set_global_seed()`, `require_billing_project()`, `summary()`. Todos os paths derivam de `Path(__file__).resolve().parent.parent` — nenhum path absoluto no código.
- `requirements.txt` com versões-alvo (serão travadas com `pip freeze` depois do `make install` na máquina do usuário). Pacotes: basedosdados, pandas, polars, pyarrow, numpy, scipy, scikit-learn, lightgbm, optuna, shap, pymc, arviz, pytensor, geopandas, geobr, folium, shapely, streamlit, plotly, pyyaml, pytest, ruff, jupyter, ipykernel.
- `Makefile` com alvos `install`, `dev`, `test`, `lint`, `freeze`, `clean`, `help`.
- `run_prod.sh` para a máquina pesada: força `mode: prod` no YAML, valida `billing_project_id`, roda scripts 01→06 em sequência.
- `colab_setup.ipynb` com receita completa para Colab Pro: montar Drive, clonar repo privado com PAT, instalar deps, autenticar GCP, symlinks de `data/` e `models/` para o Drive, trocar para prod e rodar.
- `README.md` com seções "Desenvolvendo" e "Rodando em produção" (Colab / servidor próprio / VM).
- `.gitignore` protegendo `/data/raw/`, `/data/interim/`, `/data/processed/`, `/models/*` (exceto `.gitkeep`), `.venv/`, `__pycache__/`, `.ipynb_checkpoints/`, `.env`, service accounts, artefatos de relatório.
- `tests/test_config.py` como smoke test: import de `src.config`, modo == dev, chaves de `PATHS` e `MODE_CFG`, `set_global_seed()`.
- `bootstrap_git.sh` para o usuário inicializar git local + opcionalmente apontar para o repo GitHub.

## Critério de aceite da Fase 0

`python -c "from src.config import MODE; print(MODE)"` → `dev` **OK**.

Os 4 testes do smoke test passam (rodados diretamente porque o sandbox não tem pytest instalável via proxy).

## Decisões autônomas

- **`/models/*` com `!/models/.gitkeep`** em vez de só `models/` no `.gitignore`. A regra original do briefing (`models/`) também teria excluído o pacote Python `src/models/`. A nova regra é estrita e intencional.
- **`set_global_seed()` helper** em `src.config`: cada script chama para fixar `random` e `numpy.random` globalmente. Seeds específicos de LightGBM e PyMC continuam passados por parâmetro.
- **`require_billing_project()` helper**: scripts de ingestão chamam e falham cedo com mensagem clara se o YAML estiver sem `gcp.billing_project_id`.
- **`bootstrap_git.sh`**: adicionado para contornar uma limitação do sandbox Cowork — o mount não deixa popular `.git/objects`. O script também detecta e limpa uma `.git` herdada em estado quebrado.
- **Adição de `ruff` ao requirements e target `make lint`**: não estava no briefing, mas o briefing menciona lint no Makefile. Baixo custo.

## Tempo de execução em dev

A Fase 0 é pura construção de arquivos; não há pipeline para cronometrar ainda. Em qualquer máquina o import de `src.config` é instantâneo.

## Estimativa de tempo em prod

N/A para Fase 0 (não há processamento). Para referência futura: as outras fases vão consumir a maior parte do tempo em (1) ingestão BigQuery, (2) treino PyMC.

## Problemas encontrados

- **Sandbox Cowork não consegue criar `.git/objects`** no caminho `/mnt/Modelo2026`. O `git init` do subprocess falhou com `Operation not permitted` em arquivos internos do `.git`. Verificação completa do commit foi feita em diretório fora do mount (`/sessions/gifted-optimistic-johnson/git_verify/`) e funcionou 100% — 23 arquivos rastreados, `.gitignore` validado bloqueando `data/*` e `models/*.pkl`, apenas `models/.gitkeep` preservado. **Resolução:** você roda `bash bootstrap_git.sh <url-do-repo>` na sua máquina.
- **`pytest` não pode ser instalado no sandbox** (proxy bloqueia PyPI para alguns pacotes). Os 4 testes do smoke test foram executados manualmente (import + chamadas de função) e passaram. Na sua máquina, `make test` vai rodar com o pytest real.
- **`src/models/__init__.py` estava sendo engolido pela regra `models/`** do `.gitignore` original — corrigido para `/models/*`.

## Credenciais recebidas

- **GitHub remote:** `https://github.com/chufhi/eleicao2026.git`
- **GCP billing project:** `modelo-eleitoral-2026` (já gravado em `config.yaml`)

Dry-run do `bootstrap_git.sh` com esse remote rodou limpo em dir externo. Quando você rodar na sua máquina:

```bash
cd Modelo2026
bash bootstrap_git.sh https://github.com/chufhi/eleicao2026.git
git push -u origin main
```

## Próximos passos

Fase 0 concluída. Próxima: **Fase 1 — Ingestão** (Base dos Dados + geometrias + validação de totais).

## Árvore resumida

```
Modelo2026/
├── .gitignore
├── Makefile
├── README.md
├── bootstrap_git.sh
├── briefing_cowork_codeonly.md
├── colab_setup.ipynb
├── config.yaml
├── requirements.txt
├── run_prod.sh
├── data/           (raw/interim/processed ignorados; .gitkeep)
├── models/         (tudo ignorado exceto .gitkeep)
├── notebooks/      (.gitkeep)
├── reports/        (status_fase_0.md; artefatos ignorados)
├── scripts/        (.gitkeep; entrypoints chegam na Fase 1)
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── aggregation/__init__.py
│   ├── dashboard/__init__.py
│   ├── features/__init__.py
│   ├── ingestion/__init__.py
│   └── models/__init__.py
└── tests/
    ├── __init__.py
    └── test_config.py
```
