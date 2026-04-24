# Status — Fase 1 (Ingestão)

Data: 2026-04-18
Modo: dev (config.yaml)

## O que foi feito

Módulo `src/ingestion/` e o CLI `scripts/01_ingest.py`:

- `queries.py` — templates SQL parametrizados por `MODE_CFG`:
  - `resultados_presidenciais_sql()` — 1º turno, cargo = presidente, filtros UF+ano.
  - `resultados_prefeito_sql()` — 1º turno, cargo = prefeito, anos **municipais** (derivados de `PRESIDENCIAL_TO_MUNICIPAL`: 2014→2012, 2018→2016, 2022→2020, 2026→2024).
  - `candidatos_presidenciais_sql()` e `candidatos_prefeito_sql()` — trazem `composicao_coligacao` para H3.
  - `diretorio_municipios_sql()` — IBGE (id, UF, nome, região, capital).
  - `QUERIES` dict registra tudo para iteração em `01_ingest.py`.
- `bd_client.py` — `download(name, sql, force=False, backend=None)` com cache Parquet (`data/raw/<name>.parquet`), import lazy de `basedosdados`, backend injetável. Grava atomicamente (`.tmp` + `os.replace`).
- `geo.py` — `download_geometrias()` via `geobr`, cacheia `data/raw/geometrias_municipios.parquet`, backend injetável, falha graciosamente para `01_ingest.py` continuar se `geobr` tropeçar na rede.
- `sample.py` — `choose_ids()` e `apply_dev_sampling()` com `np.random.default_rng(SEED)`. Em dev, 100 municípios; filtro aplicado **após** o download em todas as tabelas com `id_municipio` (consistência de joins).
- `validate.py` — `ValidationReport` com erros e warnings, checagens por tabela (colunas obrigatórias, nulls, votos ≥ 0, `id_municipio` de 7 dígitos, totais vs. oficial com tolerância 0,1%). Gera markdown para `reports/ingestao_validacao_<mode>.md`.
- `scripts/01_ingest.py` — CLI com flags `--force`, `--skip-geo`, `--only`, `--log-level`. Salva `data/raw/<nome>.parquet` (bruto intocado para poder virar prod depois) e `data/raw/<nome>.dev.parquet` (com amostragem).

## Testes

`tests/test_ingestion.py` — **17 testes, todos passando** (rodados manualmente no sandbox com pickle no lugar de parquet, porque a proxy do Cowork bloqueia `pip install pyarrow`):

1. SQL dev contém `'SP'`, presidenciais 2014/2018/2022 e cargo correto.
2. `anos_municipais_para_panel()` mapeia corretamente.
3. Prefeito usa **anos municipais**, não presidenciais.
4. `QUERIES` tem exatamente as 5 chaves esperadas.
5-7. `bd_client.download` honra cache, re-baixa com `force`, levanta em 0 linhas.
8-10. `choose_ids` é reprodutível, passa through quando `max_n > universo`, seeds diferentes dão resultados diferentes.
11. `apply_dev_sampling` filtra as tabelas pelo MESMO conjunto de IDs.
12-15. `validate` detecta colunas ausentes, votos negativos, ID não-IBGE, tolerância de totais.
16. `ValidationReport.to_markdown` renderiza erros e warnings.
17. `scripts/01_ingest.py` importa e expõe `main`/`rodar_queries`/`validar`.

`python scripts/01_ingest.py --help` imprime a ajuda (argparse OK).

## Decisões autônomas

- **Backend injetável** em `bd_client` e `geo` (protocol `SQLBackend`/`GeoBackend`). Testes nunca tocam BigQuery/geobr. Zero dependência de auth GCP para rodar o suite.
- **Import lazy** de `basedosdados` e `geobr` — módulos só carregam quando `download()` é chamado de verdade.
- **Mapeamento presidencial→municipal** em `queries.PRESIDENCIAL_TO_MUNICIPAL`. Preciso na Fase 2 para montar o painel. Documentado no docstring.
- **Amostragem pós-download** conforme briefing, com universo = união de `id_municipio` nas tabelas. Garante que as 100 cidades escolhidas existem em prefeito, presidencial e diretório.
- **Dois Parquets por tabela em dev**: `<nome>.parquet` (bruto) + `<nome>.dev.parquet` (amostrado). Fase 2+ lê o `.dev.parquet` em dev e o `.parquet` em prod. Evita reingestão se você quiser trocar `max_municipios`.
- **Validação contra totais oficiais** é opcional (`oficial_por_uf=None` vira warning). Adicionaremos uma fonte oficial (dadosabertos.tse.jus.br) em fase de refinamento se necessário.
- **Tabela `br_tse_eleicoes.candidatos`** usada para coligação. Se o schema BD mudar nome de coluna (`composicao_coligacao` → outro), `validate` vai acusar "coluna faltando" e paramos.

## Tempo de execução em dev

Código puro, sem BigQuery: < 1 s.
Com BigQuery real (SP, 3 presidenciais + 3 municipais + 2 candidatos + diretório + geobr SP):
- Estimativa: 2–5 min, dominada por latência do BigQuery e do `geobr`.

## Estimativa de tempo em prod

- 5 queries BigQuery sobre `br_tse_eleicoes.resultados_candidato_municipio` (tabela de dezenas de milhões de linhas) + `candidatos` + diretório: **10–30 min**, dependendo de slot do BigQuery e do plano de billing.
- `geobr.read_municipality(code_muni="all", year=2022)`: **3–10 min** para Brasil inteiro + eventual download de shapefiles do IBGE.
- Cache Parquet local faz runs subsequentes < 30 s.

## Problemas encontrados

- **`pip install pyarrow` bloqueado** pela proxy do sandbox Cowork. Validei `bd_client` com `pickle` no lugar de parquet. Na sua máquina, `make install` resolve isso.
- **Schema das tabelas da BD não verificado ao vivo**. Estou seguindo nomenclatura atual de `br_tse_eleicoes.resultados_candidato_municipio` (`ano`, `sigla_uf`, `id_municipio`, `cargo`, `numero_candidato`, `turno`, `votos`, etc.). Se algum nome mudou, `validate` já acusa e paramos — é o que o briefing recomenda.

## Próximos passos

Pronto para a **Fase 2 — Painel mestre**. Ela vai:

- `src/features/panel.py`: cruza `resultados_presidenciais` × `resultados_prefeito` usando `PRESIDENCIAL_TO_MUNICIPAL` pra colar o prefeito vigente em cada ano presidencial.
- `scripts/02_build_panel.py`: salva `data/interim/painel_mestre.parquet` (em dev, versão amostrada).
- Regra-chave: "prefeito vigente em ano presidencial X = vencedor (cargo=prefeito, turno=1) da eleição municipal imediatamente anterior", documentada.
- Alvo: < 1 min em dev, ~2 min em prod.

Posso seguir?
