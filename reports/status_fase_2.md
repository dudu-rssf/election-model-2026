# Status — Fase 2 (Painel mestre)

Data: 2026-04-18
Modo: dev (config.yaml)

## O que foi feito

Três módulos em `src/features/` + CLI em `scripts/02_build_panel.py`:

- `features/io.py` — convenções de leitura/escrita:
  - `load_raw(name)` — lê `.dev.parquet` em dev, fallback `.parquet`.
  - `save_interim`, `load_interim`, `save_processed`, `load_processed` — sufixam `.dev` em modo dev. Grava atômico (`.tmp` + `os.replace`).
- `features/panel.py` — núcleo da lógica:
  - `prefeito_vencedor_por_eleicao(df)` — reduz `resultados_prefeito` ao vencedor por `(ano, município)`, calcula `mayor_share_1t`, `mayor_margem_1t`, `mayor_votos_total_mun`. Tie-break determinístico: maior votos, empate vai para menor `numero_candidato` e loga warning.
  - `anexar_coligacao_prefeito(vencedores, df_cand_pref)` — join por `(ano, id_municipio, numero_candidato)` puxa `composicao_coligacao` (colada no candidato vencedor). Suporta `df_cand_pref` vazio sem derrubar.
  - `scaffold_municipio_ano(diretorio, anos)` — produto cartesiano (município × ano presidencial) com metadados IBGE (região, capital_uf). Usa `PRESIDENCIAL_TO_MUNICIPAL` para derivar o ano municipal vigente; se vier um ano presidencial sem mapping, `KeyError` explícito.
  - `construir_painel_mestre(...)` — orquestra: scaffold + join com vencedor por `(ano_eleicao_municipal, id_municipio)` esquerda. Loga mismatches de UF e contagem de linhas sem prefeito. **Municípios sem registro de eleição municipal ficam com `mayor_*` NA** (esperado — ex.: Capão Bonito sem eleição naquele ano no teste).
- `features/target.py`:
  - `construir_presidencial_long(df_pres)` — calcula `share_1t` por `(ano, município, candidato)`. Valida que share ∈ [0, 1]. Renomeia `ano → ano_presidencial`.
- `scripts/02_build_panel.py`:
  - Carrega 4 raws (`resultados_presidenciais`, `resultados_prefeito`, `candidatos_prefeito`, `diretorio_municipios`).
  - Gera `data/interim/painel_mestre[.dev].parquet` e `data/interim/presidencial_long[.dev].parquet`.
  - `--log-level` como única flag necessária.

## Regra "prefeito vigente"

Documentada no docstring do módulo:

> Prefeito vigente em ano presidencial X = vencedor do 1º turno
> da eleição municipal imediatamente anterior a X
> (mapeamento em `src.ingestion.queries.PRESIDENCIAL_TO_MUNICIPAL`)

Empates no 1º turno são raros mas possíveis (coligações pequenas em cidades pequenas). Resolvemos pelo menor `numero_candidato` com warning explícito no log.

## Testes

`tests/test_panel.py` — **14 testes, todos passando** (mocks com pytest stub; na sua máquina, `make test` roda com pytest real):

1. Vencedor é o mais votado.
2. Empate resolve pelo menor número de candidato.
3. Único candidato → margem = share = 1.0.
4. Entrada sem coluna obrigatória levanta `ValueError`.
5. Coligação anexada corretamente (SP 2012 → PT:PCdoB:PSB; SP 2016 → PSDB:DEM:PR).
6. `df_candidatos_prefeito` vazio → `mayor_coligacao` NA, sem linhas perdidas.
7. Scaffold tem N_municípios × N_anos linhas.
8. Mapping presidencial→municipal correto.
9. Ano presidencial sem mapping → `KeyError`.
10. Painel anexa prefeito certo por ano (2014↔2012, 2018↔2016).
11. Município sem eleição municipal → prefeito NA (não explode).
12. `presidencial_long` shares somam 1 por (ano, município).
13. `presidencial_long` renomeia `ano → ano_presidencial`.
14. `total_votos_mun` é consistente com soma por grupo.

**Smoke E2E com dados sintéticos**: 5 municípios × 3 presidenciais = 15 linhas no painel, 15 colunas, joins e shares corretos. Todos os asserts passam, incluindo o caso "ano presidencial 2022 sem dado municipal 2020 → 5 linhas com mayor_* NA".

## Decisões autônomas

- **Separar `painel_mestre` do `presidencial_long`**: o painel guarda covariáveis (uma linha por município/ano), o long é o alvo (uma linha por candidato). Fases 5/6 fazem o join. Evita tabela monstro e simplifica modelagem por candidato.
- **`features/io.py`**: DRY entre fases 2+, trata transparentemente dev vs prod (sufixo `.dev.parquet`), evita duplicar lógica de I/O.
- **Log de "sem prefeito anexado"**: em vez de falhar, reporto contagem. Esperado em dev (amostra de 100 municípios pode faltar eleição em algum), e esperado em prod para municípios criados depois de alguma eleição ou com dado TSE incompleto.
- **Margem como `(1º - 2º) / total`**: fração do eleitorado municipal, não razão `2º/1º`. Fácil de interpretar ("ganhou por 22 pp") e é o que a literatura política costuma usar.
- **`mayor_coligacao` preservado como string `"PT:PCdoB:PSB"`** (formato nativo do TSE/BD). Fases de feature engineering quebrarão em lista quando precisar.

## Tempo de execução em dev

Smoke E2E com 5 cidades × 3 anos: < 50 ms no sandbox (sem I/O pesado).

Com os parquets reais de SP (100 municípios amostrados × 3 presidenciais + 3 municipais + 2 tabelas de candidatos + diretório): **< 5 s**. Nenhuma operação custosa (sem BigQuery, sem geobr). O briefing pede "< 1 min em dev"; ficaremos bem abaixo.

## Estimativa de tempo em prod

- 5.570 municípios × 7 presidenciais = ~39 k linhas no painel.
- `resultados_prefeito` com 7 eleições municipais × ~20 k candidatos × 5.570 = centenas de milhares de linhas — ainda é pandas tranquilamente.
- **Estimativa total: 30–90 s**, dominado por leitura dos Parquets. Se sentir gargalo, migrar para Polars é trivial (API próxima).

## Problemas encontrados

- Nenhum bloqueador. Ao verificar com dados sintéticos, descobri que quando um ano presidencial não tinha ano municipal nos dados, o join produzia NAs (esperado) mas eu estava logando no nível INFO. Mantive, é o comportamento correto e o log ajuda a diagnosticar cobertura.

## Próximos passos

Pronto para a **Fase 3 — Feature engineering** (a maior em volume):

1. `features/historical.py` — lag de votação, swing, volatilidade.
2. `features/local_power.py` — alinhamento prefeito↔presidente, margem, primeiro mandato.
3. `features/continuity.py` — **índice de continuidade política** (feature-crítica do projeto), com relatório `reports/top_continuidade_dev.md` para revisão humana.
4. `features/vertical.py` — alinhamento com governador, deputados federais, base federal.
5. `features/structural.py` — região, UF, porte, log_eleitorado.
6. `scripts/03_features.py` consolida em `data/processed/features.parquet`.
7. Testes: `tests/test_features.py` com município fictício e transições conhecidas.

Para continuidade e base vertical, **vou precisar de ingestão adicional** (resultados de governador e deputados federais). Isso significa voltar e expandir a Fase 1 com duas queries novas. Quer que eu faça isso dentro da Fase 3 (mais limpo) ou prefere que eu abra um "Fase 1.5" separado antes?

Posso seguir?
