# Status — Fase 1.5 (Expansão da ingestão: governador + deputado federal)

Data: 2026-04-22
Modo: dev (config.yaml)

## Motivação

Fase 3 precisa de features verticais (alinhamento com governador e base
federal) — isso exige baixar dados que Fase 1 não cobriu. Separei essa
expansão como Fase 1.5 ao invés de enfiar dentro da Fase 3: mantém o
script `01_ingest.py` como porta única de entrada de dados brutos e evita
misturar I/O com feature engineering.

## O que foi feito

Quatro queries novas em `src/ingestion/queries.py`:

- `resultados_governador_sql()` — 1º turno, por candidato × município.
- `candidatos_governador_sql()` — ficha + `composicao_coligacao` estadual.
- `resultados_deputado_federal_sql()` — 1º turno, por candidato × município (a tabela do BD já tem o breakdown municipal mesmo o cargo sendo estadual).
- `candidatos_deputado_federal_sql()` — ficha + `composicao_coligacao` federal.

Tabelas alvo na Base dos Dados: as mesmas que já usávamos
(`basedosdados.br_tse_eleicoes.resultados_candidato_municipio` e
`basedosdados.br_tse_eleicoes.candidatos`), só muda o filtro `cargo`.

Cargo no TSE (confirmado nos dicionários públicos de metadados do BD):
`'governador'` e `'deputado federal'` (exatamente assim, minúsculo, com espaço).

Novo mapeamento + helper:

```python
PRESIDENCIAL_TO_ESTADUAL_ANTERIOR = {
    1998: 1994, 2002: 1998, 2006: 2002, 2010: 2006,
    2014: 2010, 2018: 2014, 2022: 2018, 2026: 2022,
}

def anos_estaduais_para_panel(anos_presidenciais):
    """Retorna { X, X-4 } para cada X presidencial."""
```

Motivo: eleições gerais (presidencial + governador + dep.fed.) acontecem
no mesmo ano. Queremos os dois recortes disponíveis para a Fase 3:

- **Governador vigente** em X = eleito em X-4 (quem está no cargo durante a campanha presidencial).
- **Governador concorrente** em X = eleito em X (alinhamento do resultado estadual simultâneo).

A Fase 3 escolhe qual usar (ou usa ambos). Em dev
(`anos_presidencial = [2014, 2018, 2022]`), o set de anos estaduais
baixado é `{2010, 2014, 2018, 2022}`.

Registry atualizado em `queries.QUERIES`:

```
resultados_presidenciais
resultados_prefeito
resultados_governador             ← novo
resultados_deputado_federal       ← novo
candidatos_presidenciais
candidatos_prefeito
candidatos_governador             ← novo
candidatos_deputado_federal       ← novo
diretorio_municipios
```

Validação em `src/ingestion/validate.py`:

- `validate_resultados_governador` — mesmas checagens que prefeito
  (schema, no-nulls em chaves, votos ≥ 0, id_municipio IBGE 7 dígitos).
- `validate_resultados_deputado_federal` — idem.
- Ficha de candidatos reusa `validate_candidatos` (já genérico).

`scripts/01_ingest.py` não precisou mudar o loop (já itera
`queries.QUERIES`), só a função `validar()`: agora também chama as novas
validações quando as tabelas estão presentes.

## Testes

Novos testes em `tests/test_ingestion.py`:

- `test_registry_completo` atualizado (9 tabelas esperadas).
- `test_anos_estaduais_inclui_concorrente_e_anterior` — union correta.
- `test_resultados_governador_sql_contem_cargo_e_anos_corretos`
- `test_candidatos_governador_sql_tem_composicao_coligacao` (e não filtra turno, pois tabela de candidatos não tem).
- `test_resultados_deputado_federal_sql_usa_cargo_correto`
- `test_candidatos_deputado_federal_sql_ok`
- `test_validate_resultados_governador_aceita_df_valido`
- `test_validate_resultados_deputado_federal_detecta_id_invalido`

Todos os testes rodaram no sandbox (sem pytest, via asserts diretos) e
passaram. Na sua máquina, `make test` roda a suite completa (Fase 1 +
Fase 1.5 + Fase 2).

## Decisões autônomas

- **Separar como Fase 1.5 e não como parte da Fase 3**: `01_ingest.py`
  continua sendo a única porta de entrada de dados brutos. Misturar
  ingestão com feature engineering quebra a separação que o briefing pede.
- **Baixar governador dos dois anos (X e X-4)**: custo marginal irrisório
  (são ~27 UFs × 2 anos × ~10 candidatos por eleição) e dá liberdade
  total pra Fase 3 escolher o recorte temporal.
- **Deputado federal baixado por município**: a tabela do BD já expõe os
  votos municipais, não precisei reescrever. Consequência: para SP com
  100 municípios amostrados, ainda ficamos com ~20 dep. federais × 100
  mun × 4 anos = ~8k linhas. Tranquilo.
- **Cargo como string minúscula exata**: `'governador'`, `'deputado federal'`.
  Se o schema do BD mudar, a validação do total vs oficial na Fase 1
  original falharia — e agora as queries retornariam 0 linhas, que
  `bd_client.download` transforma em `RuntimeError` explícito.

## Tempo de execução em dev

Nada foi efetivamente baixado aqui (isto é Fase 1.5 em termos de código
apenas). Quando você rodar `python scripts/01_ingest.py` na sua máquina
com credenciais GCP, a estimativa é:

- Governador SP 4 anos × ~10 candidatos × ~645 municípios originais = ~25k linhas, +/- 3 s.
- Deputado federal SP 4 anos × ~200 candidatos × ~645 municípios = ~500k linhas, +/- 20-40 s.
- **Dev total (com as 4 queries novas):** +30-60 s sobre o baseline de Fase 1. Continuamos bem abaixo do orçamento de 5 min.

Depois do download, `apply_dev_sampling` reduz tudo para os mesmos 100
municípios escolhidos na Fase 1 — as novas tabelas são filtradas pelo
mesmo set de IDs e os joins continuam consistentes.

## Estimativa de tempo em prod

- Governador Brasil 27 UFs × 7 anos × ~10 candidatos × 5570 mun = ~10 M linhas. Com BigQuery, 10-30 s de execução de query + transferência ~5 min (depende do perfil da conexão Colab).
- Deputado federal Brasil 27 UFs × 7 anos × ~200 candidatos × 5570 mun = ~200 M linhas (!!). **Aqui precisamos atenção:**
  - A query atual baixa o breakdown completo município × candidato. Para prod, talvez valha agregar em BigQuery antes do download (soma por `(ano, uf, id_municipio, sigla_partido)`) para reduzir volume em 20x.
  - Alternativa mais conservadora: deixar a query como está e aceitar 10-15 min de download. Simples e testado.
- **Recomendação:** manter a query como está para prod. Se estourar memória no Colab Pro (dispositivo com 12-25 GB RAM), migrar para Polars + leitura streaming. Código já está com tipos explícitos, migração é 1 h de trabalho.

## Problemas encontrados

Nenhum. A arquitetura da Fase 1 foi fácil de estender: tudo que precisei
foi adicionar entradas no dicionário `QUERIES` e dois validadores novos.
A ausência de coluna `turno` na tabela `candidatos` (contrastada com
`resultados_candidato_municipio` que tem `turno`) é consistente com o
que já fazíamos para prefeito, então replicar o padrão foi direto.

## Próximos passos

Com os dados necessários agora mapeados, estou pronto para **Fase 3 —
Feature engineering**. Plano operacional:

1. `src/features/historical.py` — lag de votação presidencial, swing, volatilidade.
2. `src/features/local_power.py` — alinhamento prefeito↔presidente, margem, primeiro mandato.
3. `src/features/continuity.py` — **índice de continuidade política** + relatório `reports/top_continuidade_dev.md` para revisão humana (briefing pede parada nesta etapa se o top 20 não fizer sentido).
4. `src/features/vertical.py` — alinhamento com governador (vigente + concorrente) e força local da coligação federal via deputados federais.
5. `src/features/structural.py` — região, UF, porte, log_eleitorado.
6. `scripts/03_features.py` consolida tudo em `data/processed/features.parquet`.
7. `tests/test_features.py` com município fictício e transições conhecidas.

Começo com `historical.py` e `local_power.py` (os mais simples — puro
pandas sobre o painel já construído) e vou subindo em complexidade.

Posso seguir para a Fase 3?
