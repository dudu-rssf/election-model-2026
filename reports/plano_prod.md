# Plano de rodada prod — Fase 4 (presidencial + prefeito)

Após Fase 4 completa em dev (1 UF × 100 municípios × 2-3 anos), o próximo passo natural é rodar o pipeline em **prod** (todas as UFs × todos os municípios × todos os anos disponíveis). Este documento é o checklist para essa rodada — pré-requisitos, ordem de execução, custos esperados, critérios de sucesso, e armadilhas conhecidas.

## Por que rodar prod agora

A Fase 4 deixou três descobertas que **estruturalmente** dependem de mais anos pra fechar:

1. **PL 2022 MAE = 0.51** (modelo prevê 0.02 quando real é 0.55). Causa: em dev o LGBM só tem 2 anos de treino (2014, 2018) onde `lag_share_1t_sucessao` nunca varia significativamente (PL pré-Bolsonaro era partido pequeno, sucessão não disparava). Em prod com 6 anos de treino, PSL 2018 entra com share ~0.55, PL 2022 com `lag_sucessao` ~0.55 — a relação fica visível ao LGBM.

2. **Decis intermediários sub-cobertos pelo conformal Mondrian**. Causa: o conjunto de calibração (holdout 2018) não contém PL com perfil pós-migração; o q̂ do bin específico onde PL 2022 cai está calibrado num regime sem esse tipo de erro. Mais anos = mais regimes históricos no calib.

3. **UNIÃO 2022 → DEM sucessão silenciosa** continuará em prod (DEM não teve candidato presidencial em nenhum ano). Mitigação só vem com #60 (pesquisas como feature). Em prod, o WARNING adicionado em `aplicar_sucessao` deve disparar — validar que aparece é parte do critério de sucesso.

## Pré-requisitos

- [ ] `config.yaml`: trocar `mode: dev` para `mode: prod` (linha 1).
- [ ] `config.yaml > gcp.billing_project_id` preenchido (já está: `modelo-eleitoral-2026`).
- [ ] Credenciais GCP ativas: `gcloud auth application-default login` ou `GOOGLE_APPLICATION_CREDENTIALS` apontando pra service account com permissão de leitura em `basedosdados.br_tse_eleicoes.*`.
- [ ] Espaço em disco: estimar **~2-5 GB** para `data/raw/` (parquet × 8 anos × ~5 tabelas × todas as UFs). O `data/processed/` adiciona ~500 MB.
- [ ] Tempo de wallclock estimado: **2-4 horas** para o pipeline completo (a maior parte é ingestão + agregação geo).
- [ ] (opcional) Testar primeiro em prod-light: editar `config.yaml > prod > ufs: ["SP", "RJ", "MG"]` para uma rodada parcial de validação antes da nacional.

## Ordem de execução (presidencial)

```bash
# 1. Ingestão (BigQuery → data/raw/*.parquet)
python scripts/01_ingest.py
# Validação automática em reports/ingestao_validacao_prod.md.
# Re-rodar com --force se schema mudou em algum ano.

# 2. Painel longo (data/raw → data/interim/presidencial_long.parquet)
python scripts/02_build_panel.py

# 3. Features (data/interim → data/processed/features.parquet)
python scripts/03_features.py
# Atenção: este script chama aplicar_sucessao. O log deve mostrar
# WARNING de UNIÃO→DEM nas linhas presidenciais 2022. Se não aparecer,
# investigar (provável: features_historical não está chamando
# _lag_por_sigla_canonica com sucessoes do config).

# 4. Treino + relatório
python scripts/04_train.py --calibrate --conformal --conformal-mondrian
# Saídas:
#   models/lgbm_v1.pkl              modelo + calibrador + conformal
#   data/processed/preds.parquet    pred + intervalos
#   reports/status_fase_4.md        relatório completo
```

## Ordem de execução (prefeito)

```bash
# Os scripts 02 e 03 já são parametrizados pelo eixo. Para prefeito:
python scripts/03_features_prefeito.py
python scripts/04_train_prefeito.py --calibrate --conformal --conformal-mondrian
# Saídas em models/lgbm_prefeito_v1.pkl, preds_prefeito.parquet,
# reports/status_fase_4_5.md.
```

## Custos BigQuery estimados

Base dos Dados é **gratuita** para queries até 1 TB/mês via Google Public Datasets. A ingestão prod consome:

| Tabela | Anos | Linhas estimadas | TB scaneado (aprox) |
| --- | --- | --- | --- |
| resultados_candidato_municipio (presidencial) | 7 | ~150 k | < 0.01 |
| resultados_candidato_municipio (prefeito) | 8 | ~200 k | < 0.01 |
| candidatos | 7 + 8 | ~50 k | < 0.01 |
| partidos | 7 | ~7 k | < 0.001 |
| Geometria municípios (geobr, fora do BQ) | — | 5570 | (cache local) |

Total esperado: **< 0.05 TB/mês** — folgadamente dentro do free tier.

## Critérios de sucesso

Comparar com dev (status_fase_4.md atual):

| Métrica | Dev (atual) | Prod (esperado) |
| --- | --- | --- |
| `LightGBM_v1` MAE geral | 0.0558 | 0.04 - 0.06 (similar — mais anos compensa mais ruído) |
| PL 2022 MAE | 0.5096 | **< 0.20** (LGBM aprende lag_sucessao) |
| Decil 7 cobertura Mondrian | 0.836 | **> 0.85** |
| Decis 7-8 cobertura split | 0.64 / 0.82 | > 0.80 / > 0.85 |
| WARNING UNIÃO→DEM no log | n/a | **deve aparecer** (validação) |
| `lag_share_1t_sucessao` em feature_importance | 874 (10º lugar) | top 5 (sobe) |

Se PL 2022 MAE não cair pra < 0.30, há um bug estrutural além de "não tem anos suficiente" — investigar:
- O LGBM está usando `lag_share_1t_sucessao` ou só `lag_share_1t`? Conferir feature_importance.
- A coluna `lag_share_1t_sucessao` está populada para PL 2022 em prod? `WARNING` de aplicar_sucessao não deve aparecer pra `PL → PSL`.

## Armadilhas conhecidas

1. **Schema variável em anos antigos**: `partidos_*` 1998-2002 pode não ter os mesmos campos de 2014-2022. A Fase 1.5/3.5 já documentou problemas com `composicao_coligacao` (cobertura zerada em anos antigos no BD). Possível: feature `alinhado_gov_vigente_coligacao` será NaN em ~50% das linhas treino prod. Não é bloqueio (LGBM lida com NaN), mas reduz o sinal dessa feature.

2. **`share_prefeito_local` em 1998-2002**: depende de eleição municipal anterior (1996, 2000). Eleições 1996 podem não estar disponíveis no BD (verificar `peek_partidos_pref_2020.py` style — fazer um peek 1996 antes de rodar). Se faltar, ano 1998 vai com `share_prefeito_local = NaN`.

3. **Geometria geobr**: o cache fica em `~/.geobr/`. Se vazio, a primeira rodada baixa ~200 MB e demora. Em máquina nova, fazer `python -c "import geobr; geobr.read_municipality(year=2020)"` antes pra pré-aquecer.

4. **`local_power.py` gap temporal**: para presidencial 2022, busca prefeito 2020 (gap 2 anos, ok). Para presidencial 1998, buscaria prefeito 1996 (gap 2 anos, ok), mas 1996 pode não existir → NaN. Aceito.

5. **Memória do LGBM**: prod tem ~50-100 k linhas treino (vs ~2 k em dev). Com 500 estimators × num_leaves=63, deve caber em < 4 GB RAM. Se estourar, reduzir num_leaves para 31.

6. **Tempo de `_lag_por_sigla_canonica`**: vetorização atual é O(n_partidos × n_anos × n_municipios). Em prod com 5570 municípios × 7 anos × ~30 partidos pode passar de 10s — não bloqueia, mas vale cronometrar.

## Pós-rodada

1. Comparar `reports/status_fase_4.md` prod com versão dev — copiar a tabela "MAE por partido" lado-a-lado pra evidenciar onde melhorou.
2. Se PL 2022 estiver bem calibrado, **encerrar Fase 4 oficialmente** e abrir Fase 5 (escopo: agregação UF→nacional, conformal estratificado por região, ou início de #60 pesquisas).
3. Snapshot do `models/lgbm_v1.pkl` prod com tag git, pra reprodutibilidade.

## Decisão pendente: qual a Fase 5?

Três caminhos plausíveis (escolha após validar prod):

- **5a. Agregação UF → nacional**: o modelo prevê share municipal; pra previsão presidencial precisa agregar com peso (eleitorado, comparecimento esperado). Trabalho moderado, value alto.
- **5b. Conformal estratificado por (sigla, região)**: melhora cobertura condicional. Trabalho leve, value médio.
- **5c. #60 pesquisas como feature**: encara o débito técnico da identidade partidária. Trabalho pesado, value alto mas longo.

A escolha depende do quão bem prod fechar PL 2022. Se fechar (esperado), 5a faz mais sentido. Se não fechar, 5c.
