# Fase 4 — status: primeiro modelo presidencial

**Modo:** dev | **UFs:** ['SP'] | **Máx municípios:** 100
**Split temporal:** treino = [2014, 2018] (2330 linhas) | teste = 2022 (1100 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 1100 | 0.0751 | 0.1715 | 0.0587 |
| B1_lag_share | 1100 | 0.0782 | 0.1757 | 0.0643 |
| B2_blend | 1100 | 0.0764 | 0.1730 | 0.0615 |
| LightGBM_v1 | 1100 | 0.0558 | 0.1575 | 0.0505 |
| LightGBM_v1_iso | 1100 | 0.0568 | 0.1589 | 0.0514 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 100 | 0.5096 | 0.5166 | 0.5096 | 0.5288 |
| PT | 100 | 0.0533 | 0.0678 | 0.0360 | 0.3928 |
| MDB | 100 | 0.0288 | 0.0310 | 0.0288 | 0.0473 |
| PDT | 100 | 0.0113 | 0.0139 | -0.0096 | 0.0219 |
| NOVO | 100 | 0.0091 | 0.0102 | -0.0091 | 0.0034 |
| UNIÃO | 100 | 0.0008 | 0.0010 | 0.0007 | 0.0043 |
| PTB | 100 | 0.0006 | 0.0007 | -0.0005 | 0.0006 |
| UP | 100 | 0.0003 | 0.0003 | -0.0003 | 0.0003 |
| DC | 100 | 0.0002 | 0.0002 | -0.0002 | 0.0001 |
| PSTU | 100 | 0.0002 | 0.0002 | -0.0001 | 0.0002 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| SP | 1100 | 0.0558 | 0.1575 | 0.0505 | 0.0909 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.000844, 0.000278] | 110 | 0.0002 | 0.0001 | -0.0002 |
| (0.000278, 0.000351] | 110 | 0.0003 | 0.0001 | -0.0002 |
| (0.000351, 0.000573] | 110 | 0.0004 | 0.0003 | -0.0002 |
| (0.000573, 0.00106] | 110 | 0.0008 | 0.0006 | -0.0002 |
| (0.00106, 0.00349] | 110 | 0.0019 | 0.0017 | -0.0002 |
| (0.00349, 0.0121] | 110 | 0.0066 | 0.0042 | -0.0024 |
| (0.0121, 0.0178] | 110 | 0.0157 | 0.1838 | 0.1681 |
| (0.0178, 0.0223] | 110 | 0.0200 | 0.2204 | 0.2004 |
| (0.0223, 0.0394] | 110 | 0.0295 | 0.1383 | 0.1088 |
| (0.0394, 0.54] | 110 | 0.3289 | 0.3596 | 0.0307 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Calibração por decil (LightGBM + isotonic)

**Modo:** holdout (ano_calib=2018) | **Cobertura:** asimétrico (raw quando pred < 0.50)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.000844, 0.000278] | 110 | 0.0002 | 0.0001 | -0.0002 |
| (0.000278, 0.000351] | 110 | 0.0003 | 0.0001 | -0.0002 |
| (0.000351, 0.000573] | 110 | 0.0004 | 0.0003 | -0.0002 |
| (0.000573, 0.00106] | 110 | 0.0008 | 0.0006 | -0.0002 |
| (0.00106, 0.00349] | 110 | 0.0019 | 0.0017 | -0.0002 |
| (0.00349, 0.0121] | 110 | 0.0066 | 0.0042 | -0.0024 |
| (0.0121, 0.0178] | 110 | 0.0157 | 0.1838 | 0.1681 |
| (0.0178, 0.0223] | 110 | 0.0200 | 0.2204 | 0.2004 |
| (0.0223, 0.0394] | 110 | 0.0295 | 0.1383 | 0.1088 |
| (0.0394, 0.486] | 110 | 0.3192 | 0.3596 | 0.0403 |

> Calibrador isotônico ajustado em predições do modelo no treino. Aplicado pós-hoc na predição do teste.

## MAE por partido — versão calibrada (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 100 | 0.5096 | 0.5166 | 0.5096 | 0.5288 |
| PT | 100 | 0.0639 | 0.0987 | 0.0466 | 0.3928 |
| MDB | 100 | 0.0288 | 0.0310 | 0.0288 | 0.0473 |
| PDT | 100 | 0.0113 | 0.0139 | -0.0096 | 0.0219 |
| NOVO | 100 | 0.0091 | 0.0102 | -0.0091 | 0.0034 |
| UNIÃO | 100 | 0.0008 | 0.0010 | 0.0007 | 0.0043 |
| PTB | 100 | 0.0006 | 0.0007 | -0.0005 | 0.0006 |
| UP | 100 | 0.0003 | 0.0003 | -0.0003 | 0.0003 |
| DC | 100 | 0.0002 | 0.0002 | -0.0002 | 0.0001 |
| PSTU | 100 | 0.0002 | 0.0002 | -0.0001 | 0.0002 |

## Cobertura conformal (split)

**Cobertura nominal:** 90% (α = 0.10) | **q̂ split:** 0.3190 | **Cobertura observada (test):** 0.908

**Cobertura por decil de pred — split:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 110.0000 | 0.0002 | 0.0003 | 1.0000 |
| 1.0000 | 110.0000 | 0.0003 | 0.0004 | 1.0000 |
| 2.0000 | 110.0000 | 0.0004 | 0.0006 | 1.0000 |
| 3.0000 | 110.0000 | 0.0006 | 0.0011 | 1.0000 |
| 4.0000 | 110.0000 | 0.0011 | 0.0035 | 1.0000 |
| 5.0000 | 110.0000 | 0.0035 | 0.0121 | 1.0000 |
| 6.0000 | 110.0000 | 0.0122 | 0.0178 | 0.6545 |
| 7.0000 | 110.0000 | 0.0179 | 0.0223 | 0.6364 |
| 8.0000 | 110.0000 | 0.0223 | 0.0394 | 0.8182 |
| 9.0000 | 110.0000 | 0.0395 | 0.4857 | 0.9727 |

> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é homogêneo, cobertura por decil pode divergir do nominal (intervalo grande demais nos baixos, pequeno demais nos altos).

## Cobertura conformal (Mondrian — estratificado por bin de pred)

**Cobertura observada (test):** 0.961 | **q̂ por bin:** [0.001, 0.038, 0.052, 0.336, 0.346, 0.610, 0.545, 0.541, 0.116, 0.345]

**Cobertura por decil de pred — Mondrian:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 110.0000 | 0.0002 | 0.0003 | 1.0000 |
| 1.0000 | 110.0000 | 0.0003 | 0.0004 | 1.0000 |
| 2.0000 | 110.0000 | 0.0004 | 0.0006 | 1.0000 |
| 3.0000 | 110.0000 | 0.0006 | 0.0011 | 1.0000 |
| 4.0000 | 110.0000 | 0.0011 | 0.0035 | 1.0000 |
| 5.0000 | 110.0000 | 0.0035 | 0.0121 | 1.0000 |
| 6.0000 | 110.0000 | 0.0122 | 0.0178 | 1.0000 |
| 7.0000 | 110.0000 | 0.0179 | 0.0223 | 0.8364 |
| 8.0000 | 110.0000 | 0.0223 | 0.0394 | 0.8182 |
| 9.0000 | 110.0000 | 0.0395 | 0.4857 | 0.9545 |

> Mondrian deve dar cobertura aproximadamente uniforme ao longo dos decis (cobertura condicional).

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| sigla_partido | 39823.9765 |
| share_dep_federal_partido | 27129.9255 |
| swing_share_1t | 26637.3758 |
| log_eleitorado | 18941.4443 |
| share_prefeito_local | 16515.3463 |
| margem_prefeito | 15155.2084 |
| lag_share_1t | 6546.7000 |
| porte | 1798.6705 |
| continuidade_classe | 956.6161 |
| lag_share_1t_sucessao | 874.3352 |
| alinhado_gov_vigente_coligacao | 527.7202 |
| alinhado_gov_vigente_partido | 354.5505 |
| primeiro_mandato_prefeito | 326.9766 |
| alinhado_prefeito_partido | 75.6420 |
| indice_continuidade | 42.0674 |

## Notas

- Target modelado em `logit(share)`, predição destransformada com sigmoid.
- LightGBM com `objective=regression_l1` (MAE) — robusto a caudas.
- Features categóricas (sigla_uf, regiao, porte, continuidade_classe, sigla_partido) tratadas nativamente pelo LightGBM.
- Amostra dev = 1 UF × 100 municípios × 3 anos × ~10 partidos = ~3 mil linhas. Signal-to-noise é limitado — resultados aqui servem pra validar pipeline, não pra conclusões sobre 2026.
- Versão `_iso` corrige saturação no top decil via regressão isotônica treinada num ano holdout do conjunto de treino. Não toca o LGBM.
- Cobertura conformal é uma propriedade do conjunto de calibração: o número observado no test é descritivo, não uma garantia (a garantia vale sob exchangeability calib↔test).

## Próximos passos

- **Fase 5+**: revisar incerteza com mais anos em prod, e considerar conformal por estratos de UF/região (não só por bin de pred).
- **Investigar**: se o LightGBM não bate B1 (`lag_share_1t`) de forma contundente, reavaliar features históricas — pode ser que precisemos de mais anos no treino (rodar em prod).