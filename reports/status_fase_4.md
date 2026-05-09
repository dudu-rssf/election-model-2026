# Fase 4 — status: primeiro modelo presidencial

**Modo:** prod | **UFs:** all | **Máx municípios:** None
**Split temporal:** treino = [1998, 2002, 2006, 2010, 2014, 2018] (300896 linhas) | teste = 2022 (61270 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 61270 | 0.0517 | 0.1351 | 0.0397 |
| B1_lag_share | 61270 | 0.0582 | 0.1418 | 0.0421 |
| B2_blend | 61270 | 0.0533 | 0.1359 | 0.0409 |
| LightGBM_v1 | 61270 | 0.0174 | 0.0526 | 0.0163 |
| LightGBM_v1_iso | 61270 | 0.0174 | 0.0527 | 0.0165 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 5570 | 0.1540 | 0.1727 | 0.1540 | 0.3943 |
| MDB | 5570 | 0.0140 | 0.0175 | 0.0139 | 0.0312 |
| PDT | 5570 | 0.0116 | 0.0138 | 0.0116 | 0.0222 |
| PT | 5570 | 0.0089 | 0.0132 | -0.0012 | 0.5436 |
| NOVO | 5570 | 0.0017 | 0.0028 | 0.0011 | 0.0030 |
| UNIÃO | 5570 | 0.0004 | 0.0005 | 0.0004 | 0.0043 |
| DC | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| PSTU | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| UP | 5570 | 0.0001 | 0.0001 | -0.0000 | 0.0002 |
| PTB | 5570 | 0.0000 | 0.0001 | 0.0000 | 0.0007 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| DF | 11 | 0.0286 | 0.0703 | 0.0238 | 0.0909 |
| AC | 242 | 0.0278 | 0.0794 | 0.0269 | 0.0909 |
| RO | 572 | 0.0278 | 0.0819 | 0.0270 | 0.0909 |
| RR | 165 | 0.0275 | 0.0834 | 0.0272 | 0.0909 |
| SC | 3245 | 0.0248 | 0.0709 | 0.0236 | 0.0909 |
| SP | 7095 | 0.0245 | 0.0670 | 0.0215 | 0.0909 |
| MT | 1551 | 0.0238 | 0.0730 | 0.0234 | 0.0909 |
| RS | 5467 | 0.0235 | 0.0661 | 0.0228 | 0.0909 |
| ES | 858 | 0.0225 | 0.0649 | 0.0219 | 0.0909 |
| MS | 869 | 0.0218 | 0.0631 | 0.0212 | 0.0909 |
| PR | 4389 | 0.0216 | 0.0638 | 0.0209 | 0.0909 |
| RJ | 1012 | 0.0216 | 0.0610 | 0.0207 | 0.0909 |
| GO | 2706 | 0.0214 | 0.0602 | 0.0207 | 0.0909 |
| AP | 176 | 0.0182 | 0.0498 | 0.0176 | 0.0909 |
| MG | 9383 | 0.0174 | 0.0495 | 0.0165 | 0.0909 |
| PA | 1584 | 0.0173 | 0.0505 | 0.0169 | 0.0909 |
| TO | 1529 | 0.0152 | 0.0447 | 0.0146 | 0.0909 |
| AL | 1122 | 0.0132 | 0.0349 | 0.0125 | 0.0909 |
| AM | 682 | 0.0122 | 0.0362 | 0.0114 | 0.0909 |
| SE | 825 | 0.0106 | 0.0294 | 0.0102 | 0.0909 |
| RN | 1837 | 0.0102 | 0.0290 | 0.0092 | 0.0909 |
| CE | 2024 | 0.0099 | 0.0246 | 0.0089 | 0.0909 |
| MA | 2387 | 0.0095 | 0.0274 | 0.0087 | 0.0909 |
| PB | 2453 | 0.0090 | 0.0241 | 0.0082 | 0.0909 |
| PE | 2035 | 0.0086 | 0.0268 | 0.0078 | 0.0909 |
| BA | 4587 | 0.0084 | 0.0252 | 0.0078 | 0.0909 |
| PI | 2464 | 0.0061 | 0.0157 | 0.0055 | 0.0909 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.0009131, 0.000105] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000105, 0.000136] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000136, 0.000273] | 6127 | 0.0002 | 0.0002 | -0.0000 |
| (0.000273, 0.000608] | 6127 | 0.0004 | 0.0004 | -0.0000 |
| (0.000608, 0.00166] | 6127 | 0.0011 | 0.0009 | -0.0002 |
| (0.00166, 0.00387] | 6127 | 0.0025 | 0.0040 | 0.0014 |
| (0.00387, 0.0101] | 6127 | 0.0069 | 0.0129 | 0.0060 |
| (0.0101, 0.0227] | 6127 | 0.0158 | 0.0280 | 0.0122 |
| (0.0227, 0.334] | 6127 | 0.1784 | 0.2727 | 0.0944 |
| (0.334, 0.888] | 6127 | 0.5404 | 0.5900 | 0.0496 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Calibração por decil (LightGBM + isotonic)

**Modo:** holdout (ano_calib=2018) | **Cobertura:** asimétrico (raw quando pred < 0.50)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.0009131, 0.000105] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000105, 0.000136] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000136, 0.000273] | 6127 | 0.0002 | 0.0002 | -0.0000 |
| (0.000273, 0.000608] | 6127 | 0.0004 | 0.0004 | -0.0000 |
| (0.000608, 0.00166] | 6127 | 0.0011 | 0.0009 | -0.0002 |
| (0.00166, 0.00387] | 6127 | 0.0025 | 0.0040 | 0.0014 |
| (0.00387, 0.0101] | 6127 | 0.0069 | 0.0129 | 0.0060 |
| (0.0101, 0.0227] | 6127 | 0.0158 | 0.0280 | 0.0122 |
| (0.0227, 0.334] | 6127 | 0.1784 | 0.2727 | 0.0944 |
| (0.334, 0.925] | 6127 | 0.5390 | 0.5900 | 0.0510 |

> Calibrador isotônico ajustado em predições do modelo no treino. Aplicado pós-hoc na predição do teste.

## MAE por partido — versão calibrada (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 5570 | 0.1540 | 0.1727 | 0.1540 | 0.3943 |
| MDB | 5570 | 0.0140 | 0.0175 | 0.0139 | 0.0312 |
| PDT | 5570 | 0.0116 | 0.0138 | 0.0116 | 0.0222 |
| PT | 5570 | 0.0093 | 0.0136 | 0.0004 | 0.5436 |
| NOVO | 5570 | 0.0017 | 0.0028 | 0.0011 | 0.0030 |
| UNIÃO | 5570 | 0.0004 | 0.0005 | 0.0004 | 0.0043 |
| DC | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| PSTU | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| UP | 5570 | 0.0001 | 0.0001 | -0.0000 | 0.0002 |
| PTB | 5570 | 0.0000 | 0.0001 | 0.0000 | 0.0007 |

## Cobertura conformal (split)

**Cobertura nominal:** 90% (α = 0.10) | **q̂ split:** 0.0628 | **Cobertura observada (test):** 0.923

**Cobertura por decil de pred — split:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0006 | 1.0000 |
| 4.0000 | 6127.0000 | 0.0006 | 0.0017 | 1.0000 |
| 5.0000 | 6127.0000 | 0.0017 | 0.0039 | 1.0000 |
| 6.0000 | 6127.0000 | 0.0039 | 0.0101 | 0.9998 |
| 7.0000 | 6127.0000 | 0.0101 | 0.0227 | 0.9997 |
| 8.0000 | 6127.0000 | 0.0227 | 0.3337 | 0.4448 |
| 9.0000 | 6127.0000 | 0.3337 | 0.9254 | 0.7901 |

> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é homogêneo, cobertura por decil pode divergir do nominal (intervalo grande demais nos baixos, pequeno demais nos altos).

## Cobertura conformal (Mondrian — estratificado por bin de pred)

**Cobertura observada (test):** 0.935 | **q̂ por bin:** [0.013, 0.013, 0.013, 0.013, 0.013, 0.013, 0.016, 0.059, 0.208, 0.385]

**Cobertura por decil de pred — Mondrian:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0006 | 1.0000 |
| 4.0000 | 6127.0000 | 0.0006 | 0.0017 | 1.0000 |
| 5.0000 | 6127.0000 | 0.0017 | 0.0039 | 0.9928 |
| 6.0000 | 6127.0000 | 0.0039 | 0.0101 | 0.8647 |
| 7.0000 | 6127.0000 | 0.0101 | 0.0227 | 0.6690 |
| 8.0000 | 6127.0000 | 0.0227 | 0.3337 | 0.8255 |
| 9.0000 | 6127.0000 | 0.3337 | 0.9254 | 0.9990 |

> Mondrian deve dar cobertura aproximadamente uniforme ao longo dos decis (cobertura condicional).

## Cobertura conformal (MondrianCategorical — estratos por `sigla_partido`, `regiao`)

**Cobertura observada (test):** 0.868 | **Estratos:** 65 (fallback global: 0)

**Cobertura por decil de pred — MondrianCategorical:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0006 | 1.0000 |
| 4.0000 | 6127.0000 | 0.0006 | 0.0017 | 1.0000 |
| 5.0000 | 6127.0000 | 0.0017 | 0.0039 | 0.9928 |
| 6.0000 | 6127.0000 | 0.0039 | 0.0101 | 0.9674 |
| 7.0000 | 6127.0000 | 0.0101 | 0.0227 | 0.7349 |
| 8.0000 | 6127.0000 | 0.0227 | 0.3337 | 0.2544 |
| 9.0000 | 6127.0000 | 0.3337 | 0.9254 | 0.7302 |

**Top 10 estratos com menor cobertura empírica:**

| estrato | n | cobertura |
| --- | --- | --- |
| PL|Sul | 1191 | 0.0000 |
| PL|Centro-Oeste | 467 | 0.0021 |
| PL|Sudeste | 1668 | 0.0192 |
| PL|Norte | 450 | 0.0444 |
| MDB|Sul | 1191 | 0.2233 |
| MDB|Sudeste | 1668 | 0.3177 |
| MDB|Centro-Oeste | 467 | 0.3212 |
| MDB|Norte | 450 | 0.4311 |
| PL|Nordeste | 1794 | 0.4783 |
| PT|Sudeste | 1668 | 0.8147 |

> MondrianCategorical foca em estratos categóricos (e.g., partido) onde o regime de erro pode ser distinto. Estratos sub-cobertos sinalizam exchangeability quebrada entre calib↔test (caso típico: PL 2022 com migração do Bolsonaro).

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| sigla_partido | 4102028.6820 |
| swing_share_1t | 3661333.4990 |
| lag_share_1t | 1693919.8741 |
| share_dep_federal_partido | 557479.9926 |
| sigla_uf | 315826.5358 |
| volatilidade_partido | 305781.1596 |
| lag2_share_1t | 227926.6759 |
| log_eleitorado | 126253.8324 |
| lag_share_1t_sucessao | 107987.7524 |
| alinhado_gov_vigente_coligacao | 34532.7717 |
| alinhado_gov_concorrente_coligacao | 18700.9376 |
| regiao | 7082.1936 |
| porte | 5506.3101 |
| alinhado_gov_concorrente_partido | 3701.3872 |
| alinhado_gov_vigente_partido | 3697.1461 |

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