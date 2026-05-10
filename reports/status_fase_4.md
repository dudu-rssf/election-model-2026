# Fase 4 — status: primeiro modelo presidencial

**Modo:** prod | **UFs:** all | **Máx municípios:** None
**Split temporal:** treino = [1998, 2002, 2006, 2010, 2014, 2018] (300896 linhas) | teste = 2022 (61270 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 61270 | 0.0517 | 0.1351 | 0.0397 |
| B1_lag_share | 61270 | 0.0582 | 0.1418 | 0.0421 |
| B2_blend | 61270 | 0.0533 | 0.1359 | 0.0409 |
| LightGBM_v1 | 61270 | 0.0134 | 0.0371 | -0.0086 |
| LightGBM_v1_iso | 61270 | 0.0197 | 0.0511 | -0.0153 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 5570 | 0.1175 | 0.1210 | -0.1171 | 0.3943 |
| PDT | 5570 | 0.0128 | 0.0155 | 0.0127 | 0.0222 |
| MDB | 5570 | 0.0082 | 0.0109 | 0.0080 | 0.0312 |
| PT | 5570 | 0.0069 | 0.0097 | 0.0012 | 0.5436 |
| NOVO | 5570 | 0.0014 | 0.0025 | 0.0009 | 0.0030 |
| UNIÃO | 5570 | 0.0003 | 0.0004 | 0.0002 | 0.0043 |
| PTB | 5570 | 0.0001 | 0.0001 | 0.0001 | 0.0007 |
| DC | 5570 | 0.0001 | 0.0001 | -0.0000 | 0.0001 |
| PSTU | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| UP | 5570 | 0.0001 | 0.0001 | -0.0000 | 0.0002 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| CE | 2024 | 0.0165 | 0.0401 | -0.0057 | 0.0909 |
| DF | 11 | 0.0165 | 0.0384 | -0.0057 | 0.0909 |
| RJ | 1012 | 0.0163 | 0.0428 | -0.0087 | 0.0909 |
| RN | 1837 | 0.0157 | 0.0427 | -0.0097 | 0.0909 |
| SE | 825 | 0.0152 | 0.0439 | -0.0115 | 0.0909 |
| PB | 2453 | 0.0147 | 0.0395 | -0.0086 | 0.0909 |
| GO | 2706 | 0.0146 | 0.0398 | -0.0092 | 0.0909 |
| MG | 9383 | 0.0144 | 0.0402 | -0.0100 | 0.0909 |
| AP | 176 | 0.0143 | 0.0393 | -0.0093 | 0.0909 |
| RS | 5467 | 0.0141 | 0.0372 | -0.0073 | 0.0909 |
| MA | 2387 | 0.0140 | 0.0394 | -0.0104 | 0.0909 |
| PA | 1584 | 0.0137 | 0.0366 | -0.0081 | 0.0909 |
| TO | 1529 | 0.0136 | 0.0381 | -0.0094 | 0.0909 |
| BA | 4587 | 0.0132 | 0.0377 | -0.0097 | 0.0909 |
| SP | 7095 | 0.0130 | 0.0362 | -0.0087 | 0.0909 |
| PE | 2035 | 0.0129 | 0.0369 | -0.0101 | 0.0909 |
| ES | 858 | 0.0127 | 0.0349 | -0.0083 | 0.0909 |
| AL | 1122 | 0.0127 | 0.0320 | -0.0065 | 0.0909 |
| AM | 682 | 0.0127 | 0.0361 | -0.0096 | 0.0909 |
| MS | 869 | 0.0126 | 0.0357 | -0.0088 | 0.0909 |
| PR | 4389 | 0.0120 | 0.0339 | -0.0080 | 0.0909 |
| SC | 3245 | 0.0112 | 0.0315 | -0.0064 | 0.0909 |
| PI | 2464 | 0.0105 | 0.0273 | -0.0063 | 0.0909 |
| RO | 572 | 0.0101 | 0.0287 | -0.0063 | 0.0909 |
| MT | 1551 | 0.0096 | 0.0300 | -0.0075 | 0.0909 |
| RR | 165 | 0.0085 | 0.0232 | -0.0035 | 0.0909 |
| AC | 242 | 0.0081 | 0.0186 | 0.0009 | 0.0909 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.0009185, 0.000106] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000106, 0.000131] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000131, 0.000275] | 6127 | 0.0002 | 0.0002 | -0.0000 |
| (0.000275, 0.000547] | 6127 | 0.0004 | 0.0004 | -0.0000 |
| (0.000547, 0.00157] | 6127 | 0.0009 | 0.0009 | -0.0000 |
| (0.00157, 0.00434] | 6127 | 0.0028 | 0.0037 | 0.0010 |
| (0.00434, 0.00937] | 6127 | 0.0074 | 0.0133 | 0.0059 |
| (0.00937, 0.0326] | 6127 | 0.0169 | 0.0283 | 0.0114 |
| (0.0326, 0.503] | 6127 | 0.3031 | 0.2575 | -0.0456 |
| (0.503, 0.893] | 6127 | 0.6626 | 0.6047 | -0.0579 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Calibração por decil (LightGBM + isotonic)

**Modo:** holdout (ano_calib=2018) | **Cobertura:** asimétrico (raw quando pred < 0.50)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.0009185, 0.000106] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000106, 0.000131] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000131, 0.000275] | 6127 | 0.0002 | 0.0002 | -0.0000 |
| (0.000275, 0.000547] | 6127 | 0.0004 | 0.0004 | -0.0000 |
| (0.000547, 0.00157] | 6127 | 0.0009 | 0.0009 | -0.0000 |
| (0.00157, 0.00434] | 6127 | 0.0028 | 0.0037 | 0.0010 |
| (0.00434, 0.00937] | 6127 | 0.0074 | 0.0133 | 0.0059 |
| (0.00937, 0.0326] | 6127 | 0.0169 | 0.0283 | 0.0114 |
| (0.0326, 0.531] | 6130 | 0.3035 | 0.2576 | -0.0458 |
| (0.531, 0.932] | 6124 | 0.7297 | 0.6048 | -0.1250 |

> Calibrador isotônico ajustado em predições do modelo no treino. Aplicado pós-hoc na predição do teste.

## MAE por partido — versão calibrada (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 5570 | 0.1550 | 0.1608 | -0.1550 | 0.3943 |
| PT | 5570 | 0.0387 | 0.0503 | -0.0349 | 0.5436 |
| PDT | 5570 | 0.0128 | 0.0155 | 0.0127 | 0.0222 |
| MDB | 5570 | 0.0082 | 0.0109 | 0.0080 | 0.0312 |
| NOVO | 5570 | 0.0014 | 0.0025 | 0.0009 | 0.0030 |
| UNIÃO | 5570 | 0.0003 | 0.0004 | 0.0002 | 0.0043 |
| PTB | 5570 | 0.0001 | 0.0001 | 0.0001 | 0.0007 |
| DC | 5570 | 0.0001 | 0.0001 | -0.0000 | 0.0001 |
| PSTU | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| UP | 5570 | 0.0001 | 0.0001 | -0.0000 | 0.0002 |

## Cobertura conformal (split)

**Cobertura nominal:** 90% (α = 0.10) | **q̂ split:** 0.0356 | **Cobertura observada (test):** 0.860

**Cobertura por decil de pred — split:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0005 | 1.0000 |
| 4.0000 | 6127.0000 | 0.0005 | 0.0016 | 1.0000 |
| 5.0000 | 6127.0000 | 0.0016 | 0.0043 | 1.0000 |
| 6.0000 | 6127.0000 | 0.0043 | 0.0094 | 0.9992 |
| 7.0000 | 6127.0000 | 0.0094 | 0.0326 | 0.9765 |
| 8.0000 | 6130.0000 | 0.0326 | 0.5310 | 0.6010 |
| 9.0000 | 6124.0000 | 0.5338 | 0.9324 | 0.0194 |

> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é homogêneo, cobertura por decil pode divergir do nominal (intervalo grande demais nos baixos, pequeno demais nos altos).

## Cobertura conformal (Mondrian — estratificado por bin de pred)

**Cobertura observada (test):** 0.868 | **q̂ por bin:** [0.007, 0.007, 0.007, 0.007, 0.007, 0.007, 0.033, 0.064, 0.041, 0.102]

**Cobertura por decil de pred — Mondrian:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0005 | 1.0000 |
| 4.0000 | 6127.0000 | 0.0005 | 0.0016 | 1.0000 |
| 5.0000 | 6127.0000 | 0.0016 | 0.0043 | 0.9765 |
| 6.0000 | 6127.0000 | 0.0043 | 0.0094 | 0.6249 |
| 7.0000 | 6127.0000 | 0.0094 | 0.0326 | 0.9585 |
| 8.0000 | 6130.0000 | 0.0326 | 0.5310 | 0.6155 |
| 9.0000 | 6124.0000 | 0.5338 | 0.9324 | 0.5087 |

> Mondrian deve dar cobertura aproximadamente uniforme ao longo dos decis (cobertura condicional).

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| share_pesquisa_nacional | 3835698.3958 |
| swing_share_1t | 3400580.5125 |
| sigla_partido | 1517377.7155 |
| lag_share_1t | 1515930.2682 |
| sigla_uf | 406076.6435 |
| lag_share_1t_sucessao | 196332.6197 |
| volatilidade_partido | 195707.3512 |
| share_dep_federal_partido | 166938.8241 |
| log_eleitorado | 160251.6325 |
| lag2_share_1t | 120205.4882 |
| porte | 19514.1593 |
| regiao | 17918.5086 |
| alinhado_gov_vigente_coligacao | 16109.2955 |
| alinhado_gov_concorrente_coligacao | 13025.1826 |
| alinhado_gov_concorrente_partido | 6200.4773 |

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