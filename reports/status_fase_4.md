# Fase 4 — status: primeiro modelo presidencial

**Modo:** prod | **UFs:** all | **Máx municípios:** None
**Split temporal:** treino = [1998, 2002, 2006, 2010, 2014, 2018] (300896 linhas) | teste = 2022 (61270 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 61270 | 0.0517 | 0.1351 | 0.0397 |
| B1_lag_share | 61270 | 0.0582 | 0.1418 | 0.0421 |
| B2_blend | 61270 | 0.0533 | 0.1359 | 0.0409 |
| LightGBM_v1 | 61270 | 0.0099 | 0.0266 | -0.0051 |
| LightGBM_v1_iso | 61270 | 0.0137 | 0.0344 | -0.0095 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 5570 | 0.0802 | 0.0857 | -0.0789 | 0.3943 |
| PDT | 5570 | 0.0122 | 0.0146 | 0.0122 | 0.0222 |
| PT | 5570 | 0.0090 | 0.0125 | 0.0054 | 0.5436 |
| MDB | 5570 | 0.0052 | 0.0079 | 0.0041 | 0.0312 |
| NOVO | 5570 | 0.0014 | 0.0026 | 0.0012 | 0.0030 |
| UNIÃO | 5570 | 0.0001 | 0.0002 | -0.0000 | 0.0043 |
| DC | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| PSTU | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| UP | 5570 | 0.0000 | 0.0001 | -0.0000 | 0.0002 |
| PCB | 5570 | 0.0000 | 0.0001 | -0.0000 | 0.0002 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PB | 2453 | 0.0132 | 0.0348 | -0.0075 | 0.0909 |
| RJ | 1012 | 0.0129 | 0.0304 | -0.0035 | 0.0909 |
| MA | 2387 | 0.0128 | 0.0358 | -0.0092 | 0.0909 |
| RN | 1837 | 0.0126 | 0.0329 | -0.0071 | 0.0909 |
| SE | 825 | 0.0119 | 0.0334 | -0.0082 | 0.0909 |
| AM | 682 | 0.0117 | 0.0337 | -0.0088 | 0.0909 |
| DF | 11 | 0.0116 | 0.0263 | -0.0033 | 0.0909 |
| PI | 2464 | 0.0115 | 0.0306 | -0.0072 | 0.0909 |
| MG | 9383 | 0.0112 | 0.0301 | -0.0067 | 0.0909 |
| RS | 5467 | 0.0109 | 0.0271 | -0.0040 | 0.0909 |
| TO | 1529 | 0.0108 | 0.0305 | -0.0075 | 0.0909 |
| AL | 1122 | 0.0107 | 0.0260 | -0.0045 | 0.0909 |
| PA | 1584 | 0.0106 | 0.0275 | -0.0056 | 0.0909 |
| GO | 2706 | 0.0105 | 0.0281 | -0.0060 | 0.0909 |
| CE | 2024 | 0.0104 | 0.0238 | -0.0021 | 0.0909 |
| AP | 176 | 0.0098 | 0.0264 | -0.0065 | 0.0909 |
| ES | 858 | 0.0096 | 0.0271 | -0.0061 | 0.0909 |
| PR | 4389 | 0.0094 | 0.0234 | -0.0031 | 0.0909 |
| PE | 2035 | 0.0089 | 0.0234 | -0.0055 | 0.0909 |
| MS | 869 | 0.0086 | 0.0246 | -0.0061 | 0.0909 |
| SC | 3245 | 0.0083 | 0.0242 | -0.0048 | 0.0909 |
| BA | 4587 | 0.0073 | 0.0189 | -0.0039 | 0.0909 |
| AC | 242 | 0.0072 | 0.0177 | -0.0002 | 0.0909 |
| MT | 1551 | 0.0068 | 0.0206 | -0.0041 | 0.0909 |
| RO | 572 | 0.0067 | 0.0182 | -0.0028 | 0.0909 |
| SP | 7095 | 0.0063 | 0.0166 | -0.0024 | 0.0909 |
| RR | 165 | 0.0063 | 0.0168 | -0.0014 | 0.0909 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.0009205, 0.000105] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000105, 0.000131] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000131, 0.000266] | 6127 | 0.0002 | 0.0002 | -0.0000 |
| (0.000266, 0.000558] | 6127 | 0.0004 | 0.0004 | -0.0000 |
| (0.000558, 0.00155] | 6128 | 0.0009 | 0.0009 | -0.0000 |
| (0.00155, 0.00444] | 6126 | 0.0027 | 0.0038 | 0.0011 |
| (0.00444, 0.00995] | 6127 | 0.0075 | 0.0137 | 0.0062 |
| (0.00995, 0.0412] | 6127 | 0.0191 | 0.0276 | 0.0085 |
| (0.0412, 0.48] | 6127 | 0.2882 | 0.2552 | -0.0330 |
| (0.48, 0.894] | 6127 | 0.6409 | 0.6073 | -0.0336 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Calibração por decil (LightGBM + isotonic)

**Modo:** holdout (ano_calib=2018) | **Cobertura:** asimétrico (raw quando pred < 0.50)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.0009205, 0.000105] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000105, 0.000131] | 6127 | 0.0001 | 0.0000 | -0.0001 |
| (0.000131, 0.000266] | 6127 | 0.0002 | 0.0002 | -0.0000 |
| (0.000266, 0.000558] | 6127 | 0.0004 | 0.0004 | -0.0000 |
| (0.000558, 0.00155] | 6128 | 0.0009 | 0.0009 | -0.0000 |
| (0.00155, 0.00444] | 6126 | 0.0027 | 0.0038 | 0.0011 |
| (0.00444, 0.00995] | 6127 | 0.0075 | 0.0137 | 0.0062 |
| (0.00995, 0.0412] | 6127 | 0.0191 | 0.0276 | 0.0085 |
| (0.0412, 0.48] | 6127 | 0.2882 | 0.2552 | -0.0330 |
| (0.48, 0.91] | 6127 | 0.6851 | 0.6073 | -0.0778 |

> Calibrador isotônico ajustado em predições do modelo no treino. Aplicado pós-hoc na predição do teste.

## MAE por partido — versão calibrada (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 5570 | 0.1008 | 0.1058 | -0.1006 | 0.3943 |
| PT | 5570 | 0.0311 | 0.0390 | -0.0215 | 0.5436 |
| PDT | 5570 | 0.0122 | 0.0146 | 0.0122 | 0.0222 |
| MDB | 5570 | 0.0052 | 0.0079 | 0.0041 | 0.0312 |
| NOVO | 5570 | 0.0014 | 0.0026 | 0.0012 | 0.0030 |
| UNIÃO | 5570 | 0.0001 | 0.0002 | -0.0000 | 0.0043 |
| DC | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| PSTU | 5570 | 0.0001 | 0.0001 | -0.0001 | 0.0001 |
| UP | 5570 | 0.0000 | 0.0001 | -0.0000 | 0.0002 |
| PCB | 5570 | 0.0000 | 0.0001 | -0.0000 | 0.0002 |

## Cobertura conformal (split)

**Cobertura nominal:** 90% (α = 0.10) | **q̂ split:** 0.0314 | **Cobertura observada (test):** 0.867

**Cobertura por decil de pred — split:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0006 | 1.0000 |
| 4.0000 | 6128.0000 | 0.0006 | 0.0015 | 1.0000 |
| 5.0000 | 6126.0000 | 0.0015 | 0.0044 | 1.0000 |
| 6.0000 | 6127.0000 | 0.0044 | 0.0100 | 0.9984 |
| 7.0000 | 6127.0000 | 0.0100 | 0.0412 | 0.9785 |
| 8.0000 | 6127.0000 | 0.0412 | 0.4805 | 0.5619 |
| 9.0000 | 6127.0000 | 0.4805 | 0.9101 | 0.1276 |

> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é homogêneo, cobertura por decil pode divergir do nominal (intervalo grande demais nos baixos, pequeno demais nos altos).

## Cobertura conformal (Mondrian — estratificado por bin de pred)

**Cobertura observada (test):** 0.863 | **q̂ por bin:** [0.006, 0.006, 0.006, 0.006, 0.006, 0.006, 0.022, 0.063, 0.051, 0.097]

**Cobertura por decil de pred — Mondrian:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 1.0000 | 6127.0000 | 0.0001 | 0.0001 | 1.0000 |
| 2.0000 | 6127.0000 | 0.0001 | 0.0003 | 1.0000 |
| 3.0000 | 6127.0000 | 0.0003 | 0.0006 | 1.0000 |
| 4.0000 | 6128.0000 | 0.0006 | 0.0015 | 1.0000 |
| 5.0000 | 6126.0000 | 0.0015 | 0.0044 | 0.9545 |
| 6.0000 | 6127.0000 | 0.0044 | 0.0100 | 0.5722 |
| 7.0000 | 6127.0000 | 0.0100 | 0.0412 | 0.7061 |
| 8.0000 | 6127.0000 | 0.0412 | 0.4805 | 0.7415 |
| 9.0000 | 6127.0000 | 0.4805 | 0.9101 | 0.6597 |

> Mondrian deve dar cobertura aproximadamente uniforme ao longo dos decis (cobertura condicional).

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| swing_share_1t | 3462610.0162 |
| share_pesquisa_nacional | 2954508.2379 |
| lag_share_1t | 1665391.2018 |
| share_pesquisa_uf | 1183659.8044 |
| sigla_partido | 823355.6223 |
| sigla_uf | 317661.2814 |
| share_dep_federal_partido | 164168.8711 |
| lag_share_1t_sucessao | 145515.2386 |
| volatilidade_partido | 125266.2249 |
| log_eleitorado | 114315.2009 |
| lag2_share_1t | 92153.7813 |
| pesquisa_uf_disponivel | 13414.1309 |
| alinhado_gov_vigente_coligacao | 12241.3441 |
| porte | 9709.0245 |
| alinhado_gov_concorrente_coligacao | 9164.6929 |

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