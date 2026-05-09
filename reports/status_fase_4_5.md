# Fase 4.5 — status: primeiro modelo de prefeito

**Modo:** prod | **UFs:** all | **Máx municípios:** None
**Eixo:** `ano_municipal` | **Split temporal:** treino = [1996, 2000, 2004, 2008, 2012, 2016, 2020] (96121 linhas) | teste = 2024 (15047 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 15047 | 0.1839 | 0.2357 | 0.0416 |
| B1_lag_share | 15047 | 0.2758 | 0.3572 | 0.2337 |
| B2_blend | 15047 | 0.2046 | 0.2649 | 0.1377 |
| LightGBM_prefeito_v1 | 15047 | 0.0056 | 0.0145 | 0.0000 |
| LightGBM_prefeito_v1_iso | 15047 | 0.0063 | 0.0143 | 0.0025 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PC do B | 55 | 0.0100 | 0.0341 | 0.0050 | 0.3508 |
| CIDADANIA | 106 | 0.0091 | 0.0282 | -0.0014 | 0.3425 |
| REPUBLICANOS | 1079 | 0.0078 | 0.0162 | 0.0029 | 0.3986 |
| MDB | 1873 | 0.0076 | 0.0186 | -0.0004 | 0.4535 |
| PSD | 1707 | 0.0075 | 0.0178 | -0.0010 | 0.4695 |
| PP | 1486 | 0.0060 | 0.0171 | -0.0013 | 0.4548 |
| PSDB | 702 | 0.0060 | 0.0167 | -0.0007 | 0.3909 |
| AVANTE | 373 | 0.0060 | 0.0164 | 0.0014 | 0.3575 |
| PSB | 776 | 0.0053 | 0.0123 | 0.0001 | 0.4009 |
| PDT | 602 | 0.0049 | 0.0133 | -0.0002 | 0.3167 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PB | 509 | 0.0072 | 0.0215 | -0.0017 | 0.4381 |
| PI | 484 | 0.0069 | 0.0240 | -0.0022 | 0.4607 |
| MS | 226 | 0.0063 | 0.0132 | -0.0011 | 0.3496 |
| MT | 356 | 0.0063 | 0.0162 | -0.0013 | 0.3961 |
| AP | 52 | 0.0061 | 0.0113 | 0.0023 | 0.3077 |
| AL | 254 | 0.0060 | 0.0143 | -0.0004 | 0.4016 |
| TO | 316 | 0.0060 | 0.0103 | 0.0028 | 0.4399 |
| SP | 2035 | 0.0058 | 0.0138 | 0.0010 | 0.3170 |
| CE | 467 | 0.0058 | 0.0127 | -0.0002 | 0.3940 |
| PR | 1114 | 0.0057 | 0.0162 | -0.0015 | 0.3582 |
| RR | 42 | 0.0057 | 0.0114 | -0.0019 | 0.3571 |
| MG | 2286 | 0.0057 | 0.0160 | -0.0003 | 0.3731 |
| RN | 375 | 0.0056 | 0.0129 | 0.0005 | 0.4453 |
| GO | 623 | 0.0056 | 0.0151 | 0.0017 | 0.3933 |
| PE | 498 | 0.0056 | 0.0139 | 0.0005 | 0.3675 |
| PA | 453 | 0.0055 | 0.0129 | -0.0003 | 0.3179 |
| RJ | 357 | 0.0055 | 0.0126 | 0.0012 | 0.2577 |
| BA | 1091 | 0.0054 | 0.0148 | 0.0001 | 0.3813 |
| ES | 272 | 0.0053 | 0.0124 | 0.0006 | 0.2868 |
| SC | 828 | 0.0050 | 0.0121 | -0.0004 | 0.3563 |
| RS | 1192 | 0.0049 | 0.0116 | 0.0000 | 0.4161 |
| RO | 161 | 0.0048 | 0.0111 | 0.0010 | 0.3230 |
| AC | 61 | 0.0047 | 0.0075 | 0.0012 | 0.3607 |
| MA | 604 | 0.0042 | 0.0101 | 0.0003 | 0.3543 |
| AM | 198 | 0.0039 | 0.0074 | 0.0007 | 0.3131 |
| SE | 193 | 0.0037 | 0.0076 | 0.0012 | 0.3731 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.000887, 0.0255] | 1505 | 0.0115 | 0.0114 | -0.0002 |
| (0.0255, 0.0915] | 1505 | 0.0525 | 0.0531 | 0.0006 |
| (0.0915, 0.219] | 1504 | 0.1545 | 0.1561 | 0.0016 |
| (0.219, 0.32] | 1505 | 0.2724 | 0.2732 | 0.0007 |
| (0.32, 0.394] | 1505 | 0.3575 | 0.3583 | 0.0008 |
| (0.394, 0.453] | 1504 | 0.4243 | 0.4234 | -0.0009 |
| (0.453, 0.504] | 1505 | 0.4778 | 0.4784 | 0.0006 |
| (0.504, 0.565] | 1504 | 0.5319 | 0.5329 | 0.0009 |
| (0.565, 0.666] | 1505 | 0.6083 | 0.6103 | 0.0020 |
| (0.666, 1.0] | 1505 | 0.8020 | 0.7960 | -0.0060 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Calibração por decil (LightGBM + isotonic)

**Modo:** holdout (ano_calib=2020) | **Cobertura:** asimétrico (raw quando pred < 0.50)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.000887, 0.0255] | 1505 | 0.0115 | 0.0114 | -0.0002 |
| (0.0255, 0.0915] | 1505 | 0.0525 | 0.0531 | 0.0006 |
| (0.0915, 0.219] | 1504 | 0.1545 | 0.1561 | 0.0016 |
| (0.219, 0.32] | 1505 | 0.2724 | 0.2732 | 0.0007 |
| (0.32, 0.394] | 1505 | 0.3575 | 0.3583 | 0.0008 |
| (0.394, 0.453] | 1504 | 0.4243 | 0.4234 | -0.0009 |
| (0.453, 0.504] | 1505 | 0.4774 | 0.4784 | 0.0010 |
| (0.504, 0.556] | 1525 | 0.5298 | 0.5333 | 0.0035 |
| (0.556, 0.67] | 1494 | 0.5995 | 0.6114 | 0.0119 |
| (0.67, 1.0] | 1495 | 0.7910 | 0.7968 | 0.0059 |

> Calibrador isotônico ajustado em predições do modelo no treino. Aplicado pós-hoc na predição do teste.

## MAE por partido — versão calibrada (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PC do B | 55 | 0.0108 | 0.0342 | 0.0073 | 0.3508 |
| CIDADANIA | 106 | 0.0095 | 0.0267 | 0.0021 | 0.3425 |
| REPUBLICANOS | 1079 | 0.0086 | 0.0162 | 0.0055 | 0.3986 |
| PSD | 1707 | 0.0085 | 0.0168 | 0.0033 | 0.4695 |
| MDB | 1873 | 0.0083 | 0.0175 | 0.0032 | 0.4535 |
| PP | 1486 | 0.0075 | 0.0178 | 0.0021 | 0.4548 |
| PSDB | 702 | 0.0068 | 0.0163 | 0.0017 | 0.3909 |
| AVANTE | 373 | 0.0067 | 0.0150 | 0.0037 | 0.3575 |
| PSB | 776 | 0.0064 | 0.0123 | 0.0031 | 0.4009 |
| UNIÃO | 1250 | 0.0056 | 0.0120 | 0.0028 | 0.4431 |

## Cobertura conformal (split)

**Cobertura nominal:** 90% (α = 0.10) | **q̂ split:** 0.0128 | **Cobertura observada (test):** 0.867

**Cobertura por decil de pred — split:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 1505.0000 | 0.0001 | 0.0255 | 0.9907 |
| 1.0000 | 1505.0000 | 0.0255 | 0.0915 | 0.9555 |
| 2.0000 | 1504.0000 | 0.0916 | 0.2193 | 0.9368 |
| 3.0000 | 1505.0000 | 0.2194 | 0.3205 | 0.9442 |
| 4.0000 | 1505.0000 | 0.3205 | 0.3938 | 0.9442 |
| 5.0000 | 1504.0000 | 0.3938 | 0.4527 | 0.9282 |
| 6.0000 | 1505.0000 | 0.4527 | 0.5040 | 0.9581 |
| 7.0000 | 1525.0000 | 0.5052 | 0.5563 | 0.9023 |
| 8.0000 | 1494.0000 | 0.5589 | 0.6704 | 0.6024 |
| 9.0000 | 1495.0000 | 0.6705 | 1.0000 | 0.5057 |

> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é homogêneo, cobertura por decil pode divergir do nominal (intervalo grande demais nos baixos, pequeno demais nos altos).

## Cobertura conformal (Mondrian — estratificado por bin de pred)

**Cobertura observada (test):** 0.905 | **q̂ por bin:** [0.001, 0.005, 0.007, 0.009, 0.010, 0.010, 0.009, 0.010, 0.013, 0.045]

**Cobertura por decil de pred — Mondrian:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 1505.0000 | 0.0001 | 0.0255 | 0.9076 |
| 1.0000 | 1505.0000 | 0.0255 | 0.0915 | 0.8784 |
| 2.0000 | 1504.0000 | 0.0916 | 0.2193 | 0.8956 |
| 3.0000 | 1505.0000 | 0.2194 | 0.3205 | 0.9282 |
| 4.0000 | 1505.0000 | 0.3205 | 0.3938 | 0.8977 |
| 5.0000 | 1504.0000 | 0.3938 | 0.4527 | 0.8770 |
| 6.0000 | 1505.0000 | 0.4527 | 0.5040 | 0.9176 |
| 7.0000 | 1525.0000 | 0.5052 | 0.5563 | 0.8984 |
| 8.0000 | 1494.0000 | 0.5589 | 0.6704 | 0.9605 |
| 9.0000 | 1495.0000 | 0.6705 | 1.0000 | 0.8856 |

> Mondrian deve dar cobertura aproximadamente uniforme ao longo dos decis (cobertura condicional).

## Cobertura conformal (MondrianCategorical — estratos por `sigla_partido`, `regiao`)

**Cobertura observada (test):** 0.883 | **Estratos:** 158 (fallback global: 22)

**Cobertura por decil de pred — MondrianCategorical:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 1505.0000 | 0.0001 | 0.0255 | 0.9894 |
| 1.0000 | 1505.0000 | 0.0255 | 0.0915 | 0.9488 |
| 2.0000 | 1504.0000 | 0.0916 | 0.2193 | 0.9362 |
| 3.0000 | 1505.0000 | 0.2194 | 0.3205 | 0.9535 |
| 4.0000 | 1505.0000 | 0.3205 | 0.3938 | 0.9462 |
| 5.0000 | 1504.0000 | 0.3938 | 0.4527 | 0.9335 |
| 6.0000 | 1505.0000 | 0.4527 | 0.5040 | 0.9701 |
| 7.0000 | 1525.0000 | 0.5052 | 0.5563 | 0.9154 |
| 8.0000 | 1494.0000 | 0.5589 | 0.6704 | 0.6680 |
| 9.0000 | 1495.0000 | 0.6705 | 1.0000 | 0.5639 |

**Top 10 estratos com menor cobertura empírica:**

| estrato | n | cobertura |
| --- | --- | --- |
| REPUBLICANOS|Norte | 133 | 0.4511 |
| AVANTE|Centro-Oeste | 6 | 0.5000 |
| PV|Nordeste | 18 | 0.5556 |
| PODE|Centro-Oeste | 36 | 0.7222 |
| PRTB|Centro-Oeste | 4 | 0.7500 |
| PRTB|Sul | 9 | 0.7778 |
| PC do B|Nordeste | 41 | 0.7805 |
| PRTB|Nordeste | 14 | 0.7857 |
| UNIÃO|Centro-Oeste | 239 | 0.7866 |
| AVANTE|Sudeste | 147 | 0.7959 |

## Cobertura conformal (CQR — Conformalized Quantile Regression)

**Cobertura observada (test):** 0.910 | **q̂ CQR:** +0.0006

> CQR usa 2 LGBMs quantile pra modelar `[q_low(x), q_hi(x)]` diretamente, depois conformaliza a margem. Intervalos adaptativos: largura cresce onde o modelo prevê dispersão maior.

**Cobertura por decil de pred — CQR:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 1505.0000 | 0.0001 | 0.0255 | 0.9927 |
| 1.0000 | 1505.0000 | 0.0255 | 0.0915 | 0.9661 |
| 2.0000 | 1504.0000 | 0.0916 | 0.2193 | 0.9016 |
| 3.0000 | 1505.0000 | 0.2194 | 0.3205 | 0.8837 |
| 4.0000 | 1505.0000 | 0.3205 | 0.3938 | 0.8724 |
| 5.0000 | 1504.0000 | 0.3938 | 0.4527 | 0.8763 |
| 6.0000 | 1505.0000 | 0.4527 | 0.5040 | 0.8950 |
| 7.0000 | 1525.0000 | 0.5052 | 0.5563 | 0.9082 |
| 8.0000 | 1494.0000 | 0.5589 | 0.6704 | 0.8855 |
| 9.0000 | 1495.0000 | 0.6705 | 1.0000 | 0.9177 |

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| swing_share_1t | 1547655.1939 |
| lag_share_1t | 820654.0036 |
| log_eleitorado | 133895.8061 |
| share_dep_federal_partido | 102007.7966 |
| sigla_partido | 79274.4707 |
| lag_share_1t_sucessao | 74855.4578 |
| sigla_uf | 58700.4810 |
| share_prefeito_local | 24126.4344 |
| volatilidade_partido | 18088.0819 |
| margem_prefeito | 16519.3072 |
| alinhado_gov_vigente_partido | 14765.2696 |
| porte | 9202.2677 |
| lag2_share_1t | 4940.2258 |
| regiao | 1851.2526 |
| alinhado_prefeito_partido | 1307.9467 |

## Notas

- Target modelado em `logit(share)`, predição destransformada com sigmoid.
- LightGBM com `objective=regression_l1` (MAE) — robusto a caudas.
- Eixo `ano_municipal`: prefeito vigente no momento da próxima eleição = vencedor da eleição municipal anterior (X-4).
- Sem features de governador concorrente (não há eleição estadual no ano municipal); apenas `alinhado_gov_vigente_*`.
- Universo de partidos no painel municipal é maior (especialmente em eleições proporcionais locais), e a granularidade do `numero_candidato` é local — diferente do número nacional do candidato presidencial.
- Versão `_iso` corrige saturação no top decil via regressão isotônica treinada com predições out-of-fold (leave-one-year-out) no conjunto de treino. Não toca o LGBM.

## Próximos passos

- Replicar `--calibrate` no eixo presidencial (`scripts/04_train.py`).
- Investigar comparativo com Fase 4: o lag local (`lag_share_1t`) é mais ruidoso aqui — partidos lançam candidatos diferentes em cada eleição municipal — por isso B1 deve performar pior que no presidencial.
- Avaliar uso da feature `continuidade` como driver: por hipótese, é ainda mais predictive aqui que no presidencial.