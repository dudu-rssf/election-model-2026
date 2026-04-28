# Fase 4.5 — status: primeiro modelo de prefeito

**Modo:** dev | **UFs:** ['SP'] | **Máx municípios:** 100
**Eixo:** `ano_municipal` | **Split temporal:** treino = [2012, 2016, 2020] (1000 linhas) | teste = 2024 (291 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 291 | 0.2012 | 0.2545 | 0.0625 |
| B1_lag_share | 291 | 0.2871 | 0.3729 | 0.2604 |
| B2_blend | 291 | 0.2234 | 0.2903 | 0.1614 |
| LightGBM_prefeito_v1 | 291 | 0.0279 | 0.0623 | 0.0169 |
| LightGBM_prefeito_v1_iso | 291 | 0.0282 | 0.0569 | 0.0024 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PSD | 67 | 0.0546 | 0.1021 | 0.0388 | 0.4698 |
| MDB | 25 | 0.0470 | 0.0855 | 0.0397 | 0.4646 |
| REPUBLICANOS | 26 | 0.0308 | 0.0686 | 0.0222 | 0.4828 |
| PSDB | 10 | 0.0249 | 0.0322 | 0.0029 | 0.3052 |
| PL | 44 | 0.0247 | 0.0403 | 0.0060 | 0.3895 |
| PT | 21 | 0.0159 | 0.0339 | 0.0114 | 0.0897 |
| PP | 14 | 0.0152 | 0.0208 | 0.0067 | 0.3880 |
| PODE | 17 | 0.0127 | 0.0205 | 0.0043 | 0.3008 |
| UNIÃO | 20 | 0.0068 | 0.0092 | -0.0027 | 0.3362 |
| PSB | 11 | 0.0063 | 0.0080 | 0.0023 | 0.1738 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| SP | 291 | 0.0279 | 0.0623 | 0.0169 | 0.3436 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (0.00465, 0.0193] | 30 | 0.0140 | 0.0151 | 0.0011 |
| (0.0193, 0.0638] | 29 | 0.0390 | 0.0410 | 0.0020 |
| (0.0638, 0.157] | 29 | 0.1106 | 0.1133 | 0.0027 |
| (0.157, 0.27] | 29 | 0.2174 | 0.2205 | 0.0030 |
| (0.27, 0.356] | 29 | 0.3143 | 0.3204 | 0.0060 |
| (0.356, 0.409] | 29 | 0.3829 | 0.3812 | -0.0017 |
| (0.409, 0.465] | 29 | 0.4425 | 0.4414 | -0.0010 |
| (0.465, 0.526] | 29 | 0.4967 | 0.5267 | 0.0300 |
| (0.526, 0.595] | 29 | 0.5623 | 0.5951 | 0.0327 |
| (0.595, 0.742] | 29 | 0.6983 | 0.7930 | 0.0947 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Calibração por decil (LightGBM + isotonic)

**Modo:** holdout (ano_calib=2020) | **Cobertura:** asimétrico (raw quando pred < 0.50)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (0.00465, 0.0193] | 30 | 0.0140 | 0.0151 | 0.0011 |
| (0.0193, 0.0638] | 29 | 0.0390 | 0.0410 | 0.0020 |
| (0.0638, 0.157] | 29 | 0.1106 | 0.1133 | 0.0027 |
| (0.157, 0.27] | 29 | 0.2174 | 0.2205 | 0.0030 |
| (0.27, 0.356] | 29 | 0.3143 | 0.3204 | 0.0060 |
| (0.356, 0.409] | 29 | 0.3829 | 0.3812 | -0.0017 |
| (0.409, 0.465] | 29 | 0.4425 | 0.4414 | -0.0010 |
| (0.465, 0.536] | 31 | 0.5016 | 0.5273 | 0.0257 |
| (0.536, 0.71] | 31 | 0.6157 | 0.6188 | 0.0031 |
| (0.71, 0.822] | 25 | 0.8217 | 0.8000 | -0.0216 |

> Calibrador isotônico ajustado em predições do modelo no treino. Aplicado pós-hoc na predição do teste.

## MAE por partido — versão calibrada (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PSD | 67 | 0.0507 | 0.0842 | 0.0098 | 0.4698 |
| MDB | 25 | 0.0388 | 0.0766 | 0.0140 | 0.4646 |
| REPUBLICANOS | 26 | 0.0365 | 0.0696 | 0.0057 | 0.4828 |
| PL | 44 | 0.0308 | 0.0527 | -0.0125 | 0.3895 |
| PSDB | 10 | 0.0253 | 0.0322 | 0.0018 | 0.3052 |
| PT | 21 | 0.0153 | 0.0315 | 0.0108 | 0.0897 |
| PODE | 17 | 0.0130 | 0.0213 | 0.0021 | 0.3008 |
| PP | 14 | 0.0129 | 0.0165 | 0.0034 | 0.3880 |
| UNIÃO | 20 | 0.0124 | 0.0281 | -0.0104 | 0.3362 |
| PSB | 11 | 0.0063 | 0.0080 | 0.0023 | 0.1738 |

## Cobertura conformal (split)

**Cobertura nominal:** 90% (α = 0.10) | **q̂ split:** 0.0986 | **Cobertura observada (test):** 0.900

**Cobertura por decil de pred — split:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 30.0000 | 0.0057 | 0.0193 | 1.0000 |
| 1.0000 | 29.0000 | 0.0205 | 0.0638 | 1.0000 |
| 2.0000 | 29.0000 | 0.0641 | 0.1565 | 1.0000 |
| 3.0000 | 29.0000 | 0.1590 | 0.2704 | 1.0000 |
| 4.0000 | 29.0000 | 0.2736 | 0.3556 | 0.9655 |
| 5.0000 | 29.0000 | 0.3576 | 0.4087 | 1.0000 |
| 6.0000 | 29.0000 | 0.4130 | 0.4647 | 1.0000 |
| 7.0000 | 31.0000 | 0.4654 | 0.5365 | 0.9032 |
| 8.0000 | 31.0000 | 0.5377 | 0.7104 | 0.7419 |
| 9.0000 | 25.0000 | 0.8217 | 0.8217 | 0.3200 |

> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é homogêneo, cobertura por decil pode divergir do nominal (intervalo grande demais nos baixos, pequeno demais nos altos).

## Cobertura conformal (Mondrian — estratificado por bin de pred)

**Cobertura observada (test):** 0.959 | **q̂ por bin:** [0.009, 0.158, 0.147, 0.085, 0.181]

**Cobertura por decil de pred — Mondrian:**

| decil | n | pred_min | pred_max | cobertura |
| --- | --- | --- | --- | --- |
| 0.0000 | 30.0000 | 0.0057 | 0.0193 | 0.9333 |
| 1.0000 | 29.0000 | 0.0205 | 0.0638 | 0.8966 |
| 2.0000 | 29.0000 | 0.0641 | 0.1565 | 1.0000 |
| 3.0000 | 29.0000 | 0.1590 | 0.2704 | 1.0000 |
| 4.0000 | 29.0000 | 0.2736 | 0.3556 | 0.9655 |
| 5.0000 | 29.0000 | 0.3576 | 0.4087 | 1.0000 |
| 6.0000 | 29.0000 | 0.4130 | 0.4647 | 1.0000 |
| 7.0000 | 31.0000 | 0.4654 | 0.5365 | 0.9355 |
| 8.0000 | 31.0000 | 0.5377 | 0.7104 | 0.9677 |
| 9.0000 | 25.0000 | 0.8217 | 0.8217 | 0.8800 |

> Mondrian deve dar cobertura aproximadamente uniforme ao longo dos decis (cobertura condicional).

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| swing_share_1t | 18384.0835 |
| share_dep_federal_partido | 11795.9266 |
| log_eleitorado | 9204.5235 |
| lag_share_1t | 7306.4236 |
| margem_prefeito | 5351.5152 |
| share_prefeito_local | 5044.9094 |
| sigla_partido | 2804.2248 |
| volatilidade_partido | 1711.5458 |
| porte | 904.5156 |
| lag2_share_1t | 885.6275 |
| lag_share_1t_sucessao | 757.9168 |
| continuidade_classe | 353.5300 |
| alinhado_gov_vigente_partido | 249.9342 |
| alinhado_gov_vigente_coligacao | 142.8092 |
| primeiro_mandato_prefeito | 117.7829 |

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