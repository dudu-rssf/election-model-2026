# Fase 4 — status: primeiro modelo presidencial

**Modo:** dev | **UFs:** ['SP'] | **Máx municípios:** 100
**Split temporal:** treino = [2014, 2018] (2330 linhas) | teste = 2022 (1100 linhas)

## Comparativo geral (escala share ∈ [0,1])

| modelo | n | mae | rmse | bias |
| --- | --- | --- | --- | --- |
| B0_mediana_partido_uf | 1100 | 0.0751 | 0.1715 | 0.0587 |
| B1_lag_share | 1100 | 0.0782 | 0.1757 | 0.0643 |
| B2_blend | 1100 | 0.0764 | 0.1730 | 0.0615 |
| LightGBM_v1 | 1100 | 0.0565 | 0.1585 | 0.0507 |

> `bias` positivo = modelo subestima (predição < realidade).
> `mae` e `rmse` quanto menor, melhor.

## MAE por partido (top 10 piores)

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| PL | 100 | 0.5126 | 0.5196 | 0.5126 | 0.5288 |
| PT | 100 | 0.0589 | 0.0729 | 0.0493 | 0.3928 |
| MDB | 100 | 0.0209 | 0.0231 | 0.0209 | 0.0473 |
| PDT | 100 | 0.0156 | 0.0178 | -0.0149 | 0.0219 |
| NOVO | 100 | 0.0107 | 0.0129 | -0.0107 | 0.0034 |
| UNIÃO | 100 | 0.0012 | 0.0014 | 0.0011 | 0.0043 |
| PTB | 100 | 0.0005 | 0.0005 | -0.0004 | 0.0006 |
| UP | 100 | 0.0003 | 0.0004 | -0.0003 | 0.0003 |
| DC | 100 | 0.0002 | 0.0002 | -0.0002 | 0.0001 |
| PSTU | 100 | 0.0002 | 0.0002 | -0.0001 | 0.0002 |

## MAE por UF

| grupo | n | mae | rmse | bias | share_medio |
| --- | --- | --- | --- | --- | --- |
| SP | 1100 | 0.0565 | 0.1585 | 0.0507 | 0.0909 |

## Calibração por decil (LightGBM)

| decil | n | pred_medio | real_medio | erro_decil |
| --- | --- | --- | --- | --- |
| (-0.000814, 0.000272] | 110 | 0.0002 | 0.0000 | -0.0002 |
| (0.000272, 0.000379] | 110 | 0.0003 | 0.0002 | -0.0001 |
| (0.000379, 0.000599] | 110 | 0.0005 | 0.0002 | -0.0002 |
| (0.000599, 0.000991] | 110 | 0.0008 | 0.0006 | -0.0002 |
| (0.000991, 0.00275] | 110 | 0.0017 | 0.0018 | 0.0001 |
| (0.00275, 0.0124] | 110 | 0.0059 | 0.0208 | 0.0149 |
| (0.0124, 0.0196] | 110 | 0.0158 | 0.3760 | 0.3602 |
| (0.0196, 0.0287] | 110 | 0.0243 | 0.1187 | 0.0944 |
| (0.0287, 0.0455] | 110 | 0.0358 | 0.0312 | -0.0047 |
| (0.0455, 0.52] | 110 | 0.3172 | 0.3596 | 0.0423 |

> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).

## Top feature importance (LightGBM, gain)

| feature | importance |
| --- | --- |
| sigla_partido | 39967.6581 |
| swing_share_1t | 27148.9877 |
| share_dep_federal_partido | 26634.8730 |
| log_eleitorado | 19585.6153 |
| margem_prefeito | 15259.4538 |
| share_prefeito_local | 15019.7558 |
| lag_share_1t | 6990.6036 |
| porte | 1648.4731 |
| continuidade_classe | 930.4284 |
| alinhado_gov_vigente_partido | 577.2614 |
| alinhado_gov_vigente_coligacao | 562.0017 |
| primeiro_mandato_prefeito | 363.5473 |
| alinhado_prefeito_partido | 181.8103 |
| indice_continuidade | 33.1937 |
| alinhado_gov_concorrente_partido | 16.8624 |

## Notas

- Target modelado em `logit(share)`, predição destransformada com sigmoid.
- LightGBM com `objective=regression_l1` (MAE) — robusto a caudas.
- Features categóricas (sigla_uf, regiao, porte, continuidade_classe, sigla_partido) tratadas nativamente pelo LightGBM.
- Amostra dev = 1 UF × 100 municípios × 3 anos × ~10 partidos = ~3 mil linhas. Signal-to-noise é limitado — resultados aqui servem pra validar pipeline, não pra conclusões sobre 2026.

## Próximos passos (Fase 4.5 / Fase 5)

- **Fase 4.5**: replicar pipeline pra prefeito (target municipal, eixo temporal 2012/2016/2020).
- **Fase 5**: quantificação de incerteza (conformal prediction ou bootstrap). Fase 4 entrega só estimativa pontual; o produto final precisa de intervalo.
- **Investigar**: se o LightGBM não bate B1 (`lag_share_1t`) de forma contundente, reavaliar features históricas — pode ser que precisemos de mais anos no treino (rodar em prod).