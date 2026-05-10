# Fase 5 — Agregação presidencial (município → UF → nacional)

**Modo:** prod | **Eixo:** `ano_presidencial` | **Pred col:** `pred_LightGBM_v1_renorm` | **Renormalizar:** `mun`
**Intervalos:** lower=`pred_lower_mondrian_renorm` upper=`pred_upper_mondrian_renorm` | **MC samples:** 1000 | **α:** 0.1 (IC 90%)

## Sanity check — soma de shares por (UF, ano)

Tolerância: ±0.010 | grupos: 27 | violações: **0**

Soma — min: 1.0000 | max: 1.0000 | média: 1.0000

## Sanity check — soma de shares nacional por ano

grupos: 1 | violações: **0** | min=1.0000 max=1.0000

## Cobertura empírica dos intervalos agregados

| nível | cobertura | nominal |
| --- | --- | --- |
| UF | 0.7172 | 0.90 |
| Nacional | 0.5455 | 0.90 |

> Cobertura agregada = fração de (UF×partido) ou (ano×partido) onde `y_real` (média ponderada do `y_true`) cai dentro de [share_lower, share_upper].

## Top partidos por share nacional (último ano)

### Ano = 2022

| sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |   bias_nacional |   n_ufs |   eleitorado_total |
|:----------------|-------------:|--------------:|--------------:|---------:|----------------:|--------:|-------------------:|
| PL              |       0.4888 |        0.4824 |        0.4948 |   0.4320 |          0.0568 |      27 |     117935194.0000 |
| PT              |       0.4536 |        0.4474 |        0.4598 |   0.4843 |         -0.0307 |      27 |     117935194.0000 |
| MDB             |       0.0369 |        0.0335 |        0.0402 |   0.0416 |         -0.0047 |      27 |     117935194.0000 |
| PDT             |       0.0117 |        0.0105 |        0.0129 |   0.0304 |         -0.0187 |      27 |     117935194.0000 |
| UNIÃO           |       0.0048 |        0.0044 |        0.0052 |   0.0051 |         -0.0003 |      27 |     117935194.0000 |
| NOVO            |       0.0023 |        0.0020 |        0.0026 |   0.0047 |         -0.0024 |      27 |     117935194.0000 |
| PTB             |       0.0007 |        0.0005 |        0.0008 |   0.0007 |         -0.0000 |      27 |     117935194.0000 |
| UP              |       0.0004 |        0.0002 |        0.0006 |   0.0005 |         -0.0000 |      27 |     117935194.0000 |
| PCB             |       0.0004 |        0.0002 |        0.0006 |   0.0004 |         -0.0000 |      27 |     117935194.0000 |
| PSTU            |       0.0002 |        0.0001 |        0.0004 |   0.0002 |          0.0000 |      27 |     117935194.0000 |
| DC              |       0.0001 |        0.0000 |        0.0003 |   0.0001 |          0.0000 |      27 |     117935194.0000 |

> `bias_nacional = share_pred - y_real`. Negativo = modelo subestima o partido no agregado nacional.

## UFs onde o intervalo agregado **NÃO cobriu** y_real

### Ano = 2022  (84/297 fora)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |    erro |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|--------:|
| MA         | PL              |       0.3427 |        0.3281 |        0.3583 |   0.2602 | -0.0825 |
| PB         | PL              |       0.3784 |        0.3586 |        0.3979 |   0.2962 | -0.0822 |
| PI         | PL              |       0.2770 |        0.2563 |        0.2975 |   0.1990 | -0.0779 |
| SE         | PL              |       0.3681 |        0.3444 |        0.3916 |   0.2916 | -0.0765 |
| RN         | PL              |       0.3854 |        0.3639 |        0.4060 |   0.3102 | -0.0752 |
| RJ         | PL              |       0.5832 |        0.5502 |        0.6166 |   0.5109 | -0.0723 |
| AM         | PL              |       0.4988 |        0.4525 |        0.5431 |   0.4280 | -0.0709 |
| MG         | PL              |       0.4992 |        0.4864 |        0.5114 |   0.4360 | -0.0632 |
| PA         | PL              |       0.4654 |        0.4484 |        0.4837 |   0.4027 | -0.0627 |
| RS         | PL              |       0.5513 |        0.5379 |        0.5649 |   0.4889 | -0.0624 |
| AL         | PL              |       0.4218 |        0.3963 |        0.4480 |   0.3605 | -0.0614 |
| TO         | PL              |       0.5010 |        0.4802 |        0.5226 |   0.4400 | -0.0610 |
| CE         | PL              |       0.3134 |        0.2897 |        0.3374 |   0.2538 | -0.0597 |
| AP         | PL              |       0.4921 |        0.4435 |        0.5429 |   0.4341 | -0.0580 |
| MA         | PT              |       0.6316 |        0.6167 |        0.6464 |   0.6884 |  0.0569 |
| GO         | PL              |       0.5774 |        0.5559 |        0.5995 |   0.5216 | -0.0559 |
| PB         | PT              |       0.5880 |        0.5692 |        0.6075 |   0.6421 |  0.0540 |
| ES         | PL              |       0.5752 |        0.5548 |        0.5969 |   0.5223 | -0.0529 |
| SE         | PT              |       0.5855 |        0.5623 |        0.6091 |   0.6382 |  0.0528 |
| PI         | PT              |       0.6918 |        0.6714 |        0.7126 |   0.7425 |  0.0507 |

### Distribuição dos descobertos por partido

- `PDT`: 27 UFs descobertas
- `PL`: 24 UFs descobertas
- `PT`: 17 UFs descobertas
- `NOVO`: 10 UFs descobertas
- `MDB`: 6 UFs descobertas

> Concentração em poucos partidos indica viés estrutural do LGBM (e.g., PL 2022 — efeito migração Bolsonaro PSL→PL). Solução não é o agregador — é #60 (pesquisas como feature).

## UFs — partido com maior share por UF (último ano)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|
| AC         | PL              |       0.6559 |        0.6137 |        0.6967 |   0.6250 |
| AL         | PT              |       0.5367 |        0.5110 |        0.5629 |   0.5650 |
| AM         | PL              |       0.4988 |        0.4525 |        0.5431 |   0.4280 |
| AP         | PL              |       0.4921 |        0.4435 |        0.5429 |   0.4341 |
| BA         | PT              |       0.6712 |        0.6544 |        0.6881 |   0.6973 |
| CE         | PT              |       0.6443 |        0.6198 |        0.6689 |   0.6591 |
| DF         | PL              |       0.5775 |        0.4913 |        0.6601 |   0.5165 |
| ES         | PL              |       0.5752 |        0.5548 |        0.5969 |   0.5223 |
| GO         | PL              |       0.5774 |        0.5559 |        0.5995 |   0.5216 |
| MA         | PT              |       0.6316 |        0.6167 |        0.6464 |   0.6884 |
| MG         | PL              |       0.4992 |        0.4864 |        0.5114 |   0.4360 |
| MS         | PL              |       0.5703 |        0.5394 |        0.6002 |   0.5270 |
| MT         | PL              |       0.6292 |        0.6096 |        0.6486 |   0.5984 |
| PA         | PT              |       0.4902 |        0.4724 |        0.5080 |   0.5222 |
| PB         | PT              |       0.5880 |        0.5692 |        0.6075 |   0.6421 |
| PE         | PT              |       0.6226 |        0.6028 |        0.6417 |   0.6527 |
| PI         | PT              |       0.6918 |        0.6714 |        0.7126 |   0.7425 |
| PR         | PL              |       0.6028 |        0.5859 |        0.6193 |   0.5526 |
| RJ         | PL              |       0.5832 |        0.5502 |        0.6166 |   0.5109 |
| RN         | PT              |       0.5828 |        0.5617 |        0.6033 |   0.6298 |
| RO         | PL              |       0.6725 |        0.6448 |        0.7019 |   0.6436 |
| RR         | PL              |       0.7183 |        0.6587 |        0.7760 |   0.6957 |
| RS         | PL              |       0.5513 |        0.5379 |        0.5649 |   0.4889 |
| SC         | PL              |       0.6599 |        0.6470 |        0.6731 |   0.6221 |
| SE         | PT              |       0.5855 |        0.5623 |        0.6091 |   0.6382 |
| SP         | PL              |       0.5244 |        0.5015 |        0.5474 |   0.4771 |
| TO         | PL              |       0.5010 |        0.4802 |        0.5226 |   0.4400 |

## Notas

- `eleitorado_uf`/`eleitorado_total` é a soma de `total_votos_mun`, proxy do eleitorado registrado (correlação > 0.95).
- Intervalos agregados via Monte Carlo: para cada linha (mun, partido), sortear uniforme centrada em pred com semi-largura (hi-lo)/2; agregar ponderando por `total_votos_mun`; pegar percentis 0.050 e 0.950.
- Independência entre partidos no MC dentro do mesmo município ignora a restrição sum_partido share ~= 1. O efeito é alargar levemente os intervalos agregados (conservador).
- `--renormalizar mun` está ativo: predições foram divididas por sum_p pred[m,p] em cada município. Os intervalos foram reescalados pelo mesmo fator (preserva forma relativa).
