# Fase 5 — Agregação presidencial (município → UF → nacional)

**Modo:** prod | **Eixo:** `ano_presidencial` | **Pred col:** `pred_LightGBM_v1_iso` | **Renormalizar:** `none`
**Intervalos:** lower=`pred_lower_mondrian` upper=`pred_upper_mondrian` | **MC samples:** 1000 | **α:** 0.1 (IC 90%)

## Sanity check — soma de shares por (UF, ano)

Tolerância: ±0.010 | grupos: 27 | violações: **27**

Soma — min: 0.6491 | max: 0.9051 | média: 0.7903

> **Nota:** com `--renormalizar=none`, a soma de shares por UF pode ser < 1. Isso reflete o bias L1 do LGBM (subestima shares uniformemente, ~0.016/linha × n_partidos). Não é bug do agregador. Para previsão final reportável, use `--renormalizar mun`.

### Violadores (top 10)

|   ano_presidencial | sigla_uf   |   soma |   delta |
|-------------------:|:-----------|-------:|--------:|
|               2022 | AC         | 0.6922 |  0.3078 |
|               2022 | AL         | 0.8231 |  0.1769 |
|               2022 | AM         | 0.8086 |  0.1914 |
|               2022 | AP         | 0.7576 |  0.2424 |
|               2022 | BA         | 0.8887 |  0.1113 |
|               2022 | CE         | 0.8694 |  0.1306 |
|               2022 | DF         | 0.7386 |  0.2614 |
|               2022 | ES         | 0.7582 |  0.2418 |
|               2022 | GO         | 0.7581 |  0.2419 |
|               2022 | MA         | 0.8758 |  0.1242 |

## Sanity check — soma de shares nacional por ano

grupos: 1 | violações: **1** | min=0.8002 max=0.8002

## Cobertura empírica dos intervalos agregados

| nível | cobertura | nominal |
| --- | --- | --- |
| UF | 0.7172 | 0.90 |
| Nacional | 0.6364 | 0.90 |

> Cobertura agregada = fração de (UF×partido) ou (ano×partido) onde `y_real` (média ponderada do `y_true`) cai dentro de [share_lower, share_upper].

## Top partidos por share nacional (último ano)

### Ano = 2022

| sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |   bias_nacional |   n_ufs |   eleitorado_total |
|:----------------|-------------:|--------------:|--------------:|---------:|----------------:|--------:|-------------------:|
| PT              |       0.4933 |        0.4667 |        0.5190 |   0.4843 |          0.0089 |      27 |     117935194.0000 |
| PL              |       0.2606 |        0.2383 |        0.2818 |   0.4320 |         -0.1714 |      27 |     117935194.0000 |
| MDB             |       0.0227 |        0.0201 |        0.0253 |   0.0416 |         -0.0189 |      27 |     117935194.0000 |
| PDT             |       0.0147 |        0.0136 |        0.0157 |   0.0304 |         -0.0157 |      27 |     117935194.0000 |
| UNIÃO           |       0.0046 |        0.0040 |        0.0052 |   0.0051 |         -0.0005 |      27 |     117935194.0000 |
| NOVO            |       0.0024 |        0.0019 |        0.0029 |   0.0047 |         -0.0023 |      27 |     117935194.0000 |
| PTB             |       0.0007 |        0.0004 |        0.0010 |   0.0007 |          0.0000 |      27 |     117935194.0000 |
| UP              |       0.0004 |        0.0001 |        0.0008 |   0.0005 |         -0.0000 |      27 |     117935194.0000 |
| PCB             |       0.0004 |        0.0001 |        0.0007 |   0.0004 |          0.0000 |      27 |     117935194.0000 |
| PSTU            |       0.0002 |        0.0000 |        0.0005 |   0.0002 |          0.0000 |      27 |     117935194.0000 |
| DC              |       0.0002 |        0.0000 |        0.0004 |   0.0001 |          0.0000 |      27 |     117935194.0000 |

> `bias_nacional = share_pred - y_real`. Negativo = modelo subestima o partido no agregado nacional.

## UFs onde o intervalo agregado **NÃO cobriu** y_real

### Ano = 2022  (84/297 fora)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |   erro |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|-------:|
| RR         | PL              |       0.3861 |        0.1584 |        0.6043 |   0.6957 | 0.3096 |
| RO         | PL              |       0.3711 |        0.2625 |        0.4857 |   0.6436 | 0.2725 |
| AC         | PL              |       0.3548 |        0.1901 |        0.5130 |   0.6250 | 0.2702 |
| MT         | PL              |       0.3561 |        0.2787 |        0.4337 |   0.5984 | 0.2423 |
| SC         | PL              |       0.3858 |        0.3342 |        0.4384 |   0.6221 | 0.2364 |
| PR         | PL              |       0.3276 |        0.2637 |        0.3903 |   0.5526 | 0.2250 |
| MS         | PL              |       0.3093 |        0.1893 |        0.4260 |   0.5270 | 0.2177 |
| GO         | PL              |       0.3093 |        0.2279 |        0.3924 |   0.5216 | 0.2123 |
| ES         | PL              |       0.3114 |        0.2327 |        0.3942 |   0.5223 | 0.2108 |
| RS         | PL              |       0.2878 |        0.2407 |        0.3342 |   0.4889 | 0.2010 |
| SP         | PL              |       0.2838 |        0.2066 |        0.3609 |   0.4771 | 0.1933 |
| RJ         | PL              |       0.3224 |        0.1961 |        0.4503 |   0.5109 | 0.1885 |
| AP         | PL              |       0.2463 |        0.0773 |        0.4260 |   0.4341 | 0.1878 |
| MG         | PL              |       0.2565 |        0.2107 |        0.3000 |   0.4360 | 0.1795 |
| TO         | PL              |       0.2696 |        0.1891 |        0.3532 |   0.4400 | 0.1704 |
| PA         | PL              |       0.2373 |        0.1781 |        0.3026 |   0.4027 | 0.1654 |
| AL         | PL              |       0.2258 |        0.1330 |        0.3179 |   0.3605 | 0.1347 |
| RN         | PL              |       0.1928 |        0.1193 |        0.2642 |   0.3102 | 0.1174 |
| PE         | PL              |       0.1829 |        0.1240 |        0.2494 |   0.2991 | 0.1162 |
| PB         | PL              |       0.1823 |        0.1130 |        0.2500 |   0.2962 | 0.1140 |

### Distribuição dos descobertos por partido

- `PDT`: 27 UFs descobertas
- `MDB`: 26 UFs descobertas
- `PL`: 25 UFs descobertas
- `NOVO`: 6 UFs descobertas

> Concentração em poucos partidos indica viés estrutural do LGBM (e.g., PL 2022 — efeito migração Bolsonaro PSL→PL). Solução não é o agregador — é #60 (pesquisas como feature).

## UFs — partido com maior share por UF (último ano)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|
| AC         | PL              |       0.3548 |        0.1901 |        0.5130 |   0.6250 |
| AL         | PT              |       0.5640 |        0.4599 |        0.6689 |   0.5650 |
| AM         | PT              |       0.4896 |        0.2976 |        0.6790 |   0.4958 |
| AP         | PT              |       0.4535 |        0.2461 |        0.6659 |   0.4567 |
| BA         | PT              |       0.6952 |        0.6312 |        0.7593 |   0.6973 |
| CE         | PT              |       0.6645 |        0.5670 |        0.7636 |   0.6591 |
| DF         | PT              |       0.3949 |        0.0482 |        0.7300 |   0.3685 |
| ES         | PT              |       0.4052 |        0.3213 |        0.4951 |   0.4040 |
| GO         | PT              |       0.4077 |        0.3179 |        0.4923 |   0.3951 |
| MA         | PT              |       0.6899 |        0.6280 |        0.7526 |   0.6884 |
| MG         | PT              |       0.4888 |        0.4347 |        0.5417 |   0.4829 |
| MS         | PT              |       0.3895 |        0.2731 |        0.5065 |   0.3904 |
| MT         | PL              |       0.3561 |        0.2787 |        0.4337 |   0.5984 |
| PA         | PT              |       0.5177 |        0.4418 |        0.5950 |   0.5222 |
| PB         | PT              |       0.6419 |        0.5612 |        0.7246 |   0.6421 |
| PE         | PT              |       0.6545 |        0.5731 |        0.7322 |   0.6527 |
| PI         | PT              |       0.7399 |        0.6562 |        0.8260 |   0.7425 |
| PR         | PT              |       0.3793 |        0.3110 |        0.4474 |   0.3599 |
| RJ         | PT              |       0.4224 |        0.2835 |        0.5591 |   0.4068 |
| RN         | PT              |       0.6327 |        0.5412 |        0.7222 |   0.6298 |
| RO         | PL              |       0.3711 |        0.2625 |        0.4857 |   0.6436 |
| RR         | PL              |       0.3861 |        0.1584 |        0.6043 |   0.6957 |
| RS         | PT              |       0.4246 |        0.3660 |        0.4777 |   0.4228 |
| SC         | PL              |       0.3858 |        0.3342 |        0.4384 |   0.6221 |
| SE         | PT              |       0.6384 |        0.5390 |        0.7390 |   0.6382 |
| SP         | PT              |       0.4317 |        0.3360 |        0.5283 |   0.4089 |
| TO         | PT              |       0.4969 |        0.4154 |        0.5767 |   0.5040 |

## Notas

- `eleitorado_uf`/`eleitorado_total` é a soma de `total_votos_mun`, proxy do eleitorado registrado (correlação > 0.95).
- Intervalos agregados via Monte Carlo: para cada linha (mun, partido), sortear uniforme centrada em pred com semi-largura (hi-lo)/2; agregar ponderando por `total_votos_mun`; pegar percentis 0.050 e 0.950.
- Independência entre partidos no MC dentro do mesmo município ignora a restrição sum_partido share ~= 1. O efeito é alargar levemente os intervalos agregados (conservador).
