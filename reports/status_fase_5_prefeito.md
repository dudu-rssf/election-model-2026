# Fase 5 — Agregação prefeito (município → UF → nacional)

**Modo:** prod | **Eixo:** `ano_municipal` | **Pred col:** `pred_LightGBM_prefeito_v1_iso_renorm` | **Renormalizar:** `mun`
**Intervalos:** lower=`pred_lower_cqr_renorm` upper=`pred_upper_cqr_renorm` | **MC samples:** 1000 | **α:** 0.1 (IC 90%)

## Sanity check — soma de shares por (UF, ano)

Tolerância: ±0.010 | grupos: 26 | violações: **0**

Soma — min: 1.0000 | max: 1.0000 | média: 1.0000

## Sanity check — soma de shares nacional por ano

grupos: 1 | violações: **0** | min=1.0000 max=1.0000

## Cobertura empírica dos intervalos agregados

| nível | cobertura | nominal |
| --- | --- | --- |
| UF | 0.9734 | 0.90 |
| Nacional | 1.0000 | 0.90 |

> Cobertura agregada = fração de (UF×partido) ou (ano×partido) onde `y_real` (média ponderada do `y_true`) cai dentro de [share_lower, share_upper].

## Top partidos por share nacional (último ano)

### Ano = 2024

| sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |   bias_nacional |   n_ufs |   eleitorado_total |
|:----------------|-------------:|--------------:|--------------:|---------:|----------------:|--------:|-------------------:|
| PL              |       0.1385 |        0.1370 |        0.1401 |   0.1379 |          0.0006 |      26 |     113150500.0000 |
| PSD             |       0.1272 |        0.1250 |        0.1296 |   0.1274 |         -0.0002 |      25 |     113150500.0000 |
| MDB             |       0.1264 |        0.1249 |        0.1279 |   0.1271 |         -0.0007 |      26 |     113150500.0000 |
| UNIÃO           |       0.0998 |        0.0977 |        0.1018 |   0.0995 |          0.0003 |      26 |     113150500.0000 |
| PP              |       0.0877 |        0.0866 |        0.0889 |   0.0877 |          0.0000 |      26 |     113150500.0000 |
| PT              |       0.0781 |        0.0770 |        0.0793 |   0.0784 |         -0.0003 |      25 |     113150500.0000 |
| REPUBLICANOS    |       0.0645 |        0.0636 |        0.0656 |   0.0649 |         -0.0003 |      26 |     113150500.0000 |
| PSB             |       0.0571 |        0.0537 |        0.0603 |   0.0576 |         -0.0005 |      24 |     113150500.0000 |
| PSDB            |       0.0422 |        0.0367 |        0.0477 |   0.0412 |          0.0010 |      25 |     113150500.0000 |
| PODE            |       0.0308 |        0.0300 |        0.0316 |   0.0308 |          0.0000 |      26 |     113150500.0000 |
| PDT             |       0.0278 |        0.0269 |        0.0287 |   0.0279 |         -0.0001 |      26 |     113150500.0000 |
| PSOL            |       0.0234 |        0.0223 |        0.0246 |   0.0228 |          0.0005 |      25 |     113150500.0000 |
| AVANTE          |       0.0187 |        0.0185 |        0.0189 |   0.0188 |         -0.0001 |      21 |     113150500.0000 |
| PRTB            |       0.0158 |        0.0149 |        0.0168 |   0.0161 |         -0.0003 |      16 |     113150500.0000 |
| NOVO            |       0.0135 |        0.0118 |        0.0153 |   0.0137 |         -0.0002 |      25 |     113150500.0000 |

> `bias_nacional = share_pred - y_real`. Negativo = modelo subestima o partido no agregado nacional.

## UFs onde o intervalo agregado **NÃO cobriu** y_real

### Ano = 2024  (16/602 fora)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |    erro |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|--------:|
| AM         | CIDADANIA       |       0.1085 |        0.1062 |        0.1108 |   0.1014 | -0.0072 |
| AP         | PSOL            |       0.0594 |        0.0546 |        0.0641 |   0.0534 | -0.0061 |
| GO         | PL              |       0.2237 |        0.2189 |        0.2286 |   0.2183 | -0.0054 |
| RO         | MDB             |       0.0737 |        0.0700 |        0.0773 |   0.0786 |  0.0049 |
| RO         | UNIÃO           |       0.3166 |        0.3118 |        0.3212 |   0.3118 | -0.0047 |
| PR         | CIDADANIA       |       0.0159 |        0.0144 |        0.0174 |   0.0138 | -0.0020 |
| BA         | PSOL            |       0.0156 |        0.0149 |        0.0163 |   0.0167 |  0.0011 |
| AM         | AVANTE          |       0.1815 |        0.1807 |        0.1822 |   0.1806 | -0.0009 |
| SP         | UNIÃO           |       0.0488 |        0.0482 |        0.0493 |   0.0480 | -0.0007 |
| MT         | PDT             |       0.0013 |        0.0007 |        0.0019 |   0.0007 | -0.0006 |
| TO         | PDT             |       0.0420 |        0.0417 |        0.0423 |   0.0416 | -0.0004 |
| MA         | PRD             |       0.0272 |        0.0269 |        0.0275 |   0.0276 |  0.0003 |
| AC         | PC do B         |       0.0024 |        0.0024 |        0.0025 |   0.0027 |  0.0002 |
| MT         | DC              |       0.0007 |        0.0007 |        0.0007 |   0.0008 |  0.0000 |
| TO         | DC              |       0.0008 |        0.0008 |        0.0008 |   0.0009 |  0.0000 |
| TO         | PV              |       0.0036 |        0.0036 |        0.0037 |   0.0037 |  0.0000 |

### Distribuição dos descobertos por partido

- `CIDADANIA`: 2 UFs descobertas
- `DC`: 2 UFs descobertas
- `PDT`: 2 UFs descobertas
- `PSOL`: 2 UFs descobertas
- `UNIÃO`: 2 UFs descobertas
- `AVANTE`: 1 UFs descobertas
- `MDB`: 1 UFs descobertas
- `PC do B`: 1 UFs descobertas
- `PL`: 1 UFs descobertas
- `PRD`: 1 UFs descobertas

> Concentração em poucos partidos indica viés estrutural do LGBM (e.g., PL 2022 — efeito migração Bolsonaro PSL→PL). Solução não é o agregador — é #60 (pesquisas como feature).

## UFs — partido com maior share por UF (último ano)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|
| AC         | PP              |       0.2605 |        0.2512 |        0.2690 |   0.2605 |
| AL         | MDB             |       0.4109 |        0.3851 |        0.4353 |   0.4136 |
| AM         | UNIÃO           |       0.2253 |        0.2197 |        0.2307 |   0.2305 |
| AP         | MDB             |       0.4872 |        0.3648 |        0.6129 |   0.4957 |
| BA         | UNIÃO           |       0.2455 |        0.2218 |        0.2690 |   0.2448 |
| CE         | PT              |       0.2506 |        0.2407 |        0.2601 |   0.2558 |
| ES         | PODE            |       0.1958 |        0.1729 |        0.2174 |   0.1961 |
| GO         | UNIÃO           |       0.2788 |        0.2709 |        0.2864 |   0.2762 |
| MA         | PSB             |       0.1362 |        0.1342 |        0.1383 |   0.1346 |
| MG         | PSD             |       0.1540 |        0.1461 |        0.1617 |   0.1551 |
| MS         | PSDB            |       0.3788 |        0.3681 |        0.3904 |   0.3805 |
| MT         | UNIÃO           |       0.3103 |        0.3039 |        0.3162 |   0.3116 |
| PA         | MDB             |       0.4376 |        0.4217 |        0.4531 |   0.4356 |
| PB         | PSB             |       0.2341 |        0.2305 |        0.2373 |   0.2344 |
| PE         | PSB             |       0.2542 |        0.2304 |        0.2778 |   0.2567 |
| PI         | PT              |       0.3115 |        0.2982 |        0.3250 |   0.3091 |
| PR         | PSD             |       0.3195 |        0.3142 |        0.3246 |   0.3179 |
| RJ         | PL              |       0.2814 |        0.2674 |        0.2954 |   0.2802 |
| RN         | UNIÃO           |       0.2288 |        0.2106 |        0.2476 |   0.2291 |
| RO         | UNIÃO           |       0.3166 |        0.3118 |        0.3212 |   0.3118 |
| RR         | MDB             |       0.4338 |        0.3542 |        0.5138 |   0.4356 |
| RS         | MDB             |       0.2060 |        0.1960 |        0.2153 |   0.2061 |
| SC         | PL              |       0.2951 |        0.2904 |        0.2997 |   0.2940 |
| SE         | PSD             |       0.2621 |        0.2551 |        0.2692 |   0.2611 |
| SP         | PL              |       0.1465 |        0.1437 |        0.1496 |   0.1459 |
| TO         | UNIÃO           |       0.2790 |        0.2669 |        0.2909 |   0.2784 |

## Notas

- `eleitorado_uf`/`eleitorado_total` é a soma de `total_votos_mun`, proxy do eleitorado registrado (correlação > 0.95).
- Intervalos agregados via Monte Carlo: para cada linha (mun, partido), sortear uniforme centrada em pred com semi-largura (hi-lo)/2; agregar ponderando por `total_votos_mun`; pegar percentis 0.050 e 0.950.
- Independência entre partidos no MC dentro do mesmo município ignora a restrição sum_partido share ~= 1. O efeito é alargar levemente os intervalos agregados (conservador).
- `--renormalizar mun` está ativo: predições foram divididas por sum_p pred[m,p] em cada município. Os intervalos foram reescalados pelo mesmo fator (preserva forma relativa).
