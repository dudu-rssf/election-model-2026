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
| UF | 0.7374 | 0.90 |
| Nacional | 0.3636 | 0.90 |

> Cobertura agregada = fração de (UF×partido) ou (ano×partido) onde `y_real` (média ponderada do `y_true`) cai dentro de [share_lower, share_upper].

## Top partidos por share nacional (último ano)

### Ano = 2022

| sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |   bias_nacional |   n_ufs |   eleitorado_total |
|:----------------|-------------:|--------------:|--------------:|---------:|----------------:|--------:|-------------------:|
| PL              |       0.4937 |        0.4862 |        0.5007 |   0.4320 |          0.0617 |      27 |     117935194.0000 |
| PT              |       0.4508 |        0.4436 |        0.4579 |   0.4843 |         -0.0335 |      27 |     117935194.0000 |
| MDB             |       0.0345 |        0.0309 |        0.0379 |   0.0416 |         -0.0071 |      27 |     117935194.0000 |
| PDT             |       0.0125 |        0.0109 |        0.0140 |   0.0304 |         -0.0179 |      27 |     117935194.0000 |
| UNIÃO           |       0.0040 |        0.0035 |        0.0044 |   0.0051 |         -0.0011 |      27 |     117935194.0000 |
| NOVO            |       0.0031 |        0.0026 |        0.0035 |   0.0047 |         -0.0017 |      27 |     117935194.0000 |
| PTB             |       0.0005 |        0.0003 |        0.0007 |   0.0007 |         -0.0002 |      27 |     117935194.0000 |
| PCB             |       0.0004 |        0.0001 |        0.0006 |   0.0004 |         -0.0000 |      27 |     117935194.0000 |
| UP              |       0.0003 |        0.0001 |        0.0005 |   0.0005 |         -0.0001 |      27 |     117935194.0000 |
| PSTU            |       0.0002 |        0.0000 |        0.0004 |   0.0002 |          0.0000 |      27 |     117935194.0000 |
| DC              |       0.0001 |        0.0000 |        0.0003 |   0.0001 |          0.0000 |      27 |     117935194.0000 |

> `bias_nacional = share_pred - y_real`. Negativo = modelo subestima o partido no agregado nacional.

## UFs onde o intervalo agregado **NÃO cobriu** y_real

### Ano = 2022  (78/297 fora)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |    erro |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|--------:|
| CE         | PL              |       0.3476 |        0.3212 |        0.3743 |   0.2538 | -0.0938 |
| BA         | PL              |       0.3365 |        0.3186 |        0.3549 |   0.2431 | -0.0934 |
| SE         | PL              |       0.3811 |        0.3531 |        0.4083 |   0.2916 | -0.0895 |
| PI         | PL              |       0.2799 |        0.2556 |        0.3040 |   0.1990 | -0.0809 |
| RN         | PL              |       0.3879 |        0.3615 |        0.4132 |   0.3102 | -0.0777 |
| MA         | PL              |       0.3355 |        0.3181 |        0.3542 |   0.2602 | -0.0753 |
| PB         | PL              |       0.3704 |        0.3462 |        0.3937 |   0.2962 | -0.0741 |
| BA         | PT              |       0.6285 |        0.6104 |        0.6467 |   0.6973 |  0.0688 |
| PA         | PL              |       0.4685 |        0.4482 |        0.4909 |   0.4027 | -0.0658 |
| SE         | PT              |       0.5729 |        0.5460 |        0.6002 |   0.6382 |  0.0653 |
| PE         | PL              |       0.3614 |        0.3410 |        0.3843 |   0.2991 | -0.0623 |
| MG         | PL              |       0.4967 |        0.4814 |        0.5112 |   0.4360 | -0.0607 |
| SP         | PL              |       0.5362 |        0.5100 |        0.5624 |   0.4771 | -0.0591 |
| RJ         | PL              |       0.5697 |        0.5313 |        0.6087 |   0.5109 | -0.0588 |
| TO         | PL              |       0.4971 |        0.4723 |        0.5228 |   0.4400 | -0.0571 |
| RS         | PL              |       0.5456 |        0.5302 |        0.5613 |   0.4889 | -0.0567 |
| RN         | PT              |       0.5771 |        0.5516 |        0.6019 |   0.6298 |  0.0527 |
| PI         | PT              |       0.6899 |        0.6659 |        0.7140 |   0.7425 |  0.0526 |
| CE         | PDT             |       0.0173 |        0.0107 |        0.0242 |   0.0680 |  0.0507 |
| AL         | PL              |       0.4096 |        0.3778 |        0.4418 |   0.3605 | -0.0491 |

### Distribuição dos descobertos por partido

- `PDT`: 26 UFs descobertas
- `PL`: 21 UFs descobertas
- `PT`: 15 UFs descobertas
- `MDB`: 7 UFs descobertas
- `NOVO`: 5 UFs descobertas
- `UNIÃO`: 4 UFs descobertas

> Concentração em poucos partidos indica viés estrutural do LGBM (e.g., PL 2022 — efeito migração Bolsonaro PSL→PL). Solução não é o agregador — é #60 (pesquisas como feature).

## UFs — partido com maior share por UF (último ano)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|
| AC         | PL              |       0.6534 |        0.6013 |        0.7038 |   0.6250 |
| AL         | PT              |       0.5485 |        0.5167 |        0.5807 |   0.5650 |
| AM         | PT              |       0.4821 |        0.4281 |        0.5354 |   0.4958 |
| AP         | PL              |       0.4854 |        0.4264 |        0.5473 |   0.4341 |
| BA         | PT              |       0.6285 |        0.6104 |        0.6467 |   0.6973 |
| CE         | PT              |       0.6186 |        0.5916 |        0.6459 |   0.6591 |
| DF         | PL              |       0.5740 |        0.4693 |        0.6742 |   0.5165 |
| ES         | PL              |       0.5703 |        0.5455 |        0.5965 |   0.5223 |
| GO         | PL              |       0.5688 |        0.5437 |        0.5944 |   0.5216 |
| MA         | PT              |       0.6395 |        0.6220 |        0.6571 |   0.6884 |
| MG         | PL              |       0.4967 |        0.4814 |        0.5112 |   0.4360 |
| MS         | PL              |       0.5662 |        0.5293 |        0.6020 |   0.5270 |
| MT         | PL              |       0.6263 |        0.6030 |        0.6493 |   0.5984 |
| PA         | PT              |       0.4892 |        0.4677 |        0.5107 |   0.5222 |
| PB         | PT              |       0.5971 |        0.5742 |        0.6205 |   0.6421 |
| PE         | PT              |       0.6100 |        0.5874 |        0.6314 |   0.6527 |
| PI         | PT              |       0.6899 |        0.6659 |        0.7140 |   0.7425 |
| PR         | PL              |       0.5933 |        0.5732 |        0.6130 |   0.5526 |
| RJ         | PL              |       0.5697 |        0.5313 |        0.6087 |   0.5109 |
| RN         | PT              |       0.5771 |        0.5516 |        0.6019 |   0.6298 |
| RO         | PL              |       0.6729 |        0.6396 |        0.7083 |   0.6436 |
| RR         | PL              |       0.7183 |        0.6453 |        0.7889 |   0.6957 |
| RS         | PL              |       0.5456 |        0.5302 |        0.5613 |   0.4889 |
| SC         | PL              |       0.6702 |        0.6547 |        0.6860 |   0.6221 |
| SE         | PT              |       0.5729 |        0.5460 |        0.6002 |   0.6382 |
| SP         | PL              |       0.5362 |        0.5100 |        0.5624 |   0.4771 |
| TO         | PL              |       0.4971 |        0.4723 |        0.5228 |   0.4400 |

## Notas

- `eleitorado_uf`/`eleitorado_total` é a soma de `total_votos_mun`, proxy do eleitorado registrado (correlação > 0.95).
- Intervalos agregados via Monte Carlo: para cada linha (mun, partido), sortear uniforme centrada em pred com semi-largura (hi-lo)/2; agregar ponderando por `total_votos_mun`; pegar percentis 0.050 e 0.950.
- Independência entre partidos no MC dentro do mesmo município ignora a restrição sum_partido share ~= 1. O efeito é alargar levemente os intervalos agregados (conservador).
- `--renormalizar mun` está ativo: predições foram divididas por sum_p pred[m,p] em cada município. Os intervalos foram reescalados pelo mesmo fator (preserva forma relativa).
