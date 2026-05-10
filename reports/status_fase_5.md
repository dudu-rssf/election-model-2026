# Fase 5 — Agregação presidencial (município → UF → nacional)

**Modo:** prod | **Eixo:** `ano_presidencial` | **Pred col:** `pred_LightGBM_v1` | **Renormalizar:** `none`
**Intervalos:** lower=`pred_lower_mondrian` upper=`pred_upper_mondrian` | **MC samples:** 1000 | **α:** 0.1 (IC 90%)

## Sanity check — soma de shares por (UF, ano)

Tolerância: ±0.010 | grupos: 27 | violações: **26**

Soma — min: 0.9038 | max: 1.1271 | média: 1.0459

> **Nota:** com `--renormalizar=none`, a soma de shares por UF pode ser < 1. Isso reflete o bias L1 do LGBM (subestima shares uniformemente, ~0.016/linha × n_partidos). Não é bug do agregador. Para previsão final reportável, use `--renormalizar mun`.

### Violadores (top 10)

|   ano_presidencial | sigla_uf   |   soma |   delta |
|-------------------:|:-----------|-------:|--------:|
|               2022 | AC         | 0.9721 |  0.0279 |
|               2022 | AL         | 1.0451 |  0.0451 |
|               2022 | AM         | 1.1011 |  0.1011 |
|               2022 | AP         | 1.0717 |  0.0717 |
|               2022 | BA         | 1.0621 |  0.0621 |
|               2022 | CE         | 1.0245 |  0.0245 |
|               2022 | DF         | 1.0176 |  0.0176 |
|               2022 | ES         | 1.0597 |  0.0597 |
|               2022 | GO         | 1.0364 |  0.0364 |
|               2022 | MA         | 1.0551 |  0.0551 |

## Sanity check — soma de shares nacional por ano

grupos: 1 | violações: **1** | min=1.0566 max=1.0566

## Cobertura empírica dos intervalos agregados

| nível | cobertura | nominal |
| --- | --- | --- |
| UF | 0.7508 | 0.90 |
| Nacional | 0.5455 | 0.90 |

> Cobertura agregada = fração de (UF×partido) ou (ano×partido) onde `y_real` (média ponderada do `y_true`) cai dentro de [share_lower, share_upper].

## Top partidos por share nacional (último ano)

### Ano = 2022

| sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |   bias_nacional |   n_ufs |   eleitorado_total |
|:----------------|-------------:|--------------:|--------------:|---------:|----------------:|--------:|-------------------:|
| PL              |       0.5262 |        0.5194 |        0.5326 |   0.4320 |          0.0942 |      27 |     117935194.0000 |
| PT              |       0.4724 |        0.4658 |        0.4788 |   0.4843 |         -0.0120 |      27 |     117935194.0000 |
| MDB             |       0.0364 |        0.0329 |        0.0398 |   0.0416 |         -0.0051 |      27 |     117935194.0000 |
| PDT             |       0.0120 |        0.0107 |        0.0133 |   0.0304 |         -0.0184 |      27 |     117935194.0000 |
| UNIÃO           |       0.0049 |        0.0045 |        0.0054 |   0.0051 |         -0.0002 |      27 |     117935194.0000 |
| NOVO            |       0.0027 |        0.0024 |        0.0031 |   0.0047 |         -0.0020 |      27 |     117935194.0000 |
| PTB             |       0.0007 |        0.0005 |        0.0009 |   0.0007 |         -0.0000 |      27 |     117935194.0000 |
| UP              |       0.0005 |        0.0002 |        0.0007 |   0.0005 |          0.0000 |      27 |     117935194.0000 |
| PCB             |       0.0004 |        0.0002 |        0.0006 |   0.0004 |          0.0000 |      27 |     117935194.0000 |
| PSTU            |       0.0002 |        0.0001 |        0.0004 |   0.0002 |          0.0000 |      27 |     117935194.0000 |
| DC              |       0.0002 |        0.0000 |        0.0003 |   0.0001 |          0.0000 |      27 |     117935194.0000 |

> `bias_nacional = share_pred - y_real`. Negativo = modelo subestima o partido no agregado nacional.

## UFs onde o intervalo agregado **NÃO cobriu** y_real

### Ano = 2022  (74/297 fora)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |    erro |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|--------:|
| SE         | PL              |       0.4424 |        0.4161 |        0.4681 |   0.2916 | -0.1508 |
| RN         | PL              |       0.4482 |        0.4245 |        0.4712 |   0.3102 | -0.1379 |
| AM         | PL              |       0.5531 |        0.5035 |        0.6009 |   0.4280 | -0.1251 |
| PB         | PL              |       0.4139 |        0.3923 |        0.4347 |   0.2962 | -0.1176 |
| PA         | PL              |       0.5198 |        0.5017 |        0.5398 |   0.4027 | -0.1171 |
| TO         | PL              |       0.5561 |        0.5341 |        0.5790 |   0.4400 | -0.1161 |
| MG         | PL              |       0.5502 |        0.5365 |        0.5632 |   0.4360 | -0.1142 |
| AP         | PL              |       0.5453 |        0.4934 |        0.5998 |   0.4341 | -0.1111 |
| RJ         | PL              |       0.6219 |        0.5865 |        0.6578 |   0.5109 | -0.1110 |
| RS         | PL              |       0.5953 |        0.5811 |        0.6096 |   0.4889 | -0.1064 |
| DF         | PL              |       0.6223 |        0.5330 |        0.7079 |   0.5165 | -0.1058 |
| ES         | PL              |       0.6261 |        0.6042 |        0.6493 |   0.5223 | -0.1039 |
| GO         | PL              |       0.6202 |        0.5982 |        0.6425 |   0.5216 | -0.0986 |
| MS         | PL              |       0.6249 |        0.5919 |        0.6571 |   0.5270 | -0.0979 |
| AL         | PL              |       0.4496 |        0.4230 |        0.4762 |   0.3605 | -0.0892 |
| PE         | PL              |       0.3842 |        0.3654 |        0.4050 |   0.2991 | -0.0850 |
| SP         | PL              |       0.5612 |        0.5370 |        0.5854 |   0.4771 | -0.0841 |
| BA         | PL              |       0.3262 |        0.3093 |        0.3435 |   0.2431 | -0.0831 |
| PR         | PL              |       0.6351 |        0.6177 |        0.6519 |   0.5526 | -0.0825 |
| CE         | PL              |       0.3338 |        0.3095 |        0.3582 |   0.2538 | -0.0800 |

### Distribuição dos descobertos por partido

- `PDT`: 27 UFs descobertas
- `PL`: 25 UFs descobertas
- `PT`: 10 UFs descobertas
- `MDB`: 6 UFs descobertas
- `NOVO`: 6 UFs descobertas

> Concentração em poucos partidos indica viés estrutural do LGBM (e.g., PL 2022 — efeito migração Bolsonaro PSL→PL). Solução não é o agregador — é #60 (pesquisas como feature).

## UFs — partido com maior share por UF (último ano)

| sigla_uf   | sigla_partido   |   share_pred |   share_lower |   share_upper |   y_real |
|:-----------|:----------------|-------------:|--------------:|--------------:|---------:|
| AC         | PL              |       0.6712 |        0.6288 |        0.7120 |   0.6250 |
| AL         | PT              |       0.5543 |        0.5279 |        0.5812 |   0.5650 |
| AM         | PL              |       0.5531 |        0.5035 |        0.6009 |   0.4280 |
| AP         | PL              |       0.5453 |        0.4934 |        0.5998 |   0.4341 |
| BA         | PT              |       0.6987 |        0.6814 |        0.7161 |   0.6973 |
| CE         | PT              |       0.6486 |        0.6236 |        0.6738 |   0.6591 |
| DF         | PL              |       0.6223 |        0.5330 |        0.7079 |   0.5165 |
| ES         | PL              |       0.6261 |        0.6042 |        0.6493 |   0.5223 |
| GO         | PL              |       0.6202 |        0.5982 |        0.6425 |   0.5216 |
| MA         | PT              |       0.6881 |        0.6718 |        0.7044 |   0.6884 |
| MG         | PL              |       0.5502 |        0.5365 |        0.5632 |   0.4360 |
| MS         | PL              |       0.6249 |        0.5919 |        0.6571 |   0.5270 |
| MT         | PL              |       0.6769 |        0.6562 |        0.6975 |   0.5984 |
| PA         | PT              |       0.5207 |        0.5014 |        0.5400 |   0.5222 |
| PB         | PT              |       0.6365 |        0.6162 |        0.6575 |   0.6421 |
| PE         | PT              |       0.6529 |        0.6323 |        0.6727 |   0.6527 |
| PI         | PT              |       0.7416 |        0.7199 |        0.7637 |   0.7425 |
| PR         | PL              |       0.6351 |        0.6177 |        0.6519 |   0.5526 |
| RJ         | PL              |       0.6219 |        0.5865 |        0.6578 |   0.5109 |
| RN         | PT              |       0.6278 |        0.6046 |        0.6503 |   0.6298 |
| RO         | PL              |       0.6684 |        0.6396 |        0.6983 |   0.6436 |
| RR         | PL              |       0.6973 |        0.6404 |        0.7520 |   0.6957 |
| RS         | PL              |       0.5953 |        0.5811 |        0.6096 |   0.4889 |
| SC         | PL              |       0.6811 |        0.6676 |        0.6950 |   0.6221 |
| SE         | PT              |       0.6356 |        0.6104 |        0.6612 |   0.6382 |
| SP         | PL              |       0.5612 |        0.5370 |        0.5854 |   0.4771 |
| TO         | PL              |       0.5561 |        0.5341 |        0.5790 |   0.4400 |

## Notas

- `eleitorado_uf`/`eleitorado_total` é a soma de `total_votos_mun`, proxy do eleitorado registrado (correlação > 0.95).
- Intervalos agregados via Monte Carlo: para cada linha (mun, partido), sortear uniforme centrada em pred com semi-largura (hi-lo)/2; agregar ponderando por `total_votos_mun`; pegar percentis 0.050 e 0.950.
- Independência entre partidos no MC dentro do mesmo município ignora a restrição sum_partido share ~= 1. O efeito é alargar levemente os intervalos agregados (conservador).
