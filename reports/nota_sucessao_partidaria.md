# Nota técnica — sucessão partidária (versão mínima)

**Data:** 2026-04-24 · **Tasks:** #55, #56, #57, #58, #59 fechadas; #60 permanece pendente.

## Contexto

Fase 4 mostrou catástrofe isolada em PL 2022 (MAE = 0.51, share_medio real = 0.53,
share predito ~0.02). Causa raiz documentada no `status_fase_4.md`: siglas
partidárias não são identificadores estáveis — "PL" em 2022 (base bolsonarista
migrada do PSL) é um objeto político diferente de "PL" em 2018 (partido pequeno
de centro), mas o modelo vê a mesma categoria.

A decisão foi implementar a **versão mínima** (mapeamento manual de sucessões
por dict em YAML) e deixar a **versão ideal** (feature de intenção de voto via
pesquisas) para a versão final do modelo.

## O que foi implementado

| Arquivo | Mudança |
|---|---|
| `config.yaml` | Seção `partido_sucessao` com mapeamento `PL:2022→PSL` e `UNIÃO:2022→DEM` |
| `src/features/partido_sucessao.py` | Módulo novo: `resolver_sigla_canonica`, `aplicar_sucessao` |
| `src/features/historical.py` | Nova coluna paralela `lag_share_1t_sucessao` (lag agrupando por sigla canônica em vez da sigla bruta) |
| `scripts/03_features.py` | Lê `CONFIG["partido_sucessao"]` e passa para `features_historical` |
| `src/models/features.py` | `lag_share_1t_sucessao` entra em `FEATURES_NUMERICAS` |
| `tests/test_features.py` | +2 testes cobrindo sucessão |
| `tests/test_models.py` | Fixture ajustada para incluir a nova feature |

## Resultado empírico

Pipeline re-rodado em dev (SP × 100 municípios × 3 anos):

| Métrica | Antes (sem sucessão) | Depois (com sucessão mínima) | Δ |
|---|---|---|---|
| LightGBM MAE geral | 0.0565 | 0.0558 | −0.0007 (~1%) |
| PL 2022 MAE | 0.5096 | 0.5096 | 0.00 |
| UNIÃO 2022 MAE | 0.0008 | 0.0008 | 0.00 |

Feature importance (gain) do LightGBM:

| feature | importance |
|---|---|
| `sigla_partido` | 39823.98 |
| `share_dep_federal_partido` | 27129.93 |
| `swing_share_1t` | 26637.38 |
| `log_eleitorado` | 18941.44 |
| ... | ... |
| `lag_share_1t` | 6546.70 |
| `lag_share_1t_sucessao` | 874.33 |

A feature nova ficou entre as menos usadas pelo modelo. 200 linhas remapeadas
(logs do `src.features.partido_sucessao`: 100 PL + 100 UNIÃO), mas o MAE por
partido não mexeu.

## Por que não funcionou (diagnóstico)

Dois mecanismos se compõem:

1. **Covariate shift.** O mapping só está ativo em 2022. Em 2014 e 2018,
   `sigla_canonica(PL) = PL`, logo `lag_share_1t_sucessao(PL, 2018)` é
   idêntica a `lag_share_1t(PL, 2018)` — ambas pequenas (PL era partido
   de centro). O LightGBM não vê, no treino, nenhum caso de "PL com
   `lag_share_1t_sucessao` grande". Quando encontra isso no teste (2022,
   PL, lag_sucessao ≈ 0.5), não tem como saber que deveria confiar nessa
   pista. A feature basicamente não é usada.

2. **Dominância da categoria.** `sigla_partido` tem gain ≈ 40k — 6× maior
   que `lag_share_1t`. O modelo ancora a predição na identidade da sigla
   ("PL historicamente ≈ 2%"), e qualquer feature numérica tem dificuldade
   para contrapor esse viés.

Fingerprint na calibração por decil: o decil 7 tem `pred_medio = 0.015` e
`real_medio = 0.184`. Esses são exatamente as 110 linhas de PL 2022 (SP,
100 municípios, 2022 = 100 linhas PL, e alguns vizinhos do bucket).

## Decisão

**Opção B:** aceitar PL 2022 como outlier documentado na Fase 4. Não
tentamos corrigir via troca de categoria (`sigla_partido` → `sigla_canonica`),
que seria a próxima intervenção natural, porque:

- O risco colateral — unificar PSL 2014 (pequeno) + PSL 2018 (Bolsonaro) na
  mesma categoria — compromete outras siglas sem garantia de ganho.
- A correção estrutural correta é **feature de intenção de voto via
  pesquisas** (task #60), que dá o sinal em tempo real e não exige
  mapeamento manual caso-a-caso.
- A amostra dev é pequena (3430 linhas, 3 anos). Antes de iterar muito em
  cirurgias aqui, faz mais sentido rodar a pipeline em prod (7 anos
  presidenciais, 5570 municípios) e ver se o sinal consolida.

## Infraestrutura mantida, não removida

O módulo `partido_sucessao` e o dict em `config.yaml` ficam no código:

- Zero custo (~1% perturbação em MAE geral, benigno).
- Documenta a decisão política de modelagem (vide `config.yaml` com comentário longo).
- Hook pronto para receber, na versão final, o "predecessor político" derivado
  das pesquisas (se a gente optar por continuar usando sucessão explícita em
  paralelo às pesquisas).

## Follow-up

- **Task #60 (pendente)**: substituir o dict por feature dinâmica de intenção
  de voto (Datafolha/Ipec/Quaest). Antes da versão final.
- **Fase 4.5**: replicar pipeline para prefeito (target municipal).
- **Prod run**: rodar Fase 4 com 7 anos e UFs completas — verificar se a
  catástrofe do PL atenua com mais dados históricos e cobertura geográfica.
