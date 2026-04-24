# Status — Fase 3 (Feature engineering)

Data: 2026-04-22
Modo: dev (config.yaml)

## O que foi feito

Cinco módulos em `src/features/` + CLI em `scripts/03_features.py` + bateria de testes em `tests/test_features.py`. A Fase 3 consolida todas as features em `data/processed/features[.dev].parquet` com **uma linha por (ano_presidencial × id_municipio × sigla_partido)** — formato pronto para os joins de alvo nas Fases 4–6.

### 1. `features/structural.py` — covariáveis estáticas

Features por (ano_presidencial × id_municipio):

- `sigla_uf`, `regiao`, `capital_uf` — do diretório IBGE (via painel mestre).
- `log_eleitorado` — `log1p(total_votos_mun)` do próprio `presidencial_long`. Aproximação razoável do eleitorado ativo — evita dependência do `populacao` do IBGE e fica temporalmente alinhado com a eleição.
- `porte` — tercis (`pequeno` / `medio` / `grande`) por ano usando `qcut(duplicates="drop")`. Usei tercis por ano (não por eleição global) para que a comparação faça sentido dentro do mesmo ciclo.

### 2. `features/local_power.py` — poder local

Dois blocos por design:

**Bloco A — (mun × ano):** `features_local_mun_ano(painel)`:
- `share_prefeito_local` = `mayor_share_1t` vigente.
- `margem_prefeito` = `mayor_margem_1t`.
- `primeiro_mandato_prefeito` = 1 se o partido do prefeito atual é diferente do partido do prefeito da eleição municipal imediatamente anterior no mesmo município; NA se não há histórico ou se o registro atual é NA.

**Bloco B — (mun × ano × partido):** `alinhamento_partido_com_prefeito(painel, partidos)`:
- `alinhado_prefeito_partido` — 1 se `sigla_partido == mayor_partido`, 0 se NA de prefeito → NA.
- `alinhado_prefeito_coligacao` — 1 se `sigla_partido` está no set `_split_coligacao(mayor_coligacao)`.

Cross-join `painel × universo de partidos` produz uma linha para **todo** partido presidencial no universo, mesmo que não tenha concorrido naquele município — necessário para o merge final no long.

### 3. `features/historical.py` — lag, swing, volatilidade

Features por (ano × mun × partido):

- `lag_share_1t` / `lag2_share_1t` — shift temporal dentro do grupo (município × partido).
- `swing_share_1t` = `share_1t - lag_share_1t`.
- `volatilidade_partido` — desvio padrão (ddof=0) dos shares *estritamente anteriores*, via `.shift(1).expanding(min_periods=2).std(ddof=0)`.

**Política `SHARE_ZERO_SE_AUSENTE = True`**: partidos que não lançaram candidato num ano recebem share = 0 naquele ano **se o partido aparece em pelo menos um outro ano no município** (universo por município). Documentado no topo do módulo; pode ser revertido via constante. Alternativa (NaN em vez de 0) daria muitos NaN em lags de partidos nanicos.

### 4. `features/continuity.py` — feature-crítica (H2)

**Classificação de transição** entre eleições municipais consecutivas:

| Classe   | Condição                                                                 | Índice |
|----------|--------------------------------------------------------------------------|--------|
| total    | mesmo número + mesmo partido (reeleição do candidato)                    | 1.00   |
| forte    | mesmo partido                                                            | 0.67   |
| parcial  | partido atual ∈ coligação anterior, ou anterior ∈ atual, ou ≥2 partidos em comum | 0.33 |
| ruptura  | nenhum dos acima                                                         | 0.00   |

**Contadores acumulados por município** (unidade = anos):
- `anos_consecutivos_mesmo_partido` — +4 em total/forte, 0 em parcial/ruptura.
- `anos_consecutivos_mesmo_grupo` — +4 em total/forte, +2 em parcial, 0 em ruptura.

Tanto a classe quanto os contadores ficam NA na primeira eleição observada do município (não há anterior).

**Broadcast para o long**: `features_continuity(df_prefeito, df_candidatos_prefeito, anos_presidenciais)` usa `PRESIDENCIAL_TO_MUNICIPAL` para pegar o ano municipal vigente para cada ano presidencial e extrai só as 4 colunas. Como estas features são (mun × ano), são joinadas no long em (ano_presidencial, id_municipio).

**Relatório para revisão humana** (`top_continuidade_dev.md`): 20 municípios com maior `anos_max_mesmo_partido`. O briefing manda **PARAR aqui em prod** se o top 20 não fizer sentido político — é o sinal de que a lógica de continuidade tem bug.

### 5. `features/vertical.py` — alinhamento vertical

**Governador** — `governador_vencedor_por_eleicao(df_gov, df_cand_gov)`:
- Agrega votos municipais para nível estadual (groupby UF).
- Escolhe vencedor por UF × ano; tie-break pelo menor número de candidato.
- Anexa coligação via `df_candidatos_governador`.
- Implementação direta (não reuso do `prefeito_vencedor_por_eleicao` para não bagunçar colunas).

`alinhamento_partido_com_governador(painel, df_gov, df_cand_gov, partidos)` produz 4 flags por (ano_pres × mun × partido):
- `alinhado_gov_{vigente,concorrente}_{partido,coligacao}` — vigente usa `PRESIDENCIAL_TO_ESTADUAL_ANTERIOR[ano_pres]`, concorrente usa `ano_pres` (eleição estadual simultânea).

**Deputado federal** — `share_dep_federal_por_partido(df_dep_fed, anos_presidenciais)`:
- Soma votos por (ano × mun × partido) — múltiplos candidatos de um mesmo partido vão para o mesmo bucket.
- Divide pelo total local daquele ano → `share_dep_federal_partido`.
- Ano é o próprio ano presidencial (eleição simultânea).

### 6. `scripts/03_features.py`

Orquestra em 4 passos:
1. `carregar_insumos()` — painel + pres_long + 4 raws.
2. `computar_features()` — chama os 7 blocos (2 de local_power, 2 de vertical, 1 cada dos outros).
3. `consolidar()` — left-join em `presidencial_long`:
   - Covariáveis (ano, mun) primeiro (`structural`, `local_power_mun`, `continuity`).
   - Específicas de partido (ano, mun, partido) depois (`local_power_partido`, `historical`, `align_gov`, `share_dep_federal`).
4. Salva `data/processed/features[.dev].parquet` via `fio.save_processed`.
5. Gera `reports/top_continuidade_dev.md` (a menos que `--skip-report-top`).

## Tabela final

**Formato**: 1 linha por candidato × município × ano presidencial. **Colunas** (31 na smoke):

```
chaves:     ano_presidencial, sigla_uf, id_municipio,
            numero_candidato, nome_candidato, sigla_partido
alvo-ish:   votos, total_votos_mun, share_1t
structural: regiao, capital_uf, log_eleitorado, porte
local pwr:  share_prefeito_local, margem_prefeito, primeiro_mandato_prefeito,
            alinhado_prefeito_partido, alinhado_prefeito_coligacao
continuid:  continuidade_classe, indice_continuidade,
            anos_consecutivos_mesmo_partido, anos_consecutivos_mesmo_grupo
historical: lag_share_1t, lag2_share_1t, swing_share_1t, volatilidade_partido
vertical:   alinhado_gov_vigente_partido, alinhado_gov_vigente_coligacao,
            alinhado_gov_concorrente_partido, alinhado_gov_concorrente_coligacao,
            share_dep_federal_partido
```

## Testes

`tests/test_features.py` — **14 testes** em pytest (pytest real na sua máquina; no sandbox rodei asserts diretos via `python -c`, **todos passando**):

1. `structural` tem 1 linha por (mun, ano) e colunas esperadas.
2. `porte` ∈ {pequeno, medio, grande}; `log_eleitorado` monotônico com votos.
3. `local_power` bloco A cobre todas as linhas do painel; NA propagado em municípios sem prefeito.
4. `primeiro_mandato_prefeito`: 1 em transição de partido (SP 2012 PT → 2016 PSDB), NA na primeira observação.
5. `alinhado_prefeito_partido`: SP 2014 prefeito PT + partido PT → 1; SP 2018 prefeito PSDB + PSL → 0.
6. `alinhado_prefeito_coligacao`: SP 2014 PT na coligação "PT:PCdoB:PSB" → 1.
7. Municípios sem prefeito → ambos NA.
8. `historical` lag e swing: SP PT 2014→2018 share 0.4→0.3, lag=0.4, swing=-0.1.
9. `historical` expande universo: PSL em 2014 (não concorreu) aparece com share=0.
10. `continuity` classifica SP 2012 PT → 2016 PSDB como ruptura; Adamantina 2012 PT → 2016 PSB (PT na coligação só "PT", PSB na coligação só "PSB:PP") como ruptura.
11. `CLASSE_INDICE` mapeia ruptura→0.0 e total→1.0.
12. Mesmo candidato reeleito → classe "total", anos_consecutivos_mesmo_partido=4.
13. `features_continuity` mapeia presidencial→municipal (2014 NA porque é a primeira; 2018 preenchido).
14. `vertical`:
    - `governador_vencedor_por_eleicao` escolhe PSDB 600 vs PT 400 com share 0.6.
    - `alinhamento_partido_com_governador` produz as 4 flags corretas (vigente PSDB + concorrente PT em cenário sintético).
    - `share_dep_federal_por_partido` soma múltiplos candidatos do mesmo partido (PT com 300+200 → share 0.5).
15. Script `03_features.py` importa limpo e expõe `main`, `carregar_insumos`, `computar_features`, `consolidar`.

**Smoke E2E** com fixtures sintéticas (2 municípios × 2 anos presidenciais, 2 eleições municipais, 1 eleição estadual vigente + 1 concorrente, 2 anos de deputado federal): **8 linhas × 31 colunas** no `features` final, joins consistentes, nenhum NaN inesperado.

## Tempo de execução

- **Smoke sandbox**: < 150 ms para 2 mun × 2 anos × 3 partidos.
- **Dev (SP, 100 municípios × 3 presidenciais)**: estimativa **2–5 s**. Operações custosas (todas ainda pandas): continuity itera por município com loop explícito (~100 grupos × ~3 eleições = 300 iterações triviais), historical usa transform com expanding (ok para ~10k linhas).
- **Prod (5.570 mun × 7 presidenciais)**: estimativa **60–180 s**. O gargalo provável é o loop da continuity (~5.570 grupos). Se incomodar, vetorizar com `groupby().apply()` sem loop Python é simples.

## Decisões autônomas

- **(mun × ano) separado de (mun × ano × partido)**: dois blocos explícitos em cada módulo. Reduz cardinalidade intermediária e deixa o merge final mais limpo. Só crio o produto cartesiano quando necessário (`alinhamento_*`).
- **Universo de partidos expandido com share=0** em `historical`: alternativa (NaN) dava muitos nulos em partidos pequenos, quebrando lags. Controlado por constante `SHARE_ZERO_SE_AUSENTE`.
- **Continuidade com 4 classes, não 3 ou 5**: "parcial" captura o caso real de "o PP elegeu mas estava na coligação do PMDB anterior" — frequente no interior. Critério de ≥2 partidos em comum é conservador para não inflar "parcial".
- **Contadores em anos (×4), não em "número de eleições"**: politicamente mais interpretável ("16 anos de PT") e alinha com a escala temporal de outras features.
- **Governador vigente + concorrente**: gerar as duas flags separadas custa barato e deixa a Fase 4 decidir se usa uma, outra ou ambas. Vigente é "quem já está lá e provavelmente endossou"; concorrente é "o favorito que está vencendo simultaneamente".
- **Share de deputado federal no mesmo ano presidencial**: o eleitor vota simultaneamente — é a melhor medida da base federal local contemporânea, mesmo que o deputado eleito nem sempre seja o mais votado no município.
- **Relatório top-20 gerado junto com o parquet**: não espero o usuário pedir — o briefing manda parar nesta etapa, então o artefato de revisão sai por default.

## Problemas encontrados

- **Reuso inicial de `prefeito_vencedor_por_eleicao` para governador**: tentei adaptar renomeando `sigla_uf`↔`id_municipio`, mas o merge interno produzia colunas duplicadas. Reescrevi a função de governador direto — mais limpo e mais rápido que o hack.
- **`nome_candidato` opcional em governador**: nem todos os tables do BD incluem; tratado com `if nome_col`.
- **Teste de local_power com fixture sem KASSAB**: um smoke test anterior assumia share 0.5, mas a soma real era 200/350 ≈ 0.571. Assert ajustado.

## Checkpoints de revisão humana

O briefing manda **PARAR e revisar** antes da Fase 4:

1. **Top 20 municípios de continuidade** (`reports/top_continuidade_dev.md`):
   - Este relatório é gerado quando você roda `python scripts/03_features.py` com dados reais.
   - Espere ver: interior de SP com dominância do PSDB (anos 2000), interior de MG com coligações PP/PMDB fortes, capitais com mais ruptura.
   - **Se os top 20 não fizerem sentido político, me avise** — provavelmente é bug na classificação de transição ou no parser de coligação.

2. **Sanity check do `features.parquet`**:
   - Uma linha por candidato × município × ano? Faça `df.groupby(["ano_presidencial","id_municipio"]).size()` — deve bater com o número de candidatos presidenciais daquele ano.
   - NAs esperados: `lag_share_1t` no primeiro ano, `primeiro_mandato_prefeito` na primeira eleição municipal observada, `alinhado_prefeito_*` onde o município não tem prefeito registrado.
   - NAs **inesperados**: se `share_dep_federal_partido` ou `alinhado_gov_*` tiverem muitos NAs em anos centrais, investigar cobertura das queries da Fase 1.5.

## Próximos passos

Pronto para a **Fase 4 — Análise exploratória e validação de hipóteses**:

1. Carregar `features.parquet` + target (`share_1t`).
2. Verificar estrutura das hipóteses do briefing (H1 continuidade, H2 alinhamento vertical, H3 swing histórico, H4 efeito de poder local).
3. Notebooks `notebooks/04_eda.ipynb` ou scripts em `scripts/04_eda.py` com correlações, gráficos e checks de signal.
4. **Critério de parada explícito do briefing**: se `indice_continuidade` ou `alinhamento_gov` não tiverem correlação alguma com o target, revisitar features antes de modelar.

**Posso seguir para a Fase 4?** Antes disso, dá uma olhada no `reports/top_continuidade_dev.md` (geração local) e me avise se os top 20 fazem sentido. Se sim, prossigo com a exploratória.
