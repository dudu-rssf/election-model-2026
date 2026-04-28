"""
src.features.continuity — índice de continuidade política municipal.

Hipótese H2 do projeto: municípios com mesmo grupo político dominante há
muitos anos entregam mais voto ao candidato apoiado. Esta é a feature
*crítica* do modelo e o briefing pede parada para revisão humana do top 20.

Conceitos:

1. **Transição eleitoral** — entre duas eleições municipais consecutivas,
   classificamos a passagem em quatro níveis:

       total   : reeleição do mesmo candidato (mesmo numero + partido).
       forte   : partido do vencedor se mantém.
       parcial : partido vencedor está na coligação anterior (ou o partido
                 anterior está na coligação atual), ou há ≥2 partidos em
                 comum entre coligações.
       ruptura : nenhuma sobreposição.

2. **Índice de continuidade** — mapeamento numérico da transição para
   permitir uso como feature:
       ruptura  -> 0.00
       parcial  -> 0.33
       forte    -> 0.67
       total    -> 1.00

3. **Anos consecutivos mesmo grupo** — contador acumulado em anos (múltiplos
   de 4 = duração do mandato). Semântica:

       N mandatos municipais consecutivos do mesmo partido/grupo = N × 4 anos.

   Implementação: na PRIMEIRA transição forte/total de uma sequência
   (após None ou ruptura), o mandato imediatamente anterior também faz
   parte da continuidade, então o contador inicia em 8 (4 do anterior +
   4 do atual). A cada transição forte/total subsequente soma-se +4.

       anos_consecutivos_mesmo_partido — só conta 'total'/'forte';
                                         zerado em 'parcial' e 'ruptura'.
       anos_consecutivos_mesmo_grupo   — 'total'/'forte' soma 4;
                                         'parcial' soma 2; zera em 'ruptura'.
                                         Também adiciona 4 do mandato
                                         anterior ao iniciar uma sequência.

Entradas:
    * `df_prefeito` — bruto TSE, cobre todas as eleições municipais baixadas.
    * `df_partidos_prefeito` — tabela `br_tse_eleicoes.partidos` filtrada por
      cargo='prefeito', usada para anexar `mayor_coligacao` (ano 2020 vem
      100% NA por buraco da BD).

Saída:
    DataFrame (<ano_col> × id_municipio) com 4 colunas de continuidade,
    para ser broadcast no long durante a consolidação.

Eixo configurável (`ano_col` + `map_ano_para_municipal`):
  * Eixo presidencial (Fase 3, default):
      ano_col='ano_presidencial', map=PRESIDENCIAL_TO_MUNICIPAL (X-2 — eleição
      municipal mais recente cujo vencedor está vigente no ano presidencial).
  * Eixo municipal (Fase 4.5):
      ano_col='ano_municipal', map=MUNICIPAL_TO_MUNICIPAL_ANTERIOR (X-4 — o
      prefeito vigente no momento da próxima eleição municipal foi eleito 4
      anos antes).
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.features.panel import (
    MUNICIPAL_TO_MUNICIPAL_ANTERIOR,
    anexar_coligacao_prefeito,
    prefeito_vencedor_por_eleicao,
)
from src.ingestion.queries import PRESIDENCIAL_TO_MUNICIPAL

logger = logging.getLogger(__name__)


CLASSE_INDICE: dict[str, float] = {
    "ruptura": 0.0,
    "parcial": 0.33,
    "forte": 0.67,
    "total": 1.0,
}


def _split_coligacao(s) -> set[str]:
    if s is None:
        return set()
    if isinstance(s, float) and np.isnan(s):
        return set()
    if pd.isna(s):
        return set()
    return {p.strip() for p in str(s).split(":") if p and p.strip()}


def classificar_transicao(
    atual: dict,
    anterior: dict,
) -> str | None:
    """Classifica transição entre duas eleições municipais consecutivas."""
    if atual.get("mayor_partido") is None or anterior.get("mayor_partido") is None:
        return None
    if pd.isna(atual.get("mayor_partido")) or pd.isna(anterior.get("mayor_partido")):
        return None

    mesmo_partido = atual["mayor_partido"] == anterior["mayor_partido"]
    mesmo_numero = (
        atual.get("mayor_numero") == anterior.get("mayor_numero")
        and atual.get("mayor_numero") is not None
        and not pd.isna(atual.get("mayor_numero"))
    )

    if mesmo_partido and mesmo_numero:
        return "total"
    if mesmo_partido:
        return "forte"

    col_atual = _split_coligacao(atual.get("mayor_coligacao"))
    col_anterior = _split_coligacao(anterior.get("mayor_coligacao"))

    # partido atual dentro da coligação anterior, ou vice-versa
    cross = (atual["mayor_partido"] in col_anterior) or (anterior["mayor_partido"] in col_atual)
    if cross:
        return "parcial"
    # sobreposição de coligações com ≥ 2 partidos em comum
    if len(col_atual & col_anterior) >= 2:
        return "parcial"
    return "ruptura"


def _construir_historico_vencedores(
    df_prefeito: pd.DataFrame,
    df_partidos_prefeito: pd.DataFrame,
) -> pd.DataFrame:
    """Vencedor por (id_municipio × ano_eleicao_municipal), com coligação."""
    vencedores = prefeito_vencedor_por_eleicao(df_prefeito)
    vencedores = anexar_coligacao_prefeito(vencedores, df_partidos_prefeito)
    cols = [
        "ano_eleicao_municipal",
        "id_municipio",
        "mayor_numero",
        "mayor_partido",
        "mayor_coligacao",
    ]
    return vencedores[cols].sort_values(["id_municipio", "ano_eleicao_municipal"]).reset_index(drop=True)


def _calcular_continuidade_por_municipio(grupo: pd.DataFrame) -> pd.DataFrame:
    """Para um município, percorre eleições municipais em ordem e calcula:
    classe, índice, anos_consecutivos_mesmo_partido, anos_consecutivos_mesmo_grupo.

    A primeira eleição do município no histórico fica com NA.
    """
    out = grupo.copy().sort_values("ano_eleicao_municipal").reset_index(drop=True)
    n = len(out)

    classes: list[str | None] = [None] * n
    indices: list[float | None] = [None] * n
    ac_partido: list[int | None] = [None] * n
    ac_grupo: list[int | None] = [None] * n

    for i in range(1, n):
        atual = out.iloc[i].to_dict()
        anterior = out.iloc[i - 1].to_dict()
        classe = classificar_transicao(atual, anterior)
        classes[i] = classe
        if classe is not None:
            indices[i] = CLASSE_INDICE[classe]

            prev_part = ac_partido[i - 1] if ac_partido[i - 1] is not None else 0
            prev_grupo = ac_grupo[i - 1] if ac_grupo[i - 1] is not None else 0

            # IMPORTANTE: ao INICIAR uma sequência de continuidade (prev=0,
            # seja porque é a primeira transição do histórico ou porque
            # veio de uma ruptura), o mandato imediatamente anterior também
            # faz parte da sequência e contribui com 4 anos. Por isso a
            # base é 4 quando prev == 0, e o valor da transição se soma
            # sobre essa base. Assim N mandatos consecutivos do mesmo
            # partido = N × 4 anos (antes tinha um off-by-one que gerava
            # (N-1) × 4 para sequências iniciando no começo do histórico).
            if classe in ("total", "forte"):
                base_part = prev_part if prev_part > 0 else 4
                base_grupo = prev_grupo if prev_grupo > 0 else 4
                ac_partido[i] = base_part + 4
                ac_grupo[i] = base_grupo + 4
            elif classe == "parcial":
                ac_partido[i] = 0  # parcial não conta como mesmo partido
                base_grupo = prev_grupo if prev_grupo > 0 else 4
                ac_grupo[i] = base_grupo + 2
            else:  # ruptura
                ac_partido[i] = 0
                ac_grupo[i] = 0

    out["continuidade_classe"] = classes
    out["indice_continuidade"] = indices
    out["anos_consecutivos_mesmo_partido"] = ac_partido
    out["anos_consecutivos_mesmo_grupo"] = ac_grupo
    return out


def calcular_historico_continuidade(
    df_prefeito: pd.DataFrame,
    df_partidos_prefeito: pd.DataFrame,
) -> pd.DataFrame:
    """Gera tabela (id_municipio × ano_eleicao_municipal) com as 4 colunas de continuidade."""
    vencedores = _construir_historico_vencedores(df_prefeito, df_partidos_prefeito)
    partes = [
        _calcular_continuidade_por_municipio(g)
        for _, g in vencedores.groupby("id_municipio", sort=False)
    ]
    if not partes:
        cols = [
            "ano_eleicao_municipal",
            "id_municipio",
            "mayor_numero",
            "mayor_partido",
            "mayor_coligacao",
            "continuidade_classe",
            "indice_continuidade",
            "anos_consecutivos_mesmo_partido",
            "anos_consecutivos_mesmo_grupo",
        ]
        return pd.DataFrame(columns=cols)
    return pd.concat(partes, ignore_index=True)


def features_continuity(
    df_prefeito: pd.DataFrame,
    df_partidos_prefeito: pd.DataFrame,
    anos_alvo: Iterable[int] | None = None,
    *,
    ano_col: str = "ano_presidencial",
    map_ano_para_municipal: Mapping[int, int] | None = None,
    anos_presidenciais: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Features de continuidade para cada (<ano_col> × id_municipio).

    Para cada ano-alvo, pega a eleição municipal vigente (via
    `map_ano_para_municipal`) e extrai as 4 colunas computadas em
    `calcular_historico_continuidade`.

    Args:
        df_prefeito: bruto TSE de prefeito.
        df_partidos_prefeito: tabela de partidos (cargo='prefeito').
        anos_alvo: anos do eixo (presidenciais ou municipais).
        ano_col: nome da coluna do eixo temporal — default
            `'ano_presidencial'` (Fase 3). Para Fase 4.5 usar
            `'ano_municipal'`.
        map_ano_para_municipal: dict ano_eixo→ano_eleicao_municipal_vigente.
            Default: `PRESIDENCIAL_TO_MUNICIPAL` (X-2). Para Fase 4.5 passar
            `MUNICIPAL_TO_MUNICIPAL_ANTERIOR` (X-4 — o prefeito vigente no
            momento da próxima eleição municipal foi eleito 4 anos antes).
        anos_presidenciais: alias deprecado de `anos_alvo`, mantido por
            backward compat com chamadas pré-Fase 4.5.
    """
    if anos_alvo is None and anos_presidenciais is not None:
        anos_alvo = anos_presidenciais
    if anos_alvo is None:
        raise ValueError("anos_alvo é obrigatório (ou anos_presidenciais legacy)")

    if map_ano_para_municipal is None:
        map_ano_para_municipal = PRESIDENCIAL_TO_MUNICIPAL

    hist = calcular_historico_continuidade(df_prefeito, df_partidos_prefeito)

    anos = sorted({int(a) for a in anos_alvo})
    pares = [(a, map_ano_para_municipal[a]) for a in anos]

    linhas = []
    cols_keep = [
        "continuidade_classe",
        "indice_continuidade",
        "anos_consecutivos_mesmo_partido",
        "anos_consecutivos_mesmo_grupo",
    ]
    for ano_eixo, ano_mun in pares:
        sub = hist[hist["ano_eleicao_municipal"] == ano_mun][
            ["id_municipio", *cols_keep]
        ].copy()
        sub.insert(0, ano_col, ano_eixo)
        linhas.append(sub)
    out = pd.concat(linhas, ignore_index=True) if linhas else pd.DataFrame()

    n_nulos = int(out["continuidade_classe"].isna().sum()) if len(out) else 0
    logger.info(
        "features_continuity: %d linhas; %d sem histórico anterior suficiente",
        len(out), n_nulos,
    )
    return out


# ------------------------------------------------------------
# Relatório para revisão humana (briefing pede top 20 em dev)
# ------------------------------------------------------------
def top_municipios_por_continuidade(
    hist: pd.DataFrame,
    diretorio: pd.DataFrame,
    n: int = 20,
) -> pd.DataFrame:
    """Top-N municípios com maior `anos_consecutivos_mesmo_partido` agregado.

    Critério: por município, pega o valor máximo de anos_consecutivos_mesmo_partido
    ao longo do histórico (o ápice da dominância).
    """
    agg = (
        hist.groupby("id_municipio", as_index=False)
        .agg(
            anos_max_partido=("anos_consecutivos_mesmo_partido", "max"),
            anos_max_grupo=("anos_consecutivos_mesmo_grupo", "max"),
            eleicoes_observadas=("ano_eleicao_municipal", "count"),
            ultima_classe=("continuidade_classe", "last"),
        )
    )
    agg = agg.merge(
        diretorio[["id_municipio", "nome", "sigla_uf"]].assign(
            id_municipio=lambda d: d["id_municipio"].astype("string")
        ),
        on="id_municipio",
        how="left",
    )
    agg["anos_max_partido"] = agg["anos_max_partido"].fillna(0).astype("int64")
    agg["anos_max_grupo"] = agg["anos_max_grupo"].fillna(0).astype("int64")
    agg = agg.sort_values(
        ["anos_max_partido", "anos_max_grupo"], ascending=[False, False]
    )
    return agg.head(n).reset_index(drop=True)


def salvar_relatorio_top_continuidade(
    hist: pd.DataFrame,
    diretorio: pd.DataFrame,
    caminho,
    n: int = 20,
) -> None:
    """Escreve `reports/top_continuidade_dev.md` para revisão humana."""
    top = top_municipios_por_continuidade(hist, diretorio, n=n)

    linhas = [
        f"# Top {n} municípios por continuidade política (dev)",
        "",
        "> Revisão humana obrigatória. O briefing manda PARAR nesta etapa se",
        "> os top 20 não fizerem sentido político — isso indica bug na lógica",
        "> de continuidade.",
        "",
        f"Total de municípios cobertos: {hist['id_municipio'].nunique()}",
        f"Total de eleições municipais: {hist['ano_eleicao_municipal'].nunique()}",
        "",
        "| # | UF | Município | Anos max mesmo partido | Anos max mesmo grupo | Eleições observadas | Última transição |",
        "|---|----|-----------|------------------------|----------------------|---------------------|------------------|",
    ]
    for i, row in top.iterrows():
        linhas.append(
            f"| {i+1} | {row.get('sigla_uf','?')} | {row.get('nome','?')} "
            f"| {int(row['anos_max_partido'])} | {int(row['anos_max_grupo'])} "
            f"| {int(row['eleicoes_observadas'])} | {row.get('ultima_classe') or 'NA'} |"
        )
    linhas.append("")
    linhas.append(
        "**Interpretação:** `anos_max_mesmo_partido` = maior sequência de "
        "eleições em que o MESMO partido venceu no município × 4 anos. "
        "`anos_max_mesmo_grupo` estende considerando coligações sobrepostas "
        "(parcial conta como 2 anos)."
    )
    linhas.append("")
    linhas.append(
        "**Ressalva:** em modo dev temos apenas eleições municipais "
        f"cobertas pelo mapping presidencial→municipal ({sorted(hist['ano_eleicao_municipal'].unique().tolist())}). "
        "Com histórico curto, municípios de 3ª/4ª coloção na dominância podem não aparecer."
    )
    with open(caminho, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    logger.info("salvo relatório: %s", caminho)


__all__ = [
    "CLASSE_INDICE",
    "classificar_transicao",
    "calcular_historico_continuidade",
    "features_continuity",
    "top_municipios_por_continuidade",
    "salvar_relatorio_top_continuidade",
]
