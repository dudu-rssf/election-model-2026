"""
src.features.vertical — alinhamento vertical (governador + base federal).

Features por (ano × id_municipio × sigla_partido):

    alinhado_gov_vigente_partido    — 1 se partido == governador eleito na
                                       eleição estadual anterior.
    alinhado_gov_vigente_coligacao  — 1 se partido está na coligação do gov vigente.
    alinhado_gov_concorrente_partido    (apenas no eixo presidencial)
    alinhado_gov_concorrente_coligacao  (apenas no eixo presidencial)
    share_dep_federal_partido       — fração dos votos de dep. federal do
                                       partido no município na eleição federal
                                       relevante (concorrente no eixo presidencial,
                                       anterior no eixo municipal).

Eixo presidencial (Fase 3):
  * Governador vigente em X = eleito em X-4 (map PRESIDENCIAL_TO_ESTADUAL_ANTERIOR).
  * Governador concorrente em X = eleito em X (mesmo ano — eleição geral).
  * Deputado federal concorrente em X = eleito em X.

Eixo municipal (Fase 4.5):
  * Governador vigente em ano_municipal X = eleito em X-2 (map MUNICIPAL_TO_ESTADUAL_ANTERIOR).
  * NÃO há governador concorrente (municipal não é simultânea com estadual).
  * Deputado federal vigente em ano_municipal X = eleito em X-2 (mesmo mapa).

Broadcast: o governador é único por UF × ano da eleição — aplicado a
todos os municípios daquela UF. Deputado federal é municipal (somamos votos
por partido no município).
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from src.features.local_power import _split_coligacao
from src.ingestion.queries import PRESIDENCIAL_TO_ESTADUAL_ANTERIOR


# ------------------------------------------------------------
# Mapas do eixo municipal (Fase 4.5)
# ------------------------------------------------------------
# Eleição municipal X → eleição estadual/federal imediatamente anterior (X-2).
# Governador/dep. federal eleitos em ano par de eleição geral, 2 anos antes
# da municipal. Não existe concorrente no eixo municipal.
MUNICIPAL_TO_ESTADUAL_ANTERIOR: dict[int, int] = {
    2000: 1998,
    2004: 2002,
    2008: 2006,
    2012: 2010,
    2016: 2014,
    2020: 2018,
    2024: 2022,
    2028: 2026,
}

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Governador — vencedor por UF × ano (é uma eleição estadual)
# ------------------------------------------------------------
def governador_vencedor_por_eleicao(
    df_governador: pd.DataFrame,
    df_partidos_gov: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reduz resultados_governador para o vencedor por (ano × UF).

    A tabela do BD vem com votos municipalizados; agregamos para o total
    estadual e escolhemos o candidato com mais votos. Tie-break
    determinístico: menor `numero_candidato`.

    `df_partidos_gov` (opcional): tabela `br_tse_eleicoes.partidos` filtrada
    por cargo='governador' (ver `partidos_governador_sql`). Usada apenas
    para anexar `gov_coligacao` por (ano, sigla_uf, sigla_partido). Se
    None ou sem `composicao_coligacao`, a coluna sai como NA.
    """
    required = {"ano", "sigla_uf", "numero_candidato", "sigla_partido", "votos"}
    missing = required - set(df_governador.columns)
    if missing:
        raise ValueError(f"df_governador sem colunas: {sorted(missing)}")

    # Agrega voto por (ano × UF × candidato)
    nome_col = "nome_candidato" if "nome_candidato" in df_governador.columns else None
    group_cols = ["ano", "sigla_uf", "numero_candidato", "sigla_partido"]
    if nome_col:
        group_cols.insert(3, nome_col)
    df_uf = (
        df_governador.groupby(group_cols, as_index=False)["votos"].sum()
    )

    totais = (
        df_uf.groupby(["ano", "sigla_uf"], as_index=False)["votos"]
        .sum()
        .rename(columns={"votos": "gov_votos_total_uf"})
    )

    df_sorted = df_uf.sort_values(
        ["ano", "sigla_uf", "votos", "numero_candidato"],
        ascending=[True, True, False, True],
    )
    df_sorted["rank"] = df_sorted.groupby(["ano", "sigla_uf"]).cumcount() + 1
    vencedor = df_sorted[df_sorted["rank"] == 1].copy()
    segundo = df_sorted[df_sorted["rank"] == 2][["ano", "sigla_uf", "votos"]].rename(
        columns={"votos": "_votos_2o"}
    )
    vencedor = vencedor.merge(totais, on=["ano", "sigla_uf"], how="left")
    vencedor = vencedor.merge(segundo, on=["ano", "sigla_uf"], how="left")

    vencedor["gov_share_1t"] = vencedor["votos"] / vencedor["gov_votos_total_uf"]
    vencedor["gov_margem_1t"] = (
        (vencedor["votos"] - vencedor["_votos_2o"].fillna(0))
        / vencedor["gov_votos_total_uf"]
    )

    out = pd.DataFrame({
        "ano_eleicao_estadual": vencedor["ano"].astype("int64"),
        "sigla_uf": vencedor["sigla_uf"],
        "gov_numero": vencedor["numero_candidato"],
        "gov_nome": vencedor[nome_col] if nome_col else pd.NA,
        "gov_partido": vencedor["sigla_partido"],
        "gov_share_1t": vencedor["gov_share_1t"].astype("float64"),
        "gov_margem_1t": vencedor["gov_margem_1t"].astype("float64"),
    })

    # Anexa coligação via tabela `br_tse_eleicoes.partidos` (cargo='governador').
    # Granularidade: 1 linha por (ano, sigla_uf, sigla_partido). Join por
    # SIGLA do partido vencedor.
    tem_coligacao = (
        df_partidos_gov is not None
        and len(df_partidos_gov) > 0
        and "composicao_coligacao" in df_partidos_gov.columns
    )
    if tem_coligacao:
        p_full = df_partidos_gov.copy()
        # Reconstrói composicao_coligacao a partir de sequencial_coligacao
        # (2010 SP governador vem com sequencial populado mas composicao NULL).
        if "sequencial_coligacao" in p_full.columns:
            from src.features.coligacao import reconstruir_composicao_via_sequencial
            p_full = reconstruir_composicao_via_sequencial(p_full, ["ano", "sigla_uf"])
        p = (
            p_full[["ano", "sigla_uf", "sigla_partido", "composicao_coligacao"]]
            .sort_values(
                ["ano", "sigla_uf", "sigla_partido", "composicao_coligacao"],
                na_position="last",
            )
            .drop_duplicates(subset=["ano", "sigla_uf", "sigla_partido"], keep="first")
        )
        out = out.merge(
            p,
            left_on=["ano_eleicao_estadual", "sigla_uf", "gov_partido"],
            right_on=["ano", "sigla_uf", "sigla_partido"],
            how="left",
        ).drop(columns=["ano", "sigla_partido"])
        out = out.rename(columns={"composicao_coligacao": "gov_coligacao"})

        cob = (
            out.dropna(subset=["gov_partido"])
            .assign(_ok=out["gov_coligacao"].notna())
            .groupby("ano_eleicao_estadual")["_ok"]
            .mean()
        )
        for ano, pct in cob.items():
            if pct < 0.5:
                logger.warning(
                    "gov_coligacao: ano %d com cobertura %.1f%%",
                    int(ano), 100 * pct,
                )
    else:
        if df_partidos_gov is not None and "composicao_coligacao" not in df_partidos_gov.columns:
            logger.warning(
                "partidos_governador sem 'composicao_coligacao'; gov_coligacao fica NA"
            )
        out["gov_coligacao"] = pd.NA

    return out[[
        "ano_eleicao_estadual", "sigla_uf",
        "gov_numero", "gov_nome", "gov_partido", "gov_coligacao",
        "gov_share_1t", "gov_margem_1t",
    ]].reset_index(drop=True)


# ------------------------------------------------------------
# Alinhamento com governador (vigente + concorrente)
# ------------------------------------------------------------
def alinhamento_partido_com_governador(
    painel: pd.DataFrame,
    df_governador: pd.DataFrame,
    df_partidos_gov: pd.DataFrame | None,
    partidos: Iterable[str],
    *,
    ano_col: str = "ano_presidencial",
    map_vigente: Mapping[int, int] | None = None,
    incluir_concorrente: bool = True,
) -> pd.DataFrame:
    """Produz flags de alinhamento com governador vigente (e, opcionalmente,
    concorrente — só faz sentido no eixo presidencial).

    Args:
        painel: painel_mestre com [ano_col, id_municipio, sigla_uf].
        df_governador: resultados_governador (votos por candidato × UF).
        df_partidos_gov: partidos_governador (para coligação).
        partidos: universo de siglas partidárias a considerar.
        ano_col: nome da coluna de ano no painel.
        map_vigente: {ano_eixo: ano_estadual_vigente}. Default =
            PRESIDENCIAL_TO_ESTADUAL_ANTERIOR (eixo presidencial).
            Para eixo municipal, passar MUNICIPAL_TO_ESTADUAL_ANTERIOR.
        incluir_concorrente: se True (default, eixo presidencial), usa o
            mesmo ano do eixo como ano concorrente (eleição geral). Se False
            (eixo municipal), só gera features de governador vigente.

    Returns:
        DataFrame (ano_col × id_municipio × sigla_partido) com flags.
    """
    required_painel = {ano_col, "id_municipio", "sigla_uf"}
    missing = required_painel - set(painel.columns)
    if missing:
        raise ValueError(f"painel sem colunas: {sorted(missing)}")

    map_vigente = dict(map_vigente or PRESIDENCIAL_TO_ESTADUAL_ANTERIOR)

    vencedores_gov = governador_vencedor_por_eleicao(df_governador, df_partidos_gov)
    partidos_list = sorted({str(p) for p in partidos if pd.notna(p)})

    painel_base = painel[[ano_col, "id_municipio", "sigla_uf"]].copy()
    painel_base["id_municipio"] = painel_base["id_municipio"].astype("string")
    painel_base["ano_eleicao_estadual_vigente"] = painel_base[ano_col].map(map_vigente)
    if incluir_concorrente:
        painel_base["ano_eleicao_estadual_concorrente"] = painel_base[ano_col].astype("int64")

    def _merge_gov(df, col, prefix):
        """Anexa governador por (ano, UF) usando a coluna indicada."""
        g = vencedores_gov.rename(
            columns={
                "ano_eleicao_estadual": col,
                "gov_partido": f"{prefix}_partido_gov",
                "gov_coligacao": f"{prefix}_coligacao_gov",
            }
        )[[col, "sigla_uf", f"{prefix}_partido_gov", f"{prefix}_coligacao_gov"]]
        return df.merge(g, on=[col, "sigla_uf"], how="left")

    painel_base = _merge_gov(painel_base, "ano_eleicao_estadual_vigente", "vigente")
    if incluir_concorrente:
        painel_base = _merge_gov(painel_base, "ano_eleicao_estadual_concorrente", "concorrente")

    painel_base["_k"] = 1
    partidos_df = pd.DataFrame({"sigla_partido": partidos_list, "_k": 1})
    out = painel_base.merge(partidos_df, on="_k").drop(columns="_k")

    prefixos = ["vigente"] + (["concorrente"] if incluir_concorrente else [])
    for prefix in prefixos:
        part_gov = f"{prefix}_partido_gov"
        col_gov = f"{prefix}_coligacao_gov"
        na_mask = out[part_gov].isna()

        out[f"alinhado_gov_{prefix}_partido"] = (out["sigla_partido"] == out[part_gov]).astype("Int64")
        out.loc[na_mask, f"alinhado_gov_{prefix}_partido"] = pd.NA

        col_sets = out[col_gov].apply(lambda s: set(_split_coligacao(s)))
        out[f"alinhado_gov_{prefix}_coligacao"] = [
            (pd.NA if not c else int(p in c))
            for p, c in zip(out["sigla_partido"], col_sets)
        ]
        out[f"alinhado_gov_{prefix}_coligacao"] = out[f"alinhado_gov_{prefix}_coligacao"].astype("Int64")

    cols = [ano_col, "id_municipio", "sigla_partido",
            "alinhado_gov_vigente_partido", "alinhado_gov_vigente_coligacao"]
    if incluir_concorrente:
        cols += ["alinhado_gov_concorrente_partido", "alinhado_gov_concorrente_coligacao"]
    return out[cols].reset_index(drop=True)


# ------------------------------------------------------------
# Base federal local via deputado federal
# ------------------------------------------------------------
def share_dep_federal_por_partido(
    df_dep_federal: pd.DataFrame,
    anos_alvo: Iterable[int] | None = None,
    *,
    ano_col: str = "ano_presidencial",
    map_ano_para_federal: Mapping[int, int] | None = None,
    # Alias de retrocompat (Fase 3)
    anos_presidenciais: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Share dos votos de dep. federal por (ano_eixo × mun × partido).

    No eixo presidencial (default), o ano federal concorrente é o mesmo do
    eixo (eleição geral). No eixo municipal, o ano federal é o ANTERIOR
    (X-2, via `map_ano_para_federal`).

    Args:
        df_dep_federal: resultados de dep. federal (ano, id_municipio,
            sigla_partido, votos).
        anos_alvo: valores do eixo (ano_eixo) que queremos cobrir.
            Alias: `anos_presidenciais` (deprecated, para compat com Fase 3).
        ano_col: nome da coluna de ano na saída.
        map_ano_para_federal: {ano_eixo: ano_federal}. Default = identidade
            (ano_eixo == ano_federal, eixo presidencial). No eixo municipal,
            passar `MUNICIPAL_TO_ESTADUAL_ANTERIOR` (dep. federal e governador
            são eleitos juntos).

    Returns:
        DataFrame com (<ano_col>, id_municipio, sigla_partido, share_dep_federal_partido).
    """
    required = {"ano", "id_municipio", "sigla_partido", "votos"}
    missing = required - set(df_dep_federal.columns)
    if missing:
        raise ValueError(f"df_dep_federal sem colunas: {sorted(missing)}")

    if anos_alvo is None:
        anos_alvo = anos_presidenciais  # backward compat
    if anos_alvo is None:
        raise ValueError("informe `anos_alvo` (lista de anos do eixo).")

    anos_eixo = sorted({int(a) for a in anos_alvo})
    mapa = {int(a): int(a) for a in anos_eixo}  # identity default
    if map_ano_para_federal:
        mapa.update({int(k): int(v) for k, v in map_ano_para_federal.items() if int(k) in mapa})

    anos_fed_necessarios = sorted(set(mapa.values()))
    df = df_dep_federal[df_dep_federal["ano"].isin(anos_fed_necessarios)].copy()
    df["id_municipio"] = df["id_municipio"].astype("string")
    df["votos"] = df["votos"].astype("int64")

    por_partido = (
        df.groupby(["ano", "id_municipio", "sigla_partido"], as_index=False)["votos"]
        .sum()
    )
    totais = (
        df.groupby(["ano", "id_municipio"], as_index=False)["votos"]
        .sum()
        .rename(columns={"votos": "_total"})
    )
    agg = por_partido.merge(totais, on=["ano", "id_municipio"], how="left")
    agg["share_dep_federal_partido"] = (agg["votos"] / agg["_total"]).astype("float64")
    agg = agg.drop(columns=["votos", "_total"])
    # `agg` está em coord federal (`ano` = ano federal). Faz broadcast reverso:
    # para cada ano_eixo, pega a share do ano_federal mapeado.
    remap = pd.DataFrame(
        {ano_col: list(mapa.keys()), "ano": list(mapa.values())}
    )
    out = remap.merge(agg, on="ano", how="left").drop(columns=["ano"])
    return out[[ano_col, "id_municipio", "sigla_partido", "share_dep_federal_partido"]].reset_index(drop=True)


__all__ = [
    "MUNICIPAL_TO_ESTADUAL_ANTERIOR",
    "governador_vencedor_por_eleicao",
    "alinhamento_partido_com_governador",
    "share_dep_federal_por_partido",
]
