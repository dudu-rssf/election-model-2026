"""
src.features.local_power — features de poder local (prefeito).

Separamos em dois blocos:

(A) Features invariantes por candidato (nível mun × ano):
    * `margem_prefeito` — margem de vitória do prefeito vigente (1º vs 2º).
    * `primeiro_mandato_prefeito` — 1 se o partido do prefeito atual é
      diferente do partido do prefeito da eleição municipal anterior.
    * `share_prefeito_local` — `mayor_share_1t` do prefeito vigente.

(B) Features partido-específicas (nível mun × ano × sigla_partido):
    * `alinhado_prefeito_partido` — 1 se `sigla_partido == mayor_partido`.
    * `alinhado_prefeito_coligacao` — 1 se `sigla_partido` consta da
      composição da coligação estadual/municipal do prefeito.

Esse split simplifica a consolidação: o bloco (A) dá um left-join sobre
(ano, id_municipio); o bloco (B) sobre (ano, id_municipio, sigla_partido).

Eixo configurável (`ano_col` + `ano_eleicao_anterior_col`):
  * Eixo presidencial (Fase 3, default):
      ano_col='ano_presidencial', ano_eleicao_anterior_col='ano_eleicao_municipal'
      (X-2 — prefeito vigente no ano presidencial).
  * Eixo municipal (Fase 4.5):
      ano_col='ano_municipal', ano_eleicao_anterior_col='ano_eleicao_municipal_anterior'
      (X-4 — prefeito vigente no momento da próxima eleição municipal).
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Bloco A — features ao nível (mun × ano_presidencial)
# ------------------------------------------------------------
def _primeiro_mandato_flag(
    painel: pd.DataFrame,
    ano_eleicao_anterior_col: str,
) -> pd.Series:
    """Para cada linha do painel, verifica se o partido do prefeito atual é
    diferente do partido vigente na eleição municipal anterior no mesmo
    município. Retorna Series alinhada ao painel, com NA onde não há
    histórico anterior nos dados baixados.
    """
    df = painel[["id_municipio", ano_eleicao_anterior_col, "mayor_partido"]].copy()
    df["id_municipio"] = df["id_municipio"].astype("string")

    unicos = (
        df.dropna(subset=[ano_eleicao_anterior_col])
        .drop_duplicates(subset=["id_municipio", ano_eleicao_anterior_col])
        .sort_values(["id_municipio", ano_eleicao_anterior_col])
    )
    unicos["mayor_partido_anterior"] = unicos.groupby("id_municipio")["mayor_partido"].shift(1)

    merged = painel[["id_municipio", ano_eleicao_anterior_col]].merge(
        unicos[["id_municipio", ano_eleicao_anterior_col, "mayor_partido_anterior"]],
        on=["id_municipio", ano_eleicao_anterior_col],
        how="left",
    )

    flag = pd.Series(pd.NA, index=painel.index, dtype="Int64")
    known = merged["mayor_partido_anterior"].notna() & painel["mayor_partido"].notna()
    flag.loc[known.values] = (
        (painel.loc[known.values, "mayor_partido"].values
         != merged.loc[known.values, "mayor_partido_anterior"].values)
    ).astype("int64")
    return flag


def features_local_mun_ano(
    painel: pd.DataFrame,
    *,
    ano_col: str = "ano_presidencial",
    ano_eleicao_anterior_col: str = "ano_eleicao_municipal",
) -> pd.DataFrame:
    """Bloco A: features broadcastáveis (mun × ano).

    Args:
        painel: painel_mestre. Espera [ano_col, id_municipio, mayor_partido,
            mayor_share_1t, mayor_margem_1t, ano_eleicao_anterior_col].
        ano_col: eixo temporal — default `'ano_presidencial'` (Fase 3);
            para Fase 4.5 usar `'ano_municipal'`.
        ano_eleicao_anterior_col: nome da coluna com o ano da eleição
            municipal que deu origem ao prefeito vigente. No painel
            presidencial = 'ano_eleicao_municipal' (X-2); no municipal =
            'ano_eleicao_municipal_anterior' (X-4).
    """
    required = {
        ano_col,
        "id_municipio",
        "mayor_partido",
        "mayor_share_1t",
        "mayor_margem_1t",
        ano_eleicao_anterior_col,
    }
    missing = required - set(painel.columns)
    if missing:
        raise ValueError(f"painel sem colunas: {sorted(missing)}")

    out = pd.DataFrame(
        {
            ano_col: painel[ano_col].astype("int64"),
            "id_municipio": painel["id_municipio"].astype("string"),
            "share_prefeito_local": painel["mayor_share_1t"].astype("float64"),
            "margem_prefeito": painel["mayor_margem_1t"].astype("float64"),
        }
    )
    out["primeiro_mandato_prefeito"] = _primeiro_mandato_flag(
        painel, ano_eleicao_anterior_col=ano_eleicao_anterior_col,
    ).values
    return out.reset_index(drop=True)


# ------------------------------------------------------------
# Bloco B — features partido-específicas (mun × ano × partido)
# ------------------------------------------------------------
def _split_coligacao(s: str | None) -> list[str]:
    """Quebra 'PT:PCdoB:PSB' em ['PT','PCdoB','PSB']. NA/empty -> []."""
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return []
    if pd.isna(s):
        return []
    partes = [p.strip() for p in str(s).split(":") if p and p.strip()]
    return partes


def alinhamento_partido_com_prefeito(
    painel: pd.DataFrame,
    partidos: Iterable[str],
    *,
    ano_col: str = "ano_presidencial",
) -> pd.DataFrame:
    """Para cada (ano × mun × partido), marca alinhamento com o prefeito.

    Args:
        painel: painel_mestre com [ano_col, id_municipio, mayor_partido,
            mayor_coligacao].
        partidos: lista de partidos a expandir (tipicamente o conjunto
            único de `sigla_partido` no long target).
        ano_col: eixo temporal — default `'ano_presidencial'` (Fase 3);
            para Fase 4.5 passar `'ano_municipal'`.

    Returns:
        DataFrame com [ano_col, id_municipio, sigla_partido,
        alinhado_prefeito_partido, alinhado_prefeito_coligacao].
    """
    required = {ano_col, "id_municipio", "mayor_partido", "mayor_coligacao"}
    missing = required - set(painel.columns)
    if missing:
        raise ValueError(f"painel sem colunas: {sorted(missing)}")

    partidos_list = sorted({str(p) for p in partidos if pd.notna(p)})
    if not partidos_list:
        raise ValueError("lista de partidos vazia")

    base = painel[[ano_col, "id_municipio", "mayor_partido", "mayor_coligacao"]].copy()
    base["id_municipio"] = base["id_municipio"].astype("string")
    base["_coligacao_set"] = base["mayor_coligacao"].apply(lambda s: set(_split_coligacao(s)))

    base["_k"] = 1
    partidos_df = pd.DataFrame({"sigla_partido": partidos_list, "_k": 1})
    out = base.merge(partidos_df, on="_k").drop(columns="_k")

    mayor_na = out["mayor_partido"].isna()
    out["alinhado_prefeito_partido"] = (
        (out["sigla_partido"] == out["mayor_partido"]).astype("Int64")
    )
    out.loc[mayor_na, "alinhado_prefeito_partido"] = pd.NA

    out["alinhado_prefeito_coligacao"] = out.apply(
        lambda r: pd.NA if not r["_coligacao_set"] else int(r["sigla_partido"] in r["_coligacao_set"]),
        axis=1,
    ).astype("Int64")

    cols = [
        ano_col,
        "id_municipio",
        "sigla_partido",
        "alinhado_prefeito_partido",
        "alinhado_prefeito_coligacao",
    ]
    return out[cols].reset_index(drop=True)


__all__ = [
    "features_local_mun_ano",
    "alinhamento_partido_com_prefeito",
]
