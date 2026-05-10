"""
src.features.historical — features históricas (lag/swing/volatilidade).

Saída: DataFrame ao nível (ano × id_municipio × sigla_partido) com:

    lag_share_1t                — share do partido na eleição anterior.
    lag_share_1t_sucessao       — idem, mas agrupando por sigla canônica
                                  (ver src.features.partido_sucessao). Para
                                  partidos sem mapeamento, igual ao lag_share_1t.
    lag_share_1t_uf_sucessao    — média ponderada (por eleitorado) do
                                  lag_share_1t_sucessao em todos os muns da
                                  UF. Capta tendência regional do partido
                                  (com sucessão) suavizando ruído municipal.
    lag2_share_1t               — share do partido duas eleições atrás.
    swing_share_1t              — share atual − lag.
    volatilidade_partido        — desvio padrão dos shares do partido no município
                                  considerando as eleições estritamente anteriores.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import pandas as pd

from src.features.partido_sucessao import aplicar_sucessao

logger = logging.getLogger(__name__)


SHARE_ZERO_SE_AUSENTE = True


def _long_wide_partido(long: pd.DataFrame, ano_col: str) -> pd.DataFrame:
    required = {ano_col, "id_municipio", "sigla_partido", "share_1t"}
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"long sem colunas: {sorted(missing)}")

    df = long.copy()
    df["id_municipio"] = df["id_municipio"].astype("string")

    agg = (
        df.groupby(
            [ano_col, "id_municipio", "sigla_partido"],
            as_index=False,
        )["share_1t"]
        .sum()
    )
    return agg


def _expand_universo_partido(agg: pd.DataFrame, ano_col: str) -> pd.DataFrame:
    if not SHARE_ZERO_SE_AUSENTE:
        return agg

    part_por_mun = (
        agg.drop_duplicates(subset=["id_municipio", "sigla_partido"])[
            ["id_municipio", "sigla_partido"]
        ]
    )
    anos_por_mun = (
        agg.drop_duplicates(subset=["id_municipio", ano_col])[
            ["id_municipio", ano_col]
        ]
    )
    universo = anos_por_mun.merge(part_por_mun, on="id_municipio")
    out = universo.merge(
        agg, on=[ano_col, "id_municipio", "sigla_partido"], how="left"
    )
    out["share_1t"] = out["share_1t"].fillna(0.0).astype("float64")
    return out


def _lag_por_sigla_canonica(
    exp: pd.DataFrame,
    sucessoes: Mapping[str, Mapping[int, str]] | None,
    ano_col: str,
) -> pd.Series:
    df = aplicar_sucessao(
        exp,
        sucessoes,
        col_partido="sigla_partido",
        col_ano=ano_col,
        col_saida="sigla_canonica",
    )

    canon = (
        df.groupby(
            [ano_col, "id_municipio", "sigla_canonica"],
            as_index=False,
            dropna=False,
        )["share_1t"]
        .sum()
    )

    canon = canon.sort_values(["id_municipio", "sigla_canonica", ano_col])
    grp = canon.groupby(["id_municipio", "sigla_canonica"], sort=False, dropna=False)
    canon["lag_canon"] = grp["share_1t"].shift(1)

    merged = df.merge(
        canon[[ano_col, "id_municipio", "sigla_canonica", "lag_canon"]],
        on=[ano_col, "id_municipio", "sigla_canonica"],
        how="left",
    )
    return pd.Series(
        merged["lag_canon"].to_numpy(dtype="float64"),
        index=df.index,
        name="lag_share_1t_sucessao",
    )


def _adicionar_lag_uf_sucessao(
    exp: pd.DataFrame,
    long: pd.DataFrame,
    *,
    ano_col: str,
) -> pd.DataFrame:
    """Anexa `lag_share_1t_uf_sucessao` em `exp`.

    Para cada (ano, sigla_uf, sigla_partido), computa a média ponderada
    do `lag_share_1t_sucessao` em todos os municípios da UF, ponderando
    por `total_votos_mun`. Valor NaN do lag não contribui.
    """
    if "lag_share_1t_sucessao" not in exp.columns:
        raise ValueError("exp precisa ter lag_share_1t_sucessao computado")

    mun_uf = (
        long[["id_municipio", "sigla_uf"]]
        .drop_duplicates(subset=["id_municipio"])
    )
    mun_uf["id_municipio"] = mun_uf["id_municipio"].astype("string")

    if "total_votos_mun" in long.columns:
        mun_votos = (
            long.groupby([ano_col, "id_municipio"], as_index=False)[
                "total_votos_mun"
            ]
            .first()
        )
        mun_votos["id_municipio"] = mun_votos["id_municipio"].astype("string")
    else:
        logger.warning(
            "long sem total_votos_mun — lag_share_1t_uf_sucessao usa peso uniforme"
        )
        mun_votos = (
            long[[ano_col, "id_municipio"]]
            .drop_duplicates()
            .assign(total_votos_mun=1.0)
        )
        mun_votos["id_municipio"] = mun_votos["id_municipio"].astype("string")

    exp = exp.copy()
    exp["id_municipio"] = exp["id_municipio"].astype("string")
    exp = exp.merge(mun_uf, on="id_municipio", how="left")
    exp = exp.merge(mun_votos, on=[ano_col, "id_municipio"], how="left")

    mask = exp["lag_share_1t_sucessao"].notna()
    sub = exp[mask].copy()
    sub["_wp"] = sub["lag_share_1t_sucessao"] * sub["total_votos_mun"]
    agg = (
        sub.groupby(
            [ano_col, "sigla_uf", "sigla_partido"], as_index=False, observed=True
        )
        .agg(_wp_sum=("_wp", "sum"), _w_sum=("total_votos_mun", "sum"))
    )
    agg["lag_share_1t_uf_sucessao"] = agg["_wp_sum"] / agg["_w_sum"]
    agg = agg[[ano_col, "sigla_uf", "sigla_partido", "lag_share_1t_uf_sucessao"]]

    exp = exp.merge(agg, on=[ano_col, "sigla_uf", "sigla_partido"], how="left")
    exp = exp.drop(columns=["sigla_uf", "total_votos_mun"], errors="ignore")

    n_com_lag = int(exp["lag_share_1t_uf_sucessao"].notna().sum())
    pct = n_com_lag / len(exp) * 100 if len(exp) > 0 else 0.0
    logger.info(
        "lag_share_1t_uf_sucessao: %d/%d linhas com valor (%.1f%%)",
        n_com_lag, len(exp), pct,
    )
    return exp


def features_historical(
    long: pd.DataFrame,
    anos: Iterable[int] | None = None,
    sucessoes: Mapping[str, Mapping[int, str]] | None = None,
    ano_col: str = "ano_presidencial",
) -> pd.DataFrame:
    agg = _long_wide_partido(long, ano_col=ano_col)

    if anos is not None:
        anos_set = {int(a) for a in anos}
        agg = agg[agg[ano_col].isin(anos_set)].copy()

    exp = _expand_universo_partido(agg, ano_col=ano_col)
    exp = exp.sort_values(["id_municipio", "sigla_partido", ano_col])

    grp = exp.groupby(["id_municipio", "sigla_partido"], sort=False)
    exp["lag_share_1t"] = grp["share_1t"].shift(1)
    exp["lag2_share_1t"] = grp["share_1t"].shift(2)
    exp["swing_share_1t"] = exp["share_1t"] - exp["lag_share_1t"]

    def _vol_expandida(s: pd.Series) -> pd.Series:
        return s.shift(1).expanding(min_periods=2).std(ddof=0)

    exp["volatilidade_partido"] = grp["share_1t"].transform(_vol_expandida).astype("float64")
    exp["lag_share_1t_sucessao"] = _lag_por_sigla_canonica(exp, sucessoes, ano_col=ano_col)

    exp = _adicionar_lag_uf_sucessao(exp, long, ano_col=ano_col)

    cols = [
        ano_col,
        "id_municipio",
        "sigla_partido",
        "lag_share_1t",
        "lag_share_1t_sucessao",
        "lag_share_1t_uf_sucessao",
        "lag2_share_1t",
        "swing_share_1t",
        "volatilidade_partido",
    ]
    return exp[cols].reset_index(drop=True)


__all__ = ["features_historical", "SHARE_ZERO_SE_AUSENTE"]
