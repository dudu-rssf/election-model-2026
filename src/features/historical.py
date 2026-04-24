"""
src.features.historical — features históricas (lag/swing/volatilidade).

Saída: DataFrame ao nível (ano_presidencial × id_municipio × sigla_partido) com:

    lag_share_1t                — share do partido na presidencial anterior.
    lag2_share_1t               — share do partido duas presidenciais atrás.
    swing_share_1t              — share atual − lag.
    volatilidade_partido        — desvio padrão dos shares do partido no município
                                  considerando as eleições estritamente anteriores.

As features são calculadas a partir de `presidencial_long` (Fase 2) e juntadas
no long original por (ano_presidencial, id_municipio, sigla_partido).

Notas:
  * Partidos ausentes no município num ano são tratados como share = 0 para
    o cálculo, **mas** o lag/swing só é preenchido se existe registro direto
    do partido no ano anterior (evita ruído em partidos que não concorreram).
    Essa escolha é documentada para revisão; ver `SHARE_ZERO_SE_AUSENTE`.
  * Volatilidade usa `ddof=0` (populacional, amostra inteira de eleições).
  * O primeiro ano do histórico não tem lag → NaN.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Política de preenchimento para anos em que o partido não teve candidato no
# município. True = assume share 0; False = deixa NaN.
# Para swing/volatilidade usamos True (simplifica continuidade do sinal).
# Para lag_share_1t, preferimos False (NaN): lag de um partido que não
# concorreu no ano anterior não tem interpretação direta.
SHARE_ZERO_SE_AUSENTE = True


def _pres_long_wide_partido(pres_long: pd.DataFrame) -> pd.DataFrame:
    """Retorna DataFrame wide: cada (ano, mun, partido) tem um share (0 se ausente).

    Reduz ruído quando o mesmo partido lança mais de um candidato no mesmo ano
    (raro em presidencial, mas possível em candidatos avulsos): soma shares.
    """
    required = {"ano_presidencial", "id_municipio", "sigla_partido", "share_1t"}
    missing = required - set(pres_long.columns)
    if missing:
        raise ValueError(f"pres_long sem colunas: {sorted(missing)}")

    df = pres_long.copy()
    df["id_municipio"] = df["id_municipio"].astype("string")

    # Soma por (ano, mun, partido) — trata múltiplos candidatos do mesmo partido
    agg = (
        df.groupby(
            ["ano_presidencial", "id_municipio", "sigla_partido"],
            as_index=False,
        )["share_1t"]
        .sum()
    )
    return agg


def _expand_universo_partido(agg: pd.DataFrame) -> pd.DataFrame:
    """Expande para todas combinações (ano, mun, partido_universo), preenchendo 0.

    Universo de partidos = partidos que concorreram em pelo menos um ano no mun.
    """
    if not SHARE_ZERO_SE_AUSENTE:
        return agg

    # Universo por município: partidos que apareceram em algum ano
    part_por_mun = (
        agg.drop_duplicates(subset=["id_municipio", "sigla_partido"])[
            ["id_municipio", "sigla_partido"]
        ]
    )
    anos_por_mun = (
        agg.drop_duplicates(subset=["id_municipio", "ano_presidencial"])[
            ["id_municipio", "ano_presidencial"]
        ]
    )
    universo = anos_por_mun.merge(part_por_mun, on="id_municipio")
    out = universo.merge(
        agg, on=["ano_presidencial", "id_municipio", "sigla_partido"], how="left"
    )
    out["share_1t"] = out["share_1t"].fillna(0.0).astype("float64")
    return out


def features_historical(
    pres_long: pd.DataFrame,
    anos_presidenciais: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Calcula features históricas por (ano, mun, partido).

    Args:
        pres_long: presidencial_long (Fase 2).
        anos_presidenciais: filtro opcional — por padrão usa os anos presentes
            em `pres_long`. Útil para garantir ordem consistente.

    Returns:
        DataFrame com colunas:
            ano_presidencial, id_municipio, sigla_partido,
            lag_share_1t, lag2_share_1t, swing_share_1t, volatilidade_partido.
    """
    agg = _pres_long_wide_partido(pres_long)

    if anos_presidenciais is not None:
        anos_set = {int(a) for a in anos_presidenciais}
        agg = agg[agg["ano_presidencial"].isin(anos_set)].copy()

    exp = _expand_universo_partido(agg)
    exp = exp.sort_values(["id_municipio", "sigla_partido", "ano_presidencial"])

    # Lags (shift dentro do grupo mun × partido na ordem temporal)
    grp = exp.groupby(["id_municipio", "sigla_partido"], sort=False)
    exp["lag_share_1t"] = grp["share_1t"].shift(1)
    exp["lag2_share_1t"] = grp["share_1t"].shift(2)
    exp["swing_share_1t"] = exp["share_1t"] - exp["lag_share_1t"]

    # Volatilidade: std das eleições *estritamente* anteriores
    def _vol_expandida(s: pd.Series) -> pd.Series:
        # shift primeiro para não incluir o ano corrente
        return s.shift(1).expanding(min_periods=2).std(ddof=0)

    exp["volatilidade_partido"] = grp["share_1t"].transform(_vol_expandida).astype("float64")

    cols = [
        "ano_presidencial",
        "id_municipio",
        "sigla_partido",
        "lag_share_1t",
        "lag2_share_1t",
        "swing_share_1t",
        "volatilidade_partido",
    ]
    return exp[cols].reset_index(drop=True)


__all__ = ["features_historical", "SHARE_ZERO_SE_AUSENTE"]
