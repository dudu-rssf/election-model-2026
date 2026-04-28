"""
src.features.historical — features históricas (lag/swing/volatilidade).

Saída: DataFrame ao nível (ano × id_municipio × sigla_partido) com:

    lag_share_1t                — share do partido na eleição anterior.
    lag_share_1t_sucessao       — idem, mas agrupando por sigla canônica
                                  (ver src.features.partido_sucessao). Para
                                  partidos sem mapeamento, igual ao lag_share_1t.
    lag2_share_1t               — share do partido duas eleições atrás.
    swing_share_1t              — share atual − lag.
    volatilidade_partido        — desvio padrão dos shares do partido no município
                                  considerando as eleições estritamente anteriores.

As features são calculadas a partir de `<eixo>_long` (Fase 2) e juntadas no
long original por (ano, id_municipio, sigla_partido).

A coluna de ano é parametrizada via `ano_col` — default `'ano_presidencial'`
(Fase 3). Para o modelo de prefeito (Fase 4.5) passe `ano_col='ano_municipal'`.

Notas:
  * Partidos ausentes no município num ano são tratados como share = 0 para
    o cálculo, **mas** o lag/swing só é preenchido se existe registro direto
    do partido no ano anterior (evita ruído em partidos que não concorreram).
    Essa escolha é documentada para revisão; ver `SHARE_ZERO_SE_AUSENTE`.
  * Volatilidade usa `ddof=0` (populacional, amostra inteira de eleições).
  * O primeiro ano do histórico não tem lag → NaN.
  * `lag_share_1t_sucessao` é deixado paralelo ao `lag_share_1t` propositalmente:
    o modelo escolhe qual pesar mais. Quando não há `sucessoes`, as duas
    colunas são idênticas.
"""
from __future__ import annotations

import logging
from typing import Iterable, Mapping

import pandas as pd

from src.features.partido_sucessao import aplicar_sucessao

logger = logging.getLogger(__name__)


SHARE_ZERO_SE_AUSENTE = True


def _long_wide_partido(long: pd.DataFrame, ano_col: str) -> pd.DataFrame:
    """DataFrame wide: cada (ano, mun, partido) tem um share (0 se ausente).

    Reduz ruído quando o mesmo partido lança mais de um candidato no mesmo ano
    (raro em presidencial, mais comum em eleições locais): soma shares.
    """
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
    """Expande para todas combinações (ano, mun, partido_universo), preenchendo 0.

    Universo de partidos = partidos que concorreram em pelo menos um ano no mun.
    """
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
    """Calcula lag_share_1t agrupando por sigla canônica ao invés de sigla bruta.

    Canonicalização: dado (partido, ano), resolver para sigla predecessora quando
    houver mapeamento — senão, sigla inalterada. Depois, re-agrega share_1t por
    (ano, mun, canonical) — somando quando múltiplas siglas canonicalizam juntas
    — e calcula o shift(1) na ordem temporal dentro do grupo (mun, canonical).
    """
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


def features_historical(
    long: pd.DataFrame,
    anos: Iterable[int] | None = None,
    sucessoes: Mapping[str, Mapping[int, str]] | None = None,
    ano_col: str = "ano_presidencial",
) -> pd.DataFrame:
    """Calcula features históricas por (ano, mun, partido).

    Args:
        long: tabela long (Fase 2) — presidencial_long ou prefeito_long.
        anos: filtro opcional — por padrão usa os anos presentes em `long`.
        sucessoes: dict de mapeamento (sigla → ano → predecessor) vindo do
            config.yaml. Se None ou {}, `lag_share_1t_sucessao` fica idêntico
            a `lag_share_1t`.
        ano_col: nome da coluna do eixo temporal — default `'ano_presidencial'`
            (Fase 3). Para Fase 4.5 (prefeito) usar `'ano_municipal'`.

    Returns:
        DataFrame com colunas:
            <ano_col>, id_municipio, sigla_partido,
            lag_share_1t, lag_share_1t_sucessao, lag2_share_1t,
            swing_share_1t, volatilidade_partido.
    """
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

    cols = [
        ano_col,
        "id_municipio",
        "sigla_partido",
        "lag_share_1t",
        "lag_share_1t_sucessao",
        "lag2_share_1t",
        "swing_share_1t",
        "volatilidade_partido",
    ]
    return exp[cols].reset_index(drop=True)


__all__ = ["features_historical", "SHARE_ZERO_SE_AUSENTE"]
