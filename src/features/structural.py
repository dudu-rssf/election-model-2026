"""
src.features.structural — features estruturais do município.

Saída: uma linha por (ano_presidencial, id_municipio) com:
    sigla_uf         str      — cópia do painel
    regiao           str      — cópia do painel
    capital_uf       bool     — cópia do painel
    log_eleitorado   float    — log1p(total_votos_mun) da eleição presidencial do ano
    porte            str      — tercil do log_eleitorado em 'pequeno'/'medio'/'grande'

Essas features são invariantes por candidato — serão broadcast para cada
linha do long table durante a consolidação em `scripts/03_features.py`.

Notas:
  * "eleitorado" é aproximado como `total_votos_mun` da própria eleição
    presidencial (total de votos válidos no município naquele ano).
    A Base dos Dados tem `eleitorado_municipio` como tabela separada mas
    não a ingerimos — a aproximação é boa porque o TSE-correlação entre
    votantes e eleitorado é > 0.95 em nível municipal.
  * `porte` é computado por qcut nos log_eleitorado dentro de cada ano
    presidencial. Dessa forma, o rótulo é relativo ao tamanho típico
    daquele ano (evita que "pequeno" de 2022 seja equivalente a "médio"
    de 2014 quando a população cresceu).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _eleitorado_por_mun_ano(pres_long: pd.DataFrame) -> pd.DataFrame:
    """Extrai total_votos_mun (único por ano × mun) do long table."""
    required = {"ano_presidencial", "id_municipio", "total_votos_mun"}
    missing = required - set(pres_long.columns)
    if missing:
        raise ValueError(f"pres_long sem colunas: {sorted(missing)}")

    # total_votos_mun é constante dentro de (ano, mun); first() é suficiente
    out = (
        pres_long.groupby(["ano_presidencial", "id_municipio"], as_index=False)[
            "total_votos_mun"
        ]
        .first()
        .rename(columns={"total_votos_mun": "_eleitorado"})
    )
    return out


def _porte_por_ano(df: pd.DataFrame, col: str = "log_eleitorado") -> pd.Series:
    """Tercis do `col` dentro de cada ano_presidencial."""
    labels = ["pequeno", "medio", "grande"]

    def _q(s: pd.Series) -> pd.Series:
        # qcut falha silenciosamente se todos os valores forem iguais
        try:
            return pd.qcut(s, q=3, labels=labels, duplicates="drop")
        except ValueError:
            return pd.Series([labels[0]] * len(s), index=s.index)

    out = df.groupby("ano_presidencial", group_keys=False)[col].apply(_q)
    return out.astype("string")


def features_structural(
    painel: pd.DataFrame,
    pres_long: pd.DataFrame,
) -> pd.DataFrame:
    """Monta o bloco de features estruturais (uma linha por mun × ano).

    Args:
        painel: painel_mestre (Fase 2).
        pres_long: presidencial_long (Fase 2) — usado para derivar log_eleitorado.

    Returns:
        DataFrame com ['ano_presidencial', 'id_municipio', 'sigla_uf',
        'regiao', 'capital_uf', 'log_eleitorado', 'porte'].
    """
    required = {"ano_presidencial", "id_municipio", "sigla_uf", "regiao", "capital_uf"}
    missing = required - set(painel.columns)
    if missing:
        raise ValueError(f"painel sem colunas: {sorted(missing)}")

    base = painel[["ano_presidencial", "id_municipio", "sigla_uf", "regiao", "capital_uf"]].copy()
    base["id_municipio"] = base["id_municipio"].astype("string")

    eleitorado = _eleitorado_por_mun_ano(pres_long)
    eleitorado["id_municipio"] = eleitorado["id_municipio"].astype("string")

    out = base.merge(eleitorado, on=["ano_presidencial", "id_municipio"], how="left")

    nulls = int(out["_eleitorado"].isna().sum())
    if nulls:
        logger.warning(
            "features_structural: %d município(s) sem dado presidencial; log_eleitorado=NaN",
            nulls,
        )
    out["log_eleitorado"] = np.log1p(out["_eleitorado"].fillna(0)).astype("float64")
    out.loc[out["_eleitorado"].isna(), "log_eleitorado"] = np.nan

    out["porte"] = _porte_por_ano(out, col="log_eleitorado")

    out = out.drop(columns=["_eleitorado"])
    cols = [
        "ano_presidencial",
        "id_municipio",
        "sigla_uf",
        "regiao",
        "capital_uf",
        "log_eleitorado",
        "porte",
    ]
    return out[cols].reset_index(drop=True)


__all__ = ["features_structural"]
