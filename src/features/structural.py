"""
src.features.structural — features estruturais do município.

Saída: uma linha por (ano × id_municipio) com:
    sigla_uf         str      — cópia do painel
    regiao           str      — cópia do painel
    capital_uf       bool     — cópia do painel
    log_eleitorado   float    — log1p(total_votos_mun) da eleição do ano
    porte            str      — tercil do log_eleitorado em 'pequeno'/'medio'/'grande'

Essas features são invariantes por candidato — serão broadcast para cada
linha do long table durante a consolidação em `scripts/03_features*.py`.

O módulo é agnóstico ao eixo: `ano_col` (default `'ano_presidencial'`)
aponta para a coluna do eixo temporal no `painel` e no `long`. Para o
modelo prefeito (Fase 4.5), passar `ano_col='ano_municipal'`.

Notas:
  * "eleitorado" é aproximado como `total_votos_mun` da própria eleição
    (total de votos válidos no município naquele ano). Correlação com
    eleitorado real é > 0.95 a nível municipal.
  * `porte` é computado por qcut nos log_eleitorado dentro de cada ano,
    para que a escala seja relativa ao ano em questão (evita comparação
    espúria entre 2014 e 2022 quando a base cresceu).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _eleitorado_por_mun_ano(long: pd.DataFrame, ano_col: str) -> pd.DataFrame:
    """Extrai total_votos_mun (único por ano × mun) do long table."""
    required = {ano_col, "id_municipio", "total_votos_mun"}
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"long sem colunas: {sorted(missing)}")

    out = (
        long.groupby([ano_col, "id_municipio"], as_index=False)[
            "total_votos_mun"
        ]
        .first()
        .rename(columns={"total_votos_mun": "_eleitorado"})
    )
    return out


def _porte_por_ano(df: pd.DataFrame, ano_col: str, col: str = "log_eleitorado") -> pd.Series:
    """Tercis do `col` dentro de cada ano (`ano_col`)."""
    labels = ["pequeno", "medio", "grande"]

    def _q(s: pd.Series) -> pd.Series:
        try:
            return pd.qcut(s, q=3, labels=labels, duplicates="drop")
        except ValueError:
            return pd.Series([labels[0]] * len(s), index=s.index)

    out = df.groupby(ano_col, group_keys=False)[col].apply(_q)
    return out.astype("string")


def features_structural(
    painel: pd.DataFrame,
    long: pd.DataFrame,
    ano_col: str = "ano_presidencial",
) -> pd.DataFrame:
    """Monta o bloco de features estruturais (uma linha por mun × ano).

    Args:
        painel: painel_mestre (Fase 2).
        long: tabela long (presidencial_long ou prefeito_long). Usada para
            derivar log_eleitorado via total_votos_mun.
        ano_col: eixo temporal — default `'ano_presidencial'`; em Fase 4.5
            passar `'ano_municipal'`.

    Returns:
        DataFrame com [ano_col, 'id_municipio', 'sigla_uf', 'regiao',
        'capital_uf', 'log_eleitorado', 'porte'].
    """
    required = {ano_col, "id_municipio", "sigla_uf", "regiao", "capital_uf"}
    missing = required - set(painel.columns)
    if missing:
        raise ValueError(f"painel sem colunas: {sorted(missing)}")

    base = painel[[ano_col, "id_municipio", "sigla_uf", "regiao", "capital_uf"]].copy()
    base["id_municipio"] = base["id_municipio"].astype("string")

    eleitorado = _eleitorado_por_mun_ano(long, ano_col=ano_col)
    eleitorado["id_municipio"] = eleitorado["id_municipio"].astype("string")

    out = base.merge(eleitorado, on=[ano_col, "id_municipio"], how="left")

    nulls = int(out["_eleitorado"].isna().sum())
    if nulls:
        logger.warning(
            "features_structural: %d município(s) sem dado do ano; log_eleitorado=NaN",
            nulls,
        )
    out["log_eleitorado"] = np.log1p(out["_eleitorado"].fillna(0)).astype("float64")
    out.loc[out["_eleitorado"].isna(), "log_eleitorado"] = np.nan

    out["porte"] = _porte_por_ano(out, ano_col=ano_col, col="log_eleitorado")

    out = out.drop(columns=["_eleitorado"])
    cols = [
        ano_col,
        "id_municipio",
        "sigla_uf",
        "regiao",
        "capital_uf",
        "log_eleitorado",
        "porte",
    ]
    return out[cols].reset_index(drop=True)


__all__ = ["features_structural"]
