"""
src.features.target — tabela long de votação presidencial (o alvo do modelo).

Convenção de saída (`data/interim/presidencial_long.parquet`):

    ano_presidencial int
    id_municipio     str
    sigla_uf         str
    numero_candidato int
    nome_candidato   str
    sigla_partido    str
    votos            int
    total_votos_mun  int
    share_1t         float   # votos / total_votos_mun

Esta tabela vira o alvo das predições em fase 5/6 (uma linha por
(candidato, município, ano)). Métricas agregadas (nacional, UF) derivam
daqui ponderando por total_votos_mun.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def construir_presidencial_long(
    df_pres: pd.DataFrame,
    df_candidatos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Monta a tabela long presidencial.

    `nome_candidato` não precisa estar em `df_pres` (a tabela
    `resultados_candidato_municipio` do BD atual não traz essa coluna).
    Se `df_candidatos` for fornecido, fazemos merge em (ano, numero_candidato)
    para anexar nome. Caso contrário, `nome_candidato` fica como pd.NA.
    """
    required = {
        "ano",
        "sigla_uf",
        "id_municipio",
        "numero_candidato",
        "sigla_partido",
        "votos",
    }
    missing = required - set(df_pres.columns)
    if missing:
        raise ValueError(f"df_pres sem colunas: {sorted(missing)}")

    df = df_pres.copy()
    df["id_municipio"] = df["id_municipio"].astype("string")
    df["votos"] = df["votos"].astype("int64")

    # Anexa nome_candidato. Prioridade:
    #   1) já existe no df_pres -> usa direto
    #   2) df_candidatos fornecido -> merge em (ano, numero_candidato)
    #   3) nada -> NA
    if "nome_candidato" not in df.columns:
        if df_candidatos is not None and len(df_candidatos) > 0:
            cand = df_candidatos[
                ["ano", "numero_candidato", "nome_candidato"]
            ].drop_duplicates(subset=["ano", "numero_candidato"], keep="first")
            df = df.merge(cand, on=["ano", "numero_candidato"], how="left")
        else:
            df["nome_candidato"] = pd.NA

    totais = (
        df.groupby(["ano", "id_municipio"], as_index=False)["votos"]
        .sum()
        .rename(columns={"votos": "total_votos_mun"})
    )

    out = df.merge(totais, on=["ano", "id_municipio"], how="left")
    out["share_1t"] = (out["votos"] / out["total_votos_mun"]).astype("float64")

    # Validação leve: share deve estar em [0, 1]
    bad = ((out["share_1t"] < 0) | (out["share_1t"] > 1)).sum()
    if bad:
        raise ValueError(f"share_1t fora de [0,1] em {int(bad)} linha(s)")

    out = out.rename(columns={"ano": "ano_presidencial"})

    cols = [
        "ano_presidencial",
        "sigla_uf",
        "id_municipio",
        "numero_candidato",
        "nome_candidato",
        "sigla_partido",
        "votos",
        "total_votos_mun",
        "share_1t",
    ]
    return out[cols].reset_index(drop=True)


__all__ = ["construir_presidencial_long"]
