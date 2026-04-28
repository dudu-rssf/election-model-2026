"""
src.features.target_prefeito — tabela long de votação para prefeito.

Análogo a `src.features.target` (presidencial), mas para o alvo da Fase 4.5
(modelo de prefeito). Convenção de saída
(`data/interim/prefeito_long.parquet`):

    ano_municipal    int
    sigla_uf         str
    id_municipio     str
    numero_candidato int
    nome_candidato   str
    sigla_partido    str
    votos            int
    total_votos_mun  int
    share_1t         float   # votos / total_votos_mun

Detalhe que difere do presidencial:
  * A tabela `candidatos` do BD para prefeito tem `id_municipio` — candidatos
    a prefeito são locais. Então o merge de nome é por
    (ano, sigla_uf, id_municipio, numero_candidato), não só (ano, numero).
    Usar a chave completa evita falsos matches (número 10 em SP não é o
    mesmo candidato que número 10 em MG).
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def construir_prefeito_long(
    df_prefeito: pd.DataFrame,
    df_candidatos: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Monta a tabela long de prefeito.

    Args:
        df_prefeito: resultados_candidato_municipio filtrado por cargo=prefeito
            (saída de `resultados_prefeito_sql`).
        df_candidatos: ficha dos candidatos (saída de `candidatos_prefeito_sql`).
            Se fornecido, anexa `nome_candidato`; senão, deixa NA.

    Returns:
        DataFrame long com share_1t por (candidato, município, ano municipal).
    """
    required = {
        "ano",
        "sigla_uf",
        "id_municipio",
        "numero_candidato",
        "sigla_partido",
        "votos",
    }
    missing = required - set(df_prefeito.columns)
    if missing:
        raise ValueError(f"df_prefeito sem colunas: {sorted(missing)}")

    df = df_prefeito.copy()
    df["id_municipio"] = df["id_municipio"].astype("string")
    df["votos"] = df["votos"].astype("int64")

    # Anexa nome_candidato. Chave completa (ano, uf, municipio, numero) porque
    # o número de candidato a prefeito só é único dentro do município.
    if "nome_candidato" not in df.columns:
        if df_candidatos is not None and len(df_candidatos) > 0:
            cand_cols = ["ano", "sigla_uf", "id_municipio", "numero_candidato", "nome_candidato"]
            cand = df_candidatos[cand_cols].copy()
            cand["id_municipio"] = cand["id_municipio"].astype("string")
            cand = cand.drop_duplicates(
                subset=["ano", "sigla_uf", "id_municipio", "numero_candidato"],
                keep="first",
            )
            df = df.merge(
                cand,
                on=["ano", "sigla_uf", "id_municipio", "numero_candidato"],
                how="left",
            )
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

    out = out.rename(columns={"ano": "ano_municipal"})

    cols = [
        "ano_municipal",
        "sigla_uf",
        "id_municipio",
        "numero_candidato",
        "nome_candidato",
        "sigla_partido",
        "votos",
        "total_votos_mun",
        "share_1t",
    ]

    logger.info(
        "construir_prefeito_long: %d linhas (%d municípios × %d anos)",
        len(out),
        out["id_municipio"].nunique(),
        out["ano_municipal"].nunique(),
    )
    return out[cols].reset_index(drop=True)


__all__ = ["construir_prefeito_long"]
