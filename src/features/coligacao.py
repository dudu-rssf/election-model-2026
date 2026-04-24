"""
src.features.coligacao — utilitário para reconstruir `composicao_coligacao`
a partir de `sequencial_coligacao` em casos onde o BD entregou com NULL.

Motivação:
  Em `br_tse_eleicoes.partidos`, a coluna `composicao_coligacao` vem
  pré-agregada pela Base dos Dados. Porém há buracos conhecidos:
    * governador 2010 — sequencial populado, composicao NULL (100% recup.)
    * prefeito 2012  — sequencial populado, composicao NULL (~3.4%)
    * prefeito 2020  — sequencial TAMBÉM NULL → não reconstruível aqui.

  Para os dois primeiros, a coligação pode ser reconstruída fazendo uma
  agregação por (chaves, sequencial_coligacao) e concatenando siglas —
  exatamente o que o BD faz para os demais anos.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def reconstruir_composicao_via_sequencial(
    df: pd.DataFrame,
    chaves_coligacao: Iterable[str],
) -> pd.DataFrame:
    """Preenche `composicao_coligacao` NULL a partir de `sequencial_coligacao`.

    Estratégia:
      1. Linhas com `sequencial_coligacao` não-NULL: agrupa por
         (*chaves_coligacao, sequencial_coligacao) e concatena siglas
         distintas — esse é o "composicao reconstruída".
      2. Linhas com `sequencial_coligacao` NULL E
         `tipo_agremiacao == 'partido isolado'`: composicao = sigla do próprio
         partido (caso típico em 2010 — PSOL/PCB/PCO).
      3. Demais linhas NULL: mantém NULL (2020 prefeito, onde nem sequencial
         veio — plan B com CSV TSE continua necessário).

    Args:
      df: DataFrame da tabela `partidos` (prefeito ou governador) com colunas
          `sigla_partido`, `composicao_coligacao`, `sequencial_coligacao`.
          `tipo_agremiacao` é opcional (se ausente, regra (2) é pulada).
      chaves_coligacao: colunas que delimitam o escopo da coligação.
          Para prefeito: `["ano", "id_municipio"]`. Para governador:
          `["ano", "sigla_uf"]`.

    Returns:
      Cópia do df com `composicao_coligacao` preenchida onde possível.
    """
    chaves = list(chaves_coligacao)
    required = set(chaves) | {"sigla_partido", "composicao_coligacao", "sequencial_coligacao"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df sem colunas: {sorted(missing)}")

    out = df.copy()
    antes_na = out["composicao_coligacao"].isna().sum()
    if antes_na == 0:
        return out  # nada a fazer

    # --- Regra 1: agregação via sequencial_coligacao ---
    com_seq = out[out["sequencial_coligacao"].notna()].copy()
    if len(com_seq):
        grupos = (
            com_seq.sort_values(chaves + ["sequencial_coligacao", "sigla_partido"])
            .groupby(chaves + ["sequencial_coligacao"], dropna=False)["sigla_partido"]
            .agg(lambda s: ":".join(sorted({str(x) for x in s if pd.notna(x)})))
            .reset_index()
            .rename(columns={"sigla_partido": "_comp_via_seq"})
        )
        out = out.merge(grupos, on=chaves + ["sequencial_coligacao"], how="left")
        mask_r1 = out["composicao_coligacao"].isna() & out["_comp_via_seq"].notna()
        n_r1 = int(mask_r1.sum())
        out.loc[mask_r1, "composicao_coligacao"] = out.loc[mask_r1, "_comp_via_seq"]
        out = out.drop(columns="_comp_via_seq")
    else:
        n_r1 = 0

    # --- Regra 2: partido isolado sem sequencial → a própria sigla ---
    n_r2 = 0
    if "tipo_agremiacao" in out.columns:
        mask_r2 = (
            out["composicao_coligacao"].isna()
            & out["sequencial_coligacao"].isna()
            & (out["tipo_agremiacao"].astype("string").str.lower() == "partido isolado")
        )
        n_r2 = int(mask_r2.sum())
        out.loc[mask_r2, "composicao_coligacao"] = out.loc[mask_r2, "sigla_partido"]

    depois_na = out["composicao_coligacao"].isna().sum()
    logger.info(
        "reconstruir_composicao: NA antes=%d, reconstruídos via sequencial=%d, "
        "via partido isolado=%d, NA depois=%d",
        int(antes_na), n_r1, n_r2, int(depois_na),
    )
    return out


__all__ = ["reconstruir_composicao_via_sequencial"]
