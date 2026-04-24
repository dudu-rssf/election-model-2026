"""
src.ingestion.sample — amostragem reprodutível em modo dev.

O briefing manda aplicar `max_municipios` **depois** do download, para que
a estrutura dos dados (quem elegeu quem, onde, etc.) seja realista e não
fique viesada pela seleção no SQL.

Estratégia: em dev, escolhemos até `MODE_CFG["max_municipios"]` IDs com
`numpy.random.default_rng(SEED)`. Todas as tabelas (presidenciais, prefeito,
candidatos de prefeito, diretório, geometrias) são filtradas pelo MESMO
conjunto de IDs para garantir joins consistentes.

Em prod (`max_municipios is None`) esta função é no-op.
"""
from __future__ import annotations

import logging
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import MODE_CFG, SEED

logger = logging.getLogger(__name__)


def choose_ids(todos_ids: Iterable[str], max_n: int | None, seed: int = SEED) -> list[str]:
    """Retorna lista ordenada e deterministic de IDs a manter.

    Se max_n é None ou >= len(todos_ids), retorna todos.
    """
    todos = sorted({str(x) for x in todos_ids})
    if max_n is None or max_n >= len(todos):
        return todos
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(todos), size=max_n, replace=False)
    return sorted(todos[i] for i in idx)


def filter_by_municipios(
    df: pd.DataFrame,
    ids_keep: Iterable[str],
    col: str = "id_municipio",
) -> pd.DataFrame:
    """Filtra `df` mantendo apenas linhas em `ids_keep` (se a coluna existir)."""
    if col not in df.columns:
        return df
    ids_set = set(map(str, ids_keep))
    mask = df[col].astype("string").isin(ids_set)
    out = df.loc[mask].reset_index(drop=True)
    return out


def apply_dev_sampling(
    frames: dict[str, pd.DataFrame],
    id_col: str = "id_municipio",
) -> dict[str, pd.DataFrame]:
    """Aplica amostragem dev consistentemente em todas as tabelas.

    Args:
        frames: dict nome -> DataFrame já carregado (da ingestão).
                Precisa ter pelo menos uma tabela com `id_col` para derivar
                o universo.
        id_col: coluna de ID municipal.

    Returns:
        Novo dict com as mesmas chaves, filtrado.
    """
    max_n = MODE_CFG.get("max_municipios")
    if max_n is None:
        return frames

    # Universo: união de IDs presentes em tabelas com a coluna.
    universos: list[set[str]] = []
    for name, df in frames.items():
        if id_col in df.columns:
            universos.append(set(df[id_col].astype("string").dropna().unique()))
    if not universos:
        logger.warning("apply_dev_sampling: nenhuma tabela tem %s; no-op", id_col)
        return frames

    todos = sorted(set().union(*universos))
    ids_keep = choose_ids(todos, max_n=max_n)
    logger.info("dev sampling: %d/%d municípios mantidos", len(ids_keep), len(todos))

    return {name: filter_by_municipios(df, ids_keep, col=id_col) for name, df in frames.items()}


__all__ = ["choose_ids", "filter_by_municipios", "apply_dev_sampling"]
