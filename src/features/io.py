"""
src.features.io — convenções de leitura/escrita entre fases.

Regras:
  * Ingestão grava `data/raw/<name>.parquet` (bruto, sempre) e, em dev,
    `data/raw/<name>.dev.parquet` (amostrado). Fases 2+ leem a versão
    apropriada via `load_raw`.
  * Fase 2 grava em `data/interim/`. Fases 3+ em `data/processed/`.
    Em dev sufixamos `.dev.parquet` para separar de eventuais runs prod.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import MODE, PATHS

logger = logging.getLogger(__name__)


def _suffix() -> str:
    return ".dev.parquet" if MODE == "dev" else ".parquet"


# ------------------------------------------------------------
# Leitura da ingestão (data/raw)
# ------------------------------------------------------------
def load_raw(name: str) -> pd.DataFrame:
    """Lê `data/raw/<name>.dev.parquet` em dev, senão `data/raw/<name>.parquet`.

    Faz fallback: se o `.dev.parquet` não existe em dev, usa o `.parquet`
    (por exemplo, ao rodar só uma etapa sem reingerir).
    """
    raw = PATHS["data_raw"]
    dev_path = raw / f"{name}.dev.parquet"
    full_path = raw / f"{name}.parquet"

    if MODE == "dev" and dev_path.exists():
        logger.info("load_raw(dev): %s", dev_path.name)
        return pd.read_parquet(dev_path)
    if full_path.exists():
        logger.info("load_raw: %s", full_path.name)
        return pd.read_parquet(full_path)
    raise FileNotFoundError(
        f"Nenhum parquet encontrado para {name!r} em {raw}. "
        "Rodou `scripts/01_ingest.py`?"
    )


def save_interim(df: pd.DataFrame, name: str) -> Path:
    """Grava em data/interim/<name>[.dev].parquet, atômico."""
    out_dir = PATHS["data_interim"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}{_suffix()}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    import os
    os.replace(tmp, path)
    logger.info("save_interim: %s (%d linhas, %d colunas)", path, len(df), df.shape[1])
    return path


def load_interim(name: str) -> pd.DataFrame:
    path = PATHS["data_interim"] / f"{name}{_suffix()}"
    if not path.exists():
        raise FileNotFoundError(f"Interim não encontrado: {path}")
    return pd.read_parquet(path)


def save_processed(df: pd.DataFrame, name: str) -> Path:
    out_dir = PATHS["data_processed"]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}{_suffix()}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    import os
    os.replace(tmp, path)
    logger.info("save_processed: %s (%d linhas)", path, len(df))
    return path


def load_processed(name: str) -> pd.DataFrame:
    path = PATHS["data_processed"] / f"{name}{_suffix()}"
    if not path.exists():
        raise FileNotFoundError(f"Processed não encontrado: {path}")
    return pd.read_parquet(path)


__all__ = [
    "load_raw",
    "save_interim",
    "load_interim",
    "save_processed",
    "load_processed",
]
