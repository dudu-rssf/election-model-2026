"""
src.ingestion.geo — geometrias municipais via `geobr`, com cache local.

Em dev, apenas as UFs do `MODE_CFG["ufs"]` são baixadas.
Em prod (`ufs == "all"`), baixa Brasil inteiro numa chamada.

Saída: GeoDataFrame serializado como Parquet em `data/raw/geometrias_municipios.parquet`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Protocol

import pandas as pd

from src.config import MODE_CFG, PATHS

logger = logging.getLogger(__name__)

CACHE_NAME = "geometrias_municipios"


class GeoBackend(Protocol):
    def read_municipality(self, code_muni: str, year: int): ...  # pragma: no cover


def _default_backend() -> GeoBackend:
    import geobr

    class _GeobrBackend:
        def read_municipality(self, code_muni: str, year: int):
            return geobr.read_municipality(code_muni=code_muni, year=year)

    return _GeobrBackend()


def _cache_path() -> Path:
    raw = PATHS["data_raw"]
    raw.mkdir(parents=True, exist_ok=True)
    return raw / f"{CACHE_NAME}.parquet"


def download_geometrias(
    ufs: Iterable[str] | str | None = None,
    ano: int = 2022,
    *,
    force: bool = False,
    backend: GeoBackend | None = None,
) -> "pd.DataFrame":
    """Baixa e cacheia geometrias municipais.

    Args:
        ufs: iterável de siglas ou string "all". Default: MODE_CFG["ufs"].
        ano: malha do IBGE (geobr). 2022 é padrão para modelagem 2026.
        force: ignora cache.
        backend: injetável para testes.

    Returns:
        DataFrame com coluna `geometry` (shapely). Retornamos DataFrame
        porque GeoDataFrame não é importado de cara (evita obrigar geopandas
        nos testes de outros módulos).
    """
    cache = _cache_path()
    if cache.exists() and not force:
        logger.info("geo cache hit: %s", cache)
        # Aqui basta pandas.read_parquet; geopandas faz o upgrade depois.
        return pd.read_parquet(cache)

    ufs = MODE_CFG["ufs"] if ufs is None else ufs
    backend = backend or _default_backend()

    if ufs == "all":
        gdf = backend.read_municipality(code_muni="all", year=ano)
    else:
        ufs_list = list(ufs)
        frames = []
        for uf in ufs_list:
            logger.info("geobr: baixando %s (ano=%s)", uf, ano)
            frames.append(backend.read_municipality(code_muni=uf, year=ano))
        # concatena mantendo colunas (geopandas lida com isso via pd.concat)
        gdf = pd.concat(frames, ignore_index=True)

    if len(gdf) == 0:
        raise RuntimeError("geobr retornou 0 linhas — verifique UF/ano.")

    # geopandas GeoDataFrame.to_parquet grava geometria como WKB automaticamente.
    # Se não for GeoDataFrame (teste com mock), usa pandas.
    write_fn = getattr(gdf, "to_parquet", None)
    if write_fn is None:
        raise TypeError("objeto retornado não suporta .to_parquet()")
    tmp = cache.with_suffix(".parquet.tmp")
    write_fn(tmp, index=False)
    import os
    os.replace(tmp, cache)
    logger.info("salvo: %s (%d linhas)", cache, len(gdf))
    return gdf


__all__ = ["CACHE_NAME", "download_geometrias"]
