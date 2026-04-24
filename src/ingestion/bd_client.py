"""
src.ingestion.bd_client — wrapper fino de `basedosdados` com cache Parquet local.

Padrão de uso (em scripts/01_ingest.py):

    from src.ingestion.bd_client import download
    from src.ingestion.queries import resultados_presidenciais_sql

    df = download(
        name="resultados_presidenciais",
        sql=resultados_presidenciais_sql(),
    )

Comportamento:
  1. Se o cache local (`data/raw/<name>.parquet`) existe, lê dele.
  2. Caso contrário, executa a query via basedosdados.read_sql() (que usa o
     `billing_project_id` do `config.yaml`), grava no cache e retorna.
  3. Importação de `basedosdados` é lazy — evita requerer GCP em testes
     unitários que nunca chamam `download()`.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.config import PATHS, require_billing_project

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Protocolo do backend (facilita monkeypatch nos testes)
# ------------------------------------------------------------
class SQLBackend(Protocol):
    """Qualquer objeto com `.read_sql(query, billing_project_id) -> pd.DataFrame`."""

    def read_sql(self, query: str, billing_project_id: str) -> pd.DataFrame: ...  # pragma: no cover


def _default_backend() -> SQLBackend:
    """Carrega backend SQL sob demanda.

    Tenta `basedosdados` primeiro (wrapper oficial). Se falhar no import
    (ex.: incompatibilidade do beta com versões novas de `packaging`),
    usa `google.cloud.bigquery` direto. Os dois dão o mesmo resultado
    porque os SQLs do projeto já referenciam tabelas com caminho completo
    `basedosdados.br_tse_eleicoes.<tabela>`.
    """
    try:
        import basedosdados as bd  # noqa: WPS433  (lazy import proposital)

        class _BDBackend:
            def read_sql(self, query: str, billing_project_id: str) -> pd.DataFrame:
                return bd.read_sql(query=query, billing_project_id=billing_project_id)

        logger.info("backend SQL: basedosdados")
        return _BDBackend()
    except Exception as exc:  # import error ou erro de inicialização
        logger.warning(
            "basedosdados indisponível (%s). Caindo para google-cloud-bigquery direto.",
            exc.__class__.__name__,
        )

    # Fallback: google-cloud-bigquery direto
    from google.cloud import bigquery  # noqa: WPS433

    class _BQBackend:
        def read_sql(self, query: str, billing_project_id: str) -> pd.DataFrame:
            client = bigquery.Client(project=billing_project_id)
            job = client.query(query)
            return job.to_dataframe(create_bqstorage_client=False)

    logger.info("backend SQL: google-cloud-bigquery (fallback)")
    return _BQBackend()


# ------------------------------------------------------------
# Cache
# ------------------------------------------------------------
def _cache_path(name: str) -> Path:
    raw_dir = PATHS["data_raw"]
    raw_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir / f"{name}.parquet"


def _query_fingerprint(query: str) -> str:
    """Hash curto da query — guardado em metadados do Parquet pra detectar
    mudança de query entre runs e forçar reingestão quando `--force`."""
    return hashlib.sha256(query.encode("utf-8")).hexdigest()[:12]


# ------------------------------------------------------------
# Núcleo
# ------------------------------------------------------------
def download(
    name: str,
    sql: str,
    *,
    force: bool = False,
    backend: SQLBackend | None = None,
) -> pd.DataFrame:
    """Retorna DataFrame, priorizando cache Parquet local.

    Args:
        name: identificador (vira nome do arquivo no cache).
        sql: query BigQuery a executar se o cache estiver frio.
        force: ignora cache e re-baixa.
        backend: injetável — default chama `basedosdados`.

    Returns:
        DataFrame com as colunas retornadas pela query.
    """
    cache = _cache_path(name)
    fingerprint = _query_fingerprint(sql)

    if cache.exists() and not force:
        logger.info("cache hit: %s (%s)", cache, fingerprint)
        return pd.read_parquet(cache)

    billing = require_billing_project()
    logger.info("cache miss: %s — rodando SQL (billing=%s)", name, billing)

    backend = backend or _default_backend()
    df = backend.read_sql(query=sql, billing_project_id=billing)

    if df is None or len(df) == 0:
        raise RuntimeError(
            f"Ingestão '{name}' retornou 0 linhas. Verifique filtros SQL e permissão BigQuery."
        )

    # Atomic write: escreve em .tmp e renomeia
    tmp = cache.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, cache)
    logger.info("salvo: %s (%d linhas)", cache, len(df))
    return df


def list_cached() -> list[Path]:
    """Lista Parquets presentes em data/raw/."""
    return sorted(PATHS["data_raw"].glob("*.parquet"))


def clear_cache(name: str | None = None) -> int:
    """Remove cache; `name=None` limpa todos. Retorna nº de arquivos removidos."""
    if name is None:
        count = 0
        for p in list_cached():
            p.unlink()
            count += 1
        return count
    path = _cache_path(name)
    if path.exists():
        path.unlink()
        return 1
    return 0


__all__ = [
    "SQLBackend",
    "download",
    "list_cached",
    "clear_cache",
]
