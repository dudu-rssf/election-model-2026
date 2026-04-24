#!/usr/bin/env python
"""
scripts/peek_schema.py — descobre o schema real da BD para o dataset TSE.

v2: lista todas as tabelas do dataset `br_tse_eleicoes`, imprime as colunas
de cada uma, e também destaca onde aparece a string "coligacao" (que a BD
moveu pra outra tabela em abr/2026).

Uso:
    python scripts/peek_schema.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import require_billing_project  # noqa: E402


DATASETS = [
    "br_tse_eleicoes",
    "br_bd_diretorios_brasil",
]


def listar_tabelas(client, dataset: str) -> list[str]:
    sql = f"""
    SELECT table_name
    FROM `basedosdados.{dataset}.INFORMATION_SCHEMA.TABLES`
    ORDER BY table_name
    """
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)
    return df["table_name"].tolist()


def listar_colunas(client, dataset: str, table: str) -> list[tuple[str, str]]:
    sql = f"""
    SELECT column_name, data_type
    FROM `basedosdados.{dataset}.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = '{table}'
    ORDER BY ordinal_position
    """
    df = client.query(sql).to_dataframe(create_bqstorage_client=False)
    return list(zip(df["column_name"], df["data_type"]))


def main() -> int:
    billing = require_billing_project()
    from google.cloud import bigquery  # noqa: WPS433

    client = bigquery.Client(project=billing)

    hits_coligacao: list[tuple[str, str, str]] = []  # (dataset, table, column)

    for dataset in DATASETS:
        print("#" * 70)
        print(f"DATASET: basedosdados.{dataset}")
        print("#" * 70)
        try:
            tabelas = listar_tabelas(client, dataset)
        except Exception as e:
            print(f"  ERRO listando tabelas: {e}")
            continue

        print(f"Tabelas encontradas ({len(tabelas)}): {tabelas}\n")

        for table in tabelas:
            print("=" * 70)
            print(f"basedosdados.{dataset}.{table}")
            print("=" * 70)
            try:
                cols = listar_colunas(client, dataset, table)
                for col, dtype in cols:
                    marker = "  <-- COLIGACAO" if "coligacao" in col.lower() else ""
                    print(f"  {col:<40} {dtype}{marker}")
                    if "coligacao" in col.lower():
                        hits_coligacao.append((dataset, table, col))
            except Exception as e:
                print(f"  ERRO: {e}")
            print()

    if hits_coligacao:
        print("#" * 70)
        print("RESUMO — colunas contendo 'coligacao':")
        print("#" * 70)
        for ds, tb, col in hits_coligacao:
            print(f"  {ds}.{tb}.{col}")
    else:
        print("#" * 70)
        print("Nenhuma coluna 'coligacao' encontrada nesses datasets.")
        print("#" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
