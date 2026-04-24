#!/usr/bin/env python
"""scripts/peek_partidos.py — schema da tabela `br_tse_eleicoes.partidos`.

Schema-only — não baixa dado. Usa pra descobrir a chave de join entre
`partidos` (que agora tem composicao_coligacao) e `candidatos`/`resultados_*`.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import require_billing_project  # noqa: E402


def main() -> int:
    billing = require_billing_project()
    from google.cloud import bigquery  # noqa: WPS433

    client = bigquery.Client(project=billing)

    # 1. Schema completo de partidos
    print("=" * 70)
    print("basedosdados.br_tse_eleicoes.partidos — colunas")
    print("=" * 70)
    sql_cols = """
    SELECT column_name, data_type
    FROM `basedosdados.br_tse_eleicoes.INFORMATION_SCHEMA.COLUMNS`
    WHERE table_name = 'partidos'
    ORDER BY ordinal_position
    """
    cols = client.query(sql_cols).to_dataframe(create_bqstorage_client=False)
    for _, r in cols.iterrows():
        marker = "  <-- COLIGACAO" if "coligacao" in r["column_name"].lower() else ""
        print(f"  {r['column_name']:<40} {r['data_type']}{marker}")
    print()

    # 2. Sample 10 linhas pra entender granularidade
    print("=" * 70)
    print("Sample (10 linhas, ano=2020, SP, prefeito) — pra ver granularidade")
    print("=" * 70)
    sql_sample = """
    SELECT *
    FROM `basedosdados.br_tse_eleicoes.partidos`
    WHERE ano = 2020 AND sigla_uf = 'SP'
    LIMIT 10
    """
    try:
        sample = client.query(sql_sample).to_dataframe(create_bqstorage_client=False)
        print(sample.to_string())
    except Exception as e:
        print(f"  Falhou filtrando por ano/uf: {e}")
        print("  Tentando sem filtro:")
        sample = client.query(
            "SELECT * FROM `basedosdados.br_tse_eleicoes.partidos` LIMIT 10"
        ).to_dataframe(create_bqstorage_client=False)
        print(sample.to_string())
    print()

    # 3. Conta linhas por (ano, sigla_uf) pra confirmar particionamento
    print("=" * 70)
    print("Anos disponíveis e contagem")
    print("=" * 70)
    sql_anos = """
    SELECT ano, COUNT(*) as n
    FROM `basedosdados.br_tse_eleicoes.partidos`
    GROUP BY ano
    ORDER BY ano
    """
    try:
        anos = client.query(sql_anos).to_dataframe(create_bqstorage_client=False)
        print(anos.to_string())
    except Exception as e:
        print(f"  Falhou: {e}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
