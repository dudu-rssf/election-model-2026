#!/usr/bin/env python
"""scripts/peek_coligacao_populada.py — confere se composicao_coligacao
está realmente populada em `br_tse_eleicoes.partidos`.

Sample mostrou todos NULL. Pode ser pq REPUBLICANOS foi isolado, ou pq o
campo é raro. Antes de refazer o pipeline em cima dessa tabela, vamos ver:

1. Quantos registros têm composicao_coligacao não-null por (ano, cargo)?
2. Quais valores aparecem em SP/2020/prefeito (sample real, não filtrado)?
3. Cross-check em Santana de Parnaíba (3547304) 2012/2016/2020 — sabemos
   que PSDB ganhou as 3 eleições; queremos ver a composição de cada uma.
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

    # 1. Cobertura de composicao_coligacao por (ano, cargo)
    print("=" * 70)
    print("1. Cobertura de composicao_coligacao por (ano, cargo) em anos relevantes")
    print("=" * 70)
    sql_cov = """
    SELECT
      ano, cargo, tipo_agremiacao,
      COUNT(*) AS n_total,
      COUNTIF(composicao_coligacao IS NOT NULL) AS n_com_coligacao,
      ROUND(SAFE_DIVIDE(COUNTIF(composicao_coligacao IS NOT NULL), COUNT(*)) * 100, 1) AS pct
    FROM `basedosdados.br_tse_eleicoes.partidos`
    WHERE ano IN (2012, 2014, 2016, 2018, 2020, 2022)
      AND cargo IN ('prefeito', 'governador')
    GROUP BY ano, cargo, tipo_agremiacao
    ORDER BY ano, cargo, tipo_agremiacao
    """
    cov = client.query(sql_cov).to_dataframe(create_bqstorage_client=False)
    print(cov.to_string())
    print()

    # 2. Sample de 10 registros COM composicao_coligacao não-null em SP
    print("=" * 70)
    print("2. Sample de 10 registros COM composicao_coligacao em SP / 2020 / prefeito")
    print("=" * 70)
    sql_sample = """
    SELECT
      ano, sigla_uf, id_municipio, cargo,
      numero, sigla, tipo_agremiacao,
      composicao_coligacao
    FROM `basedosdados.br_tse_eleicoes.partidos`
    WHERE ano = 2020 AND sigla_uf = 'SP' AND cargo = 'prefeito'
      AND composicao_coligacao IS NOT NULL
    LIMIT 10
    """
    sample = client.query(sql_sample).to_dataframe(create_bqstorage_client=False)
    if len(sample) == 0:
        print("  NENHUM registro com composicao_coligacao em SP/2020/prefeito.")
    else:
        print(sample.to_string())
    print()

    # 3. Santana de Parnaíba: 2012, 2016, 2020 — todas as linhas pra prefeito
    print("=" * 70)
    print("3. Santana de Parnaíba (3547304) — partidos para prefeito 2012/2016/2020")
    print("=" * 70)
    sql_sp = """
    SELECT
      ano, id_municipio, numero, sigla, tipo_agremiacao,
      sequencial_coligacao, nome_coligacao, composicao_coligacao
    FROM `basedosdados.br_tse_eleicoes.partidos`
    WHERE id_municipio = '3547304'
      AND cargo = 'prefeito'
      AND ano IN (2012, 2016, 2020)
    ORDER BY ano, numero
    """
    sp = client.query(sql_sp).to_dataframe(create_bqstorage_client=False)
    print(sp.to_string())
    print()

    # 4. Pra governador: granularidade — id_municipio é null? ou repete por mun?
    print("=" * 70)
    print("4. Governador 2014 SP — granularidade (5 linhas)")
    print("=" * 70)
    sql_gov = """
    SELECT
      ano, sigla_uf, id_municipio, cargo,
      numero, sigla, tipo_agremiacao, composicao_coligacao
    FROM `basedosdados.br_tse_eleicoes.partidos`
    WHERE ano = 2014 AND sigla_uf = 'SP' AND cargo = 'governador'
    LIMIT 5
    """
    gov = client.query(sql_gov).to_dataframe(create_bqstorage_client=False)
    print(gov.to_string())
    print()

    # 5. Conta linhas distintas por (ano, sigla_uf, cargo='governador', numero)
    print("=" * 70)
    print("5. Granularidade governador: linhas por (ano, uf, numero) em SP/2014")
    print("=" * 70)
    sql_gran = """
    SELECT numero, sigla, COUNT(*) as n_linhas,
           COUNT(DISTINCT id_municipio) as n_munis
    FROM `basedosdados.br_tse_eleicoes.partidos`
    WHERE ano = 2014 AND sigla_uf = 'SP' AND cargo = 'governador'
    GROUP BY numero, sigla
    ORDER BY n_linhas DESC
    LIMIT 10
    """
    gran = client.query(sql_gran).to_dataframe(create_bqstorage_client=False)
    print(gran.to_string())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
