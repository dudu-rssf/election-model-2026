#!/usr/bin/env python
"""
Peek em data/raw/partidos_prefeito.dev.parquet pra investigar se o buraco
de composicao_coligacao em 2020 é reconstruível via sequencial_coligacao.

Se sequencial_coligacao vier populado em 2020 (mesmo padrão que 2010 gov),
dá pra reconstruir a composição via self-join e eliminar o plan B.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import io as fio  # noqa: E402


def main() -> int:
    df = fio.load_raw("partidos_prefeito")
    print(f"total linhas: {len(df)}")
    print(f"colunas: {list(df.columns)}")
    print()
    print("linhas por ano:")
    print(df.groupby("ano").size())
    print()
    print("cobertura por ano (composicao_coligacao e sequencial_coligacao):")
    cov = (
        df.assign(
            _cc=df["composicao_coligacao"].notna(),
            _sc=df["sequencial_coligacao"].notna(),
        )
        .groupby("ano")
        .agg(
            total=("ano", "size"),
            comp_ok=("_cc", "sum"),
            seq_ok=("_sc", "sum"),
        )
    )
    cov["comp_pct"] = (100 * cov["comp_ok"] / cov["total"]).round(1)
    cov["seq_pct"] = (100 * cov["seq_ok"] / cov["total"]).round(1)
    print(cov)
    print()
    print("amostra 2020 (até 15 linhas, municípios variados):")
    s2020 = df[df["ano"] == 2020].head(15)
    print(s2020[["ano", "sigla_uf", "id_municipio", "sigla_partido",
                 "sequencial_coligacao", "nome_coligacao",
                 "composicao_coligacao"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
