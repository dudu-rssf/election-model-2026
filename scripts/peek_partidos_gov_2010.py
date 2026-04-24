#!/usr/bin/env python
"""
Peek em data/raw/partidos_governador.dev.parquet pra diagnosticar a cobertura
0% em 2010 sinalizada por vertical.governador_vencedor_por_eleicao.

Responde:
  1. Quantas linhas por ano?
  2. Quantas linhas por ano têm composicao_coligacao não-NULL?
  3. Amostra das linhas de 2010 em SP.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import io as fio  # noqa: E402


def main() -> int:
    df = fio.load_raw("partidos_governador")
    print(f"total linhas: {len(df)}")
    print(f"colunas: {list(df.columns)}")
    print()
    print("linhas por ano:")
    print(df.groupby("ano").size())
    print()
    print("cobertura composicao_coligacao por ano (ano / total / não-NULL / %):")
    cov = (
        df.assign(_ok=df["composicao_coligacao"].notna())
        .groupby("ano")
        .agg(total=("ano", "size"), preenchidas=("_ok", "sum"))
    )
    cov["pct"] = (100 * cov["preenchidas"] / cov["total"]).round(1)
    print(cov)
    print()
    print("amostra 2010 SP (até 10 linhas):")
    sp10 = df[(df["ano"] == 2010) & (df["sigla_uf"] == "SP")].head(10)
    print(sp10.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
