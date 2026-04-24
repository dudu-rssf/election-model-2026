#!/usr/bin/env python
"""
scripts/diag_continuidade.py — diagnóstico do top_continuidade.

Responde duas perguntas após Gabriel apontar que Santana de Parnaíba deveria
ter 12 anos de PSDB (Elvis 2013-2020 + Antonio Pereira 2021-2024, todos
eleitos pelo PSDB):

  1. Santana de Parnaíba (id 3547304) está no sample dev de 100 municípios?
  2. A métrica `ano_max_mesmo_partido` (ou equivalente) consegue expressar
     12 anos, ou trunca em 8? Cross-check com contagem direta no painel.

Output: texto no console.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import io as fio  # noqa: E402


SANTANA_PARNAIBA_ID = "3547304"  # IBGE


def main() -> int:
    painel = fio.load_interim("painel_mestre")
    feats = fio.load_processed("features")

    print("=" * 70)
    print("painel_mestre.dev colunas:")
    print(list(painel.columns))
    print()
    print("features.dev colunas:")
    print(list(feats.columns))
    print()

    # 1. Santana de Parnaíba no sample?
    print("=" * 70)
    print("1. Santana de Parnaíba (id 3547304) no sample?")
    sp = painel[painel["id_municipio"] == SANTANA_PARNAIBA_ID]
    if len(sp) == 0:
        print("   -> NÃO. Só 100 de 645 municípios foram amostrados.")
        print("      Sem Santana de Parnaíba, não podemos usá-la como teste.")
    else:
        print("   -> SIM. Dados do painel:")
        print(sp[["id_municipio", "nome", "ano_presidencial",
                  "ano_eleicao_municipal", "mayor_partido", "mayor_numero"]].to_string())
    print()

    # 2. Conta direta: municípios do sample com 3 mandatos seguidos do mesmo partido
    print("=" * 70)
    print("2. Municípios do sample com 3 mandatos municipais do MESMO partido:")
    # Agrupa por município, coleciona partidos por ano municipal (ordenado)
    pivot = (
        painel.dropna(subset=["mayor_partido"])
        .groupby("id_municipio")
        .apply(lambda g: tuple(g.sort_values("ano_eleicao_municipal")["mayor_partido"].values))
        .reset_index(name="partidos_por_eleicao")
    )
    pivot["n_eleicoes"] = pivot["partidos_por_eleicao"].apply(len)
    pivot["todos_iguais"] = pivot.apply(
        lambda r: r["n_eleicoes"] == 3 and len(set(r["partidos_por_eleicao"])) == 1,
        axis=1,
    )
    pivot = pivot.merge(
        painel[["id_municipio", "nome"]].drop_duplicates(),
        on="id_municipio", how="left"
    )
    con_tres = pivot[pivot["todos_iguais"]]
    print(f"   Total com 3 mandatos: {len(con_tres)}")
    if len(con_tres) > 0:
        print("   Lista:")
        for _, r in con_tres.iterrows():
            print(f"     {r['id_municipio']} {r['nome']}: {r['partidos_por_eleicao']}")
    print()

    # 3. Agora compara com a métrica de features: quantos têm ano_max_mesmo_partido = 12?
    print("=" * 70)
    print("3. Distribuição da métrica `ano_max_mesmo_partido` (ou equivalente) em features:")
    # Procura coluna relacionada
    cols_candidatas = [c for c in feats.columns if "mesmo_partido" in c.lower() or "anos_max" in c.lower()]
    if not cols_candidatas:
        cols_candidatas = [c for c in feats.columns if "partido" in c.lower() and ("anos" in c.lower() or "cont" in c.lower())]
    print(f"   Colunas candidatas encontradas: {cols_candidatas}")
    for col in cols_candidatas:
        print(f"\n   {col} value_counts:")
        print(feats[col].value_counts(dropna=False).to_string())
    print()

    # 4. Para os municípios do (2), qual o valor da métrica em features?
    if len(con_tres) > 0 and cols_candidatas:
        print("=" * 70)
        print("4. Valor da métrica para municípios com 3 mandatos iguais (ground truth = 12):")
        mun_ids = con_tres["id_municipio"].tolist()
        sub = feats[feats["id_municipio"].isin(mun_ids)]
        if len(sub):
            print(sub[["id_municipio", "ano_presidencial"] + cols_candidatas].head(30).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
