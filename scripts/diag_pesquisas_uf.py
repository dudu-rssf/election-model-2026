#!/usr/bin/env python
"""
scripts/diag_pesquisas_uf.py — auditoria do pesquisas_uf.csv contra TSE.

Compara as estimativas em `data/raw/pesquisas_uf.csv` com o resultado
real TSE agregado por UF (vindo de `presidencial_long.parquet`).

Lógica:
  * Para cada (ano, sigla_uf, sigla_partido) na pesquisa, calcular o
    share real TSE = Σ votos_partido_uf / Σ total_votos_mun_uf.
  * Mostrar diff = share_pesquisa - share_real.
  * Pesquisa pré-1º turno tipicamente subestima vencedor por 5-10pp;
    valores ±5pp pros demais. Diff > 15pp ou sinal trocado = red flag
    para revisão.

Output: tabela markdown em reports/auditoria_pesquisas_uf.md.

Uso:
    python scripts/diag_pesquisas_uf.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.config import PATHS  # noqa: E402
from src.features import io as fio  # noqa: E402
from src.features import pesquisas as fpesq  # noqa: E402


log = logging.getLogger("diag_pesquisas_uf")


def calcular_share_real_tse(
    pres_long: pd.DataFrame,
    ano_col: str = "ano_presidencial",
) -> pd.DataFrame:
    """Agrega votos real do TSE por (ano, UF, partido).

    Returns:
        DataFrame com [ano_col, sigla_uf, sigla_partido, share_real_tse,
        votos_total_uf].
    """
    required = {ano_col, "sigla_uf", "sigla_partido", "votos", "total_votos_mun"}
    missing = required - set(pres_long.columns)
    if missing:
        raise ValueError(f"pres_long sem colunas: {sorted(missing)}")

    # Total UF (deduplicar por município pra evitar contar votos várias vezes)
    total_uf = (
        pres_long[[ano_col, "sigla_uf", "id_municipio", "total_votos_mun"]]
        .drop_duplicates([ano_col, "sigla_uf", "id_municipio"])
        .groupby([ano_col, "sigla_uf"], observed=True)["total_votos_mun"]
        .sum()
        .reset_index()
        .rename(columns={"total_votos_mun": "votos_total_uf"})
    )
    # Soma de votos por (ano, UF, partido)
    votos_partido = (
        pres_long.groupby([ano_col, "sigla_uf", "sigla_partido"], observed=True)[
            "votos"
        ]
        .sum()
        .reset_index()
    )
    out = votos_partido.merge(total_uf, on=[ano_col, "sigla_uf"], how="left")
    out["share_real_tse"] = out["votos"] / out["votos_total_uf"]
    return out[[ano_col, "sigla_uf", "sigla_partido", "share_real_tse", "votos_total_uf"]]


def comparar(
    pesquisas: pd.DataFrame,
    real: pd.DataFrame,
    *,
    ano_col: str = "ano_presidencial",
) -> pd.DataFrame:
    """Junta pesquisas com TSE e calcula diff."""
    pq = pesquisas.rename(columns={"ano": ano_col})
    pq[ano_col] = pq[ano_col].astype("int64")
    pq["sigla_uf"] = pq["sigla_uf"].astype("string")
    pq["sigla_partido"] = pq["sigla_partido"].astype("string")
    real[ano_col] = real[ano_col].astype("int64")
    real["sigla_uf"] = real["sigla_uf"].astype("string")
    real["sigla_partido"] = real["sigla_partido"].astype("string")
    df = pq.merge(real, on=[ano_col, "sigla_uf", "sigla_partido"], how="left")
    df["diff"] = df["share_pesquisa"] - df["share_real_tse"]
    df["abs_diff"] = df["diff"].abs()
    # Flag: pesquisa pode subestimar vencedor regional em 0-10pp; ±5pp
    # pros demais. abs_diff > 0.10 = atenção; > 0.15 = vermelho.
    def _flag(d: float) -> str:
        if pd.isna(d):
            return "❓ sem TSE"
        a = abs(d)
        if a > 0.15:
            return "🔴 >15pp"
        if a > 0.10:
            return "🟡 >10pp"
        if a > 0.05:
            return "🟠 >5pp"
        return "✅ <5pp"
    df["flag"] = df["diff"].apply(_flag)
    return df[[
        ano_col, "sigla_uf", "sigla_partido",
        "share_pesquisa", "share_real_tse", "diff", "abs_diff", "flag",
    ]]


def main() -> int:
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s | %(message)s",
                        level=logging.INFO)

    log.info("Carregando presidencial_long...")
    pres_long = fio.load_interim("presidencial_long")
    log.info("pres_long: %d linhas, anos=%s",
             len(pres_long),
             sorted(pres_long["ano_presidencial"].unique().tolist()))

    log.info("Calculando share real TSE por (ano, UF, partido)...")
    real = calcular_share_real_tse(pres_long)
    log.info("share_real_tse: %d combinações", len(real))

    log.info("Carregando pesquisas_uf.csv...")
    pq = fpesq.carregar_pesquisas_uf(PATHS["data_raw"] / "pesquisas_uf.csv")
    log.info("pesquisas_uf: %d entradas", len(pq))

    log.info("Comparando...")
    diag = comparar(pq, real)
    diag = diag.sort_values(["abs_diff"], ascending=False).reset_index(drop=True)

    # Estatísticas
    n_red = int((diag["flag"] == "🔴 >15pp").sum())
    n_yellow = int((diag["flag"] == "🟡 >10pp").sum())
    n_orange = int((diag["flag"] == "🟠 >5pp").sum())
    n_green = int((diag["flag"] == "✅ <5pp").sum())
    n_nan = int(diag["share_real_tse"].isna().sum())
    log.info(
        "Resumo: 🔴 %d | 🟡 %d | 🟠 %d | ✅ %d | ❓ %d (total %d)",
        n_red, n_yellow, n_orange, n_green, n_nan, len(diag),
    )

    # Relatório
    out_path = PATHS["reports"] / "auditoria_pesquisas_uf.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# Auditoria pesquisas_uf.csv vs TSE\n")
    lines.append(f"**Total entradas pesquisa**: {len(diag)}")
    lines.append(f"**🔴 >15pp**: {n_red} (revisar urgente)")
    lines.append(f"**🟡 >10pp**: {n_yellow} (revisar)")
    lines.append(f"**🟠 >5pp**: {n_orange} (esperado se vencedor regional — pesquisa subestima ~5-10pp)")
    lines.append(f"**✅ <5pp**: {n_green} (boa precisão)")
    lines.append(f"**❓ sem TSE**: {n_nan} (partido não rodou na UF)\n")
    lines.append("> Convenção: `diff = share_pesquisa - share_real`. Negativo = pesquisa subestimou.\n")
    lines.append("## Top 30 maiores discrepâncias (|diff|)\n")
    lines.append(diag.head(30).to_markdown(index=False, floatfmt=".4f"))
    lines.append("\n## Tabela completa\n")
    lines.append(diag.to_markdown(index=False, floatfmt=".4f"))

    out_path.write_text("\n".join(lines), encoding="utf-8")
    log.info("Relatório: %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
