#!/usr/bin/env python
"""
scripts/diag_decis_intermediarios.py — diagnóstico da descalibração nos
decis intermediários do LightGBM_v1 (presidencial, teste 2022).

`reports/status_fase_4.md` revela um padrão estranho:

    decil 7: pred=0.016  real=0.184  erro=+0.168
    decil 8: pred=0.020  real=0.220  erro=+0.200
    decil 9: pred=0.029  real=0.138  erro=+0.109

O modelo está prevendo MUITO baixo (1-3%) pra um conjunto de candidatos
que tiveram share real de 14-22%. ~330 linhas afetadas. A descalibração
não está nas caudas — está num "cinto" intermediário.

Hipótese principal: **troca de partido**. Em particular:
  - PL 2022 → recebe o eleitorado do PSL 2018 (Bolsonaro migrou).
  - UNIÃO 2022 → fusão DEM+PSL.
Em ambos os casos, `lag_share_1t` é zero/baixo (o partido na sigla atual
mal existia em 2018), mas `lag_share_1t_sucessao` deveria capturar via
sigla canônica (PSL 2018).

Este script responde:

    1. Que candidatos caem na faixa pred ∈ [pred_min, pred_max]?
    2. Quanto disso é PL/UNIÃO em 2022?
    3. Pra essas linhas, `lag_share_1t_sucessao` está populado?
    4. Bate com share do PSL 2018 esperado?

Uso:
    python scripts/diag_decis_intermediarios.py
    python scripts/diag_decis_intermediarios.py --pred-min 0.012 --pred-max 0.04
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import io as fio  # noqa: E402

log = logging.getLogger("diag_decis_intermediarios")


PRED_COL = "pred_LightGBM_v1"
ANO_COL = "ano_presidencial"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="diag descalibração intermediária — Fase 4")
    p.add_argument("--pred-min", type=float, default=0.012,
                   help="limite inferior da faixa de predição (default 0.012)")
    p.add_argument("--pred-max", type=float, default=0.040,
                   help="limite superior da faixa de predição (default 0.040)")
    p.add_argument("--top-k", type=int, default=40,
                   help="quantas linhas (top por |erro|) listar")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


# ------------------------------------------------------------
# Carga + join preds + features
# ------------------------------------------------------------
def _carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    preds = fio.load_processed("preds")
    feats = fio.load_processed("features")
    ano_teste = int(preds[ANO_COL].max())
    log.info("ano_teste detectado: %d", ano_teste)
    log.info("preds: %d linhas | features: %d linhas (todos os anos)",
             len(preds), len(feats))
    return preds, feats


def _join(preds: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Junta preds + features (todas as colunas relevantes do ano de teste)."""
    keys = [ANO_COL, "id_municipio", "sigla_partido", "numero_candidato"]
    feats_cols = keys + [
        "lag_share_1t", "lag_share_1t_sucessao", "lag2_share_1t",
        "swing_share_1t", "log_eleitorado",
        "share_gov_partido", "share_dep_federal_partido",
        "alinhado_gov_vigente_partido",
        "indice_continuidade", "anos_consecutivos_mesmo_partido",
        "porte", "continuidade_classe",
        "nome_candidato",
    ]
    cols_present = [c for c in feats_cols if c in feats.columns]
    ano_teste = int(preds[ANO_COL].max())
    feats_te = feats[feats[ANO_COL] == ano_teste][cols_present].drop_duplicates(
        subset=keys, keep="first",
    )
    merged = preds.merge(feats_te, on=keys, how="left", validate="one_to_one")
    log.info("merge: %d linhas (%d colunas)", len(merged), merged.shape[1])
    return merged


# ------------------------------------------------------------
# Análises
# ------------------------------------------------------------
def _resumo_geral(df: pd.DataFrame) -> None:
    y = df["y_true"].to_numpy()
    p = df[PRED_COL].to_numpy()
    err = y - p
    print("=" * 78)
    print(f"RESUMO GERAL test 2022 (n={len(df)})")
    print("=" * 78)
    print(f"  y_true:  mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")
    print(f"  pred:    mean={p.mean():.4f}  std={p.std():.4f}  "
          f"min={p.min():.4f}  max={p.max():.4f}")
    print(f"  erro signed (y-p):  mean={err.mean():+.4f}  "
          f"std={err.std():.4f}")
    print(f"  MAE: {np.abs(err).mean():.4f}")
    print()


def _selecionar_faixa(
    df: pd.DataFrame, pred_min: float, pred_max: float,
) -> pd.DataFrame:
    sel = df[(df[PRED_COL] >= pred_min) & (df[PRED_COL] <= pred_max)].copy()
    sel["erro"] = sel["y_true"] - sel[PRED_COL]
    print("=" * 78)
    print(f"FAIXA: pred ∈ [{pred_min:.3f}, {pred_max:.3f}]  ->  n={len(sel)}")
    print("=" * 78)
    if len(sel) == 0:
        return sel
    print(f"  pred  mean={sel[PRED_COL].mean():.4f}  range=[{sel[PRED_COL].min():.4f}, {sel[PRED_COL].max():.4f}]")
    print(f"  real  mean={sel['y_true'].mean():.4f}  range=[{sel['y_true'].min():.4f}, {sel['y_true'].max():.4f}]")
    print(f"  erro  mean={sel['erro'].mean():+.4f}  med={sel['erro'].median():+.4f}")
    print(f"  MAE   {sel['erro'].abs().mean():.4f}")
    print()
    return sel


def _por_partido(sel: pd.DataFrame) -> None:
    g = sel.groupby("sigla_partido").agg(
        n=("y_true", "size"),
        y_mean=("y_true", "mean"),
        pred_mean=(PRED_COL, "mean"),
        erro_mean=("erro", "mean"),
        erro_abs=("erro", lambda s: s.abs().mean()),
    ).sort_values("erro_abs", ascending=False)
    print("  Breakdown por partido (na faixa) — ordenado por |erro|:")
    print(g.to_string(float_format="%.4f"))
    print()


def _foco_partidos_suspeitos(
    sel: pd.DataFrame, partidos_alvo: list[str] = ("PL", "UNIÃO", "REPUBLICANOS"),
) -> None:
    """Foco em partidos com hipótese de troca/migração."""
    alvo = sel[sel["sigla_partido"].isin(partidos_alvo)].copy()
    if len(alvo) == 0:
        print(f"  [nenhuma linha de {partidos_alvo} na faixa]")
        return
    print(f"  FOCO em partidos suspeitos {list(partidos_alvo)}:")
    print(f"    n na faixa: {len(alvo)} ({len(alvo)/max(len(sel),1):.0%})")
    print(f"    contribuição pro |erro| total: "
          f"{alvo['erro'].abs().sum() / sel['erro'].abs().sum():.0%}")
    print()
    g = alvo.groupby("sigla_partido").agg(
        n=("y_true", "size"),
        y_mean=("y_true", "mean"),
        pred_mean=(PRED_COL, "mean"),
        lag_share=("lag_share_1t", "mean"),
        lag_sucessao=("lag_share_1t_sucessao", "mean"),
        lag_share_nan=("lag_share_1t", lambda s: int(s.isna().sum())),
        lag_sucessao_nan=("lag_share_1t_sucessao", lambda s: int(s.isna().sum())),
    )
    print("    Por partido — médias de lag_share_1t e lag_share_1t_sucessao:")
    print(g.to_string(float_format="%.4f"))
    print()


def _validar_sucessao_aplicada(df_full: pd.DataFrame) -> None:
    """Sanity check: pra cada par (sigla, ano) com mapping declarado, conta
    quantas linhas têm `lag_share_1t_sucessao` populado e diferente de
    `lag_share_1t` (= sucessão realmente aplicou)."""
    casos = [("PL", 2022, "PSL"), ("UNIÃO", 2022, "DEM")]
    print("  Validação do mapeamento partido_sucessao:")
    for sigla, ano, predecessor in casos:
        sub = df_full[(df_full["sigla_partido"] == sigla) & (df_full[ANO_COL] == ano)]
        if len(sub) == 0:
            print(f"    {sigla} {ano}: 0 linhas no merged (sem dado)")
            continue
        n = len(sub)
        n_lag_pop = int(sub["lag_share_1t"].notna().sum())
        n_suc_pop = int(sub["lag_share_1t_sucessao"].notna().sum())
        n_diff = int(
            (sub["lag_share_1t_sucessao"].fillna(-1) != sub["lag_share_1t"].fillna(-1)).sum()
        )
        lag_med = sub["lag_share_1t"].mean()
        suc_med = sub["lag_share_1t_sucessao"].mean()
        print(f"    {sigla} {ano} -> canônica '{predecessor}':")
        print(f"        n_linhas={n}")
        print(f"        lag_share_1t            populado: {n_lag_pop}/{n} "
              f"(mean={lag_med:.4f})")
        print(f"        lag_share_1t_sucessao   populado: {n_suc_pop}/{n} "
              f"(mean={suc_med:.4f})")
        print(f"        casos onde divergem (sucessão aplicou): {n_diff}/{n}")
    print()


def _tabela_detalhe(sel: pd.DataFrame, k: int = 40) -> None:
    show = sel.sort_values("erro", ascending=False).head(k)
    cols = [
        "id_municipio", "sigla_partido", "numero_candidato", "nome_candidato",
        "y_true", PRED_COL, "erro",
        "lag_share_1t", "lag_share_1t_sucessao",
        "share_gov_partido", "alinhado_gov_vigente_partido",
    ]
    cols_present = [c for c in cols if c in show.columns]
    print(f"  Top {min(k, len(sel))} linhas — ordenadas por erro signed (subestimação primeiro):")
    print()
    pd.set_option("display.max_colwidth", 24)
    pd.set_option("display.width", 220)
    print(show[cols_present].to_string(index=False, float_format="%.3f"))
    print()


def _contraste_resto(df: pd.DataFrame, pred_min: float, pred_max: float) -> None:
    fora = df[~((df[PRED_COL] >= pred_min) & (df[PRED_COL] <= pred_max))]
    err_fora = fora["y_true"] - fora[PRED_COL]
    print("=" * 78)
    print(f"CONTRASTE: resto do teste (pred fora da faixa)")
    print("=" * 78)
    print(f"  n={len(fora)}")
    print(f"  erro mean={err_fora.mean():+.4f}  med={err_fora.median():+.4f}")
    print(f"  |erro| mean={err_fora.abs().mean():.4f}")
    print()


# ------------------------------------------------------------
# main
# ------------------------------------------------------------
def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)

    preds, feats = _carregar()
    df = _join(preds, feats)

    _resumo_geral(df)

    sel = _selecionar_faixa(df, args.pred_min, args.pred_max)
    if len(sel) == 0:
        print("Nada na faixa — encerrando.")
        return 0

    _por_partido(sel)
    _foco_partidos_suspeitos(sel)
    _validar_sucessao_aplicada(df)
    _tabela_detalhe(sel, k=args.top_k)
    _contraste_resto(df, args.pred_min, args.pred_max)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
