#!/usr/bin/env python
"""
scripts/diag_topcauda.py — diagnóstico da descalibração no top decil
do LightGBM_prefeito_v1 (Fase 4.5, teste 2024).

O `reports/status_fase_4_5.md` mostra que o top decil do teste tem:
    pred_medio = 0.6983
    real_medio = 0.7930
    erro_decil = +0.0947  (subestima 9.5pp)

Este script responde:

    1. Quem são os candidatos do top decil? (nome / partido / município)
    2. O bias é uniforme ou tem outliers?
    3. Há padrão por partido / por tamanho de município?
    4. O modelo está SATURANDO em features-âncora ou aprendeu signal a mais?
       Compara pred vs lag_share_1t pra ver se o LGBM só copiou o lag,
       ou se andou na direção certa mas sem atravessar.
    5. Esses candidatos são prefeitos incumbentes / sucessores formais?

Uso:
    python scripts/diag_topcauda.py [--quantil 0.9] [--log-level INFO]

Output: texto no console.
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
from src.models import features_prefeito as mfp  # noqa: E402
from src.models import transforms as tfm  # noqa: E402

log = logging.getLogger("diag_topcauda")


PRED_COL = "pred_LightGBM_prefeito_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="diagnóstico da cauda — Fase 4.5")
    p.add_argument("--quantil", type=float, default=0.9,
                   help="corte de cauda (default 0.9 = top 10%%).")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


def _carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carrega preds_prefeito e features_prefeito (filtra ano_teste)."""
    preds = fio.load_processed("preds_prefeito")
    feats = fio.load_processed("features_prefeito")

    # ano_teste = max ano_municipal nos preds
    ano_teste = int(preds["ano_municipal"].max())
    log.info("ano_teste detectado: %d", ano_teste)

    feats_te = feats[feats["ano_municipal"] == ano_teste].copy()
    log.info("preds: %d linhas | features (ano_teste): %d linhas",
             len(preds), len(feats_te))
    return preds, feats_te


def _join(preds: pd.DataFrame, feats: pd.DataFrame) -> pd.DataFrame:
    """Junta preds + features no chave (ano_municipal, id_municipio, sigla_partido,
    numero_candidato).

    Validação: cardinalidade preservada (sem fan-out).
    """
    keys = ["ano_municipal", "id_municipio", "sigla_partido", "numero_candidato"]
    feats_subset_cols = keys + [
        "lag_share_1t", "lag_share_1t_sucessao", "lag2_share_1t",
        "swing_share_1t", "share_prefeito_local", "margem_prefeito",
        "log_eleitorado", "primeiro_mandato_prefeito",
        "alinhado_prefeito_partido", "alinhado_prefeito_coligacao",
        "indice_continuidade", "anos_consecutivos_mesmo_partido",
        "porte", "continuidade_classe", "share_dep_federal_partido",
    ]
    cols_present = [c for c in feats_subset_cols if c in feats.columns]
    # garante unicidade da chave nos dois lados antes do merge
    if preds.duplicated(subset=keys).any():
        n_dup = int(preds.duplicated(subset=keys).sum())
        log.warning("preds tem %d duplicatas na chave %s — mantendo first", n_dup, keys)
        preds = preds.drop_duplicates(subset=keys, keep="first")
    feats_uniq = feats[cols_present].drop_duplicates(subset=keys, keep="first")
    merged = preds.merge(feats_uniq, on=keys, how="left", validate="one_to_one")
    log.info("merge: %d linhas (%d colunas)", len(merged), merged.shape[1])
    return merged


def _resumo_geral(df: pd.DataFrame) -> None:
    y = df["y_true"].to_numpy()
    p = df[PRED_COL].to_numpy()
    err = y - p  # >0 = subestima
    print("=" * 78)
    print(f"RESUMO GERAL (n={len(df)})")
    print("=" * 78)
    print(f"  y_true:  mean={y.mean():.4f}  std={y.std():.4f}  "
          f"min={y.min():.4f}  max={y.max():.4f}")
    print(f"  pred:    mean={p.mean():.4f}  std={p.std():.4f}  "
          f"min={p.min():.4f}  max={p.max():.4f}")
    print(f"  erro signed (y-p):  mean={err.mean():+.4f}  "
          f"std={err.std():.4f}  p10={np.percentile(err,10):+.4f}  "
          f"p90={np.percentile(err,90):+.4f}")
    print()


def _selecionar_topcauda(df: pd.DataFrame, quantil: float) -> pd.DataFrame:
    corte = df["y_true"].quantile(quantil)
    top = df[df["y_true"] >= corte].copy()
    print("=" * 78)
    print(f"TOP CAUDA: y_true >= q{quantil:.2f} (corte={corte:.4f})  "
          f"-> n={len(top)}")
    print("=" * 78)
    print()
    return top


def _tabela_top(top: pd.DataFrame, k: int = 30) -> None:
    """Tabela detalhada dos candidatos do top decil, ordenada por erro signed."""
    show = top.copy()
    show["erro"] = show["y_true"] - show[PRED_COL]
    show = show.sort_values("erro", ascending=False)

    cols = [
        "id_municipio", "sigla_partido", "numero_candidato",
        "nome_candidato", "y_true", PRED_COL, "erro",
        "lag_share_1t", "share_prefeito_local", "margem_prefeito",
        "log_eleitorado", "primeiro_mandato_prefeito",
        "alinhado_prefeito_partido",
    ]
    cols_present = [c for c in cols if c in show.columns]
    print(f"  Top {min(k, len(show))} linhas — ordenadas por erro signed (subestimação primeiro):")
    print()
    pd.set_option("display.max_colwidth", 32)
    pd.set_option("display.width", 200)
    print(show[cols_present].head(k).to_string(index=False, float_format="%.3f"))
    print()


def _por_partido(top: pd.DataFrame) -> None:
    show = top.copy()
    show["erro"] = show["y_true"] - show[PRED_COL]
    g = show.groupby("sigla_partido").agg(
        n=("y_true", "size"),
        y_mean=("y_true", "mean"),
        pred_mean=(PRED_COL, "mean"),
        erro_mean=("erro", "mean"),
    ).sort_values("n", ascending=False)
    print("  Breakdown por partido (top cauda):")
    print(g.to_string(float_format="%.4f"))
    print()


def _comparar_com_lag(top: pd.DataFrame, df_geral: pd.DataFrame) -> None:
    """O modelo está SATURANDO em lag_share_1t? Compara para a cauda:
        pred vs lag_share_1t vs y_true.

    Se pred ≈ lag e y_true >> lag, o modelo só copiou o lag e perdeu
    a "subida". Se pred > lag mas pred < y_true, o modelo andou na
    direção certa mas conservadoramente.
    """
    if "lag_share_1t" not in top.columns:
        print("  [skip: lag_share_1t não disponível]")
        return

    show = top.dropna(subset=["lag_share_1t"]).copy()
    n_lag_nan = len(top) - len(show)

    # decompõe o erro: y - pred = (y - lag) + (lag - pred)
    #   (y - lag) = "subida real do partido naquele municipio"
    #   (lag - pred) = "quanto o modelo se afastou do lag"
    show["delta_real"] = show["y_true"] - show["lag_share_1t"]      # subida verdadeira
    show["delta_modelo"] = show[PRED_COL] - show["lag_share_1t"]    # subida do modelo
    show["erro"] = show["y_true"] - show[PRED_COL]

    print("  Decomposição do erro vs lag_share_1t (na cauda):")
    print(f"    n com lag disponível: {len(show)}  (NaN: {n_lag_nan})")
    print(f"    subida real (y - lag) : "
          f"mean={show['delta_real'].mean():+.4f}  "
          f"med={show['delta_real'].median():+.4f}")
    print(f"    subida modelo (p - lag): "
          f"mean={show['delta_modelo'].mean():+.4f}  "
          f"med={show['delta_modelo'].median():+.4f}")
    print(f"    razão de captura       : "
          f"mean(p-lag)/mean(y-lag) = "
          f"{show['delta_modelo'].mean()/show['delta_real'].mean():.2%}")
    print()

    # quanto da cauda tem lag NaN (partido novo no município)? Esses
    # casos são particularmente difíceis: o LGBM precisa adivinhar do zero.
    if n_lag_nan > 0:
        nan_rows = top[top["lag_share_1t"].isna()]
        print(f"    [{n_lag_nan} linha(s) sem lag — partido sem histórico no município]")
        print(f"    erro signed nesses casos: "
              f"mean={(nan_rows['y_true'] - nan_rows[PRED_COL]).mean():+.4f}")
        print()


def _saturacao_logit(top: pd.DataFrame) -> None:
    """No espaço logit, quanto o modelo está afastado do alvo?

    A sigmoid comprime extremos: 0.95 em share = 2.94 em logit; 0.99 = 4.6.
    Se o modelo aprendeu logits altos mas não suficientemente, esse plot
    dá pra ver. Para um candidato com share_real=0.85 e share_pred=0.70,
    o erro em logit é 1.73 - 0.85 = 0.88 (substancial).
    """
    show = top.copy()
    show["logit_y"] = tfm.logit_share(show["y_true"])
    show["logit_p"] = tfm.logit_share(show[PRED_COL])
    show["erro_logit"] = show["logit_y"] - show["logit_p"]
    print("  Erro no espaço logit (cauda):")
    print(f"    logit(y_true): mean={show['logit_y'].mean():+.3f}  "
          f"max={show['logit_y'].max():+.3f}")
    print(f"    logit(pred) : mean={show['logit_p'].mean():+.3f}  "
          f"max={show['logit_p'].max():+.3f}")
    print(f"    erro logit  : mean={show['erro_logit'].mean():+.3f}  "
          f"med={show['erro_logit'].median():+.3f}")
    print()


def _incumbencia_no_topo(top: pd.DataFrame) -> None:
    """Quanto do topo é prefeito incumbente / sucessor formal?

    `alinhado_prefeito_partido == 1` = candidato do mesmo partido do prefeito
    vigente no município. É a melhor proxy disponível pra "incumbente
    concorrendo à reeleição ou sucessor formal".
    """
    if "alinhado_prefeito_partido" not in top.columns:
        print("  [skip: alinhado_prefeito_partido não disponível]")
        return
    s = top["alinhado_prefeito_partido"]
    n = s.notna().sum()
    n_alinhado = (s == 1).sum()
    n_oposicao = (s == 0).sum()
    print("  Status incumbência (alinhado_prefeito_partido):")
    print(f"    aligned (sucessor / reeleição) : {n_alinhado}/{n} "
          f"({100*n_alinhado/max(n,1):.1f}%)")
    print(f"    oposição                       : {n_oposicao}/{n}")
    if n_alinhado > 0:
        sub_a = top[top["alinhado_prefeito_partido"] == 1]
        sub_o = top[top["alinhado_prefeito_partido"] == 0]
        print(f"    erro mean (aligned):  "
              f"{(sub_a['y_true'] - sub_a[PRED_COL]).mean():+.4f}")
        if len(sub_o) > 0:
            print(f"    erro mean (oposição): "
                  f"{(sub_o['y_true'] - sub_o[PRED_COL]).mean():+.4f}")
    print()


def _municipio_porte(top: pd.DataFrame) -> None:
    if "log_eleitorado" not in top.columns:
        return
    show = top.copy()
    show["erro"] = show["y_true"] - show[PRED_COL]
    show = show.dropna(subset=["log_eleitorado"])
    # divide em terços de log_eleitorado
    try:
        show["bucket"] = pd.qcut(show["log_eleitorado"], 3,
                                  labels=["pequeno", "medio", "grande"], duplicates="drop")
    except ValueError:
        # pouca variação no log_eleitorado da cauda — cai pra 2 buckets
        show["bucket"] = pd.qcut(show["log_eleitorado"], 2,
                                  labels=["menor", "maior"], duplicates="drop")
    g = show.groupby("bucket", observed=True).agg(
        n=("y_true", "size"),
        log_elec_med=("log_eleitorado", "median"),
        y_mean=("y_true", "mean"),
        pred_mean=(PRED_COL, "mean"),
        erro_mean=("erro", "mean"),
    )
    print("  Erro por porte de município (terços de log_eleitorado, na cauda):")
    print(g.to_string(float_format="%.4f"))
    print()


def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)

    preds, feats = _carregar()
    df = _join(preds, feats)

    _resumo_geral(df)

    top = _selecionar_topcauda(df, args.quantil)
    _tabela_top(top, k=30)
    _por_partido(top)
    _comparar_com_lag(top, df)
    _saturacao_logit(top)
    _incumbencia_no_topo(top)
    _municipio_porte(top)

    # contraste: mesmo resumo no resto pra dar régua
    print("=" * 78)
    print("CONTRASTE: resto do teste (y_true < corte)")
    print("=" * 78)
    resto = df[df["y_true"] < df["y_true"].quantile(args.quantil)]
    err_resto = resto["y_true"] - resto[PRED_COL]
    print(f"  n={len(resto)}  erro mean={err_resto.mean():+.4f}  "
          f"erro med={err_resto.median():+.4f}  "
          f"|erro| mean={err_resto.abs().mean():.4f}")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
