#!/usr/bin/env python
"""
scripts/05_aggregate.py — Fase 5: agregação município → UF → nacional.

Lê predições municipais de `data/processed/preds[_prefeito].parquet`,
agrega via média ponderada por `total_votos_mun` (proxy de eleitorado)
até UF e até nível nacional, propaga incerteza dos intervalos
conformais via Monte Carlo, salva
`data/processed/previsao_uf[_prefeito].parquet` e
`data/processed/previsao_nacional[_prefeito].parquet`, e gera relatório
em `reports/status_fase_5[_prefeito].md`.

Critério de sucesso (Fase 5):
  * Sem `--renormalizar mun`: a soma de share_pred por (UF, ano) pode
    ser != 1 — reflete o bias L1 do LGBM. Critério é só a cobertura.
  * Com `--renormalizar mun`: soma de share_pred por (UF, ano) ≈ 1.0
    (tol. 1%) por construção; checagem é sanity da matemática.
  * Cobertura nacional retrospectiva (último ano de teste) ≥ 0.85
    quando comparada com agregado oficial (y_real ponderado).

Uso:
    # Presidencial (default)
    python scripts/05_aggregate.py

    # Prefeito
    python scripts/05_aggregate.py --eixo prefeito

    # Renormalizar pra forçar soma=1 (recomendado para previsão final)
    python scripts/05_aggregate.py --renormalizar mun
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.aggregation import aggregate as agg  # noqa: E402
from src.config import MODE, PATHS  # noqa: E402
from src.features import io as fio  # noqa: E402


log = logging.getLogger("05_aggregate")


# ============================================================
# Configuração de eixo (presidencial vs prefeito)
# ============================================================
EIXO_CFG = {
    "presidencial": {
        "preds_name": "preds",
        "ano_col": "ano_presidencial",
        "pred_col": "pred_LightGBM_v1_iso",
        # Default: Mondrian (melhor cobertura no presidencial — y bimodal)
        "pred_lower_col": "pred_lower_mondrian",
        "pred_upper_col": "pred_upper_mondrian",
        "out_uf": "previsao_uf",
        "out_nacional": "previsao_nacional",
        "report_name": "status_fase_5.md",
    },
    "prefeito": {
        "preds_name": "preds_prefeito",
        "ano_col": "ano_municipal",
        "pred_col": "pred_LightGBM_prefeito_v1_iso",
        # Default: CQR (melhor cobertura no prefeito — y mais simétrico)
        "pred_lower_col": "pred_lower_cqr",
        "pred_upper_col": "pred_upper_cqr",
        "out_uf": "previsao_uf_prefeito",
        "out_nacional": "previsao_nacional_prefeito",
        "report_name": "status_fase_5_prefeito.md",
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 5 — agregação UF/nacional")
    p.add_argument("--eixo", type=str, default="presidencial",
                   choices=list(EIXO_CFG.keys()),
                   help="eixo: presidencial (default) ou prefeito.")
    p.add_argument("--pred-col", type=str, default=None,
                   help="coluna de pred pontual a agregar. Default depende do "
                        "eixo (pred_LightGBM_v1_iso para presidencial, "
                        "pred_LightGBM_prefeito_v1_iso para prefeito).")
    p.add_argument("--renormalizar", type=str, default="none",
                   choices=["none", "mun"],
                   help="estratégia de renormalização das predições antes de "
                        "agregar. 'none' (default): mantém shares como modelo "
                        "previu (soma por UF pode ser != 1 por causa do bias "
                        "L1 do LGBM). 'mun': divide preds por sum_p pred[m,p] "
                        "em cada município, forçando soma=1 — preserva razões "
                        "entre partidos mas neutraliza o bias L1.")
    p.add_argument("--pred-lower-col", type=str, default=None,
                   help="coluna de limite inferior do intervalo. Default depende "
                        "do eixo (Mondrian para presidencial, CQR para prefeito).")
    p.add_argument("--pred-upper-col", type=str, default=None,
                   help="coluna de limite superior do intervalo.")
    p.add_argument("--peso-col", type=str, default="total_votos_mun",
                   help="coluna de peso (eleitorado proxy). Default: "
                        "total_votos_mun.")
    p.add_argument("--n-samples", type=int, default=1000,
                   help="amostras Monte Carlo. Default: 1000.")
    p.add_argument("--alpha", type=float, default=0.10,
                   help="nível de erro (1-α = cobertura). Default 0.10 -> IC 90%%.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--soma-tol", type=float, default=0.01,
                   help="tolerância pra soma de shares ~= 1 por (UF, ano). "
                        "Default 0.01 (1%%).")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


def _resolver_coluna_intervalo(
    preds: pd.DataFrame, candidatos: list[str], qual: str
) -> str | None:
    for c in candidatos:
        if c in preds.columns:
            return c
    log.warning("nenhuma coluna de intervalo (%s) encontrada entre %s; "
                "agregação seguirá sem intervalos", qual, candidatos)
    return None


def main() -> None:
    args = parse_args()
    configurar_logging(args.log_level)

    cfg = EIXO_CFG[args.eixo]
    log.info("Fase 5 — eixo=%s, mode=%s, renormalizar=%s",
             args.eixo, MODE, args.renormalizar)

    # 1) Ler preds
    preds = fio.load_processed(cfg["preds_name"])
    log.info("preds: %d linhas, cols=%s", len(preds), preds.columns.tolist())

    # 2) Resolver colunas de intervalo
    if args.pred_lower_col is not None:
        lo_col = args.pred_lower_col if args.pred_lower_col in preds.columns else None
        hi_col = args.pred_upper_col if args.pred_upper_col in preds.columns else None
    else:
        lo_col = _resolver_coluna_intervalo(
            preds, [cfg["pred_lower_col"], "pred_lower"], "lower"
        )
        hi_col = _resolver_coluna_intervalo(
            preds, [cfg["pred_upper_col"], "pred_upper"], "upper"
        )
    log.info("usando intervalos: lower=%s upper=%s", lo_col, hi_col)

    pred_col = args.pred_col if args.pred_col is not None else cfg["pred_col"]
    if pred_col not in preds.columns:
        raw_candidates = [
            pred_col.replace("_iso", ""),
            "pred_LightGBM_v1",
            "pred_LightGBM_prefeito_v1",
        ]
        fallback = next((c for c in raw_candidates if c in preds.columns), None)
        if fallback is None:
            raise SystemExit(
                f"coluna {pred_col!r} não encontrada em preds. Disponíveis: "
                f"{[c for c in preds.columns if c.startswith('pred_')]}"
            )
        log.warning("coluna pred=%s não existe; caindo para %s", pred_col, fallback)
        pred_col = fallback

    # 2.1) Renormalização opcional
    if args.renormalizar == "mun":
        idx_cols = [cfg["ano_col"], "id_municipio"]
        s = preds.groupby(idx_cols, observed=True)[pred_col].transform("sum")
        s_safe = s.where(s > 0, 1.0)
        preds = preds.copy()
        preds[pred_col + "_renorm"] = preds[pred_col] / s_safe
        if lo_col is not None and hi_col is not None:
            preds[lo_col + "_renorm"] = preds[lo_col] / s_safe
            preds[hi_col + "_renorm"] = preds[hi_col] / s_safe
            lo_col, hi_col = lo_col + "_renorm", hi_col + "_renorm"
        pred_col = pred_col + "_renorm"
        log.warning("renormalizando preds por município: sum_p pred[m,p] = 1. "
                    "Neutraliza o bias L1 do LGBM mas preserva razoes entre partidos.")

    # 3) mun -> UF
    log.info("agregando mun -> UF ...")
    df_uf = agg.agregar_municipal_para_uf(
        preds,
        peso_col=args.peso_col,
        pred_col=pred_col,
        ano_col=cfg["ano_col"],
        pred_lower_col=lo_col,
        pred_upper_col=hi_col,
        n_samples=args.n_samples,
        alpha=args.alpha,
        seed=args.seed,
    )
    fio.save_processed(df_uf, cfg["out_uf"])

    sanity_uf = agg.verificar_soma_unitaria(
        df_uf,
        keys=[cfg["ano_col"], "sigla_uf"],
        share_col="share_pred",
        tolerancia=args.soma_tol,
    )

    # 4) UF -> nacional
    log.info("agregando UF -> nacional ...")
    df_nac = agg.agregar_uf_para_nacional(
        df_uf,
        peso_col="eleitorado_uf",
        share_col="share_pred",
        ano_col=cfg["ano_col"],
        share_lower_col="share_lower",
        share_upper_col="share_upper",
        n_samples=args.n_samples,
        alpha=args.alpha,
        seed=args.seed + 1,
    )
    fio.save_processed(df_nac, cfg["out_nacional"])

    sanity_nac = agg.verificar_soma_unitaria(
        df_nac,
        keys=[cfg["ano_col"]],
        share_col="share_pred",
        tolerancia=args.soma_tol,
    )

    # 5) Cobertura
    cob_nacional = agg.cobertura_agregada(df_nac) if "y_real" in df_nac.columns else float("nan")
    cob_uf = agg.cobertura_agregada(df_uf) if "y_real" in df_uf.columns else float("nan")
    log.info("cobertura agregada — UF: %.3f | nacional: %.3f", cob_uf, cob_nacional)

    # 6) Relatório
    relatorio = _montar_relatorio(
        eixo=args.eixo,
        cfg=cfg,
        df_uf=df_uf,
        df_nac=df_nac,
        sanity_uf=sanity_uf,
        sanity_nac=sanity_nac,
        cob_uf=cob_uf,
        cob_nacional=cob_nacional,
        pred_col=pred_col,
        lo_col=lo_col,
        hi_col=hi_col,
        n_samples=args.n_samples,
        alpha=args.alpha,
        renormalizar=args.renormalizar,
    )
    report_path = PATHS["reports"] / cfg["report_name"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(relatorio, encoding="utf-8")
    log.info("relatório: %s", report_path)

    # 7) Critérios
    falhas = []
    if args.renormalizar == "mun" and sanity_uf.n_violacoes > 0:
        falhas.append(
            f"{sanity_uf.n_violacoes}/{sanity_uf.n_grupos} (UF, ano) com soma "
            f"fora de [{1-args.soma_tol:.3f}, {1+args.soma_tol:.3f}] "
            "(--renormalizar mun deveria ter forçado soma=1)"
        )
    elif args.renormalizar == "none" and sanity_uf.n_violacoes > 0:
        log.info(
            "soma de shares por UF != 1 (média %.3f) — esperado quando "
            "--renormalizar=none por causa do bias L1 do LGBM. Para previsão "
            "final reportável, considerar --renormalizar mun.",
            sanity_uf.soma_media,
        )
    if not pd.isna(cob_nacional) and cob_nacional < 0.85:
        falhas.append(f"cobertura nacional {cob_nacional:.3f} < 0.85")
    if falhas:
        log.warning("Fase 5 com pendências: %s", "; ".join(falhas))
    else:
        log.info("Fase 5 OK.")


def _montar_relatorio(
    *,
    eixo: str,
    cfg: dict,
    df_uf: pd.DataFrame,
    df_nac: pd.DataFrame,
    sanity_uf,
    sanity_nac,
    cob_uf: float,
    cob_nacional: float,
    pred_col: str,
    lo_col: str | None,
    hi_col: str | None,
    n_samples: int,
    alpha: float,
    renormalizar: str,
) -> str:
    ano_col = cfg["ano_col"]
    titulo = f"Fase 5 — Agregação {eixo} (município → UF → nacional)"

    L: list[str] = []
    L.append(f"# {titulo}")
    L.append("")
    L.append(f"**Modo:** {MODE} | **Eixo:** `{ano_col}` | "
             f"**Pred col:** `{pred_col}` | **Renormalizar:** `{renormalizar}`")
    L.append(f"**Intervalos:** lower=`{lo_col}` upper=`{hi_col}` | "
             f"**MC samples:** {n_samples} | **α:** {alpha} (IC {(1-alpha)*100:.0f}%)")
    L.append("")

    L.append("## Sanity check — soma de shares por (UF, ano)")
    L.append("")
    L.append(
        f"Tolerância: ±{sanity_uf.tolerancia:.3f} | grupos: {sanity_uf.n_grupos} | "
        f"violações: **{sanity_uf.n_violacoes}**"
    )
    L.append("")
    L.append(
        f"Soma — min: {sanity_uf.soma_min:.4f} | max: {sanity_uf.soma_max:.4f} | "
        f"média: {sanity_uf.soma_media:.4f}"
    )
    L.append("")
    if renormalizar == "none" and sanity_uf.n_violacoes > 0:
        L.append(
            "> **Nota:** com `--renormalizar=none`, a soma de shares por UF "
            "pode ser < 1. Isso reflete o bias L1 do LGBM (subestima shares "
            "uniformemente, ~0.016/linha × n_partidos). Não é bug do "
            "agregador. Para previsão final reportável, use `--renormalizar mun`."
        )
        L.append("")

    if sanity_uf.n_violacoes > 0:
        L.append("### Violadores (top 10)")
        L.append("")
        L.append(sanity_uf.detalhes.head(10).to_markdown(index=False, floatfmt=".4f"))
        L.append("")

    L.append("## Sanity check — soma de shares nacional por ano")
    L.append("")
    L.append(
        f"grupos: {sanity_nac.n_grupos} | violações: **{sanity_nac.n_violacoes}** | "
        f"min={sanity_nac.soma_min:.4f} max={sanity_nac.soma_max:.4f}"
    )
    L.append("")

    L.append("## Cobertura empírica dos intervalos agregados")
    L.append("")
    L.append("| nível | cobertura | nominal |")
    L.append("| --- | --- | --- |")
    L.append(f"| UF | {cob_uf:.4f} | {1-alpha:.2f} |")
    L.append(f"| Nacional | {cob_nacional:.4f} | {1-alpha:.2f} |")
    L.append("")
    L.append(
        "> Cobertura agregada = fração de (UF×partido) ou (ano×partido) "
        "onde `y_real` (média ponderada do `y_true`) cai dentro de "
        "[share_lower, share_upper]."
    )
    L.append("")

    # Top partidos nacional + bias
    L.append("## Top partidos por share nacional (último ano)")
    L.append("")
    if not df_nac.empty:
        ano_max = df_nac[ano_col].max()
        top = df_nac[df_nac[ano_col] == ano_max].copy()
        top = top.sort_values("share_pred", ascending=False)
        cols_show = ["sigla_partido", "share_pred"]
        if "share_lower" in top.columns:
            cols_show += ["share_lower", "share_upper"]
        if "y_real" in top.columns:
            cols_show += ["y_real"]
            top["bias_nacional"] = top["share_pred"] - top["y_real"]
            cols_show += ["bias_nacional"]
        cols_show += ["n_ufs", "eleitorado_total"]
        cols_show = [c for c in cols_show if c in top.columns]
        L.append(f"### Ano = {ano_max}")
        L.append("")
        L.append(top[cols_show].head(15).to_markdown(index=False, floatfmt=".4f"))
        L.append("")
        L.append(
            "> `bias_nacional = share_pred - y_real`. Negativo = modelo "
            "subestima o partido no agregado nacional."
        )
        L.append("")

    # Diagnóstico: UFs onde o intervalo agregado nao cobriu y_real
    if not df_uf.empty and "y_real" in df_uf.columns and \
            "share_lower" in df_uf.columns:
        L.append("## UFs onde o intervalo agregado **NÃO cobriu** y_real")
        L.append("")
        ano_max = df_uf[ano_col].max()
        sub = df_uf[df_uf[ano_col] == ano_max].copy()
        fora = sub[(sub["y_real"] < sub["share_lower"]) |
                   (sub["y_real"] > sub["share_upper"])].copy()
        if fora.empty:
            L.append("_Todos os (UF, partido) cobertos._")
            L.append("")
        else:
            fora["erro"] = (fora["y_real"] - fora["share_pred"]).round(4)
            fora = fora.reindex(fora["erro"].abs().sort_values(ascending=False).index)
            cols = ["sigla_uf", "sigla_partido", "share_pred", "share_lower",
                    "share_upper", "y_real", "erro"]
            cols = [c for c in cols if c in fora.columns]
            L.append(f"### Ano = {ano_max}  ({len(fora)}/{len(sub)} fora)")
            L.append("")
            L.append(fora[cols].head(20).to_markdown(index=False, floatfmt=".4f"))
            L.append("")
            # Concentração por partido
            por_partido = fora.groupby("sigla_partido").size().sort_values(ascending=False)
            L.append("### Distribuição dos descobertos por partido")
            L.append("")
            for p, n in por_partido.head(10).items():
                L.append(f"- `{p}`: {n} UFs descobertas")
            L.append("")
            L.append(
                "> Concentração em poucos partidos indica viés estrutural do "
                "LGBM (e.g., PL 2022 — efeito migração Bolsonaro PSL→PL). "
                "Solução não é o agregador — é #60 (pesquisas como feature)."
            )
            L.append("")

    # Top UFs do partido vencedor
    L.append("## UFs — partido com maior share por UF (último ano)")
    L.append("")
    if not df_uf.empty:
        ano_max = df_uf[ano_col].max()
        sub = df_uf[df_uf[ano_col] == ano_max].copy()
        idx = sub.groupby("sigla_uf")["share_pred"].idxmax()
        winners = sub.loc[idx].sort_values("sigla_uf")
        cols_show = ["sigla_uf", "sigla_partido", "share_pred"]
        if "share_lower" in winners.columns:
            cols_show += ["share_lower", "share_upper"]
        if "y_real" in winners.columns:
            cols_show += ["y_real"]
        cols_show = [c for c in cols_show if c in winners.columns]
        L.append(winners[cols_show].to_markdown(index=False, floatfmt=".4f"))
        L.append("")

    L.append("## Notas")
    L.append("")
    L.append(
        "- `eleitorado_uf`/`eleitorado_total` é a soma de `total_votos_mun`, "
        "proxy do eleitorado registrado (correlação > 0.95)."
    )
    L.append(
        "- Intervalos agregados via Monte Carlo: para cada linha (mun, partido), "
        "sortear uniforme centrada em pred com semi-largura (hi-lo)/2; agregar "
        "ponderando por `total_votos_mun`; pegar percentis "
        f"{alpha/2:.3f} e {1-alpha/2:.3f}."
    )
    L.append(
        "- Independência entre partidos no MC dentro do mesmo município "
        "ignora a restrição sum_partido share ~= 1. O efeito é alargar "
        "levemente os intervalos agregados (conservador)."
    )
    if renormalizar == "mun":
        L.append(
            "- `--renormalizar mun` está ativo: predições foram divididas por "
            "sum_p pred[m,p] em cada município. Os intervalos foram reescalados "
            "pelo mesmo fator (preserva forma relativa)."
        )
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
