#!/usr/bin/env python
"""
scripts/04_train.py — Fase 4: primeiro modelo presidencial.

Carrega `data/processed/features.parquet`, roda split temporal
(treino=2014+2018, teste=2022), ajusta baselines (B0/B1/B2) e LightGBM,
salva predições em `data/processed/preds.parquet`, o modelo em
`models/lgbm_v1.pkl` e gera `reports/status_fase_4.md`.

Flags opt-in (replicadas do `04_train_prefeito.py`):

  * `--calibrate` ajusta IsotonicCalibrator pós-hoc (corrige saturação no
    top decil). Default `--calib-mode holdout` separa um ano de calibração.
  * `--conformal` ajusta SplitConformal sobre os resíduos do conjunto de
    calibração e adiciona pred_lower/pred_upper. `--conformal-mondrian`
    também salva versão estratificada por bin de pred.

Uso:
    python scripts/04_train.py [--log-level INFO] [--no-save-model]
    python scripts/04_train.py --calibrate --calib-min-pred 0.5
    python scripts/04_train.py --calibrate --conformal --conformal-mondrian
"""
from __future__ import annotations

import argparse
import logging
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import MODE, MODE_CFG, PATHS, set_global_seed, summary  # noqa: E402
from src.features import io as fio  # noqa: E402
from src.models import baseline as bl  # noqa: E402
from src.models import calibrate as cal  # noqa: E402
from src.models import conformal as cf  # noqa: E402
from src.models import cqr as cqr_mod  # noqa: E402
from src.models import evaluate as ev  # noqa: E402
from src.models import features as mf  # noqa: E402
from src.models import train as tr  # noqa: E402


log = logging.getLogger("04_train")

ANO_COL_PRES = "ano_presidencial"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 4 — treino e avaliação")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-save-model", action="store_true",
                   help="não escreve models/lgbm_v1.pkl")
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="fração do treino pra validação interna (early stopping). "
                        "0 desliga. Padrão: 0.")
    p.add_argument("--calibrate", action="store_true",
                   help="ajusta IsotonicCalibrator e adiciona pred calibrada "
                        "(LightGBM_v1_iso) ao comparativo.")
    p.add_argument("--calib-mode", type=str, default="holdout",
                   choices=["holdout", "oof"],
                   help="estratégia pra gerar predições no treino: "
                        "'holdout' (default, separa o último ano de treino "
                        "como conjunto de calibração) ou 'oof' "
                        "(leave-one-year-out cross-validation).")
    p.add_argument("--calib-min-pred", type=float, default=0.5,
                   help="limiar pra calibração assimétrica: predições raw "
                        "abaixo desse valor passam direto (sem isotonic). "
                        "0.0 desliga (calibra todos). Default 0.5.")
    p.add_argument("--calib-ano", type=int, default=None,
                   help="ano de holdout pra calibração. Default: último de "
                        "anos_treino. Ignorado se --calib-mode=oof.")
    p.add_argument("--conformal", action="store_true",
                   help="ajusta SplitConformal sobre resíduos do conjunto de "
                        "calibração e adiciona pred_lower/pred_upper em preds.")
    p.add_argument("--conformal-alpha", type=float, default=0.1,
                   help="nível de erro (1-α=cobertura). Default 0.1 -> IC 90%%.")
    p.add_argument("--conformal-mondrian", action="store_true",
                   help="também ajusta MondrianConformal (estratificado por bin "
                        "de pred). Salva pred_lower_mondrian/pred_upper_mondrian.")
    p.add_argument("--conformal-bins", type=int, default=10,
                   help="n_bins do MondrianConformal. Default 10 (decis).")
    p.add_argument("--conformal-min-q-factor", type=float, default=0.0,
                   help="floor mínimo no q̂ por bin do Mondrian, como fração "
                        "do q̂ global. 0 (default) = sem floor. Em prod com "
                        "muitos pontos de pred baixa, recomenda-se 0.5 pra "
                        "evitar intervalos zerados em bins degenerados.")
    p.add_argument("--cqr", action="store_true",
                   help="também ajusta CQR (Conformalized Quantile Regression). "
                        "Treina 2 LGBMs quantile (low, hi) com mesmo split e "
                        "calibra a margem conformal. Salva pred_lower_cqr/"
                        "pred_upper_cqr — intervalos adaptativos que herdam "
                        "heterocedasticidade do modelo quantile. Pressupõe "
                        "--conformal (reusa o mesmo conjunto de calibração).")
    p.add_argument("--conformal-mondrian-cat", action="store_true",
                   help="também ajusta MondrianCategorical estratificado por "
                        "(sigla_partido, regiao). Salva pred_lower_mondrian_cat/"
                        "pred_upper_mondrian_cat. Útil para cobertura "
                        "condicional por partido (e.g., decil 7 do bin-Mondrian "
                        "sub-coberto pelo PL 2022). Pressupõe --conformal.")
    p.add_argument("--conformal-mondrian-cat-cols", type=str,
                   default="sigla_partido,regiao",
                   help="colunas categóricas para o MondrianCategorical, "
                        "separadas por vírgula. Default: 'sigla_partido,regiao'. "
                        "Outras opções: 'sigla_partido' (só partido), "
                        "'sigla_partido,sigla_uf' (mais granular).")
    p.add_argument("--conformal-mondrian-cat-min-per-stratum", type=int,
                   default=10,
                   help="min_per_stratum do MondrianCategorical. Default 10.")
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


def _construir_estrato(
    *,
    df_calib: pd.DataFrame,
    prep: mf.PreparedData,
    strata_cols: list[str],
) -> np.ndarray:
    """Constrói labels de estrato pra calibração concatenando strata_cols.

    df_calib tem 'idx_original' apontando pras linhas no prep (full train).
    Para cada coluna em strata_cols, busca em prep.meta primeiro, senão
    em prep.X. Junta com '|' para virar uma label única por linha.
    """
    idx = df_calib["idx_original"].to_numpy()
    pieces: list[np.ndarray] = []
    for col in strata_cols:
        if col in prep.meta.columns:
            pieces.append(prep.meta.iloc[idx][col].astype(str).to_numpy())
        elif col in prep.X.columns:
            pieces.append(prep.X.iloc[idx][col].astype(str).to_numpy())
        else:
            raise ValueError(
                f"coluna de strata {col!r} não encontrada em prep.meta nem prep.X"
            )
    if len(pieces) == 1:
        return pieces[0]
    return np.array(["|".join(parts) for parts in zip(*pieces)])


def _construir_estrato_test(
    *,
    test: mf.PreparedData,
    strata_cols: list[str],
) -> np.ndarray:
    """Análogo a _construir_estrato mas pro test set."""
    pieces: list[np.ndarray] = []
    for col in strata_cols:
        if col in test.meta.columns:
            pieces.append(test.meta[col].astype(str).to_numpy())
        elif col in test.X.columns:
            pieces.append(test.X[col].astype(str).to_numpy())
        else:
            raise ValueError(
                f"coluna de strata {col!r} não encontrada em test.meta nem test.X"
            )
    if len(pieces) == 1:
        return pieces[0]
    return np.array(["|".join(parts) for parts in zip(*pieces)])


def determinar_split(anos_pres: list[int]) -> tuple[list[int], int]:
    """Treino = todos menos o último; teste = último.

    Em dev (2014, 2018, 2022) -> treino=[2014,2018], teste=2022.
    """
    anos = sorted(int(a) for a in anos_pres)
    if len(anos) < 2:
        raise ValueError(f"precisa de ≥ 2 anos presidenciais; got {anos}")
    return anos[:-1], anos[-1]


def rodar_baselines(
    train: mf.PreparedData,
    test: mf.PreparedData,
) -> dict[str, np.ndarray]:
    """Ajusta B0/B1/B2 e retorna dict nome→predições no teste."""
    preds: dict[str, np.ndarray] = {}

    b0 = bl.MedianaPartidoUF().fit(train.X, train.y, meta=train.meta)
    preds[b0.nome] = b0.predict(test.X, meta=test.meta)

    b1 = bl.LagShare().fit(train.X, train.y, meta=train.meta)
    preds[b1.nome] = b1.predict(test.X, meta=test.meta)

    b2 = bl.BlendB0B1(alpha=0.5).fit(train.X, train.y, meta=train.meta)
    preds[b2.nome] = b2.predict(test.X, meta=test.meta)

    return preds


def formatar_tabela_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Converte DataFrame em tabela Markdown (sem dependências extras)."""
    if len(df) == 0:
        return "_(vazio)_"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    linhas = [header, sep]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(format(v, floatfmt))
            elif isinstance(v, (int, np.integer)):
                cells.append(str(int(v)))
            else:
                cells.append(str(v))
        linhas.append("| " + " | ".join(cells) + " |")
    return "\n".join(linhas)


def gerar_relatorio(
    comp: pd.DataFrame,
    por_partido: pd.DataFrame,
    por_uf: pd.DataFrame,
    calib: pd.DataFrame,
    imp: pd.DataFrame,
    ano_teste: int,
    anos_treino: list[int],
    n_train: int,
    n_test: int,
    *,
    calib_iso: pd.DataFrame | None = None,
    por_partido_iso: pd.DataFrame | None = None,
    calib_mode: str | None = None,
    calib_min_pred: float | None = None,
    calib_ano: int | None = None,
    conformal_alpha: float | None = None,
    cobertura_split: float | None = None,
    cobertura_decil_split: pd.DataFrame | None = None,
    q_hat_split: float | None = None,
    cobertura_mondrian: float | None = None,
    cobertura_decil_mondrian: pd.DataFrame | None = None,
    mondrian_q_per_bin: list[float] | None = None,
    cobertura_cqr: float | None = None,
    cobertura_decil_cqr: pd.DataFrame | None = None,
    q_hat_cqr: float | None = None,
    cobertura_mondrian_cat: float | None = None,
    cobertura_decil_mondrian_cat: pd.DataFrame | None = None,
    cobertura_categoria_mondrian_cat: pd.DataFrame | None = None,
    mondrian_cat_strata_cols: list[str] | None = None,
    mondrian_cat_n_estratos: int | None = None,
    mondrian_cat_n_fallback: int | None = None,
) -> str:
    """Monta o markdown do status_fase_4.md."""
    linhas = [
        "# Fase 4 — status: primeiro modelo presidencial",
        "",
        f"**Modo:** {MODE} | **UFs:** {MODE_CFG['ufs']} | "
        f"**Máx municípios:** {MODE_CFG['max_municipios']}",
        f"**Split temporal:** treino = {anos_treino} ({n_train} linhas) | "
        f"teste = {ano_teste} ({n_test} linhas)",
        "",
        "## Comparativo geral (escala share ∈ [0,1])",
        "",
        formatar_tabela_md(comp),
        "",
        "> `bias` positivo = modelo subestima (predição < realidade).",
        "> `mae` e `rmse` quanto menor, melhor.",
        "",
        "## MAE por partido (top 10 piores)",
        "",
        formatar_tabela_md(por_partido.head(10)),
        "",
        "## MAE por UF",
        "",
        formatar_tabela_md(por_uf),
        "",
        "## Calibração por decil (LightGBM)",
        "",
        formatar_tabela_md(calib),
        "",
        "> Decil bem calibrado: `pred_medio ≈ real_medio` (erro_decil ≈ 0).",
        "",
    ]
    if calib_iso is not None:
        modo_label = (
            f"holdout (ano_calib={calib_ano})" if calib_mode == "holdout"
            else "OOF leave-one-year-out"
        ) if calib_mode else "isotonic"
        min_label = (
            f"asimétrico (raw quando pred < {calib_min_pred:.2f})"
            if calib_min_pred and calib_min_pred > 0
            else "global (calibra todos)"
        )
        linhas += [
            "## Calibração por decil (LightGBM + isotonic)",
            "",
            f"**Modo:** {modo_label} | **Cobertura:** {min_label}",
            "",
            formatar_tabela_md(calib_iso),
            "",
            "> Calibrador isotônico ajustado em predições do modelo no treino. "
            "Aplicado pós-hoc na predição do teste.",
            "",
        ]
        if por_partido_iso is not None:
            linhas += [
                "## MAE por partido — versão calibrada (top 10 piores)",
                "",
                formatar_tabela_md(por_partido_iso.head(10)),
                "",
            ]

    if cobertura_split is not None and conformal_alpha is not None:
        cobertura_nominal = 1.0 - conformal_alpha
        linhas += [
            "## Cobertura conformal (split)",
            "",
            f"**Cobertura nominal:** {cobertura_nominal:.0%} "
            f"(α = {conformal_alpha:.2f}) | "
            f"**q̂ split:** {q_hat_split:.4f} | "
            f"**Cobertura observada (test):** {cobertura_split:.3f}",
            "",
        ]
        if cobertura_decil_split is not None:
            linhas += [
                "**Cobertura por decil de pred — split:**",
                "",
                formatar_tabela_md(cobertura_decil_split),
                "",
                "> Decil bem coberto: `cobertura ≈ 1-α`. Quando o split é "
                "homogêneo, cobertura por decil pode divergir do nominal "
                "(intervalo grande demais nos baixos, pequeno demais nos altos).",
                "",
            ]
        if cobertura_mondrian is not None:
            linhas += [
                "## Cobertura conformal (Mondrian — estratificado por bin de pred)",
                "",
                f"**Cobertura observada (test):** {cobertura_mondrian:.3f} | "
                f"**q̂ por bin:** [{', '.join(f'{q:.3f}' for q in (mondrian_q_per_bin or []))}]",
                "",
            ]
            if cobertura_decil_mondrian is not None:
                linhas += [
                    "**Cobertura por decil de pred — Mondrian:**",
                    "",
                    formatar_tabela_md(cobertura_decil_mondrian),
                    "",
                    "> Mondrian deve dar cobertura aproximadamente uniforme "
                    "ao longo dos decis (cobertura condicional).",
                    "",
                ]
        if cobertura_mondrian_cat is not None:
            cols_label = (
                ", ".join(f"`{c}`" for c in (mondrian_cat_strata_cols or []))
                or "?"
            )
            linhas += [
                f"## Cobertura conformal (MondrianCategorical — estratos por {cols_label})",
                "",
                f"**Cobertura observada (test):** {cobertura_mondrian_cat:.3f} | "
                f"**Estratos:** {mondrian_cat_n_estratos or '?'} "
                f"(fallback global: {mondrian_cat_n_fallback or 0})",
                "",
            ]
            if cobertura_decil_mondrian_cat is not None:
                linhas += [
                    "**Cobertura por decil de pred — MondrianCategorical:**",
                    "",
                    formatar_tabela_md(cobertura_decil_mondrian_cat),
                    "",
                ]
            if cobertura_categoria_mondrian_cat is not None:
                # Top 10 piores estratos (cobertura mais baixa)
                piores = cobertura_categoria_mondrian_cat.head(10)
                linhas += [
                    "**Top 10 estratos com menor cobertura empírica:**",
                    "",
                    formatar_tabela_md(piores),
                    "",
                    "> MondrianCategorical foca em estratos categóricos (e.g., "
                    "partido) onde o regime de erro pode ser distinto. "
                    "Estratos sub-cobertos sinalizam exchangeability quebrada "
                    "entre calib↔test (caso típico: PL 2022 com migração do "
                    "Bolsonaro).",
                    "",
                ]
        if cobertura_cqr is not None:
            qhat_label = (
                f"{q_hat_cqr:+.4f}" if q_hat_cqr is not None else "n/a"
            )
            linhas += [
                "## Cobertura conformal (CQR — Conformalized Quantile Regression)",
                "",
                f"**Cobertura observada (test):** {cobertura_cqr:.3f} | "
                f"**q̂ CQR:** {qhat_label}",
                "",
                "> CQR usa 2 LGBMs quantile pra modelar `[q_low(x), q_hi(x)]` "
                "diretamente, depois conformaliza a margem. Intervalos "
                "adaptativos: largura cresce onde o modelo prevê dispersão maior.",
                "",
            ]
            if cobertura_decil_cqr is not None:
                linhas += [
                    "**Cobertura por decil de pred — CQR:**",
                    "",
                    formatar_tabela_md(cobertura_decil_cqr),
                    "",
                ]

    linhas += [
        "## Top feature importance (LightGBM, gain)",
        "",
        formatar_tabela_md(imp),
        "",
        "## Notas",
        "",
        "- Target modelado em `logit(share)`, predição destransformada com sigmoid.",
        "- LightGBM com `objective=regression_l1` (MAE) — robusto a caudas.",
        "- Features categóricas (sigla_uf, regiao, porte, continuidade_classe, sigla_partido) "
        "tratadas nativamente pelo LightGBM.",
        "- Amostra dev = 1 UF × 100 municípios × 3 anos × ~10 partidos = ~3 mil linhas. "
        "Signal-to-noise é limitado — resultados aqui servem pra validar pipeline, não pra "
        "conclusões sobre 2026.",
    ]
    if calib_iso is not None:
        linhas += [
            "- Versão `_iso` corrige saturação no top decil via regressão isotônica "
            "treinada num ano holdout do conjunto de treino. Não toca o LGBM.",
        ]
    if cobertura_split is not None:
        linhas += [
            "- Cobertura conformal é uma propriedade do conjunto de calibração: o "
            "número observado no test é descritivo, não uma garantia (a garantia "
            "vale sob exchangeability calib↔test).",
        ]
    linhas += [
        "",
        "## Próximos passos",
        "",
        "- **Fase 5+**: revisar incerteza com mais anos em prod, e considerar "
        "conformal por estratos de UF/região (não só por bin de pred).",
        "- **Investigar**: se o LightGBM não bate B1 (`lag_share_1t`) de forma contundente, "
        "reavaliar features históricas — pode ser que precisemos de mais anos no treino "
        "(rodar em prod).",
    ]
    return "\n".join(linhas)


def main() -> int:
    args = parse_args()
    configurar_logging(args.log_level)
    set_global_seed()
    log.info(summary())

    # 1) Carrega features
    df = fio.load_processed("features")
    log.info("features: %s", df.shape)

    # 2) Prep
    prep = mf.preparar_X_y(df)
    anos = MODE_CFG["anos_presidencial"]
    anos_treino, ano_teste = determinar_split(anos)
    train, test = mf.split_temporal(prep, anos_treino, ano_teste)

    # 3) Baselines
    log.info("rodando baselines...")
    preds = rodar_baselines(train, test)

    # 4) LightGBM
    log.info("treinando LightGBM...")
    model, y_pred_lgbm = tr.fit_predict(train, test, val_fraction=args.val_fraction)
    preds["LightGBM_v1"] = y_pred_lgbm

    # 4.5) Calibração isotônica (opt-in)
    calibrator: cal.IsotonicCalibrator | None = None
    y_pred_iso: np.ndarray | None = None
    calib_iso: pd.DataFrame | None = None
    por_partido_iso: pd.DataFrame | None = None
    df_calib: pd.DataFrame | None = None
    if args.calibrate:
        # min_pred=0 vira None internamente (= calibra todos)
        min_pred = args.calib_min_pred if args.calib_min_pred > 0 else None
        if args.calib_mode == "holdout":
            ano_calib = args.calib_ano if args.calib_ano is not None else anos_treino[-1]
            log.info("calibração isotônica via holdout (ano_calib=%d, min_pred=%s)...",
                     ano_calib, min_pred)
            calibrator, df_calib = cal.treinar_calibrador_holdout(
                train, ano_calib=ano_calib, anos_treino=anos_treino,
                ano_col=ANO_COL_PRES, min_pred=min_pred,
            )
        else:  # oof
            log.info("calibração isotônica via OOF leave-one-year-out "
                     "(n_folds=%d, min_pred=%s)...",
                     len(anos_treino), min_pred)
            calibrator, df_calib = cal.treinar_calibrador_oof(
                train, anos_treino, ano_col=ANO_COL_PRES, min_pred=min_pred,
            )
        y_pred_iso = calibrator.predict(y_pred_lgbm)
        preds["LightGBM_v1_iso"] = y_pred_iso
        log.info(
            "isotonic: pred raw range=[%.4f, %.4f] -> iso range=[%.4f, %.4f]",
            float(np.min(y_pred_lgbm)), float(np.max(y_pred_lgbm)),
            float(np.min(y_pred_iso)), float(np.max(y_pred_iso)),
        )
        n_passou_raw = int((y_pred_lgbm < (min_pred or 0.0)).sum()) if min_pred else 0
        if n_passou_raw:
            log.info("isotonic: %d/%d predições passaram raw (pred < %s)",
                     n_passou_raw, len(y_pred_lgbm), min_pred)

    # 4.6) Conformal prediction (opt-in)
    split_conf: cf.SplitConformal | None = None
    mondrian_conf: cf.MondrianConformal | None = None
    mondrian_cat_conf: cf.MondrianCategorical | None = None
    pred_lo_split: np.ndarray | None = None
    pred_hi_split: np.ndarray | None = None
    pred_lo_mondrian: np.ndarray | None = None
    pred_hi_mondrian: np.ndarray | None = None
    pred_lo_mondrian_cat: np.ndarray | None = None
    pred_hi_mondrian_cat: np.ndarray | None = None
    cobertura_split: float | None = None
    cobertura_mondrian: float | None = None
    cobertura_mondrian_cat: float | None = None
    cobertura_decil_split: pd.DataFrame | None = None
    cobertura_decil_mondrian: pd.DataFrame | None = None
    cobertura_decil_mondrian_cat: pd.DataFrame | None = None
    cobertura_categoria_mondrian_cat: pd.DataFrame | None = None
    mondrian_cat_strata_cols: list[str] = []

    if args.conformal:
        # Reusar (se possível) o df de calibração já gerado pelo --calibrate.
        # Caso contrário, gerar um holdout dedicado com a mesma estratégia.
        if args.calibrate:
            log.info("conformal: reusando conjunto de calibração do --calibrate")
            col_pred_calib = (
                "y_pred_holdout" if args.calib_mode == "holdout" else "y_pred_oof"
            )
            pred_calib_raw = df_calib[col_pred_calib].to_numpy()
            y_true_calib = df_calib["y_true"].to_numpy()
            pred_calib_final = (
                calibrator.predict(pred_calib_raw) if calibrator is not None
                else pred_calib_raw
            )
        else:
            ano_calib_cf = (
                args.calib_ano if args.calib_ano is not None else anos_treino[-1]
            )
            log.info("conformal: gerando holdout dedicado (ano_calib=%d)", ano_calib_cf)
            df_calib = cal.holdout_predictions_um_ano(
                train, ano_calib=ano_calib_cf, anos_treino=anos_treino,
                ano_col=ANO_COL_PRES,
            )
            pred_calib_final = df_calib["y_pred_holdout"].to_numpy()
            y_true_calib = df_calib["y_true"].to_numpy()

        residuos_abs = cf.compute_residuals(y_true_calib, pred_calib_final)

        # Predição final no teste pra qual computamos o IC
        y_pred_pontual = y_pred_iso if y_pred_iso is not None else y_pred_lgbm

        # Split conformal — sempre
        log.info("conformal: ajustando SplitConformal (alpha=%.3f, n=%d)",
                 args.conformal_alpha, len(residuos_abs))
        split_conf = cf.SplitConformal(alpha=args.conformal_alpha).fit(residuos_abs)
        pred_lo_split, pred_hi_split = split_conf.predict_interval(y_pred_pontual)
        cobertura_split = cf.coverage_observed(
            test.y.values, pred_lo_split, pred_hi_split,
        )
        log.info(
            "SplitConformal: q_hat=%.4f, cobertura observada (test)=%.3f",
            split_conf.q_hat, cobertura_split,
        )
        cobertura_decil_split = cf.coverage_por_decil(
            test.y.values, y_pred_pontual, pred_lo_split, pred_hi_split,
            n_quantis=10,
        )

        # Mondrian conformal — opcional
        if args.conformal_mondrian:
            log.info("conformal: ajustando MondrianConformal (n_bins=%d, min_q_factor=%.2f)",
                     args.conformal_bins, args.conformal_min_q_factor)
            mondrian_conf = cf.MondrianConformal(
                alpha=args.conformal_alpha,
                n_bins=args.conformal_bins,
                min_per_bin=10,
                min_q_factor=args.conformal_min_q_factor,
            ).fit(pred_calib_final, residuos_abs)
            pred_lo_mondrian, pred_hi_mondrian = mondrian_conf.predict_interval(
                y_pred_pontual,
            )
            cobertura_mondrian = cf.coverage_observed(
                test.y.values, pred_lo_mondrian, pred_hi_mondrian,
            )
            log.info(
                "MondrianConformal: cobertura observada (test)=%.3f, fallback bins=%s",
                cobertura_mondrian, mondrian_conf.bins_fallback,
            )
            cobertura_decil_mondrian = cf.coverage_por_decil(
                test.y.values, y_pred_pontual, pred_lo_mondrian, pred_hi_mondrian,
                n_quantis=10,
            )

        # MondrianCategorical — opcional (estratificação por sigla/regiao)
        if args.conformal_mondrian_cat:
            mondrian_cat_strata_cols = [
                c.strip() for c in args.conformal_mondrian_cat_cols.split(",")
                if c.strip()
            ]
            log.info(
                "conformal: ajustando MondrianCategorical (cols=%s, "
                "min_per_stratum=%d)",
                mondrian_cat_strata_cols,
                args.conformal_mondrian_cat_min_per_stratum,
            )
            strata_calib = _construir_estrato(
                df_calib=df_calib, prep=train,
                strata_cols=mondrian_cat_strata_cols,
            )
            strata_test = _construir_estrato_test(
                test=test, strata_cols=mondrian_cat_strata_cols,
            )
            mondrian_cat_conf = cf.MondrianCategorical(
                alpha=args.conformal_alpha,
                min_per_stratum=args.conformal_mondrian_cat_min_per_stratum,
                min_q_factor=args.conformal_min_q_factor,
            ).fit(strata_calib, residuos_abs)
            pred_lo_mondrian_cat, pred_hi_mondrian_cat = (
                mondrian_cat_conf.predict_interval(y_pred_pontual, strata_test)
            )
            cobertura_mondrian_cat = cf.coverage_observed(
                test.y.values, pred_lo_mondrian_cat, pred_hi_mondrian_cat,
            )
            log.info(
                "MondrianCategorical: cobertura observada (test)=%.3f, "
                "n_estratos=%d, fallback=%d, floored=%d",
                cobertura_mondrian_cat,
                len(mondrian_cat_conf.q_per_stratum),
                len(mondrian_cat_conf.strata_fallback),
                len(mondrian_cat_conf.strata_floored),
            )
            cobertura_decil_mondrian_cat = cf.coverage_por_decil(
                test.y.values, y_pred_pontual,
                pred_lo_mondrian_cat, pred_hi_mondrian_cat,
                n_quantis=10,
            )
            cobertura_categoria_mondrian_cat = cf.coverage_por_categoria(
                test.y.values, pred_lo_mondrian_cat, pred_hi_mondrian_cat,
                strata_test,
            )

    # 4.7) CQR — Conformalized Quantile Regression (opt-in)
    cqr_obj: cqr_mod.CQR | None = None
    pred_lo_cqr: np.ndarray | None = None
    pred_hi_cqr: np.ndarray | None = None
    cobertura_cqr: float | None = None
    cobertura_decil_cqr: pd.DataFrame | None = None
    if args.cqr:
        if not args.conformal:
            raise SystemExit("--cqr requer --conformal (reusa o conjunto de calibração)")

        from src.models.transforms import sigmoid_logit  # local import: opt-in

        alpha_cqr = args.conformal_alpha
        a_low = alpha_cqr / 2.0
        a_hi = 1.0 - alpha_cqr / 2.0

        # Treina os 2 LGBMs quantile FINAIS (em todo o treino) — gerarão
        # os intervalos no test.
        log.info("CQR: treinando LGBMs quantile finais (alpha_low=%.3f, alpha_hi=%.3f)",
                 a_low, a_hi)
        m_low_final = tr.treinar_lgbm(
            train.X, train.y, train.cat_features,
            overrides={"objective": "quantile", "alpha": a_low, "metric": "quantile"},
            early_stopping_rounds=None,
        )
        m_hi_final = tr.treinar_lgbm(
            train.X, train.y, train.cat_features,
            overrides={"objective": "quantile", "alpha": a_hi, "metric": "quantile"},
            early_stopping_rounds=None,
        )
        q_low_test = sigmoid_logit(m_low_final.predict(test.X))
        q_hi_test = sigmoid_logit(m_hi_final.predict(test.X))
        # Modelos quantile podem cruzar (low > hi pontualmente) — garantir ordem.
        q_low_test, q_hi_test = (
            np.minimum(q_low_test, q_hi_test),
            np.maximum(q_low_test, q_hi_test),
        )

        # Para o CALIB set: precisa de quantis vindos de modelos que NÃO viram
        # essas linhas. Usa o mesmo split holdout que o calibrator/conformal usa.
        ano_calib_cqr = (
            args.calib_ano if args.calib_ano is not None else anos_treino[-1]
        )
        anos_treino_holdout = [a for a in anos_treino if a != ano_calib_cqr]
        log.info("CQR: treinando LGBMs quantile holdout (anos=%s, calib=%d)",
                 anos_treino_holdout, ano_calib_cqr)
        train_holdout, calib_holdout = mf.split_temporal(
            prep, anos_treino_holdout, ano_calib_cqr,
        )
        m_low_h = tr.treinar_lgbm(
            train_holdout.X, train_holdout.y, train_holdout.cat_features,
            overrides={"objective": "quantile", "alpha": a_low, "metric": "quantile"},
            early_stopping_rounds=None,
        )
        m_hi_h = tr.treinar_lgbm(
            train_holdout.X, train_holdout.y, train_holdout.cat_features,
            overrides={"objective": "quantile", "alpha": a_hi, "metric": "quantile"},
            early_stopping_rounds=None,
        )
        q_low_calib = sigmoid_logit(m_low_h.predict(calib_holdout.X))
        q_hi_calib = sigmoid_logit(m_hi_h.predict(calib_holdout.X))
        q_low_calib, q_hi_calib = (
            np.minimum(q_low_calib, q_hi_calib),
            np.maximum(q_low_calib, q_hi_calib),
        )
        y_calib_cqr = calib_holdout.y.to_numpy()

        cqr_obj = cqr_mod.CQR(alpha=alpha_cqr).fit(q_low_calib, q_hi_calib, y_calib_cqr)
        pred_lo_cqr, pred_hi_cqr = cqr_obj.predict_interval(q_low_test, q_hi_test)
        cobertura_cqr = cf.coverage_observed(
            test.y.values, pred_lo_cqr, pred_hi_cqr,
        )
        log.info(
            "CQR: q_hat=%+.4f, cobertura observada (test)=%.3f",
            cqr_obj.q_hat, cobertura_cqr,
        )
        cobertura_decil_cqr = cf.coverage_por_decil(
            test.y.values, y_pred_pontual, pred_lo_cqr, pred_hi_cqr,
            n_quantis=10,
        )

    # 5) Métricas
    comp = ev.tabela_comparativa(test.y.values, preds)
    log.info("comparativo geral:\n%s", comp.to_string(index=False))

    por_partido = ev.metricas_por_grupo(
        test.y.values, y_pred_lgbm, test.meta["sigla_partido"], min_n=5,
    )
    por_uf = ev.metricas_por_grupo(
        test.y.values, y_pred_lgbm, test.meta["sigla_uf"], min_n=5,
    )
    calib = ev.calibracao_por_decil(test.y.values, y_pred_lgbm, n_quantis=10)
    imp = ev.top_feature_importance(model, top_k=15, importance_type="gain")

    if y_pred_iso is not None:
        calib_iso = ev.calibracao_por_decil(test.y.values, y_pred_iso, n_quantis=10)
        por_partido_iso = ev.metricas_por_grupo(
            test.y.values, y_pred_iso, test.meta["sigla_partido"], min_n=5,
        )
        log.info("calibracao_por_decil (iso):\n%s",
                 calib_iso.to_string(index=False))

    # 6) Salva predições + modelo
    preds_df = test.meta.copy()
    preds_df["y_true"] = test.y.values
    for nome, yp in preds.items():
        preds_df[f"pred_{nome}"] = yp
    if pred_lo_split is not None:
        preds_df["pred_lower"] = pred_lo_split
        preds_df["pred_upper"] = pred_hi_split
    if pred_lo_mondrian is not None:
        preds_df["pred_lower_mondrian"] = pred_lo_mondrian
        preds_df["pred_upper_mondrian"] = pred_hi_mondrian
    if pred_lo_cqr is not None:
        preds_df["pred_lower_cqr"] = pred_lo_cqr
        preds_df["pred_upper_cqr"] = pred_hi_cqr
    if pred_lo_mondrian_cat is not None:
        preds_df["pred_lower_mondrian_cat"] = pred_lo_mondrian_cat
        preds_df["pred_upper_mondrian_cat"] = pred_hi_mondrian_cat
    fio.save_processed(preds_df, "preds")

    if not args.no_save_model:
        PATHS["models"].mkdir(parents=True, exist_ok=True)
        model_path = PATHS["models"] / "lgbm_v1.pkl"
        with open(model_path, "wb") as f:
            pickle.dump({
                "model": model,
                "feature_names": model.feature_name(),
                "cat_features": train.cat_features,
                "params": tr.params_lgbm(),
                "anos_treino": anos_treino,
                "ano_teste": ano_teste,
                "ano_col": ANO_COL_PRES,
                "calibrator": calibrator,            # None se --calibrate não foi passado
                "split_conformal": split_conf,       # None se --conformal não foi passado
                "mondrian_conformal": mondrian_conf,  # None se --conformal-mondrian não foi passado
                "mondrian_categorical": mondrian_cat_conf,  # None se --conformal-mondrian-cat não foi passado
                "mondrian_categorical_strata_cols": (
                    mondrian_cat_strata_cols if mondrian_cat_conf is not None else None
                ),
                "cqr": cqr_obj,                       # None se --cqr não foi passado
            }, f)
        log.info("modelo salvo em %s", model_path)

    # 7) Relatório
    calib_ano_resolved = (
        (args.calib_ano if args.calib_ano is not None else anos_treino[-1])
        if args.calibrate and args.calib_mode == "holdout" else None
    )
    md = gerar_relatorio(
        comp=comp, por_partido=por_partido, por_uf=por_uf,
        calib=calib, imp=imp,
        ano_teste=ano_teste, anos_treino=anos_treino,
        n_train=len(train), n_test=len(test),
        calib_iso=calib_iso, por_partido_iso=por_partido_iso,
        calib_mode=args.calib_mode if args.calibrate else None,
        calib_min_pred=args.calib_min_pred if args.calibrate else None,
        calib_ano=calib_ano_resolved,
        conformal_alpha=args.conformal_alpha if args.conformal else None,
        cobertura_split=cobertura_split,
        cobertura_decil_split=cobertura_decil_split,
        q_hat_split=split_conf.q_hat if split_conf is not None else None,
        cobertura_mondrian=cobertura_mondrian,
        cobertura_decil_mondrian=cobertura_decil_mondrian,
        mondrian_q_per_bin=(
            mondrian_conf.q_per_bin.tolist() if mondrian_conf is not None else None
        ),
        cobertura_cqr=cobertura_cqr,
        cobertura_decil_cqr=cobertura_decil_cqr,
        q_hat_cqr=cqr_obj.q_hat if cqr_obj is not None else None,
        cobertura_mondrian_cat=cobertura_mondrian_cat,
        cobertura_decil_mondrian_cat=cobertura_decil_mondrian_cat,
        cobertura_categoria_mondrian_cat=cobertura_categoria_mondrian_cat,
        mondrian_cat_strata_cols=(
            mondrian_cat_strata_cols if mondrian_cat_conf is not None else None
        ),
        mondrian_cat_n_estratos=(
            len(mondrian_cat_conf.q_per_stratum)
            if mondrian_cat_conf is not None else None
        ),
        mondrian_cat_n_fallback=(
            len(mondrian_cat_conf.strata_fallback)
            if mondrian_cat_conf is not None else None
        ),
    )
    report_path = PATHS["reports"] / "status_fase_4.md"
    report_path.write_text(md, encoding="utf-8")
    log.info("relatório salvo em %s", report_path)

    log.info("Fase 4 OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
