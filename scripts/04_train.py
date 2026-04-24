#!/usr/bin/env python
"""
scripts/04_train.py — Fase 4: primeiro modelo presidencial.

Carrega `data/processed/features.parquet`, roda split temporal
(treino=2014+2018, teste=2022), ajusta baselines (B0/B1/B2) e LightGBM,
salva predições em `data/processed/preds.parquet`, o modelo em
`models/lgbm_v1.pkl` e gera `reports/status_fase_4.md`.

Uso:
    python scripts/04_train.py [--log-level INFO] [--no-save-model]
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

from src.config import MODE_CFG, PATHS, set_global_seed, summary  # noqa: E402
from src.features import io as fio  # noqa: E402
from src.models import baseline as bl  # noqa: E402
from src.models import evaluate as ev  # noqa: E402
from src.models import features as mf  # noqa: E402
from src.models import train as tr  # noqa: E402


log = logging.getLogger("04_train")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fase 4 — treino e avaliação")
    p.add_argument("--log-level", type=str, default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--no-save-model", action="store_true",
                   help="não escreve models/lgbm_v1.pkl")
    p.add_argument("--val-fraction", type=float, default=0.0,
                   help="fração do treino pra validação interna (early stopping). "
                        "0 desliga. Padrão: 0.")
    return p.parse_args()


def configurar_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        level=getattr(logging, level),
    )


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
) -> str:
    """Monta o markdown do status_fase_4.md."""
    linhas = [
        "# Fase 4 — status: primeiro modelo presidencial",
        "",
        f"**Modo:** dev | **UFs:** {MODE_CFG['ufs']} | "
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
        "",
        "## Próximos passos (Fase 4.5 / Fase 5)",
        "",
        "- **Fase 4.5**: replicar pipeline pra prefeito (target municipal, eixo temporal 2012/2016/2020).",
        "- **Fase 5**: quantificação de incerteza (conformal prediction ou bootstrap). "
        "Fase 4 entrega só estimativa pontual; o produto final precisa de intervalo.",
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

    # 6) Salva predições + modelo
    preds_df = test.meta.copy()
    preds_df["y_true"] = test.y.values
    for nome, yp in preds.items():
        preds_df[f"pred_{nome}"] = yp
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
            }, f)
        log.info("modelo salvo em %s", model_path)

    # 7) Relatório
    md = gerar_relatorio(
        comp=comp, por_partido=por_partido, por_uf=por_uf,
        calib=calib, imp=imp,
        ano_teste=ano_teste, anos_treino=anos_treino,
        n_train=len(train), n_test=len(test),
    )
    report_path = PATHS["reports"] / "status_fase_4.md"
    report_path.write_text(md, encoding="utf-8")
    log.info("relatório salvo em %s", report_path)

    log.info("Fase 4 OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
