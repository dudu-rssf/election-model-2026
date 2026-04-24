"""
src.models.evaluate — métricas e diagnósticos de modelos.

Métricas na escala original do share_1t (∈ [0,1]), não em logit. Dois
motivos: (1) MAE de 0.02 em logit é difícil de interpretar; 0.02 em share
é "errei 2 pontos percentuais"; (2) comparação justa entre baselines
(que vivem em share) e LightGBM (que vive em logit internamente).

API central:

    metricas_gerais(y_true, y_pred) -> dict
    metricas_por_grupo(y_true, y_pred, grupo: Series, k: int = 10) -> DataFrame
    tabela_comparativa(preds: dict[str, np.ndarray], y_true) -> DataFrame
    calibracao_por_decil(y_true, y_pred, n_quantis: int = 10) -> DataFrame

Nenhuma função plota — Fase 4 é só tabelas em Markdown. Plots entram em
Fase 5 (quando já houver modelo + ajustes).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Métricas simples
# ------------------------------------------------------------
def _mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Erro médio com sinal: positivo = modelo subestima."""
    return float(np.mean(y_true - y_pred))


def metricas_gerais(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    yt = np.asarray(y_true, dtype="float64")
    yp = np.asarray(y_pred, dtype="float64")
    mask = ~(np.isnan(yt) | np.isnan(yp))
    if not mask.any():
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "bias": float("nan")}
    yt = yt[mask]
    yp = yp[mask]
    return {
        "n": int(mask.sum()),
        "mae": _mae(yt, yp),
        "rmse": _rmse(yt, yp),
        "bias": _bias(yt, yp),
    }


# ------------------------------------------------------------
# Breakdown por grupo (partido, UF, etc.)
# ------------------------------------------------------------
def metricas_por_grupo(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    grupo: pd.Series,
    min_n: int = 5,
) -> pd.DataFrame:
    """Métricas estratificadas por `grupo` (ex: sigla_partido, sigla_uf).

    Filtra grupos com menos de `min_n` obs (ruído demais).
    """
    df = pd.DataFrame({
        "grupo": grupo.astype("string").values,
        "y_true": np.asarray(y_true, dtype="float64"),
        "y_pred": np.asarray(y_pred, dtype="float64"),
    })
    df = df.dropna(subset=["y_true", "y_pred"])

    rows = []
    for g, sub in df.groupby("grupo", observed=True):
        if len(sub) < min_n:
            continue
        yt = sub["y_true"].values
        yp = sub["y_pred"].values
        rows.append({
            "grupo": g,
            "n": int(len(sub)),
            "mae": _mae(yt, yp),
            "rmse": _rmse(yt, yp),
            "bias": _bias(yt, yp),
            "share_medio": float(yt.mean()),
        })
    out = pd.DataFrame(rows)
    if len(out) == 0:
        return out
    return out.sort_values("mae", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------
# Comparação lado a lado de múltiplos modelos
# ------------------------------------------------------------
def tabela_comparativa(
    y_true: pd.Series | np.ndarray,
    preds: dict[str, np.ndarray],
) -> pd.DataFrame:
    """DataFrame: linhas = modelos, colunas = métricas."""
    yt_arr = np.asarray(y_true, dtype="float64")
    rows = []
    for nome, yp in preds.items():
        m = metricas_gerais(yt_arr, yp)
        rows.append({"modelo": nome, **m})
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# Calibração: decil predito vs realizado
# ------------------------------------------------------------
def calibracao_por_decil(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    n_quantis: int = 10,
) -> pd.DataFrame:
    """Bucketiza predições em decis e compara média prevista vs realizada.

    Modelo bem calibrado: pred_medio ≈ real_medio dentro de cada bucket.
    """
    df = pd.DataFrame({
        "y_true": np.asarray(y_true, dtype="float64"),
        "y_pred": np.asarray(y_pred, dtype="float64"),
    }).dropna()
    if len(df) == 0:
        return pd.DataFrame(
            columns=["decil", "n", "pred_medio", "real_medio", "erro_decil"]
        )

    try:
        df["decil"] = pd.qcut(df["y_pred"], q=n_quantis, duplicates="drop")
    except ValueError:
        df["decil"] = "all"

    agg = (
        df.groupby("decil", observed=True)
        .agg(n=("y_true", "size"),
             pred_medio=("y_pred", "mean"),
             real_medio=("y_true", "mean"))
        .reset_index()
    )
    agg["erro_decil"] = agg["real_medio"] - agg["pred_medio"]
    agg["decil"] = agg["decil"].astype("string")
    return agg


# ------------------------------------------------------------
# Feature importance (LightGBM)
# ------------------------------------------------------------
def top_feature_importance(
    model,
    top_k: int = 20,
    importance_type: str = "gain",
) -> pd.DataFrame:
    """Extrai feature importance do Booster. `importance_type` ∈ {gain, split}."""
    nomes = model.feature_name()
    vals = model.feature_importance(importance_type=importance_type)
    df = pd.DataFrame({"feature": nomes, "importance": vals})
    df = df.sort_values("importance", ascending=False).head(top_k).reset_index(drop=True)
    return df


__all__ = [
    "metricas_gerais",
    "metricas_por_grupo",
    "tabela_comparativa",
    "calibracao_por_decil",
    "top_feature_importance",
]
