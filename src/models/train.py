"""
src.models.train — pipeline de treino do LightGBM pro target presidencial.

Função central: `treinar_lgbm(X_train, y_train, X_val=None, y_val=None, ...)`
que retorna um `lightgbm.Booster` ajustado. Wrapper em cima da API nativa
pra padronizar:

  * target = logit(share), predição destransformada com sigmoid
  * objective = regression_l1 (MAE — mais robusto a outliers que MSE)
  * early stopping com validação opcional
  * categorical_feature vem da `PreparedData.cat_features`
  * seed fixa via src.config.SEED
  * hiperparâmetros de config.yaml → model.lgbm

Também expõe `prever(model, X)` que já aplica sigmoid e clipa em [0,1].
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.config import CONFIG, SEED
from src.models.features import PreparedData
from src.models.transforms import EPS_DEFAULT, logit_share, sigmoid_logit

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Hiperparâmetros (carregados do config.yaml)
# ------------------------------------------------------------
def params_lgbm(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Retorna dict de params pro LightGBM, partindo do config.yaml."""
    cfg = CONFIG.get("model", {}).get("lgbm", {})
    params: dict[str, Any] = {
        "objective": "regression_l1",
        "metric": "mae",
        "n_estimators": int(cfg.get("n_estimators", 500)),
        "learning_rate": float(cfg.get("learning_rate", 0.05)),
        "num_leaves": int(cfg.get("num_leaves", 63)),
        "min_data_in_leaf": int(cfg.get("min_data_in_leaf", 20)),
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "seed": SEED,
    }
    if overrides:
        params.update(overrides)
    return params


# ------------------------------------------------------------
# Treino
# ------------------------------------------------------------
def treinar_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cat_features: list[str],
    X_val: pd.DataFrame | None = None,
    y_val: pd.Series | None = None,
    eps: float = EPS_DEFAULT,
    overrides: dict[str, Any] | None = None,
    early_stopping_rounds: int | None = 50,
) -> "lightgbm.Booster":  # type: ignore[name-defined]
    """Treina LightGBM em logit(share). Retorna Booster.

    Args:
        X_train, y_train: features e target (share ∈ [0,1]).
        cat_features: lista de nomes das colunas categóricas.
        X_val, y_val: opcionais — ativa early stopping.
        eps: para logit transform.
        overrides: sobrescreve params padrão.
        early_stopping_rounds: rodadas sem melhoria em val antes de parar.
            Ignorado se X_val for None.

    Returns:
        lightgbm.Booster treinado.
    """
    import lightgbm as lgb

    params = params_lgbm(overrides)
    y_tr_logit = logit_share(y_train, eps=eps)

    # Categóricas via pandas Categorical + param explícito
    dtrain = lgb.Dataset(
        X_train,
        label=y_tr_logit,
        categorical_feature=cat_features,
        free_raw_data=False,
    )
    valid_sets = [dtrain]
    valid_names = ["train"]
    if X_val is not None and y_val is not None:
        y_va_logit = logit_share(y_val, eps=eps)
        dval = lgb.Dataset(
            X_val,
            label=y_va_logit,
            categorical_feature=cat_features,
            reference=dtrain,
            free_raw_data=False,
        )
        valid_sets.append(dval)
        valid_names.append("val")

    callbacks: list[Any] = [lgb.log_evaluation(period=0)]
    if X_val is not None and early_stopping_rounds:
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    n_est = params.pop("n_estimators")
    model = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=n_est,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )
    logger.info(
        "treinar_lgbm: best_iter=%s, n_features=%d",
        model.best_iteration, X_train.shape[1],
    )
    return model


# ------------------------------------------------------------
# Predição
# ------------------------------------------------------------
def prever(
    model: "lightgbm.Booster",  # type: ignore[name-defined]
    X: pd.DataFrame,
    clip_pred: bool = True,
) -> np.ndarray:
    """Prediz em share (já destransforma de logit). Opcionalmente clipa em [0,1]."""
    logit_pred = model.predict(X, num_iteration=model.best_iteration)
    share = sigmoid_logit(np.asarray(logit_pred, dtype="float64"))
    if clip_pred:
        share = np.clip(share, 0.0, 1.0)
    return share


# ------------------------------------------------------------
# Pipeline fit+predict (orquestra PreparedData train/test)
# ------------------------------------------------------------
def fit_predict(
    train: PreparedData,
    test: PreparedData,
    val_fraction: float = 0.0,
    overrides: dict[str, Any] | None = None,
) -> tuple["lightgbm.Booster", np.ndarray]:  # type: ignore[name-defined]
    """Fit no train, predict no test. Retorna (modelo, y_pred_test).

    Se `val_fraction > 0`, separa uma fração aleatória do train pra validação
    interna (early stopping). Não há split temporal no treino (só 2 anos em
    dev, 2014 e 2018 — com val_fraction=0 evitamos desperdiçar metade da
    amostra; 0.0 pula early stopping).

    A seed é fixa via `src.config.SEED`, então a divisão é reprodutível.
    """
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction fora de [0,1): {val_fraction}")

    X_val, y_val = None, None
    X_tr, y_tr = train.X, train.y
    if val_fraction > 0:
        rng = np.random.default_rng(SEED)
        idx = np.arange(len(X_tr))
        rng.shuffle(idx)
        n_val = max(1, int(round(val_fraction * len(idx))))
        val_idx = idx[:n_val]
        tr_idx = idx[n_val:]
        X_val = X_tr.iloc[val_idx].reset_index(drop=True)
        y_val = y_tr.iloc[val_idx].reset_index(drop=True)
        X_tr = X_tr.iloc[tr_idx].reset_index(drop=True)
        y_tr = y_tr.iloc[tr_idx].reset_index(drop=True)
        logger.info("fit_predict: train=%d / val=%d", len(X_tr), len(X_val))

    model = treinar_lgbm(
        X_tr, y_tr, cat_features=train.cat_features,
        X_val=X_val, y_val=y_val,
        overrides=overrides,
    )
    y_pred = prever(model, test.X)
    return model, y_pred


__all__ = [
    "params_lgbm",
    "treinar_lgbm",
    "prever",
    "fit_predict",
]
