"""
src.models.transforms — transformações do target `share_1t`.

O target é uma proporção em [0, 1] (voto do candidato ÷ votos válidos no
município). Modelar diretamente a proporção com regressão quadrática tem
dois problemas:

  1. Predições ficam livres pra sair de [0, 1] — numericamente possível,
     politicamente sem sentido.
  2. A variância do share não é uniforme: partidos pequenos vivem em
     [0, 0.05], partidos grandes em [0.2, 0.6]. Erro absoluto de 0.02
     é muito grande pro PSOL e desprezível pro PT.

Solução padrão: logit-transform. Modelamos `logit(share)` ∈ (-∞, +∞),
o que (a) impede extrapolação inválida após sigmoid e (b) equaliza a
sensibilidade multiplicativamente em vez de aditivamente.

Clipping: `share = 0` e `share = 1` vão pra ±∞ após logit. Clipamos em
[eps, 1-eps] antes de transformar. EPS_DEFAULT = 1e-4 é suficiente: na
escala logit equivale a ±9.2 (longe da zona de treino típica em [-6, +1]).

Funções:
    logit_share(share, eps=EPS_DEFAULT) -> logit
    sigmoid_logit(logit) -> share
    clip_share(share, eps=EPS_DEFAULT) -> share clipado

logit_share e sigmoid_logit aceitam escalar, np.ndarray ou pd.Series.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS_DEFAULT: float = 1e-4


def clip_share(
    share: float | np.ndarray | pd.Series,
    eps: float = EPS_DEFAULT,
) -> np.ndarray | pd.Series | float:
    """Clipa valores em [eps, 1-eps]. Preserva o tipo de entrada."""
    if not 0 < eps < 0.5:
        raise ValueError(f"eps fora de (0, 0.5): {eps}")
    if isinstance(share, pd.Series):
        return share.clip(lower=eps, upper=1.0 - eps)
    arr = np.asarray(share, dtype="float64")
    clipped = np.clip(arr, eps, 1.0 - eps)
    return clipped if arr.ndim else float(clipped)


def logit_share(
    share: float | np.ndarray | pd.Series,
    eps: float = EPS_DEFAULT,
) -> np.ndarray | pd.Series | float:
    """log(share / (1 - share)) após clip. NaN propaga como NaN."""
    if isinstance(share, pd.Series):
        clipped = clip_share(share, eps=eps)
        out = np.log(clipped / (1.0 - clipped))
        return out
    arr = np.asarray(share, dtype="float64")
    clipped = np.clip(arr, eps, 1.0 - eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.log(clipped / (1.0 - clipped))
    # Propaga NaN onde o input era NaN
    if arr.ndim:
        out = np.where(np.isnan(arr), np.nan, out)
        return out
    if np.isnan(arr):
        return float("nan")
    return float(out)


def sigmoid_logit(
    logit: float | np.ndarray | pd.Series,
) -> np.ndarray | pd.Series | float:
    """1 / (1 + exp(-logit)). Estável numericamente pra |x| grande."""
    if isinstance(logit, pd.Series):
        return pd.Series(sigmoid_logit(logit.values), index=logit.index)
    arr = np.asarray(logit, dtype="float64")
    # Implementação estável: dois ramos
    out = np.empty_like(arr)
    pos = arr >= 0
    neg = ~pos
    out[pos] = 1.0 / (1.0 + np.exp(-arr[pos]))
    ez = np.exp(arr[neg])
    out[neg] = ez / (1.0 + ez)
    # NaN propaga
    out = np.where(np.isnan(arr), np.nan, out)
    if arr.ndim == 0:
        return float(out)
    return out


__all__ = ["EPS_DEFAULT", "clip_share", "logit_share", "sigmoid_logit"]
