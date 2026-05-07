"""
src.models.cqr — Conformalized Quantile Regression.

Diferença pro `SplitConformal`/`MondrianConformal` (em src.models.conformal):

    Split / Mondrian: intervalos da forma [ŷ - q̂, ŷ + q̂] (simétrico,
    homocedástico ou estratificado por bin de pred). Quando a dispersão
    do erro varia muito com x (heterocedasticidade — caso PL 2022 no
    presidencial), o intervalo fica grande demais nos casos fáceis e
    pequeno demais nos difíceis.

    CQR: usa **dois modelos quantile** já treinados — q_low(x) ≈ quantil
    α/2 condicional, q_hi(x) ≈ quantil 1-α/2 condicional. O intervalo
    base é [q_low(x), q_hi(x)]. Conformaliza calculando uma margem q̂
    sobre os resíduos conformais E_i:

        E_i = max(q_low(x_i) - y_i, y_i - q_hi(x_i))

    E_i > 0 quando y_i caiu fora do intervalo base; E_i < 0 com folga
    (mede o quanto o intervalo está "errado pra menos" ou "sobra").

    q̂_cqr = quantil corrigido de amostra finita das E_i (mesmo nível
    ⌈(n+1)(1-α)⌉/n usado no split).

    Intervalo final: [q_low(x) - q̂_cqr, q_hi(x) + q̂_cqr]. Cobre 1-α
    marginalmente (mesma garantia do split conformal), mas a forma do
    intervalo herda a heterocedasticidade modelada pelos quantile
    regressors.

Referência: Romano, Patterson, Candès (2019) — "Conformalized Quantile
Regression". https://arxiv.org/abs/1905.03222

API:

    cqr = CQR(alpha=0.1).fit(q_low_calib, q_hi_calib, y_calib)
    lo, hi = cqr.predict_interval(q_low_test, q_hi_test)

Uso típico em pipeline com LGBM:

    1. Treina LGBM principal (objective=regression_l1) em logit(y).
    2. Treina LGBM_low (objective=quantile, alpha=α/2) em logit(y).
    3. Treina LGBM_hi (objective=quantile, alpha=1-α/2) em logit(y).
    4. No calib set: q_low = sigmoid(LGBM_low.predict(X_calib)), idem hi.
    5. cqr.fit(q_low, q_hi, y_calib).
    6. No test: idem, depois cqr.predict_interval(q_low_test, q_hi_test).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# Reusa a correção de amostra finita do split conformal — mesma estatística.
from src.models.conformal import _finite_sample_quantile_level  # noqa: E402


class CQR:
    """Conformalized Quantile Regression.

    Args:
        alpha: nível de erro alvo (1-α = cobertura nominal). Default 0.1.

    Attrs:
        q_hat: margem conformal. Pode ser negativa quando os modelos
            quantile já são conservadores (intervalo base cobre mais
            que 1-α — CQR aperta).
        n_calib: tamanho do conjunto de calibração após drop NaN.
    """

    nome: str = "cqr"

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha fora de (0,1): {alpha}")
        self.alpha = float(alpha)
        self.q_hat: float | None = None
        self.n_calib: int = 0

    @staticmethod
    def _resid(
        q_low: np.ndarray, q_hi: np.ndarray, y: np.ndarray
    ) -> np.ndarray:
        """Resíduo CQR: max(q_low - y, y - q_hi). Positivo = fora do intervalo."""
        return np.maximum(q_low - y, y - q_hi)

    def fit(
        self,
        q_low_calib: np.ndarray | pd.Series,
        q_hi_calib: np.ndarray | pd.Series,
        y_calib: np.ndarray | pd.Series,
    ) -> "CQR":
        """Ajusta a margem conformal sobre o conjunto de calibração."""
        ql = np.asarray(q_low_calib, dtype="float64").ravel()
        qh = np.asarray(q_hi_calib, dtype="float64").ravel()
        y = np.asarray(y_calib, dtype="float64").ravel()
        if not (ql.shape == qh.shape == y.shape):
            raise ValueError(
                f"shapes desiguais: q_low={ql.shape}, q_hi={qh.shape}, y={y.shape}"
            )
        if np.any(qh < ql):
            n_inv = int((qh < ql).sum())
            raise ValueError(
                f"q_hi < q_low em {n_inv} pontos — passe os modelos quantile "
                "na ordem certa (low primeiro, hi depois)."
            )

        mask = ~(np.isnan(ql) | np.isnan(qh) | np.isnan(y))
        ql, qh, y = ql[mask], qh[mask], y[mask]
        n = len(y)
        if n < 5:
            raise ValueError(
                f"poucos pontos válidos pra calibrar CQR: {n} (mínimo 5)"
            )

        E = self._resid(ql, qh, y)
        level = _finite_sample_quantile_level(n, self.alpha)
        # method='higher' alinhado com a prova conformal (split).
        q_hat = float(np.quantile(E, level, method="higher"))

        self.q_hat = q_hat
        self.n_calib = int(n)
        # Cobertura empírica do intervalo base (sem conformalização) —
        # informativo: se já está acima de 1-α, q_hat será negativo
        # (CQR aperta o intervalo).
        cob_base = float(((y >= ql) & (y <= qh)).mean())
        logger.info(
            "CQR.fit: n=%d, alpha=%.3f, q_hat=%+.4f, "
            "cob_intervalo_base=%.3f (nominal=%.2f)",
            n, self.alpha, q_hat, cob_base, 1.0 - self.alpha,
        )
        return self

    def predict_interval(
        self,
        q_low: np.ndarray | pd.Series,
        q_hi: np.ndarray | pd.Series,
        *,
        clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo conformalizado: [q_low - q̂, q_hi + q̂]."""
        if self.q_hat is None:
            raise RuntimeError("CQR não foi ajustado — chame .fit antes")
        ql = np.asarray(q_low, dtype="float64").ravel()
        qh = np.asarray(q_hi, dtype="float64").ravel()
        if ql.shape != qh.shape:
            raise ValueError(f"shapes desiguais: q_low={ql.shape}, q_hi={qh.shape}")
        lo = ql - self.q_hat
        hi = qh + self.q_hat
        if clip:
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        return lo, hi


__all__ = ["CQR"]
