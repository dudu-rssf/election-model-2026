"""
src.models.conformal — quantificação de incerteza via conformal prediction.

API:
    sc = SplitConformal(alpha=0.1).fit(residuals_abs)
    lo, hi = sc.predict_interval(y_pred)

    mc = MondrianConformal(alpha=0.1, n_bins=10).fit(pred_calib, residuals_abs)
    lo, hi = mc.predict_interval(y_pred)

    mcat = MondrianCategorical(alpha=0.1, min_per_stratum=10).fit(strata, residuals_abs)
    lo, hi = mcat.predict_interval(y_pred, strata)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================
def compute_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    absolute: bool = True,
) -> np.ndarray:
    y = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y_true={y.shape} vs y_pred={p.shape}")
    r = y - p
    return np.abs(r) if absolute else r


def coverage_observed(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
) -> float:
    y = np.asarray(y_true, dtype="float64").ravel()
    lo = np.asarray(lower, dtype="float64").ravel()
    hi = np.asarray(upper, dtype="float64").ravel()
    if not (y.shape == lo.shape == hi.shape):
        raise ValueError(
            f"shapes desiguais: y={y.shape}, lower={lo.shape}, upper={hi.shape}"
        )
    in_band = (y >= lo) & (y <= hi)
    return float(in_band.mean())


def _finite_sample_quantile_level(n: int, alpha: float) -> float:
    if n <= 0:
        raise ValueError(f"n_calib={n}: precisa de >=1 ponto")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha fora de (0,1): {alpha}")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    level = min(k / n, 1.0)
    return float(level)


def _quantile_residuos(residuos_abs: np.ndarray, alpha: float) -> float:
    r = np.asarray(residuos_abs, dtype="float64").ravel()
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        raise ValueError("residuos vazios após drop NaN")
    level = _finite_sample_quantile_level(n, alpha)
    return float(np.quantile(r, level, method="higher"))


# ============================================================
# SplitConformal
# ============================================================
class SplitConformal:
    nome: str = "split_conformal"

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha fora de (0,1): {alpha}")
        self.alpha = float(alpha)
        self.q_hat: float | None = None
        self.n_calib: int = 0

    def fit(self, residuos_abs: np.ndarray | pd.Series) -> "SplitConformal":
        r = np.asarray(residuos_abs, dtype="float64").ravel()
        r = r[~np.isnan(r)]
        if len(r) < 5:
            raise ValueError(f"poucos resíduos válidos: {len(r)} (mínimo 5)")
        if np.any(r < 0):
            raise ValueError("residuos_abs tem valores negativos — passe |r|")
        self.q_hat = _quantile_residuos(r, self.alpha)
        self.n_calib = int(len(r))
        logger.info("SplitConformal.fit: n=%d, alpha=%.3f, q_hat=%.4f",
                    self.n_calib, self.alpha, self.q_hat)
        return self

    def predict_interval(
        self, y_pred: np.ndarray | pd.Series, *, clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.q_hat is None:
            raise RuntimeError("SplitConformal não foi ajustado")
        p = np.asarray(y_pred, dtype="float64").ravel()
        lo = p - self.q_hat
        hi = p + self.q_hat
        if clip:
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        return lo, hi


# ============================================================
# MondrianConformal — bin de pred
# ============================================================
class MondrianConformal:
    nome: str = "mondrian_conformal"

    def __init__(
        self,
        alpha: float = 0.1,
        n_bins: int = 10,
        min_per_bin: int = 10,
        min_q_factor: float = 0.0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha fora de (0,1): {alpha}")
        if n_bins < 2:
            raise ValueError(f"n_bins>=2 (got {n_bins})")
        if min_per_bin < 1:
            raise ValueError(f"min_per_bin>=1 (got {min_per_bin})")
        if not 0.0 <= min_q_factor <= 1.0:
            raise ValueError(f"min_q_factor em [0,1] (got {min_q_factor})")
        self.alpha = float(alpha)
        self.n_bins = int(n_bins)
        self.min_per_bin = int(min_per_bin)
        self.min_q_factor = float(min_q_factor)
        self.bin_edges: np.ndarray | None = None
        self.q_per_bin: np.ndarray | None = None
        self.q_global: float | None = None
        self.bins_fallback: list[int] = []
        self.bins_floored: list[int] = []
        self.n_calib: int = 0

    def fit(
        self,
        pred_calib: np.ndarray | pd.Series,
        residuos_abs: np.ndarray | pd.Series,
    ) -> "MondrianConformal":
        p = np.asarray(pred_calib, dtype="float64").ravel()
        r = np.asarray(residuos_abs, dtype="float64").ravel()
        if p.shape != r.shape:
            raise ValueError(f"shapes não batem: pred={p.shape} vs residuos={r.shape}")
        mask = ~(np.isnan(p) | np.isnan(r))
        p, r = p[mask], r[mask]
        n = len(p)
        if n < self.n_bins * 2:
            raise ValueError(
                f"poucos pontos ({n}) pra n_bins={self.n_bins}; precisa "
                f">= {self.n_bins * 2}"
            )
        if np.any(r < 0):
            raise ValueError("residuos_abs tem valores negativos — passe |r|")

        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(p, qs)
        edges_u = np.unique(edges)
        if len(edges_u) < self.n_bins + 1:
            logger.warning(
                "MondrianConformal: pred_calib tem ties — bordas reduzidas "
                "de %d pra %d", self.n_bins + 1, len(edges_u),
            )
        edges_u[0] = -np.inf
        edges_u[-1] = np.inf
        self.bin_edges = edges_u
        n_bins_eff = len(edges_u) - 1

        q_global = _quantile_residuos(r, self.alpha)
        self.q_global = q_global

        bin_idx = np.digitize(p, edges_u[1:-1], right=False)

        q_per_bin = np.empty(n_bins_eff, dtype="float64")
        bins_fallback: list[int] = []
        for b in range(n_bins_eff):
            r_b = r[bin_idx == b]
            n_b = len(r_b)
            if n_b < self.min_per_bin:
                q_per_bin[b] = q_global
                bins_fallback.append(b)
                logger.info(
                    "MondrianConformal: bin %d com n=%d < %d -> fallback "
                    "global (q̂=%.4f)", b, n_b, self.min_per_bin, q_global,
                )
            else:
                q_per_bin[b] = _quantile_residuos(r_b, self.alpha)

        bins_floored: list[int] = []
        if self.min_q_factor > 0.0:
            floor = self.min_q_factor * q_global
            for b in range(n_bins_eff):
                if q_per_bin[b] < floor:
                    logger.info(
                        "MondrianConformal: bin %d q=%.4f < floor=%.4f -> "
                        "ajustado", b, q_per_bin[b], floor,
                    )
                    q_per_bin[b] = floor
                    bins_floored.append(b)

        self.q_per_bin = q_per_bin
        self.bins_fallback = bins_fallback
        self.bins_floored = bins_floored
        self.n_calib = int(n)
        logger.info(
            "MondrianConformal.fit: n=%d, alpha=%.3f, n_bins_eff=%d, "
            "q_global=%.4f, q_per_bin=[%s], min_q_factor=%.2f, n_floored=%d",
            n, self.alpha, n_bins_eff, q_global,
            ", ".join(f"{q:.3f}" for q in q_per_bin),
            self.min_q_factor, len(bins_floored),
        )
        return self

    def _bin_for(self, pred: np.ndarray) -> np.ndarray:
        assert self.bin_edges is not None
        return np.digitize(pred, self.bin_edges[1:-1], right=False)

    def predict_interval(
        self, y_pred: np.ndarray | pd.Series, *, clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.q_per_bin is None or self.bin_edges is None:
            raise RuntimeError("MondrianConformal não foi ajustado")
        p = np.asarray(y_pred, dtype="float64").ravel()
        bin_idx = self._bin_for(p)
        q = self.q_per_bin[bin_idx]
        lo = p - q
        hi = p + q
        if clip:
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        return lo, hi


# ============================================================
# MondrianCategorical — quantil por estrato categórico
# ============================================================
class MondrianCategorical:
    """Conformal estratificado por etiquetas categóricas (e.g.,
    `sigla_partido`, ou combinação `sigla|regiao`).

    Cada estrato `s` recebe q̂_s = quantil dos resíduos com etiqueta s.
    Estratos com `< min_per_stratum` pontos ou nunca vistos no calib
    caem pro q̂ global.
    """
    nome: str = "mondrian_categorical"

    def __init__(
        self,
        alpha: float = 0.1,
        min_per_stratum: int = 10,
        min_q_factor: float = 0.0,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha fora de (0,1): {alpha}")
        if min_per_stratum < 1:
            raise ValueError(f"min_per_stratum>=1 (got {min_per_stratum})")
        if not 0.0 <= min_q_factor <= 1.0:
            raise ValueError(f"min_q_factor em [0,1] (got {min_q_factor})")
        self.alpha = float(alpha)
        self.min_per_stratum = int(min_per_stratum)
        self.min_q_factor = float(min_q_factor)
        self.q_per_stratum: dict[str, float] = {}
        self.q_global: float | None = None
        self.strata_fallback: list[str] = []
        self.strata_floored: list[str] = []
        self.n_calib: int = 0

    def fit(
        self,
        strata: np.ndarray | pd.Series,
        residuos_abs: np.ndarray | pd.Series,
    ) -> "MondrianCategorical":
        s = pd.Series(strata).reset_index(drop=True).astype("string")
        r = np.asarray(residuos_abs, dtype="float64").ravel()
        if len(s) != len(r):
            raise ValueError(
                f"shapes não batem: strata={len(s)} vs residuos={len(r)}"
            )
        mask = (~s.isna()) & (~np.isnan(r))
        s = s[mask].reset_index(drop=True)
        r = r[mask.to_numpy()]
        n = len(s)
        if n < 5:
            raise ValueError(f"poucos resíduos válidos: {n} (mínimo 5)")
        if np.any(r < 0):
            raise ValueError("residuos_abs tem valores negativos — passe |r|")

        q_global = _quantile_residuos(r, self.alpha)
        self.q_global = q_global

        q_per_stratum: dict[str, float] = {}
        strata_fallback: list[str] = []
        for label in s.unique():
            mask_s = (s == label).to_numpy()
            r_s = r[mask_s]
            n_s = len(r_s)
            label_str = str(label)
            if n_s < self.min_per_stratum:
                q_per_stratum[label_str] = q_global
                strata_fallback.append(label_str)
                logger.info(
                    "MondrianCategorical: estrato %r n=%d < %d -> "
                    "fallback global (q̂=%.4f)",
                    label_str, n_s, self.min_per_stratum, q_global,
                )
            else:
                q_per_stratum[label_str] = _quantile_residuos(r_s, self.alpha)

        strata_floored: list[str] = []
        if self.min_q_factor > 0.0:
            floor = self.min_q_factor * q_global
            for label, q in list(q_per_stratum.items()):
                if q < floor:
                    logger.info(
                        "MondrianCategorical: estrato %r q=%.4f < floor=%.4f"
                        " -> ajustado", label, q, floor,
                    )
                    q_per_stratum[label] = floor
                    strata_floored.append(label)

        self.q_per_stratum = q_per_stratum
        self.strata_fallback = strata_fallback
        self.strata_floored = strata_floored
        self.n_calib = int(n)
        logger.info(
            "MondrianCategorical.fit: n=%d, alpha=%.3f, n_estratos=%d, "
            "q_global=%.4f, n_fallback=%d, n_floored=%d",
            n, self.alpha, len(q_per_stratum), q_global,
            len(strata_fallback), len(strata_floored),
        )
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray | pd.Series,
        strata: np.ndarray | pd.Series,
        *,
        clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.q_per_stratum or self.q_global is None:
            raise RuntimeError("MondrianCategorical não foi ajustado")
        p = np.asarray(y_pred, dtype="float64").ravel()
        s = pd.Series(strata).astype("string").reset_index(drop=True)
        if len(p) != len(s):
            raise ValueError(
                f"shapes não batem: y_pred={len(p)} vs strata={len(s)}"
            )
        q = s.map(self.q_per_stratum).fillna(self.q_global).to_numpy(
            dtype="float64"
        )
        lo = p - q
        hi = p + q
        if clip:
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        return lo, hi


# ============================================================
# Helpers
# ============================================================
def coverage_por_decil(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
    *,
    n_quantis: int = 10,
) -> pd.DataFrame:
    y = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    lo = np.asarray(lower, dtype="float64").ravel()
    hi = np.asarray(upper, dtype="float64").ravel()
    if not (y.shape == p.shape == lo.shape == hi.shape):
        raise ValueError("shapes desiguais entre y_true, y_pred, lower, upper")

    df = pd.DataFrame({"y": y, "pred": p, "lo": lo, "hi": hi})
    try:
        df["decil"] = pd.qcut(df["pred"], q=n_quantis, labels=False, duplicates="drop")
    except ValueError as e:
        raise ValueError(f"qcut falhou: {e}") from e

    rows = []
    for b, sub in df.groupby("decil", observed=True):
        rows.append({
            "decil": int(b),
            "n": int(len(sub)),
            "pred_min": float(sub["pred"].min()),
            "pred_max": float(sub["pred"].max()),
            "cobertura": float(((sub["y"] >= sub["lo"]) & (sub["y"] <= sub["hi"])).mean()),
        })
    return pd.DataFrame(rows).sort_values("decil").reset_index(drop=True)


def coverage_por_categoria(
    y_true: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
    strata: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """Cobertura empírica por estrato categórico."""
    y = np.asarray(y_true, dtype="float64").ravel()
    lo = np.asarray(lower, dtype="float64").ravel()
    hi = np.asarray(upper, dtype="float64").ravel()
    s = pd.Series(strata).astype("string").reset_index(drop=True)
    if not (y.shape == lo.shape == hi.shape == (len(s),)):
        raise ValueError("shapes desiguais entre y_true, lower, upper, strata")

    df = pd.DataFrame({"y": y, "lo": lo, "hi": hi, "estrato": s})
    rows = []
    for est, sub in df.groupby("estrato", observed=True):
        in_band = (sub["y"] >= sub["lo"]) & (sub["y"] <= sub["hi"])
        rows.append({
            "estrato": str(est),
            "n": int(len(sub)),
            "cobertura": float(in_band.mean()),
        })
    return pd.DataFrame(rows).sort_values(
        "cobertura", ascending=True
    ).reset_index(drop=True)


__all__ = [
    "SplitConformal",
    "MondrianConformal",
    "MondrianCategorical",
    "compute_residuals",
    "coverage_observed",
    "coverage_por_decil",
    "coverage_por_categoria",
]
