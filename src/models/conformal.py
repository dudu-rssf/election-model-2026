"""
src.models.conformal — quantificação de incerteza via conformal prediction.

Fase 5: queremos intervalos de confiança em torno da predição pontual do
LightGBM. A abordagem escolhida é **split conformal** sobre resíduos
absolutos:

    1. Treina LGBM em (anos_treino - {ano_calib}).
    2. Prediz no ano_calib (conjunto de calibração).
    3. Calcula resíduos absolutos r_i = |y_i - ŷ_i|.
    4. Quantil empírico q̂ = quantile(r_1..r_n, level) com correção de
       amostra finita: level = ⌈(n+1)(1-α)⌉ / n.
    5. Para cada predição nova ŷ, intervalo = [ŷ - q̂, ŷ + q̂], clipado
       em [0, 1].

Garantia (split conformal): cobertura marginal ≥ 1 - α, condicional ao
calibration set, sob exchangeability entre calibração e teste.

Duas variantes:

  * `SplitConformal` — quantil global. Mesmo q̂ para todas as predições.
                       Cobertura marginal correta, mas potencialmente
                       conservador no meio (resíduos pequenos) e
                       inadequado nas pontas (resíduos grandes).
  * `MondrianConformal` — quantil por bin de predição (decis por
                          default). Cobertura aproximadamente condicional
                          em ŷ. Bin com poucos pontos cai pro q̂ global.

Ambos reusam o mesmo conjunto de calibração que o IsotonicCalibrator
(`treinar_calibrador_holdout` ou `_oof`). Resíduos podem ser computados
sobre predições raw ou já calibradas — o que importa é que sejam
exchangeáveis com as predições do teste.

API:
    sc = SplitConformal(alpha=0.1).fit(residuals_abs)
    lo, hi = sc.predict_interval(y_pred)

    mc = MondrianConformal(alpha=0.1, n_bins=10).fit(pred_calib, residuals_abs)
    lo, hi = mc.predict_interval(y_pred)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def compute_residuals(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    *,
    absolute: bool = True,
) -> np.ndarray:
    """Resíduos = y_true - y_pred. Por default em valor absoluto."""
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
    """Fração de y_true que cai em [lower, upper] (inclusive)."""
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
    """Nível corrigido pra amostra finita: ⌈(n+1)(1-α)⌉ / n.

    Esse ajuste vem da prova do split conformal: a posição do quantil no
    conjunto de calibração precisa cobrir a amostra de teste com prob
    ≥ 1-α. Se ⌈(n+1)(1-α)⌉ > n, devolve 1.0 (não dá pra alcançar a
    cobertura desejada com tão poucos pontos — q̂ vira o máximo dos
    resíduos).
    """
    if n <= 0:
        raise ValueError(f"n_calib={n}: precisa de >=1 ponto")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha fora de (0,1): {alpha}")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    level = min(k / n, 1.0)
    return float(level)


def _quantile_residuos(residuos_abs: np.ndarray, alpha: float) -> float:
    """q̂ = quantil corrigido (amostra finita) dos resíduos absolutos."""
    r = np.asarray(residuos_abs, dtype="float64").ravel()
    r = r[~np.isnan(r)]
    n = len(r)
    if n == 0:
        raise ValueError("residuos vazios após drop NaN")
    level = _finite_sample_quantile_level(n, alpha)
    # method='higher' garante que pelo menos ⌈(n+1)(1-α)⌉ pontos ficam
    # abaixo de q̂ — alinhado com a prova do split conformal.
    q = float(np.quantile(r, level, method="higher"))
    return q


# ============================================================
# SplitConformal — quantil global
# ============================================================
class SplitConformal:
    """Conformal split com quantil global dos resíduos absolutos.

    Args:
        alpha: nível de erro alvo (1-α = cobertura nominal).
               Default 0.1 → IC 90%.

    Attrs:
        q_hat: quantil ajustado após fit (None antes).
        n_calib: tamanho do conjunto de calibração.
        alpha: salvo do __init__.
    """

    nome: str = "split_conformal"

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0 < alpha < 1:
            raise ValueError(f"alpha fora de (0,1): {alpha}")
        self.alpha = float(alpha)
        self.q_hat: float | None = None
        self.n_calib: int = 0

    def fit(self, residuos_abs: np.ndarray | pd.Series) -> "SplitConformal":
        """Ajusta q̂ a partir dos resíduos absolutos do conjunto de calibração."""
        r = np.asarray(residuos_abs, dtype="float64").ravel()
        r = r[~np.isnan(r)]
        if len(r) < 5:
            raise ValueError(
                f"poucos resíduos válidos pra calibrar: {len(r)} (mínimo 5)"
            )
        if np.any(r < 0):
            raise ValueError("residuos_abs tem valores negativos — passe |r|")
        self.q_hat = _quantile_residuos(r, self.alpha)
        self.n_calib = int(len(r))
        logger.info(
            "SplitConformal.fit: n=%d, alpha=%.3f, q_hat=%.4f",
            self.n_calib, self.alpha, self.q_hat,
        )
        return self

    def predict_interval(
        self,
        y_pred: np.ndarray | pd.Series,
        *,
        clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo [y_pred - q̂, y_pred + q̂]. Por default clipa em [0,1]."""
        if self.q_hat is None:
            raise RuntimeError("SplitConformal não foi ajustado — chame .fit antes")
        p = np.asarray(y_pred, dtype="float64").ravel()
        lo = p - self.q_hat
        hi = p + self.q_hat
        if clip:
            lo = np.clip(lo, 0.0, 1.0)
            hi = np.clip(hi, 0.0, 1.0)
        return lo, hi


# ============================================================
# MondrianConformal — quantil por bin de pred
# ============================================================
class MondrianConformal:
    """Conformal estratificado por bin de predição (decis).

    Define bins via `np.quantile(pred_calib, ...)` no fit, calcula q̂_b
    em cada bin, e na predição atribui novas amostras pelo seu pred.

    Bins com `< min_per_bin` pontos caem pro q̂ global (amostra
    insuficiente pra estimar quantil específico).

    Args:
        alpha: nível de erro alvo.
        n_bins: número de bins por quantil de pred (default 10 = decis).
        min_per_bin: bins menores que isso fallback pro global.
        min_q_factor: floor mínimo no q̂ por bin como fração do q̂ global
            (default 0.0 = sem floor, comportamento original). Em prod com
            n grande e bins de pred baixa, a distribuição de resíduos no
            bin pode degenerar (delta em zero) e o quantil 90% também
            colapsar pra zero — o intervalo zerado não cobre as caudas.
            Setando 0.3 ou 0.5 garante que `q_per_bin >= min_q_factor *
            q_global` em todos os bins. Recomendado: 0.5 em prod.

    Attrs:
        bin_edges: ndarray (n_bins+1,) com as bordas (após fit).
        q_per_bin: ndarray (n_bins,) com o q̂ de cada bin.
        q_global: float — fallback para bins com pouco dado.
        bins_fallback: lista de índices que caíram no global.
        bins_floored: lista de índices que receberam floor min_q_factor.
    """

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
        """Ajusta um q̂ por bin de pred. Bin pequeno → fallback ao global."""
        p = np.asarray(pred_calib, dtype="float64").ravel()
        r = np.asarray(residuos_abs, dtype="float64").ravel()
        if p.shape != r.shape:
            raise ValueError(
                f"shapes não batem: pred={p.shape} vs residuos={r.shape}"
            )
        mask = ~(np.isnan(p) | np.isnan(r))
        p, r = p[mask], r[mask]
        n = len(p)
        if n < self.n_bins * 2:
            raise ValueError(
                f"poucos pontos ({n}) pra n_bins={self.n_bins}; precisa de >= "
                f"{self.n_bins * 2}"
            )
        if np.any(r < 0):
            raise ValueError("residuos_abs tem valores negativos — passe |r|")

        # Bordas por quantil dos preds — 0 e 1 explícitos pra cobrir caudas
        qs = np.linspace(0.0, 1.0, self.n_bins + 1)
        edges = np.quantile(p, qs)
        # Empate em quantis pode produzir bordas iguais — colapsamos pra
        # bordas estritamente crescentes (np.unique). Se isso reduzir o
        # número de bins efetivos, o usuário verá warning no log.
        edges_u = np.unique(edges)
        if len(edges_u) < self.n_bins + 1:
            logger.warning(
                "MondrianConformal: pred_calib tem ties — bordas reduzidas de "
                "%d pra %d", self.n_bins + 1, len(edges_u),
            )
        edges_u[0] = -np.inf       # cobre extremos baixos no predict_interval
        edges_u[-1] = np.inf
        self.bin_edges = edges_u
        n_bins_eff = len(edges_u) - 1

        # Quantil global como fallback
        q_global = _quantile_residuos(r, self.alpha)
        self.q_global = q_global

        # bin index ∈ [0, n_bins_eff-1]
        # np.digitize com right=False: edges[i] <= x < edges[i+1] cai no bin i
        bin_idx = np.digitize(p, edges_u[1:-1], right=False)  # 0..n_bins_eff-1

        q_per_bin = np.empty(n_bins_eff, dtype="float64")
        bins_fallback: list[int] = []
        for b in range(n_bins_eff):
            r_b = r[bin_idx == b]
            n_b = len(r_b)
            if n_b < self.min_per_bin:
                q_per_bin[b] = q_global
                bins_fallback.append(b)
                logger.info(
                    "MondrianConformal: bin %d com n=%d < %d -> fallback global "
                    "(q̂=%.4f)", b, n_b, self.min_per_bin, q_global,
                )
            else:
                q_per_bin[b] = _quantile_residuos(r_b, self.alpha)

        # Floor: bins com q̂ degenerado (~0) recebem ao menos
        # min_q_factor * q_global pra evitar intervalos zerados.
        bins_floored: list[int] = []
        if self.min_q_factor > 0.0:
            floor = self.min_q_factor * q_global
            for b in range(n_bins_eff):
                if q_per_bin[b] < floor:
                    logger.info(
                        "MondrianConformal: bin %d q=%.4f < floor=%.4f "
                        "(min_q_factor=%.2f * q_global=%.4f) -> ajustado",
                        b, q_per_bin[b], floor, self.min_q_factor, q_global,
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
        """Atribui um índice de bin a cada predição."""
        assert self.bin_edges is not None
        return np.digitize(pred, self.bin_edges[1:-1], right=False)

    def predict_interval(
        self,
        y_pred: np.ndarray | pd.Series,
        *,
        clip: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Intervalo [pred - q̂_b, pred + q̂_b], onde b é o bin do pred."""
        if self.q_per_bin is None or self.bin_edges is None:
            raise RuntimeError("MondrianConformal não foi ajustado — chame .fit antes")
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
# Helpers de alto nível: cobertura observada por decil
# ============================================================
def coverage_por_decil(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    lower: np.ndarray | pd.Series,
    upper: np.ndarray | pd.Series,
    *,
    n_quantis: int = 10,
) -> pd.DataFrame:
    """Cobertura empírica por decil do y_pred.

    Útil pra detectar bins onde o intervalo está sub/sobrecobrindo. Ideal:
    cobertura empírica ≈ 1-α em cada decil (cobertura condicional).

    Returns:
        DataFrame com colunas [decil, n, pred_min, pred_max, cobertura].
    """
    y = np.asarray(y_true, dtype="float64").ravel()
    p = np.asarray(y_pred, dtype="float64").ravel()
    lo = np.asarray(lower, dtype="float64").ravel()
    hi = np.asarray(upper, dtype="float64").ravel()
    if not (y.shape == p.shape == lo.shape == hi.shape):
        raise ValueError("shapes desiguais entre y_true, y_pred, lower, upper")

    df = pd.DataFrame({"y": y, "pred": p, "lo": lo, "hi": hi})
    # qcut com duplicates='drop' pra empates em pred
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


__all__ = [
    "SplitConformal",
    "MondrianConformal",
    "compute_residuals",
    "coverage_observed",
    "coverage_por_decil",
]
