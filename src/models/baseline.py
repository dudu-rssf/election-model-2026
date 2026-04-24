"""
src.models.baseline — baselines simples pro target `share_1t`.

Três baselines, ordenados por sofisticação:

    B0: mediana histórica do share do partido na UF
        -> um número por (sigla_partido, sigla_uf).
        -> prediz isso pra toda linha. Não usa nada específico do município.
        -> **teto de simplicidade.** Se o LightGBM perde disso, algo tá errado.

    B1: lag_share_1t (share do mesmo partido no mesmo município na eleição
        presidencial anterior).
        -> **feature única.** Captura inércia eleitoral local.
        -> Quando lag é NaN (partido não concorreu no ano anterior ou é
           ano inicial do histórico), cai pra B0 como fallback.

    B2: blend linear de B0 e B1, com peso alpha (padrão 0.5).
        -> útil pra suavizar: B1 é barulhento em municípios com poucos
           votos, B0 é lento pra atualizar mudanças locais.

Todos seguem API sklearn-like:

    b = BaselineX()
    b.fit(X_train, y_train, meta_train=..., partido_col="sigla_partido",
          uf_col="sigla_uf")
    y_pred = b.predict(X_test, meta_test=...)

`meta_*` carrega as colunas que o baseline precisa pra lookup (sigla_partido,
sigla_uf, lag_share_1t). Passamos separado pra deixar claro que não são
features no sentido ML — são identificadores/lookups.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Baseline(ABC):
    """Interface comum."""

    nome: str = "baseline"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs) -> "Baseline":
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        ...


# ------------------------------------------------------------
# B0 — mediana histórica do (partido, UF)
# ------------------------------------------------------------
class MedianaPartidoUF(Baseline):
    """Prediz a mediana histórica do share do partido na UF.

    Fallback cascateado: se (partido, UF) não existe no treino, usa
    mediana só do partido; se nem isso, global.
    """

    nome = "B0_mediana_partido_uf"

    def __init__(self) -> None:
        self._mediana_partido_uf: dict[tuple[str, str], float] = {}
        self._mediana_partido: dict[str, float] = {}
        self._mediana_global: float = float("nan")

    def fit(
        self,
        X: pd.DataFrame,  # noqa: ARG002  (não usado, assinatura uniforme)
        y: pd.Series,
        *,
        meta: pd.DataFrame,
        partido_col: str = "sigla_partido",
        uf_col: str = "sigla_uf",
    ) -> "MedianaPartidoUF":
        if partido_col not in meta.columns or uf_col not in meta.columns:
            raise ValueError(f"meta precisa de {partido_col} e {uf_col}")
        df = pd.DataFrame({
            "partido": meta[partido_col].astype("string"),
            "uf": meta[uf_col].astype("string"),
            "y": y.values,
        })
        self._mediana_partido_uf = (
            df.groupby(["partido", "uf"])["y"].median().to_dict()
        )
        self._mediana_partido = df.groupby("partido")["y"].median().to_dict()
        self._mediana_global = float(df["y"].median())
        logger.info(
            "B0 fit: %d (partido,UF), %d partidos, global=%.4f",
            len(self._mediana_partido_uf),
            len(self._mediana_partido),
            self._mediana_global,
        )
        return self

    def predict(
        self,
        X: pd.DataFrame,  # noqa: ARG002
        *,
        meta: pd.DataFrame,
        partido_col: str = "sigla_partido",
        uf_col: str = "sigla_uf",
    ) -> np.ndarray:
        if partido_col not in meta.columns or uf_col not in meta.columns:
            raise ValueError(f"meta precisa de {partido_col} e {uf_col}")
        partidos = meta[partido_col].astype("string")
        ufs = meta[uf_col].astype("string")

        out = np.empty(len(meta), dtype="float64")
        for i, (p, u) in enumerate(zip(partidos, ufs)):
            v = self._mediana_partido_uf.get((p, u))
            if v is None or np.isnan(v):
                v = self._mediana_partido.get(p, self._mediana_global)
            out[i] = v if v is not None else self._mediana_global
        return out


# ------------------------------------------------------------
# B1 — lag_share_1t (fallback B0)
# ------------------------------------------------------------
class LagShare(Baseline):
    """Prediz o lag_share_1t (share da presidencial anterior). Fallback B0."""

    nome = "B1_lag_share"

    def __init__(self) -> None:
        self._b0 = MedianaPartidoUF()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        meta: pd.DataFrame,
        partido_col: str = "sigla_partido",
        uf_col: str = "sigla_uf",
    ) -> "LagShare":
        self._b0.fit(X, y, meta=meta, partido_col=partido_col, uf_col=uf_col)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        meta: pd.DataFrame,
        partido_col: str = "sigla_partido",
        uf_col: str = "sigla_uf",
        lag_col: str = "lag_share_1t",
    ) -> np.ndarray:
        if lag_col not in X.columns:
            raise ValueError(f"X precisa da coluna {lag_col!r}")
        lag = X[lag_col].astype("float64").values
        b0_pred = self._b0.predict(
            X, meta=meta, partido_col=partido_col, uf_col=uf_col
        )
        # Fallback onde lag é NaN
        out = np.where(np.isnan(lag), b0_pred, lag)
        # Clip pra [0, 1] por garantia (lag vem direto do dado, já deveria
        # estar em [0,1], mas proteção não machuca)
        out = np.clip(out, 0.0, 1.0)
        return out


# ------------------------------------------------------------
# B2 — blend linear de B0 e B1
# ------------------------------------------------------------
class BlendB0B1(Baseline):
    """Blend linear: alpha * B1 + (1 - alpha) * B0. alpha=0.5 por padrão."""

    nome = "B2_blend"

    def __init__(self, alpha: float = 0.5) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha fora de [0,1]: {alpha}")
        self.alpha = float(alpha)
        self._b0 = MedianaPartidoUF()
        self._b1 = LagShare()

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        meta: pd.DataFrame,
        partido_col: str = "sigla_partido",
        uf_col: str = "sigla_uf",
    ) -> "BlendB0B1":
        self._b0.fit(X, y, meta=meta, partido_col=partido_col, uf_col=uf_col)
        self._b1.fit(X, y, meta=meta, partido_col=partido_col, uf_col=uf_col)
        return self

    def predict(
        self,
        X: pd.DataFrame,
        *,
        meta: pd.DataFrame,
        partido_col: str = "sigla_partido",
        uf_col: str = "sigla_uf",
        lag_col: str = "lag_share_1t",
    ) -> np.ndarray:
        p0 = self._b0.predict(X, meta=meta, partido_col=partido_col, uf_col=uf_col)
        p1 = self._b1.predict(
            X, meta=meta, partido_col=partido_col, uf_col=uf_col, lag_col=lag_col
        )
        return self.alpha * p1 + (1.0 - self.alpha) * p0


__all__ = [
    "Baseline",
    "MedianaPartidoUF",
    "LagShare",
    "BlendB0B1",
]
