"""
src.models.calibrate — calibração pós-hoc das predições do LightGBM.

Motivação (Fase 4.5): no eixo municipal, o LGBM com `objective=regression_l1`
satura num teto de logit ~+1.0 (pred.max ≈ 0.74). Isso enviesa o top decil:
candidatos com share_real entre 0.7 e 1.0 ficam todos predizidos perto de
0.7. A causa é uma combinação de (a) ausência de candidaturas únicas no
treino e (b) MAE como objetivo, que deliberadamente sacrifica caudas.

Solução leve, sem mexer no LGBM: regressão isotônica em (pred, real),
ajustada com predições do modelo no treino, aplicada nas predições
do teste.

Dois modos de gerar predições no treino:

  * `oof`     — leave-one-year-out cross-validation. Cada ano vira fold de
                teste; treina LGBM nos outros, prediz no held-out. Ruidoso
                quando há poucos anos (cada fold tem ~2/3 do treino, e o
                ruído se acumula nos N folds).
  * `holdout` — separa um único ano do treino como conjunto de calibração.
                Treina LGBM nos restantes (uma vez), prediz no holdout.
                Mais estável que OOF para pouco dado; recomendado em dev.

API:
    cal = IsotonicCalibrator(min_pred=0.5).fit(pred_calib, y_calib)
    pred_cal = cal.predict(pred_test)

    cal, df = treinar_calibrador_holdout(prep_train, ano_calib, anos_treino, ano_col)
    cal, df = treinar_calibrador_oof(prep_train, anos_treino, ano_col)

`min_pred` (calibração assimétrica): valores de pred abaixo do limiar
passam direto (sem aplicar isotonic). Útil quando o calibrator inflou
indevidamente a cauda baixa (caso típico quando o modelo de calibração
tem viés de overshoot em decis baixos por menos dados de treino).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# IsotonicCalibrator: wrap fino do sklearn
# ------------------------------------------------------------
class IsotonicCalibrator:
    """Regressão isotônica pred → real, monotônica não-decrescente.

    Usa `sklearn.isotonic.IsotonicRegression(out_of_bounds='clip')`. Com
    clip, predição em pontos fora do intervalo de treino é projetada na
    borda — comportamento estável, não extrapolam.

    Args:
        out_of_bounds: passado direto pro sklearn ('clip' ou 'nan').
        increasing: True (default) — preserva a ordem das predições.
        min_pred: se fornecido, predições abaixo desse valor passam direto
            (sem aplicar isotonic). Útil quando o calibrator inflou
            indevidamente a cauda baixa. None desliga (calibra todos).
    """

    nome: str = "isotonic"

    def __init__(
        self,
        out_of_bounds: str = "clip",
        increasing: bool = True,
        min_pred: float | None = None,
    ) -> None:
        self.out_of_bounds = out_of_bounds
        self.increasing = increasing
        self.min_pred = min_pred
        self._iso: Any = None
        self._n_fit: int = 0

    def fit(
        self,
        pred: np.ndarray | pd.Series,
        real: np.ndarray | pd.Series,
    ) -> "IsotonicCalibrator":
        from sklearn.isotonic import IsotonicRegression

        p = np.asarray(pred, dtype="float64").ravel()
        y = np.asarray(real, dtype="float64").ravel()
        if p.shape != y.shape:
            raise ValueError(f"shape mismatch: pred={p.shape} vs real={y.shape}")
        mask = ~(np.isnan(p) | np.isnan(y))
        if mask.sum() < 5:
            raise ValueError(f"poucos pontos válidos pra calibrar: {int(mask.sum())}")
        self._iso = IsotonicRegression(
            out_of_bounds=self.out_of_bounds,
            increasing=self.increasing,
            y_min=0.0,
            y_max=1.0,
        )
        self._iso.fit(p[mask], y[mask])
        self._n_fit = int(mask.sum())
        logger.info("IsotonicCalibrator.fit: n=%d, range pred=[%.4f, %.4f], min_pred=%s",
                    self._n_fit, p[mask].min(), p[mask].max(), self.min_pred)
        return self

    def predict(self, pred: np.ndarray | pd.Series) -> np.ndarray:
        if self._iso is None:
            raise RuntimeError("IsotonicCalibrator não foi ajustado — chame .fit antes")
        p = np.asarray(pred, dtype="float64").ravel()
        out_iso = self._iso.predict(p)
        if self.min_pred is None:
            out = out_iso
        else:
            # calibração assimétrica: usa raw quando pred < min_pred
            mask_calibrar = p >= float(self.min_pred)
            out = np.where(mask_calibrar, out_iso, p)
        return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------
# OOF predictions — leave-one-year-out
# ------------------------------------------------------------
def oof_predictions_por_ano(
    prep,                                   # PreparedData (presidencial ou prefeito; duck typed)
    anos_treino: list[int],
    *,
    ano_col: str = "ano_presidencial",
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Treina LGBM N vezes com leave-one-year-out, retorna preds OOF concatenadas.

    Para cada ano em `anos_treino`:
        - treina em (anos_treino - {ano})
        - prediz em {ano}
    Concatena todas as predições, junto com o y_true correspondente.

    Args:
        prep: PreparedData de TREINO já filtrado pros anos_treino.
              Precisa ter `prep.meta[ano_col]` populado.
        anos_treino: anos a usar como folds (cada um vira teste em sua iter).
        ano_col: coluna em prep.meta que carrega o ano (default 'ano_presidencial').
        overrides: passa pro params_lgbm.

    Returns:
        DataFrame com colunas:
            [ano, idx_original, y_true, y_pred_oof]
        idx_original = posição da linha em `prep` (0..len-1).
    """
    from src.models import train as tr  # local import: evita ciclo

    if ano_col not in prep.meta.columns:
        raise ValueError(f"prep.meta sem coluna {ano_col!r}")
    anos = sorted({int(a) for a in anos_treino})
    if len(anos) < 2:
        raise ValueError(f"oof precisa de >=2 anos; got {anos}")

    serie_ano = prep.meta[ano_col].astype("int64").reset_index(drop=True)
    pieces: list[pd.DataFrame] = []

    for ano_holdout in anos:
        mask_te = (serie_ano == ano_holdout).to_numpy()
        mask_tr = ~mask_te
        n_te = int(mask_te.sum())
        n_tr = int(mask_tr.sum())
        if n_te == 0 or n_tr == 0:
            logger.warning("oof: pulando ano %d (n_treino=%d, n_teste=%d)",
                           ano_holdout, n_tr, n_te)
            continue

        X_tr = prep.X.loc[mask_tr].reset_index(drop=True)
        y_tr = prep.y.loc[mask_tr].reset_index(drop=True)
        X_te = prep.X.loc[mask_te].reset_index(drop=True)
        y_te = prep.y.loc[mask_te].reset_index(drop=True)

        logger.info("oof: holdout=%d  treino=%d  teste=%d", ano_holdout, n_tr, n_te)
        model = tr.treinar_lgbm(
            X_tr, y_tr, cat_features=prep.cat_features,
            overrides=overrides,
            early_stopping_rounds=None,    # sem val explícito; modelo final size
        )
        y_pred = tr.prever(model, X_te)

        idx_originais = np.where(mask_te)[0]
        pieces.append(pd.DataFrame({
            "ano": ano_holdout,
            "idx_original": idx_originais,
            "y_true": y_te.to_numpy(),
            "y_pred_oof": y_pred,
        }))

    if not pieces:
        raise RuntimeError("oof: nenhum fold válido")

    out = pd.concat(pieces, ignore_index=True).sort_values("idx_original").reset_index(drop=True)
    logger.info("oof: total %d linhas em %d folds", len(out), len(pieces))
    return out


# ------------------------------------------------------------
# Holdout: 1 ano único como conjunto de calibração
# ------------------------------------------------------------
def holdout_predictions_um_ano(
    prep,
    ano_calib: int,
    anos_treino: list[int],
    *,
    ano_col: str = "ano_presidencial",
    overrides: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Treina LGBM em (anos_treino - {ano_calib}) e prediz em {ano_calib}.

    Mais estável que OOF quando há poucos anos: treina UMA vez com
    `len(anos_treino) - 1` anos. O modelo final (em main()) é re-treinado
    com TODOS os anos_treino, mas o calibrator vem desse fold único.

    Args:
        prep: PreparedData de TREINO (já filtrado pra anos_treino).
        ano_calib: ano que vira holdout.
        anos_treino: lista de anos disponíveis em prep.
        ano_col: coluna de ano em prep.meta.

    Returns:
        DataFrame com (ano, idx_original, y_true, y_pred_holdout).
    """
    from src.models import train as tr

    if ano_col not in prep.meta.columns:
        raise ValueError(f"prep.meta sem coluna {ano_col!r}")
    anos = sorted({int(a) for a in anos_treino})
    if ano_calib not in anos:
        raise ValueError(f"ano_calib {ano_calib} não está em anos_treino {anos}")
    anos_modelo_calib = [a for a in anos if a != ano_calib]
    if not anos_modelo_calib:
        raise ValueError("anos_treino tem só um ano — sem treino para holdout")

    serie_ano = prep.meta[ano_col].astype("int64").reset_index(drop=True)
    mask_te = (serie_ano == ano_calib).to_numpy()
    mask_tr = ~mask_te
    n_tr = int(mask_tr.sum())
    n_te = int(mask_te.sum())
    if n_tr == 0 or n_te == 0:
        raise RuntimeError(
            f"holdout: dados insuficientes (treino={n_tr}, calib={n_te})"
        )
    logger.info("holdout: ano_calib=%d  treino=%d (anos %s)  calib=%d",
                ano_calib, n_tr, anos_modelo_calib, n_te)

    X_tr = prep.X.loc[mask_tr].reset_index(drop=True)
    y_tr = prep.y.loc[mask_tr].reset_index(drop=True)
    X_te = prep.X.loc[mask_te].reset_index(drop=True)
    y_te = prep.y.loc[mask_te].reset_index(drop=True)

    model = tr.treinar_lgbm(
        X_tr, y_tr, cat_features=prep.cat_features,
        overrides=overrides, early_stopping_rounds=None,
    )
    y_pred = tr.prever(model, X_te)
    idx_originais = np.where(mask_te)[0]
    out = pd.DataFrame({
        "ano": ano_calib,
        "idx_original": idx_originais,
        "y_true": y_te.to_numpy(),
        "y_pred_holdout": y_pred,
    })
    return out


# ------------------------------------------------------------
# Helpers de alto nível
# ------------------------------------------------------------
def treinar_calibrador_oof(
    prep_train,
    anos_treino: list[int],
    *,
    ano_col: str = "ano_presidencial",
    overrides: dict[str, Any] | None = None,
    min_pred: float | None = None,
) -> tuple[IsotonicCalibrator, pd.DataFrame]:
    """Atalho: gera OOF preds + ajusta IsotonicCalibrator.

    Returns:
        (calibrator, df_oof)
    """
    df_oof = oof_predictions_por_ano(
        prep_train, anos_treino, ano_col=ano_col, overrides=overrides,
    )
    cal = IsotonicCalibrator(min_pred=min_pred).fit(
        df_oof["y_pred_oof"], df_oof["y_true"],
    )
    return cal, df_oof


def treinar_calibrador_holdout(
    prep_train,
    ano_calib: int,
    anos_treino: list[int],
    *,
    ano_col: str = "ano_presidencial",
    overrides: dict[str, Any] | None = None,
    min_pred: float | None = None,
) -> tuple[IsotonicCalibrator, pd.DataFrame]:
    """Atalho: gera holdout preds + ajusta IsotonicCalibrator.

    Returns:
        (calibrator, df_holdout)
    """
    df_h = holdout_predictions_um_ano(
        prep_train, ano_calib, anos_treino,
        ano_col=ano_col, overrides=overrides,
    )
    cal = IsotonicCalibrator(min_pred=min_pred).fit(
        df_h["y_pred_holdout"], df_h["y_true"],
    )
    return cal, df_h


__all__ = [
    "IsotonicCalibrator",
    "oof_predictions_por_ano",
    "holdout_predictions_um_ano",
    "treinar_calibrador_oof",
    "treinar_calibrador_holdout",
]
