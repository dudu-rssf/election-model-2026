"""
Testes de `src.models.calibrate` — IsotonicCalibrator + oof_predictions_por_ano.

Cobre:
  * IsotonicCalibrator: monotonicidade, clip [0,1], corrige saturação,
    é identidade pra dados já calibrados, raise em casos degenerados.
  * oof_predictions_por_ano: shape correto, ordem preservada, pula ano
    sem dados, monkeypatch evita treinar LGBM de verdade.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import calibrate as cal


# ============================================================
# IsotonicCalibrator
# ============================================================
def test_isotonic_monotonico_simples() -> None:
    """f(x_baixo) <= f(x_alto) — propriedade central da regressão isotônica."""
    rng = np.random.default_rng(0)
    pred = rng.uniform(0, 1, size=200)
    real = pred + rng.normal(0, 0.05, size=200)   # ruído
    real = np.clip(real, 0.0, 1.0)
    c = cal.IsotonicCalibrator().fit(pred, real)

    grade = np.linspace(0.05, 0.95, 50)
    saida = c.predict(grade)
    assert np.all(np.diff(saida) >= -1e-9), "predict não-monotônico"
    assert np.all(saida >= 0.0) and np.all(saida <= 1.0)


def test_isotonic_corrige_saturacao() -> None:
    """Cenário Fase 4.5: pred satura em 0.7, real chega em ~0.9 na cauda.

    Esperado: calibrator(0.7) > 0.7. Sem exigir igualdade exata, só correção
    na direção certa.
    """
    rng = np.random.default_rng(1)
    n_baixo = 200
    n_alto = 50
    pred_baixo = rng.uniform(0.05, 0.5, size=n_baixo)
    real_baixo = pred_baixo + rng.normal(0, 0.02, size=n_baixo)
    # Cauda: pred saturada em ~0.65-0.72, mas real em 0.85-0.99
    pred_alto = rng.uniform(0.65, 0.72, size=n_alto)
    real_alto = rng.uniform(0.85, 0.99, size=n_alto)
    pred = np.concatenate([pred_baixo, pred_alto])
    real = np.clip(np.concatenate([real_baixo, real_alto]), 0.0, 1.0)

    c = cal.IsotonicCalibrator().fit(pred, real)
    saida_no_topo = c.predict(np.array([0.70]))[0]
    assert saida_no_topo > 0.80, (
        f"calibrator não corrigiu cauda: predict(0.70)={saida_no_topo:.3f}"
    )


def test_isotonic_clip_em_zero_um() -> None:
    """Predições fora de [0,1] são clipadas."""
    pred = np.linspace(0.05, 0.95, 50)
    real = np.linspace(0.05, 0.95, 50)
    c = cal.IsotonicCalibrator().fit(pred, real)
    out = c.predict(np.array([-0.5, 0.5, 1.5]))
    assert out[0] >= 0.0 and out[0] <= 1.0
    assert out[2] >= 0.0 and out[2] <= 1.0


def test_isotonic_identidade_em_dados_lineares() -> None:
    """Se pred=real exatamente, calibrator(x) ≈ x."""
    pred = np.linspace(0.05, 0.95, 100)
    real = pred.copy()
    c = cal.IsotonicCalibrator().fit(pred, real)
    saida = c.predict(np.array([0.2, 0.5, 0.8]))
    np.testing.assert_allclose(saida, [0.2, 0.5, 0.8], atol=0.01)


def test_isotonic_predict_sem_fit_raise() -> None:
    c = cal.IsotonicCalibrator()
    with pytest.raises(RuntimeError):
        c.predict(np.array([0.1, 0.5]))


def test_isotonic_fit_pouco_dado_raise() -> None:
    c = cal.IsotonicCalibrator()
    with pytest.raises(ValueError):
        c.fit(np.array([0.1, 0.2]), np.array([0.15, 0.25]))


def test_isotonic_fit_shape_mismatch_raise() -> None:
    c = cal.IsotonicCalibrator()
    with pytest.raises(ValueError):
        c.fit(np.array([0.1, 0.2, 0.3]), np.array([0.1, 0.2]))


def test_isotonic_ignora_nan() -> None:
    pred = np.array([0.1, 0.2, np.nan, 0.5, 0.8, 0.9])
    real = np.array([0.15, 0.18, 0.4, np.nan, 0.7, 0.85])
    # mask remove pred[2] e real[3] -> 4 pares válidos. Mas o fit exige >=5.
    # Adicionamos mais um par válido pra passar.
    pred = np.append(pred, 0.6)
    real = np.append(real, 0.65)
    c = cal.IsotonicCalibrator().fit(pred, real)
    assert c._n_fit == 5    # 7 - 2 NaN entries


# ============================================================
# IsotonicCalibrator.min_pred — calibração assimétrica
# ============================================================
def test_min_pred_passa_baixos_raw() -> None:
    """Quando pred < min_pred, predict retorna o pred raw (não passa pelo iso)."""
    rng = np.random.default_rng(2)
    pred = rng.uniform(0, 1, size=200)
    # Calibrator que infla tudo: real = pred + 0.3
    real = np.clip(pred + 0.3, 0, 1)
    c = cal.IsotonicCalibrator(min_pred=0.5).fit(pred, real)

    saida = c.predict(np.array([0.10, 0.30, 0.49, 0.51, 0.80]))
    # Abaixo de 0.5: passa raw
    np.testing.assert_allclose(saida[:3], [0.10, 0.30, 0.49], atol=1e-9)
    # >= 0.5: passa pelo isotonic (que aprende real = pred + 0.3)
    assert saida[3] > 0.51
    assert saida[4] > 0.80


def test_min_pred_zero_equivale_a_none() -> None:
    """min_pred=None deve dar mesmo output de min_pred=0 (calibrar todos).

    Nota: o script usa convenção 0 desliga, mas internamente o objeto
    aceita None como 'sem assimetria'. Aqui testamos a equivalência
    funcional pra pred > 0 (que é o caso real).
    """
    rng = np.random.default_rng(3)
    pred = rng.uniform(0.05, 0.95, size=100)
    real = np.clip(pred + rng.normal(0, 0.05, size=100), 0, 1)

    c_none = cal.IsotonicCalibrator(min_pred=None).fit(pred, real)
    c_zero = cal.IsotonicCalibrator(min_pred=0.0).fit(pred, real)

    teste = np.array([0.1, 0.5, 0.9])
    np.testing.assert_allclose(c_none.predict(teste), c_zero.predict(teste))


def test_min_pred_monotonia_no_break() -> None:
    """A junção raw/iso pode introduzir descontinuidade no ponto de corte;
    aqui só garantimos que predict não estoura range [0,1]."""
    rng = np.random.default_rng(4)
    pred = rng.uniform(0, 1, size=200)
    real = pred * 0.5    # iso retorna ~metade do pred — abaixo do raw
    c = cal.IsotonicCalibrator(min_pred=0.4).fit(pred, real)

    grade = np.linspace(0, 1, 100)
    out = c.predict(grade)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


# ============================================================
# oof_predictions_por_ano
# ============================================================
class _FakePrep:
    """Mock duck-typed PreparedData (X, y, meta, cat_features)."""
    def __init__(self, X: pd.DataFrame, y: pd.Series, meta: pd.DataFrame,
                 cat_features: list[str] | None = None) -> None:
        self.X = X.reset_index(drop=True)
        self.y = y.reset_index(drop=True)
        self.meta = meta.reset_index(drop=True)
        self.cat_features = cat_features or []


@pytest.fixture
def prep_fake() -> _FakePrep:
    """3 anos × 5 linhas/ano = 15 linhas. ano_municipal na meta."""
    anos = [2012, 2016, 2020]
    rows = []
    for ano in anos:
        for i in range(5):
            rows.append({"ano_municipal": ano, "feat_a": float(i), "feat_b": float(ano % 100)})
    df = pd.DataFrame(rows)
    X = df[["feat_a", "feat_b"]].copy()
    y = pd.Series(np.linspace(0.1, 0.9, len(df)), name="share_1t")
    meta = df[["ano_municipal"]].copy()
    return _FakePrep(X, y, meta)


def test_oof_shape_e_ordem(prep_fake, monkeypatch) -> None:
    """OOF retorna len(prep) linhas, ordenadas por idx_original (0..N-1)."""
    # monkeypatch treinar_lgbm e prever pra evitar dependência de LGBM
    from src.models import train as tr

    def _fake_treinar(X_tr, y_tr, **kwargs):
        return ("fake_model", len(X_tr))

    def _fake_prever(model, X, **kwargs):
        # retorna 0.5 fixo (qualquer constante)
        return np.full(len(X), 0.5, dtype="float64")

    monkeypatch.setattr(tr, "treinar_lgbm", _fake_treinar)
    monkeypatch.setattr(tr, "prever", _fake_prever)

    df_oof = cal.oof_predictions_por_ano(
        prep_fake, anos_treino=[2012, 2016, 2020], ano_col="ano_municipal",
    )
    assert len(df_oof) == len(prep_fake.X)
    # cada ano aparece exatamente 5 vezes
    assert df_oof["ano"].value_counts().to_dict() == {2012: 5, 2016: 5, 2020: 5}
    # idx_original cobre 0..14 sem repetição
    assert sorted(df_oof["idx_original"].tolist()) == list(range(15))
    # ordenação por idx_original ascendente
    assert df_oof["idx_original"].is_monotonic_increasing
    # y_true bate com prep.y na ordem
    np.testing.assert_array_equal(
        df_oof["y_true"].to_numpy(),
        prep_fake.y.iloc[df_oof["idx_original"]].to_numpy(),
    )


def test_oof_pula_ano_sem_dados(prep_fake, monkeypatch, caplog) -> None:
    """Pede oof em ano que não está em prep -> warning + pula sem quebrar."""
    from src.models import train as tr

    monkeypatch.setattr(tr, "treinar_lgbm", lambda *a, **k: "fake_model")
    monkeypatch.setattr(tr, "prever",
                        lambda model, X, **k: np.full(len(X), 0.5, dtype="float64"))

    df_oof = cal.oof_predictions_por_ano(
        prep_fake, anos_treino=[2012, 2016, 2020, 2099],   # 2099 não existe
        ano_col="ano_municipal",
    )
    # 2099 não deve aparecer
    assert 2099 not in df_oof["ano"].unique()
    assert len(df_oof) == 15


def test_oof_raise_ano_col_ausente(prep_fake) -> None:
    with pytest.raises(ValueError, match="ano_col"):
        cal.oof_predictions_por_ano(
            prep_fake, anos_treino=[2012, 2016], ano_col="ano_inexistente",
        )


def test_oof_raise_anos_unico() -> None:
    """Precisa de >=2 anos pra fazer leave-one-out."""
    df = pd.DataFrame({"feat_a": [1.0, 2.0]})
    prep = _FakePrep(df, pd.Series([0.3, 0.7]), pd.DataFrame({"ano_municipal": [2020, 2020]}))
    with pytest.raises(ValueError, match=">=2"):
        cal.oof_predictions_por_ano(prep, anos_treino=[2020], ano_col="ano_municipal")


# ============================================================
# treinar_calibrador_oof — integração leve
# ============================================================
def test_treinar_calibrador_oof_devolve_calibrador(prep_fake, monkeypatch) -> None:
    from src.models import train as tr

    monkeypatch.setattr(tr, "treinar_lgbm", lambda *a, **k: "fake_model")

    # Predições OOF "ruins" que precisam de correção: pred constante 0.4,
    # mas y_true varia em [0.1, 0.9]. Calibrator vai aprender que pred=0.4
    # corresponde a y médio.
    monkeypatch.setattr(tr, "prever",
                        lambda model, X, **k: np.full(len(X), 0.4, dtype="float64"))

    calibrator, df_oof = cal.treinar_calibrador_oof(
        prep_fake, anos_treino=[2012, 2016, 2020], ano_col="ano_municipal",
    )
    assert isinstance(calibrator, cal.IsotonicCalibrator)
    assert len(df_oof) == 15
    out = calibrator.predict(np.array([0.4]))[0]
    # Com pred constante = 0.4 e y_mean ≈ 0.5, calibrator deve mapear pra ~0.5
    assert 0.3 <= out <= 0.7


# ============================================================
# holdout_predictions_um_ano + treinar_calibrador_holdout
# ============================================================
def test_holdout_um_ano_shape(prep_fake, monkeypatch) -> None:
    """Treina UMA vez (não N), prediz só no ano_calib."""
    from src.models import train as tr
    chamadas = {"treinar": 0}

    def _fake_treinar(*a, **k):
        chamadas["treinar"] += 1
        return "fake_model"

    monkeypatch.setattr(tr, "treinar_lgbm", _fake_treinar)
    monkeypatch.setattr(tr, "prever",
                        lambda model, X, **k: np.full(len(X), 0.5, dtype="float64"))

    df_h = cal.holdout_predictions_um_ano(
        prep_fake, ano_calib=2020, anos_treino=[2012, 2016, 2020],
        ano_col="ano_municipal",
    )
    # treinou exatamente 1 vez (vs OOF que treina N=3 vezes)
    assert chamadas["treinar"] == 1
    # apenas linhas do ano_calib
    assert len(df_h) == 5
    assert (df_h["ano"] == 2020).all()
    # idx_original aponta pras linhas certas (10..14 em prep_fake; 2020 é o último ano)
    assert sorted(df_h["idx_original"].tolist()) == [10, 11, 12, 13, 14]


def test_holdout_raise_ano_fora(prep_fake) -> None:
    with pytest.raises(ValueError, match="não está em anos_treino"):
        cal.holdout_predictions_um_ano(
            prep_fake, ano_calib=2099, anos_treino=[2012, 2016, 2020],
            ano_col="ano_municipal",
        )


def test_treinar_calibrador_holdout_devolve_calibrador(prep_fake, monkeypatch) -> None:
    from src.models import train as tr

    monkeypatch.setattr(tr, "treinar_lgbm", lambda *a, **k: "fake_model")
    monkeypatch.setattr(tr, "prever",
                        lambda model, X, **k: np.full(len(X), 0.4, dtype="float64"))

    calibrator, df_h = cal.treinar_calibrador_holdout(
        prep_fake, ano_calib=2020, anos_treino=[2012, 2016, 2020],
        ano_col="ano_municipal", min_pred=0.3,
    )
    assert isinstance(calibrator, cal.IsotonicCalibrator)
    assert calibrator.min_pred == 0.3
    assert len(df_h) == 5
