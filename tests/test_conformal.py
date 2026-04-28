"""
Testes de `src.models.conformal`.

Cobre:
  * compute_residuals: shapes, NaN, valor absoluto vs assinado.
  * coverage_observed: contagem correta dentro/fora do intervalo.
  * Quantil corrigido pra amostra finita: ⌈(n+1)(1-α)⌉ / n.
  * SplitConformal: cobertura empírica ≈ 1-α em dados sintéticos
    exchangeáveis; intervalo simétrico em torno de y_pred; clip [0,1].
  * MondrianConformal: bins corretos, fallback global pra bins pequenos,
    cobertura condicional aproximadamente 1-α em cada bin.
  * coverage_por_decil: tabela com colunas esperadas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import conformal as cf


# ============================================================
# Helpers / utilidades
# ============================================================
def test_compute_residuals_basico() -> None:
    y = np.array([0.1, 0.5, 0.9])
    p = np.array([0.2, 0.4, 0.95])
    out_abs = cf.compute_residuals(y, p)
    np.testing.assert_allclose(out_abs, [0.1, 0.1, 0.05])
    out_signed = cf.compute_residuals(y, p, absolute=False)
    np.testing.assert_allclose(out_signed, [-0.1, 0.1, -0.05])


def test_compute_residuals_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        cf.compute_residuals(np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]))


def test_coverage_observed() -> None:
    y = np.array([0.1, 0.5, 0.9, 0.3])
    lo = np.array([0.0, 0.4, 0.85, 0.5])
    hi = np.array([0.2, 0.6, 0.95, 0.7])
    # 0.1 ∈ [0,0.2] ✓, 0.5 ∈ [0.4,0.6] ✓, 0.9 ∈ [0.85,0.95] ✓, 0.3 ∉ [0.5,0.7] ✗
    cov = cf.coverage_observed(y, lo, hi)
    assert cov == 0.75


def test_coverage_observed_shape_mismatch() -> None:
    y = np.array([0.1, 0.5])
    lo = np.array([0.0])
    hi = np.array([0.2, 0.6])
    with pytest.raises(ValueError, match="shapes desiguais"):
        cf.coverage_observed(y, lo, hi)


def test_finite_sample_quantile_level() -> None:
    """⌈(n+1)(1-α)⌉ / n. Ex.: n=100, α=0.1 -> ⌈101·0.9⌉ / 100 = 91/100 = 0.91."""
    assert cf._finite_sample_quantile_level(100, 0.1) == pytest.approx(0.91)
    # Caso onde k > n -> clipa em 1.0
    # n=10, alpha=0.01 -> ⌈11·0.99⌉ = 11 > 10 -> 1.0
    assert cf._finite_sample_quantile_level(10, 0.01) == 1.0


def test_finite_sample_quantile_level_invalido() -> None:
    with pytest.raises(ValueError):
        cf._finite_sample_quantile_level(0, 0.1)
    with pytest.raises(ValueError):
        cf._finite_sample_quantile_level(10, 0.0)
    with pytest.raises(ValueError):
        cf._finite_sample_quantile_level(10, 1.0)


# ============================================================
# SplitConformal
# ============================================================
def test_split_conformal_intervalo_simetrico() -> None:
    """Intervalo é simétrico: hi - pred == pred - lo == q̂."""
    rng = np.random.default_rng(0)
    residuos = np.abs(rng.normal(0, 0.1, size=200))
    sc = cf.SplitConformal(alpha=0.1).fit(residuos)
    assert sc.q_hat is not None
    assert sc.n_calib == 200

    pred = np.array([0.3, 0.5, 0.7])
    lo, hi = sc.predict_interval(pred, clip=False)
    np.testing.assert_allclose(hi - pred, sc.q_hat)
    np.testing.assert_allclose(pred - lo, sc.q_hat)


def test_split_conformal_clip() -> None:
    """Predições perto da borda têm intervalo clipado em [0,1]."""
    residuos = np.abs(np.random.default_rng(1).normal(0, 0.2, size=100))
    sc = cf.SplitConformal(alpha=0.1).fit(residuos)
    pred = np.array([0.05, 0.95])
    lo, hi = sc.predict_interval(pred, clip=True)
    assert lo[0] >= 0.0
    assert hi[1] <= 1.0
    # sem clip, lo[0] e hi[1] estouram o range
    lo_uc, hi_uc = sc.predict_interval(pred, clip=False)
    assert lo_uc[0] < 0.0
    assert hi_uc[1] > 1.0


def test_split_conformal_cobertura_empirica() -> None:
    """Em dados exchangeáveis, cobertura empírica ≈ 1 - α (com slack)."""
    rng = np.random.default_rng(42)
    n_calib = 500
    n_test = 500
    y_calib = rng.uniform(0, 1, size=n_calib)
    y_pred_calib = y_calib + rng.normal(0, 0.05, size=n_calib)
    y_test = rng.uniform(0, 1, size=n_test)
    y_pred_test = y_test + rng.normal(0, 0.05, size=n_test)

    residuos = cf.compute_residuals(y_calib, y_pred_calib)
    sc = cf.SplitConformal(alpha=0.1).fit(residuos)
    lo, hi = sc.predict_interval(y_pred_test, clip=False)
    cov = cf.coverage_observed(y_test, lo, hi)
    # Cobertura nominal = 0.9. Slack ±5pp pra n=500.
    assert 0.85 <= cov <= 0.97, f"cobertura fora do esperado: {cov:.3f}"


def test_split_conformal_predict_sem_fit_raise() -> None:
    sc = cf.SplitConformal()
    with pytest.raises(RuntimeError):
        sc.predict_interval(np.array([0.3, 0.5]))


def test_split_conformal_alpha_invalido() -> None:
    with pytest.raises(ValueError):
        cf.SplitConformal(alpha=0.0)
    with pytest.raises(ValueError):
        cf.SplitConformal(alpha=1.0)
    with pytest.raises(ValueError):
        cf.SplitConformal(alpha=-0.1)


def test_split_conformal_residuos_negativos_raise() -> None:
    """Resíduos absolutos não podem ser negativos."""
    sc = cf.SplitConformal()
    with pytest.raises(ValueError, match="negativos"):
        sc.fit(np.array([0.1, -0.2, 0.3, 0.4, 0.5]))


def test_split_conformal_poucos_dados() -> None:
    sc = cf.SplitConformal()
    with pytest.raises(ValueError, match="poucos"):
        sc.fit(np.array([0.1, 0.2]))


def test_split_conformal_ignora_nan() -> None:
    residuos = np.array([0.1, 0.2, np.nan, 0.4, 0.5, 0.6])
    sc = cf.SplitConformal(alpha=0.1).fit(residuos)
    assert sc.n_calib == 5    # sem o NaN


# ============================================================
# MondrianConformal
# ============================================================
def test_mondrian_fit_basico() -> None:
    rng = np.random.default_rng(0)
    n = 500
    pred = rng.uniform(0, 1, size=n)
    residuos = np.abs(rng.normal(0, 0.05, size=n))
    mc = cf.MondrianConformal(alpha=0.1, n_bins=5, min_per_bin=10).fit(pred, residuos)
    assert mc.bin_edges is not None
    assert mc.q_per_bin is not None
    assert len(mc.q_per_bin) == 5
    assert mc.n_calib == n
    # Sem fallback (todos os bins têm ~100 pontos)
    assert len(mc.bins_fallback) == 0


def test_mondrian_intervalo_por_bin() -> None:
    """Predições no mesmo bin recebem o mesmo q̂."""
    rng = np.random.default_rng(1)
    n = 400
    pred = rng.uniform(0, 1, size=n)
    # Resíduos crescem com pred — bins altos terão q̂ maior
    residuos = np.abs(rng.normal(0, 0.05 + 0.2 * pred))
    mc = cf.MondrianConformal(alpha=0.1, n_bins=4, min_per_bin=10).fit(pred, residuos)

    # Pred=0.05 (bin baixo) e pred=0.95 (bin alto) devem ter intervalos
    # de larguras diferentes
    lo_baixo, hi_baixo = mc.predict_interval(np.array([0.05]), clip=False)
    lo_alto, hi_alto = mc.predict_interval(np.array([0.95]), clip=False)
    largura_baixo = (hi_baixo - lo_baixo)[0]
    largura_alto = (hi_alto - lo_alto)[0]
    assert largura_alto > largura_baixo, (
        f"bin alto deveria ter intervalo maior: baixo={largura_baixo:.3f} "
        f"alto={largura_alto:.3f}"
    )


def test_mondrian_fallback_global() -> None:
    """Bins com poucos pontos caem pro q_global."""
    rng = np.random.default_rng(2)
    # 100 pontos em pred=[0, 0.5], 5 pontos em pred=[0.9, 1.0]
    pred = np.concatenate([
        rng.uniform(0, 0.5, size=100),
        rng.uniform(0.9, 1.0, size=5),
    ])
    residuos = np.abs(rng.normal(0, 0.05, size=len(pred)))
    mc = cf.MondrianConformal(alpha=0.1, n_bins=5, min_per_bin=10).fit(pred, residuos)
    # último bin tem só ~5 pontos -> fallback
    assert len(mc.bins_fallback) >= 1
    assert mc.q_per_bin[-1] == mc.q_global


def test_mondrian_cobertura_condicional() -> None:
    """Cobertura por bin ≈ 1-α em cenário heteroscedástico."""
    rng = np.random.default_rng(7)
    n_calib = 1000
    n_test = 1000
    pred_calib = rng.uniform(0, 1, size=n_calib)
    # heteroscedástico: ruído cresce com pred
    y_calib = pred_calib + rng.normal(0, 0.03 + 0.15 * pred_calib)
    pred_test = rng.uniform(0, 1, size=n_test)
    y_test = pred_test + rng.normal(0, 0.03 + 0.15 * pred_test)

    residuos = cf.compute_residuals(y_calib, pred_calib)
    mc = cf.MondrianConformal(alpha=0.1, n_bins=5, min_per_bin=20).fit(pred_calib, residuos)
    lo, hi = mc.predict_interval(pred_test, clip=False)

    # Cobertura marginal próxima de 0.9
    cov = cf.coverage_observed(y_test, lo, hi)
    assert 0.83 <= cov <= 0.97, f"cobertura marginal: {cov:.3f}"


def test_mondrian_predict_sem_fit_raise() -> None:
    mc = cf.MondrianConformal()
    with pytest.raises(RuntimeError):
        mc.predict_interval(np.array([0.3, 0.5]))


def test_mondrian_n_bins_invalido() -> None:
    with pytest.raises(ValueError):
        cf.MondrianConformal(n_bins=1)
    with pytest.raises(ValueError):
        cf.MondrianConformal(min_per_bin=0)


def test_mondrian_poucos_pontos_raise() -> None:
    """Precisa de >= n_bins * 2 pontos."""
    mc = cf.MondrianConformal(n_bins=10)
    pred = np.linspace(0, 1, 15)
    res = np.abs(np.random.default_rng(0).normal(0, 0.1, size=15))
    with pytest.raises(ValueError, match="poucos pontos"):
        mc.fit(pred, res)


def test_mondrian_residuos_negativos_raise() -> None:
    mc = cf.MondrianConformal(n_bins=3, min_per_bin=2)
    pred = np.linspace(0, 1, 30)
    res = np.full(30, 0.1)
    res[5] = -0.2
    with pytest.raises(ValueError, match="negativos"):
        mc.fit(pred, res)


def test_mondrian_shape_mismatch() -> None:
    mc = cf.MondrianConformal(n_bins=3, min_per_bin=2)
    pred = np.linspace(0, 1, 30)
    res = np.full(15, 0.1)
    with pytest.raises(ValueError, match="shapes não batem"):
        mc.fit(pred, res)


# ============================================================
# coverage_por_decil
# ============================================================
def test_coverage_por_decil_shape() -> None:
    rng = np.random.default_rng(0)
    n = 500
    y = rng.uniform(0, 1, size=n)
    p = y + rng.normal(0, 0.05, size=n)
    lo = p - 0.1
    hi = p + 0.1
    df = cf.coverage_por_decil(y, p, lo, hi, n_quantis=10)
    assert set(df.columns) == {"decil", "n", "pred_min", "pred_max", "cobertura"}
    # 10 decis (com slack: empates podem reduzir)
    assert 5 <= len(df) <= 10
    # cobertura ∈ [0,1]
    assert df["cobertura"].between(0, 1).all()


def test_coverage_por_decil_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes desiguais"):
        cf.coverage_por_decil(
            y_true=np.array([0.1, 0.5]),
            y_pred=np.array([0.2]),
            lower=np.array([0.0, 0.4]),
            upper=np.array([0.2, 0.6]),
        )
