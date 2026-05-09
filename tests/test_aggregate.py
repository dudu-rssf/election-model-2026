"""
Testes de `src.aggregation.aggregate`.

Cobre:
  * agregar_municipal_para_uf: média ponderada correta; identidade de
    soma quando shares de input somam 1; agregação de y_true; partido
    faltando em alguns muns (caso prefeito).
  * agregar_uf_para_nacional: média ponderada correta; idempotência
    quando há 1 só UF.
  * Monte Carlo: cobertura empírica ≈ 1-α em dados sintéticos com
    intervalos honestos; intervalos contidos em [0,1] após clip.
  * verificar_soma_unitaria: detecta violadores além da tolerância.
  * cobertura_agregada: contagem correta de y_real dentro do intervalo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.aggregation import aggregate as agg


def _preds_sinteticos(
    seed: int = 0,
    *,
    n_munis_por_uf: int = 50,
    ufs: tuple = ("AA", "BB", "CC"),
    partidos: tuple = ("P1", "P2", "P3"),
    ano: int = 2022,
    margem: float = 0.05,
) -> pd.DataFrame:
    """Gera preds sintéticos com shares somando 1 por município."""
    rng = np.random.default_rng(seed)
    rows = []
    for uf in ufs:
        for m in range(n_munis_por_uf):
            mun_id = f"{uf}{m:03d}"
            eleitorado = float(rng.integers(1_000, 100_000))
            shares = rng.dirichlet(np.ones(len(partidos)))
            for p, s in zip(partidos, shares):
                rows.append({
                    "ano_presidencial": ano,
                    "id_municipio": mun_id,
                    "sigla_uf": uf,
                    "sigla_partido": p,
                    "total_votos_mun": eleitorado,
                    "pred_LightGBM_v1": float(s),
                    "pred_lower": float(np.clip(s - margem, 0.0, 1.0)),
                    "pred_upper": float(np.clip(s + margem, 0.0, 1.0)),
                    "y_true": float(s),
                })
    return pd.DataFrame(rows)


def test_agregar_municipal_para_uf_media_ponderada() -> None:
    df = pd.DataFrame({
        "ano_presidencial": [2022] * 4,
        "id_municipio": ["M1", "M1", "M2", "M2"],
        "sigla_uf": ["SP"] * 4,
        "sigla_partido": ["PT", "PL", "PT", "PL"],
        "total_votos_mun": [100, 100, 300, 300],
        "pred_LightGBM_v1": [0.6, 0.4, 0.2, 0.8],
    })
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
        incluir_y_true=False,
    )
    pt = out[out["sigla_partido"] == "PT"].iloc[0]
    pl = out[out["sigla_partido"] == "PL"].iloc[0]
    assert pt["share_pred"] == pytest.approx(0.30)
    assert pl["share_pred"] == pytest.approx(0.70)
    assert pt["eleitorado_uf"] == 400
    assert pt["n_municipios_partido"] == 2
    assert pt["n_municipios_uf"] == 2


def test_agregar_municipal_para_uf_soma_unitaria() -> None:
    df = _preds_sinteticos(seed=7)
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
    )
    res = agg.verificar_soma_unitaria(
        out, keys=["ano_presidencial", "sigla_uf"], tolerancia=1e-9,
    )
    assert res.n_violacoes == 0
    assert res.soma_min == pytest.approx(1.0)
    assert res.soma_max == pytest.approx(1.0)


def test_agregar_municipal_para_uf_partido_faltando_em_muns() -> None:
    """Quando partido não compete em todos os muns (caso prefeito), o
    denominador é o eleitorado TOTAL da UF."""
    df = pd.DataFrame({
        "ano_presidencial": [2022] * 3,
        "id_municipio": ["M1", "M1", "M2"],
        "sigla_uf": ["SP"] * 3,
        "sigla_partido": ["PT", "PL", "PT"],
        "total_votos_mun": [100, 100, 300],
        "pred_LightGBM_v1": [0.6, 0.4, 1.0],
    })
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
        incluir_y_true=False,
    )
    pt = out[out["sigla_partido"] == "PT"].iloc[0]
    pl = out[out["sigla_partido"] == "PL"].iloc[0]
    # eleitorado_uf = 100 + 300 = 400 (dedup por mun)
    assert pt["eleitorado_uf"] == 400
    assert pl["eleitorado_uf"] == 400
    # PT: (100*0.6 + 300*1.0) / 400 = 0.90
    assert pt["share_pred"] == pytest.approx(0.90)
    # PL: (100*0.4) / 400 = 0.10 (denominador total!)
    assert pl["share_pred"] == pytest.approx(0.10)
    assert (pt["share_pred"] + pl["share_pred"]) == pytest.approx(1.0)
    assert pt["n_municipios_partido"] == 2
    assert pl["n_municipios_partido"] == 1
    assert pt["n_municipios_uf"] == 2
    assert pl["n_municipios_uf"] == 2


def test_agregar_municipal_para_uf_y_true() -> None:
    df = pd.DataFrame({
        "ano_presidencial": [2022] * 2,
        "id_municipio": ["M1", "M2"],
        "sigla_uf": ["SP", "SP"],
        "sigla_partido": ["PT", "PT"],
        "total_votos_mun": [100, 400],
        "pred_LightGBM_v1": [0.5, 0.5],
        "y_true": [0.6, 0.2],
    })
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
    )
    assert out.iloc[0]["y_real"] == pytest.approx(0.28)


def test_agregar_municipal_para_uf_intervalos_contem_pontual() -> None:
    df = _preds_sinteticos(seed=13, margem=0.10)
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
        pred_lower_col="pred_lower", pred_upper_col="pred_upper",
        n_samples=500, alpha=0.10, seed=42,
    )
    sub = out.dropna(subset=["share_lower", "share_upper"])
    fora = sub[(sub["share_pred"] < sub["share_lower"]) |
               (sub["share_pred"] > sub["share_upper"])]
    assert len(fora) / len(sub) <= 0.01


def test_agregar_intervalos_dentro_de_zero_um() -> None:
    df = _preds_sinteticos(seed=21)
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
        pred_lower_col="pred_lower", pred_upper_col="pred_upper",
        n_samples=200, seed=1,
    )
    sub = out.dropna(subset=["share_lower", "share_upper"])
    assert (sub["share_lower"] >= 0.0).all()
    assert (sub["share_upper"] <= 1.0).all()
    assert (sub["share_lower"] <= sub["share_upper"]).all()


def test_monte_carlo_cobertura_marginal() -> None:
    rng = np.random.default_rng(2024)
    rows = []
    for u in range(20):
        share_real_uf = rng.uniform(0.2, 0.8)
        for m in range(30):
            real = float(np.clip(share_real_uf + rng.normal(0, 0.02), 0, 1))
            err = float(rng.normal(0, 0.05))
            pred = float(np.clip(real + err, 0, 1))
            margem = 0.10
            rows.append({
                "ano_presidencial": 2022,
                "id_municipio": f"U{u:02d}M{m:03d}",
                "sigla_uf": f"UF{u:02d}",
                "sigla_partido": "P1",
                "total_votos_mun": float(rng.integers(5_000, 50_000)),
                "pred_LightGBM_v1": pred,
                "pred_lower": float(np.clip(pred - margem, 0, 1)),
                "pred_upper": float(np.clip(pred + margem, 0, 1)),
                "y_true": real,
            })
    df = pd.DataFrame(rows)
    out = agg.agregar_municipal_para_uf(
        df, peso_col="total_votos_mun",
        pred_col="pred_LightGBM_v1", ano_col="ano_presidencial",
        pred_lower_col="pred_lower", pred_upper_col="pred_upper",
        n_samples=1000, alpha=0.10, seed=0,
    )
    cob = agg.cobertura_agregada(out)
    assert cob >= 0.85


def test_agregar_uf_para_nacional_media_ponderada() -> None:
    df_uf = pd.DataFrame({
        "ano_presidencial": [2022] * 2,
        "sigla_uf": ["SP", "MG"],
        "sigla_partido": ["PT", "PT"],
        "share_pred": [0.40, 0.55],
        "eleitorado_uf": [10_000, 5_000],
    })
    out = agg.agregar_uf_para_nacional(
        df_uf, peso_col="eleitorado_uf", share_col="share_pred",
        share_lower_col=None, share_upper_col=None,
        incluir_y_real=False,
    )
    assert len(out) == 1
    assert out.iloc[0]["share_pred"] == pytest.approx(0.45)
    assert out.iloc[0]["eleitorado_total"] == 15_000
    assert out.iloc[0]["n_ufs"] == 2


def test_agregar_uf_para_nacional_idempotente_uma_uf() -> None:
    df_uf = pd.DataFrame({
        "ano_presidencial": [2022, 2022],
        "sigla_uf": ["SP", "SP"],
        "sigla_partido": ["PT", "PL"],
        "share_pred": [0.4, 0.6],
        "eleitorado_uf": [10_000, 10_000],
        "share_lower": [0.35, 0.55],
        "share_upper": [0.45, 0.65],
    })
    out = agg.agregar_uf_para_nacional(
        df_uf, peso_col="eleitorado_uf", share_col="share_pred",
        ano_col="ano_presidencial", incluir_y_real=False,
        n_samples=200, seed=0,
    )
    pt = out[out["sigla_partido"] == "PT"].iloc[0]
    pl = out[out["sigla_partido"] == "PL"].iloc[0]
    assert pt["share_pred"] == pytest.approx(0.4)
    assert pl["share_pred"] == pytest.approx(0.6)


def test_agregar_uf_para_nacional_partido_em_subset_de_ufs() -> None:
    """Partido só aparece em algumas UFs — denominador deve ser
    eleitorado total nacional."""
    df_uf = pd.DataFrame({
        "ano_presidencial": [2022] * 3,
        "sigla_uf": ["SP", "MG", "RJ"],
        "sigla_partido": ["X", "X", "Y"],  # X em SP+MG, Y só em RJ
        "share_pred": [0.5, 0.4, 1.0],
        "eleitorado_uf": [10_000, 5_000, 4_000],
    })
    out = agg.agregar_uf_para_nacional(
        df_uf, peso_col="eleitorado_uf", share_col="share_pred",
        share_lower_col=None, share_upper_col=None,
        incluir_y_real=False,
    )
    # eleitorado_total = 10k + 5k + 4k = 19k (mesmo pra X e Y)
    assert (out["eleitorado_total"] == 19_000).all()
    # X: (10k*0.5 + 5k*0.4) / 19k = 7000/19000 = 0.3684
    x = out[out["sigla_partido"] == "X"].iloc[0]
    assert x["share_pred"] == pytest.approx(7000/19000)
    # Y: (4k*1.0) / 19k = 0.2105
    y = out[out["sigla_partido"] == "Y"].iloc[0]
    assert y["share_pred"] == pytest.approx(4000/19000)


def test_verificar_soma_unitaria_dentro_tolerancia() -> None:
    df = pd.DataFrame({
        "ano": [2022, 2022, 2022],
        "uf": ["SP", "SP", "SP"],
        "partido": ["PT", "PL", "MDB"],
        "share_pred": [0.4, 0.59, 0.005],
    })
    res = agg.verificar_soma_unitaria(df, keys=["ano", "uf"], tolerancia=0.01)
    assert res.n_grupos == 1
    assert res.n_violacoes == 0
    assert res.soma_min == pytest.approx(0.995)


def test_verificar_soma_unitaria_violacao() -> None:
    df = pd.DataFrame({
        "ano": [2022, 2022, 2022, 2022],
        "uf": ["SP", "SP", "RJ", "RJ"],
        "partido": ["PT", "PL", "PT", "PL"],
        "share_pred": [0.5, 0.49, 0.4, 0.4],
    })
    res = agg.verificar_soma_unitaria(df, keys=["ano", "uf"], tolerancia=0.05)
    assert res.n_violacoes == 1
    assert (res.detalhes["uf"] == "RJ").all()


def test_cobertura_agregada_basica() -> None:
    df = pd.DataFrame({
        "y_real":       [0.10, 0.50, 0.90, 0.30],
        "share_lower":  [0.05, 0.40, 0.85, 0.40],
        "share_upper":  [0.20, 0.60, 0.95, 0.60],
    })
    assert agg.cobertura_agregada(df) == 0.75


def test_cobertura_agregada_ignora_nan() -> None:
    df = pd.DataFrame({
        "y_real":       [0.10, np.nan, 0.30],
        "share_lower":  [0.05, 0.40,   0.40],
        "share_upper":  [0.20, 0.60,   0.60],
    })
    assert agg.cobertura_agregada(df) == 0.5


def test_agregar_falta_coluna_obrigatoria() -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with pytest.raises(ValueError, match="colunas ausentes"):
        agg.agregar_municipal_para_uf(
            df, peso_col="peso", pred_col="pred", ano_col="ano",
        )
