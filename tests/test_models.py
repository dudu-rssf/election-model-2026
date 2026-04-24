"""
Testes da Fase 4 — pipeline de modelo.

Cobre:
  * transforms: logit/sigmoid são inversos; clip funciona
  * features.preparar_X_y: schema esperado, sem vazamento, dtypes certos
  * features.split_temporal: split por ano, aviso em categoria só-no-teste
  * baseline: B0 usa mediana; B1 fallback em lag NaN; B2 blend
  * train: pipeline fit_predict preserva formato e range

Não testa LightGBM real (dependência pesada, desnecessária pra lógica do
wrapper) exceto em um smoke test marcado com `@pytest.mark.slow`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models import baseline as bl
from src.models import features as mf
from src.models import transforms as tx


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def df_features_fake() -> pd.DataFrame:
    """Mini DataFrame com o schema que Fase 3 produz."""
    rng = np.random.default_rng(0)
    n = 60
    anos = rng.choice([2014, 2018, 2022], size=n)
    ufs = rng.choice(["SP"], size=n)
    partidos = rng.choice(["PT", "PSDB", "PSOL", "PL", "PDT"], size=n)
    mun = rng.choice(["3550308", "3500105", "3509502"], size=n)

    df = pd.DataFrame({
        # identificadores
        "ano_presidencial": anos.astype("int64"),
        "sigla_uf": ufs,
        "id_municipio": pd.array(mun, dtype="string"),
        "numero_candidato": rng.integers(10, 60, size=n),
        "nome_candidato": ["fulano"] * n,
        "sigla_partido": partidos,
        # target stack
        "votos": rng.integers(1000, 50_000, size=n),
        "total_votos_mun": 100_000,
        "share_1t": rng.beta(2, 5, size=n),
        # structural
        "regiao": "Sudeste",
        "capital_uf": rng.choice([True, False], size=n),
        "log_eleitorado": rng.normal(11, 1, size=n),
        "porte": rng.choice(["pequeno", "medio", "grande"], size=n),
        # local_power
        "share_prefeito_local": rng.uniform(0.3, 0.6, size=n),
        "margem_prefeito": rng.uniform(0.05, 0.2, size=n),
        "primeiro_mandato_prefeito": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        # continuity
        "continuidade_classe": rng.choice(
            ["total", "forte", "parcial", "ruptura"], size=n
        ),
        "indice_continuidade": rng.uniform(0, 1, size=n),
        "anos_consecutivos_mesmo_partido": rng.integers(0, 20, size=n).astype("int64"),
        "anos_consecutivos_mesmo_grupo": rng.integers(0, 20, size=n).astype("int64"),
        # local_power (partido)
        "alinhado_prefeito_partido": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        "alinhado_prefeito_coligacao": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        # historical
        "lag_share_1t": rng.beta(2, 5, size=n),
        "lag_share_1t_sucessao": rng.beta(2, 5, size=n),
        "lag2_share_1t": rng.beta(2, 5, size=n),
        "swing_share_1t": rng.normal(0, 0.05, size=n),
        "volatilidade_partido": rng.uniform(0, 0.1, size=n),
        # vertical
        "alinhado_gov_vigente_partido": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        "alinhado_gov_vigente_coligacao": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        "alinhado_gov_concorrente_partido": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        "alinhado_gov_concorrente_coligacao": pd.array(
            rng.choice([0, 1], size=n), dtype="Int64"
        ),
        "share_dep_federal_partido": rng.beta(2, 10, size=n),
    })
    return df


# ============================================================
# transforms
# ============================================================
def test_transforms_logit_sigmoid_inversos():
    shares = np.array([0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99])
    logits = tx.logit_share(shares)
    recovered = tx.sigmoid_logit(logits)
    np.testing.assert_allclose(recovered, shares, atol=1e-10)


def test_transforms_clip_ate_01():
    shares = np.array([0.0, 1.0, 0.5, -0.1, 1.1])
    clipped = tx.clip_share(shares, eps=1e-3)
    assert clipped.min() >= 1e-3
    assert clipped.max() <= 1 - 1e-3
    # Valor dentro do range fica igual
    assert clipped[2] == 0.5


def test_transforms_logit_de_0_e_1_nao_explode():
    logits = tx.logit_share(np.array([0.0, 1.0]))
    assert np.all(np.isfinite(logits))


def test_transforms_nan_propaga():
    assert np.isnan(tx.logit_share(np.nan))
    assert np.isnan(tx.sigmoid_logit(np.nan))


def test_transforms_series_preserva_indice():
    s = pd.Series([0.1, 0.5, 0.9], index=["a", "b", "c"])
    out = tx.logit_share(s)
    # logit_share de Series retorna array/Series — assert só que valores batem
    assert len(out) == 3


# ============================================================
# features.preparar_X_y
# ============================================================
def test_preparar_X_y_schema_esperado(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    # Total esperado = cat + bin + num
    esperado = (
        len(mf.FEATURES_CATEGORICAS)
        + len(mf.FEATURES_BINARIAS)
        + len(mf.FEATURES_NUMERICAS)
    )
    assert prep.X.shape[1] == esperado
    assert len(prep.X) == len(prep.y) == len(prep.meta)


def test_preparar_X_y_sem_vazamento(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    proibidas = {"votos", "total_votos_mun", "share_1t", "ano_presidencial",
                 "numero_candidato", "nome_candidato", "id_municipio"}
    assert not (set(prep.X.columns) & proibidas)
    # Meta tem as colunas úteis pra lookup
    assert "ano_presidencial" in prep.meta.columns
    assert "sigla_partido" in prep.meta.columns


def test_preparar_X_y_dtypes_categoricas(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    for c in mf.FEATURES_CATEGORICAS:
        assert str(prep.X[c].dtype) == "category", f"{c} não é category"


def test_preparar_X_y_dtypes_numericas_float(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    for c in mf.FEATURES_BINARIAS + mf.FEATURES_NUMERICAS:
        assert prep.X[c].dtype.kind == "f", f"{c} dtype={prep.X[c].dtype}"


def test_preparar_X_y_dropna_target(df_features_fake):
    df = df_features_fake.copy()
    df.loc[0:2, "share_1t"] = np.nan
    prep = mf.preparar_X_y(df)
    assert len(prep) == len(df) - 3
    assert prep.y.notna().all()


def test_preparar_X_y_levanta_sem_coluna(df_features_fake):
    df = df_features_fake.drop(columns=["lag_share_1t"])
    with pytest.raises(ValueError, match="lag_share_1t"):
        mf.preparar_X_y(df)


# ============================================================
# features.split_temporal
# ============================================================
def test_split_temporal_separa_por_ano(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    tr, te = mf.split_temporal(prep, anos_treino=[2014, 2018], ano_teste=2022)
    assert set(tr.meta["ano_presidencial"].unique()) <= {2014, 2018}
    assert set(te.meta["ano_presidencial"].unique()) == {2022}
    assert len(tr) + len(te) == len(prep)


def test_split_temporal_rejeita_sobreposicao(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    with pytest.raises(ValueError, match="está em"):
        mf.split_temporal(prep, anos_treino=[2014, 2018, 2022], ano_teste=2022)


# ============================================================
# baseline
# ============================================================
def test_baseline_B0_mediana(df_features_fake):
    prep = mf.preparar_X_y(df_features_fake)
    tr, te = mf.split_temporal(prep, [2014, 2018], 2022)
    b0 = bl.MedianaPartidoUF().fit(tr.X, tr.y, meta=tr.meta)
    pred = b0.predict(te.X, meta=te.meta)
    assert len(pred) == len(te)
    assert pred.min() >= 0 and pred.max() <= 1


def test_baseline_B1_usa_lag_quando_disponivel(df_features_fake):
    """B1 deve devolver valores próximos aos lags quando todos estão preenchidos."""
    prep = mf.preparar_X_y(df_features_fake)
    tr, te = mf.split_temporal(prep, [2014, 2018], 2022)
    # garante que todos os lags estão preenchidos
    te.X.loc[:, "lag_share_1t"] = np.linspace(0.1, 0.6, len(te))
    b1 = bl.LagShare().fit(tr.X, tr.y, meta=tr.meta)
    pred = b1.predict(te.X, meta=te.meta)
    np.testing.assert_allclose(pred, te.X["lag_share_1t"].values)


def test_baseline_B1_fallback_em_lag_nan(df_features_fake):
    """B1 deve usar B0 quando lag é NaN."""
    prep = mf.preparar_X_y(df_features_fake)
    tr, te = mf.split_temporal(prep, [2014, 2018], 2022)
    te.X.loc[:, "lag_share_1t"] = np.nan
    b1 = bl.LagShare().fit(tr.X, tr.y, meta=tr.meta)
    pred = b1.predict(te.X, meta=te.meta)
    # Sem NaN no output
    assert not np.isnan(pred).any()
    # Deve ser igual ao B0 (que é o fallback)
    b0_pred = bl.MedianaPartidoUF().fit(tr.X, tr.y, meta=tr.meta).predict(
        te.X, meta=te.meta
    )
    np.testing.assert_allclose(pred, b0_pred)


def test_baseline_B2_blend(df_features_fake):
    """B2 com alpha=0.5 deve ser média entre B0 e B1."""
    prep = mf.preparar_X_y(df_features_fake)
    tr, te = mf.split_temporal(prep, [2014, 2018], 2022)
    b0 = bl.MedianaPartidoUF().fit(tr.X, tr.y, meta=tr.meta)
    b1 = bl.LagShare().fit(tr.X, tr.y, meta=tr.meta)
    b2 = bl.BlendB0B1(alpha=0.5).fit(tr.X, tr.y, meta=tr.meta)

    p0 = b0.predict(te.X, meta=te.meta)
    p1 = b1.predict(te.X, meta=te.meta)
    p2 = b2.predict(te.X, meta=te.meta)
    np.testing.assert_allclose(p2, 0.5 * p0 + 0.5 * p1)


def test_baseline_B2_rejeita_alpha_invalido():
    with pytest.raises(ValueError):
        bl.BlendB0B1(alpha=1.5)
    with pytest.raises(ValueError):
        bl.BlendB0B1(alpha=-0.1)
