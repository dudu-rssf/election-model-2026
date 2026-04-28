"""
Testes da Fase 4.5 — modelo de prefeito (eixo municipal).

Cobre:
  * `target_prefeito.construir_prefeito_long`
  * `panel.construir_painel_mestre_municipal` + `scaffold_municipio_ano_municipal`
  * Versão parametrizada (ano_col) dos módulos compartilhados:
    structural, historical, local_power, vertical, continuity.
  * `src.models.features_prefeito.preparar_X_y` + `split_temporal`.
  * Smoke-import dos scripts/03_features_prefeito.py e scripts/04_train_prefeito.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    continuity,
    historical,
    local_power,
    structural,
    vertical,
)
from src.features.panel import (
    MUNICIPAL_TO_MUNICIPAL_ANTERIOR,
    construir_painel_mestre_municipal,
    scaffold_municipio_ano_municipal,
)
from src.features.target_prefeito import construir_prefeito_long
from src.features.vertical import MUNICIPAL_TO_ESTADUAL_ANTERIOR
from src.models import features_prefeito as mfp


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def diretorio() -> pd.DataFrame:
    return pd.DataFrame([
        {"id_municipio": "3550308", "sigla_uf": "SP", "nome": "São Paulo",
         "regiao": "Sudeste", "capital_uf": True},
        {"id_municipio": "3500105", "sigla_uf": "SP", "nome": "Adamantina",
         "regiao": "Sudeste", "capital_uf": False},
    ])


@pytest.fixture
def df_prefeito() -> pd.DataFrame:
    """Bruto TSE: 2 municípios × 4 eleições municipais × 2 candidatos por eleição."""
    rows = []
    cenarios = {
        "3550308": [(2012, "PT", 13), (2016, "PSDB", 45), (2020, "PSDB", 45), (2024, "PT", 13)],
        "3500105": [(2012, "PT", 13), (2016, "PT", 13), (2020, "PT", 13), (2024, "PSDB", 45)],
    }
    for mun, lista in cenarios.items():
        for ano, partido_v, num_v in lista:
            rows.append({
                "ano": ano, "id_municipio": mun, "sigla_uf": "SP",
                "numero_candidato": num_v, "nome_candidato": f"vencedor-{partido_v}-{ano}",
                "sigla_partido": partido_v, "votos": 800,
            })
            rows.append({
                "ano": ano, "id_municipio": mun, "sigla_uf": "SP",
                "numero_candidato": 99, "nome_candidato": f"perdedor-{ano}",
                "sigla_partido": "OUTRO", "votos": 200,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def df_candidatos_prefeito(df_prefeito: pd.DataFrame) -> pd.DataFrame:
    """Tabela candidatos análoga à de presidencial, mas com chave municipal."""
    return (
        df_prefeito[["ano", "sigla_uf", "id_municipio", "numero_candidato", "nome_candidato"]]
        .drop_duplicates()
        .copy()
    )


@pytest.fixture
def df_partidos_prefeito(df_prefeito: pd.DataFrame) -> pd.DataFrame:
    """Composição de coligação por (ano, mun, partido)."""
    return pd.DataFrame({
        "ano": df_prefeito["ano"],
        "id_municipio": df_prefeito["id_municipio"],
        "sigla_uf": df_prefeito["sigla_uf"],
        "sigla_partido": df_prefeito["sigla_partido"],
        "composicao_coligacao": "PT/PCdoB:PSDB",  # qualquer string razoável
    }).drop_duplicates()


@pytest.fixture
def painel_municipal(diretorio, df_prefeito, df_partidos_prefeito) -> pd.DataFrame:
    return construir_painel_mestre_municipal(
        diretorio=diretorio,
        df_prefeito=df_prefeito,
        df_partidos_prefeito=df_partidos_prefeito,
        anos_municipais=[2016, 2020, 2024],
    )


@pytest.fixture
def prefeito_long(df_prefeito, df_candidatos_prefeito) -> pd.DataFrame:
    return construir_prefeito_long(df_prefeito, df_candidatos_prefeito)


# ============================================================
# target_prefeito
# ============================================================
def test_prefeito_long_shares_somam_1_por_mun_ano(prefeito_long):
    grp = prefeito_long.groupby(["ano_municipal", "id_municipio"])["share_1t"].sum()
    np.testing.assert_allclose(grp.values, 1.0, atol=1e-9)


def test_prefeito_long_eixo_municipal_e_total_consistente(prefeito_long):
    assert "ano_municipal" in prefeito_long.columns
    assert "ano_presidencial" not in prefeito_long.columns
    # total_votos_mun é o mesmo para cada (ano, mun)
    grp = prefeito_long.groupby(["ano_municipal", "id_municipio"])["total_votos_mun"].nunique()
    assert (grp == 1).all()


def test_prefeito_long_inclui_nome_candidato(prefeito_long):
    # quando df_candidatos é fornecido, nome_candidato vem dele
    assert prefeito_long["nome_candidato"].notna().all()


# ============================================================
# panel: scaffold + painel_municipal
# ============================================================
def test_scaffold_municipal_anos_corretos(diretorio):
    sc = scaffold_municipio_ano_municipal(diretorio, anos_municipais=[2020, 2024])
    assert set(sc["ano_municipal"].unique()) == {2020, 2024}
    assert "ano_eleicao_municipal_anterior" in sc.columns
    # X-4 mapping
    sub_2020 = sc[sc["ano_municipal"] == 2020]
    sub_2024 = sc[sc["ano_municipal"] == 2024]
    assert (sub_2020["ano_eleicao_municipal_anterior"] == 2016).all()
    assert (sub_2024["ano_eleicao_municipal_anterior"] == 2020).all()


def test_scaffold_levanta_em_ano_sem_mapping(diretorio):
    # 2030 não está em MUNICIPAL_TO_MUNICIPAL_ANTERIOR
    with pytest.raises(KeyError):
        scaffold_municipio_ano_municipal(diretorio, anos_municipais=[2030])


def test_painel_municipal_anexa_prefeito_de_x_minus_4(painel_municipal):
    # 3550308: vencedor 2012=PT, 2016=PSDB, 2020=PSDB → painel 2024 deve ter PSDB (vigente, eleito em 2020)
    sub = painel_municipal[
        (painel_municipal["id_municipio"] == "3550308") & (painel_municipal["ano_municipal"] == 2024)
    ]
    assert len(sub) == 1
    assert sub["mayor_partido"].iloc[0] == "PSDB"
    assert sub["ano_eleicao_municipal_anterior"].iloc[0] == 2020


def test_painel_municipal_eixo_municipal(painel_municipal):
    assert "ano_municipal" in painel_municipal.columns
    assert "ano_presidencial" not in painel_municipal.columns


# ============================================================
# structural (parametrizado)
# ============================================================
def test_structural_aceita_ano_municipal(painel_municipal, prefeito_long):
    out = structural.features_structural(
        painel_municipal, prefeito_long, ano_col="ano_municipal"
    )
    assert "ano_municipal" in out.columns
    assert "log_eleitorado" in out.columns
    # 1 linha por (ano, mun)
    assert out.duplicated(subset=["ano_municipal", "id_municipio"]).sum() == 0


# ============================================================
# local_power (parametrizado)
# ============================================================
def test_local_power_features_local_aceita_municipal(painel_municipal):
    out = local_power.features_local_mun_ano(
        painel_municipal,
        ano_col="ano_municipal",
        ano_eleicao_anterior_col="ano_eleicao_municipal_anterior",
    )
    assert "ano_municipal" in out.columns
    assert {"share_prefeito_local", "margem_prefeito", "primeiro_mandato_prefeito"} <= set(out.columns)


def test_local_power_alinhamento_partido_municipal(painel_municipal):
    out = local_power.alinhamento_partido_com_prefeito(
        painel_municipal, partidos=["PT", "PSDB"], ano_col="ano_municipal"
    )
    assert "ano_municipal" in out.columns
    assert set(out["sigla_partido"].unique()) == {"PT", "PSDB"}


# ============================================================
# historical (parametrizado)
# ============================================================
def test_historical_aceita_ano_municipal(prefeito_long):
    out = historical.features_historical(
        prefeito_long, anos=[2020, 2024], ano_col="ano_municipal"
    )
    assert "ano_municipal" in out.columns
    assert {"lag_share_1t", "swing_share_1t", "volatilidade_partido"} <= set(out.columns)


# ============================================================
# continuity (parametrizado para X-4)
# ============================================================
def test_continuity_features_municipal_olha_x_minus_4(df_prefeito, df_partidos_prefeito):
    out = continuity.features_continuity(
        df_prefeito, df_partidos_prefeito,
        anos_alvo=[2024],
        ano_col="ano_municipal",
        map_ano_para_municipal=MUNICIPAL_TO_MUNICIPAL_ANTERIOR,
    )
    assert "ano_municipal" in out.columns
    # 3550308: 2020 PSDB, 2016 PSDB → forte (na linha de 2024 olhando pra eleição 2020)
    sub_sp = out[out["id_municipio"] == "3550308"]
    assert sub_sp["continuidade_classe"].iloc[0] in {"forte", "total"}
    # 3500105: 2020 PT, 2016 PT → forte (na linha de 2024 olhando pra eleição 2020)
    sub_ad = out[out["id_municipio"] == "3500105"]
    assert sub_ad["continuidade_classe"].iloc[0] in {"forte", "total"}


def test_continuity_backward_compat_anos_presidenciais(df_prefeito, df_partidos_prefeito):
    """Chamada legada com anos_presidenciais= ainda funciona (Fase 3)."""
    out = continuity.features_continuity(
        df_prefeito, df_partidos_prefeito, anos_presidenciais=[2018, 2022],
    )
    assert "ano_presidencial" in out.columns


# ============================================================
# vertical (parametrizado)
# ============================================================
def test_vertical_governador_municipal_sem_concorrente(painel_municipal):
    """No eixo municipal só existe vigente (sem eleição estadual concorrente)."""
    # painel precisa de gov_*_partido/coligacao para a função funcionar — usa mock vazio
    df_gov = pd.DataFrame({
        "ano": [2018, 2018, 2022, 2022],
        "id_municipio": ["3550308", "3500105", "3550308", "3500105"],
        "sigla_uf": ["SP"] * 4,
        "numero_candidato": [13, 13, 45, 45],
        "sigla_partido": ["PT", "PT", "PSDB", "PSDB"],
        "votos": [1000, 800, 1200, 900],
        "turno": [1] * 4,
    })
    df_part_gov = pd.DataFrame({
        "ano": [2018, 2018, 2022, 2022],
        "id_municipio": ["3550308", "3500105", "3550308", "3500105"],
        "sigla_uf": ["SP"] * 4,
        "sigla_partido": ["PT", "PT", "PSDB", "PSDB"],
        "composicao_coligacao": ["PT/PCdoB"] * 2 + ["PSDB/MDB"] * 2,
    })
    out = vertical.alinhamento_partido_com_governador(
        painel_municipal, df_gov, df_part_gov,
        partidos=["PT", "PSDB"],
        ano_col="ano_municipal",
        map_vigente=MUNICIPAL_TO_ESTADUAL_ANTERIOR,
        incluir_concorrente=False,
    )
    # Não deve haver colunas concorrente
    assert not any(c.startswith("alinhado_gov_concorrente") for c in out.columns)
    # Mas deve ter as vigentes
    assert {"alinhado_gov_vigente_partido", "alinhado_gov_vigente_coligacao"} <= set(out.columns)
    assert "ano_municipal" in out.columns


def test_vertical_dep_federal_municipal_usa_x_minus_2(painel_municipal):
    """Para ano_municipal X, dep. federal = eleição federal em X-2.

    Share é calculado por (ano, mun, partido) / total(ano, mun).
    """
    df_dep = pd.DataFrame({
        "ano": [2018, 2018, 2018, 2018, 2022, 2022, 2022, 2022],
        "id_municipio": ["3550308", "3550308", "3500105", "3500105"] * 2,
        "sigla_partido": ["PT", "PSDB", "PT", "PSDB"] * 2,
        "votos": [1000, 500, 600, 400, 800, 1200, 700, 300],
    })
    out = vertical.share_dep_federal_por_partido(
        df_dep, anos_alvo=[2020, 2024],
        ano_col="ano_municipal",
        map_ano_para_federal=MUNICIPAL_TO_ESTADUAL_ANTERIOR,
    )
    assert "ano_municipal" in out.columns
    assert set(out["ano_municipal"].unique()) == {2020, 2024}
    # 2020 mapeia pra 2018; 2024 mapeia pra 2022.
    sub_2020 = out[(out["ano_municipal"] == 2020) & (out["id_municipio"] == "3550308")]
    sub_2024 = out[(out["ano_municipal"] == 2024) & (out["id_municipio"] == "3550308")]
    # SP em 2018: PT 1000/(1000+500) = 0.667; em 2022: PT 800/(800+1200) = 0.4
    np.testing.assert_allclose(
        sub_2020[sub_2020["sigla_partido"] == "PT"]["share_dep_federal_partido"].iloc[0],
        1000 / 1500,
    )
    np.testing.assert_allclose(
        sub_2024[sub_2024["sigla_partido"] == "PT"]["share_dep_federal_partido"].iloc[0],
        800 / 2000,
    )


# ============================================================
# src.models.features_prefeito
# ============================================================
def _df_features_municipal_fake() -> pd.DataFrame:
    """DataFrame parecido com a saída de scripts/03_features_prefeito.py."""
    rng = np.random.default_rng(42)
    rows = []
    for ano in [2016, 2020, 2024]:
        for mun in ["3550308", "3500105"]:
            for partido in ["PT", "PSDB", "MDB"]:
                rows.append({
                    "ano_municipal": ano,
                    "id_municipio": mun,
                    "sigla_partido": partido,
                    "sigla_uf": "SP",
                    "regiao": "Sudeste",
                    "capital_uf": (mun == "3550308"),
                    "porte": "grande" if mun == "3550308" else "pequeno",
                    "continuidade_classe": rng.choice(["forte", "ruptura", "parcial"]),
                    "log_eleitorado": float(rng.uniform(8, 14)),
                    "share_prefeito_local": float(rng.uniform(0.3, 0.7)),
                    "margem_prefeito": float(rng.uniform(0.0, 0.4)),
                    "indice_continuidade": float(rng.uniform(0, 1)),
                    "anos_consecutivos_mesmo_partido": int(rng.integers(0, 16)),
                    "anos_consecutivos_mesmo_grupo": int(rng.integers(0, 20)),
                    "lag_share_1t": float(rng.uniform(0.1, 0.6)),
                    "lag_share_1t_sucessao": float(rng.uniform(0.1, 0.6)),
                    "lag2_share_1t": float(rng.uniform(0.1, 0.6)),
                    "swing_share_1t": float(rng.normal(0, 0.05)),
                    "volatilidade_partido": float(rng.uniform(0.01, 0.1)),
                    "share_dep_federal_partido": float(rng.uniform(0.1, 0.5)),
                    "primeiro_mandato_prefeito": int(rng.integers(0, 2)),
                    "alinhado_prefeito_partido": int(rng.integers(0, 2)),
                    "alinhado_prefeito_coligacao": int(rng.integers(0, 2)),
                    "alinhado_gov_vigente_partido": int(rng.integers(0, 2)),
                    "alinhado_gov_vigente_coligacao": int(rng.integers(0, 2)),
                    "votos": int(rng.integers(100, 5000)),
                    "total_votos_mun": int(rng.integers(5000, 10000)),
                    "numero_candidato": int(rng.integers(10, 99)),
                    "nome_candidato": "X",
                    "share_1t": float(rng.uniform(0.1, 0.6)),
                })
    return pd.DataFrame(rows)


def test_features_prefeito_preparar_X_y_schema():
    df = _df_features_municipal_fake()
    prep = mfp.preparar_X_y(df)
    # contagem: 5 cat + 6 bin + 12 num = 23
    assert prep.X.shape[1] == 23
    assert "ano_municipal" not in prep.X.columns  # eixo nunca é feature
    assert "ano_municipal" in prep.meta.columns
    assert "share_1t" not in prep.X.columns


def test_features_prefeito_sem_governador_concorrente():
    df = _df_features_municipal_fake()
    prep = mfp.preparar_X_y(df)
    bins_concorrente = [c for c in prep.X.columns if "concorrente" in c]
    assert bins_concorrente == [], f"Eixo municipal não deve ter governador concorrente; achei: {bins_concorrente}"


def test_features_prefeito_sem_vazamento():
    df = _df_features_municipal_fake()
    prep = mfp.preparar_X_y(df)
    proibidas = {"votos", "total_votos_mun", "share_1t", "ano_municipal", "id_municipio"}
    assert proibidas.isdisjoint(set(prep.X.columns))


def test_features_prefeito_split_temporal_separa_por_ano_municipal():
    df = _df_features_municipal_fake()
    prep = mfp.preparar_X_y(df)
    tr, te = mfp.split_temporal(prep, anos_treino=[2016, 2020], ano_teste=2024)
    assert (tr.meta["ano_municipal"].isin([2016, 2020])).all()
    assert (te.meta["ano_municipal"] == 2024).all()
    assert len(tr) + len(te) == len(prep)


def test_features_prefeito_split_temporal_rejeita_sobreposicao():
    df = _df_features_municipal_fake()
    prep = mfp.preparar_X_y(df)
    with pytest.raises(ValueError):
        mfp.split_temporal(prep, anos_treino=[2020, 2024], ano_teste=2024)


# ============================================================
# Smoke imports dos scripts
# ============================================================
def test_script_03_prefeito_importa():
    """O script deve importar sem efeitos colaterais."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_s03p", Path(__file__).resolve().parent.parent / "scripts" / "03_features_prefeito.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
    assert mod.ANO_COL == "ano_municipal"


def test_script_04_prefeito_importa():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_s04p", Path(__file__).resolve().parent.parent / "scripts" / "04_train_prefeito.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert hasattr(mod, "main")
