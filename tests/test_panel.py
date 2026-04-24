"""
Testes da Fase 2 — painel mestre.

Usa fixtures com municípios fictícios para testar:
  * Vencedor por eleição + empate (tie-break).
  * Share e margem.
  * Anexo de coligação.
  * Scaffold município × ano_presidencial.
  * Join prefeito→presidencial via PRESIDENCIAL_TO_MUNICIPAL.
  * Tabela long presidencial (shares somam 1 por município/ano).
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.features import panel as panel_mod
from src.features import target as target_mod
from src.ingestion.queries import PRESIDENCIAL_TO_MUNICIPAL


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def diretorio():
    return pd.DataFrame([
        {"id_municipio": "3550308", "sigla_uf": "SP", "nome": "São Paulo", "regiao": "Sudeste", "capital_uf": True},
        {"id_municipio": "3500105", "sigla_uf": "SP", "nome": "Adamantina", "regiao": "Sudeste", "capital_uf": False},
        {"id_municipio": "3509502", "sigla_uf": "SP", "nome": "Capão Bonito", "regiao": "Sudeste", "capital_uf": False},
    ])


@pytest.fixture
def prefeito_resultados():
    """Prefeito: eleições municipais 2012 e 2016 (cobrem presidencial 2014 e 2018)."""
    rows = [
        # 2012 — SP: PT 200, PSDB 150, PMDB 50
        {"ano": 2012, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 13, "nome_candidato": "HADDAD",
         "sigla_partido": "PT", "votos": 200},
        {"ano": 2012, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 45, "nome_candidato": "SERRA",
         "sigla_partido": "PSDB", "votos": 150},
        {"ano": 2012, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 15, "nome_candidato": "KASSAB",
         "sigla_partido": "PMDB", "votos": 50},
        # 2012 — Adamantina: empate 100/100 entre PT (13) e PSDB (45)
        {"ano": 2012, "id_municipio": "3500105", "sigla_uf": "SP",
         "numero_candidato": 45, "nome_candidato": "X",
         "sigla_partido": "PSDB", "votos": 100},
        {"ano": 2012, "id_municipio": "3500105", "sigla_uf": "SP",
         "numero_candidato": 13, "nome_candidato": "Y",
         "sigla_partido": "PT", "votos": 100},
        # 2016 — SP: PSDB 300, PT 200
        {"ano": 2016, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 45, "nome_candidato": "DORIA",
         "sigla_partido": "PSDB", "votos": 300},
        {"ano": 2016, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 13, "nome_candidato": "HADDAD",
         "sigla_partido": "PT", "votos": 200},
        # 2016 — Adamantina: PSB 80
        {"ano": 2016, "id_municipio": "3500105", "sigla_uf": "SP",
         "numero_candidato": 40, "nome_candidato": "Z",
         "sigla_partido": "PSB", "votos": 80},
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def partidos_prefeito():
    """Mock da tabela `br_tse_eleicoes.partidos` filtrada por cargo='prefeito'.

    Granularidade: 1 linha por (ano, id_municipio, sigla_partido). Join é
    feito pela SIGLA do partido vencedor (não pelo número do candidato).
    """
    return pd.DataFrame([
        {"ano": 2012, "id_municipio": "3550308", "sigla_partido": "PT",
         "composicao_coligacao": "PT:PCdoB:PSB"},
        {"ano": 2012, "id_municipio": "3500105", "sigla_partido": "PT",
         "composicao_coligacao": "PT"},
        {"ano": 2012, "id_municipio": "3500105", "sigla_partido": "PSDB",
         "composicao_coligacao": "PSDB:DEM"},
        {"ano": 2016, "id_municipio": "3550308", "sigla_partido": "PSDB",
         "composicao_coligacao": "PSDB:DEM:PR"},
        {"ano": 2016, "id_municipio": "3500105", "sigla_partido": "PSB",
         "composicao_coligacao": "PSB:PP"},
    ])


@pytest.fixture
def presidencial_resultados():
    return pd.DataFrame([
        # 2014 SP
        {"ano": 2014, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 13, "nome_candidato": "DILMA",
         "sigla_partido": "PT", "votos": 400},
        {"ano": 2014, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 45, "nome_candidato": "AECIO",
         "sigla_partido": "PSDB", "votos": 600},
        # 2018 SP
        {"ano": 2018, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 17, "nome_candidato": "BOLSONARO",
         "sigla_partido": "PSL", "votos": 700},
        {"ano": 2018, "id_municipio": "3550308", "sigla_uf": "SP",
         "numero_candidato": 13, "nome_candidato": "HADDAD",
         "sigla_partido": "PT", "votos": 300},
    ])


# ============================================================
# prefeito_vencedor_por_eleicao
# ============================================================
def test_vencedor_escolhe_maior_votado(prefeito_resultados):
    v = panel_mod.prefeito_vencedor_por_eleicao(prefeito_resultados)
    sp_2012 = v[(v["ano_eleicao_municipal"] == 2012) & (v["id_municipio"] == "3550308")].iloc[0]
    assert sp_2012["mayor_partido"] == "PT"
    assert sp_2012["mayor_votos"] == 200
    assert sp_2012["mayor_votos_total_mun"] == 400  # 200+150+50
    assert sp_2012["mayor_share_1t"] == pytest.approx(0.5)
    assert sp_2012["mayor_margem_1t"] == pytest.approx(50 / 400)  # (200-150)/400


def test_vencedor_empate_resolve_pelo_menor_numero(prefeito_resultados):
    # Adamantina 2012: empate 100/100 PSDB(45) vs PT(13). Vence menor numero (13).
    v = panel_mod.prefeito_vencedor_por_eleicao(prefeito_resultados)
    row = v[(v["ano_eleicao_municipal"] == 2012) & (v["id_municipio"] == "3500105")]
    assert len(row) == 1
    assert int(row.iloc[0]["mayor_numero"]) == 13
    assert row.iloc[0]["mayor_partido"] == "PT"


def test_vencedor_sem_segundo_colocado_margem_igual_share(prefeito_resultados):
    v = panel_mod.prefeito_vencedor_por_eleicao(prefeito_resultados)
    row = v[(v["ano_eleicao_municipal"] == 2016) & (v["id_municipio"] == "3500105")].iloc[0]
    # Único candidato -> margem = share = 1.0
    assert row["mayor_share_1t"] == pytest.approx(1.0)
    assert row["mayor_margem_1t"] == pytest.approx(1.0)


def test_vencedor_falta_coluna():
    df = pd.DataFrame({"ano": [2012], "votos": [100]})
    with pytest.raises(ValueError, match="sem colunas"):
        panel_mod.prefeito_vencedor_por_eleicao(df)


# ============================================================
# anexar_coligacao_prefeito
# ============================================================
def test_anexar_coligacao(prefeito_resultados, partidos_prefeito):
    v = panel_mod.prefeito_vencedor_por_eleicao(prefeito_resultados)
    out = panel_mod.anexar_coligacao_prefeito(v, partidos_prefeito)

    sp_2012 = out[(out["ano_eleicao_municipal"] == 2012) & (out["id_municipio"] == "3550308")].iloc[0]
    assert sp_2012["mayor_coligacao"] == "PT:PCdoB:PSB"

    # 2016 SP: Doria PSDB
    sp_2016 = out[(out["ano_eleicao_municipal"] == 2016) & (out["id_municipio"] == "3550308")].iloc[0]
    assert sp_2016["mayor_coligacao"] == "PSDB:DEM:PR"


def test_anexar_coligacao_sem_partidos_preserva_vencedores(prefeito_resultados):
    v = panel_mod.prefeito_vencedor_por_eleicao(prefeito_resultados)
    out = panel_mod.anexar_coligacao_prefeito(v, pd.DataFrame())
    assert "mayor_coligacao" in out.columns
    assert out["mayor_coligacao"].isna().all()
    # Nenhuma linha perdida
    assert len(out) == len(v)


def test_anexar_coligacao_none_explicito(prefeito_resultados):
    v = panel_mod.prefeito_vencedor_por_eleicao(prefeito_resultados)
    out = panel_mod.anexar_coligacao_prefeito(v, None)
    assert "mayor_coligacao" in out.columns
    assert out["mayor_coligacao"].isna().all()


# ============================================================
# scaffold_municipio_ano
# ============================================================
def test_scaffold_tem_linha_por_cidade_e_ano(diretorio):
    scaff = panel_mod.scaffold_municipio_ano(diretorio, [2014, 2018])
    assert len(scaff) == 6  # 3 cidades × 2 anos
    assert set(scaff["ano_presidencial"]) == {2014, 2018}
    assert set(scaff["id_municipio"]) == set(diretorio["id_municipio"])


def test_scaffold_mapeia_ano_municipal_correto(diretorio):
    scaff = panel_mod.scaffold_municipio_ano(diretorio, [2014, 2018])
    # 2014 -> 2012, 2018 -> 2016
    for _, row in scaff.iterrows():
        assert row["ano_eleicao_municipal"] == PRESIDENCIAL_TO_MUNICIPAL[row["ano_presidencial"]]


def test_scaffold_ano_sem_mapping_levanta(diretorio):
    with pytest.raises(KeyError, match="sem mapping municipal"):
        panel_mod.scaffold_municipio_ano(diretorio, [1990])


# ============================================================
# construir_painel_mestre
# ============================================================
def test_painel_mestre_anexa_prefeito_certo(diretorio, prefeito_resultados, partidos_prefeito):
    painel = panel_mod.construir_painel_mestre(
        diretorio=diretorio,
        df_prefeito=prefeito_resultados,
        df_partidos_prefeito=partidos_prefeito,
        anos_presidenciais=[2014, 2018],
    )
    # SP em 2014 deve ter prefeito vigente eleito em 2012 (PT / Haddad)
    sp_2014 = painel[(painel["id_municipio"] == "3550308") & (painel["ano_presidencial"] == 2014)].iloc[0]
    assert sp_2014["ano_eleicao_municipal"] == 2012
    assert sp_2014["mayor_partido"] == "PT"
    assert sp_2014["mayor_coligacao"] == "PT:PCdoB:PSB"
    # SP em 2018 -> eleição 2016 -> Doria PSDB
    sp_2018 = painel[(painel["id_municipio"] == "3550308") & (painel["ano_presidencial"] == 2018)].iloc[0]
    assert sp_2018["mayor_partido"] == "PSDB"
    assert sp_2018["mayor_coligacao"] == "PSDB:DEM:PR"


def test_painel_mestre_sem_prefeito_fica_na(diretorio, prefeito_resultados, partidos_prefeito):
    painel = panel_mod.construir_painel_mestre(
        diretorio=diretorio,
        df_prefeito=prefeito_resultados,
        df_partidos_prefeito=partidos_prefeito,
        anos_presidenciais=[2014, 2018],
    )
    # Capão Bonito (3509502) não aparece em prefeito → mayor_* deve ser NA
    cb = painel[painel["id_municipio"] == "3509502"]
    assert len(cb) == 2  # 2014 e 2018
    assert cb["mayor_partido"].isna().all()
    assert cb["mayor_coligacao"].isna().all()


# ============================================================
# construir_presidencial_long
# ============================================================
def test_presidencial_long_shares_somam_1(presidencial_resultados):
    out = target_mod.construir_presidencial_long(presidencial_resultados)
    s = out.groupby(["ano_presidencial", "id_municipio"])["share_1t"].sum()
    for v in s.values:
        assert v == pytest.approx(1.0)


def test_presidencial_long_renomeia_ano(presidencial_resultados):
    out = target_mod.construir_presidencial_long(presidencial_resultados)
    assert "ano_presidencial" in out.columns
    assert "ano" not in out.columns


def test_presidencial_long_total_votos_consistente(presidencial_resultados):
    out = target_mod.construir_presidencial_long(presidencial_resultados)
    for (ano, mun), g in out.groupby(["ano_presidencial", "id_municipio"]):
        assert int(g["total_votos_mun"].iloc[0]) == int(g["votos"].sum())
