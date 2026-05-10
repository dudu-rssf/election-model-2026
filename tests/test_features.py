"""
Testes da Fase 3 — feature engineering.

Cobre os 5 módulos de `src/features/` + ponta-a-ponta com município
fictício e transições conhecidas.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import (
    coligacao,
    continuity,
    historical,
    local_power,
    structural,
    vertical,
)
from src.features.panel import construir_painel_mestre
from src.features.target import construir_presidencial_long


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture
def diretorio():
    return pd.DataFrame([
        {"id_municipio": "3550308", "sigla_uf": "SP", "nome": "São Paulo",
         "regiao": "Sudeste", "capital_uf": True},
        {"id_municipio": "3500105", "sigla_uf": "SP", "nome": "Adamantina",
         "regiao": "Sudeste", "capital_uf": False},
        {"id_municipio": "3509502", "sigla_uf": "SP", "nome": "Capão Bonito",
         "regiao": "Sudeste", "capital_uf": False},
    ])


@pytest.fixture
def df_prefeito():
    """2012 e 2016 em SP e Adamantina."""
    rows = []
    # SP 2012: PT(13) 200, PSDB(45) 150, PMDB(15) 50
    for num, part, v in [(13, "PT", 200), (45, "PSDB", 150), (15, "PMDB", 50)]:
        rows.append({"ano": 2012, "sigla_uf": "SP", "id_municipio": "3550308",
                     "numero_candidato": num, "nome_candidato": "X",
                     "sigla_partido": part, "votos": v})
    # SP 2016: PSDB(45) 300, PT(13) 200
    for num, part, v in [(45, "PSDB", 300), (13, "PT", 200)]:
        rows.append({"ano": 2016, "sigla_uf": "SP", "id_municipio": "3550308",
                     "numero_candidato": num, "nome_candidato": "Y",
                     "sigla_partido": part, "votos": v})
    # Adamantina 2012: empate PSDB(45)/PT(13) 100/100 — tie-break vai para 13
    for num, part, v in [(45, "PSDB", 100), (13, "PT", 100)]:
        rows.append({"ano": 2012, "sigla_uf": "SP", "id_municipio": "3500105",
                     "numero_candidato": num, "nome_candidato": "Z",
                     "sigla_partido": part, "votos": v})
    # Adamantina 2016: PSB(40) único
    rows.append({"ano": 2016, "sigla_uf": "SP", "id_municipio": "3500105",
                 "numero_candidato": 40, "nome_candidato": "W",
                 "sigla_partido": "PSB", "votos": 80})
    return pd.DataFrame(rows)


@pytest.fixture
def df_partidos_prefeito():
    """Mock da tabela `partidos` com chave (ano, mun, sigla_partido)."""
    return pd.DataFrame([
        {"ano": 2012, "id_municipio": "3550308", "sigla_partido": "PT",
         "composicao_coligacao": "PT:PCdoB:PSB"},
        {"ano": 2012, "id_municipio": "3500105", "sigla_partido": "PT",
         "composicao_coligacao": "PT"},
        {"ano": 2016, "id_municipio": "3550308", "sigla_partido": "PSDB",
         "composicao_coligacao": "PSDB:DEM:PR"},
        {"ano": 2016, "id_municipio": "3500105", "sigla_partido": "PSB",
         "composicao_coligacao": "PSB:PP"},
    ])


@pytest.fixture
def df_pres():
    """Presidencial 2014 e 2018 em SP, Adamantina e Capão Bonito."""
    rows = []
    for ano, shares in [
        (2014, {"3550308": {"PT": 400, "PSDB": 600},
                "3500105": {"PT": 300, "PSDB": 200},
                "3509502": {"PT": 150, "PSDB": 100}}),
        (2018, {"3550308": {"PT": 300, "PSL": 700},
                "3500105": {"PT": 100, "PSL": 400},
                "3509502": {"PT": 50, "PSL": 200}}),
    ]:
        for mun, votos_por_partido in shares.items():
            for part, v in votos_por_partido.items():
                num = {"PT": 13, "PSDB": 45, "PSL": 17}[part]
                rows.append({"ano": ano, "sigla_uf": "SP", "id_municipio": mun,
                             "numero_candidato": num, "nome_candidato": part,
                             "sigla_partido": part, "votos": v})
    return pd.DataFrame(rows)


@pytest.fixture
def painel(diretorio, df_prefeito, df_partidos_prefeito):
    return construir_painel_mestre(
        diretorio=diretorio,
        df_prefeito=df_prefeito,
        df_partidos_prefeito=df_partidos_prefeito,
        anos_presidenciais=[2014, 2018],
    )


@pytest.fixture
def pres_long(df_pres):
    return construir_presidencial_long(df_pres)


# ============================================================
# structural
# ============================================================
def test_structural_tem_uma_linha_por_mun_ano(painel, pres_long):
    s = structural.features_structural(painel, pres_long)
    assert len(s) == 6  # 3 mun × 2 anos
    assert set(s.columns) >= {
        "ano_presidencial", "id_municipio", "sigla_uf", "regiao",
        "capital_uf", "log_eleitorado", "porte",
    }


def test_structural_porte_tem_3_categorias_no_max(painel, pres_long):
    s = structural.features_structural(painel, pres_long)
    niveis = {"pequeno", "medio", "grande"}
    valores = set(s["porte"].dropna().unique())
    assert valores <= niveis
    # log_eleitorado monotônico crescente com votos
    sp_14 = s[(s["id_municipio"] == "3550308") & (s["ano_presidencial"] == 2014)].iloc[0]
    cb_14 = s[(s["id_municipio"] == "3509502") & (s["ano_presidencial"] == 2014)].iloc[0]
    assert sp_14["log_eleitorado"] > cb_14["log_eleitorado"]


# ============================================================
# local_power
# ============================================================
def test_local_power_bloco_A(painel):
    a = local_power.features_local_mun_ano(painel)
    assert len(a) == len(painel)
    # Capão Bonito não tem prefeito → NA
    cb = a[a["id_municipio"] == "3509502"]
    assert cb["share_prefeito_local"].isna().all()
    assert cb["margem_prefeito"].isna().all()


def test_local_power_primeiro_mandato(painel):
    a = local_power.features_local_mun_ano(painel)
    # SP: 2012 PT, 2016 PSDB → em 2018 presidencial, primeiro_mandato=1
    sp18 = a[(a["id_municipio"] == "3550308") & (a["ano_presidencial"] == 2018)].iloc[0]
    assert sp18["primeiro_mandato_prefeito"] == 1
    # SP 2014 → não tem histórico anterior a 2012 → NA
    sp14 = a[(a["id_municipio"] == "3550308") & (a["ano_presidencial"] == 2014)].iloc[0]
    assert pd.isna(sp14["primeiro_mandato_prefeito"])


def test_local_power_alinhamento_partido(painel, pres_long):
    partidos = pres_long["sigla_partido"].unique()
    out = local_power.alinhamento_partido_com_prefeito(painel, partidos)
    # SP 2014: prefeito PT, partido=PT → alinhado_partido=1
    r = out[(out["id_municipio"] == "3550308") & (out["ano_presidencial"] == 2014)
            & (out["sigla_partido"] == "PT")].iloc[0]
    assert int(r["alinhado_prefeito_partido"]) == 1
    assert int(r["alinhado_prefeito_coligacao"]) == 1  # PT na própria coligação
    # SP 2018: prefeito PSDB (coligação PSDB:DEM:PR), partido=PSL → 0/0
    r2 = out[(out["id_municipio"] == "3550308") & (out["ano_presidencial"] == 2018)
             & (out["sigla_partido"] == "PSL")].iloc[0]
    assert int(r2["alinhado_prefeito_partido"]) == 0
    assert int(r2["alinhado_prefeito_coligacao"]) == 0


def test_local_power_alinhamento_com_prefeito_na(painel, pres_long):
    partidos = pres_long["sigla_partido"].unique()
    out = local_power.alinhamento_partido_com_prefeito(painel, partidos)
    # Capão Bonito sem prefeito → ambos NA
    cb = out[out["id_municipio"] == "3509502"]
    assert cb["alinhado_prefeito_partido"].isna().all()
    assert cb["alinhado_prefeito_coligacao"].isna().all()


# ============================================================
# historical
# ============================================================
def test_historical_lag_e_swing(pres_long):
    hist = historical.features_historical(pres_long)
    # SP PT: 2014 share=400/1000=0.4; 2018 share=300/1000=0.3.
    # Em 2018, lag = 0.4, swing = 0.3 - 0.4 = -0.1
    r = hist[(hist["id_municipio"] == "3550308") & (hist["ano_presidencial"] == 2018)
             & (hist["sigla_partido"] == "PT")].iloc[0]
    assert r["lag_share_1t"] == pytest.approx(0.4)
    assert r["swing_share_1t"] == pytest.approx(-0.1)
    # Primeiro ano (2014) não tem lag
    r0 = hist[(hist["id_municipio"] == "3550308") & (hist["ano_presidencial"] == 2014)
              & (hist["sigla_partido"] == "PT")].iloc[0]
    assert pd.isna(r0["lag_share_1t"])


def test_historical_expande_universo_de_partidos(pres_long):
    """PSL não apareceu em 2014 mas o universo deve incluir share = 0 para ele."""
    hist = historical.features_historical(pres_long)
    sp14_psl = hist[(hist["id_municipio"] == "3550308") & (hist["ano_presidencial"] == 2014)
                    & (hist["sigla_partido"] == "PSL")]
    # universo deve conter PSL em 2014 com share = 0 (porque apareceu em 2018)
    assert len(sp14_psl) == 1


def test_historical_sucessao_none_equivale_a_lag_original(pres_long):
    """Sem mapeamento de sucessão, `lag_share_1t_sucessao` == `lag_share_1t`."""
    hist = historical.features_historical(pres_long, sucessoes=None)
    iguais = (
        hist["lag_share_1t"].fillna(-999.0)
        == hist["lag_share_1t_sucessao"].fillna(-999.0)
    ).all()
    assert iguais, "sem sucessão, as duas colunas devem ser idênticas"


def test_historical_sucessao_remapeia_lag():
    """Com mapping {PSL:2018:PSDB}, lag_sucessao de PSL_2018 aponta para PSDB_2014."""
    # Fixture compacto: 1 município, 2 anos, partidos PT/PSDB/PSL.
    # 2014: PT 400, PSDB 600. 2018: PT 300, PSL 700.
    rows = []
    for ano, d in [
        (2014, {"PT": 400, "PSDB": 600}),
        (2018, {"PT": 300, "PSL": 700}),
    ]:
        total = sum(d.values())
        for partido, votos in d.items():
            rows.append({
                "ano_presidencial": ano, "id_municipio": "M1",
                "sigla_partido": partido, "share_1t": votos / total,
            })
    pres_long_fake = pd.DataFrame(rows)

    hist = historical.features_historical(
        pres_long_fake, sucessoes={"PSL": {2018: "PSDB"}},
    )
    psl_2018 = hist[
        (hist["sigla_partido"] == "PSL") & (hist["ano_presidencial"] == 2018)
    ].iloc[0]

    # Original: PSL em 2014 foi expandido pra 0 → lag_share_1t == 0
    assert psl_2018["lag_share_1t"] == pytest.approx(0.0)
    # Sucessão: canonical(PSL, 2018) = PSDB → lag = PSDB_2014 = 0.6
    assert psl_2018["lag_share_1t_sucessao"] == pytest.approx(0.6)


def test_historical_lag_uf_sucessao_pondera_por_eleitorado() -> None:
    """lag_share_1t_uf_sucessao = média ponderada por eleitorado do
    lag_sucessao em todos os muns da UF.

    Setup: 2 UFs (SP, RJ) × 2 muns cada × 2 anos.
    PSL 2018 em SP: 0.50 (mun A 100k votos) e 0.60 (mun B 100k votos).
    PSL 2018 em RJ: 0.30 (mun C 50k votos) e 0.30 (mun D 50k votos).
    Esperado: PL 2022 SP UF lag = 0.55; PL 2022 RJ UF lag = 0.30.
    """
    rows = []
    config = {
        ("3550308", "SP", 100_000): {2018: {"PSL": 0.50, "PT": 0.30},
                                     2022: {"PL": 0.45, "PT": 0.40}},
        ("3506003", "SP", 100_000): {2018: {"PSL": 0.60, "PT": 0.20},
                                     2022: {"PL": 0.55, "PT": 0.30}},
        ("3304557", "RJ", 50_000): {2018: {"PSL": 0.30, "PT": 0.50},
                                    2022: {"PL": 0.20, "PT": 0.65}},
        ("3303302", "RJ", 50_000): {2018: {"PSL": 0.30, "PT": 0.50},
                                    2022: {"PL": 0.25, "PT": 0.60}},
    }
    for (mun, uf, eleitorado), por_ano in config.items():
        for ano, shares in por_ano.items():
            for partido, share in shares.items():
                rows.append({
                    "ano_presidencial": ano, "id_municipio": mun,
                    "sigla_uf": uf, "sigla_partido": partido,
                    "share_1t": share, "total_votos_mun": eleitorado,
                })
    long = pd.DataFrame(rows)

    hist = historical.features_historical(
        long, anos=[2018, 2022], sucessoes={"PL": {2022: "PSL"}},
        ano_col="ano_presidencial",
    )
    pl_2022 = hist[
        (hist["sigla_partido"] == "PL") & (hist["ano_presidencial"] == 2022)
    ]
    sp = pl_2022[pl_2022["id_municipio"].isin(["3550308", "3506003"])]
    rj = pl_2022[pl_2022["id_municipio"].isin(["3304557", "3303302"])]
    # SP: (100k*0.50 + 100k*0.60) / 200k = 0.55
    assert sp["lag_share_1t_uf_sucessao"].iloc[0] == pytest.approx(0.55)
    # RJ: (50k*0.30 + 50k*0.30) / 100k = 0.30
    assert rj["lag_share_1t_uf_sucessao"].iloc[0] == pytest.approx(0.30)
    # Mesmo valor pra todos os muns da mesma UF (broadcast)
    assert sp["lag_share_1t_uf_sucessao"].nunique() == 1
    assert rj["lag_share_1t_uf_sucessao"].nunique() == 1


def test_historical_lag_uf_primeiro_ano_nan() -> None:
    """No primeiro ano (sem lag disponível) lag_uf_sucessao deve ser NaN."""
    rows = [
        {"ano_presidencial": 2018, "id_municipio": "M1", "sigla_uf": "SP",
         "sigla_partido": "PT", "share_1t": 0.4, "total_votos_mun": 1000},
        {"ano_presidencial": 2018, "id_municipio": "M2", "sigla_uf": "SP",
         "sigla_partido": "PT", "share_1t": 0.6, "total_votos_mun": 1000},
    ]
    long = pd.DataFrame(rows)
    hist = historical.features_historical(long, anos=[2018], ano_col="ano_presidencial")
    assert hist["lag_share_1t_uf_sucessao"].isna().all()


# ============================================================
# continuity
# ============================================================
def test_continuity_classifica_transicoes(df_prefeito, df_partidos_prefeito):
    hist = continuity.calcular_historico_continuidade(df_prefeito, df_partidos_prefeito)
    # SP 2012 PT → 2016 PSDB: partidos diferentes, e 'PSDB' não está em 'PT:PCdoB:PSB'
    # nem 'PT' está em 'PSDB:DEM:PR' → ruptura
    sp16 = hist[(hist["id_municipio"] == "3550308") & (hist["ano_eleicao_municipal"] == 2016)].iloc[0]
    assert sp16["continuidade_classe"] == "ruptura"
    # Adamantina 2012 PT (coligação 'PT') → 2016 PSB: PSB não está em 'PT',
    # PT não está em 'PSB:PP' → ruptura também
    ad16 = hist[(hist["id_municipio"] == "3500105") & (hist["ano_eleicao_municipal"] == 2016)].iloc[0]
    assert ad16["continuidade_classe"] == "ruptura"


def test_continuity_indice_mapeia_classe():
    assert continuity.CLASSE_INDICE["ruptura"] == 0.0
    assert continuity.CLASSE_INDICE["total"] == 1.0


def test_continuity_total_em_reeleicao_do_mesmo_candidato():
    """Mesmo partido + mesmo número → total.

    2 mandatos municipais consecutivos do mesmo partido = 8 anos
    (2012-2016 + 2016-2020 = período total de governo do partido).
    """
    df = pd.DataFrame([
        {"ano": 2012, "sigla_uf": "SP", "id_municipio": "X", "numero_candidato": 13,
         "nome_candidato": "A", "sigla_partido": "PT", "votos": 100},
        {"ano": 2016, "sigla_uf": "SP", "id_municipio": "X", "numero_candidato": 13,
         "nome_candidato": "A", "sigla_partido": "PT", "votos": 120},
    ])
    hist = continuity.calcular_historico_continuidade(df, pd.DataFrame())
    r = hist[hist["ano_eleicao_municipal"] == 2016].iloc[0]
    assert r["continuidade_classe"] == "total"
    assert r["anos_consecutivos_mesmo_partido"] == 8


def test_continuity_tres_mandatos_consecutivos_dao_12_anos():
    """Regressão: 3 mandatos municipais consecutivos do mesmo partido = 12 anos.

    Caso real que triggeou a correção: Santana de Parnaíba (SP) teve PSDB
    em 2012 (Elvis), 2016 (Elvis reeleito), 2020 (Antonio Pereira). Total:
    12 anos consecutivos do partido. A versão buggy retornava 8.
    """
    df = pd.DataFrame([
        {"ano": 2012, "sigla_uf": "SP", "id_municipio": "Y", "numero_candidato": 45,
         "nome_candidato": "A", "sigla_partido": "PSDB", "votos": 100},
        {"ano": 2016, "sigla_uf": "SP", "id_municipio": "Y", "numero_candidato": 45,
         "nome_candidato": "A", "sigla_partido": "PSDB", "votos": 120},
        {"ano": 2020, "sigla_uf": "SP", "id_municipio": "Y", "numero_candidato": 45,
         "nome_candidato": "B", "sigla_partido": "PSDB", "votos": 140},
    ])
    hist = continuity.calcular_historico_continuidade(df, pd.DataFrame())
    r2020 = hist[hist["ano_eleicao_municipal"] == 2020].iloc[0]
    assert r2020["continuidade_classe"] in ("total", "forte")
    assert r2020["anos_consecutivos_mesmo_partido"] == 12
    assert r2020["anos_consecutivos_mesmo_grupo"] == 12


def test_continuity_ruptura_reinicia_contagem_corretamente():
    """Após ruptura, próxima transição forte deve iniciar em 8, não 4.

    Cenário: 2012 PT, 2016 PSDB (ruptura), 2020 PSDB (forte).
    Em 2020 o PSDB governou por 8 anos (2017-2024), não 4.
    """
    df = pd.DataFrame([
        {"ano": 2012, "sigla_uf": "SP", "id_municipio": "Z", "numero_candidato": 13,
         "nome_candidato": "A", "sigla_partido": "PT", "votos": 100},
        {"ano": 2016, "sigla_uf": "SP", "id_municipio": "Z", "numero_candidato": 45,
         "nome_candidato": "B", "sigla_partido": "PSDB", "votos": 110},
        {"ano": 2020, "sigla_uf": "SP", "id_municipio": "Z", "numero_candidato": 45,
         "nome_candidato": "B", "sigla_partido": "PSDB", "votos": 130},
    ])
    hist = continuity.calcular_historico_continuidade(df, pd.DataFrame())
    r2016 = hist[hist["ano_eleicao_municipal"] == 2016].iloc[0]
    r2020 = hist[hist["ano_eleicao_municipal"] == 2020].iloc[0]
    assert r2016["continuidade_classe"] == "ruptura"
    assert r2016["anos_consecutivos_mesmo_partido"] == 0
    assert r2020["continuidade_classe"] == "total"
    assert r2020["anos_consecutivos_mesmo_partido"] == 8


def test_continuity_features_mapeia_presidencial_para_municipal(
    df_prefeito, df_partidos_prefeito
):
    out = continuity.features_continuity(
        df_prefeito, df_partidos_prefeito, anos_presidenciais=[2014, 2018]
    )
    # 2014 presidencial → eleição municipal 2012 (primeira observada → NA)
    r14 = out[out["ano_presidencial"] == 2014]
    assert r14["continuidade_classe"].isna().all()
    # 2018 presidencial → eleição municipal 2016
    r18 = out[out["ano_presidencial"] == 2018]
    assert not r18["continuidade_classe"].isna().all()


# ============================================================
# vertical
# ============================================================
def test_vertical_governador_vencedor():
    df_gov = pd.DataFrame([
        {"ano": 2010, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 45, "nome_candidato": "A", "sigla_partido": "PSDB", "votos": 600},
        {"ano": 2010, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 13, "nome_candidato": "B", "sigla_partido": "PT", "votos": 400},
    ])
    venc = vertical.governador_vencedor_por_eleicao(df_gov, pd.DataFrame())
    assert venc.iloc[0]["gov_partido"] == "PSDB"
    assert venc.iloc[0]["gov_share_1t"] == pytest.approx(0.6)


def test_vertical_alinhamento_vigente_e_concorrente():
    df_gov = pd.DataFrame([
        # 2010 SP PSDB vence
        {"ano": 2010, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 45, "nome_candidato": "A", "sigla_partido": "PSDB", "votos": 600},
        {"ano": 2010, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 13, "nome_candidato": "B", "sigla_partido": "PT", "votos": 400},
        # 2014 SP PT vence
        {"ano": 2014, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 13, "nome_candidato": "C", "sigla_partido": "PT", "votos": 600},
        {"ano": 2014, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 45, "nome_candidato": "D", "sigla_partido": "PSDB", "votos": 400},
    ])
    df_partidos_gov = pd.DataFrame([
        {"ano": 2010, "sigla_uf": "SP", "sigla_partido": "PSDB",
         "composicao_coligacao": "PSDB:DEM"},
        {"ano": 2014, "sigla_uf": "SP", "sigla_partido": "PT",
         "composicao_coligacao": "PT:PSB"},
    ])
    painel = pd.DataFrame([
        {"ano_presidencial": 2014, "id_municipio": "3550308", "sigla_uf": "SP"},
    ])
    out = vertical.alinhamento_partido_com_governador(
        painel, df_gov, df_partidos_gov, partidos=["PT", "PSDB", "DEM", "PSB"]
    )
    # Vigente (2010) PSDB: PSDB=1/1, DEM=0/1, PT=0/0
    vig_psdb = out[out["sigla_partido"] == "PSDB"].iloc[0]
    assert vig_psdb["alinhado_gov_vigente_partido"] == 1
    vig_dem = out[out["sigla_partido"] == "DEM"].iloc[0]
    assert vig_dem["alinhado_gov_vigente_coligacao"] == 1
    assert vig_dem["alinhado_gov_vigente_partido"] == 0
    # Concorrente (2014) PT: PT=1/1, PSB=0/1
    conc_pt = out[out["sigla_partido"] == "PT"].iloc[0]
    assert conc_pt["alinhado_gov_concorrente_partido"] == 1
    conc_psb = out[out["sigla_partido"] == "PSB"].iloc[0]
    assert conc_psb["alinhado_gov_concorrente_coligacao"] == 1


def test_vertical_share_dep_federal_soma_por_partido():
    df_dep = pd.DataFrame([
        {"ano": 2014, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 1313, "sigla_partido": "PT", "votos": 300},
        {"ano": 2014, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 1714, "sigla_partido": "PT", "votos": 200},  # mesmo partido
        {"ano": 2014, "sigla_uf": "SP", "id_municipio": "3550308",
         "numero_candidato": 4545, "sigla_partido": "PSDB", "votos": 500},
    ])
    out = vertical.share_dep_federal_por_partido(df_dep, anos_presidenciais=[2014])
    pt = out[out["sigla_partido"] == "PT"].iloc[0]
    assert pt["share_dep_federal_partido"] == pytest.approx(0.5)  # 500/1000


# ============================================================
# coligacao — reconstrução via sequencial
# ============================================================
def test_coligacao_reconstroi_via_sequencial_governador_2010():
    """Cenário 2010 SP gov: sequencial populado, composicao NULL.

    PDT+PT+PT do B compartilham o sequencial '...34' (coligação UNIÃO PARA
    MUDAR). Depois da reconstrução, todos 3 devem ter
    composicao_coligacao = "PDT:PT:PT do B" (ordem alfabética).
    Partido isolado PSOL deve virar composicao_coligacao = "PSOL".
    """
    df = pd.DataFrame([
        {"ano": 2010, "sigla_uf": "SP", "sigla_partido": "PDT",
         "tipo_agremiacao": "coligacao",
         "sequencial_coligacao": "250000000034", "composicao_coligacao": None},
        {"ano": 2010, "sigla_uf": "SP", "sigla_partido": "PT",
         "tipo_agremiacao": "coligacao",
         "sequencial_coligacao": "250000000034", "composicao_coligacao": None},
        {"ano": 2010, "sigla_uf": "SP", "sigla_partido": "PT do B",
         "tipo_agremiacao": "coligacao",
         "sequencial_coligacao": "250000000034", "composicao_coligacao": None},
        {"ano": 2010, "sigla_uf": "SP", "sigla_partido": "PSOL",
         "tipo_agremiacao": "partido isolado",
         "sequencial_coligacao": None, "composicao_coligacao": None},
    ])
    out = coligacao.reconstruir_composicao_via_sequencial(df, ["ano", "sigla_uf"])
    # Coligação — todos 3 partidos com mesma composicao ordenada alfabeticamente
    pdt = out[out["sigla_partido"] == "PDT"].iloc[0]
    assert pdt["composicao_coligacao"] == "PDT:PT:PT do B"
    pt = out[out["sigla_partido"] == "PT"].iloc[0]
    assert pt["composicao_coligacao"] == "PDT:PT:PT do B"
    # Partido isolado — composicao = sigla
    psol = out[out["sigla_partido"] == "PSOL"].iloc[0]
    assert psol["composicao_coligacao"] == "PSOL"


def test_coligacao_preserva_valores_existentes():
    """Se composicao_coligacao já veio populada, a função não sobrescreve."""
    df = pd.DataFrame([
        {"ano": 2014, "sigla_uf": "SP", "sigla_partido": "PT",
         "tipo_agremiacao": "coligacao",
         "sequencial_coligacao": "999", "composicao_coligacao": "PT:PCdoB"},
        {"ano": 2014, "sigla_uf": "SP", "sigla_partido": "PCdoB",
         "tipo_agremiacao": "coligacao",
         "sequencial_coligacao": "999", "composicao_coligacao": "PT:PCdoB"},
    ])
    out = coligacao.reconstruir_composicao_via_sequencial(df, ["ano", "sigla_uf"])
    assert out["composicao_coligacao"].tolist() == ["PT:PCdoB", "PT:PCdoB"]


def test_coligacao_2020_sem_sequencial_fica_na():
    """Cenário 2020 prefeito: nem composicao nem sequencial — mantém NA.

    Esse é o caso onde só o plan B (CSV TSE) resolve.
    """
    df = pd.DataFrame([
        {"ano": 2020, "sigla_uf": "SP", "id_municipio": "3550308",
         "sigla_partido": "REPUBLICANOS", "tipo_agremiacao": "coligacao",
         "sequencial_coligacao": None, "composicao_coligacao": None},
    ])
    out = coligacao.reconstruir_composicao_via_sequencial(
        df, ["ano", "id_municipio"]
    )
    assert out["composicao_coligacao"].isna().all()


# ============================================================
# Sanity do script (import dinâmico)
# ============================================================
def test_script_03_importa():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "features_script",
        Path(__file__).resolve().parent.parent / "scripts" / "03_features.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert callable(mod.carregar_insumos)
    assert callable(mod.computar_features)
    assert callable(mod.consolidar)
