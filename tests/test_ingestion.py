"""
Testes unitários da Fase 1 — ingestão.

Objetivo: **zero dependência de BigQuery**. Tudo é mockado via `backend`
injetável ou via monkeypatch de `basedosdados`.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import MODE_CFG, PATHS, SEED
from src.ingestion import bd_client, queries, sample, validate


# ============================================================
# queries.py
# ============================================================
def test_uf_clause_dev_contem_SP():
    sql = queries.resultados_presidenciais_sql()
    assert "'SP'" in sql
    assert "r.cargo = 'presidente'" in sql
    assert "r.turno = 1" in sql
    # Anos dev
    for ano in MODE_CFG["anos_presidencial"]:
        assert str(ano) in sql


def test_anos_municipais_para_panel_mapeamento():
    assert queries.anos_municipais_para_panel([2014, 2018, 2022]) == [2012, 2016, 2020]
    assert queries.anos_municipais_para_panel([2026]) == [2024]
    assert queries.anos_municipais_para_panel([2018, 2014]) == [2012, 2016]


def test_prefeito_usa_ano_municipal_nao_presidencial():
    sql = queries.resultados_prefeito_sql()
    assert "r.cargo = 'prefeito'" in sql
    # Usa MODE_CFG["anos_municipal"] direto; em dev = [2012,2016,2020,2024].
    # 2024 é o ano municipal alvo da Fase 4.5 (não derivável de anos_presidencial).
    for ano_mun in MODE_CFG["anos_municipal"]:
        assert str(ano_mun) in sql
    # E NÃO deve mencionar os anos presidenciais diretamente
    assert "IN (2014" not in sql
    assert "IN (2018" not in sql


def test_registry_completo():
    esperados = {
        "resultados_presidenciais",
        "resultados_prefeito",
        "resultados_governador",
        "resultados_deputado_federal",
        "candidatos_presidenciais",
        "candidatos_prefeito",
        "candidatos_governador",
        "candidatos_deputado_federal",
        "diretorio_municipios",
    }
    assert set(queries.QUERIES) == esperados
    for fn in queries.QUERIES.values():
        assert callable(fn)
        assert isinstance(fn(), str) and fn()


# ------------------------------------------------------------
# Fase 1.5 — governador & deputado federal
# ------------------------------------------------------------
def test_anos_estaduais_inclui_concorrente_e_anterior():
    # dev: [2014, 2018, 2022] -> { 2010, 2014, 2018, 2022 }
    out = queries.anos_estaduais_para_panel([2014, 2018, 2022])
    assert out == [2010, 2014, 2018, 2022]
    # Presidencial 2026 -> {2022, 2026}
    assert queries.anos_estaduais_para_panel([2026]) == [2022, 2026]
    # Idempotente quanto à ordem
    assert queries.anos_estaduais_para_panel([2022, 2014]) == [2010, 2014, 2018, 2022]


def test_resultados_governador_sql_contem_cargo_e_anos_corretos():
    sql = queries.resultados_governador_sql()
    assert "r.cargo = 'governador'" in sql
    assert "r.turno = 1" in sql
    # Para dev: anos estaduais = {2010, 2014, 2018, 2022}
    for ano in [2010, 2014, 2018, 2022]:
        assert str(ano) in sql
    # UF dev deve aparecer
    assert "'SP'" in sql


def test_candidatos_governador_sql_tem_composicao_coligacao():
    sql = queries.candidatos_governador_sql()
    assert "c.cargo = 'governador'" in sql
    assert "composicao_coligacao" in sql
    # Não filtra turno (tabela de candidatos não tem turno)
    assert "turno" not in sql


def test_resultados_deputado_federal_sql_usa_cargo_correto():
    sql = queries.resultados_deputado_federal_sql()
    assert "r.cargo = 'deputado federal'" in sql
    assert "r.turno = 1" in sql
    # Anos estaduais cobrem concorrente + vigente
    for ano in [2010, 2014, 2018, 2022]:
        assert str(ano) in sql


def test_candidatos_deputado_federal_sql_ok():
    sql = queries.candidatos_deputado_federal_sql()
    assert "c.cargo = 'deputado federal'" in sql
    assert "composicao_coligacao" in sql


def test_validate_resultados_governador_aceita_df_valido():
    df = pd.DataFrame({
        "ano": [2014],
        "sigla_uf": ["SP"],
        "id_municipio": ["3550308"],
        "cargo": ["governador"],
        "sigla_partido": ["PSDB"],
        "numero_candidato": [45],
        "turno": [1],
        "votos": [1000],
    })
    rep = validate.ValidationReport()
    validate.validate_resultados_governador(df, rep)
    assert rep.ok


def test_validate_resultados_deputado_federal_detecta_id_invalido():
    df = pd.DataFrame({
        "ano": [2014],
        "sigla_uf": ["SP"],
        "id_municipio": ["XYZ"],  # inválido
        "cargo": ["deputado federal"],
        "sigla_partido": ["PT"],
        "numero_candidato": [1313],
        "turno": [1],
        "votos": [500],
    })
    rep = validate.ValidationReport()
    validate.validate_resultados_deputado_federal(df, rep)
    assert not rep.ok
    assert any("IBGE" in i.mensagem or "7 dígitos" in i.mensagem for i in rep.errors)


# ============================================================
# bd_client.py — cache, download, backend injetável
# ============================================================
class _FakeBackend:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.calls = 0

    def read_sql(self, query: str, billing_project_id: str) -> pd.DataFrame:
        self.calls += 1
        return self.df.copy()


@pytest.fixture
def raw_tmp(tmp_path, monkeypatch):
    """Redireciona PATHS['data_raw'] para tmp_path. Idempotente."""
    old = PATHS["data_raw"]
    monkeypatch.setitem(PATHS, "data_raw", tmp_path)
    yield tmp_path
    monkeypatch.setitem(PATHS, "data_raw", old)


def test_download_cache_miss_then_hit(raw_tmp):
    df_in = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    backend = _FakeBackend(df_in)

    out1 = bd_client.download("fake", sql="SELECT 1", backend=backend)
    out2 = bd_client.download("fake", sql="SELECT 1", backend=backend)

    pd.testing.assert_frame_equal(out1, df_in)
    pd.testing.assert_frame_equal(out2, df_in)
    assert backend.calls == 1, "segunda chamada deveria vir do cache"
    assert (raw_tmp / "fake.parquet").exists()


def test_download_force_ignora_cache(raw_tmp):
    df_in = pd.DataFrame({"a": [1]})
    backend = _FakeBackend(df_in)

    bd_client.download("fake", sql="SELECT 1", backend=backend)
    bd_client.download("fake", sql="SELECT 1", backend=backend, force=True)

    assert backend.calls == 2


def test_download_zero_linhas_levanta(raw_tmp):
    backend = _FakeBackend(pd.DataFrame({"a": []}))
    with pytest.raises(RuntimeError, match="0 linhas"):
        bd_client.download("fake", sql="SELECT 1", backend=backend)


# ============================================================
# sample.py — amostragem reprodutível
# ============================================================
def test_choose_ids_reprodutivel():
    ids = [f"{i:07d}" for i in range(200)]
    a = sample.choose_ids(ids, max_n=50, seed=SEED)
    b = sample.choose_ids(ids, max_n=50, seed=SEED)
    assert a == b
    assert len(a) == 50
    assert len(set(a)) == 50


def test_choose_ids_max_n_maior_que_universo():
    ids = ["0000001", "0000002"]
    assert sample.choose_ids(ids, max_n=10) == sorted(ids)


def test_choose_ids_seed_diferente_resultado_diferente():
    ids = [f"{i:07d}" for i in range(200)]
    a = sample.choose_ids(ids, max_n=50, seed=1)
    b = sample.choose_ids(ids, max_n=50, seed=2)
    assert a != b


def test_apply_dev_sampling_filtra_consistentemente():
    pres = pd.DataFrame({"id_municipio": ["3550308", "3500105", "3509502"], "votos": [1, 2, 3]})
    pref = pd.DataFrame({"id_municipio": ["3550308", "3500105"], "votos": [10, 20]})
    dire = pd.DataFrame({"id_municipio": ["3550308", "3500105", "3509502"], "nome": ["SP", "AM", "AR"]})

    frames = {"pres": pres, "pref": pref, "diretorio": dire}

    # Força max_municipios=2 via monkeypatch do MODE_CFG
    old = MODE_CFG.get("max_municipios")
    MODE_CFG["max_municipios"] = 2
    try:
        out = sample.apply_dev_sampling(frames)
    finally:
        MODE_CFG["max_municipios"] = old

    ids_pres = set(out["pres"]["id_municipio"])
    ids_pref = set(out["pref"]["id_municipio"])
    ids_dir = set(out["diretorio"]["id_municipio"])
    # O conjunto de IDs mantidos deve ser o mesmo nas tabelas que contêm esses IDs
    assert ids_pres == ids_dir
    assert ids_pref <= ids_pres  # pref só tem 2 dos 3


# ============================================================
# validate.py
# ============================================================
def test_validator_detecta_colunas_ausentes():
    df = pd.DataFrame({"ano": [2014], "votos": [100]})
    rep = validate.ValidationReport()
    validate.validate_resultados_presidenciais(df, rep)
    assert not rep.ok
    msgs = " ".join(i.mensagem for i in rep.errors)
    assert "sigla_uf" in msgs or "id_municipio" in msgs


def test_validator_detecta_votos_negativos():
    df = pd.DataFrame({
        "ano": [2014, 2014],
        "sigla_uf": ["SP", "SP"],
        "id_municipio": ["3550308", "3500105"],
        "cargo": ["presidente", "presidente"],
        "numero_candidato": [13, 13],
        "turno": [1, 1],
        "votos": [100, -5],
    })
    rep = validate.ValidationReport()
    validate.validate_resultados_presidenciais(df, rep)
    assert not rep.ok
    assert any("votos" in i.mensagem for i in rep.errors)


def test_validator_id_municipio_nao_7_digitos():
    df = pd.DataFrame({
        "ano": [2014],
        "sigla_uf": ["SP"],
        "id_municipio": ["355030"],  # 6 dígitos
        "cargo": ["presidente"],
        "numero_candidato": [13],
        "turno": [1],
        "votos": [100],
    })
    rep = validate.ValidationReport()
    validate.validate_resultados_presidenciais(df, rep)
    assert not rep.ok
    assert any("IBGE" in i.mensagem or "7 dígitos" in i.mensagem for i in rep.errors)


def test_validator_totais_oficiais_tolerancia():
    df = pd.DataFrame({
        "ano": [2018, 2018],
        "sigla_uf": ["SP", "SP"],
        "id_municipio": ["3550308", "3500105"],
        "cargo": ["presidente", "presidente"],
        "numero_candidato": [17, 13],
        "turno": [1, 1],
        "votos": [1_000_000, 600_000],
    })
    oficial = {(2018, "SP"): 1_600_000}  # bate exato
    rep = validate.ValidationReport()
    validate.validate_resultados_presidenciais(df, rep, oficial_por_uf=oficial)
    assert rep.ok

    # Agora com diff > 0.1%
    oficial_bad = {(2018, "SP"): 1_800_000}
    rep2 = validate.ValidationReport()
    validate.validate_resultados_presidenciais(df, rep2, oficial_por_uf=oficial_bad)
    assert not rep2.ok


def test_validation_report_markdown():
    rep = validate.ValidationReport()
    rep.add("x", "error", "coluna Y faltando")
    rep.add("x", "warning", "Z é estranho")
    md = rep.to_markdown()
    assert "# Relatório" in md
    assert "Errors" in md
    assert "Warnings" in md


# ============================================================
# 01_ingest.py (import dinâmico)
# ============================================================
def test_script_importa_e_tem_main():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "ingest_script",
        Path(__file__).resolve().parent.parent / "scripts" / "01_ingest.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert callable(mod.main)
    assert callable(mod.rodar_queries)
    assert callable(mod.validar)
