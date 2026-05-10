"""
Testes de `src.features.pesquisas`.

Cobre:
  * carregar_pesquisas_nacional: schema válido, valores fora de [0,1] raise.
  * aplicar_pesquisa_nacional: merge correto, NaN para anos/partidos
    sem pesquisa, não duplica linhas, log de cobertura.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import pesquisas as fpesq


def _csv_minimo(tmp_path: Path) -> Path:
    """CSV de pesquisas com 3 linhas de exemplo."""
    p = tmp_path / "pesquisas_test.csv"
    p.write_text(
        "ano,sigla_partido,nome_candidato,share_pesquisa,fonte,obs\n"
        "2018,PSL,Bolsonaro,0.30,test,nada\n"
        "2018,PT,Haddad,0.23,test,nada\n"
        "2022,PL,Bolsonaro,0.33,test,nada\n",
        encoding="utf-8",
    )
    return p


def test_carregar_pesquisas_nacional_schema(tmp_path: Path) -> None:
    p = _csv_minimo(tmp_path)
    df = fpesq.carregar_pesquisas_nacional(p)
    assert list(df.columns) == ["ano", "sigla_partido", "share_pesquisa"]
    assert len(df) == 3
    assert df["ano"].dtype == np.int64
    assert df.iloc[0]["share_pesquisa"] == pytest.approx(0.30)


def test_carregar_pesquisas_nacional_arquivo_inexistente(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        fpesq.carregar_pesquisas_nacional(tmp_path / "nao_existe.csv")


def test_carregar_pesquisas_nacional_share_fora_de_zero_um(tmp_path: Path) -> None:
    p = tmp_path / "ruim.csv"
    p.write_text(
        "ano,sigla_partido,share_pesquisa\n"
        "2022,PT,1.5\n",  # > 1
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=r"share_pesquisa fora"):
        fpesq.carregar_pesquisas_nacional(p)


def test_aplicar_pesquisa_nacional_merge_basico(tmp_path: Path) -> None:
    pq = fpesq.carregar_pesquisas_nacional(_csv_minimo(tmp_path))
    long = pd.DataFrame({
        "ano_presidencial": [2018, 2018, 2022, 2010],
        "sigla_partido": ["PSL", "PT", "PL", "NOVO"],
        "id_municipio": ["m1", "m2", "m3", "m4"],
    })
    out = fpesq.aplicar_pesquisa_nacional(long, pq)
    assert "share_pesquisa_nacional" in out.columns
    # PSL/2018: 0.30
    assert out.iloc[0]["share_pesquisa_nacional"] == pytest.approx(0.30)
    # PT/2018: 0.23
    assert out.iloc[1]["share_pesquisa_nacional"] == pytest.approx(0.23)
    # PL/2022: 0.33
    assert out.iloc[2]["share_pesquisa_nacional"] == pytest.approx(0.33)
    # NOVO/2010 não está no CSV → NaN
    assert pd.isna(out.iloc[3]["share_pesquisa_nacional"])


def test_aplicar_pesquisa_nacional_nao_duplica_linhas(tmp_path: Path) -> None:
    pq = fpesq.carregar_pesquisas_nacional(_csv_minimo(tmp_path))
    # Pesquisa duplicada (ano, partido) deveria explodir o merge
    pq_bad = pd.concat([pq, pq.iloc[[0]]], ignore_index=True)
    long = pd.DataFrame({
        "ano_presidencial": [2018],
        "sigla_partido": ["PSL"],
    })
    with pytest.raises(RuntimeError, match=r"merge expandiu"):
        fpesq.aplicar_pesquisa_nacional(long, pq_bad)


def test_aplicar_pesquisa_nacional_falta_coluna_ano() -> None:
    pq = pd.DataFrame({
        "ano": [2022], "sigla_partido": ["PL"], "share_pesquisa": [0.33],
    })
    long = pd.DataFrame({
        "sigla_partido": ["PL"],  # sem ano_presidencial
        "id_municipio": ["m1"],
    })
    with pytest.raises(ValueError, match=r"long sem colunas"):
        fpesq.aplicar_pesquisa_nacional(long, pq)


def test_features_pesquisa_pipe_completo(tmp_path: Path) -> None:
    p = _csv_minimo(tmp_path)
    long = pd.DataFrame({
        "ano_presidencial": [2018, 2022],
        "sigla_partido": ["PSL", "PL"],
        "id_municipio": ["m1", "m2"],
    })
    out = fpesq.features_pesquisa(long, p)
    assert out["share_pesquisa_nacional"].notna().all()
    assert out.iloc[0]["share_pesquisa_nacional"] == pytest.approx(0.30)
