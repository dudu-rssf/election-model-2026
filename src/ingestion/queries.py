"""
src.ingestion.queries — templates SQL parametrizados para Base dos Dados.

Todas as queries respeitam o filtro do modo ativo (`MODE_CFG`):
  - dev  -> UF=SP, anos presidenciais [2014, 2018, 2022]
  - prod -> todas UFs, anos 1998..2022

As tabelas alvo na Base dos Dados (BigQuery):

  br_tse_eleicoes.resultados_candidato_municipio
  br_tse_eleicoes.candidatos
  br_bd_diretorios_brasil.municipio

Schemas podem mudar; se mudar, `validate.py` falha rápido e a fase para.
"""
from __future__ import annotations

from typing import Iterable

from src.config import MODE_CFG


# ------------------------------------------------------------
# Helpers de fragmentos SQL (usam nome de coluna COMPLETO: já com alias)
# ------------------------------------------------------------
def _uf_clause(ufs: Iterable[str] | str, column: str) -> str:
    """Gera `AND <column> IN ('SP', ...)`; se `ufs == 'all'`, retorna vazio."""
    if ufs == "all" or ufs is None:
        return ""
    ufs_list = list(ufs)
    if not ufs_list:
        return ""
    quoted = ", ".join(f"'{u}'" for u in ufs_list)
    return f"AND {column} IN ({quoted})"


def _anos_clause(anos: Iterable[int], column: str) -> str:
    anos_list = [int(a) for a in anos]
    if not anos_list:
        return ""
    joined = ", ".join(str(a) for a in anos_list)
    return f"AND {column} IN ({joined})"


# ------------------------------------------------------------
# Mapeamento presidencial -> eleição municipal imediatamente anterior
# ------------------------------------------------------------
# Regra: prefeito vigente em ano presidencial X = prefeito eleito na última
# eleição municipal antes de X (src.features.panel documenta isso).
#
#   presidencial 1998 -> prefeito eleito em 1996
#   presidencial 2002 -> 2000
#   presidencial 2006 -> 2004
#   presidencial 2010 -> 2008
#   presidencial 2014 -> 2012
#   presidencial 2018 -> 2016
#   presidencial 2022 -> 2020
#   presidencial 2026 -> 2024
PRESIDENCIAL_TO_MUNICIPAL: dict[int, int] = {
    1998: 1996,
    2002: 2000,
    2006: 2004,
    2010: 2008,
    2014: 2012,
    2018: 2016,
    2022: 2020,
    2026: 2024,
}


# Eleição estadual anterior ao pleito presidencial (governador/dep.fed. vigente).
# No Brasil eleições gerais (presidencial+estadual) acontecem no mesmo ano. O
# governador VIGENTE em 2014 foi eleito em 2010; o CONCORRENTE é eleito em 2014.
# Baixamos ambos: `anos_estaduais_para_panel` devolve a união { X, X-4 }.
PRESIDENCIAL_TO_ESTADUAL_ANTERIOR: dict[int, int] = {
    1998: 1994,
    2002: 1998,
    2006: 2002,
    2010: 2006,
    2014: 2010,
    2018: 2014,
    2022: 2018,
    2026: 2022,
}


def anos_municipais_para_panel(anos_presidenciais: Iterable[int]) -> list[int]:
    """Retorna anos de eleição municipal necessários para o painel."""
    return sorted({PRESIDENCIAL_TO_MUNICIPAL[int(a)] for a in anos_presidenciais})


def anos_estaduais_para_panel(anos_presidenciais: Iterable[int]) -> list[int]:
    """Anos de eleição estadual relevantes = {X, X-4} para X presidencial.

    Inclui o ano concorrente (governador/dep.fed. eleito junto com o presidente)
    e o ano anterior (governador/dep.fed. vigente no momento da presidencial).
    A Fase 3 decide quais usar como feature ("alinhamento vigente" vs
    "alinhamento concorrente").
    """
    anos = {int(a) for a in anos_presidenciais}
    anteriores = {PRESIDENCIAL_TO_ESTADUAL_ANTERIOR[int(a)] for a in anos_presidenciais}
    return sorted(anos | anteriores)


# ------------------------------------------------------------
# Queries
# ------------------------------------------------------------
def resultados_presidenciais_sql() -> str:
    """Resultados presidenciais (1º turno) por candidato e município."""
    uf = _uf_clause(MODE_CFG["ufs"], column="r.sigla_uf")
    anos = _anos_clause(MODE_CFG["anos_presidencial"], column="r.ano")
    return f"""
    SELECT
      r.ano,
      r.sigla_uf,
      r.id_municipio,
      r.id_municipio_tse,
      r.cargo,
      r.numero_candidato,
      r.sigla_partido,
      r.turno,
      r.votos
    FROM `basedosdados.br_tse_eleicoes.resultados_candidato_municipio` r
    WHERE r.cargo = 'presidente'
      AND r.turno = 1
      {anos}
      {uf}
    """.strip()


def resultados_prefeito_sql() -> str:
    """Resultados de prefeito (1º turno) por candidato e município.

    Usa diretamente `MODE_CFG["anos_municipal"]` (dev/prod no YAML). Isso
    cobre as duas finalidades:
      - Fase 3 (presidencial): lag do prefeito vigente em X-2 para features
        de local_power. Os anos derivados de `anos_municipais_para_panel`
        são sempre subconjunto de `anos_municipal`.
      - Fase 4.5 (prefeito): anos de eleição municipal alvo (incluindo 2024,
        que não precede nenhum ano presidencial coberto).
    """
    uf = _uf_clause(MODE_CFG["ufs"], column="r.sigla_uf")
    anos = _anos_clause(MODE_CFG["anos_municipal"], column="r.ano")
    return f"""
    SELECT
      r.ano,
      r.sigla_uf,
      r.id_municipio,
      r.cargo,
      r.numero_candidato,
      r.sigla_partido,
      r.turno,
      r.votos
    FROM `basedosdados.br_tse_eleicoes.resultados_candidato_municipio` r
    WHERE r.cargo = 'prefeito'
      AND r.turno = 1
      {anos}
      {uf}
    """.strip()


def candidatos_presidenciais_sql() -> str:
    """Ficha dos candidatos presidenciais (para coligações/partido).

    Nota de schema: na tabela `candidatos`, BD usa `numero`/`nome`. Aliasamos
    para `numero_candidato`/`nome_candidato` para manter a API consistente
    com a tabela `resultados_candidato_municipio`.
    """
    anos = _anos_clause(MODE_CFG["anos_presidencial"], column="c.ano")
    return f"""
    SELECT
      c.ano,
      c.numero AS numero_candidato,
      c.nome AS nome_candidato,
      c.sigla_partido,
      c.cargo
    FROM `basedosdados.br_tse_eleicoes.candidatos` c
    WHERE c.cargo = 'presidente'
      {anos}
    """.strip()


def candidatos_prefeito_sql() -> str:
    """Ficha dos candidatos a prefeito (1º turno).

    Ver `resultados_prefeito_sql` — anos cobertos = `MODE_CFG["anos_municipal"]`.
    """
    uf = _uf_clause(MODE_CFG["ufs"], column="c.sigla_uf")
    anos = _anos_clause(MODE_CFG["anos_municipal"], column="c.ano")
    return f"""
    SELECT
      c.ano,
      c.sigla_uf,
      c.id_municipio,
      c.numero AS numero_candidato,
      c.nome AS nome_candidato,
      c.sigla_partido,
      c.cargo
    FROM `basedosdados.br_tse_eleicoes.candidatos` c
    WHERE c.cargo = 'prefeito'
      {anos}
      {uf}
    """.strip()


def resultados_governador_sql() -> str:
    """Resultados de governador (1º turno) por candidato e município.

    Cobre `anos_estaduais_para_panel(anos_presidencial)` — a Fase 3 escolhe
    se agrega pelo governador vigente (X-4) ou concorrente (X).
    """
    uf = _uf_clause(MODE_CFG["ufs"], column="r.sigla_uf")
    anos_gov = anos_estaduais_para_panel(MODE_CFG["anos_presidencial"])
    anos = _anos_clause(anos_gov, column="r.ano")
    return f"""
    SELECT
      r.ano,
      r.sigla_uf,
      r.id_municipio,
      r.cargo,
      r.numero_candidato,
      r.sigla_partido,
      r.turno,
      r.votos
    FROM `basedosdados.br_tse_eleicoes.resultados_candidato_municipio` r
    WHERE r.cargo = 'governador'
      AND r.turno = 1
      {anos}
      {uf}
    """.strip()


def candidatos_governador_sql() -> str:
    """Ficha dos candidatos a governador (com coligação estadual)."""
    uf = _uf_clause(MODE_CFG["ufs"], column="c.sigla_uf")
    anos_gov = anos_estaduais_para_panel(MODE_CFG["anos_presidencial"])
    anos = _anos_clause(anos_gov, column="c.ano")
    return f"""
    SELECT
      c.ano,
      c.sigla_uf,
      c.numero AS numero_candidato,
      c.nome AS nome_candidato,
      c.sigla_partido,
      c.cargo
    FROM `basedosdados.br_tse_eleicoes.candidatos` c
    WHERE c.cargo = 'governador'
      {anos}
      {uf}
    """.strip()


def resultados_deputado_federal_sql() -> str:
    """Resultados de deputado federal por candidato × município.

    O cargo é estadual (circunscrição = UF) mas a tabela do BD tem o
    detalhamento municipal dos votos, que é o que precisamos para a Fase 3
    (share local por coligação federal).
    """
    uf = _uf_clause(MODE_CFG["ufs"], column="r.sigla_uf")
    anos_gov = anos_estaduais_para_panel(MODE_CFG["anos_presidencial"])
    anos = _anos_clause(anos_gov, column="r.ano")
    return f"""
    SELECT
      r.ano,
      r.sigla_uf,
      r.id_municipio,
      r.cargo,
      r.numero_candidato,
      r.sigla_partido,
      r.turno,
      r.votos
    FROM `basedosdados.br_tse_eleicoes.resultados_candidato_municipio` r
    WHERE r.cargo = 'deputado federal'
      AND r.turno = 1
      {anos}
      {uf}
    """.strip()


def candidatos_deputado_federal_sql() -> str:
    """Ficha dos candidatos a deputado federal (com coligação federal)."""
    uf = _uf_clause(MODE_CFG["ufs"], column="c.sigla_uf")
    anos_gov = anos_estaduais_para_panel(MODE_CFG["anos_presidencial"])
    anos = _anos_clause(anos_gov, column="c.ano")
    return f"""
    SELECT
      c.ano,
      c.sigla_uf,
      c.numero AS numero_candidato,
      c.nome AS nome_candidato,
      c.sigla_partido,
      c.cargo
    FROM `basedosdados.br_tse_eleicoes.candidatos` c
    WHERE c.cargo = 'deputado federal'
      {anos}
      {uf}
    """.strip()


def partidos_prefeito_sql() -> str:
    """Composição de coligação por partido × município × eleição de prefeito.

    Em abr/2026 a BD moveu `composicao_coligacao` de `candidatos` para uma
    tabela nova `br_tse_eleicoes.partidos`. Esta query traz uma linha por
    (ano, sigla_uf, id_municipio, sigla_partido) com a string de composição
    da coligação que o partido integrou na eleição majoritária local.

    Cobertura observada (peek): 100% em 2012 e 2016; 0% em 2020 (buraco
    da BD). Plano B em backlog para 2020. Cobertura 2024 a confirmar via
    scripts/01_ingest.py.
    """
    uf = _uf_clause(MODE_CFG["ufs"], column="p.sigla_uf")
    anos = _anos_clause(MODE_CFG["anos_municipal"], column="p.ano")
    return f"""
    SELECT
      p.ano,
      p.sigla_uf,
      p.id_municipio,
      p.numero AS numero_partido,
      p.sigla AS sigla_partido,
      p.tipo_agremiacao,
      p.sequencial_coligacao,
      p.nome_coligacao,
      p.composicao_coligacao
    FROM `basedosdados.br_tse_eleicoes.partidos` p
    WHERE p.cargo = 'prefeito'
      AND p.turno = 1
      {anos}
      {uf}
    """.strip()


def partidos_governador_sql() -> str:
    """Composição de coligação por partido × UF × eleição de governador.

    Granularidade: 1 linha por (ano, sigla_uf, sigla_partido). Em `partidos`
    o campo `id_municipio` vem NULL para cargo='governador' (cargo estadual).

    Cobertura observada: 100% em 2014, 2018 e 2022.
    """
    uf = _uf_clause(MODE_CFG["ufs"], column="p.sigla_uf")
    anos_gov = anos_estaduais_para_panel(MODE_CFG["anos_presidencial"])
    anos = _anos_clause(anos_gov, column="p.ano")
    # GROUP BY pra colapsar replicações idempotentes (mesmo (ano,uf,partido)
    # vindo de turnos diferentes ou linhas duplicadas).
    return f"""
    SELECT
      p.ano,
      p.sigla_uf,
      p.numero AS numero_partido,
      p.sigla AS sigla_partido,
      ANY_VALUE(p.tipo_agremiacao) AS tipo_agremiacao,
      ANY_VALUE(p.sequencial_coligacao) AS sequencial_coligacao,
      ANY_VALUE(p.nome_coligacao) AS nome_coligacao,
      ANY_VALUE(p.composicao_coligacao) AS composicao_coligacao
    FROM `basedosdados.br_tse_eleicoes.partidos` p
    WHERE p.cargo = 'governador'
      AND p.turno = 1
      {anos}
      {uf}
    GROUP BY p.ano, p.sigla_uf, p.numero, p.sigla
    """.strip()


def diretorio_municipios_sql() -> str:
    """Diretório IBGE de municípios (id, UF, nome, região, capital)."""
    uf = _uf_clause(MODE_CFG["ufs"], column="sigla_uf")
    return f"""
    SELECT
      id_municipio,
      id_municipio_tse,
      sigla_uf,
      nome,
      nome_regiao AS regiao,
      capital_uf
    FROM `basedosdados.br_bd_diretorios_brasil.municipio`
    WHERE 1=1
      {uf}
    """.strip()


# ------------------------------------------------------------
# Registry usado por scripts/01_ingest.py
# ------------------------------------------------------------
QUERIES: dict[str, callable] = {
    "resultados_presidenciais": resultados_presidenciais_sql,
    "resultados_prefeito": resultados_prefeito_sql,
    "resultados_governador": resultados_governador_sql,
    "resultados_deputado_federal": resultados_deputado_federal_sql,
    "candidatos_presidenciais": candidatos_presidenciais_sql,
    "candidatos_prefeito": candidatos_prefeito_sql,
    "candidatos_governador": candidatos_governador_sql,
    "candidatos_deputado_federal": candidatos_deputado_federal_sql,
    "partidos_prefeito": partidos_prefeito_sql,
    "partidos_governador": partidos_governador_sql,
    "diretorio_municipios": diretorio_municipios_sql,
}


__all__ = [
    "PRESIDENCIAL_TO_MUNICIPAL",
    "PRESIDENCIAL_TO_ESTADUAL_ANTERIOR",
    "anos_municipais_para_panel",
    "anos_estaduais_para_panel",
    "resultados_presidenciais_sql",
    "resultados_prefeito_sql",
    "resultados_governador_sql",
    "resultados_deputado_federal_sql",
    "candidatos_presidenciais_sql",
    "candidatos_prefeito_sql",
    "candidatos_governador_sql",
    "candidatos_deputado_federal_sql",
    "partidos_prefeito_sql",
    "partidos_governador_sql",
    "diretorio_municipios_sql",
    "QUERIES",
]
