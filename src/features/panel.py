"""
src.features.panel — construção do painel mestre.

O painel tem **uma linha por (id_municipio, ano_presidencial)** e contém:
  * metadados do município (UF, nome, região, capital)
  * info do **prefeito vigente** naquele ano presidencial

Regra de ouro — prefeito vigente:
    prefeito vigente em ano_presidencial X = vencedor do 1º turno
    da eleição municipal em `PRESIDENCIAL_TO_MUNICIPAL[X]`
    (ex.: 2014 -> eleição municipal de 2012)

Em caso de empate (improvável mas possível), mantemos o candidato com
`numero_candidato` menor apenas para determinismo e logamos um warning.

Este módulo não faz feature engineering — só o scaffold. Fase 3 anexa
features históricas, de continuidade, verticais e estruturais.
"""
from __future__ import annotations

import logging
from typing import Iterable

import pandas as pd

from src.config import MODE_CFG
from src.ingestion.queries import PRESIDENCIAL_TO_MUNICIPAL

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# 1. Vencedor por eleição municipal
# ------------------------------------------------------------
def prefeito_vencedor_por_eleicao(df_prefeito: pd.DataFrame) -> pd.DataFrame:
    """Reduz `resultados_prefeito` ao vencedor por (ano, id_municipio).

    Entrada: DataFrame com pelo menos
        ano, id_municipio, numero_candidato, nome_candidato,
        sigla_partido, votos (1º turno, já filtrado em scripts/01_ingest).

    Saída: DataFrame com
        ano_eleicao_municipal, id_municipio, sigla_uf,
        mayor_numero, mayor_nome, mayor_partido,
        mayor_votos, mayor_votos_total_mun, mayor_share_1t,
        mayor_margem_1t (vs 2º colocado).
    """
    required = {"ano", "id_municipio", "numero_candidato", "sigla_partido", "votos"}
    missing = required - set(df_prefeito.columns)
    if missing:
        raise ValueError(f"df_prefeito sem colunas: {sorted(missing)}")

    df = df_prefeito.copy()
    df["id_municipio"] = df["id_municipio"].astype("string")
    df["votos"] = df["votos"].astype("int64")

    # Total de votos por (ano, município) — denominador do share
    totais = (
        df.groupby(["ano", "id_municipio"], as_index=False)["votos"]
        .sum()
        .rename(columns={"votos": "mayor_votos_total_mun"})
    )

    # Rank por votos desc (estável via numero_candidato asc)
    df_sorted = df.sort_values(
        ["ano", "id_municipio", "votos", "numero_candidato"],
        ascending=[True, True, False, True],
    )
    df_sorted["rank"] = df_sorted.groupby(["ano", "id_municipio"]).cumcount() + 1

    vencedor = df_sorted.loc[df_sorted["rank"] == 1].copy()
    segundo = df_sorted.loc[df_sorted["rank"] == 2, ["ano", "id_municipio", "votos"]].rename(
        columns={"votos": "_votos_2o"}
    )

    vencedor = vencedor.merge(totais, on=["ano", "id_municipio"], how="left")
    vencedor = vencedor.merge(segundo, on=["ano", "id_municipio"], how="left")

    # Colunas de saída
    sigla_uf = vencedor["sigla_uf"] if "sigla_uf" in vencedor.columns else pd.NA
    nome = vencedor["nome_candidato"] if "nome_candidato" in vencedor.columns else pd.NA

    vencedor["mayor_share_1t"] = vencedor["votos"] / vencedor["mayor_votos_total_mun"]
    vencedor["mayor_margem_1t"] = (
        (vencedor["votos"] - vencedor["_votos_2o"].fillna(0))
        / vencedor["mayor_votos_total_mun"]
    )

    out = pd.DataFrame(
        {
            "ano_eleicao_municipal": vencedor["ano"].astype("int64"),
            "id_municipio": vencedor["id_municipio"],
            "sigla_uf": sigla_uf,
            "mayor_numero": vencedor["numero_candidato"],
            "mayor_nome": nome,
            "mayor_partido": vencedor["sigla_partido"],
            "mayor_votos": vencedor["votos"].astype("int64"),
            "mayor_votos_total_mun": vencedor["mayor_votos_total_mun"].astype("int64"),
            "mayor_share_1t": vencedor["mayor_share_1t"].astype("float64"),
            "mayor_margem_1t": vencedor["mayor_margem_1t"].astype("float64"),
        }
    ).reset_index(drop=True)

    # Warning se detectarmos empate no 1º turno (mesmos votos entre 1º e 2º)
    empates = (
        (vencedor["votos"] == vencedor["_votos_2o"])
        & vencedor["_votos_2o"].notna()
    ).sum()
    if empates:
        logger.warning(
            "prefeito_vencedor_por_eleicao: %d empate(s) no 1º turno; resolvido por menor número de candidato",
            int(empates),
        )
    return out


# ------------------------------------------------------------
# 2. Anexa coligação do prefeito via `partidos_prefeito`
# ------------------------------------------------------------
def anexar_coligacao_prefeito(
    vencedores: pd.DataFrame,
    df_partidos_prefeito: pd.DataFrame | None,
) -> pd.DataFrame:
    """Adiciona `mayor_coligacao` via join em (ano, id_municipio, sigla_partido).

    Fonte: `br_tse_eleicoes.partidos` filtrada por cargo='prefeito'. A BD
    moveu `composicao_coligacao` da tabela `candidatos` para `partidos`
    em abr/2026.

    Granularidade da fonte: 1 linha por (ano, sigla_uf, id_municipio,
    sigla_partido). Join é feito pelo PARTIDO do prefeito vencedor
    (sigla, não número), porque a tabela é indexada por partido.

    Buracos conhecidos: ano 2020 vem 100% NA (limitação da BD). Plano B
    pra 2020 tracked em backlog.
    """
    if df_partidos_prefeito is None or len(df_partidos_prefeito) == 0:
        logger.warning("sem partidos_prefeito — mayor_coligacao fica NA")
        out = vencedores.copy()
        out["mayor_coligacao"] = pd.NA
        return out

    if "composicao_coligacao" not in df_partidos_prefeito.columns:
        logger.warning(
            "partidos_prefeito sem 'composicao_coligacao'; mayor_coligacao fica NA"
        )
        out = vencedores.copy()
        out["mayor_coligacao"] = pd.NA
        return out

    p = df_partidos_prefeito.copy()
    p["id_municipio"] = p["id_municipio"].astype("string")

    # Reconstrói composicao_coligacao quando o BD veio com NULL mas
    # sequencial_coligacao está populado. Cobre:
    #   * 2012 prefeito (~3.4% das linhas com composicao NULL mas sequencial OK)
    #   * caso geral onde o BD não agregou (semelhante a 2010 governador).
    # Em 2020 prefeito, sequencial_coligacao TAMBÉM vem NULL → fica NA (plan B).
    if "sequencial_coligacao" in p.columns:
        from src.features.coligacao import reconstruir_composicao_via_sequencial
        p = reconstruir_composicao_via_sequencial(p, ["ano", "id_municipio"])

    # Em 2016 SP a tabela tem a sigla "REPUBLICANOS" pra número 10 (ex-PRB);
    # candidatos antigos podem aparecer com "PRB". Como joinamos por sigla,
    # qualquer rebatismo histórico vira NA — aceitável (poucos casos no SP dev).
    # Deduplica a fonte por (ano, mun, sigla) preservando coligação não-nula.
    p = (
        p.sort_values(
            ["ano", "id_municipio", "sigla_partido", "composicao_coligacao"],
            na_position="last",
        )
        .drop_duplicates(
            subset=["ano", "id_municipio", "sigla_partido"], keep="first"
        )[["ano", "id_municipio", "sigla_partido", "composicao_coligacao"]]
    )

    out = vencedores.merge(
        p,
        left_on=["ano_eleicao_municipal", "id_municipio", "mayor_partido"],
        right_on=["ano", "id_municipio", "sigla_partido"],
        how="left",
    ).drop(columns=["ano", "sigla_partido"])
    out = out.rename(columns={"composicao_coligacao": "mayor_coligacao"})

    # Log de cobertura por ano — útil pra acompanhar a degradação 2020.
    cob = (
        out.dropna(subset=["mayor_partido"])
        .assign(_ok=out["mayor_coligacao"].notna())
        .groupby("ano_eleicao_municipal")["_ok"]
        .mean()
    )
    for ano, pct in cob.items():
        if pct < 0.5:
            logger.warning(
                "mayor_coligacao: ano %d com cobertura %.1f%% (BD com buraco)",
                int(ano), 100 * pct,
            )
    return out


# ------------------------------------------------------------
# 3. Scaffold município × ano_presidencial
# ------------------------------------------------------------
def scaffold_municipio_ano(
    diretorio: pd.DataFrame,
    anos_presidenciais: Iterable[int],
) -> pd.DataFrame:
    """Produto cartesiano (municípios × anos presidenciais) com metadados do IBGE."""
    required = {"id_municipio", "sigla_uf", "nome"}
    missing = required - set(diretorio.columns)
    if missing:
        raise ValueError(f"diretorio sem colunas: {sorted(missing)}")

    d = diretorio.copy()
    d["id_municipio"] = d["id_municipio"].astype("string")
    d = d.drop_duplicates(subset=["id_municipio"])

    anos = sorted({int(a) for a in anos_presidenciais})
    if not anos:
        raise ValueError("anos_presidenciais vazio")

    # cross join via key constante
    d["_k"] = 1
    a = pd.DataFrame({"ano_presidencial": anos, "_k": 1})
    out = d.merge(a, on="_k").drop(columns="_k")

    # mapeia para ano municipal vigente
    out["ano_eleicao_municipal"] = out["ano_presidencial"].map(PRESIDENCIAL_TO_MUNICIPAL)
    if out["ano_eleicao_municipal"].isna().any():
        anos_sem_map = sorted(
            set(out.loc[out["ano_eleicao_municipal"].isna(), "ano_presidencial"].unique())
        )
        raise KeyError(
            f"anos presidenciais sem mapping municipal: {anos_sem_map}. "
            "Atualize PRESIDENCIAL_TO_MUNICIPAL em src.ingestion.queries."
        )
    out["ano_eleicao_municipal"] = out["ano_eleicao_municipal"].astype("int64")

    # Colunas de metadados opcionais
    for col in ("regiao", "capital_uf", "id_municipio_tse"):
        if col not in out.columns:
            out[col] = pd.NA

    cols = [
        "id_municipio",
        "sigla_uf",
        "nome",
        "regiao",
        "capital_uf",
        "ano_presidencial",
        "ano_eleicao_municipal",
    ]
    return out[cols].reset_index(drop=True)


# ------------------------------------------------------------
# 4. Painel final
# ------------------------------------------------------------
def construir_painel_mestre(
    diretorio: pd.DataFrame,
    df_prefeito: pd.DataFrame,
    df_partidos_prefeito: pd.DataFrame | None = None,
    anos_presidenciais: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Orquestra: scaffold × vencedor prefeito × coligação.

    `anos_presidenciais` default = MODE_CFG["anos_presidencial"] (lista de inteiros).
    `df_partidos_prefeito`: tabela `br_tse_eleicoes.partidos` filtrada por
    cargo='prefeito' (ver `src.ingestion.queries.partidos_prefeito_sql`).
    Se None, `mayor_coligacao` fica NA com warning.
    """
    anos = list(anos_presidenciais or MODE_CFG["anos_presidencial"])

    vencedores = prefeito_vencedor_por_eleicao(df_prefeito)
    vencedores = anexar_coligacao_prefeito(vencedores, df_partidos_prefeito)

    scaff = scaffold_municipio_ano(diretorio, anos)

    painel = scaff.merge(
        vencedores,
        on=["ano_eleicao_municipal", "id_municipio"],
        how="left",
        suffixes=("", "_prefeito"),
    )

    # Consistência de UF: se o diretório tem UF e o prefeito tem UF, devem bater.
    if "sigla_uf_prefeito" in painel.columns:
        mismatches = (
            (painel["sigla_uf_prefeito"].notna())
            & (painel["sigla_uf"] != painel["sigla_uf_prefeito"])
        ).sum()
        if mismatches:
            logger.warning(
                "UF diretório ≠ UF prefeito em %d linha(s); mantendo UF do diretório",
                int(mismatches),
            )
        painel = painel.drop(columns=["sigla_uf_prefeito"])

    # Log de cobertura: quantas linhas ficaram sem prefeito (município sem eleição ou sem vencedor registrado)
    sem_prefeito = painel["mayor_numero"].isna().sum()
    logger.info(
        "painel: %d linhas (%d municípios × %d anos); %d sem prefeito anexado",
        len(painel),
        painel["id_municipio"].nunique(),
        len(set(painel["ano_presidencial"])),
        int(sem_prefeito),
    )

    return painel


# ============================================================
# Fase 4.5 — painel municipal (target = prefeito)
# ------------------------------------------------------------
# O painel do MODELO PREFEITO é indexado por (id_municipio, ano_municipal),
# análogo ao painel presidencial (mun × ano_presidencial). O conceito de
# "prefeito vigente" na hora da eleição municipal X é o vencedor da eleição
# municipal anterior, X-4:
#
#   ano_municipal 2016 -> prefeito vigente = vencedor em 2012
#   ano_municipal 2020 -> vencedor em 2016
#   ano_municipal 2024 -> vencedor em 2020
#   ano_municipal 2028 -> vencedor em 2024
#
# Gap temporal: 4 anos (vs. 2 anos no presidencial).
# ============================================================
MUNICIPAL_TO_MUNICIPAL_ANTERIOR: dict[int, int] = {
    # Histórico brasileiro pós-Constituição — eleições municipais a cada 4 anos.
    2000: 1996,
    2004: 2000,
    2008: 2004,
    2012: 2008,
    2016: 2012,
    2020: 2016,
    2024: 2020,
    2028: 2024,
}


def scaffold_municipio_ano_municipal(
    diretorio: pd.DataFrame,
    anos_municipais: Iterable[int],
) -> pd.DataFrame:
    """Produto cartesiano (municípios × anos municipais) com metadados do IBGE.

    Para cada `ano_municipal` deriva `ano_eleicao_municipal_anterior = X - 4`
    via `MUNICIPAL_TO_MUNICIPAL_ANTERIOR`. Levanta se algum ano não estiver
    mapeado (eixo do modelo prefeito depende desse gap).
    """
    required = {"id_municipio", "sigla_uf", "nome"}
    missing = required - set(diretorio.columns)
    if missing:
        raise ValueError(f"diretorio sem colunas: {sorted(missing)}")

    d = diretorio.copy()
    d["id_municipio"] = d["id_municipio"].astype("string")
    d = d.drop_duplicates(subset=["id_municipio"])

    anos = sorted({int(a) for a in anos_municipais})
    if not anos:
        raise ValueError("anos_municipais vazio")

    d["_k"] = 1
    a = pd.DataFrame({"ano_municipal": anos, "_k": 1})
    out = d.merge(a, on="_k").drop(columns="_k")

    out["ano_eleicao_municipal_anterior"] = out["ano_municipal"].map(
        MUNICIPAL_TO_MUNICIPAL_ANTERIOR
    )
    if out["ano_eleicao_municipal_anterior"].isna().any():
        anos_sem_map = sorted(
            set(
                out.loc[
                    out["ano_eleicao_municipal_anterior"].isna(), "ano_municipal"
                ].unique()
            )
        )
        # Anos antes do primeiro ano mapeado (ex: 1996) não têm eleição
        # municipal anterior nos dados — sem prefeito vigente pra anexar,
        # eles caem fora do painel modelado. Mantidos só em prefeito_long
        # como base histórica para lag dos anos seguintes.
        anos_min = min(MUNICIPAL_TO_MUNICIPAL_ANTERIOR.keys())
        anos_anteriores = [a for a in anos_sem_map if a < anos_min]
        anos_desconhecidos = [a for a in anos_sem_map if a >= anos_min]
        if anos_desconhecidos:
            raise KeyError(
                f"anos municipais sem mapping anterior: {anos_desconhecidos}. "
                "Atualize MUNICIPAL_TO_MUNICIPAL_ANTERIOR em src.features.panel."
            )
        if anos_anteriores:
            logger.warning(
                "scaffold_municipio_ano_municipal: anos %s não têm eleição "
                "municipal anterior mapeada — removidos do painel modelado "
                "(serão usados só como histórico para lag).",
                anos_anteriores,
            )
            out = out[out["ano_eleicao_municipal_anterior"].notna()].copy()
    out["ano_eleicao_municipal_anterior"] = out[
        "ano_eleicao_municipal_anterior"
    ].astype("int64")

    for col in ("regiao", "capital_uf", "id_municipio_tse"):
        if col not in out.columns:
            out[col] = pd.NA

    cols = [
        "id_municipio",
        "sigla_uf",
        "nome",
        "regiao",
        "capital_uf",
        "ano_municipal",
        "ano_eleicao_municipal_anterior",
    ]
    return out[cols].reset_index(drop=True)


def construir_painel_mestre_municipal(
    diretorio: pd.DataFrame,
    df_prefeito: pd.DataFrame,
    df_partidos_prefeito: pd.DataFrame | None = None,
    anos_municipais: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Painel mestre para o modelo prefeito: 1 linha por (mun, ano_municipal).

    Anexa info do PREFEITO VIGENTE (vencedor da eleição municipal X-4) —
    mesma semântica que o painel presidencial, só com gap diferente.

    Args:
        diretorio: IBGE municípios.
        df_prefeito: resultados_candidato_municipio (cargo=prefeito, 1t).
        df_partidos_prefeito: partidos_prefeito (pra coligação).
        anos_municipais: default = MODE_CFG["anos_municipal"].

    Returns:
        DataFrame com id_municipio × ano_municipal e colunas mayor_*
        (do prefeito VIGENTE à época, eleito em X-4).
    """
    anos = list(anos_municipais or MODE_CFG["anos_municipal"])

    vencedores = prefeito_vencedor_por_eleicao(df_prefeito)
    vencedores = anexar_coligacao_prefeito(vencedores, df_partidos_prefeito)

    scaff = scaffold_municipio_ano_municipal(diretorio, anos)

    painel = scaff.merge(
        vencedores,
        left_on=["ano_eleicao_municipal_anterior", "id_municipio"],
        right_on=["ano_eleicao_municipal", "id_municipio"],
        how="left",
        suffixes=("", "_prefeito"),
    )
    # `ano_eleicao_municipal` veio do vencedor (= X-4). Mantemos o nome
    # consistente do painel presidencial pra reuso de features.
    if "ano_eleicao_municipal" in painel.columns:
        # Se existirem ambas, mantém ano_eleicao_municipal como X-4.
        pass

    if "sigla_uf_prefeito" in painel.columns:
        mismatches = (
            (painel["sigla_uf_prefeito"].notna())
            & (painel["sigla_uf"] != painel["sigla_uf_prefeito"])
        ).sum()
        if mismatches:
            logger.warning(
                "UF diretório ≠ UF prefeito em %d linha(s); mantendo UF do diretório",
                int(mismatches),
            )
        painel = painel.drop(columns=["sigla_uf_prefeito"])

    sem_prefeito = painel["mayor_numero"].isna().sum()
    logger.info(
        "painel_municipal: %d linhas (%d municípios × %d anos_municipal); %d sem prefeito anexado",
        len(painel),
        painel["id_municipio"].nunique(),
        len(set(painel["ano_municipal"])),
        int(sem_prefeito),
    )
    return painel


__all__ = [
    "PRESIDENCIAL_TO_MUNICIPAL",
    "MUNICIPAL_TO_MUNICIPAL_ANTERIOR",
    "prefeito_vencedor_por_eleicao",
    "anexar_coligacao_prefeito",
    "scaffold_municipio_ano",
    "scaffold_municipio_ano_municipal",
    "construir_painel_mestre",
    "construir_painel_mestre_municipal",
]
