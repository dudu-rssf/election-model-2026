"""
src.features.partido_sucessao — resolução de identidade partidária.

Siglas não são IDs estáveis: PL 2018 (partido pequeno de centro) ≠ PL 2022
(partido do Bolsonaro). Este módulo centraliza o mapeamento
`partido_sucessao` do config.yaml em uma função que, dado (partido, ano),
devolve a "sigla canônica" — a sigla sob a qual o campo político deveria
ser agrupado pra cálculos de lag/histórico.

Convenção: sigla_canonica = sigla_predecessora se houver mapeamento,
senão a própria sigla. Isso faz com que:

    PL_2022 e PSL_2018   → ambos mapeiam pra canônica "PSL" → compartilham histórico
    PL_2014 e PL_2018    → ambos sem mapeamento → canônica = "PL"
    UNIÃO_2022 e DEM_2018 → canônica "DEM"

Se quiser desabilitar a sucessão, passe `sucessoes=None` ou `{}`. A
coluna `sigla_canonica` vira uma cópia de `sigla_partido`.

NOTA: Este é um workaround explícito. A solução ideal é substituir por
feature dinâmica de intenção de voto via pesquisas eleitorais (task #60).
"""
from __future__ import annotations

import logging
from typing import Any, Mapping

import pandas as pd

logger = logging.getLogger(__name__)


SucessoesDict = Mapping[str, Mapping[int, str]]


def _normalizar_sucessoes(sucessoes: Any) -> dict[str, dict[int, str]]:
    """Valida e normaliza o dicionário vindo do YAML pra tipos certos."""
    if sucessoes is None:
        return {}
    if not isinstance(sucessoes, dict):
        raise TypeError(f"partido_sucessao deve ser dict; recebi {type(sucessoes).__name__}")
    out: dict[str, dict[int, str]] = {}
    for sigla, mapa in sucessoes.items():
        if not isinstance(sigla, str):
            raise TypeError(f"chave de partido_sucessao deve ser str; recebi {sigla!r}")
        if not isinstance(mapa, dict):
            raise TypeError(f"partido_sucessao[{sigla!r}] deve ser dict; recebi {type(mapa).__name__}")
        inner: dict[int, str] = {}
        for ano, predecessor in mapa.items():
            if not isinstance(predecessor, str):
                raise TypeError(
                    f"partido_sucessao[{sigla!r}][{ano!r}] deve ser str; recebi {predecessor!r}"
                )
            inner[int(ano)] = predecessor
        out[sigla] = inner
    return out


def resolver_sigla_canonica(
    partido: str | None,
    ano: int | None,
    sucessoes: SucessoesDict | None = None,
) -> str | None:
    """Resolve (partido, ano) pra sigla canônica.

    Args:
        partido: sigla (ex: "PL"). NA ou None devolve None.
        ano: ano da eleição. NA ou None devolve `partido` inalterado.
        sucessoes: dict vindo de config.yaml ou equivalente.

    Returns:
        Sigla canônica (string) ou None se entrada for NA.
    """
    if partido is None or pd.isna(partido):
        return None
    partido_s = str(partido)
    if ano is None or pd.isna(ano):
        return partido_s
    mapa = _normalizar_sucessoes(sucessoes)
    mapa_partido = mapa.get(partido_s, {})
    return mapa_partido.get(int(ano), partido_s)


def aplicar_sucessao(
    df: pd.DataFrame,
    sucessoes: SucessoesDict | None,
    col_partido: str = "sigla_partido",
    col_ano: str = "ano_presidencial",
    col_saida: str = "sigla_canonica",
) -> pd.DataFrame:
    """Adiciona coluna `sigla_canonica` ao DataFrame.

    Args:
        df: entrada. Precisa ter `col_partido` e `col_ano`.
        sucessoes: dicionário de sucessões. None ou {} = sem mapeamento
            (sigla_canonica = sigla_partido).
        col_partido, col_ano: nomes das colunas origem.
        col_saida: nome da nova coluna.

    Returns:
        Cópia de df com coluna extra `col_saida`.
    """
    for c in (col_partido, col_ano):
        if c not in df.columns:
            raise ValueError(f"df sem coluna {c!r}")

    mapa = _normalizar_sucessoes(sucessoes)
    out = df.copy()

    if not mapa:
        # Sem mapeamento: canônica = sigla pura
        out[col_saida] = out[col_partido].astype("string")
        return out

    # Aplicação vetorizada: monta Series só pros pares que têm mapping
    partido_arr = out[col_partido].astype("string")
    ano_arr = pd.to_numeric(out[col_ano], errors="coerce").astype("Int64")

    canonica = partido_arr.copy()
    n_aplicados = 0
    for sigla, por_ano in mapa.items():
        for ano, predecessor in por_ano.items():
            mask = (partido_arr == sigla) & (ano_arr == ano)
            n = int(mask.sum())
            if n > 0:
                canonica.loc[mask] = predecessor
                n_aplicados += n
                logger.info(
                    "partido_sucessao: %s %d → %s (%d linhas)",
                    sigla, ano, predecessor, n,
                )
                # Sucessão silenciosa: quando o predecessor declarado não aparece
                # como sigla_partido em nenhum ano anterior do DataFrame, o lookup
                # de histórico (e.g. _lag_por_sigla_canonica em historical.py) vai
                # retornar NaN — a sucessão fica formalmente aplicada mas sem
                # efeito prático. Bug típico: predecessor que só existe em outro
                # eixo eleitoral (e.g. UNIÃO 2022 → DEM, mas DEM não teve
                # candidato presidencial). Avisar no log torna o caso audível.
                tem_historico = bool(
                    ((partido_arr == predecessor) & (ano_arr < ano)).any()
                )
                if not tem_historico:
                    logger.warning(
                        "partido_sucessao: %s %d → %s aplicado em %d linhas, "
                        "mas %r não aparece como sigla_partido em nenhum ano "
                        "< %d no DataFrame. Lag canônico ficará NaN — "
                        "predecessor pode existir só em outro eixo eleitoral.",
                        sigla, ano, predecessor, n, predecessor, ano,
                    )
    logger.info("aplicar_sucessao: %d linhas remapeadas", n_aplicados)
    out[col_saida] = canonica
    return out


__all__ = [
    "SucessoesDict",
    "resolver_sigla_canonica",
    "aplicar_sucessao",
]
