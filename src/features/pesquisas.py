"""
src.features.pesquisas — feature de intenção de voto pré-eleição.

Endereça #60 (débito identificado em Fase 4): partidos cujo regime
muda entre calibração e teste (ex.: PL 2022 com migração Bolsonaro
PSL→PL) não são captados pelos lags históricos. Pesquisas pré-1º turno
são o sinal mais direto disponível pra esse caso.

Saída: uma coluna `share_pesquisa_nacional` por (ano × sigla_partido),
broadcast pra todas as linhas do long table.

Notas:
  * O CSV de entrada (`data/raw/pesquisas_nacional.csv`) é manual e
    tem coluna `fonte` pra rastrear procedência. Atualizar a cada nova
    eleição (incluindo 2026 quando rolar o pleito).
  * Granularidade nacional é PoC. Próxima iteração: granularidade UF
    via Datafolha estadual onde houver, com fallback para o nacional.
  * Anos sem pesquisa ou partidos não pesquisados → NaN. LightGBM
    lida com NaN nativamente.
  * Universo: a CSV cobre só candidatos relevantes em cada eleição
    (~3-7 por ano). Outros partidos no painel ficam NaN — comportamento
    correto: ausência de pesquisa É sinal de irrelevância.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def carregar_pesquisas_nacional(path: str | Path) -> pd.DataFrame:
    """Lê CSV de pesquisas nacionais.

    Espera colunas: ano, sigla_partido, share_pesquisa, [nome_candidato,
    fonte, obs].

    Returns:
        DataFrame com [ano (int), sigla_partido (str), share_pesquisa
        (float ∈ [0,1])].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"pesquisas_nacional não encontrado em {path}. "
            "Veja data/raw/pesquisas_nacional.csv pra schema."
        )
    df = pd.read_csv(path)
    cols_obrig = {"ano", "sigla_partido", "share_pesquisa"}
    missing = cols_obrig - set(df.columns)
    if missing:
        raise ValueError(f"pesquisas: colunas ausentes {missing}")
    df["ano"] = df["ano"].astype("int64")
    df["sigla_partido"] = df["sigla_partido"].astype("string")
    df["share_pesquisa"] = df["share_pesquisa"].astype("float64")
    if (df["share_pesquisa"] < 0).any() or (df["share_pesquisa"] > 1).any():
        raise ValueError("pesquisas: share_pesquisa fora de [0, 1]")
    n_pesq = len(df)
    n_anos = df["ano"].nunique()
    n_partidos = df["sigla_partido"].nunique()
    logger.info(
        "carregar_pesquisas_nacional: %d linhas (%d anos × %d partidos únicos)",
        n_pesq, n_anos, n_partidos,
    )
    return df[["ano", "sigla_partido", "share_pesquisa"]].copy()


def aplicar_pesquisa_nacional(
    long: pd.DataFrame,
    pesquisas: pd.DataFrame,
    *,
    ano_col: str = "ano_presidencial",
    partido_col: str = "sigla_partido",
) -> pd.DataFrame:
    """Anexa `share_pesquisa_nacional` ao long via merge por (ano, partido).

    Args:
        long: DataFrame com pelo menos [ano_col, partido_col].
        pesquisas: saída de carregar_pesquisas_nacional() — colunas
            [ano, sigla_partido, share_pesquisa].
        ano_col: 'ano_presidencial' (default) ou 'ano_municipal'. No
            eixo municipal a feature ainda usa pesquisa presidencial
            do mesmo ano (proxy nacional do humor político).
        partido_col: 'sigla_partido' (default).

    Returns:
        long com nova coluna `share_pesquisa_nacional` (float, NaN onde
        não há pesquisa).
    """
    required = {ano_col, partido_col}
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"long sem colunas: {sorted(missing)}")

    pq = pesquisas.rename(columns={
        "ano": ano_col,
        "sigla_partido": partido_col,
        "share_pesquisa": "share_pesquisa_nacional",
    })

    # Garantir tipos compatíveis pro merge
    long_copy = long.copy()
    long_copy[ano_col] = long_copy[ano_col].astype("int64")
    long_copy[partido_col] = long_copy[partido_col].astype("string")
    pq[ano_col] = pq[ano_col].astype("int64")
    pq[partido_col] = pq[partido_col].astype("string")

    n0 = len(long_copy)
    out = long_copy.merge(pq, on=[ano_col, partido_col], how="left")
    if len(out) != n0:
        raise RuntimeError(
            f"aplicar_pesquisa_nacional: merge expandiu linhas "
            f"({n0} -> {len(out)}). Pesquisas tem (ano, partido) duplicado?"
        )

    n_com_pesq = int(out["share_pesquisa_nacional"].notna().sum())
    pct = n_com_pesq / n0 * 100 if n0 > 0 else 0.0
    logger.info(
        "aplicar_pesquisa_nacional: %d/%d linhas com pesquisa (%.1f%%)",
        n_com_pesq, n0, pct,
    )
    # Sanity: anos cobertos
    anos_cobertos = sorted(
        out.loc[out["share_pesquisa_nacional"].notna(), ano_col].unique().tolist()
    )
    if anos_cobertos:
        logger.info("pesquisas cobrem anos: %s", anos_cobertos)

    return out


def features_pesquisa(
    long: pd.DataFrame,
    pesquisas_path: str | Path,
    *,
    ano_col: str = "ano_presidencial",
    partido_col: str = "sigla_partido",
) -> pd.DataFrame:
    """Conveniência: carrega CSV e aplica feature em um passo.

    Returns:
        long + coluna `share_pesquisa_nacional`.
    """
    pesquisas = carregar_pesquisas_nacional(pesquisas_path)
    return aplicar_pesquisa_nacional(
        long, pesquisas, ano_col=ano_col, partido_col=partido_col,
    )


def carregar_pesquisas_uf(path: str | Path) -> pd.DataFrame:
    """Lê CSV de pesquisas estaduais.

    Schema esperado: ano, sigla_uf, sigla_partido, share_pesquisa,
    [nome_candidato, fonte, obs].

    UFs ausentes do CSV ficam sem entrada e cairão no fallback nacional
    em `aplicar_pesquisa_uf`.

    Returns:
        DataFrame com [ano (int), sigla_uf (str), sigla_partido (str),
        share_pesquisa (float ∈ [0,1])].
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"pesquisas_uf não encontrado em {path}. "
            "Veja data/raw/pesquisas_uf.csv pra schema."
        )
    df = pd.read_csv(path)
    cols_obrig = {"ano", "sigla_uf", "sigla_partido", "share_pesquisa"}
    missing = cols_obrig - set(df.columns)
    if missing:
        raise ValueError(f"pesquisas_uf: colunas ausentes {missing}")
    df["ano"] = df["ano"].astype("int64")
    df["sigla_uf"] = df["sigla_uf"].astype("string")
    df["sigla_partido"] = df["sigla_partido"].astype("string")
    df["share_pesquisa"] = df["share_pesquisa"].astype("float64")
    if (df["share_pesquisa"] < 0).any() or (df["share_pesquisa"] > 1).any():
        raise ValueError("pesquisas_uf: share_pesquisa fora de [0, 1]")
    n_pesq = len(df)
    n_anos = df["ano"].nunique()
    n_ufs = df["sigla_uf"].nunique()
    n_partidos = df["sigla_partido"].nunique()
    logger.info(
        "carregar_pesquisas_uf: %d linhas (%d anos × %d UFs × %d partidos únicos)",
        n_pesq, n_anos, n_ufs, n_partidos,
    )
    return df[["ano", "sigla_uf", "sigla_partido", "share_pesquisa"]].copy()


def aplicar_pesquisa_uf(
    long: pd.DataFrame,
    pesquisas_uf: pd.DataFrame,
    pesquisas_nacional: pd.DataFrame,
    *,
    ano_col: str = "ano_presidencial",
    uf_col: str = "sigla_uf",
    partido_col: str = "sigla_partido",
) -> pd.DataFrame:
    """Anexa `share_pesquisa_uf` ao long via merge UF com fallback nacional.

    Lógica:
      1. Para cada (ano, UF, partido): tentar pesquisa estadual primeiro.
      2. Se estadual ausente: usar pesquisa nacional do mesmo (ano, partido).
      3. Se nacional também ausente: NaN.

    Adiciona também coluna binária `pesquisa_uf_disponivel` (1 se valor
    veio de pesquisa estadual, 0 se veio de fallback nacional, NaN se
    ausente em ambos). Útil pro LGBM modular confiança regional.

    Args:
        long: DataFrame com [ano_col, uf_col, partido_col].
        pesquisas_uf: saída de carregar_pesquisas_uf().
        pesquisas_nacional: saída de carregar_pesquisas_nacional().
        ano_col, uf_col, partido_col: nomes das colunas-chave.

    Returns:
        long + colunas [share_pesquisa_uf, pesquisa_uf_disponivel].
    """
    required = {ano_col, uf_col, partido_col}
    missing = required - set(long.columns)
    if missing:
        raise ValueError(f"long sem colunas: {sorted(missing)}")

    # Renomear colunas dos pesquisas pra alinhar com long
    pq_uf = pesquisas_uf.rename(columns={
        "ano": ano_col,
        "sigla_uf": uf_col,
        "sigla_partido": partido_col,
        "share_pesquisa": "share_pesquisa_estadual",
    })
    pq_nac = pesquisas_nacional.rename(columns={
        "ano": ano_col,
        "sigla_partido": partido_col,
        "share_pesquisa": "share_pesquisa_nac_fallback",
    })

    # Tipos consistentes pro merge
    long_copy = long.copy()
    long_copy[ano_col] = long_copy[ano_col].astype("int64")
    long_copy[uf_col] = long_copy[uf_col].astype("string")
    long_copy[partido_col] = long_copy[partido_col].astype("string")
    pq_uf[ano_col] = pq_uf[ano_col].astype("int64")
    pq_uf[uf_col] = pq_uf[uf_col].astype("string")
    pq_uf[partido_col] = pq_uf[partido_col].astype("string")
    pq_nac[ano_col] = pq_nac[ano_col].astype("int64")
    pq_nac[partido_col] = pq_nac[partido_col].astype("string")

    n0 = len(long_copy)
    out = long_copy.merge(
        pq_uf, on=[ano_col, uf_col, partido_col], how="left",
    )
    if len(out) != n0:
        raise RuntimeError(
            f"aplicar_pesquisa_uf: merge UF expandiu linhas "
            f"({n0} -> {len(out)}). pesquisas_uf tem (ano, uf, partido) duplicado?"
        )
    out = out.merge(
        pq_nac, on=[ano_col, partido_col], how="left",
    )
    if len(out) != n0:
        raise RuntimeError(
            f"aplicar_pesquisa_uf: merge nacional expandiu linhas "
            f"({n0} -> {len(out)}). pesquisas_nacional tem (ano, partido) duplicado?"
        )

    # Fallback: usar UF se houver, senão nacional
    out["share_pesquisa_uf"] = out["share_pesquisa_estadual"].fillna(
        out["share_pesquisa_nac_fallback"]
    )
    # Disponibilidade: 1.0 se UF original (não-NaN), 0.0 se caiu no
    # fallback nacional, NaN se não há nem nacional.
    import numpy as np
    out["pesquisa_uf_disponivel"] = np.where(
        out["share_pesquisa_estadual"].notna(),
        1.0,
        np.where(out["share_pesquisa_nac_fallback"].notna(), 0.0, np.nan),
    )

    out = out.drop(columns=["share_pesquisa_estadual", "share_pesquisa_nac_fallback"])

    n_uf = int((out["pesquisa_uf_disponivel"] == 1.0).sum())
    n_fallback = int((out["pesquisa_uf_disponivel"] == 0.0).sum())
    n_nan = int(out["pesquisa_uf_disponivel"].isna().sum())
    logger.info(
        "aplicar_pesquisa_uf: %d linhas pesquisa estadual, %d fallback nacional, %d NaN",
        n_uf, n_fallback, n_nan,
    )
    return out


def features_pesquisa_com_uf(
    long: pd.DataFrame,
    pesquisas_uf_path: str | Path,
    pesquisas_nacional_path: str | Path,
    *,
    ano_col: str = "ano_presidencial",
    uf_col: str = "sigla_uf",
    partido_col: str = "sigla_partido",
) -> pd.DataFrame:
    """Conveniência: carrega ambos CSVs e aplica feature em um passo."""
    pesquisas_uf = carregar_pesquisas_uf(pesquisas_uf_path)
    pesquisas_nac = carregar_pesquisas_nacional(pesquisas_nacional_path)
    return aplicar_pesquisa_uf(
        long, pesquisas_uf, pesquisas_nac,
        ano_col=ano_col, uf_col=uf_col, partido_col=partido_col,
    )


__all__ = [
    "carregar_pesquisas_nacional",
    "carregar_pesquisas_uf",
    "aplicar_pesquisa_nacional",
    "aplicar_pesquisa_uf",
    "features_pesquisa",
    "features_pesquisa_com_uf",
]
