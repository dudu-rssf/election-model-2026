"""
src.models.features — seleção e preparação de features pro pipeline de ML.

Função principal: `preparar_X_y(df_features, ...)` que devolve:
    X (DataFrame com dtypes corretos pro LightGBM),
    y (Series com share_1t),
    meta (DataFrame com ano_presidencial/id_municipio/... pra análise).

Design:
  * **Listagem explícita** das features. Nada de "usa tudo menos o target".
    A Fase 3 adiciona colunas auxiliares (sufixadas `_historical`, `_continuity`
    etc. via suffixes do merge); se deixássemos "usa tudo", uma coluna nova
    acidentalmente vazaria pro modelo. Listar é mais chato, mas seguro.
  * **Categóricas como pandas Categorical**: o LightGBM aceita nativamente
    via `categorical_feature=auto` ou lista explícita, e lida com NAs.
  * **Int64 (pandas nullable) → float64**: o LightGBM não aceita Int64
    nullable diretamente. Convertemos pra float; a info de NA é preservada
    com NaN (LightGBM respeita NaN natural).
  * **Validação de vazamento**: asserta que `votos` e `total_votos_mun`
    não entram no X.
  * **Validação de ano**: confere que `ano_presidencial` é apenas pra
    particionar, nunca feature.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Contrato de colunas
# ------------------------------------------------------------
FEATURES_CATEGORICAS: list[str] = [
    "sigla_uf",
    "regiao",
    "porte",
    "continuidade_classe",
    "sigla_partido",
]

# Binárias 0/1 entram como numéricas (LGBM trata igual, e o NaN é limpo)
FEATURES_BINARIAS: list[str] = [
    "capital_uf",
    "primeiro_mandato_prefeito",
    "alinhado_prefeito_partido",
    "alinhado_prefeito_coligacao",
    "alinhado_gov_vigente_partido",
    "alinhado_gov_vigente_coligacao",
    "alinhado_gov_concorrente_partido",
    "alinhado_gov_concorrente_coligacao",
]

FEATURES_NUMERICAS: list[str] = [
    "log_eleitorado",
    "share_prefeito_local",
    "margem_prefeito",
    "indice_continuidade",
    "anos_consecutivos_mesmo_partido",
    "anos_consecutivos_mesmo_grupo",
    "lag_share_1t",
    "lag2_share_1t",
    "swing_share_1t",
    "volatilidade_partido",
    "share_dep_federal_partido",
]

# Colunas auxiliares que não entram no X mas ficam em `meta` pra análise.
META_COLS: list[str] = [
    "ano_presidencial",
    "id_municipio",
    "sigla_uf",
    "sigla_partido",
    "numero_candidato",
    "nome_candidato",
    "votos",
    "total_votos_mun",
]

TARGET_COL: str = "share_1t"

# Explicitamente NÃO entra no X (evita vazamento)
COLUNAS_PROIBIDAS: set[str] = {
    "votos",
    "total_votos_mun",
    "share_1t",
    "ano_presidencial",
    "numero_candidato",
    "nome_candidato",
    "id_municipio",
}


@dataclass
class PreparedData:
    """Container com dados prontos pro LightGBM."""
    X: pd.DataFrame
    y: pd.Series
    meta: pd.DataFrame
    cat_features: list[str]

    def __len__(self) -> int:
        return len(self.X)


def _validar_schema(df: pd.DataFrame) -> None:
    esperadas = (
        set(FEATURES_CATEGORICAS) | set(FEATURES_BINARIAS) | set(FEATURES_NUMERICAS)
        | {TARGET_COL, "ano_presidencial", "id_municipio"}
    )
    faltando = esperadas - set(df.columns)
    if faltando:
        raise ValueError(
            f"df_features sem colunas obrigatórias: {sorted(faltando)}. "
            "Rode `scripts/03_features.py` antes."
        )


def _to_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Converte colunas pra pandas Categorical (LightGBM aceita nativo)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        # Padroniza como string antes (tratamento uniforme de NA)
        s = out[c].astype("string")
        out[c] = s.astype("category")
    return out


def _binarias_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Int64 nullable → float64 (LGBM aceita, NA vira NaN)."""
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        # Int64 → float preserva NaN; bool → float (0.0/1.0)
        out[c] = out[c].astype("Float64").astype("float64")
    return out


def _numericas_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float64")
    return out


def preparar_X_y(
    df_features: pd.DataFrame,
    dropna_target: bool = True,
) -> PreparedData:
    """Monta X, y, meta a partir do DataFrame de features consolidado.

    Args:
        df_features: saída de `scripts/03_features.py` (features.parquet).
        dropna_target: se True, remove linhas com target NaN (não tem como
            treinar/avaliar sem target; mantemos só para debug).

    Returns:
        PreparedData com X, y, meta, cat_features.
    """
    _validar_schema(df_features)

    df = df_features.copy()
    if dropna_target:
        n0 = len(df)
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        n1 = len(df)
        if n0 != n1:
            logger.info("preparar_X_y: dropei %d linhas com target NaN", n0 - n1)

    # X: só as colunas selecionadas, na ordem (categóricas + binárias + numéricas)
    feature_cols = FEATURES_CATEGORICAS + FEATURES_BINARIAS + FEATURES_NUMERICAS
    X = df[feature_cols].copy()
    X = _to_categorical(X, FEATURES_CATEGORICAS)
    X = _binarias_to_float(X, FEATURES_BINARIAS)
    X = _numericas_to_float(X, FEATURES_NUMERICAS)

    # Guard: nenhuma coluna proibida escapou pro X
    vazamento = set(X.columns) & COLUNAS_PROIBIDAS
    if vazamento:
        raise RuntimeError(
            f"vazamento de feature detectado: {sorted(vazamento)}. "
            "Revise FEATURES_* em src/models/features.py."
        )

    y = df[TARGET_COL].astype("float64")
    meta_presentes = [c for c in META_COLS if c in df.columns]
    meta = df[meta_presentes].reset_index(drop=True)

    logger.info(
        "preparar_X_y: %d linhas × %d features (%d cat + %d bin + %d num)",
        len(X), X.shape[1],
        len(FEATURES_CATEGORICAS), len(FEATURES_BINARIAS), len(FEATURES_NUMERICAS),
    )
    return PreparedData(X=X, y=y, meta=meta, cat_features=list(FEATURES_CATEGORICAS))


def split_temporal(
    prep: PreparedData,
    anos_treino: list[int],
    ano_teste: int,
) -> tuple[PreparedData, PreparedData]:
    """Separa `prep` em treino/teste pelo ano_presidencial (na meta).

    O ano vive em `prep.meta["ano_presidencial"]`, não em `X` — split é
    feito por índice da meta.
    """
    if "ano_presidencial" not in prep.meta.columns:
        raise ValueError("meta sem ano_presidencial — impossível fazer split temporal")
    anos_treino_set = {int(a) for a in anos_treino}
    ano_teste_i = int(ano_teste)
    if ano_teste_i in anos_treino_set:
        raise ValueError(f"ano_teste {ano_teste_i} está em anos_treino {anos_treino_set}")

    ano = prep.meta["ano_presidencial"].astype("int64")
    mask_tr = ano.isin(anos_treino_set)
    mask_te = ano == ano_teste_i

    def _subset(mask: pd.Series) -> PreparedData:
        idx = mask[mask].index
        return PreparedData(
            X=prep.X.loc[idx].reset_index(drop=True),
            y=prep.y.loc[idx].reset_index(drop=True),
            meta=prep.meta.loc[idx].reset_index(drop=True),
            cat_features=list(prep.cat_features),
        )

    tr = _subset(mask_tr)
    te = _subset(mask_te)

    # Sanity: verifica cobertura de categorias entre treino e teste.
    # Valor categórico presente só no teste vira um "level" desconhecido
    # pro LGBM. Não é fatal (LGBM trata como NaN), mas vale logar.
    for c in prep.cat_features:
        niveis_tr = set(tr.X[c].dropna().unique().tolist())
        niveis_te = set(te.X[c].dropna().unique().tolist())
        so_no_teste = niveis_te - niveis_tr
        if so_no_teste:
            logger.warning(
                "split_temporal: categoria %r tem %d nível(is) só no teste: %s",
                c, len(so_no_teste), sorted(str(x) for x in so_no_teste)[:5],
            )

    logger.info(
        "split_temporal: treino=%d (anos %s), teste=%d (ano %d)",
        len(tr), sorted(anos_treino_set), len(te), ano_teste_i,
    )
    return tr, te


__all__ = [
    "FEATURES_CATEGORICAS",
    "FEATURES_BINARIAS",
    "FEATURES_NUMERICAS",
    "META_COLS",
    "TARGET_COL",
    "PreparedData",
    "preparar_X_y",
    "split_temporal",
]
