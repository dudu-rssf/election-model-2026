"""
src.models.features_prefeito — seleção e preparação de features para o modelo
de prefeito (Fase 4.5, eixo `ano_municipal`).

Função principal: `preparar_X_y(df_features, ...)` que devolve:
    X (DataFrame com dtypes corretos pro LightGBM),
    y (Series com share_1t),
    meta (DataFrame com ano_municipal/id_municipio/... pra análise).

Diferenças vs `src.models.features` (modelo presidencial):
  * Coluna de eixo é `ano_municipal` (não `ano_presidencial`).
  * Sem features de governador concorrente — não há eleição estadual no
    ano municipal. Restam apenas as `alinhado_gov_vigente_*`.
  * Demais features são as mesmas (estruturais, históricas, continuidade,
    poder local, dep. federal vigente).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------
# Contrato de colunas
# ------------------------------------------------------------
ANO_COL: str = "ano_municipal"

FEATURES_CATEGORICAS: list[str] = [
    "sigla_uf",
    "regiao",
    "porte",
    "continuidade_classe",
    "sigla_partido",
]

# Binárias 0/1 entram como numéricas (LGBM trata igual, e o NaN é limpo).
# Sem `alinhado_gov_concorrente_*` — não há eleição estadual no ano municipal.
FEATURES_BINARIAS: list[str] = [
    "capital_uf",
    "primeiro_mandato_prefeito",
    "alinhado_prefeito_partido",
    "alinhado_prefeito_coligacao",
    "alinhado_gov_vigente_partido",
    "alinhado_gov_vigente_coligacao",
]

FEATURES_NUMERICAS: list[str] = [
    "log_eleitorado",
    "share_prefeito_local",
    "margem_prefeito",
    "indice_continuidade",
    "anos_consecutivos_mesmo_partido",
    "anos_consecutivos_mesmo_grupo",
    "lag_share_1t",
    "lag_share_1t_sucessao",
    "lag2_share_1t",
    "swing_share_1t",
    "volatilidade_partido",
    "share_dep_federal_partido",
]

# Colunas auxiliares que não entram no X mas ficam em `meta` pra análise.
META_COLS: list[str] = [
    ANO_COL,
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
    ANO_COL,
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
        | {TARGET_COL, ANO_COL, "id_municipio"}
    )
    faltando = esperadas - set(df.columns)
    if faltando:
        raise ValueError(
            f"df_features sem colunas obrigatórias: {sorted(faltando)}. "
            "Rode `scripts/03_features_prefeito.py` antes."
        )


def _to_categorical(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
        s = out[c].astype("string")
        out[c] = s.astype("category")
    return out


def _binarias_to_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            continue
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
    """Monta X, y, meta a partir do DataFrame de features consolidado (eixo municipal).

    Args:
        df_features: saída de `scripts/03_features_prefeito.py`
            (`features_prefeito.parquet`).
        dropna_target: se True, remove linhas com target NaN.
    """
    _validar_schema(df_features)

    df = df_features.copy()
    if dropna_target:
        n0 = len(df)
        df = df.dropna(subset=[TARGET_COL]).reset_index(drop=True)
        n1 = len(df)
        if n0 != n1:
            logger.info("preparar_X_y: dropei %d linhas com target NaN", n0 - n1)

    feature_cols = FEATURES_CATEGORICAS + FEATURES_BINARIAS + FEATURES_NUMERICAS
    X = df[feature_cols].copy()
    X = _to_categorical(X, FEATURES_CATEGORICAS)
    X = _binarias_to_float(X, FEATURES_BINARIAS)
    X = _numericas_to_float(X, FEATURES_NUMERICAS)

    vazamento = set(X.columns) & COLUNAS_PROIBIDAS
    if vazamento:
        raise RuntimeError(
            f"vazamento de feature detectado: {sorted(vazamento)}. "
            "Revise FEATURES_* em src/models/features_prefeito.py."
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
    """Separa `prep` em treino/teste pelo ano_municipal (na meta)."""
    if ANO_COL not in prep.meta.columns:
        raise ValueError(f"meta sem {ANO_COL} — impossível fazer split temporal")
    anos_treino_set = {int(a) for a in anos_treino}
    ano_teste_i = int(ano_teste)
    if ano_teste_i in anos_treino_set:
        raise ValueError(f"ano_teste {ano_teste_i} está em anos_treino {anos_treino_set}")

    ano = prep.meta[ANO_COL].astype("int64")
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
    "ANO_COL",
    "FEATURES_CATEGORICAS",
    "FEATURES_BINARIAS",
    "FEATURES_NUMERICAS",
    "META_COLS",
    "TARGET_COL",
    "PreparedData",
    "preparar_X_y",
    "split_temporal",
]
