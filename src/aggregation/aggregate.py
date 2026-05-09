"""
src.aggregation.aggregate — agregação município → UF → nacional.

Fase 5: o LightGBM produz `share` por (município, partido). Para previsão
nacional/UF precisamos agregar via média ponderada pelo eleitorado, e
propagar a incerteza dos intervalos conformais (pred_lower/pred_upper de
SplitConformal/Mondrian/CQR) até o agregado.

Estratégia de propagação — Monte Carlo:

    1. Para cada linha (mun, partido), sortear `n_samples` valores
       de uma uniforme centrada em `pred` com semi-largura
       w_i = (pred_upper_i - pred_lower_i) / 2:
           x_i ~ Unif(pred_i - w_i, pred_i + w_i).
       Equivale a Unif(pred_lower, pred_upper) quando o intervalo é
       simétrico em torno de pred (caso do SplitConformal/Mondrian sem
       clip mordendo). A construção centrada preserva
       E[x_i] = pred_i mesmo quando o intervalo conformal foi clipado
       em 0/1 ou é assimétrico (e.g., CQR), evitando viés sistemático
       no agregado.
    2. Para cada sample, computar a média ponderada por eleitorado
       dentro do grupo de agregação (UF ou nacional) usando o
       eleitorado TOTAL como denominador (não o subset onde o
       partido competiu).
    3. Pegar percentis empíricos (alpha/2, 1-alpha/2) das `n_samples`
       agregações como [share_lower, share_upper].

Limitações conhecidas:

  * Independência entre partidos no mesmo município: o MC sortei cada
    (mun, partido) de forma independente, ignorando a restrição
    sum share = 1 por município. O efeito é alargar levemente o
    intervalo agregado (conservador).
  * Distribuição uniforme: os intervalos conformais são quantis, não
    densidades. Uniforme centrada em pred é aproximação simples.

API:
    df_uf = agregar_municipal_para_uf(
        preds, peso_col="total_votos_mun", pred_col="pred_LightGBM_v1_iso",
        ano_col="ano_presidencial",
        pred_lower_col="pred_lower_mondrian", pred_upper_col="pred_upper_mondrian",
        n_samples=1000, alpha=0.10, seed=42,
    )
    df_nac = agregar_uf_para_nacional(df_uf, ano_col="ano_presidencial")
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================
# Helpers
# ============================================================
def _validar_cols(df: pd.DataFrame, cols: list[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: colunas ausentes {missing}")


# ============================================================
# Agregação municipal -> UF
# ============================================================
def agregar_municipal_para_uf(
    preds: pd.DataFrame,
    *,
    peso_col: str,
    pred_col: str,
    ano_col: str,
    partido_col: str = "sigla_partido",
    uf_col: str = "sigla_uf",
    id_mun_col: str = "id_municipio",
    pred_lower_col: str | None = None,
    pred_upper_col: str | None = None,
    n_samples: int = 1000,
    alpha: float = 0.10,
    seed: int = 42,
    incluir_y_true: bool = True,
) -> pd.DataFrame:
    """Agrega `pred_col` de município para UF, ponderado por `peso_col`.

    Denominador: `eleitorado_uf` é a soma de `peso_col` por (ano, uf)
    deduplicando por `id_mun_col` — total da UF, igual para todos os
    partidos. Importante quando partidos não competem em todos os
    municípios da UF (caso prefeito).

        share_pred(uf, p) = sum_m peso[m] * pred[m, p] / sum_m peso[m]
                                                       (todos os m da UF)

    No presidencial os 11 candidatos competem em todos os municípios,
    então o denominador é o mesmo seja ou não condicional.

    Se `pred_lower_col` e `pred_upper_col` forem dados, computa
    share_lower/share_upper via MC com o mesmo denominador total.
    Se `incluir_y_true` e 'y_true' presente, agrega -> y_real.
    """
    cols_obrig = [ano_col, uf_col, partido_col, peso_col, pred_col]
    _validar_cols(preds, cols_obrig, "agregar_municipal_para_uf")
    if id_mun_col not in preds.columns:
        raise ValueError(
            f"agregar_municipal_para_uf: coluna {id_mun_col!r} ausente."
        )

    com_intervalos = pred_lower_col is not None and pred_upper_col is not None
    if com_intervalos:
        _validar_cols(preds, [pred_lower_col, pred_upper_col],
                      "agregar_municipal_para_uf")

    # Eleitorado total por (ano, uf), dedup por município
    elf_uf = (
        preds[[ano_col, uf_col, id_mun_col, peso_col]]
        .drop_duplicates([ano_col, uf_col, id_mun_col])
        .groupby([ano_col, uf_col], observed=True)[peso_col]
        .sum()
        .reset_index()
        .rename(columns={peso_col: "eleitorado_uf"})
    )
    n_mun_uf = (
        preds.drop_duplicates([ano_col, uf_col, id_mun_col])
        .groupby([ano_col, uf_col], observed=True)
        .size()
        .reset_index(name="n_municipios_uf")
    )

    df = preds[cols_obrig].copy()
    df = df.rename(columns={pred_col: "_pred", peso_col: "_w"})
    if incluir_y_true and "y_true" in preds.columns:
        df["_y"] = preds["y_true"].to_numpy()

    n0 = len(df)
    df = df.dropna(subset=["_pred", "_w"]).reset_index(drop=True)
    if len(df) < n0:
        logger.info("agregar_municipal_para_uf: dropei %d linhas com NaN",
                    n0 - len(df))

    keys = [ano_col, uf_col, partido_col]

    df["_wp"] = df["_w"] * df["_pred"]
    g = df.groupby(keys, observed=True, sort=True)
    out = g.agg(
        _wp_sum=("_wp", "sum"),
        n_municipios_partido=("_pred", "size"),
    ).reset_index()
    out = out.merge(elf_uf, on=[ano_col, uf_col], how="left")
    out = out.merge(n_mun_uf, on=[ano_col, uf_col], how="left")
    out["share_pred"] = out["_wp_sum"] / out["eleitorado_uf"]
    out = out.drop(columns=["_wp_sum"])

    if "_y" in df.columns:
        df["_wy"] = df["_w"] * df["_y"]
        gy = df.groupby(keys, observed=True, sort=True).agg(
            _wy_sum=("_wy", "sum"),
        ).reset_index()
        gy = gy.merge(elf_uf, on=[ano_col, uf_col], how="left")
        gy["y_real"] = gy["_wy_sum"] / gy["eleitorado_uf"]
        out = out.merge(gy[keys + ["y_real"]], on=keys, how="left")

    if com_intervalos:
        df_int = preds[cols_obrig + [id_mun_col, pred_lower_col, pred_upper_col]].copy()
        df_int = df_int.rename(columns={
            pred_col: "_pred", peso_col: "_w",
            pred_lower_col: "_lo", pred_upper_col: "_hi",
        })
        df_int = df_int.dropna(subset=["_pred", "_w", "_lo", "_hi"])
        df_int = df_int.merge(elf_uf, on=[ano_col, uf_col], how="left")
        df_int = df_int.rename(columns={"eleitorado_uf": "_w_total"})
        df_int = df_int.reset_index(drop=True)
        intervals = _monte_carlo_por_grupo(
            df_int, keys=keys, n_samples=n_samples, alpha=alpha,
            seed=seed, clip_unit=True, denom_col="_w_total",
        )
        out = out.merge(intervals, on=keys, how="left")

    logger.info(
        "agregar_municipal_para_uf: %d (mun×partido) -> %d (uf×partido), "
        "anos=%s, ufs=%d, partidos=%d",
        len(preds), len(out),
        sorted(out[ano_col].unique().tolist()),
        out[uf_col].nunique(),
        out[partido_col].nunique(),
    )
    return out


# ============================================================
# Agregação UF -> nacional
# ============================================================
def agregar_uf_para_nacional(
    preds_uf: pd.DataFrame,
    *,
    peso_col: str = "eleitorado_uf",
    share_col: str = "share_pred",
    ano_col: str = "ano_presidencial",
    partido_col: str = "sigla_partido",
    uf_col: str = "sigla_uf",
    share_lower_col: str | None = "share_lower",
    share_upper_col: str | None = "share_upper",
    incluir_y_real: bool = True,
    n_samples: int = 1000,
    alpha: float = 0.10,
    seed: int = 43,
) -> pd.DataFrame:
    """Agrega UF -> nacional, ponderado por `peso_col` (eleitorado_uf).

    Denominador nacional: soma de peso_col deduplicada por UF — total
    nacional, igual para todos os partidos. Importante quando partidos
    não aparecem em todas as UFs (caso prefeito).
    """
    cols_obrig = [ano_col, uf_col, partido_col, peso_col, share_col]
    _validar_cols(preds_uf, cols_obrig, "agregar_uf_para_nacional")

    elf_nac = (
        preds_uf[[ano_col, uf_col, peso_col]]
        .drop_duplicates([ano_col, uf_col])
        .groupby([ano_col], observed=True)[peso_col]
        .sum()
        .reset_index()
        .rename(columns={peso_col: "eleitorado_total"})
    )
    n_uf_total = (
        preds_uf.drop_duplicates([ano_col, uf_col])
        .groupby([ano_col], observed=True)
        .size()
        .reset_index(name="n_ufs_total")
    )

    df = preds_uf[cols_obrig].copy()
    df = df.rename(columns={share_col: "_pred", peso_col: "_w"})
    if incluir_y_real and "y_real" in preds_uf.columns:
        df["_y"] = preds_uf["y_real"].to_numpy()

    df = df.dropna(subset=["_pred", "_w"]).reset_index(drop=True)
    keys = [ano_col, partido_col]

    df["_wp"] = df["_w"] * df["_pred"]
    g = df.groupby(keys, observed=True, sort=True)
    out = g.agg(
        _wp_sum=("_wp", "sum"),
        n_ufs_partido=("_pred", "size"),
    ).reset_index()
    out = out.merge(elf_nac, on=[ano_col], how="left")
    out = out.merge(n_uf_total, on=[ano_col], how="left")
    out["share_pred"] = out["_wp_sum"] / out["eleitorado_total"]
    out = out.drop(columns=["_wp_sum"])
    out["n_ufs"] = out["n_ufs_partido"]  # compat

    if "_y" in df.columns:
        df["_wy"] = df["_w"] * df["_y"]
        gy = df.groupby(keys, observed=True, sort=True).agg(
            _wy_sum=("_wy", "sum"),
        ).reset_index()
        gy = gy.merge(elf_nac, on=[ano_col], how="left")
        gy["y_real"] = gy["_wy_sum"] / gy["eleitorado_total"]
        out = out.merge(gy[keys + ["y_real"]], on=keys, how="left")

    com_intervalos = (
        share_lower_col is not None
        and share_upper_col is not None
        and share_lower_col in preds_uf.columns
        and share_upper_col in preds_uf.columns
    )
    if com_intervalos:
        df_int = preds_uf[cols_obrig + [share_lower_col, share_upper_col]].copy()
        df_int = df_int.rename(columns={
            share_col: "_pred", peso_col: "_w",
            share_lower_col: "_lo", share_upper_col: "_hi",
        })
        df_int = df_int.dropna(subset=["_pred", "_w", "_lo", "_hi"])
        df_int = df_int.merge(elf_nac, on=[ano_col], how="left")
        df_int = df_int.rename(columns={"eleitorado_total": "_w_total"})
        df_int = df_int.reset_index(drop=True)
        intervals = _monte_carlo_por_grupo(
            df_int, keys=keys, n_samples=n_samples, alpha=alpha,
            seed=seed, clip_unit=True, denom_col="_w_total",
        )
        out = out.merge(intervals, on=keys, how="left")

    logger.info(
        "agregar_uf_para_nacional: %d (uf×partido) -> %d (ano×partido), "
        "anos=%s, partidos=%d",
        len(preds_uf), len(out),
        sorted(out[ano_col].unique().tolist()),
        out[partido_col].nunique(),
    )
    return out


# ============================================================
# Monte Carlo: kernel privado
# ============================================================
def _monte_carlo_por_grupo(
    df: pd.DataFrame,
    *,
    keys: list[str],
    n_samples: int,
    alpha: float,
    seed: int,
    clip_unit: bool = True,
    denom_col: str | None = None,
) -> pd.DataFrame:
    """Para cada grupo, sortear amostras uniformes centradas em _pred
    com semi-largura w_i = (_hi - _lo)/2, computar a média ponderada
    por _w, e devolver percentis (alpha/2, 1-alpha/2).

    Args:
        denom_col: se != None, usa o valor (constante por grupo) dessa
            coluna como denominador da média ponderada em vez de sum_i w_i.
            Útil quando o agregado tem denominador externo (e.g.,
            eleitorado total da UF).
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha fora de (0,1): {alpha}")
    if n_samples < 2:
        raise ValueError(f"n_samples >= 2 (got {n_samples})")
    if denom_col is not None and denom_col not in df.columns:
        raise ValueError(f"denom_col={denom_col!r} ausente em df")

    rng = np.random.default_rng(seed)

    rows = []
    q_lo_p = alpha / 2.0
    q_hi_p = 1.0 - alpha / 2.0

    for key_vals, sub in df.groupby(keys, observed=True, sort=True):
        n_g = len(sub)
        if n_g == 0:
            continue
        pred = sub["_pred"].to_numpy(dtype="float64")
        lo = sub["_lo"].to_numpy(dtype="float64")
        hi = sub["_hi"].to_numpy(dtype="float64")
        w = sub["_w"].to_numpy(dtype="float64")
        lo_eff = np.minimum(lo, hi)
        hi_eff = np.maximum(lo, hi)
        if denom_col is not None:
            denom_vals = sub[denom_col].to_numpy(dtype="float64")
            denom = float(denom_vals[0])
            if not np.allclose(denom_vals, denom):
                logger.warning(
                    "_monte_carlo_por_grupo: denom_col=%s não é constante "
                    "no grupo %s (min=%.2f max=%.2f); usando o primeiro.",
                    denom_col, key_vals, denom_vals.min(), denom_vals.max(),
                )
        else:
            denom = float(w.sum())
        if denom <= 0:
            continue
        half_w = 0.5 * (hi_eff - lo_eff)
        u = rng.uniform(-1.0, 1.0, size=(n_g, n_samples))
        samples = pred[:, None] + half_w[:, None] * u
        agg_arr = (samples * w[:, None]).sum(axis=0) / denom
        if clip_unit:
            np.clip(agg_arr, 0.0, 1.0, out=agg_arr)
        rows.append({
            **dict(zip(keys, key_vals if isinstance(key_vals, tuple) else (key_vals,))),
            "share_lower": float(np.quantile(agg_arr, q_lo_p)),
            "share_upper": float(np.quantile(agg_arr, q_hi_p)),
        })

    if not rows:
        return pd.DataFrame(columns=keys + ["share_lower", "share_upper"])
    return pd.DataFrame(rows)


# ============================================================
# Validação / diagnósticos
# ============================================================
@dataclass
class SomaUnitariaResult:
    soma_min: float
    soma_max: float
    soma_media: float
    n_grupos: int
    n_violacoes: int
    tolerancia: float
    detalhes: pd.DataFrame


def verificar_soma_unitaria(
    df_agg: pd.DataFrame,
    *,
    keys: list[str],
    share_col: str = "share_pred",
    tolerancia: float = 0.01,
) -> SomaUnitariaResult:
    """Verifica que share_col soma ~= 1 dentro de cada grupo `keys`."""
    _validar_cols(df_agg, keys + [share_col], "verificar_soma_unitaria")

    g = df_agg.groupby(keys, observed=True, sort=True)[share_col]
    g = g.sum().reset_index()
    g = g.rename(columns={share_col: "soma"})
    g["delta"] = (g["soma"] - 1.0).abs()
    violacoes = g[g["delta"] > tolerancia].copy()

    res = SomaUnitariaResult(
        soma_min=float(g["soma"].min()),
        soma_max=float(g["soma"].max()),
        soma_media=float(g["soma"].mean()),
        n_grupos=int(len(g)),
        n_violacoes=int(len(violacoes)),
        tolerancia=float(tolerancia),
        detalhes=violacoes,
    )
    logger.info(
        "verificar_soma_unitaria: %d grupos, soma in [%.4f, %.4f] "
        "(media %.4f), tol=%.3f, violacoes=%d",
        res.n_grupos, res.soma_min, res.soma_max, res.soma_media,
        res.tolerancia, res.n_violacoes,
    )
    return res


def cobertura_agregada(
    df_agg: pd.DataFrame,
    *,
    y_col: str = "y_real",
    lower_col: str = "share_lower",
    upper_col: str = "share_upper",
) -> float:
    """Fração de linhas onde y_col em [lower_col, upper_col]."""
    _validar_cols(df_agg, [y_col, lower_col, upper_col], "cobertura_agregada")
    sub = df_agg.dropna(subset=[y_col, lower_col, upper_col])
    if sub.empty:
        return float("nan")
    in_band = (sub[y_col] >= sub[lower_col]) & (sub[y_col] <= sub[upper_col])
    return float(in_band.mean())


__all__ = [
    "agregar_municipal_para_uf",
    "agregar_uf_para_nacional",
    "verificar_soma_unitaria",
    "cobertura_agregada",
    "SomaUnitariaResult",
]
