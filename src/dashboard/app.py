"""Streamlit Dashboard — 2026 Presidential Electoral Model."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import MODE, PATHS  # noqa: E402

# ── Paleta de partidos ────────────────────────────────────────────────────────
CORES: dict[str, str] = {
    "PT":     "#c0392b",   # vermelho muted
    "PL":     "#2e86c1",   # azul aço
    "MISSÃO": "#c8a951",   # dourado
    "NOVO":   "#ca6f1e",   # âmbar
    "PSD":    "#1e8449",   # verde floresta
    "Outros": "#5d7a96",   # azul-cinza
    # partidos legados (não aparecem nos outputs, mas mantidos pra fallback)
    "UNIÃO":  "#7d6608",
    "PTB":    "#566573",
    "UP":     "#922b21",
    "PSTU":   "#7b241c",
    "PCB":    "#641e16",
    "DC":     "#2e4053",
}

_BG      = "#0d1b2a"   # fundo principal
_BG2     = "#122236"   # fundo secundário (cards, sidebar)
_GOLD    = "#c8a951"   # dourado BTG
_TEXT    = "#dce6f0"   # texto primário
_MUTED   = "#7a9ab5"   # texto secundário
_BORDER  = "#1e3a5f"   # bordas e grids

_FINANCE_CSS = """
<style>
/* ── Fundo ────────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"], .main {
    background-color: #0d1b2a !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #0a1520 !important;
    border-right: 1px solid #1e3a5f !important;
}
[data-testid="stSidebar"] * { color: #b8cfe0 !important; }

/* ── Títulos ──────────────────────────────────────────────────── */
h1 {
    color: #dce6f0 !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #c8a951;
    padding-bottom: 10px;
    margin-bottom: 6px;
}
h2, h3 { color: #c8d8e8 !important; font-weight: 500 !important; }

/* ── Métricas ─────────────────────────────────────────────────── */
[data-testid="stMetricValue"] {
    color: #c8a951 !important;
    font-weight: 700 !important;
    font-size: 1.7rem !important;
}
[data-testid="stMetricLabel"]  { color: #7a9ab5 !important; font-size: 0.8rem !important; }
[data-testid="stMetricDelta"]  { color: #7a9ab5 !important; }

/* ── Abas ─────────────────────────────────────────────────────── */
[data-testid="stTabs"] button {
    color: #7a9ab5 !important;
    font-weight: 500;
    border-radius: 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #c8a951 !important;
    border-bottom: 2px solid #c8a951 !important;
    font-weight: 600 !important;
}

/* ── Selectbox ────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div {
    background-color: #122236 !important;
    border: 1px solid #1e3a5f !important;
    color: #dce6f0 !important;
}

/* ── Expanders ────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #1e3a5f !important;
    border-radius: 4px !important;
    background-color: #0f1e30 !important;
}
[data-testid="stExpander"] summary { color: #7a9ab5 !important; }

/* ── Info / Warning ───────────────────────────────────────────── */
[data-testid="stInfo"] {
    background-color: #0d2035 !important;
    border-left: 3px solid #c8a951 !important;
    color: #b8cfe0 !important;
}
[data-testid="stWarning"] {
    background-color: #1a1200 !important;
    border-left: 3px solid #ca6f1e !important;
}

/* ── Divisor ──────────────────────────────────────────────────── */
hr { border-color: #1e3a5f !important; }

/* ── Caption ──────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] p { color: #7a9ab5 !important; }

/* ── Texto geral ──────────────────────────────────────────────── */
p, li, span, label { color: #b8cfe0; }

/* ── Scrollbar ────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0d1b2a; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* ── Mobile responsivo ────────────────────────────────────────── */
@media (max-width: 768px) {
    [data-testid="column"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }
    [data-testid="stPlotlyChart"] > div {
        height: auto !important;
        min-height: 300px;
    }
    h1 { font-size: 1.4rem !important; }
    h2, h3 { font-size: 1.1rem !important; }
    [data-testid="stMetricValue"] { font-size: 1.2rem !important; }
}
</style>
"""

_TAB_JS = """
<script>
(function() {
    const KEY = 'el2026_tab';
    function getTabs() {
        return Array.from(window.parent.document.querySelectorAll(
            '[data-testid="stTabs"] button[role="tab"]'
        ));
    }
    function restore() {
        const idx = parseInt(sessionStorage.getItem(KEY) || '0');
        const tabs = getTabs();
        if (tabs.length > idx && tabs[idx].getAttribute('aria-selected') !== 'true') {
            tabs[idx].click();
        }
    }
    function bind() {
        getTabs().forEach((t, i) => {
            if (!t._el2026_bound) {
                t.addEventListener('click', () => sessionStorage.setItem(KEY, i));
                t._el2026_bound = true;
            }
        });
    }
    const poll = setInterval(() => {
        if (getTabs().length > 0) { clearInterval(poll); restore(); bind(); }
    }, 80);
})();
</script>
"""


CANDIDATOS: dict[str, str] = {
    "PT":     "Lula",
    "PL":     "Flávio Bolsonaro",
    "MISSÃO": "Renan Santos",
    "NOVO":   "Romeu Zema",
    "PSD":    "Caiado",
}


def label_partido(sigla: str) -> str:
    nome = CANDIDATOS.get(sigla)
    return f"{sigla} ({nome})" if nome else sigla


NOMES_UF: dict[str, str] = {
    "AC": "Acre", "AL": "Alagoas", "AM": "Amazonas", "AP": "Amapá",
    "BA": "Bahia", "CE": "Ceará", "DF": "Distrito Federal", "ES": "Espírito Santo",
    "GO": "Goiás", "MA": "Maranhão", "MG": "Minas Gerais", "MS": "Mato Grosso do Sul",
    "MT": "Mato Grosso", "PA": "Pará", "PB": "Paraíba", "PE": "Pernambuco",
    "PI": "Piauí", "PR": "Paraná", "RJ": "Rio de Janeiro", "RN": "Rio Grande do Norte",
    "RO": "Rondônia", "RR": "Roraima", "RS": "Rio Grande do Sul", "SC": "Santa Catarina",
    "SE": "Sergipe", "SP": "São Paulo", "TO": "Tocantins",
}


def label_uf(sigla: str) -> str:
    return f"{sigla} ({NOMES_UF.get(sigla, sigla)})"


def sigla_from_label(lbl: str) -> str:
    return lbl.split(" ")[0]


def cor(sigla: str) -> str:
    return CORES.get(sigla, "#9CA3AF")


def hex_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ── Carregamento com cache ────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading 1st round national forecast…")
def load_nac_1t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_nacional_2026.parquet")


@st.cache_data(show_spinner="Loading 2nd round national forecast…")
def load_nac_2t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_2t_nacional_2026.parquet")


@st.cache_data(show_spinner="Loading 1st round forecast by state…")
def load_uf_1t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_uf_2026.parquet")


@st.cache_data(show_spinner="Loading 2nd round forecast by state…")
def load_uf_2t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_2t_2026.parquet")


@st.cache_data(show_spinner="Loading Brazil map…", ttl=86400)
def load_geodata():
    import geobr
    states = geobr.read_state(year=2020)
    states = states.rename(columns={"abbrev_state": "sigla_uf"})
    states["geometry"] = states["geometry"].simplify(tolerance=0.05)
    return states


# ── Helpers de gráfico ────────────────────────────────────────────────────────
def bar_chart_1t(df: pd.DataFrame, titulo: str) -> go.Figure:
    df = df.sort_values("share_pred", ascending=True).copy()
    df["_label"] = df["sigla_partido"].map(label_partido)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["_label"],
        x=df["share_pred"],
        orientation="h",
        marker_color=[cor(p) for p in df["sigla_partido"]],
        error_x=dict(
            type="data",
            symmetric=False,
            array=(df["share_upper"] - df["share_pred"]).tolist(),
            arrayminus=(df["share_pred"] - df["share_lower"]).tolist(),
            color="#374151",
            thickness=1.5,
        ),
        text=[f"{v:.1%}" for v in df["share_pred"]],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.update_layout(
        title=dict(text=titulo, font=dict(color=_TEXT, size=14)),
        xaxis=dict(tickformat=".0%", range=[0, df["share_upper"].max() * 1.15],
                   color=_MUTED, gridcolor=_BORDER, tickcolor=_MUTED),
        yaxis=dict(tickfont=dict(size=13, color=_TEXT), showgrid=False),
        margin=dict(l=60, r=40, t=48, b=32),
        height=max(300, len(df) * 44),
        plot_bgcolor=_BG2,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
    )
    return fig


def gauge_2t(partido_a: str, share_a: float, lo_a: float, hi_a: float,
             partido_b: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(share_a * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"<b>{label_partido(partido_a)}</b> in the 2nd round", "font": {"size": 18}},
        delta={"reference": 50, "valueformat": ".1f",
               "increasing": {"color": cor(partido_a)},
               "decreasing": {"color": cor(partido_b)}},
        gauge={
            "axis": {"range": [30, 70], "tickformat": ".0f"},
            "bar": {"color": cor(partido_a), "thickness": 0.35},
            "bgcolor": "white",
            "borderwidth": 1,
            "steps": [
                {"range": [30, 50], "color": hex_rgba(cor(partido_b), 0.18)},
                {"range": [50, 70], "color": hex_rgba(cor(partido_a), 0.18)},
            ],
            "threshold": {"line": {"color": "#374151", "width": 3},
                          "thickness": 0.85, "value": 50},
        },
    ))
    ic_txt = f"90% CI: [{lo_a:.1%} – {hi_a:.1%}]"
    fig.add_annotation(text=ic_txt, x=0.5, y=-0.08, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color=_MUTED))
    fig.update_layout(height=280, margin=dict(t=40, b=60, l=20, r=20),
                      paper_bgcolor=_BG, font=dict(color=_TEXT))
    return fig


def mapa_2t(df_uf: pd.DataFrame, geo) -> go.Figure:
    import json

    df = df_uf.copy()
    df["spread"] = df["share_pred_A"] - df["share_pred_B"]  # PT − PL

    merged = geo.merge(df, on="sigla_uf", how="left")
    geojson = json.loads(merged.to_json())

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=merged["sigla_uf"],
        z=merged["spread"],
        featureidkey="properties.sigla_uf",
        colorscale=[[0, "#2e86c1"], [0.5, "#1e3a5f"], [1, "#c0392b"]],
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        colorbar=dict(
            title=f"← {label_partido('PL')} | {label_partido('PT')} →",
            showticklabels=False,
            len=0.6,
            tickfont=dict(color=_TEXT),
            titlefont=dict(color=_MUTED),
        ),
        customdata=merged[["share_pred_A", "share_pred_B"]].values,
        hovertemplate=(
            "<b>%{location}</b><br>"
            f"{label_partido('PT')}: %{{customdata[0]:.1%}}<br>"
            f"{label_partido('PL')}: %{{customdata[1]:.1%}}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor=_BG)
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=480,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
    )
    return fig


def mapa_1t(df_uf: pd.DataFrame, geo) -> go.Figure:
    import json

    pt = df_uf[df_uf["sigla_partido"] == "PT"].set_index("sigla_uf")["share_pred"]
    pl = df_uf[df_uf["sigla_partido"] == "PL"].set_index("sigla_uf")["share_pred"]
    spread = (pt - pl).rename("spread").reset_index()

    merged = geo.merge(spread, on="sigla_uf", how="left")
    merged = merged.merge(pt.rename("pt").reset_index(), on="sigla_uf", how="left")
    merged = merged.merge(pl.rename("pl").reset_index(), on="sigla_uf", how="left")
    geojson = json.loads(merged.to_json())

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=merged["sigla_uf"],
        z=merged["spread"],
        featureidkey="properties.sigla_uf",
        colorscale=[[0, "#2e86c1"], [0.5, "#1e3a5f"], [1, "#c0392b"]],
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        colorbar=dict(
            title=f"← {label_partido('PL')} | {label_partido('PT')} →",
            showticklabels=False,
            len=0.6,
            tickfont=dict(color=_TEXT),
            titlefont=dict(color=_MUTED),
        ),
        customdata=merged[["pt", "pl"]].values,
        hovertemplate=(
            "<b>%{location}</b><br>"
            f"{label_partido('PT')}: %{{customdata[0]:.1%}}<br>"
            f"{label_partido('PL')}: %{{customdata[1]:.1%}}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor=_BG)
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=420,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
    )
    return fig


# ── Layout principal ──────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="2026 Electoral Model",
        page_icon="🗳️",
        layout="wide",
    )
    st.markdown(_FINANCE_CSS, unsafe_allow_html=True)
    import streamlit.components.v1 as _components
    _components.html(_TAB_JS, height=0, scrolling=False)

    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    # ── Carrega dados ────────────────────────────────────────────────────────
    nac_1t = load_nac_1t()
    nac_2t = load_nac_2t()
    uf_1t  = load_uf_1t()
    uf_2t  = load_uf_2t()

    # ── Header com headline numbers ──────────────────────────────────────────
    st.title("2026 Presidential Elections | Model Forecast")
    st.caption("Based on TSE historical data + May/2026 polls.")

    row = st.columns(4)
    nac_sorted = nac_1t.sort_values("share_pred", ascending=False).reset_index(drop=True)
    lider     = nac_sorted.iloc[0]
    segundo   = nac_sorted.iloc[1]
    pt_row    = nac_2t[nac_2t["sigla_partido"] == "PT"].iloc[0]
    pl_row    = nac_2t[nac_2t["sigla_partido"] == "PL"].iloc[0]
    margem_2t = abs(pt_row["share_pred"] - pl_row["share_pred"])

    with row[0]:
        st.metric(f"🥇 R1 — {label_partido(lider['sigla_partido'])}",
                  f"{lider['share_pred']:.1%}",
                  f"CI [{lider['share_lower']:.1%} – {lider['share_upper']:.1%}]")
    with row[1]:
        st.metric(f"🥈 R1 — {label_partido(segundo['sigla_partido'])}",
                  f"{segundo['share_pred']:.1%}",
                  f"CI [{segundo['share_lower']:.1%} – {segundo['share_upper']:.1%}]")
    vencedor_2t = "PT" if pt_row["share_pred"] > pl_row["share_pred"] else "PL"
    with row[2]:
        st.metric(f"🏆 R2 — {label_partido(vencedor_2t)} projected",
                  f"{max(pt_row['share_pred'], pl_row['share_pred']):.1%}",
                  f"margin {margem_2t:.1%}")
    pt_ufs = int((uf_2t["vencedor"] == "PT").sum())
    pl_ufs = int((uf_2t["vencedor"] == "PL").sum())
    with row[3]:
        st.metric("🗺️ States R2 (PT × PL)", f"{pt_ufs} × {pl_ufs}",
                  f"PT electorate: "
                  f"{uf_2t.loc[uf_2t['vencedor']=='PT','eleitorado_uf'].sum()/uf_2t['eleitorado_uf'].sum():.1%}")

    st.divider()

    # Precarrega geodata fora das tabs — evita spinner dentro da aba causar reset
    try:
        _geo = load_geodata()
        _geo_ok = True
    except Exception as _geo_err:
        _geo = None
        _geo_ok = False

    # ── Tabs ─────────────────────────────────────────────────────────────────
    tab1t, tab2t, tab_uf1, tab_uf2, tab_analysis, tab_dados = st.tabs([
        "National Round 1",
        "National Round 2",
        "Round 1 by State",
        "Round 2 by State",
        "Analysis",
        "Tables",
    ])

    # ── Tab 1: 1º Turno Nacional ─────────────────────────────────────────────
    with tab1t:
        st.subheader("1st Round Forecast — National")
        st.plotly_chart(
            bar_chart_1t(nac_1t, "Vote share by party — Brazil (90% CI)"),
            use_container_width=True,
        )
        soma = nac_1t["share_pred"].sum()
        if abs(soma - 1.0) > 0.02:
            st.warning(f"Sum of shares = {soma:.3f} (outside ±2%). Interpret as relative.")
        else:
            st.caption(f"Sum of shares: {soma:.4f} ✓")

    # ── Tab 2: 2º Turno Nacional ─────────────────────────────────────────────
    with tab2t:
        st.subheader(f"2nd Round Forecast — National  ({label_partido('PT')} × {label_partido('PL')})")
        col_g, col_info = st.columns([2, 1])
        with col_g:
            pt_share = float(pt_row["share_pred"])
            st.plotly_chart(
                gauge_2t(
                    "PT", pt_share, float(pt_row["share_lower"]), float(pt_row["share_upper"]),
                    "PL",
                ),
                use_container_width=True,
            )
            pl_share = float(pl_row["share_pred"])
            st.plotly_chart(
                gauge_2t(
                    "PL", pl_share, float(pl_row["share_lower"]), float(pl_row["share_upper"]),
                    "PT",
                ),
                use_container_width=True,
            )
        with col_info:
            st.markdown("### Results")
            for _, r in nac_2t.sort_values("share_pred", ascending=False).iterrows():
                destaque = "**" if r["sigla_partido"] == vencedor_2t else ""
                st.markdown(
                    f"{destaque}{label_partido(r['sigla_partido'])}: {r['share_pred']:.2%}{destaque}  \n"
                    f"90% CI: [{r['share_lower']:.2%} – {r['share_upper']:.2%}]"
                )
                st.markdown("")
            st.info(
                "The CIs **overlap**: the margin of "
                f"{margem_2t:.1%} is within the model's uncertainty.",
                icon="ℹ️",
            )
            st.markdown("#### Projected winner by state")
            cnt = uf_2t["vencedor"].value_counts()
            for p, n in cnt.items():
                st.markdown(f"- **{label_partido(p)}**: {n} states")

    # ── Tab 3: 1º Turno por UF ───────────────────────────────────────────────
    with tab_uf1:
        st.subheader("1st Round Forecast by State")
        ufs_sorted = sorted(uf_1t["sigla_uf"].unique().tolist())
        uf_options = [label_uf(u) for u in ufs_sorted]
        default_idx = ufs_sorted.index("SP") if "SP" in ufs_sorted else 0

        col_map1, col_bar1 = st.columns([2, 3])
        with col_bar1:
            uf_sel_lbl = st.selectbox("Select a state", uf_options, index=default_idx,
                                      key="sel_uf1t")
            uf_sel = sigla_from_label(uf_sel_lbl)
            df_uf_sel = uf_1t[uf_1t["sigla_uf"] == uf_sel].copy()
            eleitorado = df_uf_sel["eleitorado_uf"].iloc[0]
            n_mun = int(df_uf_sel["n_municipios_uf"].iloc[0])
            st.caption(f"{uf_sel_lbl} — {n_mun} municipalities · {eleitorado:,.0f} voters")
            st.plotly_chart(
                bar_chart_1t(df_uf_sel, f"Round 1 share — {uf_sel_lbl} (90% CI)"),
                use_container_width=True,
            )
        with col_map1:
            if _geo_ok:
                st.plotly_chart(mapa_1t(uf_1t, _geo), use_container_width=True,
                                config={"responsive": True})
                st.caption("Red = PT leads · Blue = PL leads")
            else:
                st.warning(f"Map unavailable: {_geo_err}")

    # ── Tab 4: 2º Turno por UF ───────────────────────────────────────────────
    with tab_uf2:
        st.subheader(f"2nd Round Forecast by State  ({label_partido('PT')} × {label_partido('PL')})")
        col_map2, col_detail2 = st.columns([3, 2])

        with col_map2:
            if _geo_ok:
                st.plotly_chart(mapa_2t(uf_2t, _geo), use_container_width=True,
                                config={"responsive": True})
                st.caption("Red = PT leads · Blue = PL leads")
            else:
                st.warning(f"Map unavailable: {_geo_err}")

        with col_detail2:
            ufs_2t = sorted(uf_2t["sigla_uf"].unique().tolist())
            uf2_options = [label_uf(u) for u in ufs_2t]
            uf2_sel_lbl = st.selectbox("State details", uf2_options,
                                       index=ufs_2t.index("SP") if "SP" in ufs_2t else 0,
                                       key="sel_uf2t")
            uf2_sel = sigla_from_label(uf2_sel_lbl)
            row_2t = uf_2t[uf_2t["sigla_uf"] == uf2_sel].iloc[0]

            venc = row_2t["vencedor"]
            perd = "PL" if venc == "PT" else "PT"
            st.markdown(f"### {uf2_sel_lbl}")
            st.markdown(
                f"**Projected winner:** `{label_partido(venc)}`  \n"
                f"{label_partido(venc)}: **{row_2t['share_pred_A']:.1%}** "
                f"(90% CI: {row_2t['share_lower_A']:.1%} – {row_2t['share_upper_A']:.1%})  \n"
                f"{label_partido(perd)}: **{row_2t['share_pred_B']:.1%}** "
                f"(90% CI: {row_2t['share_lower_B']:.1%} – {row_2t['share_upper_B']:.1%})  \n"
                f"Electorate: {row_2t['eleitorado_uf']:,.0f}"
            )
            st.divider()

            # Tabela completa
            display = uf_2t[["sigla_uf", "vencedor", "share_pred_A",
                              "share_pred_B", "eleitorado_uf"]].copy()
            display["Estado"] = display["sigla_uf"].map(
                lambda s: label_uf(s))
            display = display[["Estado", "vencedor", "share_pred_A",
                                "share_pred_B", "eleitorado_uf"]]
            display.columns = ["State", "Winner", "PT %", "PL %", "Electorate"]
            display["Winner"] = display["Winner"].map(label_partido)
            display["PT %"] = display["PT %"].map("{:.1%}".format)
            display["PL %"] = display["PL %"].map("{:.1%}".format)
            display["Electorate"] = display["Electorate"].map("{:,.0f}".format)
            display = display.sort_values("State")

            def _highlight(row):
                c = hex_rgba(cor(row["Winner"]), 0.15)
                return [f"background-color: {c}"] * len(row)

            st.dataframe(
                display.style.apply(_highlight, axis=1),
                use_container_width=True,
                height=420,
                hide_index=True,
            )

    # ── Tab 5: Analysis ──────────────────────────────────────────────────────
    with tab_analysis:
        st.subheader("Forecast Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 2nd Round Margin by State (PT − PL)")
            st.markdown(
                "<small style='color:#7a9ab5'>Projected vote margin per state in the 2nd round. "
                "Positive values mean PT leads; negative mean PL leads. "
                "The further the bar extends from center, the more decisive the projected win.</small>",
                unsafe_allow_html=True,
            )
            df_m = uf_2t.copy()
            df_m["margin"] = df_m["share_pred_A"] - df_m["share_pred_B"]
            df_m = df_m.sort_values("margin")
            fig_m = go.Figure(go.Bar(
                y=df_m["sigla_uf"],
                x=df_m["margin"],
                orientation="h",
                marker_color=[cor("PT") if m > 0 else cor("PL") for m in df_m["margin"]],
                text=[f"{m:+.1%}" for m in df_m["margin"]],
                textposition="outside",
                cliponaxis=False,
                customdata=df_m[["share_pred_A", "share_pred_B", "eleitorado_uf"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{label_partido('PT')}: %{{customdata[0]:.1%}}<br>"
                    f"{label_partido('PL')}: %{{customdata[1]:.1%}}<br>"
                    "Electorate: %{customdata[2]:,.0f}<extra></extra>"
                ),
            ))
            fig_m.add_vline(x=0, line_color=_MUTED, line_width=1)
            fig_m.update_layout(
                xaxis=dict(tickformat="+.0%", color=_MUTED, gridcolor=_BORDER,
                           range=[-0.38, 0.38]),
                yaxis=dict(tickfont=dict(size=11, color=_TEXT), showgrid=False),
                margin=dict(l=40, r=70, t=10, b=20),
                height=max(500, len(df_m) * 22),
                plot_bgcolor=_BG2, paper_bgcolor=_BG, font=dict(color=_TEXT),
            )
            st.plotly_chart(fig_m, use_container_width=True)
            st.caption("Positive = PT leads; negative = PL (Flávio) leads.")

        with col_b:
            st.markdown("#### Forecast Uncertainty by State (90% CI width)")
            st.markdown(
                "<small style='color:#7a9ab5'>Width of the 90% confidence interval for PT's 2nd round share per state. "
                "Wider bars signal higher model uncertainty — driven by sparse local data or high sensitivity to vote transfer assumptions. "
                "Color indicates the projected winner.</small>",
                unsafe_allow_html=True,
            )
            df_ci = uf_2t.copy()
            df_ci["ci_width"] = df_ci["share_upper_A"] - df_ci["share_lower_A"]
            df_ci = df_ci.sort_values("ci_width", ascending=True)
            fig_ci = go.Figure(go.Bar(
                y=df_ci["sigla_uf"],
                x=df_ci["ci_width"],
                orientation="h",
                marker_color=[cor(v) for v in df_ci["vencedor"]],
                text=[f"{w:.1%}" for w in df_ci["ci_width"]],
                textposition="outside",
                cliponaxis=False,
                hovertemplate="<b>%{y}</b><br>CI width: %{x:.1%}<extra></extra>",
            ))
            fig_ci.update_layout(
                xaxis=dict(tickformat=".0%", color=_MUTED, gridcolor=_BORDER),
                yaxis=dict(tickfont=dict(size=11, color=_TEXT), showgrid=False),
                margin=dict(l=40, r=70, t=10, b=20),
                height=max(500, len(df_ci) * 22),
                plot_bgcolor=_BG2, paper_bgcolor=_BG, font=dict(color=_TEXT),
            )
            st.plotly_chart(fig_ci, use_container_width=True)
            st.caption("Color = projected 2nd round winner. Wider bar = more uncertain prediction.")

        st.divider()

        st.markdown("#### Round 1: PT vs PL vote share by state")
        st.markdown(
            "<small style='color:#7a9ab5'>Each bubble is a state. "
            "The vertical axis shows PT's Round 1 share; the horizontal axis shows PL's. "
            "States above the diagonal favor PT; below favor PL. "
            "Bubble size is proportional to the state electorate — larger bubbles carry more weight in the national result.</small>",
            unsafe_allow_html=True,
        )
        pt_r1 = (uf_1t[uf_1t["sigla_partido"] == "PT"]
                 [["sigla_uf", "share_pred", "eleitorado_uf"]]
                 .rename(columns={"share_pred": "pt"}))
        pl_r1 = (uf_1t[uf_1t["sigla_partido"] == "PL"]
                 [["sigla_uf", "share_pred"]]
                 .rename(columns={"share_pred": "pl"}))
        sc = pt_r1.merge(pl_r1, on="sigla_uf")
        lim = max(sc["pt"].max(), sc["pl"].max()) * 1.12

        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=[0, lim], y=[0, lim], mode="lines",
            line=dict(color=_BORDER, dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
        ))
        fig_sc.add_trace(go.Scatter(
            x=sc["pl"], y=sc["pt"],
            mode="markers+text",
            text=sc["sigla_uf"],
            textposition="top center",
            textfont=dict(size=9, color=_MUTED),
            marker=dict(
                size=(sc["eleitorado_uf"] / sc["eleitorado_uf"].max() * 44 + 7).tolist(),
                color=[cor("PT") if pt > pl else cor("PL")
                       for pt, pl in zip(sc["pt"], sc["pl"])],
                opacity=0.85,
                line=dict(width=1, color=_BORDER),
            ),
            customdata=sc[["sigla_uf", "pt", "pl", "eleitorado_uf"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                f"{label_partido('PT')}: %{{customdata[1]:.1%}}<br>"
                f"{label_partido('PL')}: %{{customdata[2]:.1%}}<br>"
                "Electorate: %{customdata[3]:,.0f}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_sc.update_layout(
            xaxis=dict(title=f"{label_partido('PL')} share (R1)", tickformat=".0%",
                       color=_MUTED, gridcolor=_BORDER, range=[0, lim]),
            yaxis=dict(title=f"{label_partido('PT')} share (R1)", tickformat=".0%",
                       color=_MUTED, gridcolor=_BORDER, range=[0, lim]),
            margin=dict(l=60, r=20, t=10, b=60),
            height=500,
            plot_bgcolor=_BG2, paper_bgcolor=_BG, font=dict(color=_TEXT),
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Bubble size = electorate size. "
            "Above the diagonal = PT leads in that state; below = PL (Flávio) leads."
        )

        st.divider()

        # ── Swing States ─────────────────────────────────────────────────────
        st.markdown("#### Swing States — Most Competitive (margin < 10pp)")
        st.markdown(
            "<small style='color:#7a9ab5'>Only states where the projected 2nd round margin is under 10pp — "
            "these are the most likely to flip the final outcome. "
            "States inside the gold band (±5pp) are statistical toss-ups where model uncertainty spans both candidates.</small>",
            unsafe_allow_html=True,
        )
        df_swing = uf_2t.copy()
        df_swing["margin"] = df_swing["share_pred_A"] - df_swing["share_pred_B"]
        df_swing["abs_margin"] = df_swing["margin"].abs()
        df_swing = df_swing[df_swing["abs_margin"] < 0.10].sort_values("abs_margin")
        if df_swing.empty:
            st.info("No states within 10pp margin.")
        else:
            fig_sw = go.Figure(go.Bar(
                y=df_swing["sigla_uf"],
                x=df_swing["margin"],
                orientation="h",
                marker_color=[cor("PT") if m > 0 else cor("PL") for m in df_swing["margin"]],
                text=[f"{m:+.1%}" for m in df_swing["margin"]],
                textposition="outside",
                cliponaxis=False,
                customdata=df_swing[["share_pred_A", "share_pred_B", "eleitorado_uf"]].values,
                hovertemplate=(
                    "<b>%{y}</b><br>"
                    f"{label_partido('PT')}: %{{customdata[0]:.1%}}<br>"
                    f"{label_partido('PL')}: %{{customdata[1]:.1%}}<br>"
                    "Electorate: %{customdata[2]:,.0f}<extra></extra>"
                ),
            ))
            fig_sw.add_vline(x=0, line_color=_MUTED, line_width=1)
            fig_sw.add_vrect(x0=-0.05, x1=0.05, fillcolor=_GOLD, opacity=0.06,
                             line_width=0, annotation_text="toss-up zone",
                             annotation_position="top left",
                             annotation_font=dict(color=_GOLD, size=9))
            fig_sw.update_layout(
                xaxis=dict(tickformat="+.0%", color=_MUTED, gridcolor=_BORDER,
                           range=[-0.12, 0.12]),
                yaxis=dict(tickfont=dict(size=11, color=_TEXT), showgrid=False),
                margin=dict(l=40, r=70, t=20, b=20),
                height=max(260, len(df_swing) * 38),
                plot_bgcolor=_BG2, paper_bgcolor=_BG, font=dict(color=_TEXT),
            )
            st.plotly_chart(fig_sw, use_container_width=True)
            st.caption(f"{len(df_swing)} states within 10pp — highlighted band = within 5pp (toss-up).")

        st.divider()

        # ── Breakdown por região ──────────────────────────────────────────────
        REGIOES = {
            "Norte":        ["AC", "AP", "AM", "PA", "RO", "RR", "TO"],
            "Nordeste":     ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"],
            "Centro-Oeste": ["DF", "GO", "MS", "MT"],
            "Sudeste":      ["ES", "MG", "RJ", "SP"],
            "Sul":          ["PR", "RS", "SC"],
        }
        uf2_reg = uf_2t.copy()
        uf2_reg["regiao"] = uf2_reg["sigla_uf"].map(
            {uf: r for r, ufs in REGIOES.items() for uf in ufs}
        )
        reg_agg = (
            uf2_reg.groupby("regiao")
            .apply(lambda g: pd.Series({
                "pt_share": (g["share_pred_A"] * g["eleitorado_uf"]).sum() / g["eleitorado_uf"].sum(),
                "pl_share": (g["share_pred_B"] * g["eleitorado_uf"]).sum() / g["eleitorado_uf"].sum(),
                "eleitorado": g["eleitorado_uf"].sum(),
            }), include_groups=False)
            .reset_index()
        )
        reg_order = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
        reg_agg["regiao"] = pd.Categorical(reg_agg["regiao"], categories=reg_order, ordered=True)
        reg_agg = reg_agg.sort_values("regiao")

        col_reg, col_eleit = st.columns(2)

        with col_reg:
            st.markdown("#### 2nd Round Share by Region (electorate-weighted)")
            st.markdown(
                "<small style='color:#7a9ab5'>State-level 2nd round projections aggregated into Brazil's 5 geographic regions, "
                "weighted by electorate size. Reveals the structural regional base of each candidate — "
                "PT's dominance in the Northeast vs PL's strength in the South and Center-West.</small>",
                unsafe_allow_html=True,
            )
            fig_reg = go.Figure()
            fig_reg.add_trace(go.Bar(
                name=label_partido("PT"),
                x=reg_agg["regiao"],
                y=reg_agg["pt_share"],
                marker_color=cor("PT"),
                text=[f"{v:.1%}" for v in reg_agg["pt_share"]],
                textposition="outside",
            ))
            fig_reg.add_trace(go.Bar(
                name=label_partido("PL"),
                x=reg_agg["regiao"],
                y=reg_agg["pl_share"],
                marker_color=cor("PL"),
                text=[f"{v:.1%}" for v in reg_agg["pl_share"]],
                textposition="outside",
            ))
            fig_reg.add_hline(y=0.5, line_dash="dash", line_color=_MUTED, line_width=1)
            fig_reg.update_layout(
                barmode="group",
                xaxis=dict(color=_MUTED, gridcolor=_BORDER),
                yaxis=dict(tickformat=".0%", color=_MUTED, gridcolor=_BORDER,
                           range=[0, 0.80]),
                legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT)),
                margin=dict(l=40, r=20, t=10, b=40),
                height=380,
                plot_bgcolor=_BG2, paper_bgcolor=_BG, font=dict(color=_TEXT),
            )
            st.plotly_chart(fig_reg, use_container_width=True)
            st.caption("Shares are weighted by state electorate within each region.")

        with col_eleit:
            st.markdown("#### Electorate Distribution by 2nd Round Winner")
            st.markdown(
                "<small style='color:#7a9ab5'>Share of the total Brazilian electorate living in states projected to be won by each candidate. "
                "Winning more states is not enough — this chart shows the true electoral weight behind each candidate's map.</small>",
                unsafe_allow_html=True,
            )
            eleit_pt = uf_2t.loc[uf_2t["vencedor"] == "PT", "eleitorado_uf"].sum()
            eleit_pl = uf_2t.loc[uf_2t["vencedor"] == "PL", "eleitorado_uf"].sum()
            total = eleit_pt + eleit_pl
            fig_donut = go.Figure(go.Pie(
                labels=[label_partido("PT"), label_partido("PL")],
                values=[eleit_pt, eleit_pl],
                hole=0.55,
                marker_colors=[cor("PT"), cor("PL")],
                textinfo="label+percent",
                textfont=dict(size=12, color=_TEXT),
                hovertemplate="<b>%{label}</b><br>%{value:,.0f} voters<br>%{percent}<extra></extra>",
            ))
            fig_donut.add_annotation(
                text=f"{total/1e6:.1f}M<br>voters",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=13, color=_MUTED),
            )
            fig_donut.update_layout(
                showlegend=False,
                margin=dict(l=20, r=20, t=10, b=10),
                height=380,
                paper_bgcolor=_BG, font=dict(color=_TEXT),
            )
            st.plotly_chart(fig_donut, use_container_width=True)
            st.caption(
                f"{label_partido('PT')} leads in states with {eleit_pt/total:.1%} of the electorate · "
                f"{label_partido('PL')} leads in {eleit_pl/total:.1%}."
            )

        st.divider()

        # ── R1 → R2 shift ────────────────────────────────────────────────────
        st.markdown("#### Round 1 → Round 2 Vote Transfer by State")
        st.markdown(
            "<small style='color:#7a9ab5'>Difference between each candidate's projected 2nd round share and their 1st round share, per state. "
            "Positive bars mean the candidate gained votes from eliminated parties; negative means they lost relative share. "
            "PT benefits from left-leaning transfers (PDT, MDB, UP); PL from right-leaning ones (NOVO, UNIÃO, DC).</small>",
            unsafe_allow_html=True,
        )
        pt_r1_s = (uf_1t[uf_1t["sigla_partido"] == "PT"]
                   [["sigla_uf", "share_pred"]].rename(columns={"share_pred": "pt_r1"}))
        pl_r1_s = (uf_1t[uf_1t["sigla_partido"] == "PL"]
                   [["sigla_uf", "share_pred"]].rename(columns={"share_pred": "pl_r1"}))
        df_shift = (uf_2t[["sigla_uf", "share_pred_A", "share_pred_B", "vencedor"]]
                    .merge(pt_r1_s, on="sigla_uf").merge(pl_r1_s, on="sigla_uf"))
        df_shift["pt_gain"] = df_shift["share_pred_A"] - df_shift["pt_r1"]
        df_shift["pl_gain"] = df_shift["share_pred_B"] - df_shift["pl_r1"]
        df_shift = df_shift.sort_values("pt_gain")

        fig_shift = go.Figure()
        fig_shift.add_trace(go.Bar(
            name=f"{label_partido('PT')} R1→R2 gain",
            x=df_shift["sigla_uf"],
            y=df_shift["pt_gain"],
            marker_color=cor("PT"),
            text=[f"{v:+.1%}" for v in df_shift["pt_gain"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_shift.add_trace(go.Bar(
            name=f"{label_partido('PL')} R1→R2 gain",
            x=df_shift["sigla_uf"],
            y=df_shift["pl_gain"],
            marker_color=cor("PL"),
            text=[f"{v:+.1%}" for v in df_shift["pl_gain"]],
            textposition="outside",
            cliponaxis=False,
        ))
        fig_shift.add_hline(y=0, line_color=_MUTED, line_width=1)
        fig_shift.update_layout(
            barmode="group",
            xaxis=dict(color=_MUTED, gridcolor=_BORDER, tickangle=-45),
            yaxis=dict(tickformat="+.0%", color=_MUTED, gridcolor=_BORDER),
            legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=_TEXT),
                        orientation="h", y=1.05),
            margin=dict(l=40, r=20, t=40, b=60),
            height=420,
            plot_bgcolor=_BG2, paper_bgcolor=_BG, font=dict(color=_TEXT),
        )
        st.plotly_chart(fig_shift, use_container_width=True)
        st.caption(
            "Positive = candidate gained share from vote transfer; negative = lost share. "
            "PT typically gains from left-leaning party transfers; PL from right-leaning."
        )

    # ── Tab 6: Tabelas raw ───────────────────────────────────────────────────
    with tab_dados:
        st.subheader("Raw Data")
        with st.expander("National Round 1", expanded=False):
            st.dataframe(
                nac_1t.sort_values("share_pred", ascending=False)
                       .style.format({"share_pred": "{:.4f}", "share_lower": "{:.4f}",
                                      "share_upper": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("National Round 2", expanded=False):
            st.dataframe(nac_2t, use_container_width=True, hide_index=True)
        with st.expander("Round 1 by State (complete)", expanded=False):
            st.dataframe(
                uf_1t.sort_values(["sigla_uf", "share_pred"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("Round 2 by State", expanded=False):
            st.dataframe(uf_2t.sort_values("sigla_uf"), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
