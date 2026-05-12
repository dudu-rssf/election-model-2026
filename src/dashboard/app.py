"""Dashboard Streamlit — Modelo Eleitoral Presidencial 2026."""
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
@st.cache_data(show_spinner="Carregando previsão 1º turno nacional…")
def load_nac_1t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_nacional_2026.parquet")


@st.cache_data(show_spinner="Carregando previsão 2º turno nacional…")
def load_nac_2t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_2t_nacional_2026.parquet")


@st.cache_data(show_spinner="Carregando previsão 1º turno por UF…")
def load_uf_1t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_uf_2026.parquet")


@st.cache_data(show_spinner="Carregando previsão 2º turno por UF…")
def load_uf_2t() -> pd.DataFrame:
    return pd.read_parquet(PATHS["data_processed"] / "previsao_2t_2026.parquet")


@st.cache_data(show_spinner="Carregando mapa do Brasil…", ttl=86400)
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
        title={"text": f"<b>{label_partido(partido_a)}</b> no 2º turno", "font": {"size": 18}},
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
    # IC como anotação
    ic_txt = f"IC 90%: [{lo_a:.1%} – {hi_a:.1%}]"
    fig.add_annotation(text=ic_txt, x=0.5, y=-0.08, xref="paper", yref="paper",
                       showarrow=False, font=dict(size=12, color=_MUTED))
    fig.update_layout(height=280, margin=dict(t=40, b=60, l=20, r=20),
                      paper_bgcolor=_BG, font=dict(color=_TEXT))
    return fig


def mapa_2t(df_uf: pd.DataFrame, geo) -> go.Figure:
    import json

    merged = geo.merge(df_uf, on="sigla_uf", how="left")
    geojson = json.loads(merged.to_json())

    # Encode vencedor como número pra colorscale
    partido_a = df_uf.loc[df_uf["share_pred_A"].idxmax(), "sigla_uf"]  # noqa: F841
    venc_list = df_uf["vencedor"].unique().tolist()
    if "PT" in venc_list and "PL" in venc_list:
        cor_a, cor_b = "#CC0000", "#1A56DB"
        label_a, label_b = "PT", "PL"
    else:
        cor_a, cor_b = cor(venc_list[0]), cor(venc_list[-1])
        label_a, label_b = venc_list[0], venc_list[-1]

    fig = go.Figure(go.Choropleth(
        geojson=geojson,
        locations=df_uf["sigla_uf"],
        z=df_uf["share_pred_A"],
        featureidkey="properties.sigla_uf",
        colorscale=[[0, cor_b], [0.5, "#1e3a5f"], [1, cor_a]],
        zmid=0.5,
        zmin=0.3,
        zmax=0.7,
        colorbar=dict(
            title=f"← {label_partido(label_b)} | {label_partido(label_a)} →",
            tickformat=".0%",
            len=0.6,
        ),
        hovertemplate=(
            "<b>%{location}</b><br>"
            f"{label_partido(label_a)}: %{{z:.1%}}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor=_BG,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=480,
        paper_bgcolor=_BG,
        font=dict(color=_TEXT),
    )
    return fig


def mapa_1t(df_uf: pd.DataFrame, geo) -> go.Figure:
    """Mapa 1º turno: spread PT − PL por UF (divergente vermelho/azul)."""
    import json

    # Pivot: share por partido por UF
    pivot = df_uf.pivot_table(index="sigla_uf", columns="sigla_partido",
                              values="share_pred", aggfunc="first").fillna(0)
    pt_share = pivot.get("PT", pd.Series(0, index=pivot.index))
    pl_share = pivot.get("PL", pd.Series(0, index=pivot.index))
    spread = (pt_share - pl_share).reset_index()
    spread.columns = ["sigla_uf", "spread"]

    lider = df_uf.loc[df_uf.groupby("sigla_uf")["share_pred"].idxmax(),
                      ["sigla_uf", "sigla_partido", "share_pred"]].set_index("sigla_uf")

    merged = geo.merge(spread, on="sigla_uf", how="left")
    merged = merged.merge(
        lider.rename(columns={"sigla_partido": "lider", "share_pred": "share_lider"}),
        on="sigla_uf", how="left",
    )
    geojson = json.loads(merged.to_json())

    hover_text = [
        f"<b>{row['sigla_uf']}</b> — {NOMES_UF.get(row['sigla_uf'], '')}<br>"
        f"Líder 1t: {label_partido(row['lider'])} ({row['share_lider']:.1%})"
        for _, row in merged.iterrows()
    ]

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
            tickformat=".0%",
            len=0.6,
            tickfont=dict(color=_TEXT),
            titlefont=dict(color=_MUTED),
        ),
        text=hover_text,
        hovertemplate="%{text}<extra></extra>",
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
        page_title="Modelo Eleitoral 2026",
        page_icon="🗳️",
        layout="wide",
    )
    st.markdown(_FINANCE_CSS, unsafe_allow_html=True)
    import streamlit.components.v1 as _components
    _components.html(_TAB_JS, height=0, scrolling=False)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🗳️ Modelo 2026")
        st.caption(f"Modo: **{MODE.upper()}** · Python {sys.version.split()[0]}")
        st.markdown(
            "Previsão bottom-up: município → UF → nacional.  \n"
            "**Modelo:** LightGBM + calibração isotônica + Mondrian conformal.  \n"
            "**Snapshot de pesquisa:** mai/2026."
        )
        st.divider()
        st.markdown(
            "**Caveats principais**\n"
            "- Pesquisa de mai/2026 (~5 meses antes). Re-rodar pré-1t (set/2026).\n"
            "- PSD (Caiado ~5%) e MISSÃO (~2.7%) fora do universo do modelo.\n"
            "- PT pode estar subestimado em ~5pp (padrão em 2022).\n"
            "- Cobertura IC nacional = 63.6%; UF = 77.8%.\n"
            "- 2º turno usa matriz de transferência qualitativa, não treinada."
        )
        st.divider()
        st.caption("Intervalos = IC 90% (Mondrian conformal por bin)")

    # ── Carrega dados ────────────────────────────────────────────────────────
    nac_1t = load_nac_1t()
    nac_2t = load_nac_2t()
    uf_1t  = load_uf_1t()
    uf_2t  = load_uf_2t()

    # ── Header com headline numbers ──────────────────────────────────────────
    st.title("Eleições Presidenciais 2026 — Previsão do Modelo")
    st.caption("Baseado em dados históricos TSE + pesquisas mai/2026. Veja caveats na barra lateral.")

    row = st.columns(4)
    nac_sorted = nac_1t.sort_values("share_pred", ascending=False).reset_index(drop=True)
    lider     = nac_sorted.iloc[0]
    segundo   = nac_sorted.iloc[1]
    pt_row    = nac_2t[nac_2t["sigla_partido"] == "PT"].iloc[0]
    pl_row    = nac_2t[nac_2t["sigla_partido"] == "PL"].iloc[0]
    margem_2t = abs(pt_row["share_pred"] - pl_row["share_pred"])

    with row[0]:
        st.metric(f"🥇 1t — {label_partido(lider['sigla_partido'])}",
                  f"{lider['share_pred']:.1%}",
                  f"IC [{lider['share_lower']:.1%} – {lider['share_upper']:.1%}]")
    with row[1]:
        st.metric(f"🥈 1t — {label_partido(segundo['sigla_partido'])}",
                  f"{segundo['share_pred']:.1%}",
                  f"IC [{segundo['share_lower']:.1%} – {segundo['share_upper']:.1%}]")
    vencedor_2t = "PT" if pt_row["share_pred"] > pl_row["share_pred"] else "PL"
    with row[2]:
        st.metric(f"🏆 2t — {label_partido(vencedor_2t)} previsto",
                  f"{max(pt_row['share_pred'], pl_row['share_pred']):.1%}",
                  f"margem {margem_2t:.1%}")
    pt_ufs = int((uf_2t["vencedor"] == "PT").sum())
    pl_ufs = int((uf_2t["vencedor"] == "PL").sum())
    with row[3]:
        st.metric("🗺️ UFs 2t (PT × PL)", f"{pt_ufs} × {pl_ufs}",
                  f"eleitorado PT: "
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
    tab1t, tab2t, tab_uf1, tab_uf2, tab_dados = st.tabs([
        "1º Turno Nacional",
        "2º Turno Nacional",
        "1º Turno por UF",
        "2º Turno por UF",
        "Tabelas",
    ])

    # ── Tab 1: 1º Turno Nacional ─────────────────────────────────────────────
    with tab1t:
        st.subheader("Previsão 1º Turno — Nacional")
        st.plotly_chart(
            bar_chart_1t(nac_1t, "Share de votos por partido — Brasil (IC 90%)"),
            use_container_width=True,
        )
        soma = nac_1t["share_pred"].sum()
        if abs(soma - 1.0) > 0.02:
            st.warning(f"Soma dos shares = {soma:.3f} (fora de ±2%). Interprete como relativo.")
        else:
            st.caption(f"Soma dos shares: {soma:.4f} ✓")

    # ── Tab 2: 2º Turno Nacional ─────────────────────────────────────────────
    with tab2t:
        st.subheader(f"Previsão 2º Turno — Nacional  ({label_partido('PT')} × {label_partido('PL')})")
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
            st.markdown("### Resultados")
            for _, r in nac_2t.sort_values("share_pred", ascending=False).iterrows():
                destaque = "**" if r["sigla_partido"] == vencedor_2t else ""
                st.markdown(
                    f"{destaque}{label_partido(r['sigla_partido'])}: {r['share_pred']:.2%}{destaque}  \n"
                    f"IC 90%: [{r['share_lower']:.2%} – {r['share_upper']:.2%}]"
                )
                st.markdown("")
            st.info(
                "Os ICs se **sobrepõem**: a margem de "
                f"{margem_2t:.1%} está dentro da incerteza do modelo.",
                icon="ℹ️",
            )
            st.markdown("#### Vencedor previsto por UF")
            cnt = uf_2t["vencedor"].value_counts()
            for p, n in cnt.items():
                st.markdown(f"- **{label_partido(p)}**: {n} UFs")

    # ── Tab 3: 1º Turno por UF ───────────────────────────────────────────────
    with tab_uf1:
        st.subheader("Previsão 1º Turno por UF")
        ufs_sorted = sorted(uf_1t["sigla_uf"].unique().tolist())
        uf_options = [label_uf(u) for u in ufs_sorted]
        default_idx = ufs_sorted.index("SP") if "SP" in ufs_sorted else 0

        col_map1, col_bar1 = st.columns([2, 3])
        with col_bar1:
            uf_sel_lbl = st.selectbox("Selecione o estado", uf_options, index=default_idx,
                                      key="sel_uf1t")
            uf_sel = sigla_from_label(uf_sel_lbl)
            df_uf_sel = uf_1t[uf_1t["sigla_uf"] == uf_sel].copy()
            eleitorado = df_uf_sel["eleitorado_uf"].iloc[0]
            n_mun = int(df_uf_sel["n_municipios_uf"].iloc[0])
            st.caption(f"{uf_sel_lbl} — {n_mun} municípios · {eleitorado:,.0f} eleitores")
            st.plotly_chart(
                bar_chart_1t(df_uf_sel, f"Share 1t — {uf_sel_lbl} (IC 90%)"),
                use_container_width=True,
            )
        with col_map1:
            if _geo_ok:
                st.plotly_chart(mapa_1t(uf_1t, _geo), use_container_width=True,
                                config={"responsive": True})
                st.caption("Vermelho = PT lidera · Azul = PL lidera")
            else:
                st.warning(f"Mapa indisponível: {_geo_err}")

    # ── Tab 4: 2º Turno por UF ───────────────────────────────────────────────
    with tab_uf2:
        st.subheader(f"Previsão 2º Turno por UF  ({label_partido('PT')} × {label_partido('PL')})")
        col_map2, col_detail2 = st.columns([3, 2])

        with col_map2:
            if _geo_ok:
                st.plotly_chart(mapa_2t(uf_2t, _geo), use_container_width=True,
                                config={"responsive": True})
                st.caption("Vermelho = PT lidera · Azul = PL lidera")
            else:
                st.warning(f"Mapa não disponível: {_geo_err}")

        with col_detail2:
            # Selectbox estado
            ufs_2t = sorted(uf_2t["sigla_uf"].unique().tolist())
            uf2_options = [label_uf(u) for u in ufs_2t]
            uf2_sel_lbl = st.selectbox("Detalhe por estado", uf2_options,
                                       index=ufs_2t.index("SP") if "SP" in ufs_2t else 0,
                                       key="sel_uf2t")
            uf2_sel = sigla_from_label(uf2_sel_lbl)
            row_2t = uf_2t[uf_2t["sigla_uf"] == uf2_sel].iloc[0]

            venc = row_2t["vencedor"]
            perd = "PL" if venc == "PT" else "PT"
            st.markdown(f"### {uf2_sel_lbl}")
            st.markdown(
                f"**Vencedor previsto:** `{label_partido(venc)}`  \n"
                f"{label_partido(venc)}: **{row_2t['share_pred_A']:.1%}** "
                f"(IC 90%: {row_2t['share_lower_A']:.1%} – {row_2t['share_upper_A']:.1%})  \n"
                f"{label_partido(perd)}: **{row_2t['share_pred_B']:.1%}** "
                f"(IC 90%: {row_2t['share_lower_B']:.1%} – {row_2t['share_upper_B']:.1%})  \n"
                f"Eleitorado: {row_2t['eleitorado_uf']:,.0f}"
            )
            st.divider()

            # Tabela completa
            display = uf_2t[["sigla_uf", "vencedor", "share_pred_A",
                              "share_pred_B", "eleitorado_uf"]].copy()
            display["Estado"] = display["sigla_uf"].map(
                lambda s: label_uf(s))
            display = display[["Estado", "vencedor", "share_pred_A",
                                "share_pred_B", "eleitorado_uf"]]
            display.columns = ["Estado", "Vencedor", "PT %", "PL %", "Eleitorado"]
            display["Vencedor"] = display["Vencedor"].map(label_partido)
            display["PT %"] = display["PT %"].map("{:.1%}".format)
            display["PL %"] = display["PL %"].map("{:.1%}".format)
            display["Eleitorado"] = display["Eleitorado"].map("{:,.0f}".format)
            display = display.sort_values("Estado")

            def _highlight(row):
                c = hex_rgba(cor(row["Vencedor"]), 0.15)
                return [f"background-color: {c}"] * len(row)

            st.dataframe(
                display.style.apply(_highlight, axis=1),
                use_container_width=True,
                height=420,
                hide_index=True,
            )

    # ── Tab 5: Tabelas raw ───────────────────────────────────────────────────
    with tab_dados:
        st.subheader("Dados brutos")
        with st.expander("1º Turno Nacional", expanded=False):
            st.dataframe(
                nac_1t.sort_values("share_pred", ascending=False)
                       .style.format({"share_pred": "{:.4f}", "share_lower": "{:.4f}",
                                      "share_upper": "{:.4f}"}),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("2º Turno Nacional", expanded=False):
            st.dataframe(nac_2t, use_container_width=True, hide_index=True)
        with st.expander("1º Turno por UF (completo)", expanded=False):
            st.dataframe(
                uf_1t.sort_values(["sigla_uf", "share_pred"], ascending=[True, False]),
                use_container_width=True,
                hide_index=True,
            )
        with st.expander("2º Turno por UF", expanded=False):
            st.dataframe(uf_2t.sort_values("sigla_uf"), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
