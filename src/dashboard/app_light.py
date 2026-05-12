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
    "PT":     "#CC0000",
    "PL":     "#1A56DB",
    "MISSÃO": "#E8650A",
    "PSD":    "#009933",
    "NOVO":   "#FF6600",
    "UNIÃO":  "#8B5CF6",
    "PTB":    "#6B7280",
    "UP":     "#DC2626",
    "PSTU":   "#9B1C1C",
    "PCB":    "#B91C1C",
    "DC":     "#374151",
}


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
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df["sigla_partido"],
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
        title=titulo,
        xaxis=dict(tickformat=".0%", range=[0, df["share_upper"].max() * 1.15]),
        yaxis=dict(tickfont=dict(size=13)),
        margin=dict(l=60, r=40, t=48, b=32),
        height=max(300, len(df) * 44),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    return fig


def gauge_2t(partido_a: str, share_a: float, lo_a: float, hi_a: float,
             partido_b: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(share_a * 100, 1),
        number={"suffix": "%", "font": {"size": 40}},
        title={"text": f"<b>{partido_a}</b> no 2º turno", "font": {"size": 18}},
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
                       showarrow=False, font=dict(size=12, color="#6B7280"))
    fig.update_layout(height=280, margin=dict(t=40, b=60, l=20, r=20),
                      paper_bgcolor="white")
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
        colorscale=[[0, cor_b], [0.5, "#F3F4F6"], [1, cor_a]],
        zmid=0.5,
        zmin=0.3,
        zmax=0.7,
        colorbar=dict(
            title=f"← {label_b} | {label_a} →",
            tickformat=".0%",
            len=0.6,
        ),
        hovertemplate=(
            "<b>%{location}</b><br>"
            f"{label_a}: %{{z:.1%}}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_geos(
        fitbounds="locations",
        visible=False,
        bgcolor="white",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=480,
        paper_bgcolor="white",
    )
    return fig


# ── Layout principal ──────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Modelo Eleitoral 2026",
        page_icon="🗳️",
        layout="wide",
    )

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
        st.metric(f"🥇 1t — {lider['sigla_partido']}",
                  f"{lider['share_pred']:.1%}",
                  f"IC [{lider['share_lower']:.1%} – {lider['share_upper']:.1%}]")
    with row[1]:
        st.metric(f"🥈 1t — {segundo['sigla_partido']}",
                  f"{segundo['share_pred']:.1%}",
                  f"IC [{segundo['share_lower']:.1%} – {segundo['share_upper']:.1%}]")
    vencedor_2t = "PT" if pt_row["share_pred"] > pl_row["share_pred"] else "PL"
    with row[2]:
        st.metric(f"🏆 2t — {vencedor_2t} previsto",
                  f"{max(pt_row['share_pred'], pl_row['share_pred']):.1%}",
                  f"margem {margem_2t:.1%}")
    pt_ufs = int((uf_2t["vencedor"] == "PT").sum())
    pl_ufs = int((uf_2t["vencedor"] == "PL").sum())
    with row[3]:
        st.metric("🗺️ UFs 2t (PT × PL)", f"{pt_ufs} × {pl_ufs}",
                  f"eleitorado PT: "
                  f"{uf_2t.loc[uf_2t['vencedor']=='PT','eleitorado_uf'].sum()/uf_2t['eleitorado_uf'].sum():.1%}")

    st.divider()

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
        st.subheader("Previsão 2º Turno — Nacional  (PT × PL)")
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
                    f"{destaque}{r['sigla_partido']}: {r['share_pred']:.2%}{destaque}  \n"
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
                st.markdown(f"- **{p}**: {n} UFs")

    # ── Tab 3: 1º Turno por UF ───────────────────────────────────────────────
    with tab_uf1:
        st.subheader("Previsão 1º Turno por UF")
        ufs = sorted(uf_1t["sigla_uf"].unique().tolist())
        uf_sel = st.selectbox("Selecione a UF", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
        df_uf_sel = uf_1t[uf_1t["sigla_uf"] == uf_sel].copy()
        eleitorado = df_uf_sel["eleitorado_uf"].iloc[0]
        n_mun = int(df_uf_sel["n_municipios_uf"].iloc[0])
        st.caption(f"{uf_sel} — {n_mun} municípios · eleitorado {eleitorado:,.0f}")
        st.plotly_chart(
            bar_chart_1t(df_uf_sel, f"Share 1t — {uf_sel} (IC 90%)"),
            use_container_width=True,
        )

    # ── Tab 4: 2º Turno por UF ───────────────────────────────────────────────
    with tab_uf2:
        st.subheader("Previsão 2º Turno por UF")
        col_map, col_tab = st.columns([3, 2])
        with col_map:
            try:
                geo = load_geodata()
                st.plotly_chart(mapa_2t(uf_2t, geo), use_container_width=True)
                st.caption("Vermelho = PT liderando · Azul = PL liderando")
            except Exception as exc:
                st.warning(f"Mapa não disponível: {exc}")
                st.info("Instale geobr e verifique conexão com internet para o mapa.")
        with col_tab:
            display = uf_2t[["sigla_uf", "vencedor", "share_pred_A",
                              "share_pred_B", "eleitorado_uf"]].copy()
            display.columns = ["UF", "Vencedor", "PT %", "PL %", "Eleitorado"]
            display["PT %"] = display["PT %"].map("{:.1%}".format)
            display["PL %"] = display["PL %"].map("{:.1%}".format)
            display["Eleitorado"] = display["Eleitorado"].map("{:,.0f}".format)
            display = display.sort_values("UF")

            def _highlight(row):
                c = cor(row["Vencedor"]) + "22"
                return [f"background-color: {c}"] * len(row)

            st.dataframe(
                display.style.apply(_highlight, axis=1),
                use_container_width=True,
                height=500,
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
