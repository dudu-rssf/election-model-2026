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
        title={"text": f"<b>{partido_a}</b> in the 2nd round", "font": {"size": 18}},
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
                       showarrow=False, font=dict(size=12, color="#6B7280"))
    fig.update_layout(height=280, margin=dict(t=40, b=60, l=20, r=20),
                      paper_bgcolor="white")
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
        colorscale=[[0, "#1A56DB"], [0.5, "#F3F4F6"], [1, "#CC0000"]],
        zmid=0,
        zmin=-0.4,
        zmax=0.4,
        colorbar=dict(
            title="← PL | PT →",
            showticklabels=False,
            len=0.6,
        ),
        customdata=merged[["share_pred_A", "share_pred_B"]].values,
        hovertemplate=(
            "<b>%{location}</b><br>"
            "PT: %{customdata[0]:.1%}<br>"
            "PL: %{customdata[1]:.1%}<br>"
            "<extra></extra>"
        ),
    ))
    fig.update_geos(fitbounds="locations", visible=False, bgcolor="white")
    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=480,
        paper_bgcolor="white",
    )
    return fig


# ── Layout principal ──────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="2026 Electoral Model",
        page_icon="🗳️",
        layout="wide",
    )

    st.markdown("<style>[data-testid='stSidebar']{display:none}</style>", unsafe_allow_html=True)

    # ── Carrega dados ────────────────────────────────────────────────────────
    nac_1t = load_nac_1t()
    nac_2t = load_nac_2t()
    uf_1t  = load_uf_1t()
    uf_2t  = load_uf_2t()

    # ── Header com headline numbers ──────────────────────────────────────────
    st.title("2026 Presidential Elections — Model Forecast")
    st.caption("Based on TSE historical data + May/2026 polls.")

    row = st.columns(4)
    nac_sorted = nac_1t.sort_values("share_pred", ascending=False).reset_index(drop=True)
    lider     = nac_sorted.iloc[0]
    segundo   = nac_sorted.iloc[1]
    pt_row    = nac_2t[nac_2t["sigla_partido"] == "PT"].iloc[0]
    pl_row    = nac_2t[nac_2t["sigla_partido"] == "PL"].iloc[0]
    margem_2t = abs(pt_row["share_pred"] - pl_row["share_pred"])

    with row[0]:
        st.metric(f"🥇 R1 — {lider['sigla_partido']}",
                  f"{lider['share_pred']:.1%}",
                  f"CI [{lider['share_lower']:.1%} – {lider['share_upper']:.1%}]")
    with row[1]:
        st.metric(f"🥈 R1 — {segundo['sigla_partido']}",
                  f"{segundo['share_pred']:.1%}",
                  f"CI [{segundo['share_lower']:.1%} – {segundo['share_upper']:.1%}]")
    vencedor_2t = "PT" if pt_row["share_pred"] > pl_row["share_pred"] else "PL"
    with row[2]:
        st.metric(f"🏆 R2 — {vencedor_2t} projected",
                  f"{max(pt_row['share_pred'], pl_row['share_pred']):.1%}",
                  f"margin {margem_2t:.1%}")
    pt_ufs = int((uf_2t["vencedor"] == "PT").sum())
    pl_ufs = int((uf_2t["vencedor"] == "PL").sum())
    with row[3]:
        st.metric("🗺️ States R2 (PT × PL)", f"{pt_ufs} × {pl_ufs}",
                  f"PT electorate: "
                  f"{uf_2t.loc[uf_2t['vencedor']=='PT','eleitorado_uf'].sum()/uf_2t['eleitorado_uf'].sum():.1%}")

    st.divider()

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
        st.subheader("2nd Round Forecast — National  (PT × PL)")
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
                    f"{destaque}{r['sigla_partido']}: {r['share_pred']:.2%}{destaque}  \n"
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
                st.markdown(f"- **{p}**: {n} states")

    # ── Tab 3: 1º Turno por UF ───────────────────────────────────────────────
    with tab_uf1:
        st.subheader("1st Round Forecast by State")
        ufs = sorted(uf_1t["sigla_uf"].unique().tolist())
        uf_sel = st.selectbox("Select state", ufs, index=ufs.index("SP") if "SP" in ufs else 0)
        df_uf_sel = uf_1t[uf_1t["sigla_uf"] == uf_sel].copy()
        eleitorado = df_uf_sel["eleitorado_uf"].iloc[0]
        n_mun = int(df_uf_sel["n_municipios_uf"].iloc[0])
        st.caption(f"{uf_sel} — {n_mun} municipalities · electorate {eleitorado:,.0f}")
        st.plotly_chart(
            bar_chart_1t(df_uf_sel, f"Round 1 share — {uf_sel} (90% CI)"),
            use_container_width=True,
        )

    # ── Tab 4: 2º Turno por UF ───────────────────────────────────────────────
    with tab_uf2:
        st.subheader("2nd Round Forecast by State")
        col_map, col_tab = st.columns([3, 2])
        with col_map:
            try:
                geo = load_geodata()
                st.plotly_chart(mapa_2t(uf_2t, geo), use_container_width=True)
                st.caption("Red = PT leading · Blue = PL leading")
            except Exception as exc:
                st.warning(f"Map unavailable: {exc}")
                st.info("Install geobr and check internet connection for the map.")
        with col_tab:
            display = uf_2t[["sigla_uf", "vencedor", "share_pred_A",
                              "share_pred_B", "eleitorado_uf"]].copy()
            display.columns = ["State", "Winner", "PT %", "PL %", "Electorate"]
            display["PT %"] = display["PT %"].map("{:.1%}".format)
            display["PL %"] = display["PL %"].map("{:.1%}".format)
            display["Electorate"] = display["Electorate"].map("{:,.0f}".format)
            display = display.sort_values("State")

            def _highlight(row):
                c = cor(row["Winner"]) + "22"
                return [f"background-color: {c}"] * len(row)

            st.dataframe(
                display.style.apply(_highlight, axis=1),
                use_container_width=True,
                height=500,
                hide_index=True,
            )

    # ── Tab 5: Analysis ──────────────────────────────────────────────────────
    with tab_analysis:
        st.subheader("Forecast Analysis")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("#### 2nd Round Margin by State (PT − PL)")
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
                    "PT: %{customdata[0]:.1%}<br>"
                    "PL: %{customdata[1]:.1%}<br>"
                    "Electorate: %{customdata[2]:,.0f}<extra></extra>"
                ),
            ))
            fig_m.add_vline(x=0, line_color="#6B7280", line_width=1)
            fig_m.update_layout(
                xaxis=dict(tickformat="+.0%", gridcolor="#E5E7EB", range=[-0.38, 0.38]),
                yaxis=dict(tickfont=dict(size=11), showgrid=False),
                margin=dict(l=40, r=70, t=10, b=20),
                height=max(500, len(df_m) * 22),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_m, use_container_width=True)
            st.caption("Positive = PT leads; negative = PL (Flávio) leads.")

        with col_b:
            st.markdown("#### Forecast Uncertainty by State (90% CI width)")
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
                xaxis=dict(tickformat=".0%", gridcolor="#E5E7EB"),
                yaxis=dict(tickfont=dict(size=11), showgrid=False),
                margin=dict(l=40, r=70, t=10, b=20),
                height=max(500, len(df_ci) * 22),
                plot_bgcolor="white", paper_bgcolor="white",
            )
            st.plotly_chart(fig_ci, use_container_width=True)
            st.caption("Color = projected 2nd round winner. Wider bar = more uncertain prediction.")

        st.divider()

        st.markdown("#### Round 1: PT vs PL vote share by state")
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
            line=dict(color="#D1D5DB", dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
        ))
        fig_sc.add_trace(go.Scatter(
            x=sc["pl"], y=sc["pt"],
            mode="markers+text",
            text=sc["sigla_uf"],
            textposition="top center",
            textfont=dict(size=9, color="#6B7280"),
            marker=dict(
                size=(sc["eleitorado_uf"] / sc["eleitorado_uf"].max() * 44 + 7).tolist(),
                color=[cor("PT") if pt > pl else cor("PL")
                       for pt, pl in zip(sc["pt"], sc["pl"])],
                opacity=0.85,
                line=dict(width=1, color="#D1D5DB"),
            ),
            customdata=sc[["sigla_uf", "pt", "pl", "eleitorado_uf"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "PT: %{customdata[1]:.1%}<br>"
                "PL: %{customdata[2]:.1%}<br>"
                "Electorate: %{customdata[3]:,.0f}<extra></extra>"
            ),
            showlegend=False,
        ))
        fig_sc.update_layout(
            xaxis=dict(title="PL share (R1)", tickformat=".0%",
                       gridcolor="#E5E7EB", range=[0, lim]),
            yaxis=dict(title="PT share (R1)", tickformat=".0%",
                       gridcolor="#E5E7EB", range=[0, lim]),
            margin=dict(l=60, r=20, t=10, b=60),
            height=500,
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_sc, use_container_width=True)
        st.caption(
            "Bubble size = electorate size. "
            "Above the diagonal = PT leads in that state; below = PL (Flávio) leads."
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
