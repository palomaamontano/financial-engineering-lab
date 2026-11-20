"""Elementos de interfaz compartidos: estilo, tarjetas, gráficas e inputs."""
from __future__ import annotations

import html

import plotly.graph_objects as go
import streamlit as st

PALETTE = {
    "navy": "#14213D",
    "blue": "#2F5D8A",
    "teal": "#2A9D8F",
    "gold": "#C9A227",
    "red": "#B5443C",
    "grey": "#6B7280",
    "light": "#F4F6F9",
}
SERIES = [PALETTE["blue"], PALETTE["gold"], PALETTE["teal"], PALETTE["red"], PALETTE["grey"]]

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1250px;}
h1, h2, h3 {color: #14213D; letter-spacing: -0.01em;}
div[data-testid="stMetric"] {
    background: #F4F6F9; border: 1px solid #E3E7EE; border-radius: 10px; padding: 0.7rem 0.9rem;
}
[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #4B5563;}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {
    white-space: normal !important; overflow: visible !important; text-overflow: clip !important;
}
[data-testid="stMetricValue"] {font-size: 1.55rem;}
[data-testid="stMetricValue"] * {overflow: visible !important; text-overflow: clip !important;}
@media (max-width: 1200px) {[data-testid="stMetricValue"] {font-size: 1.2rem;}}
.fel-tag {display:inline-block; padding: 0.15rem 0.55rem; border-radius: 999px; font-size: 0.75rem;
          background:#E8EEF6; color:#2F5D8A; margin-right: 0.35rem; font-weight: 600;}
.fel-box {border-left: 4px solid #2F5D8A; background:#F4F6F9; padding: 0.7rem 1rem; border-radius: 6px;
          margin: 0.4rem 0 0.8rem 0;}
.fel-out {border-left: 4px solid #C9A227;}
.fel-chart-title {font-weight: 600; color: #14213D; font-size: 1.0rem; line-height: 1.6; padding-bottom: 2px; margin: 0.3rem 0 0 0;}
</style>
"""


def setup_page():
    st.markdown(CSS, unsafe_allow_html=True)


def header(title, subtitle, tags=()):
    st.title(title)
    if tags:
        st.markdown("".join(f'<span class="fel-tag">{t}</span>' for t in tags), unsafe_allow_html=True)
    st.caption(subtitle)


def section(title):
    st.markdown(f"### {title}")


def note(text, output=False):
    cls = "fel-box fel-out" if output else "fel-box"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)


def show_chart(fig: go.Figure):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13, color="#1F2937"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(bgcolor="white"),
    )
    # El título va fuera de la figura: así la leyenda (arriba, a veces en dos renglones) nunca lo tapa.
    title = fig.layout.title.text if fig.layout.title else None
    if title:
        fig.update_layout(title=None, margin=dict(t=40))
        safe = html.escape(title).replace("_", "&#95;").replace("*", "&#42;").replace("$", "&#36;")
        st.markdown(f'<div class="fel-chart-title">{safe}</div>', unsafe_allow_html=True)
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def show_table(df, formats=None):
    styler = df.style.format(formats or {}, na_rep="")
    st.dataframe(styler, hide_index=True)


def init_state(defaults: dict):
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def apply_scenario(values: dict):
    for key, value in values.items():
        st.session_state[key] = value


def scenario_picker(label, scenarios: dict, key):
    """Selector que carga un escenario de clase en los inputs."""
    names = list(scenarios.keys())

    def _load():
        choice = st.session_state[key]
        if choice in scenarios and scenarios[choice]:
            apply_scenario(scenarios[choice])

    st.selectbox(label, names, key=key, on_change=_load,
                 help="Carga en los inputs los datos de un ejercicio visto en clase.")


def pct_input(label, key, step=0.1, min_value=None, max_value=None, help=None, fmt="%.4f"):
    """Input capturado en %; devuelve el decimal."""
    v = st.number_input(label + " (%)", key=key, step=step, min_value=min_value,
                        max_value=max_value, help=help, format=fmt)
    return v / 100.0


def fmt_money(x, d=4):
    return f"{x:,.{d}f}"


def fmt_pct(x, d=4):
    return f"{100 * x:,.{d}f}%"
