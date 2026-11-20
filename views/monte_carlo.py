import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import lognorm

from components import ui
from models import black_scholes as bs
from models import monte_carlo as mc

ui.setup_page()
ui.header(
    "Monte Carlo",
    "Valoración de opciones europeas simulando el subyacente con movimiento browniano geométrico neutral al riesgo.",
    tags=("Clase 16", "GBM", "Error estándar e IC"),
)

DEFAULTS = {"mc_S": 41.0, "mc_K": 40.0, "mc_r": 8.0, "mc_q": 0.0, "mc_sigma": 30.0, "mc_T": 0.25,
            "mc_n": 50_000, "mc_steps": 60, "mc_seed": 42, "mc_type": "call", "mc_anti": False}
ui.init_state(DEFAULTS)
SCENARIOS = {"Personalizado": None, "Clase 16 · call S=41, K=40, σ=30%, r=8%, T=3/12": {**DEFAULTS}}
N_OPTIONS = [1_000, 5_000, 10_000, 50_000, 100_000, 200_000, 500_000]


@st.cache_data(show_spinner=False)
def run_mc(S, K, r, q, sigma, T, n, seed, opt, anti):
    return mc.mc_european(S, K, r, q, sigma, T, n, seed, opt, anti)


@st.cache_data(show_spinner=False)
def run_paths(S, r, q, sigma, T, steps, seed):
    return mc.simulate_paths(S, r, q, sigma, T, steps, 60, seed)


with st.sidebar:
    st.subheader("Inputs")
    ui.scenario_picker("Escenario de clase", SCENARIOS, "mc_scn")
    S = st.number_input("Precio del subyacente S(0)", key="mc_S", min_value=0.0001, step=1.0, format="%.4f")
    K = st.number_input("Strike K", key="mc_K", min_value=0.0001, step=1.0, format="%.4f")
    r = ui.pct_input("Tasa libre de riesgo r (continua)", "mc_r")
    q = ui.pct_input("Rendimiento por dividendos δ", "mc_q")
    sigma = ui.pct_input("Volatilidad σ", "mc_sigma", min_value=0.01)
    T = st.number_input("Plazo T (años)", key="mc_T", min_value=0.0001, step=0.25, format="%.4f")
    n = st.select_slider("Número de simulaciones", N_OPTIONS, key="mc_n")
    steps = int(st.number_input("Pasos temporales (para trayectorias)", key="mc_steps", min_value=1,
                                max_value=mc.MAX_STEPS, step=10))
    seed = int(st.number_input("Semilla aleatoria", key="mc_seed", min_value=0, step=1))
    opt = st.radio("Tipo", ["call", "put"], key="mc_type", horizontal=True)
    anti = st.toggle("Variables antitéticas (reduce varianza)", key="mc_anti")

try:
    res = run_mc(S, K, r, q, sigma, T, n, seed, opt, anti)
    exact = bs.bs_price(S, K, r, q, sigma, T, opt).price
except ValueError as err:
    st.error(str(err))
    st.stop()

price, se = res["precio"], res["error_estandar"]
tab_res, tab_conv, tab_theory = st.tabs(["Resultado", "Convergencia y número de simulaciones", "Fundamento y defensa"])

with tab_res:
    c = st.columns(3)
    c[0].metric(f"Precio Monte Carlo ({opt})", ui.fmt_money(price))
    c[1].metric("Black-Scholes", ui.fmt_money(exact), f"MC − BSM = {price - exact:+.4f}")
    c[2].metric("Diferencia en errores estándar", f"{(price - exact) / se:+.2f}")
    c = st.columns(3)
    c[0].metric("Error estándar", f"{se:.5f}")
    c[1].metric("IC 95%", f"[{res['ic_inferior']:.4f}, {res['ic_superior']:.4f}]")
    c[2].metric("Ancho del IC", f"{res['ic_superior'] - res['ic_inferior']:.4f}")
    inside = res["ic_inferior"] <= exact <= res["ic_superior"]
    ui.note(("El precio de Black-Scholes está dentro del intervalo de confianza: la diferencia es error de "
             "simulación, no un error del modelo." if inside else
             "El precio de Black-Scholes quedó fuera del IC 95%. Ocurre en 1 de cada 20 semillas por azar; "
             "prueba otra semilla o más simulaciones."), output=True)

    st.latex(r"S_T=S_0\exp\left[(r-\delta-\tfrac{\sigma^2}{2})T+\sigma\sqrt{T}Z\right],\qquad "
             r"\hat c=e^{-rT}\frac{1}{n}\sum_{i=1}^{n}\max(S_T^{(i)}-K,0),\qquad EE=\frac{s}{\sqrt n}")
    st.latex(rf"\hat V={price:.4f},\quad EE={se:.5f},\quad IC_{{95\%}}={price:.4f}\pm 1.96({se:.5f})")

    left, right = st.columns(2)
    with left:
        times, paths = run_paths(S, r, q, sigma, T, steps, seed)
        fig = go.Figure()
        for p in paths[:60]:
            fig.add_trace(go.Scatter(x=times, y=p, mode="lines", line=dict(width=1), opacity=0.45,
                                     showlegend=False, hoverinfo="skip"))
        fig.add_hline(y=K, line_dash="dash", line_color=ui.PALETTE["red"], annotation_text="Strike K")
        fig.add_trace(go.Scatter(x=times, y=S * np.exp((r - q) * times), name="E*[S_t] = S e^((r−δ)t)",
                                 line=dict(color="black", width=2.5)))
        fig.update_layout(title="60 trayectorias simuladas", xaxis_title="t (años)", yaxis_title="S_t", height=400)
        ui.show_chart(fig)
    with right:
        st_sample = res["ST_muestra"]
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=st_sample, nbinsx=70, histnorm="probability density", name="S_T simulado",
                                   marker_color=ui.PALETTE["blue"], opacity=0.75))
        grid = np.linspace(st_sample.min(), st_sample.max(), 300)
        mu = math.log(S) + (r - q - 0.5 * sigma ** 2) * T
        dens = lognorm.pdf(grid, s=sigma * math.sqrt(T), scale=math.exp(mu))
        fig.add_trace(go.Scatter(x=grid, y=dens, name="Lognormal teórica", line=dict(color=ui.PALETTE["gold"], width=2.5)))
        fig.add_vline(x=K, line_dash="dash", line_color=ui.PALETTE["red"])
        fig.update_layout(title="Distribución de S_T", xaxis_title="S_T", yaxis_title="Densidad", height=400)
        ui.show_chart(fig)
    st.caption(f"Media simulada de S_T: {res['media_ST']:.4f} vs forward teórico S e^((r−δ)T) = {res['forward_teorico']:.4f}. "
               f"Probabilidad simulada de ejercicio: {res['prob_ejercicio']:.2%}.")

with tab_conv:
    ns, means, ses = mc.running_estimate(res["payoff_descontado"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=means + 1.96 * ses, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ns, y=means - 1.96 * ses, fill="tonexty", fillcolor="rgba(47,93,138,0.15)",
                             line=dict(width=0), name="IC 95%"))
    fig.add_trace(go.Scatter(x=ns, y=means, name="Estimación acumulada", line=dict(color=ui.PALETTE["blue"], width=2)))
    fig.add_hline(y=exact, line_dash="dash", line_color=ui.PALETTE["gold"], annotation_text=f"BSM = {exact:.4f}")
    fig.update_layout(title="Convergencia de la estimación", xaxis_type="log", xaxis_title="Simulaciones (escala log)",
                      yaxis_title="Precio", height=420)
    ui.show_chart(fig)

    ui.section("Efecto del número de simulaciones (misma semilla)")
    rows = []
    for m in N_OPTIONS[:6]:
        rr = run_mc(S, K, r, q, sigma, T, m, seed, opt, anti)
        rows.append({"Simulaciones": m, "Precio": rr["precio"], "Error estándar": rr["error_estandar"],
                     "IC inferior": rr["ic_inferior"], "IC superior": rr["ic_superior"],
                     "MC − BSM": rr["precio"] - exact})
    df = pd.DataFrame(rows)
    ui.show_table(df, {"Simulaciones": "{:,.0f}", "Precio": "{:.4f}", "Error estándar": "{:.5f}",
                       "IC inferior": "{:.4f}", "IC superior": "{:.4f}", "MC − BSM": "{:+.4f}"})
    fig = go.Figure(go.Scatter(x=df["Simulaciones"], y=df["Error estándar"], mode="lines+markers",
                               line=dict(color=ui.PALETTE["red"], width=2.5), name="Error estándar"))
    fig.update_layout(title="El error estándar cae como 1/√n: 4 veces más simulaciones, la mitad de error",
                      xaxis_type="log", yaxis_type="log", xaxis_title="n", yaxis_title="EE", height=360)
    ui.show_chart(fig)

with tab_theory:
    st.markdown("""
**Metodología.** Bajo la medida neutral al riesgo el subyacente sigue un GBM con deriva r − δ (clase 16):
ln S_T ~ N(ln S₀ + (r − δ − σ²/2)T, σ²T). Se simulan n valores de S_T, se calcula el payoff de cada uno,
se promedia y se descuenta a la tasa libre de riesgo.

**Por qué no da exactamente Black-Scholes.** El precio simulado es un estimador: tiene error estadístico
que cae como 1/√n. La diferencia con BSM no es metodológica (es el mismo modelo), es de muestreo.
Por eso se reporta con su error estándar y su intervalo de confianza.

**Pasos temporales.** Para una opción europea sólo importa S_T, y la solución del GBM es exacta: los pasos
no cambian el precio y aquí sólo se usan para dibujar trayectorias. Importarían en opciones que dependen
de la trayectoria (asiáticas, barreras) o si se discretizara la ecuación con un esquema aproximado.

**Semilla.** Fijarla hace el experimento reproducible: con la misma semilla el profesor obtiene el mismo número.

**Variables antitéticas.** Usar Z y −Z por pares reduce la varianza sin sesgar el estimador.

**Limitaciones.** Es más lento que la fórmula cerrada; aquí se limita a 500,000 simulaciones. No valúa
americanas: para eso se usa el árbol (o métodos como Longstaff-Schwartz, fuera del alcance).

**¿Cómo validamos?** Con 200,000 simulaciones el precio queda a menos de 3 errores estándar de BSM, y en
200 semillas el IC 95% contiene al precio exacto entre 90% y 99% de las veces.
""")
