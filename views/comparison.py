import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from models import binomial as bn
from models import black_scholes as bs
from models import monte_carlo as mc

ui.setup_page()
ui.header(
    "Model Comparison Lab",
    "Un mismo instrumento valuado con Black-Scholes, árbol CRR y Monte Carlo, con los mismos inputs y convenciones.",
    tags=("BSM", "CRR", "Monte Carlo"),
)

ui.init_state({"cmp_S": 100.0, "cmp_K": 100.0, "cmp_r": 5.0, "cmp_q": 0.0, "cmp_sigma": 20.0, "cmp_T": 1.0,
               "cmp_type": "put", "cmp_N": 200, "cmp_n": 100_000, "cmp_seed": 42})


@st.cache_data(show_spinner=False)
def mc_cached(S, K, r, q, sigma, T, n, seed, opt):
    out = mc.mc_european(S, K, r, q, sigma, T, n, seed, opt)
    out.pop("ST_muestra")
    return out


@st.cache_data(show_spinner=False)
def conv_cached(S, K, r, q, sigma, T, opt, n_max):
    return bn.convergence(S, K, r, q, sigma, T, opt, "european", n_max)


with st.sidebar:
    st.subheader("Inputs comunes")
    S = st.number_input("S(0)", key="cmp_S", min_value=0.0001, step=1.0)
    K = st.number_input("K", key="cmp_K", min_value=0.0001, step=1.0)
    r = ui.pct_input("r (continua)", "cmp_r")
    q = ui.pct_input("δ", "cmp_q")
    sigma = ui.pct_input("σ", "cmp_sigma", min_value=0.01)
    T = st.number_input("T (años)", key="cmp_T", min_value=0.0001, step=0.25)
    opt = st.radio("Tipo", ["call", "put"], key="cmp_type", horizontal=True)
    N = int(st.number_input("Pasos del árbol N", key="cmp_N", min_value=1, max_value=1000, step=10))
    n = st.select_slider("Simulaciones Monte Carlo", [1_000, 10_000, 50_000, 100_000, 200_000, 500_000], key="cmp_n")
    seed = int(st.number_input("Semilla", key="cmp_seed", min_value=0, step=1))

try:
    exact = bs.bs_price(S, K, r, q, sigma, T, opt).price
    tree_eu = bn.price_binomial(S, K, r, q, T, N, opt, "european", "crr", sigma, store=False).price
    tree_am = bn.price_binomial(S, K, r, q, T, N, opt, "american", "crr", sigma, store=False).price
    sim = mc_cached(S, K, r, q, sigma, T, n, seed, opt)
except ValueError as err:
    st.error(str(err))
    st.stop()

ui.section(f"Opción {opt} europea")
rows = [
    {"Modelo": "Black-Scholes-Merton", "Precio": exact, "Dif. vs BSM": 0.0, "Dif. %": 0.0,
     "Incertidumbre": "Exacto (fórmula cerrada)", "Tipo de diferencia": "Referencia analítica"},
    {"Modelo": f"Árbol CRR (N = {N})", "Precio": tree_eu, "Dif. vs BSM": tree_eu - exact,
     "Dif. %": 100 * (tree_eu / exact - 1) if exact else 0.0,
     "Incertidumbre": "Determinista", "Tipo de diferencia": "Numérica (discretización)"},
    {"Modelo": f"Monte Carlo (n = {n:,})", "Precio": sim["precio"], "Dif. vs BSM": sim["precio"] - exact,
     "Dif. %": 100 * (sim["precio"] / exact - 1) if exact else 0.0,
     "Incertidumbre": f"EE {sim['error_estandar']:.4f}; IC [{sim['ic_inferior']:.4f}, {sim['ic_superior']:.4f}]",
     "Tipo de diferencia": "Estadística (muestreo)"},
]
df = pd.DataFrame(rows)
ui.show_table(df, {"Precio": "{:.5f}", "Dif. vs BSM": "{:+.5f}", "Dif. %": "{:+.3f}%"})
z = (sim["precio"] - exact) / sim["error_estandar"]
ui.note(f"Monte Carlo difiere de BSM en {z:+.2f} errores estándar: "
        + ("dentro de lo esperado por azar (|z| < 1.96)." if abs(z) < 1.96 else "fuera del 95%; prueba otra semilla o más simulaciones.")
        + f" El árbol difiere en {tree_eu - exact:+.5f}, un error numérico que se reduce al aumentar N.", output=True)

fig = go.Figure(go.Bar(x=df["Modelo"], y=df["Precio"], marker_color=[ui.PALETTE["gold"], ui.PALETTE["blue"], ui.PALETTE["teal"]],
                       error_y=dict(type="data", array=[0, 0, 1.96 * sim["error_estandar"]], visible=True),
                       text=[f"{v:.4f}" for v in df["Precio"]], textposition="outside"))
lo = min(df["Precio"]) - 4 * sim["error_estandar"]
hi = max(df["Precio"]) + 4 * sim["error_estandar"]
fig.update_layout(title="Precio por modelo (barra de error: IC 95% de Monte Carlo)", yaxis=dict(range=[lo, hi]), height=380)
ui.show_chart(fig)

left, right = st.columns(2)
with left:
    steps, prices = conv_cached(S, K, r, q, sigma, T, opt, 150)
    fig = go.Figure(go.Scatter(x=steps, y=prices, name="CRR", line=dict(color=ui.PALETTE["blue"])))
    fig.add_hline(y=exact, line_dash="dash", line_color=ui.PALETTE["gold"], annotation_text="BSM")
    fig.update_layout(title="Convergencia del árbol al aumentar N", xaxis_title="N", height=360)
    ui.show_chart(fig)
with right:
    ns, means, ses = mc.running_estimate(sim["payoff_descontado"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=means + 1.96 * ses, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=ns, y=means - 1.96 * ses, fill="tonexty", fillcolor="rgba(42,157,143,0.18)",
                             line=dict(width=0), name="IC 95%"))
    fig.add_trace(go.Scatter(x=ns, y=means, name="Monte Carlo", line=dict(color=ui.PALETTE["teal"])))
    fig.add_hline(y=exact, line_dash="dash", line_color=ui.PALETTE["gold"], annotation_text="BSM")
    fig.update_layout(title="Convergencia de Monte Carlo", xaxis_type="log", xaxis_title="n", height=360)
    ui.show_chart(fig)

ui.section("¿Y si la opción es americana?")
c1, c2, c3 = st.columns(3)
c1.metric("Árbol americano", f"{tree_am:.5f}")
c2.metric("Árbol europeo", f"{tree_eu:.5f}")
c3.metric("Prima por ejercicio anticipado", f"{tree_am - tree_eu:.5f}")
st.caption("Black-Scholes y este Monte Carlo sólo valúan europeas: no se comparan contra la americana.")

with st.expander("Preguntas de defensa", expanded=True):
    st.markdown("""
**¿Por qué Monte Carlo no da exactamente lo mismo que Black-Scholes?** Usan el mismo modelo (GBM con tasa y
volatilidad constantes); Monte Carlo estima la esperanza con una muestra finita. La diferencia es error
estadístico, cae como 1/√n y se mide con el error estándar. No es una diferencia metodológica.

**¿Por qué el árbol se acerca a BSM al aumentar los pasos?** Con u = e^(σ√h) y p* neutral al riesgo, la
distribución binomial de ln S_T converge a la normal del GBM cuando h → 0 (teorema central del límite). La
diferencia es numérica y oscila entre N pares e impares.

**¿Cuál modelo es más apropiado para una americana?** El árbol, porque en cada nodo compara continuar
con ejercer. Black-Scholes no considera el ejercicio anticipado y Monte Carlo estándar no conoce el valor de
continuación en cada fecha.

**¿Cuándo coinciden las tres?** Para una europea con los mismos inputs, cuando N y n son grandes.
""")
