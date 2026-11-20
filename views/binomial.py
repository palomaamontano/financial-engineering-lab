import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from models import binomial as bn
from models import black_scholes as bs

ui.setup_page()
ui.header(
    "Árbol binomial",
    "Opciones europeas y americanas, call y put, por cartera réplica y probabilidad neutral al riesgo.",
    tags=("Clases 13 y 14", "Europeas y americanas", "CRR"),
)

DEFAULTS = {"bn_S": 50.0, "bn_K": 52.0, "bn_r": 5.0, "bn_q": 0.0, "bn_T": 2.0, "bn_N": 2,
            "bn_mode": "Manual (u y d dados)", "bn_u": 1.2, "bn_d": 0.8, "bn_sigma": 30.0,
            "bn_type": "put", "bn_style": "Europea"}
ui.init_state(DEFAULTS)
SCENARIOS = {
    "Personalizado": None,
    "Clase 14 · put europeo de 2 periodos (S=50, K=52, u=1.2, d=0.8)": {**DEFAULTS},
    "Clase 14 · call de 1 periodo (S=100, K=100, u=1.05, d=0.95)": {
        "bn_S": 100.0, "bn_K": 100.0, "bn_r": 2.0, "bn_q": 0.0, "bn_T": 1.0, "bn_N": 1,
        "bn_mode": "Manual (u y d dados)", "bn_u": 1.05, "bn_d": 0.95, "bn_type": "call", "bn_style": "Europea"},
    "CRR · put americano con ejercicio anticipado (S=100, K=110, σ=20%)": {
        "bn_S": 100.0, "bn_K": 110.0, "bn_r": 8.0, "bn_q": 0.0, "bn_T": 1.0, "bn_N": 6,
        "bn_mode": "CRR (desde σ)", "bn_sigma": 20.0, "bn_type": "put", "bn_style": "Americana"},
}
MODES = {"Manual (u y d dados)": "manual", "CRR (desde σ)": "crr", "Árbol forward de McDonald (desde σ)": "forward"}

with st.sidebar:
    st.subheader("Inputs")
    ui.scenario_picker("Escenario de clase", SCENARIOS, "bn_scn")
    S = st.number_input("Precio del subyacente S(0)", key="bn_S", min_value=0.0001, step=1.0, format="%.4f")
    K = st.number_input("Strike K", key="bn_K", min_value=0.0001, step=1.0, format="%.4f")
    r = ui.pct_input("Tasa libre de riesgo r (continua)", "bn_r")
    q = ui.pct_input("Rendimiento por dividendos δ", "bn_q")
    T = st.number_input("Plazo T (años)", key="bn_T", min_value=0.0001, step=0.25, format="%.4f")
    N = int(st.number_input("Número de pasos N", key="bn_N", min_value=1, max_value=1000, step=1))
    mode_label = st.selectbox("Cómo obtener u y d", list(MODES), key="bn_mode")
    mode = MODES[mode_label]
    if mode == "manual":
        u_in = st.number_input("Factor de subida u", key="bn_u", min_value=0.0001, step=0.01, format="%.4f")
        d_in = st.number_input("Factor de bajada d", key="bn_d", min_value=0.0001, step=0.01, format="%.4f")
        sigma = None
    else:
        sigma = ui.pct_input("Volatilidad σ", "bn_sigma", min_value=0.01)
        u_in = d_in = None
    opt = st.radio("Tipo", ["call", "put"], key="bn_type", horizontal=True)
    style_label = st.radio("Ejercicio", ["Europea", "Americana"], key="bn_style", horizontal=True)
    style = "european" if style_label == "Europea" else "american"

try:
    res = bn.price_binomial(S, K, r, q, T, N, opt, style, mode, sigma, u_in, d_in)
    other_style = "american" if style == "european" else "european"
    res_other = bn.price_binomial(S, K, r, q, T, N, opt, other_style, mode, sigma, u_in, d_in, store=False)
except ValueError as err:
    st.error(str(err))
    st.stop()

prm = res.params
eu_price = res.price if style == "european" else res_other.price
am_price = res.price if style == "american" else res_other.price

tab_tree, tab_steps, tab_conv, tab_theory = st.tabs(
    ["Árbol", "Valoración hacia atrás", "Convergencia y sensibilidad", "Fundamento y defensa"])

with tab_tree:
    c = st.columns(5)
    c[0].metric(f"Precio {opt} {style_label.lower()}", ui.fmt_money(res.price))
    c[1].metric("u", f"{prm.u:.6f}")
    c[2].metric("d", f"{prm.d:.6f}")
    c[3].metric("p*", f"{prm.p:.6f}")
    c[4].metric("h = T/N", f"{prm.h:.4f}")
    e1, e2, e3, e4 = st.columns(4)
    e1.metric("Europea", ui.fmt_money(eu_price))
    e2.metric("Americana", ui.fmt_money(am_price))
    e3.metric("Prima por ejercicio anticipado", ui.fmt_money(am_price - eu_price))
    e4.metric("Nodos con ejercicio anticipado", f"{res.early_exercise_nodes}")

    st.latex(rf"p^*=\frac{{e^{{(r-\delta)h}}-d}}{{u-d}}=\frac{{e^{{({r:.4f}-{q:.4f})({prm.h:.4g})}}-{prm.d:.4f}}}"
             rf"{{{prm.u:.4f}-{prm.d:.4f}}}={prm.p:.4f}\qquad d<e^{{(r-\delta)h}}={prm.growth:.4f}<u")

    if N <= 25 and res.stock:
        show_labels = N <= 6
        fig = go.Figure()
        for n in range(N):
            for j in range(n + 1):
                for jj in (j, j + 1):
                    fig.add_trace(go.Scatter(x=[n, n + 1], y=[2 * j - n, 2 * jj - (n + 1)], mode="lines",
                                             line=dict(color="#C9D3E0", width=1.4), hoverinfo="skip", showlegend=False))
        xs, ys, colors, texts, hovers = [], [], [], [], []
        for n in range(N + 1):
            for j in range(n + 1):
                s_val, v_val, ex = res.stock[n][j], res.value[n][j], bool(res.exercise[n][j])
                xs.append(n)
                ys.append(2 * j - n)
                colors.append(ui.PALETTE["red"] if ex else (ui.PALETTE["navy"] if n == 0 else ui.PALETTE["blue"]))
                texts.append(f"S={s_val:.2f}<br>V={v_val:.2f}")
                extra = ""
                if n < N:
                    extra = f"<br>Δ={res.delta[n][j]:.4f}<br>B={res.bond[n][j]:.4f}"
                hovers.append(f"Paso {n}, subidas {j}<br>S={s_val:.4f}<br>Valor={v_val:.4f}{extra}"
                              + ("<br><b>Ejercicio anticipado</b>" if ex else ""))
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text" if show_labels else "markers",
            marker=dict(size=34 if show_labels else 12, color=colors, line=dict(color="white", width=1.5)),
            text=texts if show_labels else None, textposition="top center",
            textfont=dict(size=11, color="#14213D"), hovertext=hovers, hoverinfo="text", showlegend=False))
        fig.update_layout(title="Árbol del subyacente (S) y de la opción (V). Rojo: conviene ejercer anticipadamente",
                          xaxis=dict(title="Paso n", dtick=1), yaxis=dict(visible=False),
                          height=420 + 18 * N)
        ui.show_chart(fig)
        st.caption("Pasa el cursor sobre un nodo para ver Δ (acciones) y B (bono) de la cartera réplica.")
    else:
        st.info("Con más de 25 pasos el árbol completo no es legible: se muestran el precio, la convergencia y la "
                "sensibilidad. Reduce N para ver los nodos.")

with tab_steps:
    if res.stock and N <= 10:
        rows = []
        for n in range(N, -1, -1):
            for j in range(n, -1, -1):
                row = {"Paso n": n, "Subidas j": j, "S": res.stock[n][j], "Valor V": res.value[n][j],
                       "¿Ejerce?": "Sí" if res.exercise[n][j] else ""}
                if n < N:
                    vu, vd = res.value[n + 1][j + 1], res.value[n + 1][j]
                    row.update({"V_u": vu, "V_d": vd,
                                "Continuación": math.exp(-r * prm.h) * (prm.p * vu + (1 - prm.p) * vd),
                                "Δ": res.delta[n][j], "B": res.bond[n][j]})
                rows.append(row)
        df = pd.DataFrame(rows)
        ui.show_table(df, {c: "{:.4f}" for c in ["S", "Valor V", "V_u", "V_d", "Continuación", "Δ", "B"]})
        st.latex(r"V=e^{-rh}\left[p^*V_u+(1-p^*)V_d\right],\quad \Delta=e^{-\delta h}\frac{V_u-V_d}{S(u-d)},"
                 r"\quad B=e^{-rh}\frac{uV_d-dV_u}{u-d},\quad V=\Delta S+B")
        if res.delta:
            st.latex(rf"\text{{Nodo inicial: }}\Delta S+B=({res.delta[0][0]:.4f})({S:.4g})+({res.bond[0][0]:.4f})={res.price:.4f}")
        if style == "american":
            st.markdown("En la americana cada nodo toma el máximo entre la continuación y el ejercicio inmediato.")
    else:
        st.info("La tabla nodo por nodo se muestra hasta N = 10.")

with tab_conv:
    if mode == "manual":
        st.info("La convergencia hacia Black-Scholes requiere σ: cambia a CRR o al árbol forward. "
                "Con u y d fijos, cambiar N cambia el modelo, no sólo la discretización.")
    else:
        n_max = st.slider("N máximo", 10, 300, 150, step=10)
        steps, prices = bn.convergence(S, K, r, q, sigma, T, opt, style, n_max, mode)
        bsm = bs.bs_price(S, K, r, q, sigma, T, opt).price
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=steps, y=prices, name=f"Árbol ({style_label.lower()})",
                                 line=dict(color=ui.PALETTE["blue"], width=1.8)))
        fig.add_hline(y=bsm, line_dash="dash", line_color=ui.PALETTE["gold"],
                      annotation_text=f"BSM europea = {bsm:.4f}")
        fig.update_layout(title="Precio del árbol conforme aumentan los pasos", xaxis_title="N", yaxis_title="Precio", height=400)
        ui.show_chart(fig)
        if style == "european":
            st.caption(f"Con N = {n_max}: árbol {prices[-1]:.4f}, BSM {bsm:.4f}, error {prices[-1] - bsm:+.5f}. "
                       "La oscilación entre N pares e impares es propia de CRR.")
        else:
            st.caption("La americana converge a su propio valor, que no tiene fórmula cerrada; la línea de BSM "
                       "es la europea y sirve de piso cuando no hay dividendos en el put.")

    ui.section("Sensibilidad al spot: europea vs americana")
    xs = np.linspace(0.5 * S, 1.5 * S, 41)
    fig = go.Figure()
    for sty, name, color in (("european", "Europea", ui.PALETTE["blue"]), ("american", "Americana", ui.PALETTE["red"])):
        ys = []
        for x in xs:
            try:
                ys.append(bn.price_binomial(x, K, r, q, T, min(N, 200), opt, sty, mode, sigma, u_in, d_in, store=False).price)
            except ValueError:
                ys.append(np.nan)
        fig.add_trace(go.Scatter(x=xs, y=ys, name=name, line=dict(color=color, width=2.3)))
    payoff = np.maximum(xs - K, 0) if opt == "call" else np.maximum(K - xs, 0)
    fig.add_trace(go.Scatter(x=xs, y=payoff, name="Valor de ejercicio inmediato", line=dict(color=ui.PALETTE["grey"], dash="dot")))
    fig.update_layout(xaxis_title="S(0)", yaxis_title="Precio", height=380)
    ui.show_chart(fig)

with tab_theory:
    st.markdown("""
**Qué calcula.** El precio de una opción replicándola con Δ acciones y B pesos en el bono libre de riesgo
en cada nodo. Si la cartera paga lo mismo que la opción en ambos estados, por no arbitraje cuesta lo mismo.

**Probabilidad neutral al riesgo.** p* = (e^((r−δ)h) − d)/(u − d) no es la probabilidad real: es la que hace
que el subyacente crezca a la tasa libre de riesgo, E*[S_h] = S·e^((r−δ)h) = F₀,ₕ(S).
Sólo existe si d < e^((r−δ)h) < u; si no, hay arbitraje y la app se niega a valuar.

**Europea vs americana.** La americana puede ejercerse en cualquier nodo: V = max(continuación, ejercicio).
Por eso nunca vale menos que la europea. Un call americano sin dividendos nunca se ejerce antes (vale lo
mismo que el europeo); un put americano sí puede ejercerse antes, porque cobrar K hoy genera intereses.

**Modos de u y d.** En los ejercicios de clase u y d vienen dados. CRR fija u = e^(σ√h) y d = 1/u.
El árbol forward de McDonald usa u = e^((r−δ)h+σ√h), d = e^((r−δ)h−σ√h).

**Convergencia.** Con u y d derivados de σ, al aumentar N el árbol europeo converge a Black-Scholes; el error
oscila y decrece del orden de 1/N (diferencia metodológica numérica, no error de programación).

**¿Cómo validamos?** Ejemplos de la clase 14 (3.44, 15.69 con el straddle como call + put, 4.19),
paridad put-call en el árbol europeo, americana ≥ europea, call americano sin dividendos = europeo y
convergencia a BSM con error menor a 0.01 con N = 500.
""")
