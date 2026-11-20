import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from models import black_scholes as bs
from utils import rules

ui.setup_page()
ui.header(
    "Black-Scholes-Merton",
    "Opciones europeas sobre acciones con rendimiento continuo δ o sobre divisas (Garman-Kohlhagen).",
    tags=("Clases 16 y 17", "Europeas", "Fórmula cerrada"),
)

DEFAULTS = {"bs_S": 41.0, "bs_K": 40.0, "bs_r": 8.0, "bs_q": 0.0, "bs_sigma": 30.0, "bs_T": 0.25,
            "bs_type": "call", "bs_fx": False, "bs_table": False}
ui.init_state(DEFAULTS)
SCENARIOS = {
    "Personalizado": None,
    "Clase 16 · call S=41, K=40, σ=30%, r=8%, T=3/12": {**DEFAULTS},
    "Clase 17 · put USD/EUR X=1.2, K=1.1, σ=15%, r$=6%, r€=3%, T=0.5": {
        "bs_S": 1.2, "bs_K": 1.1, "bs_r": 6.0, "bs_q": 3.0, "bs_sigma": 15.0, "bs_T": 0.5,
        "bs_type": "put", "bs_fx": True, "bs_table": False},
}

with st.sidebar:
    st.subheader("Inputs")
    ui.scenario_picker("Escenario de clase", SCENARIOS, "bs_scn")
    fx_mode = st.toggle("Modo divisas (Garman-Kohlhagen)", key="bs_fx")
    S = st.number_input("Tipo de cambio X(t)" if fx_mode else "Precio del subyacente S(t)",
                        key="bs_S", min_value=0.0001, step=1.0, format="%.4f")
    K = st.number_input("Strike K", key="bs_K", min_value=0.0001, step=1.0, format="%.4f")
    r = ui.pct_input("Tasa doméstica r (continua)" if fx_mode else "Tasa libre de riesgo r (continua)", "bs_r")
    q = ui.pct_input("Tasa extranjera r_f" if fx_mode else "Rendimiento por dividendos δ", "bs_q")
    sigma = ui.pct_input("Volatilidad σ", "bs_sigma", min_value=0.01)
    T = st.number_input("Plazo T − t (años)", key="bs_T", min_value=0.0001, step=0.25, format="%.4f")
    opt = st.radio("Tipo de opción", ["call", "put"], key="bs_type", horizontal=True)
    table_mode = st.toggle("Usar tabla Z (redondeo de clase)", key="bs_table",
                           help="Redondea d₁ y d₂ a dos decimales y N(·) a cuatro, como en los ejercicios a mano.")

try:
    res = bs.bs_price(S, K, r, q, sigma, T, opt, table_mode)
    other = bs.bs_price(S, K, r, q, sigma, T, "put" if opt == "call" else "call", table_mode)
    g = bs.greeks(S, K, r, q, sigma, T, opt)
    parity = bs.put_call_parity(S, K, r, q, sigma, T)
except ValueError as err:
    st.error(str(err))
    st.stop()

tab_res, tab_sens, tab_theory = st.tabs(["Resultado", "Sensibilidad", "Fundamento y defensa"])

with tab_res:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(f"Precio {opt}", ui.fmt_money(res.price))
    c2.metric("d₁", f"{res.d1:.4f}")
    c3.metric("d₂", f"{res.d2:.4f}")
    lbl = "N(d₁)" if opt == "call" else "N(−d₁)"
    lbl2 = "N(d₂)" if opt == "call" else "N(−d₂)"
    c4.metric(lbl, f"{res.n_d1:.4f}")
    c5.metric(lbl2, f"{res.n_d2:.4f}")
    if table_mode:
        exact = bs.bs_price(S, K, r, q, sigma, T, opt).price
        ui.note(f"Con tabla Z el precio es {res.price:.4f}; con N(·) exacta es {exact:.4f}. "
                f"Diferencia: {res.price - exact:+.4f} ({(res.price / exact - 1) * 100:+.2f}%).")

    ui.section("Fórmula con los valores sustituidos")
    qsym = "r_f" if fx_mode else r"\delta"
    st.latex(rf"d_1=\frac{{\ln(S/K)+(r-{qsym}+\sigma^2/2)T}}{{\sigma\sqrt{{T}}}}"
             rf"=\frac{{\ln({S:.4g}/{K:.4g})+({r:.4f}-{q:.4f}+{sigma:.4f}^2/2)({T:.4g})}}{{{sigma:.4f}\sqrt{{{T:.4g}}}}}={res.d1:.4f}")
    st.latex(rf"d_2=d_1-\sigma\sqrt{{T}}={res.d1:.4f}-{sigma:.4f}\sqrt{{{T:.4g}}}={res.d2:.4f}")
    if opt == "call":
        st.latex(rf"c=S e^{{-{qsym} T}}N(d_1)-K e^{{-rT}}N(d_2)="
                 rf"{S:.4g}e^{{-{q:.4f}({T:.4g})}}({res.n_d1:.4f})-{K:.4g}e^{{-{r:.4f}({T:.4g})}}({res.n_d2:.4f})={res.price:.4f}")
    else:
        st.latex(rf"p=K e^{{-rT}}N(-d_2)-S e^{{-{qsym} T}}N(-d_1)="
                 rf"{K:.4g}e^{{-{r:.4f}({T:.4g})}}({res.n_d2:.4f})-{S:.4g}e^{{-{q:.4f}({T:.4g})}}({res.n_d1:.4f})={res.price:.4f}")

    left, right = st.columns([3, 2])
    with left:
        ui.section("Greeks")
        gdf = pd.DataFrame({
            "Greek": list(g.keys()),
            "Valor": list(g.values()),
            "Lectura": [
                "Cambio en el precio por +1 unidad del subyacente",
                "Cambio en Delta por +1 unidad del subyacente",
                "Cambio en el precio si σ sube 1 punto porcentual",
                "Cambio en el precio por el paso de un año",
                "Cambio en el precio por el paso de un día (año de 365)",
                "Cambio en el precio si r sube 1 punto porcentual (100 pb)",
            ],
        })
        ui.show_table(gdf, {"Valor": "{:.6f}"})
    with right:
        ui.section("Paridad put-call")
        st.latex(r"c-p=S e^{-\delta T}-K e^{-rT}")
        p1, p2 = st.columns(2)
        p1.metric("c − p", f"{parity['c_menos_p']:.6f}")
        p2.metric("S e^(−δT) − K e^(−rT)", f"{parity['S_e^-δT_menos_K_e^-rT']:.6f}")
        st.caption(f"Diferencia: {parity['diferencia']:.2e}. Debe ser cero salvo error de máquina.")
        st.metric(f"Precio de la {'put' if opt == 'call' else 'call'} con los mismos inputs", ui.fmt_money(other.price))

with tab_sens:
    ui.section("Precio frente a cada parámetro (los demás fijos)")
    grid = {
        "Spot S": ("S", np.linspace(0.5 * S, 1.5 * S, 81), 1.0),
        "Volatilidad σ (%)": ("sigma", np.linspace(max(0.01, 0.2 * sigma), 2.5 * sigma, 81), 100.0),
        "Tasa r (%)": ("r", np.linspace(max(0.0, r - 0.05), r + 0.05, 81), 100.0),
        "Plazo T (años)": ("T", np.linspace(max(0.01, 0.05 * T), 3 * T, 81), 1.0),
    }
    base = dict(S=S, K=K, r=r, delta=q, sigma=sigma, T=T)
    cols = st.columns(2)
    for i, (name, (param, xs, scale)) in enumerate(grid.items()):
        fig = go.Figure()
        for o, color in (("call", ui.PALETTE["blue"]), ("put", ui.PALETTE["gold"])):
            ys = [bs.bs_price(option=o, **{**base, param: x}).price for x in xs]
            fig.add_trace(go.Scatter(x=xs * scale, y=ys, name=o, line=dict(color=color, width=2.5)))
        if param == "S":
            fig.add_trace(go.Scatter(x=xs, y=np.maximum(xs - K, 0) if opt == "call" else np.maximum(K - xs, 0),
                                     name=f"payoff {opt} al vencimiento", line=dict(color=ui.PALETTE["grey"], dash="dot")))
        fig.add_vline(x=base[param] * scale, line_dash="dash", line_color=ui.PALETTE["red"])
        fig.update_layout(title=f"Precio vs {name}", xaxis_title=name, yaxis_title="Precio", height=340)
        with cols[i % 2]:
            ui.show_chart(fig)

    ui.section("Mapa de calor: precio según spot y volatilidad")
    s_axis = np.linspace(0.7 * S, 1.3 * S, 25)
    v_axis = np.linspace(max(0.02, 0.4 * sigma), 2.0 * sigma, 25)
    z = [[bs.bs_price(s, K, r, q, v, T, opt).price for s in s_axis] for v in v_axis]
    heat = go.Figure(go.Heatmap(x=s_axis, y=v_axis * 100, z=z, colorscale="Blues", colorbar=dict(title="Precio")))
    heat.update_layout(xaxis_title="Spot S", yaxis_title="σ (%)", height=420, title=f"Precio de la {opt}")
    ui.show_chart(heat)

    ui.section("Choque rápido")
    c1, c2, c3 = st.columns(3)
    param_label = c1.selectbox("Parámetro", ["S", "K", "r", "delta", "sigma", "T"],
                               format_func=lambda p: {"S": "Spot", "K": "Strike", "r": "Tasa r", "delta": "δ / r_f",
                                                      "sigma": "Volatilidad", "T": "Plazo"}[p])
    kind = c2.radio("Tipo de choque", ["Relativo (%)", "Absoluto"], horizontal=True)
    size = c3.number_input("Tamaño del choque", value=5.0, step=1.0,
                           help="Relativo: +5 = el parámetro sube 5% de su valor. Absoluto en tasas: +1 = +1 punto porcentual (100 pb).")
    old = base[param_label]
    if kind == "Relativo (%)":
        new = old * (1 + size / 100)
    else:
        new = old + (size / 100 if param_label in ("r", "delta", "sigma") else size)
    try:
        shocked = bs.bs_price(option=opt, **{**base, param_label: new}).price
        sign, typ, why = rules.option_rule(param_label, opt, "european", q)
        m1, m2, m3 = st.columns(3)
        m1.metric("Precio base", ui.fmt_money(res.price))
        m2.metric("Precio con choque", ui.fmt_money(shocked), f"{shocked - res.price:+.4f}")
        m3.metric("Cambio relativo", f"{(shocked / res.price - 1) * 100:+.2f}%" if res.price else "n.d.")
        ui.note(f"<b>{typ}.</b> {why}<br>{rules.verdict(sign, res.price, shocked)}", output=True)
    except ValueError as err:
        st.error(str(err))

with tab_theory:
    st.markdown("""
**Qué calcula.** El precio de una opción europea como el valor presente, bajo la medida neutral al riesgo,
de su payoff: $c = e^{-rT}E^*[\\max(S_T-K,0)]$. Con $S_T$ lognormal se obtiene la fórmula cerrada.

**Supuestos (clase 16).** 1) El subyacente sigue un movimiento browniano geométrico. 2) Puede pagar
dividendos a tasa continua δ. 3) La tasa libre de riesgo es constante. 4) Sin comisiones ni impuestos.
5) Se permiten ventas en corto. 6) Se presta y se pide prestado a la misma tasa. 7) No hay arbitraje.

**Lectura de N(d₂) y N(d₁).** N(d₂) es la probabilidad neutral al riesgo de que el call termine dentro del
dinero. S e^(−δT) N(d₁) es el valor presente de recibir el subyacente sólo si se ejerce.

**Limitaciones.** Volatilidad constante (en el mercado hay sonrisa de volatilidad), sin saltos, tasa
constante y sólo ejercicio europeo. Para americanas se usa el árbol binomial.

**¿Cómo validamos?** Paridad put-call al error de máquina, greeks contra diferencias finitas, el límite
σ → 0 y los ejemplos de las clases 16 (3.40) y 17 (0.0107 exacto; 0.009 con tabla Z).

**Pregunta probable: ¿por qué tu resultado no coincide con el de clase?** Porque en clase N(·) se lee de la
tabla redondeando d a dos decimales. En el put de divisas el precio es la resta de dos términos casi iguales
(0.1940 − 0.1833), así que ese redondeo cambia el resultado alrededor de 16%. Activa "Usar tabla Z" para
reproducir el número de clase.
""")
