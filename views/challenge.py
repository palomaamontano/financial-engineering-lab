import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from models import binomial as bn
from models import black_scholes as bs
from models import fx_forward as fx
from models import monte_carlo as mc
from models import rates_curve as rc
from utils import rules

ui.setup_page()
ui.header(
    "Professor Challenge | Modo Defensa",
    "Escenario base, cambio de un parámetro, resultado inmediato y la explicación teórica de lo que debía pasar.",
    tags=("Sensibilidad", "Reglas verificadas", "Base vs choque"),
)

CALCS = ["Black-Scholes-Merton", "Árbol binomial (CRR)", "Monte Carlo", "Curva de forwards", "Curva IRS"]
calc = st.radio("Calculadora", CALCS, horizontal=True)

ASSUMPTIONS = {
    "Black-Scholes-Merton": "GBM con σ y r constantes, dividendos continuos δ, sin costos ni impuestos, ventas en corto "
                            "permitidas, misma tasa para prestar y pedir prestado, sin arbitraje, ejercicio europeo.",
    "Árbol binomial (CRR)": "En cada paso el subyacente sólo sube (u) o baja (d); d < e^((r−δ)h) < u; tasa constante; "
                            "valoración por cartera réplica; ejercicio europeo o americano.",
    "Monte Carlo": "Mismo modelo que BSM (GBM neutral al riesgo); el precio es un estimador con error estadístico "
                   "que cae como 1/√n; semilla fija para reproducibilidad.",
    "Curva de forwards": "Paridad cubierta continua, puntos en pips (÷10,000), spot t+2, días/365, curva USD (SOFR) "
                         "como referencia; sin base cross-currency ni costos.",
    "Curva IRS": "Forwards simples por periodo anual, P(0,T) = Π 1/(1 + f), tasa swap par = (1 − Pₙ)/Σ P.",
}
PARAM_LABEL = {"S": "Spot S", "K": "Strike K", "r": "Tasa r", "delta": "Dividendos δ", "sigma": "Volatilidad σ",
               "T": "Plazo T", "N": "Pasos N", "n_sims": "Simulaciones n"}
RATE_PARAMS = ("r", "delta", "sigma")


def shock_value(old, param):
    c1, c2 = st.columns(2)
    if param in ("N", "n_sims"):
        new = c1.number_input("Nuevo valor", value=int(old * 4), min_value=1, step=1)
        return float(new)
    kind = c1.radio("Tipo de choque", ["Absoluto", "Relativo (%)"], horizontal=True,
                    help="Absoluto en tasas y σ: +1 = +1 punto porcentual (100 pb). Relativo: +5 = el valor sube 5%.")
    size = c2.number_input("Tamaño", value=1.0 if kind == "Absoluto" else 5.0, step=0.5)
    if kind == "Relativo (%)":
        return old * (1 + size / 100)
    return old + (size / 100 if param in RATE_PARAMS else size)


def show_result(label, base, new, sign, typ, why, fmt="{:.5f}", note_extra=""):
    m = st.columns(4)
    m[0].metric(f"{label} · base", fmt.format(base))
    m[1].metric(f"{label} · con choque", fmt.format(new))
    m[2].metric("Cambio absoluto", ("{:+" + fmt[2:]).format(new - base))
    m[3].metric("Cambio relativo", f"{(new / base - 1) * 100:+.3f}%" if base else "n.d.")
    ui.section("¿Qué debería suceder y por qué?")
    ui.note(f"<span class='fel-tag'>{typ}</span><br>{why}<br><br><b>{rules.verdict(sign, base, new)}</b>{note_extra}",
            output=True)


if calc in CALCS[:3]:
    st.markdown("#### Escenario base")
    c = st.columns(7)
    S = c[0].number_input("S", value=100.0, min_value=0.0001)
    K = c[1].number_input("K", value=100.0, min_value=0.0001)
    r = c[2].number_input("r %", value=5.0) / 100
    q = c[3].number_input("δ %", value=0.0) / 100
    sigma = c[4].number_input("σ %", value=20.0, min_value=0.01) / 100
    T = c[5].number_input("T", value=1.0, min_value=0.0001)
    opt = c[6].radio("Tipo", ["call", "put"])
    base = dict(S=S, K=K, r=r, delta=q, sigma=sigma, T=T)
    style = "european"
    extra = []
    if calc == CALCS[1]:
        e1, e2 = st.columns(2)
        N = int(e1.number_input("Pasos N", value=50, min_value=1, max_value=1000))
        style = "american" if e2.radio("Ejercicio", ["Europea", "Americana"], horizontal=True) == "Americana" else "european"
        base["N"] = N
        extra = ["N"]
    if calc == CALCS[2]:
        e1, e2 = st.columns(2)
        n0 = int(e1.select_slider("Simulaciones n", [1_000, 10_000, 50_000, 100_000], value=10_000))
        seed = int(e2.number_input("Semilla", value=42, min_value=0))
        base["n_sims"] = n0
        extra = ["n_sims"]

    def price(p):
        if calc == CALCS[0]:
            return bs.bs_price(p["S"], p["K"], p["r"], p["delta"], p["sigma"], p["T"], opt).price, None
        if calc == CALCS[1]:
            return bn.price_binomial(p["S"], p["K"], p["r"], p["delta"], p["T"], int(p["N"]), opt, style, "crr",
                                     p["sigma"], store=False).price, None
        out = mc.mc_european(p["S"], p["K"], p["r"], p["delta"], p["sigma"], p["T"], int(p["n_sims"]), seed, opt)
        return out["precio"], out["error_estandar"]

    st.markdown("#### Cambio de parámetro")
    param = st.selectbox("¿Qué cambia el profesor?", ["S", "K", "r", "delta", "sigma", "T"] + extra,
                         format_func=lambda p: PARAM_LABEL[p])
    new_val = shock_value(base[param], param)
    shocked = {**base, param: new_val}
    try:
        p0, se0 = price(base)
        p1, se1 = price(shocked)
    except ValueError as err:
        st.error(str(err))
        st.stop()
    shown = (lambda v: f"{100 * v:.4f}%") if param in RATE_PARAMS else (lambda v: f"{v:,.4f}")
    st.caption(f"{PARAM_LABEL[param]}: {shown(base[param])} → {shown(new_val)}")
    sign, typ, why = rules.option_rule(param, opt, style, q)
    extra_note = ""
    if se0 is not None:
        extra_note = (f"<br>Error estándar base {se0:.5f} y con choque {se1:.5f}. Con la misma semilla las dos "
                      "corridas usan los mismos números aleatorios, así que la dirección del cambio es confiable; "
                      "un cambio menor a 2 errores estándar no es distinguible del ruido.")
        if param != "n_sims":
            typ = f"{typ} · estimado con simulación"
    show_result(f"Precio {opt}", p0, p1, sign, typ, why, note_extra=extra_note)
    if calc == CALCS[0]:
        g = bs.greeks(option=opt, **base)
        approx = {"S": g["Delta"] * (new_val - base["S"]), "sigma": g["Vega (por +1% σ)"] * 100 * (new_val - base["sigma"]),
                  "r": g["Rho (por +1% r)"] * 100 * (new_val - base["r"]), "T": -g["Theta (por año)"] * (new_val - base["T"])}
        if param in approx:
            st.caption(f"Aproximación de primer orden con el greek: {approx[param]:+.5f} (el cambio exacto es {p1 - p0:+.5f}; "
                       "la diferencia es el efecto de segundo orden, por ejemplo Gamma).")

elif calc == "Curva de forwards":
    e = fx.EXERCISE_14
    rows = e["rows"]
    st.markdown("#### Escenario base: Tarea 1, ejercicio 14")
    tenor = st.selectbox("Plazo a observar", [r["tenor"] for r in rows], index=len(rows) - 1)
    idx = [r["tenor"] for r in rows].index(tenor)
    param = st.selectbox("¿Qué cambia el profesor?", ["spot", "points", "r_usd"],
                         format_func={"spot": "Spot (MXN)", "points": "Puntos forward (pips, todos los plazos)",
                                      "r_usd": "Curva USD (pb, paralelo)"}.get)
    size = st.number_input("Tamaño del choque", value={"spot": 0.10, "points": 10.0, "r_usd": 25.0}[param], step=0.05)
    spot0, spot1, rows1 = e["spot_rate"], e["spot_rate"], rows
    if param == "spot":
        spot1 = spot0 + size
    elif param == "points":
        rows1 = fx.shocked_rows(rows, points_shift=size)
    else:
        rows1 = fx.shocked_rows(rows, usd_shift=size / 10_000)
    b = fx.build_fx_curve(e["trade_date"], e["spot_date"], spot0, rows).table.iloc[idx]
    s = fx.build_fx_curve(e["trade_date"], e["spot_date"], spot1, rows1).table.iloc[idx]
    sign, typ, why = rules.fx_rule(param)
    st.metric(f"Forward {tenor}", f"{s['Forward F(t,T+s)']:.5f}", f"{s['Forward F(t,T+s)'] - b['Forward F(t,T+s)']:+.5f}")
    show_result(f"r MXN implícita {tenor} (%)", 100 * b["r MXN implícita (cont.)"], 100 * s["r MXN implícita (cont.)"],
                sign, typ, why, fmt="{:.4f}")

else:
    st.markdown("#### Escenario base: forwards anuales editables")
    base_df = st.data_editor(pd.DataFrame({"Forward %": [5.0, 5.5, 6.0, 6.5, 7.0]}), num_rows="dynamic")
    fwd = base_df["Forward %"].dropna().astype(float).to_numpy() / 100
    shift = st.number_input("Choque paralelo a las forwards (pb)", value=50.0, step=5.0)
    try:
        t0 = rc.curve_table(rc.discount_from_forwards(fwd))
        t1 = rc.curve_table(rc.discount_from_forwards(fwd + shift / 10_000))
    except ValueError as err:
        st.error(str(err))
        st.stop()
    sign, typ, why = rules.irs_rule("forwards")
    st.metric(f"P(0,{len(fwd)})", f"{t1['P(0,T)'].iloc[-1]:.6f}", f"{t1['P(0,T)'].iloc[-1] - t0['P(0,T)'].iloc[-1]:+.6f}")
    show_result(f"Tasa swap par {len(fwd)} años (%)", 100 * t0["Tasa swap par"].iloc[-1], 100 * t1["Tasa swap par"].iloc[-1],
                sign, typ, why, fmt="{:.4f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t0["T (años)"], y=100 * t0["Tasa swap par"], name="Base", line=dict(color=ui.PALETTE["blue"], width=2.5)))
    fig.add_trace(go.Scatter(x=t1["T (años)"], y=100 * t1["Tasa swap par"], name="Con choque", line=dict(color=ui.PALETTE["red"], width=2.5, dash="dash")))
    fig.update_layout(title="Tasa swap par por plazo", xaxis_title="T (años)", yaxis_title="%", height=360)
    ui.show_chart(fig)

with st.expander("Supuestos del modelo"):
    st.write(ASSUMPTIONS[calc])
st.caption("Tipos de resultado: Determinista = se deduce del signo de una derivada de la fórmula. Depende de los "
           "supuestos = la dirección cambia según el caso. Estadístico = sujeto a error de muestreo. Numérico = "
           "efecto de la discretización.")
