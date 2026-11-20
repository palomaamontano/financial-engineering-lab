import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from models import rates_curve as rc
from utils import rules

ui.setup_page()
ui.header(
    "Curva IRS",
    "Factores de descuento, tasas cero, tasas forward y tasas swap par a partir de tasas forward (input del profesor).",
    tags=("Clases 7, 8 y 11", "Forward rates → P(0,T)", "Bootstrap"),
)

MODES = ["Tasas forward (input del profesor)", "Factores de descuento P(0,T)", "Tasas swap (bootstrap)"]
FREQ = {"Anual (τ = 1)": 1.0, "Semestral (τ = 0.5)": 0.5, "Trimestral (τ = 0.25)": 0.25, "Mensual (τ = 1/12)": 1 / 12}

CLASS7_FWD = [100 * (1 / 0.95 - 1), 100 * (0.95 / 0.90 - 1), 100 * (0.90 / 0.85 - 1), 100 * (0.85 / 0.80 - 1)]
SCENARIOS = {
    "Personalizado": None,
    "Clase 7 · forwards implícitos en P = 0.95, 0.90, 0.85, 0.80": {
        "irs_mode": MODES[0], "irs_freq": "Anual (τ = 1)", "irs_comp": "Discreta (simple por periodo)",
        "irs_data": pd.DataFrame({"Valor": CLASS7_FWD})},
    "Clase 8 · P = 0.9237, 0.8545, 0.7764, 0.6851": {
        "irs_mode": MODES[1], "irs_freq": "Anual (τ = 1)",
        "irs_data": pd.DataFrame({"Valor": [0.9237, 0.8545, 0.7764, 0.6851]})},
    "Clase 11 · bootstrap con swaps 9%, 10%, 11%": {
        "irs_mode": MODES[2], "irs_freq": "Anual (τ = 1)", "irs_data": pd.DataFrame({"Valor": [9.0, 10.0, 11.0]})},
}
ui.init_state({"irs_mode": MODES[0], "irs_freq": "Anual (τ = 1)", "irs_comp": "Discreta (simple por periodo)",
               "irs_data": pd.DataFrame({"Valor": [5.0, 5.5, 6.0, 6.5, 7.0]}), "irs_K": 6.0, "irs_notional": 100.0})


def _load_scn():
    choice = st.session_state["irs_scn"]
    if SCENARIOS.get(choice):
        st.session_state.update(SCENARIOS[choice])
        st.session_state.pop("irs_editor", None)


with st.sidebar:
    st.subheader("Inputs")
    st.selectbox("Escenario de clase", list(SCENARIOS), key="irs_scn", on_change=_load_scn)
    mode = st.radio("Dato de entrada", MODES, key="irs_mode")
    tau = FREQ[st.selectbox("Frecuencia de los periodos", list(FREQ), key="irs_freq")]
    comp = st.radio("Capitalización de las forwards", ["Discreta (simple por periodo)", "Continua"], key="irs_comp",
                    disabled=mode != MODES[0])

unit = "%" if mode != MODES[1] else "factor"
ui.section("Datos por periodo (editable)")
st.caption({MODES[0]: "Tasa forward r_f(T_{i−1}, T_i) de cada periodo, anualizada, en %.",
            MODES[1]: "Factor de descuento P(0,T_i) de cada vencimiento.",
            MODES[2]: "Tasa swap par de cada plazo, en %."}[mode])
edited = st.data_editor(st.session_state["irs_data"], key="irs_editor", num_rows="dynamic", hide_index=False,
                        column_config={"Valor": st.column_config.NumberColumn(
                            "Valor (%)" if unit == "%" else "P(0,T)", format="%.6f", required=True)})
values = edited["Valor"].dropna().astype(float).to_numpy()

try:
    if mode == MODES[0]:
        P = rc.discount_from_forwards(values / 100, tau, "discreta" if comp.startswith("Discreta") else "continua")
    elif mode == MODES[1]:
        P = values
    else:
        P = rc.bootstrap_from_swaps(values / 100, tau)
    table = rc.curve_table(P, tau)
except ValueError as err:
    st.error(str(err))
    st.stop()

tab_res, tab_swap, tab_sens, tab_theory = st.tabs(
    ["Resultado", "Valor de un swap", "Sensibilidad", "Fundamento y defensa"])
pct_cols = ["Forward r_f(T_{i-1},T_i)", "Forward continua", "Tasa cero anual", "Tasa cero continua",
            "Tasa swap par", "Tasa swap (Σ fP / Σ P)"]
fmts = {c: (lambda v: f"{100 * v:.4f}%") for c in pct_cols}
fmts.update({"T (años)": "{:.4f}", "Precio forward del bono": "{:.6f}", "P(0,T)": "{:.6f}", "Anualidad Σ τP": "{:.6f}"})

with tab_res:
    c = st.columns(4)
    c[0].metric(f"P(0,{table['T (años)'].iloc[-1]:g})", f"{table['P(0,T)'].iloc[-1]:.6f}")
    c[1].metric("Tasa cero anual (último plazo)", ui.fmt_pct(table["Tasa cero anual"].iloc[-1]))
    c[2].metric("Tasa cero continua (último plazo)", ui.fmt_pct(table["Tasa cero continua"].iloc[-1]))
    c[3].metric("Tasa swap par (último plazo)", ui.fmt_pct(table["Tasa swap par"].iloc[-1]))
    ui.show_table(table, fmts)
    st.caption("Control: la tasa swap par y Σ r_f·P / Σ P deben coincidir en cada plazo (son la misma identidad).")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        t = table["T (años)"]
        for col, name, color in (("Tasa cero anual", "Cero (anual)", ui.PALETTE["blue"]),
                                 ("Forward r_f(T_{i-1},T_i)", "Forward por periodo", ui.PALETTE["gold"]),
                                 ("Tasa swap par", "Swap par", ui.PALETTE["teal"])):
            fig.add_trace(go.Scatter(x=t, y=100 * table[col], name=name, mode="lines+markers",
                                     line=dict(color=color, width=2.5, shape="hv" if "Forward" in name else "linear")))
        fig.update_layout(title="Curvas de tasas", xaxis_title="T (años)", yaxis_title="%", height=390)
        ui.show_chart(fig)
    with right:
        fig = go.Figure(go.Scatter(x=np.r_[0, t], y=np.r_[1, table["P(0,T)"]], mode="lines+markers",
                                   line=dict(color=ui.PALETTE["navy"], width=2.5), name="P(0,T)"))
        fig.update_layout(title="Curva de descuento", xaxis_title="T (años)", yaxis_title="P(0,T)", height=390)
        ui.show_chart(fig)

    ui.section("Fórmulas")
    st.latex(r"P(0,T_n)=\prod_{i=1}^{n}\frac{1}{1+r_f(T_{i-1},T_i)\,\tau}\qquad r(0,T)=P(0,T)^{-1/T}-1"
             r"\qquad r^{c}(0,T)=-\frac{\ln P(0,T)}{T}")
    st.latex(r"r_f(T_{i-1},T_i)=\frac{1}{\tau}\left(\frac{P(0,T_{i-1})}{P(0,T_i)}-1\right)\qquad "
             r"S_n=\frac{\sum_{i=1}^{n} r_f(T_{i-1},T_i)\,\tau\,P(0,T_i)}{\sum_{i=1}^{n}\tau P(0,T_i)}=\frac{1-P(0,T_n)}{\sum_{i=1}^{n}\tau P(0,T_i)}")
    n = len(table)
    st.latex(rf"S_{{{n}}}=\frac{{1-{table['P(0,T)'].iloc[-1]:.6f}}}{{{table['Anualidad Σ τP'].iloc[-1]:.6f}}}"
             rf"={100 * table['Tasa swap par'].iloc[-1]:.4f}\%")

with tab_swap:
    st.markdown("Valor de un IRS existente con tasa fija K (clase 8): **pata flotante − pata fija** para quien paga fijo.")
    c1, c2, c3 = st.columns(3)
    K = c1.number_input("Tasa fija K (%)", key="irs_K", step=0.25, format="%.4f") / 100
    notional = c2.number_input("Nocional", key="irs_notional", min_value=0.0, step=100.0)
    side = c3.radio("Posición", ["Paga fija (payer)", "Recibe fija (receiver)"], horizontal=True)
    mtm = rc.swap_mark_to_market(P, K, notional, tau, payer=side.startswith("Paga"))
    m = st.columns(3)
    m[0].metric("Pata flotante N(1 − P(0,Tₙ))", f"{mtm['pata_flotante']:,.4f}")
    m[1].metric("Pata fija N·K·Σ τP", f"{mtm['pata_fija']:,.4f}")
    m[2].metric("Valor del swap", f"{mtm['valor']:,.4f}")
    ui.note(f"A la tasa par {100 * table['Tasa swap par'].iloc[-1]:.4f}% el swap vale cero. Si K es menor que la tasa "
            "par, quien paga fijo gana: paga menos de lo que hoy vale la pata flotante.", output=True)

with tab_sens:
    shift_bp = st.slider("Choque paralelo a las tasas forward (pb)", -200, 200, 50, step=5)
    fwd_base = table["Forward r_f(T_{i-1},T_i)"].to_numpy()
    P_up = rc.discount_from_forwards(fwd_base + shift_bp / 10_000, tau)
    t_up = rc.curve_table(P_up, tau)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=table["T (años)"], y=100 * table["Tasa swap par"], name="Swap par base",
                             line=dict(color=ui.PALETTE["teal"], width=2.5)))
    fig.add_trace(go.Scatter(x=t_up["T (años)"], y=100 * t_up["Tasa swap par"], name=f"Swap par {shift_bp:+d} pb",
                             line=dict(color=ui.PALETTE["red"], width=2.5, dash="dash")))
    fig.add_trace(go.Scatter(x=table["T (años)"], y=100 * table["Tasa cero anual"], name="Cero base",
                             line=dict(color=ui.PALETTE["blue"], width=2)))
    fig.add_trace(go.Scatter(x=t_up["T (años)"], y=100 * t_up["Tasa cero anual"], name=f"Cero {shift_bp:+d} pb",
                             line=dict(color=ui.PALETTE["gold"], width=2, dash="dash")))
    fig.update_layout(title="Efecto de un choque paralelo", xaxis_title="T (años)", yaxis_title="%", height=400)
    ui.show_chart(fig)
    comp_df = pd.DataFrame({"T (años)": table["T (años)"], "P base": table["P(0,T)"], "P con choque": t_up["P(0,T)"],
                            "Δ swap par (pb)": 10_000 * (t_up["Tasa swap par"] - table["Tasa swap par"]),
                            "Δ tasa cero (pb)": 10_000 * (t_up["Tasa cero anual"] - table["Tasa cero anual"])})
    ui.show_table(comp_df, {"T (años)": "{:.4f}", "P base": "{:.6f}", "P con choque": "{:.6f}",
                            "Δ swap par (pb)": "{:+.2f}", "Δ tasa cero (pb)": "{:+.2f}"})
    dv01 = rc.swap_mark_to_market(rc.discount_from_forwards(fwd_base + 1e-4, tau), K, notional, tau)["valor"] - \
        rc.swap_mark_to_market(P, K, notional, tau)["valor"]
    st.metric("DV01 del swap payer (cambio de valor por +1 pb en las forwards)", f"{dv01:,.6f}")
    _, typ, why = rules.irs_rule("forwards")
    ui.note(f"<b>{typ}.</b> {why}", output=True)

with tab_theory:
    st.markdown("""
**Qué calcula.** Con las tasas forward de cada periodo se encadenan los factores de descuento; de ellos salen
las tasas cero (anual y continua), los precios forward de bonos y la tasa swap par de cada plazo.

**Forward rates, zero rates, discount factors y swap rates no son lo mismo.**
- La *forward* r_f(T_{i−1},T_i) es la tasa pactada hoy para un periodo futuro.
- La *tasa cero* r(0,T) es el rendimiento de un bono cupón cero de hoy a T: un promedio geométrico de las forwards.
- El *factor de descuento* P(0,T) es el precio hoy de 1 peso pagado en T.
- La *tasa swap par* es la tasa fija que iguala el valor presente de la pata fija y la flotante: un promedio de
  las forwards ponderado por P(0,T_i).

**Dos metodologías distintas.** Con forwards como input (lo que pidió el profesor) la curva sale directo.
El *bootstrap* (clase 11) es el camino inverso: parte de tasas swap cotizadas y despeja P(0,T_n) plazo por plazo
con P_n = (1 − S_n Σ_{i<n} τP_i)/(1 + S_n τ). Ambos modos están disponibles y son consistentes entre sí.

**Convenciones.** Periodos anuales (τ = 1) como en clase; las forwards son simples por periodo; la tasa cero se
reporta con capitalización anual y continua.

**Notas de clase revisadas.** En la clase 7 la tasa swap a 4 años es exactamente (1 − 0.80)/3.5 = 5.714%;
las notas dan 5.69% porque truncan la primera forward (5.263%) a 5.2%.

**¿Cómo validamos?** Clase 8 (9.72%), clase 11 (ceros 9%, 10.05%, 11.15%), clase 7 (ceros y precios forward),
identidad Σ fP/ΣP = (1 − Pₙ)/ΣP, y el bootstrap recupera exactamente la curva que generó las tasas swap.
""")
