import datetime as dt

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import ui
from models import fx_forward as fx
from utils import rules

ui.setup_page()
ui.header(
    "Curva de forwards USD/MXN",
    "A partir del spot y los puntos forward: tipo de cambio forward por plazo, X(t) descontado y curva implícita en pesos.",
    tags=("Clases 5 y 7", "Tarea 1 · ej. 14", "Paridad cubierta"),
)

E = fx.EXERCISE_14


def _default_table():
    return pd.DataFrame([{
        "Plazo": r["tenor"], "Inicio T": r["start"], "Vencimiento T+s": r["end"],
        "Puntos forward": r["points"], "r USD % (continua)": r["r_usd"] * 100} for r in E["rows"]])


ui.init_state({"fx_t": E["trade_date"], "fx_spot_date": E["spot_date"], "fx_spot": E["spot_rate"],
               "fx_basis": 365, "fx_table": _default_table()})

with st.sidebar:
    st.subheader("Inputs")
    if st.button("Cargar Tarea 1 · ejercicio 14"):
        st.session_state.update({"fx_t": E["trade_date"], "fx_spot_date": E["spot_date"],
                                 "fx_spot": E["spot_rate"], "fx_basis": 365, "fx_table": _default_table()})
        st.session_state.pop("fx_editor", None)
    trade_date = st.date_input("Fecha de operación t", key="fx_t", format="DD/MM/YYYY")
    spot_date = st.date_input("Fecha valor spot (t+2)", key="fx_spot_date", format="DD/MM/YYYY")
    spot = st.number_input("Tipo de cambio spot F(t,t+2) MXN por USD", key="fx_spot",
                           min_value=0.0001, step=0.01, format="%.4f")
    basis = st.radio("Base de días", [365, 360], key="fx_basis", horizontal=True,
                     help="La Tarea 1 usa días/365 con tasas continuas.")

ui.section("Plazos de mercado (editable)")
st.caption("Puntos en pips: X(T,T+s) = (F(t,T+s) − F(t,T))·10,000. Puedes agregar o borrar filas. "
           "El O/N y el T/N conectan el spot con la fecha t para obtener X(t).")
edited = st.data_editor(
    st.session_state["fx_table"], key="fx_editor", num_rows="dynamic", hide_index=True,
    column_config={
        "Plazo": st.column_config.TextColumn("Plazo", required=True),
        "Inicio T": st.column_config.DateColumn("Inicio T", format="DD/MM/YYYY", required=True),
        "Vencimiento T+s": st.column_config.DateColumn("Vencimiento T+s", format="DD/MM/YYYY", required=True),
        "Puntos forward": st.column_config.NumberColumn("Puntos forward (pips)", format="%.2f", required=True),
        "r USD % (continua)": st.column_config.NumberColumn("r USD % (SOFR, continua)", format="%.4f", required=True),
    },
)


def rows_from(df):
    df = df.dropna(how="any")
    return [{"tenor": str(x["Plazo"]), "start": x["Inicio T"], "end": x["Vencimiento T+s"],
             "points": float(x["Puntos forward"]), "r_usd": float(x["r USD % (continua)"]) / 100} for _, x in df.iterrows()]


rows = rows_from(edited)
try:
    res = fx.build_fx_curve(trade_date, spot_date, spot, rows, basis=basis)
except ValueError as err:
    st.error(str(err))
    st.stop()
for w in res.warnings:
    st.warning(w)

tbl = res.table
tab_res, tab_sens, tab_teo, tab_theory = st.tabs(
    ["Resultado", "Sensibilidad", "Forward teórico y arbitraje", "Fundamento y defensa"])

with tab_res:
    c = st.columns(4)
    c[0].metric("X(t) · tipo de cambio descontado", f"{res.x_t:.4f}")
    c[1].metric("Spot F(t,t+2)", f"{spot:.4f}")
    c[2].metric(f"Forward {tbl['Plazo'].iloc[-1]}", f"{tbl['Forward F(t,T+s)'].iloc[-1]:.4f}")
    c[3].metric(f"r MXN implícita {tbl['Plazo'].iloc[-1]}", ui.fmt_pct(tbl["r MXN implícita (cont.)"].iloc[-1], 3))
    pct_cols = ["r USD (cont.)", "r MXN implícita (cont.)", "r MXN efectiva anual", "Diferencial r MXN − r USD"]
    fmts = {c: (lambda v: f"{100 * v:.4f}%") for c in pct_cols}
    fmts.update({"Forward F(t,T+s)": "{:.5f}", "τ (años)": "{:.6f}", "Puntos": "{:.2f}",
                 "P MXN (factor desc.)": "{:.6f}", "P USD (factor desc.)": "{:.6f}",
                 "Inicio T": lambda d: d.strftime("%d/%m/%Y"), "Vencimiento T+s": lambda d: d.strftime("%d/%m/%Y")})
    ui.show_table(tbl, fmts)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tbl["Días desde t"], y=tbl["Forward F(t,T+s)"], mode="lines+markers+text",
                                 text=tbl["Plazo"], textposition="top center", name="Forward",
                                 line=dict(color=ui.PALETTE["blue"], width=2.5)))
        fig.add_hline(y=res.x_t, line_dash="dot", line_color=ui.PALETTE["grey"], annotation_text="X(t)")
        fig.update_layout(title="Curva forward USD/MXN", xaxis_title="Días desde t", yaxis_title="MXN por USD", height=380)
        ui.show_chart(fig)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=tbl["Días desde t"], y=100 * tbl["r MXN implícita (cont.)"], name="r MXN implícita",
                                 mode="lines+markers", line=dict(color=ui.PALETTE["gold"], width=2.5)))
        fig.add_trace(go.Scatter(x=tbl["Días desde t"], y=100 * tbl["r USD (cont.)"], name="r USD (SOFR)",
                                 mode="lines+markers", line=dict(color=ui.PALETTE["blue"], width=2.5)))
        fig.update_layout(title="Curvas de tasas cero (continuas)", xaxis_title="Días desde t", yaxis_title="%", height=380)
        ui.show_chart(fig)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tbl["Días desde t"], y=tbl["P MXN (factor desc.)"], name="P MXN", mode="lines+markers",
                             line=dict(color=ui.PALETTE["gold"], width=2.5)))
    fig.add_trace(go.Scatter(x=tbl["Días desde t"], y=tbl["P USD (factor desc.)"], name="P USD", mode="lines+markers",
                             line=dict(color=ui.PALETTE["blue"], width=2.5)))
    fig.update_layout(title="Curvas de descuento P(t,T) = e^(−r·τ)", xaxis_title="Días desde t", yaxis_title="Factor", height=340)
    ui.show_chart(fig)

    ui.section("Cálculo paso a paso")
    i = st.selectbox("Plazo a explicar", list(range(len(tbl))), format_func=lambda k: tbl["Plazo"].iloc[k])
    row = tbl.iloc[i]
    st.latex(rf"X(t)={res.x_t:.4f}\quad(\text{{spot}}-\text{{puntos T/N}}/10{{,}}000-\text{{puntos O/N}}/10{{,}}000)")
    st.latex(rf"F(t,T+s)={row['Forward F(t,T+s)']:.5f},\quad \tau=\frac{{{row['Días desde t']}}}{{{basis}}}={row['τ (años)']:.6f}")
    st.latex(rf"r_{{MXN}}=r_{{USD}}+\frac{{\ln\left(F/X(t)\right)}}{{\tau}}={row['r USD (cont.)']:.5f}+"
             rf"\frac{{\ln({row['Forward F(t,T+s)']:.5f}/{res.x_t:.4f})}}{{{row['τ (años)']:.6f}}}={100 * row['r MXN implícita (cont.)']:.4f}\%")

with tab_sens:
    c1, c2, c3 = st.columns(3)
    d_spot = c1.number_input("Choque al spot (MXN)", value=0.10, step=0.05, format="%.4f")
    d_pts = c2.number_input("Choque a los puntos (pips, todos los plazos)", value=10.0, step=5.0)
    d_usd = c3.number_input("Choque a la curva USD (pb)", value=25.0, step=5.0)
    cases = {
        "Base": (spot, rows),
        f"Spot {d_spot:+.4f}": (spot + d_spot, rows),
        f"Puntos {d_pts:+.0f} pips": (spot, fx.shocked_rows(rows, points_shift=d_pts)),
        f"USD {d_usd:+.0f} pb": (spot, fx.shocked_rows(rows, usd_shift=d_usd / 10_000)),
    }
    out, fig = [], go.Figure()
    for k, (name, (s_val, rr)) in enumerate(cases.items()):
        t2 = fx.build_fx_curve(trade_date, spot_date, s_val, rr, basis=basis).table
        fig.add_trace(go.Scatter(x=t2["Días desde t"], y=100 * t2["r MXN implícita (cont.)"], name=name,
                                 mode="lines+markers", line=dict(color=ui.SERIES[k % 5], width=2.3)))
        out.append({"Escenario": name, "F último plazo": t2["Forward F(t,T+s)"].iloc[-1],
                    "r MXN último plazo": t2["r MXN implícita (cont.)"].iloc[-1],
                    "Cambio en r MXN (pb)": 10_000 * (t2["r MXN implícita (cont.)"].iloc[-1] - tbl["r MXN implícita (cont.)"].iloc[-1])})
    fig.update_layout(title="Curva MXN implícita bajo cada choque", xaxis_title="Días desde t", yaxis_title="%", height=400)
    ui.show_chart(fig)
    ui.show_table(pd.DataFrame(out), {"F último plazo": "{:.5f}", "r MXN último plazo": lambda v: f"{100 * v:.4f}%",
                                      "Cambio en r MXN (pb)": "{:+.2f}"})
    for p, label in (("spot", "Spot"), ("points", "Puntos forward"), ("r_usd", "Curva USD")):
        _, typ, why = rules.fx_rule(p)
        ui.note(f"<b>{label} · {typ}.</b> {why}")

    ui.section("Mapa de calor: r MXN del último plazo según spot y puntos")
    s_axis = np.linspace(spot - 0.5, spot + 0.5, 21)
    p_axis = np.linspace(-50, 50, 21)
    z = [[100 * fx.build_fx_curve(trade_date, spot_date, s, fx.shocked_rows(rows, points_shift=p), basis=basis)
          .table["r MXN implícita (cont.)"].iloc[-1] for s in s_axis] for p in p_axis]
    heat = go.Figure(go.Heatmap(x=s_axis, y=p_axis, z=z, colorscale="Blues", colorbar=dict(title="r MXN %")))
    heat.update_layout(xaxis_title="Spot", yaxis_title="Choque a los puntos (pips)", height=420)
    ui.show_chart(heat)

with tab_teo:
    st.markdown("Paridad cubierta continua (clase 5): $F_{t,T}(X)=X(t)\\,e^{(r_{MXN}-r_{USD})(T-t)}$.")
    c1, c2, c3, c4 = st.columns(4)
    x0 = c1.number_input("X(t)", value=17.0, step=0.1, format="%.4f")
    rd = c2.number_input("r MXN % (continua)", value=8.0, step=0.25, format="%.4f") / 100
    rf_ = c3.number_input("r USD % (continua)", value=3.0, step=0.25, format="%.4f") / 100
    tt = c4.number_input("T − t (años)", value=1.0, step=0.25, format="%.4f", min_value=0.0001)
    f_teo = fx.theoretical_fx_forward(x0, rd, rf_, tt)
    m_fwd = st.number_input("Forward observado en el mercado (opcional)", value=round(f_teo, 4), step=0.01, format="%.4f")
    a, b = st.columns(2)
    a.metric("Forward teórico", f"{f_teo:.4f}")
    b.metric("Mercado − teórico", f"{m_fwd - f_teo:+.4f}")
    if m_fwd > f_teo + 1e-9:
        ui.note("El forward de mercado está caro: <b>cash and carry</b>. Vender el forward, pedir prestados pesos, "
                f"comprar dólares e invertirlos en USD. Ganancia al vencimiento por dólar: {m_fwd - f_teo:.4f} MXN.", output=True)
    elif m_fwd < f_teo - 1e-9:
        ui.note("El forward de mercado está barato: <b>reverse cash and carry</b>. Comprar el forward, pedir prestados "
                f"dólares, venderlos e invertir los pesos. Ganancia al vencimiento por dólar: {f_teo - m_fwd:.4f} MXN.", output=True)
    else:
        ui.note("El forward de mercado coincide con el teórico: no hay arbitraje.", output=True)

with tab_theory:
    st.markdown("""
**Qué calcula.** Con el spot y los puntos forward se obtiene el tipo de cambio forward de cada plazo. Con la
curva USD (SOFR) y la paridad cubierta se despeja la tasa implícita en pesos: es la tasa que hace que no haya
arbitraje entre invertir en pesos o convertir a dólares, invertir y cubrir con el forward.

**Por qué se necesita la curva USD.** La paridad cubierta sólo identifica el diferencial r_MXN − r_USD.
Spot y puntos solos no determinan ninguna de las dos curvas: hace falta una de referencia.

**Convenciones (Tarea 1, ej. 14).** Cotización MXN por USD; puntos en pips (÷10,000); spot con valor t+2;
tasas continuas; plazo en días/365 desde la fecha de operación t.

**X(t).** El spot liquida en t+2. Restando los puntos T/N y O/N se lleva a la fecha de operación:
X(t) = 17.3545 − 0.0014 − 0.0014 = 17.3517.

**Sensibilidad.** Con puntos fijos, mover la curva USD mueve la curva MXN uno a uno y no cambia los forwards.
Más puntos implican mayor diferencial y mayor tasa MXN. Subir el spot con puntos fijos baja ligeramente la tasa
implícita, porque los mismos pips representan un porcentaje menor de un tipo de cambio más alto.

**Limitaciones.** Supone que la paridad cubierta se cumple (sin base cross-currency ni costos de transacción)
y que la curva USD es la correcta para descontar. En la práctica existe una base que separa ambas.

**¿Cómo validamos?** Reproduce los cinco forwards y las cinco tasas de la Tarea 1; el forward teórico
recalculado con la tasa implícita reproduce el forward de mercado al error de máquina.
""")
