import streamlit as st

from components import ui

ui.setup_page()
ui.header(
    "Financial Engineering Lab",
    "Plataforma de calculadoras de ingeniería financiera · Especialidad en Finanzas Corporativas y Bursátiles · ULSA",
    tags=("Python", "Streamlit", "Validado con pruebas automáticas"),
)

st.markdown("Cinco calculadoras, un comparador de modelos y un modo de defensa. Los inputs se capturan a la "
            "izquierda; los resultados se calculan y no se pueden editar.")

cols = st.columns(3)
cards = [
    ("💱 Curva de forwards", "Spot y puntos forward → forwards por plazo, X(t) y curva implícita en MXN (paridad cubierta)."),
    ("📉 Curva IRS", "Tasas forward → factores de descuento, tasas cero, precios forward y tasa swap par. Incluye bootstrap."),
    ("🌳 Árbol binomial", "Call y put, europeas y americanas. Cartera réplica Δ y B, p*, ejercicio anticipado y convergencia."),
    ("∑ Black-Scholes-Merton", "Precio, d₁, d₂, N(d₁), N(d₂), greeks con unidades, paridad put-call y modo divisas."),
    ("🎲 Monte Carlo", "GBM neutral al riesgo: trayectorias, error estándar, IC 95% y convergencia contra BSM."),
    ("⚖️ Comparador y 🎓 Modo Defensa", "Mismo instrumento con tres modelos; base vs choque con la explicación teórica."),
]
for i, (title, text) in enumerate(cards):
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(text)

ui.section("Notación")
st.markdown("""
| Símbolo | Significado | Convención |
| --- | --- | --- |
| S(t), X(t) | Precio del subyacente / tipo de cambio | MXN por USD en divisas |
| K | Strike | |
| r, δ | Tasa libre de riesgo y rendimiento por dividendos | Continuas, en % anual |
| σ | Volatilidad | Anual |
| T − t, h | Plazo y longitud del paso | Años; h = T/N |
| P(t,T) | Factor de descuento | Precio de 1 pagado en T |
| F_{t,T} | Precio forward | |
| p* | Probabilidad neutral al riesgo | (e^((r−δ)h) − d)/(u − d) |
""")

ui.section("Cómo sabemos que los números son correctos")
st.markdown("""
- **78 pruebas automáticas:** 54 del núcleo matemático y 24 que ejecutan cada página como lo haría un usuario.
- **Ejemplos de clase reproducidos:** curva MXN implícita de la Tarea 1 (X(t) = 17.3517, r MXN 1M = 6.795%),
  tasa IRS de la clase 8 (9.72%), bootstrap de la clase 11, árboles de la clase 14 (3.44 y 4.19),
  BSM de la clase 16 (3.40) y put de divisas de la clase 17.
- **Identidades exactas:** paridad put-call, P(0,T) ↔ tasa cero ↔ forward, tasa swap par = Σ fP / Σ P.
- **Propiedades:** americana ≥ europea, call americano sin dividendos = europeo, convergencia CRR → BSM,
  Monte Carlo dentro de 3 errores estándar y cobertura del IC 95%.
""")
st.caption("Diferencias con las notas detectadas al validar: tasa swap de la clase 7 (5.714% exacto vs 5.69% por "
           "truncar una forward), put de divisas de la clase 17 (0.0107 exacto vs 0.009 con tabla Z) y volatilidad "
           "histórica de la clase 17 (82.64% corresponde a varianza muestral con datos mensuales).")
