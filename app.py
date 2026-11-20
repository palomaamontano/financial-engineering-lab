"""Financial Engineering Lab | Quantitative Finance

Punto de entrada de Streamlit. Ejecutar con:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(
    page_title="Financial Engineering Lab",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = {
    "Inicio": [
        st.Page("views/home.py", title="Presentación", icon="🏠", default=True),
    ],
    "Calculadoras": [
        st.Page("views/fx_forward.py", title="1 · Curva de forwards", icon="💱"),
        st.Page("views/irs_curve.py", title="2 · Curva IRS", icon="📉"),
        st.Page("views/binomial.py", title="3 · Árbol binomial", icon="🌳"),
        st.Page("views/black_scholes.py", title="4 · Black-Scholes-Merton", icon="🧮"),
        st.Page("views/monte_carlo.py", title="5 · Monte Carlo", icon="🎲"),
    ],
    "Análisis": [
        st.Page("views/comparison.py", title="Model Comparison Lab", icon="⚖️"),
        st.Page("views/challenge.py", title="Professor Challenge | Modo Defensa", icon="🎓"),
    ],
}

st.sidebar.markdown("**Financial Engineering Lab**  \nIngeniería Financiera · ULSA")
nav = st.navigation(pages)
nav.run()
