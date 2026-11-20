# Financial Engineering Lab | Quantitative Finance

Plataforma de calculadoras de Ingeniería Financiera (Especialidad en Finanzas Corporativas y Bursátiles, ULSA), hecha en Python con Streamlit.

## Módulos

| Página | Qué hace | Base en clase |
| --- | --- | --- |
| 1 · Curva de forwards | Spot y puntos forward → forwards por plazo, X(t) y curva implícita MXN | Clases 5 y 7, Tarea 1 ej. 14 |
| 2 · Curva IRS | Tasas forward → P(0,T), tasas cero, precios forward, tasa swap par; bootstrap | Clases 7, 8 y 11 |
| 3 · Árbol binomial | Call y put, europeas y americanas; Δ, B, p*, ejercicio anticipado | Clases 13 y 14 |
| 4 · Black-Scholes-Merton | Precio, d₁, d₂, N(·), greeks, paridad, modo divisas, tabla Z | Clases 16 y 17 |
| 5 · Monte Carlo | GBM neutral al riesgo, error estándar, IC 95%, convergencia | Clase 16 |
| Model Comparison Lab | BSM vs CRR vs Monte Carlo con los mismos inputs | |
| Professor Challenge | Base vs choque y la explicación teórica verificada | |

## Estructura

```
app.py              navegación
views/              una página por calculadora
models/             fórmulas (sin Streamlit; se prueban solas)
components/ui.py    estilo, tarjetas y gráficas
utils/rules.py      reglas de "¿qué debería suceder y por qué?"
tests/              78 pruebas con pytest (54 del núcleo, 24 de interfaz con AppTest)
```

## Ejecutar las pruebas

```
pip install -r requirements.txt pytest
python -m pytest -q
```

## Ejecutar la app

```
streamlit run app.py
```

## Publicar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub.
2. En share.streamlit.io: **Create app → Deploy a public app from GitHub**.
3. Repositorio: el tuyo. Branch: `main`. Main file path: `app.py`.
4. En **Advanced settings** elige Python 3.12.
5. Deploy. La app queda en una dirección `https://<nombre>.streamlit.app`.
