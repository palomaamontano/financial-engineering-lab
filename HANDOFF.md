# Traspaso: Financial Engineering Lab

Estado al 19 de noviembre de 2026, noche (hora de CDMX). Evaluación presencial: sábado 21 de noviembre.

## Ya está hecho
- Núcleo matemático en `models/` y reglas en `utils/rules.py`.
- 78 pruebas en `tests/` que pasan en Python 3.12 y 3.13: 54 del núcleo y 24 de interfaz (`tests/test_pages.py`,
  con `streamlit.testing.v1.AppTest`: cada página, la navegación, los escenarios de clase en pantalla,
  el aviso de arbitraje del árbol y las cinco calculadoras del Modo Defensa).
- Además, fuera de las pruebas, se recorrieron con AppTest todas las combinaciones de escenarios, modos,
  pasos, tipos de opción, choques (incluidos choques inválidos) y plazos del Modo Defensa: cero excepciones.
- Servidor real (`streamlit run app.py --server.headless true`): `/_stcore/health` responde `ok`; las 8 páginas
  se abrieron en Chrome headless a 1440, 1280 y 1024 px sin excepciones de Streamlit ni errores de la app.
- Edición de la tabla de forwards (fechas, puntos, filas nuevas, filas borradas) simulada con la lógica real
  de `st.data_editor`: funciona; borrar el O/N muestra un mensaje claro.
- Versiones fijadas en `requirements.txt` (Streamlit 1.67.0, Plotly 7.1.0, NumPy 2.5.5, pandas 3.1.1, SciPy 1.18.1).

## Correcciones del 19/11
1. `app.py`: el ícono "∑" de la página BSM no es un emoji válido y Streamlit 1.67 detenía la app completa al
   arrancar. Se cambió por 🧮.
2. `components/ui.py`: el CSS de las tarjetas apuntaba a `div[data-testid="stMetricLabel"]`, pero en
   Streamlit 1.67 la etiqueta es `<label>`; las etiquetas largas se cortaban con "…". Ahora se ajustan en dos
   líneas y la cifra baja de tamaño en pantallas menores a 1200 px.
3. `views/binomial.py`: seis tarjetas en una fila cortaban u, d y p*; ahora son dos filas (5 + 4).
4. `views/monte_carlo.py`: el IC 95% no cabía; ahora son dos filas de tres, con el ancho del IC.
5. `components/ui.py` (`show_chart`): la leyenda de Plotly se encimaba con el título en "60 trayectorias
   simuladas" y "Curvas de tasas". El título ahora se escribe arriba de la figura, fuera de Plotly.

## Lo que falta
1. Subir a GitHub y desplegar en Streamlit Community Cloud (Python 3.12 en Advanced settings).
2. Abrir la app publicada el sábado antes de la evaluación (las apps gratuitas se duermen sin tráfico).
3. Guía de defensa y simulacro de preguntas del profesor.

## Diferencias con las notas detectadas al validar
- Clase 7, tasa IRS a 4 años: 5.714% exacto; las notas dan 5.69% por truncar la primera forward a 5.2%.
- Clase 7, swap de acciones: 484/4.25 = 113.88; las notas suman 492 y dan 115.76.
- Clase 17, put de divisas: 0.0107 con N exacta; 0.009 con tabla Z (la app tiene el modo tabla Z).
- Clase 17, volatilidad histórica: 82.64% sólo con varianza muestral (n − 1) y h = 1/12.
- Clase 14, B = −46.559 (las notas truncan a −46.55).
