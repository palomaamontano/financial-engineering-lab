"""Pruebas de interfaz: cada página corre en Streamlit sin excepciones y muestra los números de clase."""
import pathlib

import pytest
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGES = ["home", "fx_forward", "irs_curve", "binomial", "black_scholes", "monte_carlo", "comparison", "challenge"]


def run(page):
    at = AppTest.from_file(str(ROOT / "views" / f"{page}.py"), default_timeout=180)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def metric(at, label):
    return next(m.value for m in at.metric if m.label == label)


@pytest.mark.parametrize("page", PAGES)
def test_page_runs(page):
    run(page)


def test_app_navigation():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=180)
    at.run()
    assert not at.exception
    for page in PAGES:
        at.switch_page(f"views/{page}.py").run()
        assert not at.exception, page


def test_fx_tarea1_on_screen():
    at = run("fx_forward")
    assert metric(at, "X(t) · tipo de cambio descontado") == "17.3517"
    assert metric(at, "r MXN implícita 1M") == "6.795%"


@pytest.mark.parametrize("scenario, expected", [
    ("Clase 7", "5.7143%"), ("Clase 8", "9.7200%"), ("Clase 11", "11.0000%")])
def test_irs_scenarios_on_screen(scenario, expected):
    at = run("irs_curve")
    sel = at.selectbox(key="irs_scn")
    sel.set_value(next(o for o in sel.options if o.startswith(scenario))).run()
    assert not at.exception
    assert metric(at, "Tasa swap par (último plazo)") == expected


@pytest.mark.parametrize("scenario, label, expected", [
    ("Clase 14 · put", "Precio put europea", "4.1927"),
    ("Clase 14 · call", "Precio call europea", "3.4406")])
def test_binomial_scenarios_on_screen(scenario, label, expected):
    at = run("binomial")
    sel = at.selectbox(key="bn_scn")
    sel.set_value(next(o for o in sel.options if o.startswith(scenario))).run()
    assert metric(at, label) == expected


@pytest.mark.parametrize("scenario, table, label, expected", [
    ("Clase 16", False, "Precio call", "3.3991"),
    ("Clase 17", False, "Precio put", "0.0107"),
    ("Clase 17", True, "Precio put", "0.0090")])
def test_bsm_scenarios_on_screen(scenario, table, label, expected):
    at = run("black_scholes")
    sel = at.selectbox(key="bs_scn")
    sel.set_value(next(o for o in sel.options if o.startswith(scenario))).run()
    at.toggle(key="bs_table").set_value(table).run()
    assert metric(at, label) == expected


def test_binomial_arbitrage_is_reported_not_crashing():
    at = run("binomial")
    at.number_input(key="bn_u").set_value(1.01).run()
    at.number_input(key="bn_d").set_value(1.005).run()
    assert not at.exception
    assert any("arbitraje" in e.value for e in at.error)


@pytest.mark.parametrize("calc", ["Black-Scholes-Merton", "Árbol binomial (CRR)", "Monte Carlo",
                                  "Curva de forwards", "Curva IRS"])
def test_challenge_each_calculator(calc):
    at = run("challenge")
    at.radio[0].set_value(calc).run()
    assert not at.exception
    assert len(at.metric) >= 4
