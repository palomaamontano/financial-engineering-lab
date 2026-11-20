"""Ejemplos resueltos en clase y en la Tarea 1.

Tolerancia contra las notas: 0.005 (las notas redondean a 2-4 decimales).
"""
import math

import numpy as np
import pytest

from models import black_scholes as bs
from models import binomial as bn
from models import fx_forward as fx
from models import rates_curve as rc

TOL_NOTES = 5e-3


# ---------- Módulo 1: forwards de tipo de cambio ----------

def test_tarea1_ej14_forward_curve():
    e = fx.EXERCISE_14
    res = fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"], e["rows"])
    assert res.x_t == pytest.approx(17.3517, abs=1e-9)
    expected_f = [17.3531, 17.3545, 17.3587, 17.36515, 17.4023]
    expected_r = [0.06625, 0.06625, 0.06624, 0.06814, 0.06795]
    assert list(res.table["Forward F(t,T+s)"]) == pytest.approx(expected_f, abs=1e-9)
    # las notas redondean la tasa a 3 decimales de porcentaje
    assert list(res.table["r MXN implícita (cont.)"]) == pytest.approx(expected_r, abs=2e-5)


def test_clase5_fx_forward_teorico():
    # X = 17, r_USD = 3%, r_MXN = 8%, T = 1  ->  17.87
    assert fx.theoretical_fx_forward(17.0, 0.08, 0.03, 1.0) == pytest.approx(17.87, abs=TOL_NOTES)


def test_tarea1_ej13_fx_forward_eur():
    # 1.20 USD/EUR, r_USD 5%, r_EUR 6%  ->  1.1880
    assert fx.theoretical_fx_forward(1.20, 0.05, 0.06, 1.0) == pytest.approx(1.1880, abs=1e-3)


# ---------- Módulo 2: curva IRS ----------

def test_clase7_curva_desde_factores():
    t = rc.curve_table([0.95, 0.90, 0.85, 0.80])
    assert list(t["Tasa cero anual"]) == pytest.approx([0.05263, 0.05409, 0.05567, 0.05737], abs=2e-5)
    assert list(t["Tasa cero continua"]) == pytest.approx([0.05129, 0.05268, 0.05417, 0.05579], abs=2e-5)
    assert list(t["Precio forward del bono"]) == pytest.approx([0.95, 0.9474, 0.9444, 0.9412], abs=1e-4)
    # Las notas dan 5.69% porque truncan el primer forward (5.263%) a 5.2%.
    # El valor exacto es (1 - 0.80) / 3.5 = 5.714%.
    assert t["Tasa swap par"].iloc[-1] == pytest.approx(0.2 / 3.5, abs=1e-12)


def test_clase7_equity_swap():
    # Las notas escriben 492 / 4.25 = 115.76, pero la suma correcta es 484 / 4.25 = 113.88.
    price = rc.commodity_swap_price([90, 105, 115, 120, 147], [0.95, 0.90, 0.85, 0.80, 0.75])
    assert price == pytest.approx(113.882, abs=1e-3)


def test_clase8_tasa_irs():
    t = rc.curve_table([0.9237, 0.8545, 0.7764, 0.6851])
    assert t["Tasa swap par"].iloc[-1] == pytest.approx(0.0972, abs=1e-4)
    assert list(t["Forward r_f(T_{i-1},T_i)"]) == pytest.approx([0.0826, 0.0810, 0.1006, 0.1333], abs=1e-4)


def test_clase8_oil_swap_y_mtm():
    f = [500.0, 500.5, 500.18, 500.32]
    p = [0.9237, 0.8545, 0.7764, 0.6851]
    price = rc.commodity_swap_price(f, p)
    assert price == pytest.approx(500.243, abs=TOL_NOTES)
    assert rc.commodity_swap_price(f[:3], p[:3]) == pytest.approx(500.22, abs=TOL_NOTES)
    mtm = sum((fi + 1) * pi for fi, pi in zip(f, p)) - price * sum(p)
    assert mtm == pytest.approx(3.24, abs=TOL_NOTES)


def test_clase11_bootstrap():
    p = rc.bootstrap_from_swaps([0.09, 0.10, 0.11])
    assert list(p) == pytest.approx([0.917431, 0.825688, 0.728159], abs=1e-6)
    t = rc.curve_table(p)
    assert list(t["Tasa cero anual"]) == pytest.approx([0.09, 0.1005, 0.1115], abs=1e-4)
    assert list(t["Tasa swap par"]) == pytest.approx([0.09, 0.10, 0.11], abs=1e-12)


# ---------- Módulo 3: árbol binomial ----------

def test_clase14_call_un_periodo():
    r = bn.price_binomial(100, 100, 0.02, 0.0, 1.0, 1, "call", "european", "manual", u=1.05, d=0.95)
    assert r.params.p == pytest.approx(0.7020, abs=1e-4)
    assert r.price == pytest.approx(3.44, abs=TOL_NOTES)
    assert r.delta[0][0] == pytest.approx(0.5, abs=1e-12)
    assert r.bond[0][0] == pytest.approx(-46.56, abs=TOL_NOTES)  # las notas truncan a -46.55


def test_clase14_straddle_como_call_mas_put():
    kw = dict(S0=60, K=50, r=0.08, delta=0.0, T=1.0, N=1, style="european", mode="manual", u=70 / 60, d=45 / 60)
    total = bn.price_binomial(option="call", **kw).price + bn.price_binomial(option="put", **kw).price
    assert total == pytest.approx(15.69, abs=TOL_NOTES)


def test_clase14_put_dos_periodos():
    r = bn.price_binomial(50, 52, 0.05, 0.0, 2.0, 2, "put", "european", "manual", u=1.2, d=0.8)
    assert r.params.p == pytest.approx(0.6281, abs=1e-4)
    assert r.price == pytest.approx(4.19, abs=TOL_NOTES)


# ---------- Módulo 4: Black-Scholes ----------

def test_clase16_call():
    r = bs.bs_price(41, 40, 0.08, 0.0, 0.30, 0.25, "call")
    assert r.d1 == pytest.approx(0.3730, abs=1e-4)
    assert r.d2 == pytest.approx(0.2230, abs=1e-4)
    assert r.price == pytest.approx(3.40, abs=TOL_NOTES)
    tabla = bs.bs_price(41, 40, 0.08, 0.0, 0.30, 0.25, "call", table_mode=True)
    assert (tabla.n_d1, tabla.n_d2) == (0.6443, 0.5871)


def test_clase17_put_divisas_exacto_y_tabla():
    exact = bs.bs_price(1.2, 1.1, 0.06, 0.03, 0.15, 0.5, "put")
    assert exact.d1 == pytest.approx(1.014805, abs=1e-6)
    assert exact.price == pytest.approx(0.01066, abs=1e-5)
    tabla = bs.bs_price(1.2, 1.1, 0.06, 0.03, 0.15, 0.5, "put", table_mode=True)
    assert tabla.n_d1 == pytest.approx(0.1562, abs=1e-4)
    assert tabla.n_d2 == pytest.approx(0.1814, abs=1e-4)
    assert tabla.price == pytest.approx(0.009, abs=TOL_NOTES)


def test_clase13_paridad_con_dividendo_discreto():
    prepaid = 64 - 0.8 * math.exp(-0.12 / 12)
    put = 8 - prepaid + 60 * math.exp(-0.12 * 4 / 12)
    assert put == pytest.approx(2.4393, abs=TOL_NOTES)


def test_clase17_volatilidad_historica_muestral_mensual():
    prices = np.array([80, 64, 80, 64, 80, 100, 80, 64, 80.0])
    u = np.diff(np.log(prices))
    sigma = math.sqrt(u.var(ddof=1) * 12)
    assert sigma == pytest.approx(0.8264, abs=1e-4)
