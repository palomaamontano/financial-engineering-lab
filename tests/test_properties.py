"""Identidades exactas, desigualdades, convergencia y validación de inputs."""
import math

import numpy as np
import pytest

from models import black_scholes as bs
from models import binomial as bn
from models import fx_forward as fx
from models import monte_carlo as mc
from models import rates_curve as rc
from utils import rules

BASE = dict(S=100.0, K=100.0, r=0.05, delta=0.02, sigma=0.25, T=1.0)


# ---------- identidades de curvas (tolerancia 1e-10) ----------

@pytest.mark.parametrize("comp", ["discreta", "continua"])
def test_forwards_a_factores_y_regreso(comp):
    f = [0.05, 0.055, 0.06, 0.065, 0.07]
    p = rc.discount_from_forwards(f, tau=1.0, compounding=comp)
    t = rc.curve_table(p)
    col = "Forward r_f(T_{i-1},T_i)" if comp == "discreta" else "Forward continua"
    assert list(t[col]) == pytest.approx(f, abs=1e-12)
    T = t["T (años)"].to_numpy()
    assert np.allclose((1 + t["Tasa cero anual"]) ** (-T), p, atol=1e-12)
    assert np.allclose(np.exp(-t["Tasa cero continua"] * T), p, atol=1e-12)
    assert np.allclose(t["Tasa swap par"], t["Tasa swap (Σ fP / Σ P)"], atol=1e-12)


def test_bootstrap_inverso_de_tasa_swap_semestral():
    f = [0.06, 0.062, 0.064, 0.067]
    p = rc.discount_from_forwards(f, tau=0.5)
    swaps = rc.curve_table(p, tau=0.5)["Tasa swap par"].to_numpy()
    assert np.allclose(rc.bootstrap_from_swaps(swaps, tau=0.5), p, atol=1e-12)


def test_swap_a_tasa_par_vale_cero():
    p = rc.discount_from_forwards([0.05, 0.06, 0.07])
    s = rc.curve_table(p)["Tasa swap par"].iloc[-1]
    assert rc.swap_mark_to_market(p, s)["valor"] == pytest.approx(0.0, abs=1e-10)


# ---------- forwards de tipo de cambio ----------

def test_fx_paridad_cubierta_reproduce_forwards():
    e = fx.EXERCISE_14
    res = fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"], e["rows"])
    for _, row in res.table.iterrows():
        f = fx.theoretical_fx_forward(res.x_t, row["r MXN implícita (cont.)"], row["r USD (cont.)"], row["τ (años)"])
        assert f == pytest.approx(row["Forward F(t,T+s)"], abs=1e-10)


def test_fx_choque_usd_uno_a_uno_y_reglas():
    e = fx.EXERCISE_14
    base = fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"], e["rows"]).table
    up = fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"], fx.shocked_rows(e["rows"], usd_shift=0.01)).table
    diff = up["r MXN implícita (cont.)"] - base["r MXN implícita (cont.)"]
    assert np.allclose(diff, 0.01, atol=1e-12)
    assert np.allclose(up["Forward F(t,T+s)"], base["Forward F(t,T+s)"], atol=1e-12)
    pts = fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"], fx.shocked_rows(e["rows"], points_shift=5)).table
    assert (pts["r MXN implícita (cont.)"] > base["r MXN implícita (cont.)"]).all()
    spot = fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"] + 1, e["rows"]).table
    assert (spot["r MXN implícita (cont.)"] < base["r MXN implícita (cont.)"]).all()


def test_fx_sin_on_tn_explica_el_error():
    e = fx.EXERCISE_14
    with pytest.raises(ValueError, match="O/N"):
        fx.build_fx_curve(e["trade_date"], e["spot_date"], e["spot_rate"], e["rows"][2:])


# ---------- Black-Scholes ----------

@pytest.mark.parametrize("S", [60.0, 100.0, 140.0])
def test_paridad_put_call_bsm(S):
    par = bs.put_call_parity(S, 100.0, 0.05, 0.02, 0.25, 1.0)
    assert par["diferencia"] == pytest.approx(0.0, abs=1e-10)


@pytest.mark.parametrize("option", ["call", "put"])
def test_greeks_contra_diferencias_finitas(option):
    g = bs.greeks(option=option, **BASE)
    f = lambda **kw: bs.bs_price(option=option, **{**BASE, **kw}).price
    h = 1e-4
    assert g["Delta"] == pytest.approx((f(S=BASE["S"] + h) - f(S=BASE["S"] - h)) / (2 * h), abs=1e-6)
    assert g["Gamma"] == pytest.approx((f(S=BASE["S"] + h) - 2 * f() + f(S=BASE["S"] - h)) / h ** 2, abs=1e-4)
    assert g["Vega (por +1% σ)"] * 100 == pytest.approx((f(sigma=BASE["sigma"] + h) - f(sigma=BASE["sigma"] - h)) / (2 * h), abs=1e-5)
    assert g["Rho (por +1% r)"] * 100 == pytest.approx((f(r=BASE["r"] + h) - f(r=BASE["r"] - h)) / (2 * h), abs=1e-5)
    assert g["Theta (por año)"] == pytest.approx(-(f(T=BASE["T"] + h) - f(T=BASE["T"] - h)) / (2 * h), abs=1e-5)


def test_limite_volatilidad_cero():
    p = bs.bs_price(100, 90, 0.05, 0.0, 1e-8, 1.0, "call").price
    assert p == pytest.approx(100 - 90 * math.exp(-0.05), abs=1e-6)


@pytest.mark.parametrize("bad", [dict(S=-1), dict(K=0), dict(sigma=0), dict(T=0)])
def test_validacion_inputs_bsm(bad):
    with pytest.raises(ValueError):
        bs.bs_price(option="call", **{**BASE, **bad})


@pytest.mark.parametrize("option", ["call", "put"])
@pytest.mark.parametrize("param,step", [("S", 5.0), ("K", 5.0), ("r", 0.01), ("delta", 0.01), ("sigma", 0.05)])
def test_reglas_modo_defensa_bsm(option, param, step):
    sign, _, _ = rules.option_rule(param, option)
    base = bs.bs_price(option=option, **BASE).price
    shocked = bs.bs_price(option=option, **{**BASE, param: BASE[param] + step}).price
    assert np.sign(shocked - base) == sign


# ---------- árbol binomial ----------

def test_paridad_en_arbol_europeo():
    kw = dict(S0=100, K=95, r=0.05, delta=0.02, T=1.0, N=50, style="european", mode="crr", sigma=0.3)
    c = bn.price_binomial(option="call", **kw).price
    p = bn.price_binomial(option="put", **kw).price
    assert c - p == pytest.approx(100 * math.exp(-0.02) - 95 * math.exp(-0.05), abs=1e-10)


@pytest.mark.parametrize("option", ["call", "put"])
def test_americana_mayor_o_igual_que_europea(option):
    kw = dict(S0=100, K=100, r=0.06, delta=0.04, T=1.0, N=100, option=option, mode="crr", sigma=0.3)
    assert bn.price_binomial(style="american", **kw).price >= bn.price_binomial(style="european", **kw).price - 1e-12


def test_call_americano_sin_dividendos_igual_a_europeo():
    kw = dict(S0=100, K=100, r=0.06, delta=0.0, T=1.0, N=100, option="call", mode="crr", sigma=0.3)
    am = bn.price_binomial(style="american", **kw)
    assert am.price == pytest.approx(bn.price_binomial(style="european", **kw).price, abs=1e-8)
    assert am.early_exercise_nodes == 0


def test_put_americano_ejerce_anticipadamente():
    r = bn.price_binomial(100, 110, 0.08, 0.0, 1.0, 20, "put", "american", "crr", sigma=0.2)
    assert r.early_exercise_nodes > 0


@pytest.mark.parametrize("option", ["call", "put"])
def test_convergencia_crr_a_bsm(option):
    exact = bs.bs_price(100, 100, 0.05, 0.0, 0.2, 1.0, option).price
    tree = bn.price_binomial(100, 100, 0.05, 0.0, 1.0, 500, option, "european", "crr", sigma=0.2, store=False).price
    assert abs(tree - exact) < 0.01


def test_arbitraje_en_arbol_rechazado():
    with pytest.raises(ValueError, match="arbitraje"):
        bn.price_binomial(100, 100, 0.20, 0.0, 1.0, 1, "call", "european", "manual", u=1.1, d=0.9)


# ---------- Monte Carlo ----------

@pytest.mark.parametrize("option", ["call", "put"])
def test_mc_dentro_de_tres_errores_estandar(option):
    exact = bs.bs_price(option=option, **BASE).price
    res = mc.mc_european(BASE["S"], BASE["K"], BASE["r"], BASE["delta"], BASE["sigma"], BASE["T"], 200_000, 7, option)
    assert abs(res["precio"] - exact) < 3 * res["error_estandar"]


def test_cobertura_intervalo_confianza():
    exact = bs.bs_price(option="call", **BASE).price
    hits = 0
    for seed in range(200):
        res = mc.mc_european(BASE["S"], BASE["K"], BASE["r"], BASE["delta"], BASE["sigma"], BASE["T"], 2_000, seed, "call")
        hits += res["ic_inferior"] <= exact <= res["ic_superior"]
    assert 0.90 <= hits / 200 <= 0.99


def test_mc_semilla_reproducible_y_limite():
    a = mc.mc_european(100, 100, 0.05, 0, 0.2, 1, 10_000, 3)["precio"]
    b = mc.mc_european(100, 100, 0.05, 0, 0.2, 1, 10_000, 3)["precio"]
    assert a == b
    with pytest.raises(ValueError):
        mc.mc_european(100, 100, 0.05, 0, 0.2, 1, mc.MAX_SIMULATIONS + 1, 3)
