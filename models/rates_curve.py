"""Módulo 2. Curva IRS: factores de descuento, tasas cero, forwards y tasas swap.

Notación del profesor (clases 7, 8 y 11):

* P(0,s)                         factor de descuento (precio de un bono cupón cero)
* r(0,s) = P(0,s)^(-1/s) - 1     tasa cero con capitalización anual
* r(0,s) = -ln P(0,s) / s        tasa cero continua
* F_{0,s-1}[P(s-1,s)] = P(0,s)/P(0,s-1)       precio forward del bono
* r_f(s-1,s) = P(0,s-1)/P(0,s) - 1            tasa forward por periodo
* r_swap = Σ r_f P(0,T_i) / Σ P(0,T_i) = (1 - P(0,T_n)) / Σ P(0,T_i)

Con periodos de longitud τ (τ = 1 en los ejercicios de clase) las sumas se
ponderan por τ y la tasa forward se expresa anualizada (simple por periodo).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _check_tau(tau):
    if tau <= 0:
        raise ValueError("La longitud del periodo τ debe ser positiva.")


def discount_from_forwards(forwards, tau=1.0, compounding="discreta"):
    """P(0,T_n) a partir de tasas forward por periodo (decimales, anualizadas)."""
    _check_tau(tau)
    f = np.asarray(forwards, dtype=float)
    if f.size == 0:
        raise ValueError("Captura al menos una tasa forward.")
    if compounding == "discreta":
        growth = 1.0 + f * tau
        if np.any(growth <= 0):
            raise ValueError("Una tasa forward hace 1 + f·τ ≤ 0; revisa los datos.")
        return np.cumprod(1.0 / growth)
    if compounding == "continua":
        return np.exp(-np.cumsum(f * tau))
    raise ValueError("Capitalización desconocida: usa 'discreta' o 'continua'.")


def bootstrap_from_swaps(swap_rates, tau=1.0):
    """Bootstrap (clase 11): P_n = (1 - S_n Σ_{i<n} τ P_i) / (1 + S_n τ)."""
    _check_tau(tau)
    s = np.asarray(swap_rates, dtype=float)
    if s.size == 0:
        raise ValueError("Captura al menos una tasa swap.")
    discount, annuity = [], 0.0
    for rate in s:
        p = (1.0 - rate * annuity) / (1.0 + rate * tau)
        if p <= 0:
            raise ValueError("El bootstrap produjo un factor de descuento no positivo; revisa las tasas swap.")
        discount.append(p)
        annuity += tau * p
    return np.array(discount)


def curve_table(discount, tau=1.0):
    """Tabla completa de la curva a partir de los factores de descuento."""
    _check_tau(tau)
    p = np.asarray(discount, dtype=float)
    if np.any(p <= 0):
        raise ValueError("Los factores de descuento deben ser positivos.")
    n = p.size
    t = tau * np.arange(1, n + 1)
    p_prev = np.r_[1.0, p[:-1]]
    fwd_bond = p / p_prev
    fwd_simple = (p_prev / p - 1.0) / tau
    fwd_cont = -np.log(fwd_bond) / tau
    annuity = np.cumsum(tau * p)
    swap = (1.0 - p) / annuity
    swap_from_fwd = np.cumsum(fwd_simple * tau * p) / annuity
    return pd.DataFrame({
        "Periodo i": np.arange(1, n + 1),
        "T (años)": t,
        "Forward r_f(T_{i-1},T_i)": fwd_simple,
        "Forward continua": fwd_cont,
        "Precio forward del bono": fwd_bond,
        "P(0,T)": p,
        "Tasa cero anual": p ** (-1.0 / t) - 1.0,
        "Tasa cero continua": -np.log(p) / t,
        "Anualidad Σ τP": annuity,
        "Tasa swap par": swap,
        "Tasa swap (Σ fP / Σ P)": swap_from_fwd,
    })


def swap_mark_to_market(discount, fixed_rate, notional=100.0, tau=1.0, payer=True):
    """Valor de un IRS existente (clase 8): pata flotante − pata fija para quien paga fijo."""
    p = np.asarray(discount, dtype=float)
    floating = notional * (1.0 - p[-1])
    fixed = notional * fixed_rate * tau * p.sum()
    value = floating - fixed
    return {"pata_flotante": floating, "pata_fija": fixed, "valor": value if payer else -value}


def commodity_swap_price(forward_prices, discount):
    """Precio de un swap de commodity o equity (clases 7 y 8): Σ F P / Σ P."""
    f = np.asarray(forward_prices, dtype=float)
    p = np.asarray(discount, dtype=float)
    return float((f * p).sum() / p.sum())
