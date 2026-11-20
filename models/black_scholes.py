"""Módulo 4. Black-Scholes-Merton para opciones europeas (clases 16 y 17).

c = S e^{-δT} N(d1) - K e^{-rT} N(d2)
p = K e^{-rT} N(-d2) - S e^{-δT} N(-d1)
d1 = [ln(S/K) + (r - δ + σ²/2) T] / (σ √T),   d2 = d1 - σ √T

δ es el rendimiento continuo por dividendos. Para divisas (Garman-Kohlhagen)
r es la tasa doméstica y δ la tasa extranjera.

Modo "tabla Z": redondea d1 y d2 a dos decimales y N(·) a cuatro, igual que
los cálculos a mano de clase con la tabla de la normal.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from scipy.stats import norm


@dataclass
class BSResult:
    price: float
    d1: float
    d2: float
    n_d1: float          # N(d1) para call, N(-d1) para put
    n_d2: float          # N(d2) para call, N(-d2) para put
    disc_s: float        # S e^{-δT}
    disc_k: float        # K e^{-rT}


def validate_inputs(S, K, r, delta, sigma, T):
    if S <= 0:
        raise ValueError("El precio del subyacente S debe ser positivo.")
    if K <= 0:
        raise ValueError("El strike K debe ser positivo.")
    if sigma <= 0:
        raise ValueError("La volatilidad σ debe ser positiva.")
    if T <= 0:
        raise ValueError("El plazo T debe ser positivo.")


def d1_d2(S, K, r, delta, sigma, T):
    d1 = (math.log(S / K) + (r - delta + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return d1, d1 - sigma * math.sqrt(T)


def _cdf(x, table_mode):
    if table_mode:
        return round(float(norm.cdf(round(x, 2))), 4)
    return float(norm.cdf(x))


def bs_price(S, K, r, delta, sigma, T, option="call", table_mode=False) -> BSResult:
    validate_inputs(S, K, r, delta, sigma, T)
    d1, d2 = d1_d2(S, K, r, delta, sigma, T)
    disc_s = S * math.exp(-delta * T)
    disc_k = K * math.exp(-r * T)
    if option == "call":
        n1, n2 = _cdf(d1, table_mode), _cdf(d2, table_mode)
        price = disc_s * n1 - disc_k * n2
    elif option == "put":
        n1, n2 = _cdf(-d1, table_mode), _cdf(-d2, table_mode)
        price = disc_k * n2 - disc_s * n1
    else:
        raise ValueError("Tipo de opción desconocido: usa 'call' o 'put'.")
    return BSResult(price=price, d1=d1, d2=d2, n_d1=n1, n_d2=n2, disc_s=disc_s, disc_k=disc_k)


def greeks(S, K, r, delta, sigma, T, option="call"):
    """Greeks con unidades declaradas.

    delta, gamma: por unidad del subyacente
    vega: cambio en el precio por +1 punto de volatilidad (σ + 0.01)
    rho:  cambio en el precio por +1 punto de tasa (r + 0.01)
    theta_anual: dV/dt por año; theta_diaria = theta_anual / 365
    """
    validate_inputs(S, K, r, delta, sigma, T)
    d1, d2 = d1_d2(S, K, r, delta, sigma, T)
    sq = math.sqrt(T)
    eq, er = math.exp(-delta * T), math.exp(-r * T)
    pdf = float(norm.pdf(d1))
    gamma = eq * pdf / (S * sigma * sq)
    vega = S * eq * pdf * sq
    common = -S * eq * pdf * sigma / (2 * sq)
    if option == "call":
        dlt = eq * norm.cdf(d1)
        theta = common - r * K * er * norm.cdf(d2) + delta * S * eq * norm.cdf(d1)
        rho = K * T * er * norm.cdf(d2)
    else:
        dlt = -eq * norm.cdf(-d1)
        theta = common + r * K * er * norm.cdf(-d2) - delta * S * eq * norm.cdf(-d1)
        rho = -K * T * er * norm.cdf(-d2)
    return {
        "Delta": float(dlt),
        "Gamma": float(gamma),
        "Vega (por +1% σ)": float(vega) / 100,
        "Theta (por año)": float(theta),
        "Theta (por día)": float(theta) / 365,
        "Rho (por +1% r)": float(rho) / 100,
    }


def put_call_parity(S, K, r, delta, sigma, T):
    """c - p frente a S e^{-δT} - K e^{-rT}. La diferencia debe ser ~0."""
    c = bs_price(S, K, r, delta, sigma, T, "call").price
    p = bs_price(S, K, r, delta, sigma, T, "put").price
    rhs = S * math.exp(-delta * T) - K * math.exp(-r * T)
    return {"call": c, "put": p, "c_menos_p": c - p, "S_e^-δT_menos_K_e^-rT": rhs, "diferencia": (c - p) - rhs}
