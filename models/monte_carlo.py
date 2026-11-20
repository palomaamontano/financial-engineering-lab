"""Módulo 5. Monte Carlo bajo movimiento browniano geométrico neutral al riesgo (clase 16).

S(T) = S(0) exp[(r - δ - σ²/2) T + σ √T Z],   Z ~ N(0,1)
precio = e^{-rT} · promedio(payoff)
error estándar = desviación estándar muestral / √n
IC 95% = precio ± 1.96 · EE

Para una opción europea sólo importa S(T) y la solución del GBM es exacta, así
que el número de pasos temporales no cambia el precio: sólo sirve para dibujar
trayectorias. Los pasos importarían en opciones que dependen de la trayectoria.
"""
from __future__ import annotations

import math

import numpy as np

MAX_SIMULATIONS = 500_000
MAX_STEPS = 365
MAX_PLOT_PATHS = 200
Z_95 = 1.959963984540054


def _validate(S0, K, sigma, T, n_sims):
    if S0 <= 0 or K <= 0:
        raise ValueError("S y K deben ser positivos.")
    if sigma <= 0:
        raise ValueError("La volatilidad σ debe ser positiva.")
    if T <= 0:
        raise ValueError("El plazo T debe ser positivo.")
    if int(n_sims) != n_sims or n_sims < 2:
        raise ValueError("El número de simulaciones debe ser un entero ≥ 2.")
    if n_sims > MAX_SIMULATIONS:
        raise ValueError(f"Máximo {MAX_SIMULATIONS:,} simulaciones para no bloquear la aplicación.")


def terminal_prices(S0, r, delta, sigma, T, n_sims, seed=None, antithetic=False):
    rng = np.random.default_rng(seed)
    if antithetic:
        half = rng.standard_normal((int(n_sims) + 1) // 2)
        z = np.concatenate([half, -half])[: int(n_sims)]
    else:
        z = rng.standard_normal(int(n_sims))
    return S0 * np.exp((r - delta - 0.5 * sigma ** 2) * T + sigma * math.sqrt(T) * z)


def mc_european(S0, K, r, delta, sigma, T, n_sims=50_000, seed=42, option="call", antithetic=False):
    _validate(S0, K, sigma, T, n_sims)
    st = terminal_prices(S0, r, delta, sigma, T, n_sims, seed, antithetic)
    if option == "call":
        payoff = np.maximum(st - K, 0.0)
    elif option == "put":
        payoff = np.maximum(K - st, 0.0)
    else:
        raise ValueError("Tipo de opción desconocido: usa 'call' o 'put'.")
    disc_payoff = math.exp(-r * T) * payoff
    n = int(n_sims)
    if antithetic and n % 2 == 0:
        pairs = 0.5 * (disc_payoff[: n // 2] + disc_payoff[n // 2:])
        price = float(pairs.mean())
        se = float(pairs.std(ddof=1) / math.sqrt(pairs.size))
    else:
        price = float(disc_payoff.mean())
        se = float(disc_payoff.std(ddof=1) / math.sqrt(n))
    return {
        "precio": price,
        "error_estandar": se,
        "ic_inferior": price - Z_95 * se,
        "ic_superior": price + Z_95 * se,
        "prob_ejercicio": float((payoff > 0).mean()),
        "media_ST": float(st.mean()),
        "forward_teorico": S0 * math.exp((r - delta) * T),
        "ST_muestra": st[: min(n, 20_000)],
        "payoff_descontado": disc_payoff,
    }


def simulate_paths(S0, r, delta, sigma, T, steps=50, n_paths=50, seed=42):
    """Trayectorias para graficar (solución exacta del GBM en cada paso)."""
    steps = int(min(max(steps, 1), MAX_STEPS))
    n_paths = int(min(max(n_paths, 1), MAX_PLOT_PATHS))
    rng = np.random.default_rng(seed)
    h = T / steps
    z = rng.standard_normal((n_paths, steps))
    increments = (r - delta - 0.5 * sigma ** 2) * h + sigma * math.sqrt(h) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = S0 * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    times = np.linspace(0.0, T, steps + 1)
    return times, paths


def running_estimate(disc_payoff, points=60):
    """Estimación acumulada y su error estándar conforme crece n (convergencia)."""
    n_total = disc_payoff.size
    ns = np.unique(np.geomspace(10, n_total, num=points).astype(int))
    csum = np.cumsum(disc_payoff)
    csq = np.cumsum(disc_payoff ** 2)
    means = csum[ns - 1] / ns
    var = (csq[ns - 1] - ns * means ** 2) / (ns - 1)
    se = np.sqrt(np.maximum(var, 0.0) / ns)
    return ns, means, se
