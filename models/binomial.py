"""Módulo 3. Árboles binomiales (clases 13 y 14).

Cartera réplica en cada nodo:
    Δ = e^{-δh} (C_u - C_d) / (S (u - d))
    B = e^{-rh} (u C_d - d C_u) / (u - d)
    C = Δ S + B
Probabilidad neutral al riesgo:
    p* = (e^{(r-δ)h} - d) / (u - d),  válida sólo si d < e^{(r-δ)h} < u
Valoración hacia atrás:
    C = e^{-rh} [p* C_u + (1 - p*) C_d]
Americana: en cada nodo se compara el valor de continuación con el ejercicio inmediato.

Modos para u y d:
    "manual":  u y d dados (así vienen los ejercicios del profesor)
    "crr":     Cox-Ross-Rubinstein, u = e^{σ√h}, d = 1/u
    "forward": árbol forward de McDonald, u = e^{(r-δ)h + σ√h}, d = e^{(r-δ)h - σ√h}
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

MAX_TREE_STORE = 60  # se guardan los árboles completos sólo hasta este número de pasos


@dataclass
class TreeParams:
    u: float
    d: float
    p: float
    h: float
    growth: float


@dataclass
class BinomialResult:
    price: float
    params: TreeParams
    stock: list = field(default_factory=list)      # stock[n] = precios en el paso n (j = número de subidas)
    value: list = field(default_factory=list)      # valor de la opción
    exercise: list = field(default_factory=list)   # True donde conviene ejercer anticipadamente
    delta: list = field(default_factory=list)      # Δ en cada nodo (pasos 0..N-1)
    bond: list = field(default_factory=list)       # B en cada nodo (pasos 0..N-1)
    early_exercise_nodes: int = 0


def tree_params(r, delta, T, N, mode="crr", sigma=None, u=None, d=None) -> TreeParams:
    if T <= 0:
        raise ValueError("El plazo T debe ser positivo.")
    if int(N) != N or N < 1:
        raise ValueError("El número de pasos N debe ser un entero mayor o igual a 1.")
    h = T / N
    if mode == "crr":
        if sigma is None or sigma <= 0:
            raise ValueError("En modo CRR la volatilidad σ debe ser positiva.")
        u = math.exp(sigma * math.sqrt(h))
        d = 1.0 / u
    elif mode == "forward":
        if sigma is None or sigma <= 0:
            raise ValueError("En el árbol forward la volatilidad σ debe ser positiva.")
        u = math.exp((r - delta) * h + sigma * math.sqrt(h))
        d = math.exp((r - delta) * h - sigma * math.sqrt(h))
    elif mode == "manual":
        if u is None or d is None or d <= 0 or u <= d:
            raise ValueError("En modo manual se requiere 0 < d < u.")
    else:
        raise ValueError("Modo desconocido: usa 'manual', 'crr' o 'forward'.")
    growth = math.exp((r - delta) * h)
    if not (d < growth < u):
        raise ValueError(
            f"Hay arbitraje: se requiere d < e^((r−δ)h) < u, pero d = {d:.6f}, "
            f"e^((r−δ)h) = {growth:.6f}, u = {u:.6f}. Con estos datos p* = "
            f"{(growth - d) / (u - d):.4f} no está entre 0 y 1."
        )
    return TreeParams(u=u, d=d, p=(growth - d) / (u - d), h=h, growth=growth)


def _payoff(s, K, option):
    if option == "call":
        return np.maximum(s - K, 0.0)
    if option == "put":
        return np.maximum(K - s, 0.0)
    raise ValueError("Tipo de opción desconocido: usa 'call' o 'put'.")


def price_binomial(S0, K, r, delta, T, N, option="call", style="european",
                   mode="crr", sigma=None, u=None, d=None, store=True) -> BinomialResult:
    if S0 <= 0 or K <= 0:
        raise ValueError("S y K deben ser positivos.")
    if style not in ("european", "american"):
        raise ValueError("Estilo desconocido: usa 'european' o 'american'.")
    N = int(N)
    prm = tree_params(r, delta, T, N, mode, sigma, u, d)
    disc = math.exp(-r * prm.h)
    store = store and N <= MAX_TREE_STORE

    def stock_at(n):
        j = np.arange(n + 1)
        return S0 * prm.u ** j * prm.d ** (n - j)

    v = _payoff(stock_at(N), K, option)
    stock_tree = [None] * (N + 1)
    value_tree = [None] * (N + 1)
    ex_tree = [None] * (N + 1)
    delta_tree = [None] * N
    bond_tree = [None] * N
    if store:
        stock_tree[N] = stock_at(N)
        value_tree[N] = v.copy()
        ex_tree[N] = np.zeros(N + 1, dtype=bool)
    early = 0
    for n in range(N - 1, -1, -1):
        s = stock_at(n)
        vu, vd = v[1:n + 2], v[:n + 1]
        cont = disc * (prm.p * vu + (1 - prm.p) * vd)
        if store:
            delta_tree[n] = math.exp(-delta * prm.h) * (vu - vd) / (s * (prm.u - prm.d))
            bond_tree[n] = disc * (prm.u * vd - prm.d * vu) / (prm.u - prm.d)
        if style == "american":
            exercise_value = _payoff(s, K, option)
            ex = exercise_value > cont + 1e-12
            v = np.where(ex, exercise_value, cont)
            early += int(ex.sum())
        else:
            ex = np.zeros(n + 1, dtype=bool)
            v = cont
        if store:
            stock_tree[n], value_tree[n], ex_tree[n] = s, v.copy(), ex
    res = BinomialResult(price=float(v[0]), params=prm, early_exercise_nodes=early)
    if store:
        res.stock, res.value, res.exercise = stock_tree, value_tree, ex_tree
        res.delta, res.bond = delta_tree, bond_tree
    return res


def convergence(S0, K, r, delta, sigma, T, option="call", style="european", n_max=200, mode="crr"):
    """Precio del árbol para N = 1..n_max (para comparar contra BSM)."""
    steps = np.arange(1, int(n_max) + 1)
    prices = [price_binomial(S0, K, r, delta, T, int(n), option, style, mode, sigma, store=False).price
              for n in steps]
    return steps, np.array(prices)
