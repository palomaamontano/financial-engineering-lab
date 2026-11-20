"""Reglas verificadas para "¿Qué debería suceder y por qué?" (Modo Defensa).

Cada regla devuelve:
    signo esperado: +1 sube, -1 baja, None si depende del caso
    tipo: "Determinista", "Depende de los supuestos", "Estadístico" o "Numérico"
    explicación: razonamiento financiero y matemático

La app compara el signo esperado con el cambio calculado. Las reglas se
derivan de las fórmulas (signos de los greeks) y no de interpretaciones libres.
"""
from __future__ import annotations

DET = "Determinista"
DEP = "Depende de los supuestos"
EST = "Estadístico"
NUM = "Numérico"


def option_rule(param, option, style="european", delta=0.0):
    """Reglas para BSM, árbol binomial y Monte Carlo (dirección del precio)."""
    call = option == "call"
    if param == "S":
        return (1 if call else -1, DET,
                "Delta del call es e^(−δT)N(d₁) > 0 y la del put es −e^(−δT)N(−d₁) < 0. "
                "Un spot más alto acerca al call a estar dentro del dinero y aleja al put.")
    if param == "K":
        return (-1 if call else 1, DET,
                "∂c/∂K = −e^(−rT)N(d₂) < 0 y ∂p/∂K = e^(−rT)N(−d₂) > 0. Un strike más alto "
                "es pagar más al ejercer el call o recibir más al ejercer el put.")
    if param == "r":
        return (1 if call else -1, DET,
                "Rho del call es K·T·e^(−rT)N(d₂) > 0 y la del put es negativa. Una tasa mayor baja el "
                "valor presente del strike y sube el forward S·e^((r−δ)T): favorece al call y perjudica al put.")
    if param == "delta":
        return (-1 if call else 1, DET,
                "Un rendimiento por dividendos mayor baja el forward S·e^((r−δ)T): el subyacente "
                "pierde valor al entregar flujos, lo que perjudica al call y favorece al put.")
    if param == "sigma":
        return (1, DET,
                "Vega = S·e^(−δT)·φ(d₁)·√T > 0 para call y put. Más volatilidad aumenta la "
                "probabilidad de terminar muy dentro del dinero, y la pérdida está acotada por la prima.")
    if param == "T":
        if call and delta == 0:
            return (1, DET,
                    "Sin dividendos, un call europeo vale al menos S − K·e^(−rT), que crece con T; "
                    "más plazo también da más tiempo para que el spot suba (theta negativa).")
        if style == "american":
            return (1, DET,
                    "Una opción americana con más plazo incluye todas las fechas de ejercicio de la "
                    "de menor plazo: nunca vale menos.")
        return (None, DEP,
                "En una opción europea el efecto del plazo depende del caso. Más tiempo suma valor por "
                "volatilidad, pero un put muy dentro del dinero (o un call con dividendos altos) puede "
                "valer menos porque el pago K se recibe más tarde.")
    if param == "N":
        return (None, NUM,
                "Aumentar los pasos refina la discretización: el árbol converge al precio de "
                "Black-Scholes, oscilando entre N pares e impares, con error del orden de 1/N.")
    if param == "n_sims":
        return (None, EST,
                "Más simulaciones no mueven el precio en una dirección fija: reducen el error "
                "estándar en proporción a 1/√n. El nuevo precio debe caer dentro del intervalo de confianza.")
    return (None, DEP, "Sin regla registrada para este parámetro.")


def fx_rule(param):
    if param == "spot":
        return (-1, DET,
                "Con los puntos fijos, todos los forwards suben lo mismo que el spot, pero la tasa "
                "implícita depende de ln(F/X(t)) = ln(1 + puntos/X(t)): al crecer X(t) el diferencial "
                "baja ligeramente y la tasa MXN implícita cae un poco.")
    if param == "points":
        return (1, DET,
                "Más puntos forward significan un forward más alto frente a X(t). Por paridad cubierta "
                "el diferencial r_MXN − r_USD = ln(F/X(t))/τ aumenta: la tasa MXN implícita sube.")
    if param == "r_usd":
        return (1, DET,
                "Los puntos fijan el diferencial de tasas. Si la curva USD sube y los puntos no cambian, "
                "la tasa MXN implícita sube exactamente lo mismo (uno a uno) y los forwards no cambian.")
    return (None, DEP, "Sin regla registrada para este parámetro.")


def irs_rule(param):
    if param == "forwards":
        return (1, DET,
                "Subir las tasas forward reduce todos los factores de descuento P(0,T) = Π 1/(1+f_i τ). "
                "Las tasas cero suben y la tasa swap, que es un promedio de los forwards ponderado por "
                "P(0,T_i), sube aproximadamente lo mismo que el choque.")
    return (None, DEP, "Sin regla registrada para este parámetro.")


def verdict(expected_sign, base, shocked, tol=1e-12):
    change = shocked - base
    if expected_sign is None:
        return "Sin dirección única: se interpreta el resultado calculado."
    actual = 0 if abs(change) <= tol else (1 if change > 0 else -1)
    if actual == expected_sign:
        return "✅ El resultado coincide con la teoría."
    if actual == 0:
        return "⚠️ El cambio es prácticamente nulo; revisa si el choque es demasiado pequeño."
    return "⚠️ El resultado contradice la regla: revisa los inputs o el caso especial."
