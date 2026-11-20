"""Módulo 1. Curva de forwards de tipo de cambio y curva implícita en MXN.

Metodología (clase 5 y ejercicio 14 de la Tarea 1):

* Puntos forward:   X_t(T, T+s) = (F_{t,T+s}(X) - F_{t,T}(X)) * 10,000
* Paridad cubierta: F_{t,T}(X) = X(t) * exp((r_MXN(t,T) - r_USD(t,T)) * (T - t))
* Tasa implícita:   r_MXN(t,T) = r_USD(t,T) + ln(F_{t,T}(X) / X(t)) / (T - t)

X(t) es el tipo de cambio "descontado" a la fecha de operación t. Se obtiene
encadenando los puntos O/N y T/N hacia atrás desde el spot (valor t+2).
Las tasas son continuas y el plazo se mide en días / base (365 por defecto).
"""
from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass

import pandas as pd

PIP_FACTOR = 10_000.0


@dataclass
class FxCurveResult:
    table: pd.DataFrame        # una fila por plazo
    x_t: float                 # tipo de cambio descontado X(t)
    forwards_by_date: dict     # fecha -> F_{t,fecha}(X)
    warnings: list


def _as_date(value) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return pd.Timestamp(value).date()


def chain_forwards(spot_date, spot_rate, rows, pip_factor=PIP_FACTOR):
    """Encadena los puntos forward para obtener F en cada fecha.

    rows: lista de dicts con llaves start, end, points.
    Si se conoce F en la fecha de inicio: F(end) = F(start) + puntos / 10,000.
    Si se conoce F en la fecha final:     F(start) = F(end) - puntos / 10,000.
    """
    forwards = {_as_date(spot_date): float(spot_rate)}
    warnings = []
    pending = list(range(len(rows)))
    while pending:
        progressed = False
        for i in list(pending):
            start, end = _as_date(rows[i]["start"]), _as_date(rows[i]["end"])
            pts = float(rows[i]["points"]) / pip_factor
            if start in forwards and end not in forwards:
                forwards[end] = forwards[start] + pts
            elif end in forwards and start not in forwards:
                forwards[start] = forwards[end] - pts
            elif start in forwards and end in forwards:
                implied = (forwards[end] - forwards[start]) * pip_factor
                if abs(implied - float(rows[i]["points"])) > 1e-6:
                    warnings.append(
                        f"El plazo {rows[i].get('tenor', i)} no es consistente: "
                        f"los demás puntos implican {implied:.2f} pips."
                    )
            else:
                continue
            pending.remove(i)
            progressed = True
        if not progressed:
            tenors = ", ".join(str(rows[i].get("tenor", i)) for i in pending)
            raise ValueError(
                "No se pudo conectar estos plazos con el spot: " + tenors +
                ". Cada plazo debe empezar o terminar en una fecha ya conocida."
            )
    return forwards, warnings


def build_fx_curve(trade_date, spot_date, spot_rate, rows, basis=365, pip_factor=PIP_FACTOR):
    """Construye la curva forward y la curva implícita en MXN.

    rows: dicts con tenor, start, end, points, r_usd (decimal, continua).
    """
    if spot_rate <= 0:
        raise ValueError("El tipo de cambio spot debe ser positivo.")
    if not rows:
        raise ValueError("Captura al menos un plazo.")
    trade_date, spot_date = _as_date(trade_date), _as_date(spot_date)
    for r in rows:
        if _as_date(r["end"]) <= _as_date(r["start"]):
            raise ValueError(f"En el plazo {r.get('tenor')} el vencimiento debe ser posterior al inicio.")
    forwards, warnings = chain_forwards(spot_date, spot_rate, rows, pip_factor)
    if trade_date not in forwards:
        raise ValueError(
            "Ningún plazo conecta con la fecha de operación t. Agrega el O/N "
            "(de t a t+1) y el T/N (de t+1 a t+2) para obtener X(t)."
        )
    x_t = forwards[trade_date]
    records = []
    for r in rows:
        end = _as_date(r["end"])
        days = (end - trade_date).days
        tau = days / basis
        fwd = forwards[end]
        r_usd = float(r["r_usd"])
        r_mxn = r_usd + math.log(fwd / x_t) / tau
        records.append({
            "Plazo": r.get("tenor", ""),
            "Inicio T": _as_date(r["start"]),
            "Vencimiento T+s": end,
            "Días desde t": days,
            "τ (años)": tau,
            "Puntos": float(r["points"]),
            "Forward F(t,T+s)": fwd,
            "r USD (cont.)": r_usd,
            "r MXN implícita (cont.)": r_mxn,
            "r MXN efectiva anual": math.exp(r_mxn) - 1,
            "Diferencial r MXN − r USD": r_mxn - r_usd,
            "P MXN (factor desc.)": math.exp(-r_mxn * tau),
            "P USD (factor desc.)": math.exp(-r_usd * tau),
        })
    table = pd.DataFrame(records)
    return FxCurveResult(table=table, x_t=x_t, forwards_by_date=forwards, warnings=warnings)


def theoretical_fx_forward(x_t, r_dom, r_for, tau):
    """Forward teórico por paridad cubierta continua (clase 5)."""
    return x_t * math.exp((r_dom - r_for) * tau)


def shocked_rows(rows, points_shift=0.0, usd_shift=0.0):
    """Copia de los plazos con choque aditivo en puntos (pips) y en r USD (decimal)."""
    return [{**r, "points": float(r["points"]) + points_shift, "r_usd": float(r["r_usd"]) + usd_shift} for r in rows]


EXERCISE_14 = {
    "trade_date": dt.date(2026, 5, 27),
    "spot_date": dt.date(2026, 5, 29),
    "spot_rate": 17.3545,
    "rows": [
        {"tenor": "O/N", "start": dt.date(2026, 5, 27), "end": dt.date(2026, 5, 28), "points": 14.0, "r_usd": 0.0368},
        {"tenor": "T/N", "start": dt.date(2026, 5, 28), "end": dt.date(2026, 5, 29), "points": 14.0, "r_usd": 0.0368},
        {"tenor": "S/N", "start": dt.date(2026, 5, 29), "end": dt.date(2026, 6, 1), "points": 42.0, "r_usd": 0.0368},
        {"tenor": "1W", "start": dt.date(2026, 5, 29), "end": dt.date(2026, 6, 5), "points": 106.5, "r_usd": 0.03672},
        {"tenor": "1M", "start": dt.date(2026, 5, 29), "end": dt.date(2026, 6, 30), "points": 478.0, "r_usd": 0.03669},
    ],
}
