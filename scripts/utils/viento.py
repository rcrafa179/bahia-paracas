"""Funciones meteorológicas: viento, humedad relativa, estaciones del año."""
from __future__ import annotations

import numpy as np
import pandas as pd


def velocidad_direccion(u, v):
    """Velocidad (m/s) y dirección meteorológica (grados, de DONDE viene el viento).

    Convención: 0° = viento del norte, 90° = del este. Para u>0 (hacia el este)
    el viento viene del oeste (270°).
    """
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    vel = np.hypot(u, v)
    dirm = (270.0 - np.degrees(np.arctan2(v, u))) % 360.0
    return vel, dirm


def humedad_relativa(t_k, td_k):
    """Humedad relativa (%) a partir de temperatura y punto de rocío en K (fórmula de Magnus)."""
    t = np.asarray(t_k, dtype=float) - 273.15
    td = np.asarray(td_k, dtype=float) - 273.15
    a, b = 17.625, 243.04  # Alduchov y Eskridge (1996)
    hr = 100.0 * np.exp(a * td / (b + td) - a * t / (b + t))
    return np.clip(hr, 0, 100)


def estacion_del_anio(indice: pd.DatetimeIndex) -> np.ndarray:
    """Estación climatológica (hemisferio sur): DEF verano, MAM otoño, JJA invierno, SON primavera."""
    mapa = {12: "DEF", 1: "DEF", 2: "DEF", 3: "MAM", 4: "MAM", 5: "MAM",
            6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}
    return np.array([mapa[m] for m in indice.month])


def a_hora_local(indice_utc: pd.DatetimeIndex, desfase_h: int) -> pd.DatetimeIndex:
    """Convierte un índice UTC (naive) a hora local naive sumando el desfase (p. ej. -5)."""
    return indice_utc + pd.Timedelta(hours=desfase_h)


def diferencia_angular(a, b):
    """Diferencia angular a-b en el rango [-180, 180)."""
    return (np.asarray(a) - np.asarray(b) + 180.0) % 360.0 - 180.0
