"""Viento para el forzamiento: SPSO observado o ERA5 corregido por mapeo de cuantiles.

ERA5 subestima los vientos Paracas y, además, no acierta QUÉ días ocurren (r ≈ 0,4 por la tarde).
El mapeo de cuantiles corrige la distribución de velocidades (por hora local, porque el sesgo
depende de la hora), pero no la cronología de los eventos. Por eso el método por defecto usa el
viento observado en SPSO y deja ERA5 corregido solo para rellenar sus huecos.

Métodos (config.yaml, forzamiento_viento.metodo):
  spso     viento de SPSO; huecos de velocidad con ERA5 corregido; direcciones faltantes
           (viento variable "VRB" o calma) con la dirección de ERA5.
  era5_qm  ERA5 de la boca corregido por mapeo de cuantiles, con su propia dirección.
  era5     ERA5 de la boca sin corregir.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from utils.viento import a_hora_local

CUANTILES = np.linspace(0.01, 0.99, 99)


def ajustar_qm(era: pd.Series, obs: pd.Series, horas_locales) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Cuantiles de ERA5 y del observado para cada hora local (pares horarios simultáneos)."""
    par = pd.DataFrame({"e": era, "o": obs.reindex(era.index),
                        "h": np.asarray(horas_locales)}, index=era.index).dropna()
    tablas = {}
    for hora, g in par.groupby("h"):
        if len(g) >= 30:  # con menos pares los cuantiles extremos no son estables
            tablas[int(hora)] = (np.quantile(g.e, CUANTILES), np.quantile(g.o, CUANTILES))
    return tablas


def aplicar_qm(era: pd.Series, horas_locales, tablas) -> pd.Series:
    """Aplica la corrección hora por hora. Fuera del rango ajustado se usa el factor del cuantil extremo."""
    h = np.asarray(horas_locales)
    salida = pd.Series(np.nan, index=era.index)
    for hora, (qe, qo) in tablas.items():
        sel = h == hora
        e = era.values[sel]
        c = np.interp(e, qe, qo)
        c = np.where(e > qe[-1], e * qo[-1] / qe[-1], c)
        c = np.where(e < qe[0], e * qo[0] / max(qe[0], 1e-6), c)
        salida[sel] = c
    return salida.clip(lower=0)


def validacion_cruzada(era: pd.Series, obs: pd.Series, desfase_h: int) -> pd.Series:
    """ERA5 corregido para cada mes con un ajuste que NO usa ese mes (deja un mes fuera)."""
    hl = a_hora_local(era.index, desfase_h).hour
    meses = era.index.to_period("M")
    salida = pd.Series(np.nan, index=era.index)
    for m in meses.unique():
        fuera = meses != m
        tablas = ajustar_qm(era[fuera], obs.reindex(era.index)[fuera], hl[fuera])
        dentro = meses == m
        salida[dentro] = aplicar_qm(era[dentro], hl[dentro], tablas).values
    return salida


def componentes(vel, direccion):
    """u, v (hacia donde sopla) a partir de velocidad y dirección meteorológica (de donde viene)."""
    rad = np.deg2rad(np.asarray(direccion, dtype=float))
    vel = np.asarray(vel, dtype=float)
    return -vel * np.sin(rad), -vel * np.cos(rad)


def viento_forzamiento(boca: pd.DataFrame, obs: pd.DataFrame, cfg) -> pd.DataFrame:
    """Serie horaria de viento para el forzamiento, con la fuente de cada hora.

    boca: ERA5 en el punto de la boca (vel_ms, dir_grados), periodo completo (para ajustar).
    obs : METAR de SPSO (vel_ms, dir_grados), periodo completo.
    Devuelve, para el índice de 'boca': vel_ms, dir_grados, u10_ms, v10_ms, fuente_vel, fuente_dir,
    vel_era5_ms, vel_era5_qm_ms.
    """
    fv = cfg["forzamiento_viento"]
    desfase = cfg["proyecto"]["zona_horaria_local"]
    hl = a_hora_local(boca.index, desfase).hour
    tablas = ajustar_qm(boca.vel_ms, obs.vel_ms.reindex(boca.index), hl)
    qm = aplicar_qm(boca.vel_ms, hl, tablas)

    out = pd.DataFrame(index=boca.index)
    out["vel_era5_ms"] = boca.vel_ms
    out["vel_era5_qm_ms"] = qm
    metodo = fv["metodo"]
    if metodo == "era5":
        out["vel_ms"], out["dir_grados"] = boca.vel_ms, boca.dir_grados
        out["fuente_vel"] = out["fuente_dir"] = "era5"
    elif metodo == "era5_qm":
        out["vel_ms"], out["dir_grados"] = qm, boca.dir_grados
        out["fuente_vel"], out["fuente_dir"] = "era5_qm", "era5"
    elif metodo == "spso":
        o = obs.reindex(boca.index)
        hay_v = o.vel_ms.notna()
        out["vel_ms"] = o.vel_ms.where(hay_v, qm)
        out["fuente_vel"] = np.where(hay_v, "spso", "era5_qm")
        hay_d = o.dir_grados.notna() & hay_v
        out["dir_grados"] = o.dir_grados.where(hay_d, boca.dir_grados)
        out["fuente_dir"] = np.where(hay_d, "spso", "era5")
    else:
        raise SystemExit(f"forzamiento_viento.metodo desconocido: {metodo} (spso | era5_qm | era5)")
    out["u10_ms"], out["v10_ms"] = componentes(out.vel_ms, out.dir_grados)
    return out
