#!/usr/bin/env python
"""03 - Descarga de METAR del aeropuerto de Pisco (SPSO) desde IEM ASOS y control de calidad.

Servicio: Iowa Environmental Mesonet, https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py
(endpoint verificado: devuelve CSV con la opción format=onlycomma, horas en UTC con tz=Etc/UTC).

Salidas: data/raw/metar/SPSO_AAAA.csv (crudo) y data/processed/metar_spso.csv (horario, con QC).

Uso:
    python scripts/03_metar_download.py [--inicio 2025-01-01 --fin 2025-01-31]
"""
from __future__ import annotations

import io
import time

import numpy as np
import pandas as pd
import requests

from utils import sanidad
from utils.config import cargar_config, carpeta, parser_base, periodo, ruta
from utils.log import obtener_logger
from utils.viento import humedad_relativa

log = obtener_logger("03_metar_download")

NUDO_A_MS = 0.514444
CAMPOS = ["sknt", "drct", "tmpf", "dwpf", "relh", "mslp", "gust"]


def descargar_anio(cfg, anio: int, ini: pd.Timestamp, fin: pd.Timestamp) -> pd.DataFrame:
    """Descarga el tramo [ini, fin] del año indicado (una petición por año)."""
    a = max(ini, pd.Timestamp(f"{anio}-01-01"))
    b = min(fin + pd.Timedelta(hours=1), pd.Timestamp(f"{anio + 1}-01-01"))
    params = [("station", cfg["metar"]["estacion"])] + [("data", c) for c in CAMPOS] + [
        ("year1", a.year), ("month1", a.month), ("day1", a.day),
        ("year2", b.year), ("month2", b.month), ("day2", b.day),
        ("tz", "Etc/UTC"), ("format", "onlycomma"), ("latlon", "no"),
        ("missing", "M"), ("trace", "T"), ("direct", "no"),
    ]
    for intento in range(3):
        r = requests.get(cfg["metar"]["url"], params=params, timeout=120)
        if r.status_code == 200 and r.text.startswith("station"):
            break
        log.warning(f"Intento {intento + 1}: respuesta {r.status_code}; reintentando en 10 s")
        time.sleep(10)
    else:
        raise SystemExit(f"No se pudo descargar METAR {anio}: {r.status_code} {r.text[:200]}")
    df = pd.read_csv(io.StringIO(r.text), na_values=["M", "T"])
    df["valid"] = pd.to_datetime(df["valid"])
    return df


def control_calidad(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """QC básico: duplicados, valores imposibles, calmas y paso a serie horaria."""
    rep = {"registros_crudos": len(df)}
    df = df.sort_values("valid")
    dup = df.duplicated(subset="valid", keep="first")
    rep["duplicados"] = int(dup.sum())
    df = df[~dup].copy()

    # Unidades SI
    df["vel_ms"] = df.sknt * NUDO_A_MS
    df["rafaga_ms"] = df.gust * NUDO_A_MS
    df["dir_grados"] = df.drct.astype(float)
    df["t2m_C"] = (df.tmpf - 32) * 5 / 9
    df["td_C"] = (df.dwpf - 32) * 5 / 9

    # Valores imposibles -> NaN
    malos = {
        "vel_ms": (df.vel_ms < 0) | (df.vel_ms > 30),
        "dir_grados": (df.dir_grados < 0) | (df.dir_grados > 360),
        "t2m_C": (df.t2m_C < 0) | (df.t2m_C > 40),
        "td_C": (df.td_C < -10) | (df.td_C > df.t2m_C + 0.5),
    }
    for col, m in malos.items():
        rep[f"imposibles_{col}"] = int(m.sum())
        df.loc[m, col] = np.nan

    # Calmas: velocidad 0 -> dirección indefinida. Viento variable (VRB): sin dirección.
    calma = df.vel_ms == 0
    rep["calmas"] = int(calma.sum())
    df["calma"] = calma
    df.loc[calma, "dir_grados"] = np.nan
    df.loc[df.dir_grados == 360, "dir_grados"] = 0.0
    rep["sin_direccion_con_viento"] = int((df.vel_ms > 0).sum() - df.loc[df.vel_ms > 0, "dir_grados"].notna().sum())

    df["hr_pct"] = humedad_relativa(df.t2m_C + 273.15, df.td_C + 273.15)

    # Serie horaria: la observación más cercana a cada hora en punto (±20 min);
    # así se descartan los partes especiales (SPECI) intercalados
    df = df.set_index("valid")
    horas = pd.date_range(df.index.min().ceil("h"), df.index.max().floor("h"), freq="h")
    cols = ["vel_ms", "dir_grados", "rafaga_ms", "t2m_C", "td_C", "hr_pct", "mslp", "calma"]
    hor = df[cols].reindex(horas, method="nearest", tolerance=pd.Timedelta(minutes=20))
    hor.index.name = "tiempo_utc"
    hor = hor.rename(columns={"mslp": "msl_hPa"})
    rep["horas_periodo"] = len(hor)
    rep["horas_con_viento"] = int(hor.vel_ms.notna().sum())
    rep["horas_con_temperatura"] = int(hor.t2m_C.notna().sum())
    return hor, rep


def main():
    args = parser_base("Descarga METAR de SPSO (IEM ASOS) con control de calidad").parse_args()
    cfg = cargar_config(args.config)
    ini, fin = periodo(cfg, args)
    raw = ruta(cfg["metar"]["carpeta_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    procdir = carpeta(cfg, "carpeta_procesados")
    est = cfg["metar"]["estacion"]

    partes = []
    for anio in range(ini.year, fin.year + 1):
        destino = raw / f"{est}_{anio}_{ini:%Y%m%d}_{fin:%Y%m%d}.csv"
        if destino.exists() and not args.forzar:
            log.info(f"{anio}: ya existe {destino.name}, se usa el archivo local")
            df = pd.read_csv(destino, parse_dates=["valid"])
        else:
            log.info(f"{anio}: descargando {est} desde IEM ...")
            df = descargar_anio(cfg, anio, ini, fin)
            df.to_csv(destino, index=False)
            log.info(f"{anio}: {len(df)} registros -> {destino.name}")
        partes.append(df)
    df = pd.concat(partes, ignore_index=True)
    df = df[(df.valid >= ini) & (df.valid <= fin + pd.Timedelta(minutes=59))]

    hor, rep = control_calidad(df)
    log.info("Resumen del control de calidad:")
    for k, v in rep.items():
        log.info(f"  {k:28s}: {v}")
    log.info(f"  datos útiles de viento: {100 * rep['horas_con_viento'] / rep['horas_periodo']:.1f} % de las horas; "
             f"calmas: {100 * rep['calmas'] / max(rep['registros_crudos'], 1):.1f} % de los registros")

    sanidad.rango("METAR velocidad del viento (m/s)", hor.vel_ms, 0, 30, log)
    sanidad.rango("METAR temperatura (°C)", hor.t2m_C, 5, 40, log)

    hor.round(3).to_csv(procdir / "metar_spso.csv")
    log.info(f"Guardado data/processed/metar_spso.csv ({len(hor)} horas)")


if __name__ == "__main__":
    main()
