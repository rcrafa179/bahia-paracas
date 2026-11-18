"""Lectura de ERA5, variables derivadas y detección de vientos Paracas (compartido por 02 y 04)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from utils.config import ruta
from utils.log import obtener_logger
from utils.viento import a_hora_local, humedad_relativa, velocidad_direccion

log = obtener_logger("era5")


def leer_era5(cfg, ini, fin) -> xr.Dataset:
    carpeta_raw = ruta(cfg["era5"]["carpeta_raw"])
    meses = pd.period_range(ini, fin, freq="M")
    archivos = [carpeta_raw / f"era5_{p.year}{p.month:02d}.nc" for p in meses]
    faltan = [a.name for a in archivos if not a.exists()]
    if faltan:
        raise SystemExit(f"Faltan archivos ERA5 (correr 01 antes): {faltan}")
    ds = xr.concat([xr.open_dataset(a) for a in archivos], dim="time").sortby("time")
    ds = ds.sel(time=slice(ini, fin))
    log.info(f"ERA5 leído: {ds.sizes['time']} horas ({str(ds.time.values[0])[:13]} a {str(ds.time.values[-1])[:13]} UTC)")
    return ds


def derivar(ds: xr.Dataset) -> xr.Dataset:
    """Calcula viento, HR y convierte unidades sobre toda la malla."""
    out = xr.Dataset(coords=ds.coords)
    vel, dirm = velocidad_direccion(ds.u10, ds.v10)
    out["u10"], out["v10"] = ds.u10, ds.v10
    out["vel"] = (ds.u10.dims, vel)
    out["dir"] = (ds.u10.dims, dirm)
    out["t2m"] = ds.t2m - 273.15
    out["hr"] = (ds.t2m.dims, humedad_relativa(ds.t2m, ds.d2m))
    out["msl"] = ds.msl / 100.0
    # Acumulados horarios: J/m2 -> W/m2 (promedio de la hora que termina en 'time')
    out["ssrd"] = ds.ssrd / 3600.0
    out["strd"] = ds.strd / 3600.0
    out["tp"] = ds.tp * 1000.0  # m -> mm
    out["tcc"] = ds.tcc
    # ERA5 puede traer valores negativos minúsculos por empaquetado: se anulan
    for v in ("ssrd", "strd", "tp"):
        out[v] = out[v].where(out[v] > -1.0).clip(min=0)
    return out


def a_tabla(ds: xr.Dataset) -> pd.DataFrame:
    """Dataset 1D (solo tiempo) -> DataFrame con nombres y unidades en las columnas."""
    nombres = {"u10": "u10_ms", "v10": "v10_ms", "vel": "vel_ms", "dir": "dir_grados", "t2m": "t2m_C",
               "hr": "hr_pct", "msl": "msl_hPa", "ssrd": "ssrd_Wm2", "strd": "strd_Wm2",
               "tp": "tp_mm", "tcc": "tcc_frac"}
    df = ds[list(nombres)].to_dataframe()[list(nombres)].rename(columns=nombres)
    df.index.name = "tiempo_utc"
    return df


def punto_cercano(ds, lat, lon, etiqueta="la boca"):
    p = ds.sel(latitude=lat, longitude=lon, method="nearest")
    dist_km = np.hypot((float(p.latitude) - lat) * 111.0,
                       (float(p.longitude) - lon) * 111.0 * np.cos(np.radians(lat)))
    log.info(f"Punto de malla más cercano a {etiqueta} ({lat}, {lon}): "
             f"({float(p.latitude):.2f}, {float(p.longitude):.2f}), a {dist_km:.1f} km")
    return p



def eventos_paracas(vel: pd.Series, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Días (hora local) con viento > umbral dentro de la ventana de la tarde.

    Devuelve (tabla diaria de eventos, conteo mensual).
    """
    vp = cfg["vientos_paracas"]
    tz = cfg["proyecto"]["zona_horaria_local"]
    s = vel.copy()
    s.index = a_hora_local(s.index, tz)
    tarde = s[(s.index.hour >= vp["hora_local_inicio"]) & (s.index.hour <= vp["hora_local_fin"])]
    por_dia = pd.DataFrame({
        "horas_sobre_umbral": (tarde > vp["umbral_ms"]).groupby(tarde.index.date).sum(),
        "vel_max_tarde_ms": tarde.groupby(tarde.index.date).max(),
    })
    por_dia.index = pd.to_datetime(por_dia.index)
    por_dia.index.name = "fecha_local"
    por_dia["evento"] = por_dia.horas_sobre_umbral >= vp["horas_minimas"]
    mensual = por_dia.groupby(por_dia.index.to_period("M")).agg(
        eventos=("evento", "sum"), dias=("evento", "size"),
        vel_max_tarde_ms=("vel_max_tarde_ms", "max"),
        p95_tarde_ms=("vel_max_tarde_ms", lambda x: np.nanpercentile(x, 95)),
    )
    mensual.index.name = "mes"
    return por_dia, mensual


def elegir_mes_candidato(mensual: pd.DataFrame, fuente: str, mes_fijo="auto") -> dict:
    """Elige el mes candidato para la simulación.

    mes_fijo = "auto": el mes con más eventos (desempate por el p95 del viento de la tarde).
    mes_fijo = "AAAA-MM": se usa ese mes (vientos_paracas.mes_candidato en config.yaml).
    """
    if mes_fijo is not None and str(mes_fijo).lower() != "auto":
        per = pd.Period(str(mes_fijo), freq="M")
        if per not in mensual.index:
            raise SystemExit(f"vientos_paracas.mes_candidato = {mes_fijo} está fuera del periodo analizado")
        orden = mensual.loc[[per]]
        criterio = "fijado en config.yaml (vientos_paracas.mes_candidato)"
    elif mensual.eventos.max() > 0:
        # Más eventos; en empate, el de mayor percentil 95 de viento vespertino
        orden = mensual.sort_values(["eventos", "p95_tarde_ms"], ascending=False)
        criterio = "máximo número de eventos (desempate: p95 del viento de la tarde)"
    else:
        orden = mensual.sort_values("p95_tarde_ms", ascending=False)
        criterio = "sin eventos sobre el umbral: mes con mayor p95 del viento de la tarde"
        log.warning("No hubo ningún evento sobre el umbral; se usa el mes con viento vespertino más intenso")
    mes = orden.index[0]
    ini = mes.start_time
    fin = mes.end_time.floor("h")
    return {"mes": str(mes), "inicio_utc": str(ini), "fin_utc": str(fin),
            "eventos": int(orden.eventos.iloc[0]), "p95_tarde_ms": round(float(orden.p95_tarde_ms.iloc[0]), 2),
            "fuente": fuente, "criterio": criterio}
