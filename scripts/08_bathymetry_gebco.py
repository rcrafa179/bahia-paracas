#!/usr/bin/env python
"""08 - Batimetría de la bahía de Paracas a partir de GEBCO.

Entrada: recorte NetCDF de la malla GEBCO descargado a mano desde
https://download.gebco.net (variable de elevación en m, negativa en el mar).
La ruta va en config.yaml (batimetria.gebco_nc).

Hace:
  1. Recorte a la caja del dominio y mapa batimétrico (cartopy) con costa e isóbatas.
  2. Perfiles de profundidad en la boca y a lo largo de la bahía.
  3. Área, volumen y profundidad media dentro del polígono data/poligono_bahia.geojson.
  4. fusionar_sondajes_dhn(): fusión de sondajes digitalizados de la carta DHN con GEBCO
     (lista para usar cuando exista el CSV lon, lat, prof).

Uso:
    python scripts/08_bathymetry_gebco.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import xarray as xr
from shapely import contains_xy
from shapely.geometry import shape

from utils.config import cargar_config, carpeta, parser_base, ruta
from utils.graficos import guardar, plt
from utils.log import obtener_logger

log = obtener_logger("08_bathymetry")

RADIO_TIERRA = 6371000.0  # m


# ---------------------------------------------------------------------------
# Lectura
# ---------------------------------------------------------------------------
def leer_gebco(cfg) -> xr.DataArray:
    """Lee GEBCO y devuelve la PROFUNDIDAD (m, positiva en el mar, NaN en tierra) recortada a la caja."""
    archivo = ruta(cfg["batimetria"]["gebco_nc"])
    if not archivo.exists():
        raise SystemExit(f"No existe {archivo}. Descarga el recorte de GEBCO (ver README) y corrige "
                         "'batimetria.gebco_nc' en config.yaml.")
    ds = xr.open_dataset(archivo)
    # Nombre de la variable: 'elevation' en las descargas actuales de GEBCO; se admite 'z' por compatibilidad
    var = next((v for v in ("elevation", "z", "Band1") if v in ds.data_vars), None)
    if var is None:
        raise SystemExit(f"No se encontró la variable de elevación en {archivo.name}: {list(ds.data_vars)}")
    elev = ds[var]
    ren = {k: v for k, v in (("latitude", "lat"), ("longitude", "lon"), ("y", "lat"), ("x", "lon")) if k in elev.dims}
    elev = elev.rename(ren).sortby("lat").sortby("lon")
    d = cfg["dominio"]
    elev = elev.sel(lat=slice(d["lat_min"], d["lat_max"]), lon=slice(d["lon_min"], d["lon_max"])).astype(float)
    if elev.size == 0:
        raise SystemExit("El archivo GEBCO no cubre el dominio de config.yaml")
    res = float(np.abs(np.diff(elev.lon.values)).mean()) * 3600
    log.info(f"GEBCO '{var}' leído de {archivo.name}: {elev.sizes['lat']}x{elev.sizes['lon']} celdas, "
             f"resolución ≈ {res:.0f}\" ({res / 3600 * 111:.2f} km); elevación {float(elev.min()):.0f} a "
             f"{float(elev.max()):.0f} m")
    prof = (-elev).where(elev < 0)
    prof.name = "profundidad"
    prof.attrs = {"units": "m", "long_name": "Profundidad (positiva hacia abajo)"}
    if float(prof.max()) > 1000:
        log.warning(f"[sanidad] profundidad máxima {float(prof.max()):.0f} m: inesperada en la plataforma frente a Paracas")
    return prof, elev


def area_celdas(da: xr.DataArray) -> xr.DataArray:
    """Área (m²) de cada celda de una malla regular lat-lon sobre la esfera."""
    dlat = np.radians(float(np.abs(np.diff(da.lat.values)).mean()))
    dlon = np.radians(float(np.abs(np.diff(da.lon.values)).mean()))
    a = RADIO_TIERRA ** 2 * dlat * dlon * np.cos(np.radians(da.lat))
    return a.broadcast_like(da)


def leer_poligono(cfg):
    with open(ruta(cfg["batimetria"]["poligono_bahia"]), encoding="utf-8") as f:
        gj = json.load(f)
    feat = gj["features"][0] if gj.get("type") == "FeatureCollection" else gj
    return shape(feat["geometry"]), feat.get("properties", {}).get("nombre", "Bahía")


def mascara_poligono(da: xr.DataArray, pol) -> xr.DataArray:
    LON, LAT = np.meshgrid(da.lon.values, da.lat.values)
    return xr.DataArray(contains_xy(pol, LON, LAT), coords={"lat": da.lat, "lon": da.lon}, dims=("lat", "lon"))


# ---------------------------------------------------------------------------
# Cálculos
# ---------------------------------------------------------------------------
def morfometria(prof: xr.DataArray, pol, nombre: str) -> dict:
    dentro = mascara_poligono(prof, pol)
    agua = dentro & prof.notnull()
    A = area_celdas(prof)
    area = float(A.where(agua).sum())
    vol = float((prof * A).where(agua).sum())
    r = {"cuerpo": nombre, "celdas_agua": int(agua.sum()), "area_km2": area / 1e6, "volumen_hm3": vol / 1e6,
         "profundidad_media_m": vol / area if area > 0 else np.nan,
         "profundidad_maxima_m": float(prof.where(agua).max()),
         "area_poligono_km2": float(A.where(dentro).sum()) / 1e6}
    log.info(f"{nombre}: área de agua = {r['area_km2']:.2f} km², volumen = {r['volumen_hm3']:.1f} hm³ "
             f"({r['volumen_hm3'] / 1e3:.3f} km³), profundidad media = {r['profundidad_media_m']:.1f} m, "
             f"máxima = {r['profundidad_maxima_m']:.1f} m ({r['celdas_agua']} celdas)")
    if r["celdas_agua"] < 20:
        log.warning("[sanidad] muy pocas celdas de agua dentro del polígono: revisar el GeoJSON o la resolución")
    if not (1 <= r["profundidad_media_m"] <= 50):
        log.warning("[sanidad] profundidad media fuera de lo esperado para una bahía somera (1–50 m): REVISAR")
    else:
        log.info("[sanidad] profundidad media razonable para la bahía -> OK")
    return r


def perfil(prof: xr.DataArray, seg: dict) -> pd.DataFrame:
    lats = np.linspace(seg["lat_ini"], seg["lat_fin"], seg["n"])
    lons = np.linspace(seg["lon_ini"], seg["lon_fin"], seg["n"])
    z = prof.interp(lat=xr.DataArray(lats, dims="p"), lon=xr.DataArray(lons, dims="p")).values
    dx = np.radians(np.diff(lons)) * np.cos(np.radians(lats[:-1])) * RADIO_TIERRA
    dy = np.radians(np.diff(lats)) * RADIO_TIERRA
    dist = np.concatenate([[0], np.cumsum(np.hypot(dx, dy))]) / 1000
    return pd.DataFrame({"distancia_km": dist, "lat": lats, "lon": lons, "profundidad_m": z})


def fusionar_sondajes_dhn(prof: xr.DataArray, csv_sondajes, poligono, ancho_transicion_m: float = 500.0,
                          metodo: str = "linear") -> xr.DataArray:
    """Fusiona sondajes digitalizados de la carta DHN con la malla GEBCO.

    Prioriza la carta dentro del polígono de la bahía y hace una transición lineal
    hacia GEBCO en una franja de 'ancho_transicion_m' metros fuera del borde.

    Parámetros
    ----------
    prof : DataArray
        Profundidad GEBCO (m, positiva, NaN en tierra) con coordenadas lat/lon.
    csv_sondajes : ruta a un CSV con columnas lon, lat, prof (prof en m, positiva;
        referida al mismo nivel que se quiera usar; la carta DHN suele referirse al
        nivel de reducción de sondajes, no al nivel medio: corregir antes si hace falta).
    poligono : shapely Polygon de la bahía.
    ancho_transicion_m : ancho de la franja de mezcla fuera del polígono.
    metodo : 'linear' o 'cubic' (scipy.interpolate.griddata).

    Devuelve
    --------
    DataArray con la profundidad fusionada (misma malla que 'prof').
    """
    from scipy.interpolate import griddata

    s = pd.read_csv(csv_sondajes)
    faltan = {"lon", "lat", "prof"} - set(s.columns)
    if faltan:
        raise ValueError(f"El CSV de sondajes debe tener columnas lon, lat, prof; faltan {faltan}")
    s = s.dropna(subset=["lon", "lat", "prof"])
    log.info(f"Sondajes DHN: {len(s)} puntos, profundidad {s.prof.min():.1f} a {s.prof.max():.1f} m")

    LON, LAT = np.meshgrid(prof.lon.values, prof.lat.values)
    carta = griddata((s.lon, s.lat), s.prof, (LON, LAT), method=metodo)

    # Peso de la carta: 1 dentro del polígono, decae linealmente a 0 en la franja de transición
    dentro = contains_xy(poligono, LON, LAT)
    borde = poligono.boundary
    from shapely import distance, points

    dist_grados = distance(points(LON.ravel(), LAT.ravel()), borde).reshape(LON.shape)
    dist_m = dist_grados * 111_000.0
    peso = np.where(dentro, 1.0, np.clip(1 - dist_m / ancho_transicion_m, 0, 1))
    peso = np.where(np.isfinite(carta), peso, 0.0)  # fuera del alcance de los sondajes: GEBCO

    g = prof.values
    fusion = np.where(np.isfinite(g), (1 - peso) * g + peso * np.nan_to_num(carta), np.nan)
    # Celdas que GEBCO marca como tierra pero la carta tiene agua dentro de la bahía
    nueva_agua = ~np.isfinite(g) & dentro & np.isfinite(carta) & (np.nan_to_num(carta) > 0)
    fusion = np.where(nueva_agua, carta, fusion)
    log.info(f"Fusión: {int((peso > 0).sum())} celdas con aporte de la carta "
             f"({int(nueva_agua.sum())} celdas nuevas de agua)")
    out = prof.copy(data=fusion)
    out.attrs["fuente"] = "GEBCO + sondajes carta DHN"
    return out


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_mapa(prof, elev, pol, cfg, figdir, nombre="batimetria_mapa", titulo="Batimetría de la bahía de Paracas (GEBCO)"):
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER

    d = cfg["dominio"]
    fig = plt.figure(figsize=(9, 9))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent([d["lon_min"], d["lon_max"], d["lat_min"], d["lat_max"]], crs=ccrs.PlateCarree())
    vmax = float(np.nanpercentile(prof, 99))
    pc = ax.pcolormesh(prof.lon, prof.lat, prof, cmap="viridis_r", vmin=0, vmax=vmax, shading="auto",
                       transform=ccrs.PlateCarree())
    ax.contourf(elev.lon, elev.lat, elev.where(elev >= 0), levels=[0, 1e5], colors=["#e8dcc0"],
                transform=ccrs.PlateCarree())
    iso = [z for z in cfg["batimetria"]["isobatas"] if z < float(prof.max())]
    cs = ax.contour(prof.lon, prof.lat, prof, levels=iso, colors="k", linewidths=0.5, transform=ccrs.PlateCarree())
    ax.clabel(cs, fmt="%d m", fontsize=7)
    # Línea de costa: contorno de 0 m de GEBCO (coherente con la malla)
    ax.contour(elev.lon, elev.lat, elev, levels=[0], colors="saddlebrown", linewidths=1.2, transform=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature("physical", "coastline", "10m"), edgecolor="0.4",
                   facecolor="none", linewidth=0.6, linestyle=":")
    x, y = pol.exterior.xy
    ax.plot(x, y, "r--", lw=1.5, transform=ccrs.PlateCarree(), label="Polígono de la bahía")
    for nombre_p, seg in cfg["batimetria"]["perfiles"].items():
        ax.plot([seg["lon_ini"], seg["lon_fin"]], [seg["lat_ini"], seg["lat_fin"]], "-", lw=2,
                transform=ccrs.PlateCarree(), label=f"Perfil: {nombre_p.replace('_', ' ')}")
    mg, me = cfg["mareografo"], cfg["metar"]
    ax.plot(mg["lon"], mg["lat"], "w^", mec="k", ms=9, transform=ccrs.PlateCarree(), label="Mareógrafo Pisco")
    ax.plot(me["lon"], me["lat"], "ws", mec="k", ms=8, transform=ccrs.PlateCarree(), label="Aeropuerto SPSO")
    b = cfg["boca_bahia"]
    ax.plot(b["lon"], b["lat"], "o", color="orange", mec="k", ms=7, transform=ccrs.PlateCarree(),
            label="Boca (punto de referencia)")
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="0.5")
    gl.top_labels = gl.right_labels = False
    gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    cb = plt.colorbar(pc, ax=ax, shrink=0.75, pad=0.03)
    cb.set_label("Profundidad (m)")
    ax.legend(loc="lower left", fontsize=7, framealpha=0.9)
    ax.set_title(titulo)
    guardar(fig, figdir, nombre, log)


def fig_perfiles(perfiles: dict, figdir):
    fig, axs = plt.subplots(len(perfiles), 1, figsize=(9, 3 * len(perfiles)))
    axs = np.atleast_1d(axs)
    for ax, (nombre, p) in zip(axs, perfiles.items()):
        ax.fill_between(p.distancia_km, p.profundidad_m, 0, color="steelblue", alpha=0.5)
        ax.plot(p.distancia_km, p.profundidad_m, "k-", lw=1)
        ax.invert_yaxis()
        ax.set_ylabel("Profundidad (m)")
        ax.set_title(f"Perfil: {nombre.replace('_', ' ')} "
                     f"({p.lat.iloc[0]:.3f}, {p.lon.iloc[0]:.3f}) → ({p.lat.iloc[-1]:.3f}, {p.lon.iloc[-1]:.3f})")
        if p.profundidad_m.notna().any():
            ax.text(0.99, 0.08, f"Prof. máx. {p.profundidad_m.max():.1f} m · media {p.profundidad_m.mean():.1f} m",
                    transform=ax.transAxes, ha="right", fontsize=8)
    axs[-1].set_xlabel("Distancia a lo largo del perfil (km)")
    fig.suptitle("Perfiles batimétricos (GEBCO)", fontweight="bold")
    fig.tight_layout()
    guardar(fig, figdir, "batimetria_perfiles", log)


def fig_hipsometria(prof, pol, figdir):
    """Curva hipsométrica de la bahía: área con profundidad mayor que z."""
    dentro = mascara_poligono(prof, pol) & prof.notnull()
    A = area_celdas(prof).where(dentro).values.ravel()
    z = prof.where(dentro).values.ravel()
    ok = np.isfinite(z)
    orden = np.argsort(z[ok])
    zs, As = z[ok][orden], A[ok][orden]
    area_mayor = (As.sum() - np.cumsum(As) + As) / 1e6
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(area_mayor, zs, "b-")
    ax.invert_yaxis()
    ax.set_xlabel("Área con profundidad ≥ z (km²)")
    ax.set_ylabel("Profundidad z (m)")
    ax.set_title("Curva hipsométrica de la bahía de Paracas")
    guardar(fig, figdir, "batimetria_hipsometria", log)


# ---------------------------------------------------------------------------
def main():
    p = parser_base("Batimetría de la bahía de Paracas (GEBCO)")
    p.add_argument("--sondajes", default=None, help="CSV de sondajes DHN (lon, lat, prof) para fusionar con GEBCO")
    args = p.parse_args()
    cfg = cargar_config(args.config)
    figdir = carpeta(cfg, "carpeta_figuras")
    procdir = carpeta(cfg, "carpeta_procesados")

    prof, elev = leer_gebco(cfg)
    pol, nombre = leer_poligono(cfg)
    prof.to_netcdf(procdir / "batimetria_gebco_paracas.nc")

    resultados = [morfometria(prof, pol, f"{nombre} (GEBCO)")]
    perfiles = {n: perfil(prof, seg) for n, seg in cfg["batimetria"]["perfiles"].items()}
    for n, df in perfiles.items():
        df.round(5).to_csv(procdir / f"perfil_{n}.csv", index=False)
        log.info(f"Perfil {n}: longitud {df.distancia_km.iloc[-1]:.2f} km, "
                 f"profundidad máxima {df.profundidad_m.max():.1f} m")

    fig_mapa(prof, elev, pol, cfg, figdir)
    fig_perfiles(perfiles, figdir)
    fig_hipsometria(prof, pol, figdir)

    if args.sondajes:
        fus = fusionar_sondajes_dhn(prof, ruta(args.sondajes), pol)
        fus.to_netcdf(procdir / "batimetria_fusionada_paracas.nc")
        resultados.append(morfometria(fus, pol, f"{nombre} (GEBCO + carta DHN)"))
        fig_mapa(fus, elev, pol, cfg, figdir, "batimetria_mapa_fusionada",
                 "Batimetría de la bahía de Paracas (GEBCO + carta DHN)")

    pd.DataFrame(resultados).round(3).to_csv(procdir / "morfometria_bahia.csv", index=False)
    log.info("Guardados: batimetria_gebco_paracas.nc, morfometria_bahia.csv y perfiles")


if __name__ == "__main__":
    main()
