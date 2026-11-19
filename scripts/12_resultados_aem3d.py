#!/usr/bin/env python
"""12 - Resultados de la corrida de AEM3D (marea y viento) en la bahía de Paracas.

Lee las salidas horarias de superficie (run/ncfiles/sheet_top.nc) de la corrida de 11 (marea y viento,
agua homogénea) y evalúa:
  1. Marea en la celda del mareógrafo: AEM3D frente a FES2022 (lo que se impuso en las fronteras)
     y frente al IOC observado. El IOC de enero 2025 tiene el error de escala de Pisco: se corrige
     con el factor del QC (validacion_fes_mes_candidato.yaml, independiente de FES y de AEM3D).
  2. Corrientes de superficie dentro de la bahía (polígono data/poligono_bahia.geojson): tardes con
     evento Paracas frente a tardes sin evento (12–18 h locales), y la media de toda la corrida.
  3. Relación viento–corriente: viento del forzamiento frente a la corriente media en la bahía.

Orientación de las salidas (verificada con LATITUDE/LONGITUDE del NetCDF y el manual, sec. 2.2.2):
dimensiones (T, Y, X); X = filas, crece hacia el SUR; Y = columnas, crece hacia el ESTE.
U_VELOCITY es la componente en X (positiva hacia el sur) y V_VELOCITY en Y (positiva hacia el este):
  corriente hacia el este = V,  hacia el norte = −U.
Los tiempos del modelo están en la hora de aem3d.hora (local, UTC−5).

Salidas: data/processed/resultados_aem3d.yaml y figuras aem3d_*.png

Uso:
    python scripts/12_resultados_aem3d.py      (después de 11 --corrida)
"""
from __future__ import annotations

import json
import warnings

import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from shapely import contains_xy
from shapely.geometry import shape

from utils.config import RAIZ, cargar_config, carpeta, parser_base, ruta
from utils.era5 import eventos_paracas
from utils.graficos import guardar, plt
from utils.log import obtener_logger

log = obtener_logger("12_resultados_aem3d")
NAN_AEM3D = 1e3  # AEM3D marca tierra con -9.91e16


def limpiar(a):
    a = np.asarray(a, dtype=float)
    return np.where(np.abs(a) > NAN_AEM3D, np.nan, a)


def a_hora_modelo(indice_utc, cfg):
    if cfg["aem3d"]["hora"] == "local":
        return indice_utc + pd.Timedelta(hours=cfg["proyecto"]["zona_horaria_local"])
    return indice_utc


def leer_salidas(archivo):
    ds = xr.open_dataset(archivo, decode_times=False)
    t = pd.DatetimeIndex(pd.to_datetime(dict(year=ds.Year.values, month=ds.Month.values, day=ds.Day.values,
                                             hour=ds.Hour.values, minute=ds.Minute.values,
                                             second=ds.Second.values))).round("min")
    return {"t": t, "h": limpiar(ds.HEIGHT.values),
            "este": limpiar(ds.V_VELOCITY.values), "norte": -limpiar(ds.U_VELOCITY.values),
            "lat": limpiar(ds.LATITUDE.values), "lon": limpiar(ds.LONGITUDE.values),
            "prof": -limpiar(ds.BATH.values)}


def celda_cercana(s, lat, lon):
    d2 = (s["lat"] - lat) ** 2 + ((s["lon"] - lon) * np.cos(np.deg2rad(lat))) ** 2
    d2 = np.where(np.isnan(s["h"][0]), np.inf, d2)
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    return i, j, float(np.sqrt(d2[i, j]) * 111.0)


def metricas(a: pd.Series, b: pd.Series) -> dict:
    ok = a.notna() & b.notna()
    d = a[ok] - b[ok]
    return {"n": int(ok.sum()), "rmse_cm": round(100 * float(np.sqrt((d ** 2).mean())), 1),
            "sesgo_cm": round(100 * float(d.mean()), 1), "r": round(float(np.corrcoef(a[ok], b[ok])[0, 1]), 3),
            "razon_amplitud": round(float(a[ok].std() / b[ok].std()), 3)}


# ---------------------------------------------------------------------------
def main():
    args = parser_base("Resultados de la corrida de AEM3D").parse_args()
    cfg = cargar_config(args.config)
    proc, figdir = carpeta(cfg, "carpeta_procesados"), carpeta(cfg, "carpeta_figuras")
    caso = ruta(cfg["malla_aem3d"]["carpeta"])
    archivo = caso / "run" / "ncfiles" / "sheet_top.nc"
    if not archivo.exists():
        raise SystemExit(f"No existe {archivo}: corre antes python scripts/11_malla_aem3d.py --corrida")
    s = leer_salidas(archivo)
    t = s["t"]
    arranque = pd.Timedelta(hours=cfg["malla_aem3d"]["prueba"].get("arranque_h", 3))
    valido = t >= t[0] + arranque
    log.info(f"Salidas de AEM3D: {len(t)} horas ({t[0]:%Y-%m-%d %H:%M} a {t[-1]:%Y-%m-%d %H:%M}, "
             f"{cfg['aem3d']['hora']}); se descartan las primeras {arranque.total_seconds() / 3600:g} h de arranque")
    res = {"inicio": str(t[0]), "fin": str(t[-1]), "horas": len(t)}

    # --- 1. Marea en el mareógrafo ---------------------------------------------------------------
    mg = cfg["mareografo"]
    i, j, dist = celda_cercana(s, mg["lat"], mg["lon"])
    modelo = pd.Series(s["h"][:, i, j], index=t)
    fes = pd.read_csv(proc / "marea_frontera_mes_candidato.csv", index_col=0, parse_dates=True).mareografo
    fes.index = a_hora_modelo(fes.index, cfg)
    fes = fes.reindex(fes.index.union(t)).interpolate("time").reindex(t)
    ioc = pd.read_csv(proc / "marea_ioc_horaria_previa_qc.csv", index_col=0, parse_dates=True)[cfg["ioc"]["estaciones"][0]]
    ioc.index = a_hora_modelo(ioc.index, cfg)
    ioc = ioc.reindex(t)
    with open(proc / "validacion_fes_mes_candidato.yaml", encoding="utf-8") as f:
        factor = float(yaml.safe_load(f).get("factor_escala", 1.0))
    # Nivel respecto de la media del periodo (el IOC tiene su propio cero; el modelo, el nivel medio)
    ioc_c = (ioc - ioc[valido].mean()) * factor
    mod_c = modelo - modelo[valido].mean()
    m_fes = metricas(modelo[valido], fes[valido])
    m_ioc = metricas(mod_c[valido], ioc_c[valido])
    m_ioc_fes = metricas((fes - fes[valido].mean())[valido], ioc_c[valido])
    log.info(f"Mareógrafo (celda a {dist:.2f} km): AEM3D − FES2022: RMSE {m_fes['rmse_cm']} cm, r {m_fes['r']}")
    log.info(f"AEM3D frente al IOC observado (×{factor:.3f}): RMSE {m_ioc['rmse_cm']} cm, r {m_ioc['r']}, "
             f"razón de amplitud {m_ioc['razon_amplitud']} ({m_ioc['n']} h); FES frente al IOC: RMSE "
             f"{m_ioc_fes['rmse_cm']} cm")
    if m_fes["rmse_cm"] > 5:
        log.warning("[sanidad] AEM3D se aleja de FES2022 en el mareógrafo (> 5 cm): revisar fronteras")
    else:
        log.info("[sanidad] AEM3D transmite la marea de las fronteras al interior de la bahía -> OK")
    res["mareografo"] = {"distancia_celda_km": round(dist, 2), "factor_ioc": factor,
                         "aem3d_vs_fes": m_fes, "aem3d_vs_ioc": m_ioc, "fes_vs_ioc": m_ioc_fes}
    fig_marea(t, mod_c, fes - fes[valido].mean(), ioc_c, m_fes, m_ioc, factor, figdir)

    # --- 2. Corrientes en la bahía ----------------------------------------------------------------
    with open(ruta(cfg["batimetria"]["poligono_bahia"]), encoding="utf-8") as f:
        gj = json.load(f)
    pol = shape((gj["features"][0] if gj.get("type") == "FeatureCollection" else gj)["geometry"])
    lat_ok = np.nan_to_num(s["lat"], nan=0.0)
    lon_ok = np.nan_to_num(s["lon"], nan=0.0)
    bahia = contains_xy(pol, lon_ok, lat_ok) & ~np.isnan(s["h"][0])
    vel = np.hypot(s["este"], s["norte"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v_bahia = pd.Series(np.nanmean(np.where(bahia, vel, np.nan).reshape(len(t), -1), axis=1), index=t)
    log.info(f"Bahía: {int(bahia.sum())} celdas de superficie dentro del polígono")

    met = pd.read_csv(proc / "forzamiento_meteo_mes_candidato.csv", index_col=0, parse_dates=True)
    met.index = a_hora_modelo(met.index, cfg)  # mismo reloj que el modelo
    viento = met.vel_ms.reindex(t, method="nearest", tolerance=pd.Timedelta("31min"))
    # Eventos: el criterio de 04 trabaja en UTC y convierte a hora local; aquí se pasa el viento en UTC
    v_utc = met.vel_ms.copy()
    v_utc.index = v_utc.index - pd.Timedelta(hours=cfg["proyecto"]["zona_horaria_local"])
    dias, _ = eventos_paracas(v_utc, cfg)
    dias_ev = set(pd.DatetimeIndex(dias.index[dias.evento]).date)
    vp = cfg["vientos_paracas"]
    tarde = (t.hour >= vp["hora_local_inicio"]) & (t.hour <= vp["hora_local_fin"]) & valido
    en_evento = np.array([d in dias_ev for d in t.date])
    sel_ev, sel_calma = tarde & en_evento, tarde & ~en_evento
    n_ev_dias = len({d for d, e in zip(t.date, sel_ev) if e})
    log.info(f"Días con evento Paracas dentro de la corrida: {n_ev_dias} "
             f"({sorted(str(d) for d in dias_ev if t[0].date() <= d <= t[-1].date())})")

    def media_campo(sel):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.nanmean(s["este"][sel], axis=0), np.nanmean(s["norte"][sel], axis=0)

    campos = {n: media_campo(sel) for n, sel in (("tardes con evento", sel_ev), ("tardes sin evento", sel_calma),
                                                  ("media de la corrida", valido)) if sel.any()}
    for n, sel in (("tardes con evento", sel_ev), ("tardes sin evento", sel_calma)):
        if not sel.any():
            log.warning(f"No hay {n} en la corrida: no se calcula ese campo medio")
    resumen_c = {}
    for nombre, (ue, vn) in campos.items():
        rap = np.hypot(ue, vn)[bahia]
        dirm = (np.degrees(np.arctan2(np.nanmean(ue[bahia]), np.nanmean(vn[bahia]))) + 360) % 360
        resumen_c[nombre] = {"vel_media_cms": round(100 * float(np.nanmean(rap)), 2),
                             "vel_p90_cms": round(100 * float(np.nanpercentile(rap, 90)), 2),
                             "hacia_grados": round(float(dirm), 0)}
        log.info(f"Corriente de superficie media en la bahía, {nombre}: {resumen_c[nombre]['vel_media_cms']} cm/s "
                 f"(p90 {resumen_c[nombre]['vel_p90_cms']}), hacia {resumen_c[nombre]['hacia_grados']:.0f}°")
    inst = {n: float(v_bahia[sel].mean()) for n, sel in (("tardes con evento", sel_ev),
                                                         ("tardes sin evento", sel_calma)) if sel.any()}
    log.info("Rapidez instantánea media en la bahía: "
             + ", ".join(f"{n} {100 * v:.1f} cm/s" for n, v in inst.items()))
    res["corrientes_bahia"] = resumen_c
    res["rapidez_instantanea_bahia_cms"] = {k: round(100 * float(v), 2) for k, v in inst.items()}

    # Corriente promediada en la vertical (sheet_avg): si el viento mueve solo una capa delgada de superficie,
    # el promedio vertical es mucho menor que la corriente de superficie (con flujo de retorno abajo).
    arch_avg = caso / "run" / "ncfiles" / "sheet_avg.nc"
    if arch_avg.exists():
        sa = leer_salidas(arch_avg)
        prom = {}
        for nombre, sel in (("tardes con evento", sel_ev), ("tardes sin evento", sel_calma)):
            if sel.any():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ue, vn = np.nanmean(sa["este"][sel], axis=0), np.nanmean(sa["norte"][sel], axis=0)
                prom[nombre] = round(100 * float(np.nanmean(np.hypot(ue, vn)[bahia])), 2)
        log.info("Corriente media promediada en la vertical en la bahía: "
                 + ", ".join(f"{n} {v} cm/s" for n, v in prom.items()))
        res["corriente_vertical_media_bahia_cms"] = prom

    # --- 3. Viento y corriente ------------------------------------------------------------------
    ok = valido & viento.notna().values & v_bahia.notna().values
    r0 = float(np.corrcoef(viento[ok], v_bahia[ok])[0, 1])
    rezagos = {k: float(np.corrcoef(viento.shift(k)[ok & viento.shift(k).notna().values],
                                    v_bahia[ok & viento.shift(k).notna().values])[0, 1]) for k in range(0, 7)}
    k_max = max(rezagos, key=rezagos.get)
    log.info(f"Viento (forzamiento) frente a rapidez media en la bahía: r = {r0:.2f} sin rezago; máximo "
             f"r = {rezagos[k_max]:.2f} con {k_max} h de rezago")
    res["viento_corriente"] = {"r_sin_rezago": round(r0, 3), "rezago_h": k_max, "r_max": round(rezagos[k_max], 3)}
    vmax = float(np.nanmax(vel[valido]))
    res["vel_max_superficie_cms"] = round(100 * vmax, 1)
    log.info(f"Velocidad superficial máxima en todo el dominio: {100 * vmax:.1f} cm/s")

    if {"tardes con evento", "tardes sin evento"} <= set(campos):
        fig_corrientes(s, campos, bahia, pol, figdir)
    fig_viento(t, viento, v_bahia, sel_ev, valido, res["viento_corriente"], figdir)
    with open(proc / "resultados_aem3d.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(res, f, allow_unicode=True, sort_keys=False)
    log.info("Guardado: data/processed/resultados_aem3d.yaml")


# ---------------------------------------------------------------------------
def ejes_fecha(ax):
    """Una marca cada dos días, con formato corto, para que no se encimen las fechas."""
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    meses = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: (lambda d: f"{d.day} {meses[d.month - 1]}")(mdates.num2date(x))))


def fig_marea(t, mod, fes, ioc, m_fes, m_ioc, factor, figdir):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(ioc.index, ioc, ".", color="0.5", ms=3, label=f"IOC observado ×{factor:.3f} (error de escala corregido)")
    ax.plot(fes.index, fes, color="tab:orange", lw=1.2, label="FES2022 (impuesto en las fronteras)")
    ax.plot(mod.index, mod, color="tab:blue", lw=1.0, label="AEM3D en la celda del mareógrafo")
    ax.set_ylabel("Nivel respecto de la media (m)")
    ax.set_xlabel("Fecha (hora local)")
    ax.set_title(f"Marea en el mareógrafo de Pisco: AEM3D − FES RMSE {m_fes['rmse_cm']} cm; "
                 f"AEM3D − IOC RMSE {m_ioc['rmse_cm']} cm (r = {m_ioc['r']})")
    ax.legend(fontsize=8, loc="upper right", ncol=3)
    ejes_fecha(ax)
    guardar(fig, figdir, "aem3d_marea_mareografo", log)


def fig_corrientes(s, campos, bahia, pol, figdir):
    lat, lon = s["lat"], s["lon"]
    agua = ~np.isnan(s["h"][0])
    # Recorte alrededor de la bahía (con margen) para que se vean las flechas
    x0, y0, x1, y1 = pol.bounds
    m = 0.06
    zona = (lon >= x0 - m) & (lon <= x1 + m) & (lat >= y0 - m) & (lat <= y1 + m) & agua
    nombres = ["tardes con evento", "tardes sin evento"]
    rap = [np.hypot(*campos[n]) for n in nombres]
    vmax = float(np.nanpercentile(np.concatenate([r[zona] for r in rap]), 98))
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    for ax, n, r in zip(axs, nombres, rap):
        ue, vn = campos[n]
        im = ax.scatter(lon[zona], lat[zona], c=100 * r[zona], s=14, marker="s", cmap="viridis",
                        vmin=0, vmax=100 * vmax)
        sub = zona & (np.add.outer(np.arange(lat.shape[0]), np.arange(lat.shape[1])) % 3 == 0)
        ax.quiver(lon[sub], lat[sub], ue[sub], vn[sub], scale=vmax * 25, width=0.003, color="k")
        xs, ys = pol.exterior.xy
        ax.plot(xs, ys, "r--", lw=1, label="Polígono de la bahía")
        ax.set_title(n.capitalize())
        ax.xaxis.set_major_locator(mticker.MaxNLocator(4))
        ax.set_xlabel("Longitud (°)")
        ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(lat[zona]))))
    axs[0].set_ylabel("Latitud (°)")
    axs[0].legend(fontsize=8, loc="lower left")
    fig.colorbar(im, ax=axs, label="Rapidez de la corriente media (cm/s)", shrink=0.85)
    fig.suptitle("AEM3D (marea y viento): corriente de superficie media en la bahía de Paracas,\n"
                 "tardes de 12 a 18 h con y sin evento Paracas",
                 fontweight="bold")
    guardar(fig, figdir, "aem3d_corrientes_eventos", log)


def fig_viento(t, viento, v_bahia, sel_ev, valido, vc, figdir):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax2 = ax.twinx()
    ax.plot(t, viento, color="tab:gray", lw=0.8, label="Viento del forzamiento (SPSO)")
    ax2.plot(t, 100 * v_bahia, color="tab:blue", lw=1.0, label="Rapidez media de la corriente en la bahía")
    ax.fill_between(t, 0, viento.max(), where=sel_ev, color="crimson", alpha=0.12, label="Tarde con evento Paracas")
    ax.axvspan(t[0], t[~valido][-1] if (~valido).any() else t[0], color="0.85", label="Arranque (descartado)")
    ax.set_ylabel("Viento (m/s)")
    ax2.set_ylabel("Corriente de superficie (cm/s)")
    ax.set_xlabel("Fecha (hora local)")
    ax.set_title(f"Viento y corriente en la bahía (r = {vc['r_max']:.2f} con {vc['rezago_h']} h de rezago)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left", ncol=2)
    ejes_fecha(ax)
    guardar(fig, figdir, "aem3d_viento_corriente", log)


if __name__ == "__main__":
    main()
