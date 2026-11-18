#!/usr/bin/env python
"""07 - Marea del modelo global FES2022 con pyTMD: armónicos y forzamiento de frontera.

Los archivos de FES2022b se descargan a mano desde AVISO (requiere registro) y su
carpeta se indica en config.yaml (modelo_marea.directorio). Basta con las elevaciones
(ocean_tide_extrapolated, recomendado cerca de la costa) y solo las constituyentes que
se quieran usar: el script toma las que encuentre.

Hace:
  1. Armónicos de FES2022 en el mareógrafo y comparación con UTide (IOC y UHSLC).
  2. Armónicos en puntos de las fronteras abiertas (norte y oeste, configurable).
  3. Serie de nivel predicha cada 15 min para el mes candidato (CSV), en el mareógrafo
     y en cada punto de frontera, y comparación con lo observado si hay datos.
  4. exportar_aem3d(): función vacía y documentada para adaptar al formato del curso.

Uso:
    python scripts/07_tide_model_pytmd.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pyTMD
import timescale
import xarray as xr
import yaml

from utils.config import cargar_config, carpeta, parser_base, ruta
from utils.graficos import guardar, plt
from utils.log import obtener_logger
from utils.viento import diferencia_angular

log = obtener_logger("07_tide_model")

# Nombres de pyTMD (minúsculas) que difieren de los de UTide
NOMBRES_UTIDE = {"lambda2": "LDA2"}


def nombre_utide(c: str) -> str:
    return NOMBRES_UTIDE.get(c, c.upper())


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
def definicion_modelo(cfg) -> dict:
    """Definición del modelo para pyTMD a partir de su base de datos y del patrón de archivos del config."""
    mm = cfg["modelo_marea"]
    base = pyTMD.io.load_database()[mm["nombre"]]
    d = json.loads(json.dumps(base.__parameters__ if hasattr(base, "__parameters__") else dict(base.items())))
    # Patrón glob relativo a la carpeta: solo se usan los archivos presentes
    d["z"]["model_file"] = mm["archivos"]
    for g in ("u", "v"):
        d.pop(g, None)
    return d


def abrir_modelo(cfg):
    mm = cfg["modelo_marea"]
    directorio = ruta(mm["directorio"])
    if not directorio.exists():
        raise SystemExit(f"No existe la carpeta del modelo de marea: {directorio}\n"
                         "Corre primero scripts/00_fes2022_download.py (o corrige 'modelo_marea.directorio' en config.yaml; ver README).")
    m = pyTMD.io.model(directorio).from_dict(definicion_modelo(cfg), validate=True)
    archivos = sorted(m.z.model_file)
    if not archivos:
        raise SystemExit(f"No se encontraron archivos '{mm['archivos']}' dentro de {directorio}")
    log.info(f"Modelo {mm['nombre']}: {len(archivos)} archivos en {directorio}")
    m.z.model_file = archivos
    if mm.get("constituyentes"):
        m.reduce_constituents(mm["constituyentes"])
    ds = m.open_dataset(group="z", chunks={})
    d = cfg["dominio"]
    margen = 0.5
    ds = ds.tmd.crop([d["lon_min"] - margen, d["lon_max"] + margen,
                      d["lat_min"] - margen, d["lat_max"] + margen]).load()
    log.info(f"Constituyentes disponibles: {ds.tmd.constituents}")
    faltan = [c for c in ("m2", "s2", "n2", "k1", "o1", "p1") if c not in ds.tmd.constituents]
    if faltan:
        log.warning(f"Faltan constituyentes principales en el modelo: {faltan}")
    return m, ds


def interpolar(ds, lon, lat, cfg):
    """Interpola los armónicos del modelo a puntos (con extrapolación cerca de la costa)."""
    X, Y = ds.tmd.coords_as(np.atleast_1d(lon), np.atleast_1d(lat), type="drift")
    # pyTMD llama 'time' a la dimensión de una trayectoria; aquí son puntos fijos
    X, Y = X.rename({"time": "punto"}), Y.rename({"time": "punto"})
    return ds.tmd.interp(X, Y, extrapolate=True, cutoff=cfg["modelo_marea"].get("corte_extrapolacion_km", 10.0))


def tabla_armonicos(local, etiquetas: list[str]) -> pd.DataFrame:
    filas = []
    for c in local.tmd.constituents:
        A = np.atleast_1d(local[c].tmd.amplitude.values)
        g = np.atleast_1d(local[c].tmd.phase.values)
        for i, et in enumerate(etiquetas):
            filas.append({"punto": et, "constituyente": nombre_utide(c), "amplitud_m": A[i], "fase_g_grados": g[i]})
    return pd.DataFrame(filas)


def predecir(m, local, tiempos: pd.DatetimeIndex) -> np.ndarray:
    """Nivel predicho (m) en los puntos de 'local' para los tiempos dados (UTC).

    Devuelve un arreglo (tiempo, punto). Incluye las constituyentes menores inferidas.
    """
    ts = timescale.from_datetime(tiempos.values.astype("datetime64[ns]"))
    h = local.tmd.predict(ts.tide, deltat=ts.tt_ut1, corrections=m.corrections)
    try:
        h = h + local.tmd.infer(ts.tide, deltat=ts.tt_ut1, corrections=m.corrections, minor=m.minor)
    except KeyError as e:
        # La inferencia de menores necesita constituyentes como q1, k2 o 2n2
        log.warning(f"No se infieren constituyentes menores: falta {e} en los archivos del modelo")
    return np.asarray(h.transpose(..., "punto")).reshape(len(tiempos), -1)


def puntos_frontera(cfg) -> pd.DataFrame:
    filas = []
    for nombre, seg in cfg["modelo_marea"]["fronteras"].items():
        lats = np.linspace(seg["lat_ini"], seg["lat_fin"], seg["n"])
        lons = np.linspace(seg["lon_ini"], seg["lon_fin"], seg["n"])
        for i, (la, lo) in enumerate(zip(lats, lons)):
            filas.append({"punto": f"{nombre}_{i + 1:02d}", "frontera": nombre, "lat": la, "lon": lo})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Exportación a AEM3D (pendiente de plantilla)
# ---------------------------------------------------------------------------
def exportar_aem3d(serie_frontera: pd.DataFrame, puntos: pd.DataFrame, destino):
    """Exporta el nivel de frontera al formato de entrada de AEM3D. **PENDIENTE.**

    No se implementa todavía para no inventar el formato: se completará cuando se
    tenga la plantilla del curso.

    Parámetros
    ----------
    serie_frontera : DataFrame
        Índice 'tiempo_utc' cada 15 min; una columna por punto de frontera
        (nivel en m respecto del nivel medio), p. ej. 'norte_01', 'oeste_05'.
        Se guarda también en data/processed/marea_frontera_mes_candidato.csv.
    puntos : DataFrame
        Columnas punto, frontera, lat, lon de cada punto de frontera.
    destino : Path
        Carpeta donde escribir los archivos de AEM3D.

    Al adaptarla, revisar: formato de tiempo que espera AEM3D (p. ej. día juliano o
    fecha decimal), si el tiempo debe ir en UTC u hora local, unidades (m), el
    identificador de cada frontera abierta en la malla y el encabezado del archivo.
    """
    log.info("exportar_aem3d(): pendiente de la plantilla del curso; no se escribió ningún archivo de AEM3D")
    return None


# ---------------------------------------------------------------------------
def main():
    args = parser_base("Armónicos y forzamiento de marea con FES2022 (pyTMD)").parse_args()
    cfg = cargar_config(args.config)
    figdir = carpeta(cfg, "carpeta_figuras")
    procdir = carpeta(cfg, "carpeta_procesados")
    mm = cfg["modelo_marea"]

    m, ds = abrir_modelo(cfg)

    # Recorte regional del modelo (amplitud y fase) para mapas y para no depender de los archivos globales
    rec = xr.Dataset({f"amp_{nombre_utide(c)}": ds[c].tmd.amplitude for c in ds.tmd.constituents})
    rec = rec.merge(xr.Dataset({f"fase_{nombre_utide(c)}": ds[c].tmd.phase for c in ds.tmd.constituents}))
    rec.attrs.update({"fuente": mm["nombre"], "unidades_amplitud": "m", "unidades_fase": "grados (Greenwich)"})
    rec.to_netcdf(procdir / "fes2022_recorte_paracas.nc")
    log.info("Recorte regional guardado: data/processed/fes2022_recorte_paracas.nc")

    # 1) Mareógrafo
    mg = cfg["mareografo"]
    loc_mg = interpolar(ds, mg["lon"], mg["lat"], cfg)
    t_mg = tabla_armonicos(loc_mg, ["mareografo"]).drop(columns="punto").set_index("constituyente")
    comp = []
    for fuente, archivo in [("IOC", "armonicos_ioc.csv"), ("UHSLC", "armonicos_uhslc.csv")]:
        p = procdir / archivo
        if not p.exists():
            log.warning(f"No existe {archivo}: correr 06 antes para comparar")
            continue
        ut = pd.read_csv(p).set_index("constituyente")
        for c in t_mg.index:
            if c in ut.index:
                comp.append({"fuente_obs": fuente, "constituyente": c,
                             "A_fes_m": t_mg.loc[c, "amplitud_m"], "A_obs_m": ut.loc[c, "amplitud_m"],
                             "g_fes": t_mg.loc[c, "fase_g_grados"], "g_obs": ut.loc[c, "fase_g_grados"]})
    comp = pd.DataFrame(comp)
    if len(comp):
        comp["dA_cm"] = 100 * (comp.A_fes_m - comp.A_obs_m)
        comp["dg_grados"] = diferencia_angular(comp.g_fes, comp.g_obs)
        comp.round(4).to_csv(procdir / "armonicos_comparacion_fes_utide.csv", index=False)
        dest = comp[comp.constituyente.isin(cfg["analisis_marea"]["destacar"])]
        log.info("FES2022 − UTide en el mareógrafo (constituyentes destacadas):\n"
                 + dest.round(3).to_string(index=False))
        fig_comparacion(comp, cfg["analisis_marea"]["destacar"], figdir)

    # 2) Fronteras abiertas
    pts = puntos_frontera(cfg)
    loc_fr = interpolar(ds, pts.lon.values, pts.lat.values, cfg)
    t_fr = tabla_armonicos(loc_fr, pts.punto.tolist()).merge(pts, on="punto")
    sin_dato = t_fr[t_fr.amplitud_m.isna()].punto.unique()
    if len(sin_dato):
        log.warning(f"Puntos de frontera sin dato del modelo (tierra o fuera del corte de extrapolación): {list(sin_dato)}")
    t_fr.round(5).to_csv(procdir / "armonicos_fronteras_fes.csv", index=False)
    log.info(f"Armónicos en {len(pts)} puntos de frontera -> armonicos_fronteras_fes.csv")
    fig_fronteras(t_fr, figdir)

    # 3) Serie predicha para el mes candidato
    with open(procdir / "mes_candidato.yaml", encoding="utf-8") as f:
        cand = yaml.safe_load(f)
    tiempos = pd.date_range(cand["inicio_utc"], pd.Timestamp(cand["fin_utc"]).floor("D") + pd.Timedelta(days=1),
                            freq=f"{mm['paso_minutos']}min", inclusive="left")
    h_mg = predecir(m, loc_mg, tiempos)[:, 0]
    h_fr = predecir(m, loc_fr, tiempos)
    serie = pd.DataFrame(h_fr, index=tiempos, columns=pts.punto)
    serie.insert(0, "mareografo", h_mg)
    serie.index.name = "tiempo_utc"
    rango = np.nanmax(h_mg) - np.nanmin(h_mg)
    log.info(f"Serie predicha {cand['mes']}: {len(tiempos)} pasos de {mm['paso_minutos']} min; "
             f"rango en el mareógrafo = {rango:.2f} m")
    if not 0.5 <= rango <= 3.0:
        log.warning("[sanidad] rango de marea predicho fuera de lo esperado (~1 m): REVISAR")
    else:
        log.info("[sanidad] rango de marea predicho del orden de 1 m -> OK")
    serie.round(4).to_csv(procdir / "marea_frontera_mes_candidato.csv")
    log.info("Guardado: data/processed/marea_frontera_mes_candidato.csv (nivel respecto del nivel medio, m)")
    fig_serie_candidato(serie, cand, cfg, procdir, figdir)

    exportar_aem3d(serie.drop(columns="mareografo"), pts, procdir / "aem3d")
    log.info("Modelo de marea terminado")


# ---------------------------------------------------------------------------
def fig_comparacion(comp, destacar, figdir):
    d = comp[comp.constituyente.isin(destacar)]
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for fuente, mk in [("IOC", "o"), ("UHSLC", "s")]:
        s = d[d.fuente_obs == fuente].set_index("constituyente").reindex(destacar)
        x = np.arange(len(destacar)) + (-0.12 if fuente == "IOC" else 0.12)
        axs[0].plot(x, s.dA_cm, mk, ms=8, label=f"FES2022 − UTide ({fuente})")
        axs[1].plot(x, s.dg_grados, mk, ms=8, label=f"FES2022 − UTide ({fuente})")
    for ax, et in zip(axs, ["Diferencia de amplitud (cm)", "Diferencia de fase (grados)"]):
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(destacar)), destacar)
        ax.set_ylabel(et)
        ax.legend(fontsize=8)
    axs[0].set_title("Amplitud: modelo − observado")
    axs[1].set_title("Fase de Greenwich: modelo − observado")
    fig.suptitle("FES2022 frente al análisis armónico del mareógrafo de Pisco", fontweight="bold")
    guardar(fig, figdir, "marea_fes_vs_utide", log)


def fig_fronteras(t, figdir):
    fig, axs = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for j, fr in enumerate(t.frontera.unique()):
        sub = t[t.frontera == fr]
        coord = "lon" if fr in ("norte", "sur") else "lat"
        for c in ["M2", "S2", "K1", "O1"]:
            s = sub[sub.constituyente == c].sort_values(coord)
            axs[0, j % 2].plot(s[coord], 100 * s.amplitud_m, "o-", ms=3, label=c)
            axs[1, j % 2].plot(s[coord], s.fase_g_grados, "o-", ms=3, label=c)
        axs[0, j % 2].set_title(f"Frontera {fr}")
        axs[1, j % 2].set_xlabel("Longitud (°)" if coord == "lon" else "Latitud (°)")
    axs[0, 0].set_ylabel("Amplitud (cm)")
    axs[1, 0].set_ylabel("Fase de Greenwich (°)")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("Armónicos de FES2022 a lo largo de las fronteras abiertas", fontweight="bold")
    guardar(fig, figdir, "marea_armonicos_fronteras", log)


def observado_candidato(serie, cand, cfg, procdir):
    """IOC del mes candidato (estación principal, antes del QC contra la referencia) y su factor de escala.

    Si el mes falló el QC solo por la pendiente (r alto, amplitudes escaladas), el factor es 1/pendiente,
    tomado de la comparación con la marea predicha por UHSLC (independiente de FES2022).
    Devuelve (serie sin media, factor, pendiente) o (None, 1, nan) si no hay datos."""
    est = cfg["ioc"]["estaciones"][0]
    p = procdir / "marea_ioc_horaria_previa_qc.csv"
    if not p.exists():
        log.warning("No existe marea_ioc_horaria_previa_qc.csv: correr 05 para validar el mes candidato")
        return None, 1.0, np.nan
    obs = pd.read_csv(p, index_col=0, parse_dates=True)[est].loc[serie.index[0]:serie.index[-1]]
    if obs.notna().sum() <= 48:
        log.info(f"IOC {est} no tiene datos suficientes en {cand['mes']}: no se valida la serie predicha")
        return None, 1.0, np.nan
    obs = obs - obs.mean()
    factor, pend = 1.0, np.nan
    pq = procdir / "marea_ioc_qc_mensual.csv"
    if pq.exists():
        q = pd.read_csv(pq)
        f = q[(q.estacion == est) & (q.mes == cand["mes"])]
        if len(f):
            pend = float(f.pendiente.iloc[0])
            qc = cfg["analisis_marea"]["qc_referencia"]
            if not f.aprobado.iloc[0] and f.r.iloc[0] >= qc["r_min"]:
                factor = 1.0 / pend
                log.warning(f"IOC {est} {cand['mes']} falló el QC por escala (pendiente {pend:.3f} frente a "
                            f"UHSLC): para validar se corrige por {factor:.3f}")
            elif not f.aprobado.iloc[0]:
                log.warning(f"IOC {est} {cand['mes']} falló el QC por correlación (r = {f.r.iloc[0]:.3f}): "
                            "la validación es solo indicativa")
    return obs, factor, pend


def metricas(obs, pred):
    ok = obs.notna() & pred.notna()
    return float(np.sqrt(((obs[ok] - pred[ok]) ** 2).mean())), float(np.corrcoef(obs[ok], pred[ok])[0, 1])


def fig_serie_candidato(serie, cand, cfg, procdir, figdir):
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(serie.index, serie.mareografo, lw=1, label="FES2022 predicho (mareógrafo)")
    obs, factor, pend = observado_candidato(serie, cand, cfg, procdir)
    if obs is not None:
        pred_h = serie.mareografo.reindex(obs.index)
        rmse, r = metricas(obs, pred_h)
        val = {"mes": cand["mes"], "pendiente_vs_uhslc": None if np.isnan(pend) else round(pend, 4),
               "factor_escala": round(factor, 4), "rmse_sin_corregir_m": round(rmse, 4), "r": round(r, 4)}
        texto = f"Sin corregir: RMSE = {100 * rmse:.1f} cm, r = {r:.3f}"
        log.info(f"FES2022 vs IOC sin corregir en {cand['mes']}: RMSE = {100 * rmse:.1f} cm, r = {r:.3f}")
        if factor != 1.0:
            ax.plot(obs.index, obs, ".", color="0.65", ms=2, label="IOC observado, sin corregir (sin media)")
            corr = obs * factor
            rmse_c, _ = metricas(corr, pred_h)
            val["rmse_corregido_m"] = round(rmse_c, 4)
            ax.plot(corr.index, corr, "k.", ms=2, label=f"IOC corregido ×{factor:.3f} (error de escala)")
            texto += f"\nCorregido ×{factor:.3f}: RMSE = {100 * rmse_c:.1f} cm"
            log.info(f"FES2022 vs IOC corregido ×{factor:.3f}: RMSE = {100 * rmse_c:.1f} cm")
        else:
            ax.plot(obs.index, obs, "k.", ms=2, label="IOC observado (sin media)")
        ax.text(0.01, 0.95, texto, transform=ax.transAxes, va="top")
        with open(procdir / "validacion_fes_mes_candidato.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(val, f, allow_unicode=True, sort_keys=False)
    ax.set_ylabel("Nivel respecto del nivel medio (m)")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_title(f"Marea predicha para el mes candidato ({cand['mes']}), paso de 15 min")
    ax.legend(fontsize=8, loc="upper right")
    guardar(fig, figdir, "marea_prediccion_mes_candidato", log)


if __name__ == "__main__":
    main()
