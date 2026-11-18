#!/usr/bin/env python
"""02 - Procesamiento de ERA5: variables derivadas, figuras y eventos de vientos Paracas.

Entradas : data/raw/era5/era5_AAAAMM.nc (de 01_era5_download.py)
Salidas  : data/processed/era5_punto_boca.csv, era5_promedio_caja.csv,
           eventos_paracas_era5.csv, eventos_paracas_mensual_era5.csv,
           mes_candidato.yaml, forzamiento_meteo_mes_candidato.csv y figuras.

Uso:
    python scripts/02_era5_process.py [--inicio 2025-01-01 --fin 2025-01-31]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from utils import sanidad
from utils.config import cargar_config, carpeta, parser_base, periodo
from utils.era5 import a_tabla, derivar, elegir_mes_candidato, eventos_paracas, leer_era5, punto_cercano
from utils.graficos import ESTACIONES, MESES, eje_rosa, guardar, plt, rosa
from utils.log import obtener_logger
from utils.viento import a_hora_local, estacion_del_anio, velocidad_direccion

log = obtener_logger("02_era5_process")


def promedio_caja(ds, d):
    """Promedio de las celdas de 0.25° cuya área intersecta la caja del dominio."""
    m = 0.125
    sub = ds.sel(latitude=(ds.latitude >= d["lat_min"] - m) & (ds.latitude <= d["lat_max"] + m),
                 longitude=(ds.longitude >= d["lon_min"] - m) & (ds.longitude <= d["lon_max"] + m))
    n = sub.sizes["latitude"] * sub.sizes["longitude"]
    log.info(f"Promedio de caja: {n} celdas (lat {sub.latitude.values}, lon {sub.longitude.values})")
    caja = sub.mean(dim=("latitude", "longitude"))
    # La dirección se recalcula a partir de u y v promediados (no se promedian ángulos)
    vel, dirm = velocidad_direccion(caja.u10, caja.v10)
    caja["dir"] = ("time", dirm)
    caja["vel_vectorial"] = ("time", vel)
    return caja


def verificar(df: pd.DataFrame, nombre: str):
    sanidad.rango(f"{nombre} velocidad del viento (m/s)", df.vel_ms, 0, 30, log)
    sanidad.rango(f"{nombre} radiación solar (W/m2)", df.ssrd_Wm2, 0, 1400, log)
    sanidad.rango(f"{nombre} radiación de onda larga (W/m2)", df.strd_Wm2, 0, 600, log)
    sanidad.rango(f"{nombre} temperatura (°C)", df.t2m_C, 5, 40, log)
    sanidad.rango(f"{nombre} humedad relativa (%)", df.hr_pct, 0, 100, log)
    sanidad.rango(f"{nombre} presión (hPa)", df.msl_hPa, 990, 1030, log)


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_rosas(df, figdir, etiqueta):
    fig = plt.figure(figsize=(6, 6))
    ax = eje_rosa(fig, (1, 1, 1))
    rosa(ax, df.dir_grados, df.vel_ms, f"Rosa de vientos ERA5 – {etiqueta}\n"
         f"{df.index[0]:%Y-%m-%d} a {df.index[-1]:%Y-%m-%d}")
    guardar(fig, figdir, "era5_rosa_viento_anual", log)

    est = estacion_del_anio(df.index)
    fig = plt.figure(figsize=(11, 11.5))
    # Espacio vertical para que el título de la fila inferior no tape la "S" de la superior
    fig.subplots_adjust(hspace=0.45, top=0.90)
    for i, e in enumerate(["DEF", "MAM", "JJA", "SON"]):
        ax = eje_rosa(fig, (2, 2, i + 1))
        sel = df[est == e]
        if len(sel) == 0:
            ax.set_axis_off()
            ax.set_title(f"{ESTACIONES[e]}\n(sin datos en el periodo)")
            continue
        rosa(ax, sel.dir_grados, sel.vel_ms, f"{ESTACIONES[e]} (n = {len(sel)} h)", leyenda=(i == 1))
    fig.suptitle(f"Rosas de viento estacionales ERA5 – {etiqueta}", fontweight="bold")
    guardar(fig, figdir, "era5_rosa_viento_estaciones", log)


def fig_ciclo_diurno(df, cfg, figdir, etiqueta):
    loc = df.copy()
    loc.index = a_hora_local(loc.index, cfg["proyecto"]["zona_horaria_local"])
    loc["estacion"] = estacion_del_anio(loc.index)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    g = loc.groupby(loc.index.hour)
    m, s = g.vel_ms.mean(), g.vel_ms.std()
    a1.fill_between(m.index, m - s, m + s, color="0.8", label="±1 desv. estándar")
    a1.plot(m.index, m, "k-", lw=2, label="Todo el periodo")
    for e, c in zip(["DEF", "MAM", "JJA", "SON"], ["tab:red", "tab:orange", "tab:blue", "tab:green"]):
        sel = loc[loc.estacion == e]
        if len(sel):
            a1.plot(sel.groupby(sel.index.hour).vel_ms.mean(), color=c, lw=1.3, label=ESTACIONES[e])
    vp = cfg["vientos_paracas"]
    a1.axvspan(vp["hora_local_inicio"], vp["hora_local_fin"], color="gold", alpha=0.2, label="Ventana Paracas")
    a1.axhline(vp["umbral_ms"], color="r", ls="--", lw=1, label=f"Umbral {vp['umbral_ms']:g} m/s")
    a1.set_ylabel("Velocidad del viento (m/s)")
    a1.set_title(f"Ciclo diurno del viento a 10 m – ERA5, {etiqueta}")
    a1.legend(fontsize=7, ncol=2)
    # Dirección media vectorial por hora
    u, v = g.u10_ms.mean(), g.v10_ms.mean()
    dirm = (270 - np.degrees(np.arctan2(v, u))) % 360
    a2.plot(dirm.index, dirm, "o-", color="tab:purple")
    a2.set_ylim(0, 360)
    a2.set_yticks(range(0, 361, 45), ["N", "NE", "E", "SE", "S", "SO", "O", "NO", "N"])
    a2.set_ylabel("Dirección media (de donde viene)")
    a2.set_xlabel("Hora local (UTC−5)")
    a2.set_xticks(range(0, 24, 2))
    guardar(fig, figdir, "era5_ciclo_diurno_viento", log)


def fig_climatologia(df, figdir, etiqueta):
    mens = df.groupby(df.index.month)
    tp_mes = df.tp_mm.resample("MS").sum()
    tp_clim = tp_mes.groupby(tp_mes.index.month).mean()
    paneles = [("vel_ms", "Viento (m/s)"), ("t2m_C", "Temperatura (°C)"), ("hr_pct", "Humedad relativa (%)"),
               ("msl_hPa", "Presión a nivel del mar (hPa)"), ("ssrd_Wm2", "Radiación solar (W/m²)"),
               ("tcc_frac", "Nubosidad total (fracción)")]
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for ax, (col, tit) in zip(axs.flat, paneles):
        m, s = mens[col].mean(), mens[col].std()
        ax.errorbar(m.index, m, yerr=s, fmt="o-", capsize=3)
        ax.set_title(tit)
    axs.flat[6].bar(tp_clim.index, tp_clim, color="tab:blue")
    axs.flat[6].set_title("Precipitación (mm/mes)")
    axs.flat[7].plot(mens.strd_Wm2.mean(), "o-", color="tab:brown")
    axs.flat[7].set_title("Radiación de onda larga (W/m²)")
    axs.flat[8].axis("off")
    meses_por_mes = df.index.to_period("M").unique()
    axs.flat[8].text(0.05, 0.6, f"Periodo: {df.index[0]:%Y-%m} a {df.index[-1]:%Y-%m}\n"
                     f"{len(meses_por_mes)} meses de datos\nBarras: ±1 desv. estándar\n"
                     f"de los valores horarios", fontsize=10, va="top")
    for ax in axs.flat[:8]:
        ax.set_xticks(range(1, 13), MESES, rotation=45, fontsize=8)
        ax.set_xlim(0.5, 12.5)
    fig.suptitle(f"Climatología mensual ERA5 – {etiqueta}", fontweight="bold")
    fig.tight_layout()
    guardar(fig, figdir, "era5_climatologia_mensual", log)


def fig_series(df, figdir, etiqueta):
    dia = df.resample("D").mean()
    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axs[0].plot(df.index, df.t2m_C, color="tab:red", lw=0.3, alpha=0.5, label="Horaria")
    axs[0].plot(dia.index, dia.t2m_C, color="darkred", lw=1.5, label="Media diaria")
    axs[0].set_ylabel("Temperatura a 2 m (°C)")
    axs[1].plot(df.index, df.hr_pct, color="tab:blue", lw=0.3, alpha=0.5, label="Horaria")
    axs[1].plot(dia.index, dia.hr_pct, color="navy", lw=1.5, label="Media diaria")
    axs[1].set_ylabel("Humedad relativa (%)")
    axs[2].plot(df.index, df.ssrd_Wm2, color="orange", lw=0.3, alpha=0.5, label="Solar horaria")
    axs[2].plot(dia.index, dia.ssrd_Wm2, color="darkorange", lw=1.5, label="Solar, media diaria")
    axs[2].plot(dia.index, dia.strd_Wm2, color="tab:brown", lw=1.5, label="Onda larga, media diaria")
    axs[2].set_ylabel("Radiación descendente (W/m²)")
    axs[2].set_xlabel("Fecha (UTC)")
    for ax in axs:
        ax.legend(fontsize=7, loc="upper right")
    axs[0].set_title(f"Series de temperatura, humedad relativa y radiación – ERA5, {etiqueta}")
    guardar(fig, figdir, "era5_series_T_HR_radiacion", log)


def fig_eventos(mensual, cand, cfg, figdir):
    vp = cfg["vientos_paracas"]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    etiquetas = [f"{MESES[p.month - 1]}\n{p.year}" for p in mensual.index]
    mes_rojo = cand["mes"] if cand else None
    colores = ["crimson" if str(p) == mes_rojo else "tab:blue" for p in mensual.index]
    ax.bar(range(len(mensual)), mensual.eventos, color=colores)
    ax.set_ylim(0, max(1, mensual.eventos.max()) * 1.15)  # sin ejes negativos si no hay eventos
    ax.set_xticks(range(len(mensual)), etiquetas, fontsize=7)
    ax.set_ylabel("Días con evento")
    ax2 = ax.twinx()
    ax2.plot(range(len(mensual)), mensual.p95_tarde_ms, "k.--", lw=0.8, label="p95 del máximo vespertino")
    ax2.set_ylabel("Viento (m/s)")
    ax2.axhline(vp["umbral_ms"], color="r", ls=":", lw=1)
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_title(f"Eventos de vientos Paracas por mes (ERA5): viento > {vp['umbral_ms']:g} m/s entre las "
                 f"{vp['hora_local_inicio']} y {vp['hora_local_fin']} h locales\n"
                 + (f"En rojo: mes candidato para la simulación ({mes_rojo})" if mes_rojo
                    else "El mes candidato se define con el METAR (script 04)"))
    guardar(fig, figdir, "era5_eventos_paracas_mensual", log)


# ---------------------------------------------------------------------------
def main():
    args = parser_base("Procesa ERA5: viento, HR, radiación, figuras y vientos Paracas").parse_args()
    cfg = cargar_config(args.config)
    ini, fin = periodo(cfg, args)
    figdir = carpeta(cfg, "carpeta_figuras")
    procdir = carpeta(cfg, "carpeta_procesados")

    ds = derivar(leer_era5(cfg, ini, fin))
    b = cfg["boca_bahia"]
    punto = a_tabla(punto_cercano(ds, b["lat"], b["lon"]))
    caja_ds = promedio_caja(ds, cfg["dominio"])
    caja = a_tabla(caja_ds)
    caja["vel_vectorial_ms"] = caja_ds.vel_vectorial.values

    verificar(punto, "Punto boca:")
    verificar(caja, "Promedio caja:")

    punto.round(3).to_csv(procdir / "era5_punto_boca.csv")
    caja.round(3).to_csv(procdir / "era5_promedio_caja.csv")
    log.info(f"Guardados: era5_punto_boca.csv y era5_promedio_caja.csv ({len(punto)} filas)")

    etiqueta = "boca de la bahía de Paracas"
    fig_rosas(punto, figdir, etiqueta)
    fig_ciclo_diurno(punto, cfg, figdir, etiqueta)
    fig_climatologia(punto, figdir, etiqueta)
    fig_series(punto, figdir, etiqueta)

    # Eventos de vientos Paracas (con la serie del punto de la boca)
    dias, mensual = eventos_paracas(punto.vel_ms, cfg)
    dias.to_csv(procdir / "eventos_paracas_era5.csv")
    mensual.to_csv(procdir / "eventos_paracas_mensual_era5.csv")
    log.info(f"Eventos Paracas (ERA5): {int(dias.evento.sum())} días de {len(dias)}; "
             f"velocidad máxima vespertina = {dias.vel_max_tarde_ms.max():.1f} m/s")
    for per, fila in mensual.iterrows():
        log.info(f"  {per}: {int(fila.eventos):2d} eventos, máximo vespertino {fila.vel_max_tarde_ms:.1f} m/s")

    vp = cfg["vientos_paracas"]
    mes_fijo = vp.get("mes_candidato", "auto")
    cand = elegir_mes_candidato(mensual, "era5", mes_fijo)
    if vp["fuente"] == "era5":
        with open(procdir / "mes_candidato.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cand, f, allow_unicode=True, sort_keys=False)
        log.info(f"MES CANDIDATO para la simulación: {cand['mes']} ({cand['eventos']} eventos) – {cand['criterio']}")
        # Forzamiento meteorológico del mes candidato (CSV horario en UTC, punto de la boca)
        sel = punto.loc[cand["inicio_utc"]:cand["fin_utc"]]
        sel.round(3).to_csv(procdir / "forzamiento_meteo_mes_candidato.csv")
        log.info(f"Forzamiento meteorológico del mes candidato: {len(sel)} horas -> forzamiento_meteo_mes_candidato.csv")
    else:
        log.info(f"El mes candidato y su forzamiento los escribe el script 04 (fuente '{vp['fuente']}')")
    # En la figura solo se marca el mes si es el definitivo (fuente ERA5 o mes fijado a mano)
    definitivo = vp["fuente"] == "era5" or str(mes_fijo).lower() != "auto"
    fig_eventos(mensual, cand if definitivo else None, cfg, figdir)
    log.info("Procesamiento ERA5 terminado")


if __name__ == "__main__":
    main()
