#!/usr/bin/env python
"""04 - Validación de ERA5 contra el METAR de SPSO (aeropuerto de Pisco).

Compara velocidad, dirección y temperatura en el punto de malla ERA5 más cercano
a SPSO: sesgo, RMSE, correlación, diagramas de dispersión y rosas lado a lado.
Evalúa si ERA5 subestima los vientos Paracas.

Entradas: data/raw/era5/*.nc (01) y data/processed/metar_spso.csv (03)
Salidas : data/processed/validacion_era5_spso.csv, eventos_paracas_mensual_comparacion.csv y figuras.

Uso:
    python scripts/04_meteo_validation.py [--inicio 2025-01-01 --fin 2025-01-31]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from utils.config import cargar_config, carpeta, parser_base, periodo
from utils.era5 import a_tabla, derivar, elegir_mes_candidato, eventos_paracas, leer_era5, punto_cercano
from utils.graficos import MESES, eje_rosa, guardar, plt, rosa
from utils.log import obtener_logger
from utils.viento import a_hora_local, diferencia_angular

log = obtener_logger("04_meteo_validation")

VEL_MIN_DIRECCION = 2.0  # m/s: por debajo, la dirección es poco representativa


def metricas(obs: pd.Series, mod: pd.Series) -> dict:
    ok = obs.notna() & mod.notna()
    o, m = obs[ok], mod[ok]
    if len(o) < 3:
        return {"n": int(len(o)), "media_obs": np.nan, "media_era5": np.nan, "sesgo": np.nan, "rmse": np.nan, "r": np.nan}
    return {"n": int(len(o)), "media_obs": o.mean(), "media_era5": m.mean(), "sesgo": (m - o).mean(),
            "rmse": float(np.sqrt(((m - o) ** 2).mean())), "r": float(np.corrcoef(o, m)[0, 1])}


def metricas_direccion(obs, mod, vel_obs) -> dict:
    ok = obs.notna() & mod.notna() & (vel_obs >= VEL_MIN_DIRECCION)
    d = diferencia_angular(mod[ok], obs[ok])
    if len(d) < 3:
        return {"n": int(len(d)), "sesgo": np.nan, "rmse": np.nan, "r": np.nan}
    # Sesgo circular: ángulo medio de las diferencias
    sesgo = float(np.degrees(np.arctan2(np.sin(np.radians(d)).mean(), np.cos(np.radians(d)).mean())))
    return {"n": int(len(d)), "media_obs": np.nan, "media_era5": np.nan, "sesgo": sesgo,
            "rmse": float(np.sqrt((d ** 2).mean())), "r": np.nan,
            "error_abs_medio": float(np.abs(d).mean())}


def fig_dispersion(par, figdir):
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
    pares = [("vel_obs", "vel_era5", "Velocidad del viento (m/s)"), ("t_obs", "t_era5", "Temperatura (°C)")]
    for ax, (co, cm, tit) in zip(axs[:2], pares):
        ok = par[co].notna() & par[cm].notna()
        x, y = par.loc[ok, co], par.loc[ok, cm]
        ax.scatter(x, y, s=4, alpha=0.3)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        if len(x) > 2:
            a, b = np.polyfit(x, y, 1)
            ax.plot([lo, hi], [a * lo + b, a * hi + b], "r-", lw=1.2, label=f"Ajuste: y = {a:.2f}x + {b:.2f}")
        ax.set_xlabel(f"SPSO observado – {tit}")
        ax.set_ylabel(f"ERA5 – {tit}")
        ax.set_title(tit)
        ax.legend(fontsize=8)
        ax.set_aspect("equal", adjustable="box")
    ok = par.dir_obs.notna() & par.dir_era5.notna() & (par.vel_obs >= VEL_MIN_DIRECCION)
    axs[2].scatter(par.loc[ok, "dir_obs"], par.loc[ok, "dir_era5"], s=4, alpha=0.3)
    axs[2].plot([0, 360], [0, 360], "k--", lw=1)
    axs[2].set_xlim(0, 360)
    axs[2].set_ylim(0, 360)
    axs[2].set_xticks(range(0, 361, 90), ["N", "E", "S", "O", "N"])
    axs[2].set_yticks(range(0, 361, 90), ["N", "E", "S", "O", "N"])
    axs[2].set_xlabel("SPSO observado – dirección")
    axs[2].set_ylabel("ERA5 – dirección")
    axs[2].set_title(f"Dirección (viento obs. ≥ {VEL_MIN_DIRECCION:g} m/s)")
    axs[2].set_aspect("equal")
    fig.suptitle("Validación ERA5 vs METAR SPSO (aeropuerto de Pisco), datos horarios", fontweight="bold")
    guardar(fig, figdir, "validacion_dispersion_era5_spso", log)


def fig_rosas(par, figdir):
    ok = par.vel_obs.notna() & par.vel_era5.notna()
    p = par[ok]
    fig = plt.figure(figsize=(13, 6.8))
    # Margen superior para que el título general no tape los títulos de cada rosa
    fig.subplots_adjust(top=0.82)
    rosa(eje_rosa(fig, (1, 2, 1)), p.dir_obs, p.vel_obs, "METAR SPSO (observado)", leyenda=False)
    rosa(eje_rosa(fig, (1, 2, 2)), p.dir_era5, p.vel_era5, "ERA5 (punto más cercano)")
    fig.suptitle(f"Rosas de viento en las mismas horas ({len(p)} h, "
                 f"{p.index[0]:%Y-%m-%d} a {p.index[-1]:%Y-%m-%d})", fontweight="bold")
    guardar(fig, figdir, "validacion_rosas_era5_spso", log)


def fig_ciclo(par, cfg, figdir):
    loc = par.copy()
    loc.index = a_hora_local(loc.index, cfg["proyecto"]["zona_horaria_local"])
    g = loc.groupby(loc.index.hour)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(g.vel_obs.mean(), "o-", label="METAR SPSO (media)")
    ax.plot(g.vel_era5.mean(), "s-", label="ERA5 (media)")
    ax.plot(g.vel_obs.quantile(0.9), "o:", color="tab:blue", alpha=0.6, label="METAR SPSO (p90)")
    ax.plot(g.vel_era5.quantile(0.9), "s:", color="tab:orange", alpha=0.6, label="ERA5 (p90)")
    vp = cfg["vientos_paracas"]
    ax.axvspan(vp["hora_local_inicio"], vp["hora_local_fin"], color="gold", alpha=0.2, label="Ventana Paracas")
    ax.axhline(vp["umbral_ms"], color="r", ls="--", lw=1, label=f"Umbral {vp['umbral_ms']:g} m/s")
    ax.set_xlabel("Hora local (UTC−5)")
    ax.set_ylabel("Velocidad del viento (m/s)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Ciclo diurno del viento: ERA5 vs METAR SPSO")
    ax.legend(fontsize=7, ncol=2)
    guardar(fig, figdir, "validacion_ciclo_diurno_era5_spso", log)


def fig_eventos(comp, cfg, figdir):
    vp = cfg["vientos_paracas"]
    x = np.arange(len(comp))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.bar(x - 0.2, comp.eventos_metar, 0.4, label="METAR SPSO")
    ax.bar(x + 0.2, comp.eventos_era5, 0.4, label="ERA5")
    ax.set_xticks(x, [f"{MESES[p.month - 1]}\n{p.year}" for p in comp.index], fontsize=7)
    ax.set_ylabel("Días con evento")
    ax.set_title(f"Eventos de vientos Paracas por mes (> {vp['umbral_ms']:g} m/s, "
                 f"{vp['hora_local_inicio']}–{vp['hora_local_fin']} h local): observado vs ERA5")
    ax.legend()
    guardar(fig, figdir, "validacion_eventos_paracas_mensual", log)


def main():
    args = parser_base("Valida ERA5 contra el METAR de SPSO").parse_args()
    cfg = cargar_config(args.config)
    ini, fin = periodo(cfg, args)
    figdir = carpeta(cfg, "carpeta_figuras")
    procdir = carpeta(cfg, "carpeta_procesados")

    # ERA5 en el punto de malla más cercano al aeropuerto
    m = cfg["metar"]
    era = a_tabla(punto_cercano(derivar(leer_era5(cfg, ini, fin)), m["lat"], m["lon"], "SPSO"))
    obs = pd.read_csv(procdir / "metar_spso.csv", index_col=0, parse_dates=True).loc[ini:fin]
    if obs.empty:
        raise SystemExit("metar_spso.csv no cubre el periodo: correr 03 con las mismas fechas")

    par = pd.DataFrame({
        "vel_obs": obs.vel_ms, "dir_obs": obs.dir_grados, "t_obs": obs.t2m_C,
    }).join(pd.DataFrame({"vel_era5": era.vel_ms, "dir_era5": era.dir_grados, "t_era5": era.t2m_C}), how="inner")
    log.info(f"Horas emparejadas ERA5–SPSO: {len(par)} ({par.vel_obs.notna().sum()} con viento observado)")

    # Métricas globales y en la ventana de la tarde
    loc_h = a_hora_local(par.index, cfg["proyecto"]["zona_horaria_local"]).hour
    vp = cfg["vientos_paracas"]
    tarde = (loc_h >= vp["hora_local_inicio"]) & (loc_h <= vp["hora_local_fin"])
    filas = {
        "velocidad (m/s)": metricas(par.vel_obs, par.vel_era5),
        "velocidad tarde 12-18 h (m/s)": metricas(par.vel_obs[tarde], par.vel_era5[tarde]),
        f"velocidad cuando obs > {vp['umbral_ms']:g} m/s": metricas(par.vel_obs[par.vel_obs > vp["umbral_ms"]],
                                                                   par.vel_era5[par.vel_obs > vp["umbral_ms"]]),
        "dirección (grados)": metricas_direccion(par.dir_obs, par.dir_era5, par.vel_obs),
        "temperatura (°C)": metricas(par.t_obs, par.t_era5),
    }
    tabla = pd.DataFrame(filas).T
    tabla.index.name = "variable"
    tabla.round(3).to_csv(procdir / "validacion_era5_spso.csv")
    log.info("Métricas (sesgo = ERA5 − observado):\n" + tabla.round(2).to_string())

    # Eventos Paracas: observados vs ERA5
    _, men_obs = eventos_paracas(par.vel_obs, cfg)
    _, men_era = eventos_paracas(par.vel_era5, cfg)
    comp = pd.DataFrame({"eventos_metar": men_obs.eventos, "eventos_era5": men_era.eventos,
                         "vmax_tarde_metar": men_obs.vel_max_tarde_ms, "vmax_tarde_era5": men_era.vel_max_tarde_ms})
    comp.to_csv(procdir / "eventos_paracas_mensual_comparacion.csv")

    # Discusión: ¿subestima ERA5 los vientos Paracas?
    p95_o, p95_m = par.vel_obs[tarde].quantile(0.95), par.vel_era5[tarde].quantile(0.95)
    n_o, n_m = int(comp.eventos_metar.sum()), int(comp.eventos_era5.sum())
    log.info("=" * 70)
    log.info("DISCUSIÓN: ¿ERA5 subestima los vientos Paracas?")
    log.info(f"  Días con evento: METAR = {n_o}, ERA5 = {n_m}")
    log.info(f"  p95 del viento de la tarde: METAR = {p95_o:.1f} m/s, ERA5 = {p95_m:.1f} m/s "
             f"(razón ERA5/obs = {p95_m / p95_o:.2f})")
    fuerte = filas[f"velocidad cuando obs > {vp['umbral_ms']:g} m/s"]
    if fuerte["n"] > 0:
        log.info(f"  En las {fuerte['n']} horas con obs > {vp['umbral_ms']:g} m/s, ERA5 da en promedio "
                 f"{fuerte['media_era5']:.1f} m/s frente a {fuerte['media_obs']:.1f} m/s observados "
                 f"(sesgo {fuerte['sesgo']:+.1f} m/s)")
    if n_o > 0 and (n_m < 0.5 * n_o or p95_m / p95_o < 0.85):
        log.info("  CONCLUSIÓN: ERA5 SUBESTIMA los vientos Paracas: detecta menos eventos y sus picos vespertinos "
                 "son más débiles. Es esperable por la resolución de ~30 km, que suaviza el gradiente "
                 "tierra-mar y la canalización local. Para forzar AEM3D conviene corregir el viento de ERA5 "
                 "(p. ej. escalarlo con la razón de cuantiles) o usar el METAR.")
    elif n_o == 0:
        log.info("  CONCLUSIÓN: no hubo eventos observados en el periodo; no se puede evaluar la subestimación.")
    else:
        log.info("  CONCLUSIÓN: ERA5 reproduce razonablemente la intensidad de los vientos Paracas en este periodo.")
    log.info("=" * 70)

    fig_dispersion(par, figdir)
    fig_rosas(par, figdir)
    fig_ciclo(par, cfg, figdir)
    fig_eventos(comp, cfg, figdir)

    if vp["fuente"] == "metar":
        cand = elegir_mes_candidato(men_obs, "metar", vp.get("mes_candidato", "auto"))
        with open(procdir / "mes_candidato.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cand, f, allow_unicode=True, sort_keys=False)
        # El forzamiento sale del punto ERA5 de la boca (escrito por 02), igual que con fuente ERA5
        boca = pd.read_csv(procdir / "era5_punto_boca.csv", index_col=0, parse_dates=True)
        sel = boca.loc[cand["inicio_utc"]:cand["fin_utc"]]
        sel.round(3).to_csv(procdir / "forzamiento_meteo_mes_candidato.csv")
        log.info(f"MES CANDIDATO (según METAR): {cand['mes']} ({cand['eventos']} eventos) – {cand['criterio']}")
        log.info(f"Forzamiento meteorológico (ERA5, boca): {len(sel)} horas -> forzamiento_meteo_mes_candidato.csv. "
                 "Ojo: el viento de ERA5 sigue subestimado en los eventos")
    log.info("Validación terminada")


if __name__ == "__main__":
    main()
