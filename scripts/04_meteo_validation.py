#!/usr/bin/env python
"""04 - Validación de ERA5 contra el METAR de SPSO (aeropuerto de Pisco).

Compara velocidad, dirección y temperatura en el punto de malla ERA5 más cercano
a SPSO: sesgo, RMSE, correlación, diagramas de dispersión y rosas lado a lado.
Evalúa si ERA5 subestima los vientos Paracas, prueba corregirlo por mapeo de cuantiles (con
validación cruzada dejando un mes fuera) y escribe el forzamiento meteorológico del mes candidato
con el viento elegido en forzamiento_viento.metodo (ver utils/correccion_viento.py).

Entradas: data/raw/era5/*.nc (01), era5_punto_boca.csv (02) y metar_spso.csv (03)
Salidas : validacion_era5_spso.csv, eventos_paracas_mensual_comparacion.csv,
          forzamiento_meteo_mes_candidato.csv, viento_forzamiento_resumen.yaml y figuras.

Uso:
    python scripts/04_meteo_validation.py [--inicio 2025-01-01 --fin 2025-01-31]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from utils.config import cargar_config, carpeta, parser_base, periodo
from utils.correccion_viento import validacion_cruzada, viento_forzamiento
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


def forzamiento(cand, boca, obs, comp, cfg, procdir, figdir):
    """Forzamiento meteorológico del mes candidato: ERA5 de la boca con el viento de forzamiento_viento."""
    metodo = cfg["forzamiento_viento"]["metodo"]
    viento = viento_forzamiento(boca, obs, cfg)
    sel = boca.loc[cand["inicio_utc"]:cand["fin_utc"]].copy()
    v = viento.loc[sel.index]
    for c in ["u10_ms", "v10_ms", "vel_ms", "dir_grados"]:
        sel[c] = v[c]
    for c in ["fuente_vel", "fuente_dir", "vel_era5_ms", "vel_era5_qm_ms"]:
        sel[c] = v[c]
    sel.round(3).to_csv(procdir / "forzamiento_meteo_mes_candidato.csv")
    n_obs = int((v.fuente_vel == "spso").sum())
    _, men_f = eventos_paracas(v.vel_ms, cfg)
    ev_f = int(men_f.eventos.sum())
    resumen = {"metodo": metodo, "mes": cand["mes"], "horas": len(v), "horas_spso": n_obs,
               "horas_rellenas": len(v) - n_obs, "direcciones_era5": int((v.fuente_dir != "spso").sum()),
               "eventos_forzamiento": ev_f, "eventos_metar": int(cand.get("eventos", -1)),
               "eventos_era5_qm_cv_mes": int(comp.loc[cand["mes"], "eventos_era5_qm_cv"])
               if cand["mes"] in comp.index else None,
               "eventos_metar_total": int(comp.eventos_metar.sum()),
               "eventos_era5_qm_cv_total": int(comp.eventos_era5_qm_cv.sum()),
               "vel_media_ms": round(float(v.vel_ms.mean()), 2), "vel_max_ms": round(float(v.vel_ms.max()), 2)}
    with open(procdir / "viento_forzamiento_resumen.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resumen, f, allow_unicode=True, sort_keys=False)
    log.info(f"Forzamiento meteorológico {cand['mes']}: {len(sel)} horas -> forzamiento_meteo_mes_candidato.csv; "
             f"viento '{metodo}': {n_obs} h de SPSO, {len(v) - n_obs} h rellenas con ERA5 corregido, "
             f"{resumen['direcciones_era5']} direcciones de ERA5; {ev_f} días con evento")
    if metodo == "spso":
        log.info("Nota: SPSO es una estación en tierra (aeropuerto); sobre el agua el viento suele ser algo más "
                 "intenso por la menor rugosidad, así que el forzamiento es conservador")
    fig_forzamiento(viento.loc[sel.index], obs, comp, cand, cfg, figdir)


def fig_forzamiento(v, obs, comp, cand, cfg, figdir):
    fig, axs = plt.subplots(2, 1, figsize=(13, 8))
    ax = axs[0]
    x = np.arange(len(comp))
    for i, (col, et, c) in enumerate([("eventos_metar", "METAR SPSO (observado)", "k"),
                                      ("eventos_era5", "ERA5 sin corregir", "tab:orange"),
                                      ("eventos_era5_qm_cv", "ERA5 corregido (validación cruzada)", "tab:blue")]):
        ax.bar(x + (i - 1) * 0.27, comp[col], 0.27, color=c, label=et, alpha=0.85)
    ax.set_xticks(x, [f"{MESES[p.month - 1]}\n{p.year}" for p in pd.PeriodIndex(comp.index, freq="M")], fontsize=7)
    ax.set_ylabel("Días con evento")
    ax.set_title("Eventos Paracas por mes: la corrección por cuantiles iguala el total pero no los meses")
    ax.legend(fontsize=8)
    ax = axs[1]
    o = obs.vel_ms.reindex(v.index)
    ax.plot(v.index, v.vel_era5_ms, color="tab:orange", lw=0.8, label="ERA5 sin corregir")
    ax.plot(v.index, v.vel_era5_qm_ms, color="tab:blue", lw=0.8, alpha=0.7, label="ERA5 corregido")
    ax.plot(o.index, o, "k-", lw=1, label="SPSO observado")
    rell = v.vel_ms.where(v.fuente_vel != "spso")
    ax.plot(rell.index, rell, "r.", ms=4, label="Forzamiento: horas rellenas")
    ax.axhline(cfg["vientos_paracas"]["umbral_ms"], color="0.4", ls="--", lw=0.8, label="Umbral de evento")
    ax.set_ylabel("Velocidad del viento (m/s)")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_title(f"Viento del forzamiento, {cand['mes']} (método: {cfg['forzamiento_viento']['metodo']})")
    ax.legend(fontsize=8, ncol=5, loc="upper left")
    guardar(fig, figdir, "viento_forzamiento_mes_candidato", log)


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
    # ¿Basta con corregir ERA5? Mapeo de cuantiles por hora local, evaluado dejando un mes fuera
    desfase = cfg["proyecto"]["zona_horaria_local"]
    boca = pd.read_csv(procdir / "era5_punto_boca.csv", index_col=0, parse_dates=True).loc[ini:fin]
    obs_b = obs.vel_ms.reindex(boca.index)
    qm_cv = validacion_cruzada(boca.vel_ms, obs_b, desfase)
    _, men_qm = eventos_paracas(qm_cv[obs_b.notna()], cfg)
    comp["eventos_era5_qm_cv"] = men_qm.eventos.reindex(comp.index)
    comp.to_csv(procdir / "eventos_paracas_mensual_comparacion.csv")
    ok = obs_b.notna() & qm_cv.notna()
    hl_b = a_hora_local(boca.index, desfase).hour
    tarde_b = (hl_b >= vp["hora_local_inicio"]) & (hl_b <= vp["hora_local_fin"])
    m_qm = metricas(obs_b[ok & tarde_b], qm_cv[ok & tarde_b])
    m_raw = metricas(obs_b[ok & tarde_b], boca.vel_ms[ok & tarde_b])
    acierto = (comp.eventos_metar > 0) & (comp.eventos_era5_qm_cv > 0)
    log.info(f"ERA5 corregido por cuantiles (validación cruzada): {int(comp.eventos_era5_qm_cv.sum())} días con "
             f"evento frente a {int(comp.eventos_metar.sum())} observados; por la tarde r = {m_qm['r']:.2f} "
             f"(sin corregir {m_raw['r']:.2f}), RMSE = {m_qm['rmse']:.2f} m/s")
    log.info("Eventos por mes (METAR / ERA5 / ERA5 corregido):\n"
             + comp[["eventos_metar", "eventos_era5", "eventos_era5_qm_cv"]].T.to_string())
    r_mes = comp[["eventos_metar", "eventos_era5_qm_cv"]].corr().iloc[0, 1]
    log.info(f"Correlación mensual de eventos METAR vs ERA5 corregido: r = {r_mes:.2f}: la corrección iguala "
             "la intensidad pero no la cronología de los eventos")

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
                 "tierra-mar y la canalización local. Corregirlo por cuantiles no basta (ver arriba): para forzar "
                 "AEM3D se usa el viento observado en SPSO (forzamiento_viento.metodo).")
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
        log.info(f"MES CANDIDATO (según METAR): {cand['mes']} ({cand['eventos']} eventos) – {cand['criterio']}")
    else:
        with open(procdir / "mes_candidato.yaml", encoding="utf-8") as f:
            cand = yaml.safe_load(f)  # escrito por 02
    forzamiento(cand, boca, obs, comp, cfg, procdir, figdir)
    log.info("Validación terminada")


if __name__ == "__main__":
    main()
