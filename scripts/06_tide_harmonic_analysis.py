#!/usr/bin/env python
"""06 - Análisis armónico de la marea en Pisco con UTide.

Analiza dos series:
  * IOC 2025–2026 (ventana elegida en 05)  -> serie principal
  * UHSLC, año de referencia (p. ej. 2022) -> control de calidad
y compara sus constituyentes.

Salidas (data/processed/):
  armonicos_ioc.csv, armonicos_uhslc.csv, armonicos_comparacion_ioc_uhslc.csv,
  resumen_marea.csv y figuras (amplitudes, observado vs predicho, residuo y espectro).

Uso:
    python scripts/06_tide_harmonic_analysis.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import utide
import yaml
from scipy import signal

from utils.config import cargar_config, carpeta, parser_base
from utils.graficos import guardar, plt
from utils.log import obtener_logger
from utils.viento import diferencia_angular

log = obtener_logger("06_tide_harmonic")

CICLO_SICIGIA_DIAS = 14.765  # periodo sinódico medio de medio mes lunar (M2-S2)


def clasificar(F: float) -> str:
    if not np.isfinite(F):
        return "indeterminada"
    if F < 0.25:
        return "semidiurna"
    if F <= 1.5:
        return "mixta, predominantemente semidiurna"
    if F <= 3.0:
        return "mixta, predominantemente diurna"
    return "diurna"


def analizar(serie: pd.Series, lat: float, nombre: str):
    """Ejecuta utide.solve y reconstruye la marea. Devuelve (coef, tabla, predicha)."""
    s = serie.dropna()
    log.info(f"[{nombre}] {len(s)} horas válidas ({s.index[0]:%Y-%m-%d} a {s.index[-1]:%Y-%m-%d}), "
             f"{100 * len(s) / len(serie):.1f} % del periodo")
    coef = utide.solve(s.index.values, s.values, lat=lat, method="ols", conf_int="MC",
                       trend=False, nodal=True, constit="auto", verbose=False)
    tabla = pd.DataFrame({
        "constituyente": coef["name"],
        "frecuencia_cph": coef["aux"]["frq"],
        "periodo_h": 1.0 / coef["aux"]["frq"],
        "amplitud_m": coef["A"],
        "amplitud_ic95_m": coef["A_ci"],
        "fase_g_grados": coef["g"],
        "fase_ic95_grados": coef["g_ci"],
        "snr": coef["SNR"],
        "pe_pct": coef["PE"],
    }).sort_values("amplitud_m", ascending=False).reset_index(drop=True)
    tabla["significativa"] = tabla.snr >= 2.0
    pred = utide.reconstruct(serie.index.values, coef, verbose=False)
    predicha = pd.Series(pred.h, index=serie.index)
    log.info(f"[{nombre}] {len(tabla)} constituyentes resueltas, {int(tabla.significativa.sum())} con SNR ≥ 2; "
             f"nivel medio {coef['mean']:.3f} m")
    return coef, tabla, predicha


def amplitud(tabla: pd.DataFrame, c: str) -> float:
    fila = tabla[tabla.constituyente == c]
    return float(fila.amplitud_m.iloc[0]) if len(fila) else np.nan


def rangos_observados(serie: pd.Series) -> tuple[float, float]:
    """Rango diario (máx − mín) promediado en sicigia y cuadratura.

    Se divide la serie en ciclos de 14.77 días y en cada uno se toma el mayor y el
    menor rango diario; luego se promedian. Con datos horarios el rango se subestima
    unos pocos cm.
    """
    d = serie.dropna()
    diario = (d.resample("D").max() - d.resample("D").min())[d.resample("D").count() >= 20]
    if len(diario) < 10:
        return np.nan, np.nan
    bloque = ((diario.index - diario.index[0]).days // CICLO_SICIGIA_DIAS).astype(int)
    g = diario.groupby(bloque)
    validos = g.size() >= 10
    return float(g.max()[validos].mean()), float(g.min()[validos].mean())


def resumen(tabla, obs, pred, nombre) -> dict:
    A = {c: amplitud(tabla, c) for c in ["M2", "S2", "N2", "K1", "O1", "P1"]}
    F = (A["K1"] + A["O1"]) / (A["M2"] + A["S2"])
    sic_obs, cua_obs = rangos_observados(obs)
    sic_pred, cua_pred = rangos_observados(pred.where(obs.notna()))
    r = {"serie": nombre, **{f"A_{k}_m": v for k, v in A.items()},
         "factor_forma_F": F, "clasificacion": clasificar(F),
         "rango_sicigia_2(M2+S2)_m": 2 * (A["M2"] + A["S2"]),
         "rango_cuadratura_2(M2-S2)_m": 2 * (A["M2"] - A["S2"]),
         "rango_maximo_2(M2+S2+K1+O1)_m": 2 * (A["M2"] + A["S2"] + A["K1"] + A["O1"]),
         "rango_sicigia_observado_m": sic_obs, "rango_cuadratura_observado_m": cua_obs,
         "rango_sicigia_predicho_m": sic_pred, "rango_cuadratura_predicho_m": cua_pred,
         "varianza_explicada_pct": 100 * (1 - np.nanvar(obs - pred) / np.nanvar(obs)),
         "rms_residuo_m": float(np.sqrt(np.nanmean((obs - pred) ** 2)))}
    log.info(f"[{nombre}] F = (K1+O1)/(M2+S2) = ({A['K1']:.3f}+{A['O1']:.3f})/({A['M2']:.3f}+{A['S2']:.3f}) "
             f"= {F:.3f} -> marea {r['clasificacion'].upper()}")
    log.info(f"[{nombre}] Rango de sicigia 2(M2+S2) = {r['rango_sicigia_2(M2+S2)_m']:.2f} m "
             f"(observado {sic_obs:.2f} m); cuadratura 2(M2−S2) = {r['rango_cuadratura_2(M2-S2)_m']:.2f} m "
             f"(observado {cua_obs:.2f} m)")
    if sic_obs > 1.2 * r["rango_sicigia_2(M2+S2)_m"]:
        log.info(f"[{nombre}] El rango observado supera 2(M2+S2) porque la marea es mixta: las diurnas K1 y O1 "
                 f"se suman a las semidiurnas. Cota 2(M2+S2+K1+O1) = {r['rango_maximo_2(M2+S2+K1+O1)_m']:.2f} m")
    log.info(f"[{nombre}] Varianza explicada {r['varianza_explicada_pct']:.1f} %, RMS del residuo "
             f"{100 * r['rms_residuo_m']:.1f} cm")
    if not (0.3 <= r["rango_sicigia_2(M2+S2)_m"] <= 2.5):
        log.warning(f"[{nombre}] [sanidad] rango de sicigia fuera de lo esperado para Pisco (~1 m): REVISAR")
    else:
        log.info(f"[{nombre}] [sanidad] rango de sicigia del orden de 1 m -> OK")
    return r


def espectro(residuo: pd.Series):
    """Espectro de Welch del residuo (tramo continuo más largo, huecos ≤ 6 h interpolados)."""
    r = residuo.interpolate(limit=6, limit_area="inside")
    grupos = r.isna().cumsum()
    validos = r[r.notna()]
    largo = validos.groupby(grupos[r.notna()]).size()
    g = largo.idxmax()
    tramo = validos[grupos[r.notna()] == g]
    nper = min(len(tramo), 24 * 60)  # segmentos de 60 días
    f, P = signal.welch(tramo.values - tramo.mean(), fs=24.0, nperseg=nper, detrend="linear")  # fs en ciclos/día
    return f, P, len(tramo)


# ---------------------------------------------------------------------------
def fig_amplitudes(t_ioc, t_uh, etiq_ioc, etiq_uh, destacar, figdir):
    top = t_ioc[t_ioc.significativa].head(14).constituyente.tolist()
    for c in destacar:
        if c not in top and c in t_ioc.constituyente.values:
            top.append(c)
    x = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    a_ioc = [amplitud(t_ioc, c) for c in top]
    a_uh = [amplitud(t_uh, c) for c in top]
    e_ioc = [float(t_ioc[t_ioc.constituyente == c].amplitud_ic95_m.iloc[0]) for c in top]
    ax.bar(x - 0.2, a_ioc, 0.4, yerr=e_ioc, capsize=2, label=etiq_ioc)
    ax.bar(x + 0.2, a_uh, 0.4, label=etiq_uh, alpha=0.8)
    ax.set_xticks(x, [f"$\\bf{{{c}}}$" if c in destacar else c for c in top])
    ax.set_ylabel("Amplitud (m)")
    ax.set_title("Amplitud de las principales constituyentes de marea en Pisco (barras: IC 95 %)")
    ax.legend()
    guardar(fig, figdir, "marea_amplitudes_constituyentes", log)


def fig_obs_pred(obs, pred, nombre, figdir, archivo):
    res = obs - pred
    fig, axs = plt.subplots(3, 1, figsize=(13, 9))
    axs[0].plot(obs.index, obs, lw=0.4, label="Observado")
    axs[0].plot(pred.index, pred, lw=0.4, alpha=0.8, label="Predicho (UTide)")
    axs[0].set_ylabel("Nivel (m)")
    axs[0].set_title(f"Nivel del mar en Pisco – {nombre}")
    axs[0].legend(fontsize=8, loc="upper right")
    # Zoom de 30 días en el centro de la serie
    c = obs.index[len(obs) // 2]
    z = slice(c - pd.Timedelta(days=15), c + pd.Timedelta(days=15))
    axs[1].plot(obs.loc[z].index, obs.loc[z], "k.", ms=2, label="Observado")
    axs[1].plot(pred.loc[z].index, pred.loc[z], "r-", lw=1, label="Predicho")
    axs[1].set_ylabel("Nivel (m)")
    axs[1].set_title("Detalle de 30 días (ciclo sicigia–cuadratura)")
    axs[1].legend(fontsize=8, loc="upper right")
    axs[2].plot(res.index, res, lw=0.4, color="tab:green")
    axs[2].plot(res.rolling(25, center=True, min_periods=13).mean(), color="darkgreen", lw=1.2,
                label="Media móvil de 25 h")
    axs[2].axhline(0, color="k", lw=0.5)
    axs[2].set_ylabel("Residuo (m)")
    axs[2].set_xlabel("Fecha (UTC)")
    axs[2].set_title("Residuo = observado − predicho")
    axs[2].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    guardar(fig, figdir, archivo, log)


def fig_espectro(espectros, figdir):
    fig, ax = plt.subplots(figsize=(9, 5))
    for nombre, (f, P, n) in espectros.items():
        ax.loglog(f[1:], P[1:], lw=1, label=f"{nombre} ({n / 24:.0f} días continuos)")
    for fr, et in [(1, "1 cpd (diurna)"), (1.93, "M2"), (2, "S2"), (3.86, "M4")]:
        ax.axvline(fr, color="0.5", ls=":", lw=0.8)
        ax.text(fr, ax.get_ylim()[1], et, rotation=90, va="top", ha="right", fontsize=7)
    ax.set_xlabel("Frecuencia (ciclos por día)")
    ax.set_ylabel("Densidad espectral (m²/cpd)")
    ax.set_title("Espectro de potencia del residuo (Welch)")
    ax.legend(fontsize=8)
    guardar(fig, figdir, "marea_espectro_residuo", log)


def main():
    args = parser_base("Análisis armónico de la marea (UTide)").parse_args()
    cfg = cargar_config(args.config)
    figdir = carpeta(cfg, "carpeta_figuras")
    procdir = carpeta(cfg, "carpeta_procesados")
    lat = cfg["mareografo"]["lat"]
    destacar = cfg["analisis_marea"]["destacar"]
    with open(procdir / "seleccion_marea.yaml", encoding="utf-8") as f:
        sel = yaml.safe_load(f)

    ioc_all = pd.read_csv(procdir / "marea_ioc_horaria.csv", index_col=0, parse_dates=True).nivel_m
    obs_ioc = ioc_all.loc[sel["ioc"]["inicio"]:sel["ioc"]["fin"]]
    # Solo el tramo con datos válidos (tras el QC pueden quedar pocos meses)
    obs_ioc = obs_ioc.loc[obs_ioc.first_valid_index():obs_ioc.last_valid_index()]
    uh_all = pd.read_csv(procdir / "marea_uhslc_horaria.csv", index_col=0, parse_dates=True).nivel_m
    anio = sel["uhslc"]["anio"]
    obs_uh = uh_all.loc[str(anio)]
    etiq_ioc = f"IOC {obs_ioc.index[0]:%Y-%m} a {obs_ioc.index[-1]:%Y-%m}"
    etiq_uh = f"UHSLC {anio}"

    principal = sel.get("principal", "ioc")
    log.info(f"Serie principal: {principal.upper()} (definida en 05 según el control de calidad)")
    series = [(etiq_ioc, obs_ioc, "ioc"), (etiq_uh, obs_uh, "uhslc")]
    if principal == "uhslc":
        series = series[::-1]

    resultados, espectros, tablas = [], {}, {}
    for nombre, obs, archivo in series:
        coef, tabla, pred = analizar(obs, lat, nombre)
        tabla["destacada"] = tabla.constituyente.isin(destacar)
        tabla.round(5).to_csv(procdir / f"armonicos_{archivo}.csv", index=False)
        log.info(f"[{nombre}] Constituyentes destacadas:\n"
                 + tabla[tabla.destacada][["constituyente", "amplitud_m", "amplitud_ic95_m", "fase_g_grados",
                                           "fase_ic95_grados", "snr"]].round(3).to_string(index=False))
        obs_c = obs - coef["mean"]
        pred_c = pred - coef["mean"]
        r = resumen(tabla, obs_c, pred_c, nombre)
        r["principal"] = archivo == principal
        resultados.append(r)
        fig_obs_pred(obs_c, pred_c, nombre, figdir, f"marea_observado_predicho_{archivo}")
        espectros[nombre] = espectro(obs_c - pred_c)
        pd.DataFrame({"observado_m": obs_c, "predicho_m": pred_c, "residuo_m": obs_c - pred_c}) \
            .round(4).to_csv(procdir / f"marea_residuo_{archivo}.csv")
        tablas[archivo] = tabla

    pd.DataFrame(resultados).round(4).to_csv(procdir / "resumen_marea.csv", index=False)

    # Comparación IOC vs UHSLC (control de calidad de los datos IOC)
    t1 = tablas["ioc"].set_index("constituyente")
    t2 = tablas["uhslc"].set_index("constituyente")
    comunes = [c for c in t1.index if c in t2.index]
    comp = pd.DataFrame({
        "A_ioc_m": t1.loc[comunes, "amplitud_m"], "A_uhslc_m": t2.loc[comunes, "amplitud_m"],
        "g_ioc": t1.loc[comunes, "fase_g_grados"], "g_uhslc": t2.loc[comunes, "fase_g_grados"],
    })
    comp["dA_cm"] = 100 * (comp.A_ioc_m - comp.A_uhslc_m)
    comp["dg_grados"] = diferencia_angular(comp.g_ioc, comp.g_uhslc)
    comp = comp.sort_values("A_ioc_m", ascending=False)
    comp.round(4).to_csv(procdir / "armonicos_comparacion_ioc_uhslc.csv")
    log.info(f"Comparación {etiq_ioc} vs {etiq_uh} (constituyentes destacadas):\n"
             + comp.loc[[c for c in destacar if c in comp.index]].round(3).to_string())
    dias_ioc = obs_ioc.notna().sum() / 24
    if dias_ioc < 183:
        log.info(f"Nota: el registro IOC útil tiene {dias_ioc:.0f} días; con menos de ~6 meses UTide no separa "
                 f"K1 de P1 ni S2 de K2 (criterio de Rayleigh), por eso K1 y S2 del IOC difieren más de UHSLC")
    m2 = comp.loc["M2"] if "M2" in comp.index else None
    if m2 is not None and (abs(m2.dA_cm) > 3 or abs(m2.dg_grados) > 5):
        log.warning("M2 difiere más de 3 cm o 5° entre IOC y UHSLC: revisar datos IOC (referencia de tiempo o nivel)")
    elif m2 is not None:
        log.info(f"[sanidad] M2 IOC vs UHSLC: ΔA = {m2.dA_cm:+.1f} cm, Δg = {m2.dg_grados:+.1f}° -> OK")

    fig_amplitudes(tablas["ioc"], tablas["uhslc"], etiq_ioc, etiq_uh, destacar, figdir)
    fig_espectro(espectros, figdir)
    log.info("Análisis armónico terminado")


if __name__ == "__main__":
    main()
