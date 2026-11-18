#!/usr/bin/env python
"""07b - Control externo del mareógrafo de Pisco con estaciones vecinas y FES2022.

¿Por qué? FES2022 daba amplitudes ~26 % menores que UHSLC Pisco 2022 en TODAS las constituyentes
(misma razón para diurnas y semidiurnas, fases iguales). Para saber de qué lado está el error,
se repite la comparación en los mareógrafos vecinos (Callao al norte, San Juan de Marcona al sur)
y se sigue la amplitud de M2 año por año en los tres:

  * si FES2022 coincide con los vecinos, el modelo es confiable en esta costa;
  * si Pisco se separa de sus vecinos solo en algunos años, con el mismo factor en todas las
    constituyentes, es un error de escala del registro (calibración), no física de la bahía:
    una amplificación real dependería de la frecuencia.

Lee los archivos globales de FES2022b (variables 'amplitude' en cm y 'phase' en grados,
verificado en los NetCDF) porque Callao y San Juan quedan fuera del recorte regional de 07.

Salidas (data/processed/):
  control_vecinos_m2_anual.csv   amplitud anual de M2 en las tres estaciones y razón Pisco/vecinos
  control_vecinos_fes.csv        FES2022 vs UTide por estación en el año de comparación
  control_vecinos_nivel_medio.csv  nivel medio mensual del IOC de Pisco frente a Callao (fast delivery):
                                   si suben y bajan juntos, la variación es oceanográfica y no del sensor
Figuras: figures/marea_control_vecinos.png, figures/marea_nivel_medio_pisco_callao.png

Uso:
    python scripts/07b_control_vecinos.py      (después de 00, 05 y 06)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import utide
import xarray as xr
import yaml

from utils.config import cargar_config, carpeta, parser_base, ruta
from utils.graficos import guardar, plt
from utils.log import obtener_logger
from utils.viento import diferencia_angular

log = obtener_logger("07b_control_vecinos")
COLORES = {"Callao": "tab:blue", "Pisco": "crimson", "San Juan de Marcona": "tab:green"}


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def leer_uhslc(url: str, destino: Path, faltante: int, forzar: bool) -> pd.Series:
    """Descarga (una vez) y lee un archivo horario UHSLC: año, mes, día, hora, nivel en mm."""
    if not destino.exists() or forzar:
        log.info(f"Descargando {url}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        destino.write_bytes(r.content)
    d = pd.read_csv(destino, header=None, names=["anio", "mes", "dia", "hora", "nivel_mm"])
    t = pd.to_datetime(dict(year=d.anio, month=d.mes, day=d.dia, hour=d.hora))
    s = pd.Series(d.nivel_mm.where(d.nivel_mm != faltante).values / 1000.0, index=t, name="nivel_m")
    return s.astype(float)


def estaciones(cfg) -> list[dict]:
    """Pisco (archivo de 05, posición IOC) más los vecinos del config."""
    uh = cfg["uhslc"]
    pisco = {"nombre": "Pisco", "id": uh["id"], "lat": cfg["mareografo"]["lat"], "lon": cfg["mareografo"]["lon"],
             "url": uh["url_rq"]}
    return [cfg["control_vecinos"]["estaciones"][0], pisco] + cfg["control_vecinos"]["estaciones"][1:]


def fes_en_punto(directorio: Path, constituyentes: list[str], lat: float, lon: float) -> dict:
    """Amplitud (m) y fase (°) de FES2022 interpolando la parte compleja (evita saltos de fase)."""
    salida = {}
    lon360 = lon % 360
    for c in constituyentes:
        archivo = directorio / f"{c.lower()}_fes2022.nc"
        if not archivo.exists():
            log.warning(f"Falta {archivo.name}: no se compara {c}")
            continue
        with xr.open_dataset(archivo) as ds:
            sub = ds.sel(lat=slice(lat - 0.1, lat + 0.1), lon=slice(lon360 - 0.1, lon360 + 0.1)).load()
            escala = 0.01 if ds.amplitude.attrs.get("units", "cm") == "cm" else 1.0
        z = (sub.amplitude * np.exp(-1j * np.deg2rad(sub.phase))).interp(lat=lat, lon=lon360).item()
        if not np.isfinite(z):
            log.warning(f"FES2022 sin dato en ({lat}, {lon}) para {c}")
            continue
        salida[c] = (abs(z) * escala, (-np.rad2deg(np.angle(z))) % 360)
    return salida


def armonicos(s: pd.Series, lat: float, constit) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        c = utide.solve(s.index.values, s.values, lat=lat, method="ols", conf_int="none", trend=False,
                        nodal=True, constit=constit, verbose=False)
    return {n: (a, g) for n, a, g in zip(c["name"], c["A"], c["g"])}


# ---------------------------------------------------------------------------
def main():
    args = parser_base("Control del mareógrafo de Pisco con Callao, San Juan y FES2022").parse_args()
    cfg = cargar_config(args.config)
    cv = cfg["control_vecinos"]
    procdir = carpeta(cfg, "carpeta_procesados")
    figdir = carpeta(cfg, "carpeta_figuras")
    raw = ruta(cfg["uhslc"]["carpeta_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    cons = cv["constituyentes"]
    mm = cfg["modelo_marea"]
    dir_fes = ruta(mm["directorio"]) / Path(mm["archivos"]).parent

    anual, comp, fes_m2 = {}, [], {}
    for e in estaciones(cfg):
        s = leer_uhslc(e["url"], raw / f"h{e['id']}_rq.csv", cfg["uhslc"]["valor_faltante"], args.forzar)
        log.info(f"{e['nombre']} (UHSLC {e['id']}): {s.index[0]:%Y-%m-%d} a {s.index[-1]:%Y-%m-%d}, "
                 f"{100 * s.notna().mean():.1f} % válidas")

        fes = fes_en_punto(dir_fes, cons, e["lat"], e["lon"])
        if "M2" in fes:
            fes_m2[e["nombre"]] = fes["M2"][0]

        # Amplitud anual de M2
        horas_anio = s.groupby(s.index.year).size()
        comp_anual = s.notna().groupby(s.index.year).sum() / horas_anio.clip(lower=8760)
        m2 = {}
        for anio in comp_anual[comp_anual >= cv["completitud_minima_anual"]].index:
            m2[anio] = armonicos(s.loc[str(anio)].dropna(), e["lat"], ["M2", "S2", "K1", "O1"])["M2"][0]
        anual[e["nombre"]] = pd.Series(m2)

        # FES2022 vs UTide en el año común
        anio = cv["anio_comparacion"]
        obs_a = s.loc[str(anio)].dropna() if str(anio) in s.index.year.astype(str) else pd.Series(dtype=float)
        if len(obs_a) < cv["completitud_minima_anual"] * 8760:
            log.warning(f"{e['nombre']}: {anio} sin datos suficientes, no se compara con FES2022")
            continue
        obs = armonicos(obs_a, e["lat"], "auto")
        for c in cons:
            if c in obs and c in fes:
                comp.append({"estacion": e["nombre"], "constituyente": c, "A_obs_m": obs[c][0], "A_fes_m": fes[c][0],
                             "razon_fes_obs": fes[c][0] / obs[c][0],
                             "dg_grados": float(diferencia_angular(np.array([fes[c][1]]), np.array([obs[c][1]]))[0])})

    # --- Tablas -----------------------------------------------------------------
    t = pd.DataFrame(anual).sort_index()
    t.index.name = "anio"
    vecinos = [n for n in t.columns if n != "Pisco"]
    if set(fes_m2) >= set(t.columns):
        # Cada estación se normaliza por su M2 de FES2022: así la diferencia natural entre sitios
        # (San Juan tiene más marea que Callao) no se confunde con un error del registro
        norm = t / pd.Series(fes_m2)[t.columns]
        t["razon_pisco_vecinos"] = norm["Pisco"] / norm[vecinos].mean(axis=1)
        log.info("Razón Pisco/vecinos normalizada por el M2 de FES2022 en cada estación")
    else:
        t["razon_pisco_vecinos"] = t["Pisco"] / t[vecinos].mean(axis=1)
        log.warning("Sin FES2022 en todas las estaciones: razón Pisco/vecinos sin normalizar")
    t["sospechoso"] = (t.razon_pisco_vecinos - 1).abs() > cv["tolerancia_razon"]
    t.round(4).to_csv(procdir / "control_vecinos_m2_anual.csv")
    sosp = t.index[t.sospechoso].tolist()
    log.info("Amplitud anual de M2 (cm):\n" + (100 * t[["Callao", "Pisco", "San Juan de Marcona"]]).round(1)
             .join(t.razon_pisco_vecinos.round(2)).dropna(how="all").to_string())
    if sosp:
        r = t.loc[t.sospechoso, "razon_pisco_vecinos"]
        log.warning(f"Años de Pisco fuera de ±{100 * cv['tolerancia_razon']:.0f} % de sus vecinos: {sosp} "
                    f"(razón media {r.mean():.2f}). Revisar el año de referencia en config.yaml (uhslc.anio_referencia)")
    ref = cfg["uhslc"]["anio_referencia"]
    if str(ref) != "auto" and int(ref) in sosp:
        log.warning(f"[sanidad] el año de referencia {ref} está marcado como sospechoso")
    elif str(ref) != "auto":
        log.info(f"[sanidad] año de referencia {ref}: razón Pisco/vecinos = "
                 f"{t.loc[int(ref), 'razon_pisco_vecinos']:.2f} -> OK")

    c = pd.DataFrame(comp)
    if len(c):
        c.round(4).to_csv(procdir / "control_vecinos_fes.csv", index=False)
        res = c.groupby("estacion").agg(razon_media=("razon_fes_obs", "mean"), razon_min=("razon_fes_obs", "min"),
                                        razon_max=("razon_fes_obs", "max"), dg_medio=("dg_grados", "mean"))
        log.info(f"FES2022 / UTide en {cv['anio_comparacion']}:\n" + res.round(3).to_string())
        for est, f in res.iterrows():
            if est != "Pisco" and abs(f.razon_media - 1) > cv["tolerancia_razon"]:
                log.warning(f"FES2022 se aleja más de {100 * cv['tolerancia_razon']:.0f} % de {est}: "
                            "el modelo podría no ser confiable en esta costa")

    # IOC: M2 de la ventana analizada en 06 (si existe), para el gráfico
    ioc = None
    pa, ps = procdir / "armonicos_ioc.csv", procdir / "seleccion_marea.yaml"
    if pa.exists() and ps.exists():
        a = pd.read_csv(pa).set_index("constituyente")
        with open(ps, encoding="utf-8") as f:
            sel = yaml.safe_load(f)["ioc"]
        medio = pd.Timestamp(sel["inicio"]) + (pd.Timestamp(sel["fin"]) - pd.Timestamp(sel["inicio"])) / 2
        if "M2" in a.index:
            ioc = (medio.year + (medio.dayofyear - 1) / 365.25, a.loc["M2", "amplitud_m"],
                   f"IOC {pd.Timestamp(sel['inicio']):%Y-%m} a {pd.Timestamp(sel['fin']):%Y-%m}")

    fig_control(t, c, ioc, sosp, cv, figdir)
    nivel_medio(cfg, raw, procdir, figdir, args.forzar)


def nivel_medio(cfg, raw, procdir, figdir, forzar):
    """Anomalía mensual del nivel medio: IOC Pisco (tras QC) frente a un vecino con 'fast delivery'."""
    vec = next((e for e in cfg["control_vecinos"]["estaciones"] if e.get("url_fd")), None)
    p_ioc = procdir / "marea_ioc_horaria.csv"
    if vec is None or not p_ioc.exists():
        log.info("Sin vecino con fast delivery o sin serie IOC: no se compara el nivel medio")
        return
    fd = leer_uhslc(vec["url_fd"], raw / f"h{vec['id']}_fd.csv", cfg["uhslc"]["valor_faltante"], forzar)
    ioc = pd.read_csv(p_ioc, index_col=0, parse_dates=True).nivel_m
    meses = lambda x: x.groupby(x.index.to_period("M")).agg(["mean", "count"])
    a, b = meses(ioc.dropna()), meses(fd.loc[ioc.index[0]:ioc.index[-1]].dropna())
    t = pd.DataFrame({"pisco_ioc_m": a["mean"].where(a["count"] >= 15 * 24),
                      f"{vec['nombre'].lower()}_m": b["mean"].where(b["count"] >= 15 * 24)}).dropna()
    if len(t) < 4:
        log.info("Pocos meses en común para comparar el nivel medio")
        return
    t = t - t.mean()  # anomalías respecto de la media de los meses comunes
    t.index = t.index.astype(str)
    t.index.name = "mes"
    t.round(4).to_csv(procdir / "control_vecinos_nivel_medio.csv")
    r = t.corr().iloc[0, 1]
    dif = (t.iloc[:, 0] - t.iloc[:, 1]).abs().max()
    log.info(f"Nivel medio mensual Pisco IOC vs {vec['nombre']} ({len(t)} meses): r = {r:.2f}, "
             f"rango Pisco = {100 * (t.iloc[:, 0].max() - t.iloc[:, 0].min()):.0f} cm, "
             f"diferencia máxima = {100 * dif:.1f} cm")
    if r < 0.8:
        log.warning("El nivel medio del IOC no sigue al de la estación vecina: posible deriva del sensor")
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(t))
    ax.plot(x, 100 * t.iloc[:, 0], "o-", color="crimson", label="Pisco (IOC, tras QC)")
    ax.plot(x, 100 * t.iloc[:, 1], "s-", color="tab:blue", label=f"{vec['nombre']} (UHSLC fast delivery)")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x, t.index, rotation=45, fontsize=8)
    ax.set_ylabel("Anomalía del nivel medio mensual (cm)")
    ax.set_title(f"Nivel medio mensual: Pisco frente a {vec['nombre']} (r = {r:.2f})")
    ax.legend(fontsize=8)
    guardar(fig, figdir, "marea_nivel_medio_pisco_callao", log)


def fig_control(t, c, ioc, sosp, cv, figdir):
    fig, axs = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.7, 1]})
    ax = axs[0]
    for a in sosp:
        ax.axvspan(a - 0.5, a + 0.5, color="crimson", alpha=0.07, lw=0)
    for est in ["Callao", "Pisco", "San Juan de Marcona"]:
        if est in t:
            # Se completan los años faltantes con NaN para que la línea se corte en los huecos
            s = t[est].reindex(range(t.index.min(), t.index.max() + 1))
            ax.plot(s.index, 100 * s, "o-", ms=3, color=COLORES[est], label=f"{est} (UHSLC)")
    if len(c):
        for est, sub in c[c.constituyente == "M2"].groupby("estacion"):
            ax.axhline(100 * sub.A_fes_m.iloc[0], color=COLORES.get(est, "k"), ls="--", lw=1)
        ax.plot([], [], "k--", lw=1, label="FES2022 en cada estación")
    if ioc is not None:
        ax.plot(ioc[0], 100 * ioc[1], "D", color="crimson", mec="k", ms=8, label=f"Pisco, {ioc[2]}")
    ax.set_xlabel("Año")
    ax.set_ylabel("Amplitud de M2 (cm)")
    ax.set_title("Amplitud anual de M2 (sombreado: años de Pisco fuera de ±10 % de sus vecinos)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")

    ax = axs[1]
    if len(c):
        ests = list(dict.fromkeys(c.estacion))
        cons = cv["constituyentes"]
        ancho = 0.8 / len(ests)
        for i, est in enumerate(ests):
            s = c[c.estacion == est].set_index("constituyente").reindex(cons)
            ax.bar(np.arange(len(cons)) + (i - (len(ests) - 1) / 2) * ancho, s.razon_fes_obs, ancho,
                   color=COLORES.get(est, "0.5"), label=est)
        ax.axhline(1, color="k", lw=0.8)
        ax.axhspan(1 - cv["tolerancia_razon"], 1 + cv["tolerancia_razon"], color="green", alpha=0.08)
        ax.set_xticks(range(len(cons)), cons)
        ax.set_ylim(0, 1.3)
        ax.set_ylabel("Amplitud FES2022 / observada")
        ax.set_title(f"FES2022 frente a UTide, año {cv['anio_comparacion']}", fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Control del mareógrafo de Pisco con estaciones vecinas y FES2022", fontweight="bold")
    guardar(fig, figdir, "marea_control_vecinos", log)


if __name__ == "__main__":
    main()
