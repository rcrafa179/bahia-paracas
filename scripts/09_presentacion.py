#!/usr/bin/env python
"""09 - Presentación Beamer con los resultados.

Lee los resultados de data/processed/ y escribe presentacion/resultados.tex
(cifras y tablas como macros de LaTeX); luego compila presentacion/paracas.tex
con latexmk. Las figuras o cifras que aún no existan aparecen como "pendiente".

Uso:
    python scripts/09_presentacion.py [--sin-compilar]
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import yaml

from utils.config import RAIZ, cargar_config, carpeta, parser_base
from utils.log import obtener_logger

log = obtener_logger("09_presentacion")
PEND = r"\pendiente"


def fmt(x, nd=2, unidad=""):
    try:
        if x is None or not np.isfinite(float(x)):
            return PEND
    except (TypeError, ValueError):
        return str(x)
    return f"{float(x):.{nd}f}{unidad}"


def tex(texto) -> str:
    """Escapa los caracteres especiales de LaTeX en textos que vienen de los resultados."""
    texto = str(texto)
    for c in ["&", "%", "$", "#", "_"]:
        texto = texto.replace(c, "\\" + c)
    return texto


def leer_csv(p, **kw):
    return pd.read_csv(p, **kw) if p.exists() else None


def leer_yaml(p):
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def macros(cfg, proc) -> dict[str, str]:
    m = {}
    m["periodoMeteo"] = f"{cfg['periodo']['inicio']} a {cfg['periodo']['fin']}"
    vp = cfg["vientos_paracas"]
    m["umbralParacas"] = f"{vp['umbral_ms']:g}"
    m["ventanaParacas"] = f"{vp['hora_local_inicio']}--{vp['hora_local_fin']}"

    cand = leer_yaml(proc / "mes_candidato.yaml") or {}
    m["mesCandidato"] = cand.get("mes", PEND)
    m["eventosCandidato"] = str(cand.get("eventos", PEND))
    m["criterioCandidato"] = tex(cand["criterio"]) if "criterio" in cand else PEND
    m["fuenteCandidato"] = str(cand.get("fuente", PEND)).upper()

    comp = leer_csv(proc / "eventos_paracas_mensual_comparacion.csv")
    m["eventosMetar"] = str(int(comp.eventos_metar.sum())) if comp is not None else PEND
    m["eventosEra"] = str(int(comp.eventos_era5.sum())) if comp is not None else PEND
    if comp is not None and len(comp):
        top = comp.sort_values("eventos_metar", ascending=False).iloc[0]
        m["mesMaxMetar"] = f"{top['mes']} ({int(top.eventos_metar)} días)"
    else:
        m["mesMaxMetar"] = PEND

    val = leer_csv(proc / "validacion_era5_spso.csv", index_col=0)
    filas = []
    if val is not None:
        for var, fila in val.iterrows():
            filas.append(f"{var.replace('%', r'\%')} & {int(fila.n)} & {fmt(fila.sesgo)} & {fmt(fila.rmse)} & "
                         f"{fmt(fila.r) if np.isfinite(fila.r) else '--'} \\\\")
        fuerte = [v for v in val.index if v.startswith("velocidad cuando")]
        if fuerte:
            f = val.loc[fuerte[0]]
            m["vientoFuerteObs"], m["vientoFuerteEra"] = fmt(f.media_obs, 1), fmt(f.media_era5, 1)
    m["tablaValidacion"] = "\n".join(filas) if filas else PEND
    m.setdefault("vientoFuerteObs", PEND)
    m.setdefault("vientoFuerteEra", PEND)

    # Marea
    sel = leer_yaml(proc / "seleccion_marea.yaml") or {}
    m["seriePrincipal"] = str(sel.get("principal", PEND)).upper()
    m["anioUhslc"] = str(sel.get("uhslc", {}).get("anio", PEND))
    qc = leer_csv(proc / "marea_ioc_qc_mensual.csv")
    if qc is not None:
        p = qc[qc.estacion == cfg["ioc"]["estaciones"][0]]
        m["qcAprobados"] = f"{int(p.aprobado.sum())} de {len(p)}"
        m["pendienteIoc"] = fmt(p.loc[~p.aprobado, "pendiente"].median(), 2)
    else:
        m["qcAprobados"] = m["pendienteIoc"] = PEND
    res = leer_csv(proc / "resumen_marea.csv")
    if res is not None:
        r = res[res.principal].iloc[0] if "principal" in res and res.principal.any() else res.iloc[0]
        m["factorF"] = fmt(r.factor_forma_F, 2)
        m["claseMarea"] = r.clasificacion
        for c in ["M2", "S2", "N2", "K1", "O1", "P1"]:
            m[f"amp{c.replace('2', 'dos').replace('1', 'uno')}"] = fmt(100 * r[f"A_{c}_m"], 1)
        m["rangoSicigia"] = fmt(r["rango_sicigia_2(M2+S2)_m"])
        m["rangoCuadratura"] = fmt(r["rango_cuadratura_2(M2-S2)_m"])
        m["rangoMaximo"] = fmt(r["rango_maximo_2(M2+S2+K1+O1)_m"])
        m["rangoSicigiaObs"] = fmt(r.rango_sicigia_observado_m)
        m["rangoCuadraturaObs"] = fmt(r.rango_cuadratura_observado_m)
        m["varExplicada"] = fmt(r.varianza_explicada_pct, 1)
        m["rmsResiduo"] = fmt(100 * r.rms_residuo_m, 1)
        m["seriePrincipalNombre"] = r.serie
    arm_nombre = "armonicos_uhslc.csv" if m["seriePrincipal"] == "UHSLC" else "armonicos_ioc.csv"
    arm = leer_csv(proc / arm_nombre)
    if arm is not None:
        dest = arm[arm.constituyente.isin(cfg["analisis_marea"]["destacar"])]
        m["tablaArmonicos"] = "\n".join(
            f"{f.constituyente} & {f.periodo_h:.2f} & {100 * f.amplitud_m:.1f} $\\pm$ {100 * f.amplitud_ic95_m:.1f} & "
            f"{f.fase_g_grados:.1f} $\\pm$ {f.fase_ic95_grados:.1f} & {f.snr:.0f} \\\\" for f in dest.itertuples())
    else:
        m["tablaArmonicos"] = PEND
    cmp_ = leer_csv(proc / "armonicos_comparacion_ioc_uhslc.csv", index_col=0)
    if cmp_ is not None and "M2" in cmp_.index:
        m["dAmMdosIoc"], m["dgMdosIoc"] = fmt(cmp_.loc["M2", "dA_cm"], 1), fmt(cmp_.loc["M2", "dg_grados"], 1)
    fes = leer_csv(proc / "armonicos_comparacion_fes_utide.csv")
    if fes is not None:
        f = fes[(fes.constituyente == "M2") & (fes.fuente_obs == m["seriePrincipal"])]
        if len(f):
            m["dAmMdosFes"], m["dgMdosFes"] = fmt(f.dA_cm.iloc[0], 1), fmt(f.dg_grados.iloc[0], 1)

    # Batimetría
    mor = leer_csv(proc / "morfometria_bahia.csv")
    if mor is not None:
        r = mor.iloc[0]
        m["areaBahia"], m["volBahia"] = fmt(r.area_km2, 1), fmt(r.volumen_hm3, 0)
        m["profMedia"], m["profMax"] = fmt(r.profundidad_media_m, 1), fmt(r.profundidad_maxima_m, 1)
    for k in ["areaBahia", "volBahia", "profMedia", "profMax", "dAmMdosFes", "dgMdosFes", "dAmMdosIoc", "dgMdosIoc",
              "factorF", "claseMarea", "rangoSicigia", "rangoCuadratura", "rangoMaximo", "rangoSicigiaObs",
              "rangoCuadraturaObs", "varExplicada", "rmsResiduo", "seriePrincipalNombre"] + \
             [f"amp{c.replace('2', 'dos').replace('1', 'uno')}" for c in ["M2", "S2", "N2", "K1", "O1", "P1"]]:
        m.setdefault(k, PEND)
    return m


def escribir(m: dict, destino):
    lineas = ["% Archivo generado por scripts/09_presentacion.py: no editar a mano"]
    for k, v in m.items():
        lineas.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    destino.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    log.info(f"Cifras escritas en {destino.relative_to(RAIZ)} ({len(m)} macros; "
             f"{sum(PEND in v for v in m.values())} pendientes)")


def main():
    p = parser_base("Genera y compila la presentación Beamer")
    p.add_argument("--sin-compilar", action="store_true")
    args = p.parse_args()
    cfg = cargar_config(args.config)
    proc = carpeta(cfg, "carpeta_procesados")
    pres = RAIZ / "presentacion"
    escribir(macros(cfg, proc), pres / "resultados.tex")
    if args.sin_compilar:
        return
    if shutil.which("latexmk") is None:
        raise SystemExit("No se encontró latexmk (TeX Live)")
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "paracas.tex"],
                       cwd=pres, capture_output=True, text=True)
    if r.returncode != 0:
        log.error(r.stdout[-3000:])
        raise SystemExit("Error al compilar la presentación (ver presentacion/paracas.log)")
    pdf = pres / "paracas.pdf"
    log.info(f"Presentación compilada: {pdf.relative_to(RAIZ)} ({pdf.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
