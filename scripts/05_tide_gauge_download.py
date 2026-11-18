#!/usr/bin/env python
"""05 - Descarga y control de calidad del nivel del mar en Pisco.

Dos fuentes (verificadas):
  a) IOC Sea Level Monitoring Facility (tiempo real, SIN control de calidad), estaciones
     'pisco' (radar) y 'pisc2' (presión), 1 dato por minuto, API service.php?query=data.
     Es la fuente principal para el periodo 2025–2026.
  b) UHSLC, research quality, estación 683 (Pisco), horario, 1991–2023-04. Se usa como
     referencia de calidad (año más completo). Faltante = -32767, unidades mm.

Salidas: data/processed/marea_ioc_horaria.csv, marea_uhslc_horaria.csv,
         seleccion_marea.yaml y figuras de cobertura.

Uso:
    python scripts/05_tide_gauge_download.py [--inicio 2025-01-01 --fin 2025-01-31]
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import requests
import yaml

from utils import sanidad
from utils.config import cargar_config, carpeta, parser_base, periodo, ruta
from utils.graficos import MESES, guardar, plt
from utils.log import obtener_logger

log = obtener_logger("05_tide_gauge")


# ---------------------------------------------------------------------------
# IOC
# ---------------------------------------------------------------------------
def descargar_ioc_mes(cfg, codigo: str, mes: pd.Period, forzar: bool) -> pd.DataFrame:
    """Descarga un mes de datos de 1 minuto (todas las series del sensor) y lo guarda comprimido."""
    raw = ruta(cfg["ioc"]["carpeta_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    destino = raw / f"{codigo}_{mes.year}{mes.month:02d}.csv.gz"
    if destino.exists() and not forzar:
        return pd.read_csv(destino, parse_dates=["stime"])
    params = {"query": "data", "code": codigo, "format": "json",
              "timestart": f"{mes.start_time:%Y-%m-%d}", "timestop": f"{(mes + 1).start_time:%Y-%m-%d}"}
    for intento in range(3):
        try:
            r = requests.get(cfg["ioc"]["url"], params=params, timeout=180)
            r.raise_for_status()
            datos = r.json()
            break
        except Exception as e:  # noqa: BLE001
            log.warning(f"  {codigo} {mes}: intento {intento + 1} falló ({e}); reintento en 15 s")
            time.sleep(15)
    else:
        raise SystemExit(f"No se pudo descargar {codigo} {mes}")
    df = pd.DataFrame(datos, columns=["stime", "slevel", "sensor"])
    df["stime"] = pd.to_datetime(df["stime"])
    df.to_csv(destino, index=False, compression="gzip")
    return df


def qc_ioc(df: pd.DataFrame, sensor: str, nombre: str) -> tuple[pd.Series, dict]:
    """Control de calidad de la serie de 1 minuto y paso a valores horarios.

    1) solo el sensor elegido y sin duplicados;
    2) valores groseros: a más de 3 m de la mediana del mes;
    3) picos: desvío > max(0.15 m, 6·MAD) respecto de la mediana móvil de 31 min;
    4) sensor pegado: el mismo valor repetido más de 60 minutos seguidos;
    5) media horaria centrada (±30 min) con al menos 30 minutos válidos.
    """
    rep = {}
    s = df[df.sensor == sensor].drop_duplicates("stime").set_index("stime")["slevel"].astype(float).sort_index()
    rep["minutos_crudos"] = len(s)
    if s.empty:
        return pd.Series(dtype=float), rep
    # Malla regular de 1 minuto
    s = s.reindex(pd.date_range(s.index.min().floor("min"), s.index.max().ceil("min"), freq="min"))
    med_mes = s.groupby(s.index.to_period("M")).transform("median")
    groseros = (s - med_mes).abs() > 3.0
    rep["groseros"] = int(groseros.sum())
    s[groseros] = np.nan
    med = s.rolling(31, center=True, min_periods=10).median()
    dev = (s - med).abs()
    mad = dev.rolling(1441, center=True, min_periods=60).median()
    picos = dev > np.maximum(0.15, 6 * 1.4826 * mad)
    rep["picos"] = int(picos.sum())
    s[picos] = np.nan
    # Valores repetidos (sensor pegado)
    grupo = (s.diff() != 0).cumsum()
    largo = s.groupby(grupo).transform("size")
    pegado = (largo > 60) & s.notna()
    rep["pegados"] = int(pegado.sum())
    s[pegado] = np.nan
    # Media horaria centrada en la hora en punto
    desplazada = s.copy()
    desplazada.index = desplazada.index + pd.Timedelta(minutes=30)
    n = desplazada.resample("h").count()
    hor = desplazada.resample("h").mean().where(n >= 30)
    rep["horas_validas"] = int(hor.notna().sum())
    log.info(f"  QC {nombre} ({sensor}): " + ", ".join(f"{k}={v}" for k, v in rep.items()))
    return hor, rep


def combinar(principal: pd.Series, secundaria: pd.Series, horas: pd.DatetimeIndex) -> tuple[pd.DataFrame, float]:
    """Llena huecos de la principal con la secundaria ajustada al mismo nivel medio."""
    p = principal.reindex(horas)
    s = secundaria.reindex(horas)
    comun = p.notna() & s.notna()
    if comun.sum() >= 24 * 7:
        desfase = float((p[comun] - s[comun]).median())
        dif = (p[comun] - s[comun] - desfase)
        log.info(f"Desfase de nivel principal − secundaria = {desfase:+.3f} m "
                 f"({int(comun.sum())} h comunes; desv. estándar de la diferencia {dif.std():.3f} m)")
        # Estabilidad del desfase mes a mes (saltos de nivel de referencia)
        por_mes = (p[comun] - s[comun]).groupby(p[comun].index.to_period("M")).median()
        if (por_mes - desfase).abs().max() > 0.05:
            log.warning("El desfase entre estaciones varía más de 5 cm entre meses: posible cambio de referencia:\n"
                        + por_mes.round(3).to_string())
    else:
        # Sin solape suficiente: se igualan los niveles medios de cada serie
        desfase = float(p.mean() - s.mean()) if p.notna().any() and s.notna().any() else 0.0
        log.warning(f"Solape insuficiente entre estaciones; desfase por diferencia de medias = {desfase:+.3f} m")
    nivel = p.copy()
    usa_sec = p.isna() & s.notna()
    nivel[usa_sec] = s[usa_sec] + desfase
    fuente = pd.Series(np.where(p.notna(), "principal", np.where(usa_sec, "secundaria", "")), index=horas)
    return pd.DataFrame({"nivel_m": nivel, "fuente": fuente}), desfase


# ---------------------------------------------------------------------------
# UHSLC
# ---------------------------------------------------------------------------
def descargar_uhslc(cfg, forzar: bool) -> pd.Series:
    raw = ruta(cfg["uhslc"]["carpeta_raw"])
    raw.mkdir(parents=True, exist_ok=True)
    destino = raw / f"h{cfg['uhslc']['id']}_rq.csv"
    if not destino.exists() or forzar:
        log.info(f"Descargando UHSLC research quality: {cfg['uhslc']['url_rq']}")
        r = requests.get(cfg["uhslc"]["url_rq"], timeout=180)
        r.raise_for_status()
        destino.write_bytes(r.content)
    df = pd.read_csv(destino, header=None, names=["anio", "mes", "dia", "hora", "nivel_mm"])
    t = pd.to_datetime(dict(year=df.anio, month=df.mes, day=df.dia, hour=df.hora))
    s = pd.Series(df.nivel_mm.values, index=t, name="nivel_m").astype(float)
    s[s == cfg["uhslc"]["valor_faltante"]] = np.nan
    s = s / 1000.0  # mm -> m
    log.info(f"UHSLC: {len(s)} horas ({s.index[0]:%Y-%m-%d} a {s.index[-1]:%Y-%m-%d}), "
             f"{100 * s.notna().mean():.1f} % válidas")
    return s


def elegir_anio(completitud: pd.Series, fijo, tol: float) -> int:
    """Año más completo; los empates (dentro de 'tol') se resuelven por el más reciente."""
    if str(fijo) != "auto":
        return int(fijo)
    mejores = completitud[completitud >= completitud.max() - tol]
    return int(mejores.index.max())


def elegir_ventana(serie: pd.Series, cfg) -> tuple[pd.Timestamp, pd.Timestamp, float]:
    """Ventana de 365 días (inicio en día 1 de mes) con más horas válidas; empate -> la más reciente."""
    am = cfg["analisis_marea"]
    if str(am["anio_ioc"]) != "auto":
        ini = pd.Timestamp(am["anio_ioc"])
        fin = ini + pd.Timedelta(days=365) - pd.Timedelta(hours=1)
        return ini, fin, float(serie.loc[ini:fin].notna().mean())
    dur = pd.Timedelta(days=365)
    if serie.index[-1] - serie.index[0] < dur:
        # Periodo corto (p. ej. prueba de 1 mes): se usa todo
        return serie.index[0], serie.index[-1], float(serie.notna().mean())
    candidatos = []
    for ini in pd.date_range(serie.index[0], serie.index[-1] - dur + pd.Timedelta(hours=1), freq="MS"):
        fin = ini + dur - pd.Timedelta(hours=1)
        candidatos.append((ini, fin, float(serie.loc[ini:fin].notna().mean())))
    if not candidatos:
        return serie.index[0], serie.index[-1], float(serie.notna().mean())
    cmax = max(c[2] for c in candidatos)
    ini, fin, comp = [c for c in candidatos if c[2] >= cmax - am["tolerancia_empate"]][-1]
    return ini, fin, comp


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_cobertura_ioc(cob: pd.DataFrame, ventana, figdir):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(cob))
    ancho = 0.27
    for i, (col, c) in enumerate(zip(cob.columns, ["tab:blue", "tab:orange", "k"])):
        ax.bar(x + (i - 1) * ancho, 100 * cob[col], ancho, label=col, color=c, alpha=0.85)
    ax.set_xticks(x, [f"{MESES[p.month - 1]}\n{p.year}" for p in cob.index], fontsize=7)
    ax.set_ylabel("Horas válidas (%)")
    ax.set_ylim(0, 105)
    i0 = list(cob.index).index(ventana[0].to_period("M")) if ventana[0].to_period("M") in cob.index else 0
    i1 = list(cob.index).index(ventana[1].to_period("M")) if ventana[1].to_period("M") in cob.index else len(cob) - 1
    ax.axvspan(i0 - 0.5, i1 + 0.5, color="green", alpha=0.08, label="Ventana de análisis")
    ax.set_title("Cobertura mensual del mareógrafo de Pisco (IOC, tras control de calidad)")
    ax.legend(fontsize=8, ncol=4, loc="lower left")
    guardar(fig, figdir, "marea_cobertura_ioc_mensual", log)


def fig_cobertura_uhslc(comp: pd.Series, anio: int, figdir):
    fig, ax = plt.subplots(figsize=(12, 4))
    col = ["crimson" if a == anio else "tab:blue" for a in comp.index]
    ax.bar(comp.index, 100 * comp.values, color=col)
    ax.set_ylabel("Horas válidas (%)")
    ax.set_xlabel("Año")
    ax.set_ylim(0, 105)
    ax.set_title(f"Cobertura anual del mareógrafo de Pisco (UHSLC 683, research quality). "
                 f"En rojo: año de referencia {anio}")
    guardar(fig, figdir, "marea_cobertura_uhslc_anual", log)


def fig_serie(ioc: pd.DataFrame, figdir):
    fig, ax = plt.subplots(figsize=(13, 4))
    for f, c in [("principal", "tab:blue"), ("secundaria", "tab:orange")]:
        sel = ioc.nivel_m.where(ioc.fuente == f)
        ax.plot(sel.index, sel, lw=0.4, color=c, label=f"Estación {f}")
    ax.set_ylabel("Nivel del mar (m, referencia del sensor)")
    ax.set_xlabel("Fecha (UTC)")
    ax.set_title("Nivel del mar horario en Pisco – IOC tras control de calidad")
    ax.legend(fontsize=8)
    guardar(fig, figdir, "marea_serie_ioc_horaria", log)


# ---------------------------------------------------------------------------
# Control de calidad contra la referencia UHSLC
# ---------------------------------------------------------------------------
def prediccion_referencia(uh: pd.Series, anio: int, horas: pd.DatetimeIndex, lat: float) -> pd.Series:
    """Marea astronómica predicha (sin nivel medio) a partir del año de referencia UHSLC."""
    import utide

    r = uh.loc[str(anio)].dropna()
    coef = utide.solve(r.index.values, r.values, lat=lat, method="ols", conf_int="none",
                       trend=False, nodal=True, constit="auto", verbose=False)
    h = utide.reconstruct(horas.values, coef, verbose=False).h - coef["mean"]
    return pd.Series(h, index=horas)


def qc_referencia(serie: pd.Series, pred: pd.Series, nombre: str, cfg) -> tuple[pd.DataFrame, pd.Series]:
    """Compara cada mes con la marea predicha: correlación y pendiente de la regresión obs ~ pred.

    Un mes falla si r < r_min o si la pendiente se aleja de 1 más que 'pendiente_tol'
    (amplitudes atenuadas o amplificadas: sensor mal calibrado o mal ubicado).
    """
    q = cfg["analisis_marea"]["qc_referencia"]
    filas = []
    for mes, s in serie.groupby(serie.index.to_period("M")):
        s = s.dropna()
        if len(s) < 48:
            continue
        p = pred.reindex(s.index)
        o = s - s.mean()
        pend = float(np.polyfit(p, o, 1)[0])
        r = float(np.corrcoef(o, p)[0, 1])
        ok = (r >= q["r_min"]) and (abs(pend - 1) <= q["pendiente_tol"])
        filas.append({"estacion": nombre, "mes": str(mes), "horas": len(s), "r": r, "pendiente": pend,
                      "rms_residuo_m": float((o - p).std()), "aprobado": ok})
    tabla = pd.DataFrame(filas)
    mascara = pd.Series(True, index=serie.index)
    if len(tabla) and q["aplicar"]:
        malos = set(tabla.loc[~tabla.aprobado, "mes"])
        mascara = ~serie.index.to_period("M").astype(str).isin(malos)
        mascara = pd.Series(mascara, index=serie.index)
    if len(tabla):
        log.info(f"QC contra referencia de {nombre}: {int(tabla.aprobado.sum())} de {len(tabla)} meses aprobados "
                 f"(r ≥ {q['r_min']}, |pendiente − 1| ≤ {q['pendiente_tol']})")
    return tabla, mascara


def fig_qc_referencia(tabla: pd.DataFrame, cfg, figdir):
    q = cfg["analisis_marea"]["qc_referencia"]
    fig, axs = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
    meses = sorted(tabla.mes.unique())
    x = {m: i for i, m in enumerate(meses)}
    for est, mk in zip(tabla.estacion.unique(), ["o", "s", "^"]):
        t = tabla[tabla.estacion == est]
        for ax, col in zip(axs, ["pendiente", "r"]):
            ax.plot([x[m] for m in t.mes], t[col], mk + "-", label=est)
            mal = t[~t.aprobado]
            ax.plot([x[m] for m in mal.mes], mal[col], "x", color="red", ms=10)
    axs[0].axhspan(1 - q["pendiente_tol"], 1 + q["pendiente_tol"], color="green", alpha=0.1, label="Rango aceptado")
    axs[0].axhline(1, color="k", lw=0.8)
    axs[0].set_ylabel("Pendiente obs/pred")
    axs[0].set_title("Control de calidad mensual del IOC contra la marea predicha con UHSLC (x roja = mes descartado)")
    axs[0].legend(fontsize=8)
    axs[1].axhline(q["r_min"], color="r", ls="--", lw=1, label=f"r mínimo = {q['r_min']}")
    axs[1].set_ylabel("Correlación r")
    axs[1].legend(fontsize=8)
    axs[1].set_xticks(range(len(meses)), meses, rotation=60, fontsize=7)
    guardar(fig, figdir, "marea_qc_ioc_vs_referencia", log)


# ---------------------------------------------------------------------------
def main():
    args = parser_base("Descarga y QC del mareógrafo de Pisco (IOC + UHSLC)").parse_args()
    cfg = cargar_config(args.config)
    ini, fin = periodo(cfg, args)
    figdir = carpeta(cfg, "carpeta_figuras")
    procdir = carpeta(cfg, "carpeta_procesados")
    horas = pd.date_range(ini, fin, freq="h")
    am = cfg["analisis_marea"]

    # --- UHSLC (referencia de calidad) ---------------------------------------
    uh = descargar_uhslc(cfg, args.forzar)
    # Completitud sobre el año calendario completo (castiga los años parciales)
    horas_anio = pd.Series({a: (366 if pd.Timestamp(f"{a}-12-31").dayofyear == 366 else 365) * 24
                            for a in sorted(set(uh.index.year))})
    comp_anual = uh.notna().groupby(uh.index.year).sum() / horas_anio
    anio = elegir_anio(comp_anual, cfg["uhslc"]["anio_referencia"], am["tolerancia_empate"])
    log.info(f"Año de referencia UHSLC: {anio} ({100 * comp_anual[anio]:.2f} % completo). "
             f"Años más completos: {comp_anual.sort_values(ascending=False).head(5).round(4).to_dict()}")
    sanidad.rango_marea(f"UHSLC Pisco {anio}", uh.loc[str(anio)], log)
    uh_df = uh.to_frame()
    uh_df.index.name = "tiempo_utc"
    uh_df.round(4).to_csv(procdir / "marea_uhslc_horaria.csv")
    fig_cobertura_uhslc(comp_anual, anio, figdir)
    pred_ref = prediccion_referencia(uh, anio, horas, cfg["mareografo"]["lat"])

    # --- IOC ------------------------------------------------------------------
    codigos = cfg["ioc"]["estaciones"]
    series, previas, tablas_qc = {}, {}, []
    for cod in codigos:
        sensor = cfg["ioc"]["sensor_preferido"][cod]
        partes = []
        for mes in pd.period_range(ini, fin, freq="M"):
            df = descargar_ioc_mes(cfg, cod, mes, args.forzar)
            log.info(f"IOC {cod} {mes}: {len(df)} registros ({sorted(df.sensor.unique()) if len(df) else 'vacío'})")
            partes.append(df)
        todo = pd.concat(partes, ignore_index=True)
        hor, _ = qc_ioc(todo, sensor, cod)
        hor = hor.reindex(horas)
        previas[cod] = hor
        tabla, mascara = qc_referencia(hor, pred_ref, cod, cfg)
        tablas_qc.append(tabla)
        series[cod] = hor.where(mascara)
    tabla_qc = pd.concat(tablas_qc, ignore_index=True)
    if len(tabla_qc):
        tabla_qc.round(4).to_csv(procdir / "marea_ioc_qc_mensual.csv", index=False)
        log.info("QC mensual contra la referencia:\n" + tabla_qc.round(3).to_string(index=False))
        fig_qc_referencia(tabla_qc, cfg, figdir)

    ioc, desfase = combinar(series[codigos[0]], series[codigos[1]], horas)
    cob = pd.DataFrame({codigos[0]: series[codigos[0]].notna(), codigos[1]: series[codigos[1]].notna(),
                        "combinada": ioc.nivel_m.notna()}).groupby(horas.to_period("M")).mean()
    log.info("Cobertura mensual IOC tras QC (fracción de horas válidas):\n" + cob.round(2).to_string())
    log.info(f"Serie combinada: {100 * ioc.nivel_m.notna().mean():.1f} % de horas válidas "
             f"({(ioc.fuente == 'secundaria').sum()} h tomadas de {codigos[1]})")
    if ioc.nivel_m.notna().sum() > 48:
        sanidad.rango_marea("IOC Pisco", ioc.nivel_m, log)

    v_ini, v_fin, v_comp = elegir_ventana(ioc.nivel_m, cfg)
    log.info(f"Ventana de análisis IOC: {v_ini:%Y-%m-%d} a {v_fin:%Y-%m-%d} ({100 * v_comp:.1f} % completa)")

    # Serie principal para el análisis armónico
    principal = str(am["serie_principal"])
    if principal == "auto":
        principal = "ioc" if v_comp >= am["completitud_minima"] else "uhslc"
        if principal == "uhslc":
            log.warning(f"La ventana IOC tiene solo {100 * v_comp:.1f} % de horas válidas tras el QC "
                        f"(mínimo {100 * am['completitud_minima']:.0f} %): la serie principal será UHSLC {anio}; "
                        f"IOC queda como comparación")
    log.info(f"Serie principal para el análisis armónico: {principal.upper()}")

    ioc.index.name = "tiempo_utc"
    ioc.round(4).to_csv(procdir / "marea_ioc_horaria.csv")
    # Serie de cada estación ANTES del QC contra la referencia (con su QC básico de picos y saltos):
    # 07 la usa para validar el mes candidato aunque ese mes haya fallado por un error de escala
    prev = pd.DataFrame(previas)
    prev.index.name = "tiempo_utc"
    prev.round(4).to_csv(procdir / "marea_ioc_horaria_previa_qc.csv")
    fig_cobertura_ioc(cob, (v_ini, v_fin), figdir)
    fig_serie(ioc, figdir)

    sel = {"principal": principal,
           "ioc": {"inicio": str(v_ini), "fin": str(v_fin), "completitud": round(v_comp, 4),
                   "desfase_secundaria_m": round(desfase, 4)},
           "uhslc": {"anio": anio, "completitud": round(float(comp_anual[anio]), 4)}}
    with open(procdir / "seleccion_marea.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(sel, f, allow_unicode=True, sort_keys=False)
    log.info("Guardados: marea_ioc_horaria.csv, marea_ioc_horaria_previa_qc.csv, marea_uhslc_horaria.csv, marea_ioc_qc_mensual.csv, seleccion_marea.yaml")


if __name__ == "__main__":
    main()
