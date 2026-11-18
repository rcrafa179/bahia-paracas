#!/usr/bin/env python
"""10 - Exporta el forzamiento del mes candidato al formato de entrada de AEM3D.

Escribe dos archivos de condiciones de contorno temporales (boundary_condition_file) en el
formato "Version 1" de AEM3D, verificado en:
  * el manual: ~/AEM3D/docs/AEM3D_User.pdf, sección 3.7.2 y tabla 3.2 (palabras clave y unidades);
  * los ejemplos instalados: ~/AEM3D/Model/Lagoon/run/infiles (IdealisedMetDataJan2005.dat,
    tideHeight.dat) y la guía del curso en ~/AEM3D/Model/test1/run/crearMemteologia/indicaciones.md.

Estructura de cada archivo:
    ! comentarios
    <n> data sets                     número de columnas sin contar TIME
    0 seconds between data            línea libre
    0 <ref> <ref> ...                 conjunto de celdas de cada columna (0 = toda la superficie)
    TIME <PALABRA> <PALABRA> ...      palabras clave de la tabla 3.2
    yyyyddd.dddddd  valores ...       TIME en formato CWR (año + día juliano + fracción de día)

Unidades (tabla 3.2): WIND_SPEED m/s; WIND_DIR grados desde donde viene, horario desde el norte;
AIR_TEMP °C; REL_HUM fracción 0–1; SOLAR_RAD W/m²; ATM_PRESS Pa; RAIN m/día; CLOUDS fracción 0–1;
LW_RAD_IN W/m²; HEIGHT m.

Salidas (aem3d.carpeta):
  met_paracas_<mes>.dat     meteorología horaria (conjunto 0: toda la superficie)
  marea_paracas_<mes>.dat   nivel cada 15 min en cada frontera abierta (conjuntos de aem3d.bc_frontera)
  LEEME_aem3d.txt           líneas para run_aem3d.dat y lo que falta definir en el preprocesador

Uso:
    python scripts/10_exportar_aem3d.py      (después de 04 y 07)
"""
from __future__ import annotations

import unicodedata

import pandas as pd
import yaml

from utils.config import RAIZ, cargar_config, carpeta, parser_base, ruta
from utils.log import obtener_logger

log = obtener_logger("10_exportar_aem3d")


# ---------------------------------------------------------------------------
# Tiempo en formato CWR
# ---------------------------------------------------------------------------
def fecha_cwr(t: pd.DatetimeIndex) -> list[str]:
    """yyyyddd.dddddd: año, día juliano (1 = 1 de enero) y fracción del día (6 decimales ≈ 0,1 s)."""
    frac = (t - t.normalize()).total_seconds() / 86400.0
    return [f"{a}{d:03d}{f'{x:.6f}'[1:]}" for a, d, x in zip(t.year, t.dayofyear, frac)]


def desde_cwr(texto: str) -> pd.Timestamp:
    """Inverso de fecha_cwr (para verificar los archivos escritos)."""
    entero, frac = texto.split(".")
    base = pd.Timestamp(f"{entero[:4]}-01-01") + pd.Timedelta(days=int(entero[4:]) - 1)
    return base + pd.Timedelta(days=float("0." + frac))


def relativa(p):
    """Ruta relativa a la raíz del repositorio si está dentro; si no, la absoluta."""
    return p.relative_to(RAIZ) if p.is_relative_to(RAIZ) else p


def solo_ascii(texto: str) -> str:
    """Quita tildes y símbolos: los archivos de AEM3D (Fortran) se escriben en ASCII puro."""
    texto = texto.replace("°", " grados ").replace("²", "2").replace("–", "-").replace("≈", "~")
    return unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode("ascii")


def a_hora_modelo(indice_utc: pd.DatetimeIndex, cfg) -> pd.DatetimeIndex:
    if cfg["aem3d"]["hora"] == "local":
        return indice_utc + pd.Timedelta(hours=cfg["proyecto"]["zona_horaria_local"])
    if cfg["aem3d"]["hora"] != "utc":
        raise SystemExit("aem3d.hora debe ser 'local' o 'utc'")
    return indice_utc


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------
def escribir_bc(destino, tiempos: pd.DatetimeIndex, datos: pd.DataFrame, refs: list[int], comentarios: list[str],
                decimales: dict[str, int]):
    """Archivo de condiciones de contorno temporales, formato Version 1 de AEM3D."""
    if datos.isna().any().any():
        malas = datos.columns[datos.isna().any()].tolist()
        raise SystemExit(f"{destino.name}: quedan datos faltantes en {malas}; AEM3D exige columnas completas")
    ancho = 80
    lineas = ["! " + "-" * (ancho - 4) + " !"]
    lineas += [f"! {solo_ascii(c):<{ancho - 4}} !" for c in comentarios]
    lineas += ["! " + "-" * (ancho - 4) + " !"]
    lineas.append(f"{datos.shape[1]} data sets")
    lineas.append("0 seconds between data")
    lineas.append("  ".join(str(r) for r in [0] + refs))
    claves = [c.split("@")[0] for c in datos.columns]  # 'HEIGHT@norte' -> 'HEIGHT'
    lineas.append("TIME            " + "  ".join(f"{k:>12}" for k in claves))
    tc = fecha_cwr(tiempos)
    cols = [datos[c].to_numpy() for c in datos.columns]
    nd = [decimales.get(k, 3) for k in claves]
    for i, t in enumerate(tc):
        lineas.append(f"{t:<16}" + "  ".join(f"{col[i]:>12.{d}f}" for col, d in zip(cols, nd)))
    destino.write_text("\n".join(lineas) + "\n", encoding="ascii")
    log.info(f"Escrito {relativa(destino)}: "
             f"{len(tc)} filas x {datos.shape[1]} columnas ({tc[0]} a {tc[-1]})")


def verificar_bc(archivo, n_esperado: int) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Relee el archivo: encabezado de 4 líneas, filas completas y tiempo creciente."""
    lineas = [l for l in archivo.read_text(encoding="ascii").splitlines() if not l.startswith("!")]
    n = int(lineas[0].split()[0])
    refs, claves = lineas[2].split(), lineas[3].split()
    assert n == n_esperado and len(refs) == n + 1 and len(claves) == n + 1 and claves[0] == "TIME", archivo
    filas = [l.split() for l in lineas[4:]]
    assert all(len(f) == n + 1 for f in filas), f"{archivo.name}: filas con distinto número de columnas"
    t = [desde_cwr(f[0]) for f in filas]
    assert all(b > a for a, b in zip(t, t[1:])), f"{archivo.name}: el tiempo no es creciente"
    return t[0], t[-1]


# ---------------------------------------------------------------------------
def main():
    args = parser_base("Exporta el forzamiento del mes candidato al formato de AEM3D").parse_args()
    cfg = cargar_config(args.config)
    ae = cfg["aem3d"]
    procdir = carpeta(cfg, "carpeta_procesados")
    destino = ruta(ae["carpeta"])
    destino.mkdir(parents=True, exist_ok=True)
    with open(procdir / "mes_candidato.yaml", encoding="utf-8") as f:
        cand = yaml.safe_load(f)
    mes = cand["mes"]
    etiqueta_hora = (f"hora local (UTC{cfg['proyecto']['zona_horaria_local']:+d})" if ae["hora"] == "local"
                     else "UTC")

    # Mes completo en la hora del modelo: es lo que debe cubrir el forzamiento
    m_ini = pd.Timestamp(f"{mes}-01")
    m_fin = m_ini + pd.offsets.MonthEnd(1) + pd.Timedelta(hours=23)

    # --- Meteorología ------------------------------------------------------------------
    met = pd.read_csv(procdir / "forzamiento_meteo_mes_candidato.csv", index_col=0, parse_dates=True)
    tm = a_hora_modelo(met.index, cfg)
    huecos = met[["vel_ms", "dir_grados", "t2m_C", "hr_pct", "ssrd_Wm2", "msl_hPa", "tp_mm", "tcc_frac",
                  "strd_Wm2"]].isna().sum()
    if huecos.any():
        log.warning(f"Datos faltantes en el forzamiento (se interpolan si son ≤ 3 h): {huecos[huecos > 0].to_dict()}")
        met = met.interpolate(limit=3, limit_area="inside")
    d = pd.DataFrame({
        "WIND_SPEED": met.vel_ms.clip(lower=0),
        "WIND_DIR": met.dir_grados % 360,
        "AIR_TEMP": met.t2m_C,
        "REL_HUM": (met.hr_pct / 100).clip(0, 1),
        "SOLAR_RAD": met.ssrd_Wm2.clip(lower=0),
        "ATM_PRESS": met.msl_hPa * 100,               # hPa -> Pa
        "RAIN": (met.tp_mm.clip(lower=0) * 24 / 1000),  # mm en la hora -> m/día
    })
    if ae["onda_larga"] == "CLOUDS":
        d["CLOUDS"] = met.tcc_frac.clip(0, 1)
    elif ae["onda_larga"] == "LW_RAD_IN":
        d["LW_RAD_IN"] = met.strd_Wm2
    else:
        raise SystemExit("aem3d.onda_larga debe ser 'CLOUDS' o 'LW_RAD_IN'")
    d.index = tm
    fuente_v = cfg["forzamiento_viento"]["metodo"]
    archivo_met = destino / f"met_paracas_{mes}.dat"
    escribir_bc(archivo_met, tm, d, [0] * d.shape[1], [
        f"AEM3D: forzamiento meteorológico, bahía de Paracas, mes candidato {mes}",
        f"Generado por scripts/10_exportar_aem3d.py el {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        f"TIME en formato CWR (yyyyddd.ddd), {etiqueta_hora}; datos horarios",
        f"Viento: método '{fuente_v}' (ver forzamiento_viento en config.yaml);",
        "  SPSO = METAR del aeropuerto de Pisco; huecos con ERA5 corregido por cuantiles",
        "Resto: ERA5 en el punto de malla más cercano a la boca de la bahía",
        "REL_HUM fracción; ATM_PRESS Pa (presión a nivel del mar); RAIN m/día;",
        f"{'CLOUDS fracción (tcc)' if ae['onda_larga'] == 'CLOUDS' else 'LW_RAD_IN W/m2 (strd)'};"
        " SOLAR_RAD = onda corta incidente (ssrd)",
    ], {"WIND_SPEED": 3, "WIND_DIR": 1, "AIR_TEMP": 2, "REL_HUM": 3, "SOLAR_RAD": 1, "ATM_PRESS": 1,
        "RAIN": 6, "CLOUDS": 3, "LW_RAD_IN": 1})

    # --- Marea en las fronteras ---------------------------------------------------------------
    mar = pd.read_csv(procdir / "marea_frontera_mes_candidato.csv", index_col=0, parse_dates=True)
    refs, cols, notas = [], {}, []
    for frontera, ref in ae["bc_frontera"].items():
        pts = [c for c in mar.columns if c.startswith(f"{frontera}_")]
        if not pts:
            raise SystemExit(f"No hay puntos de la frontera '{frontera}' en marea_frontera_mes_candidato.csv")
        if int(ref) == 0:
            raise SystemExit("El conjunto 0 está reservado para la superficie: usa otro número en aem3d.bc_frontera")
        # Una serie por frontera: AEM3D aplica una columna HEIGHT a todo el conjunto de celdas
        cols[f"HEIGHT@{frontera}"] = mar[pts].mean(axis=1)
        dispersion = float((mar[pts].max(axis=1) - mar[pts].min(axis=1)).max())
        refs.append(int(ref))
        notas.append(f"  {frontera}: conjunto {ref}, promedio de {len(pts)} puntos "
                     f"(diferencia máxima entre puntos {100 * dispersion:.1f} cm)")
        log.info(f"Frontera {frontera} -> conjunto {ref}: {len(pts)} puntos, diferencia máxima entre ellos "
                 f"{100 * dispersion:.1f} cm")
    h = pd.DataFrame(cols)
    tt = a_hora_modelo(h.index, cfg)
    h.index = tt
    archivo_mar = destino / f"marea_paracas_{mes}.dat"
    escribir_bc(archivo_mar, tt, h, refs, [
        f"AEM3D: nivel del mar en las fronteras abiertas, bahía de Paracas, {mes}",
        f"Generado por scripts/10_exportar_aem3d.py el {pd.Timestamp.now():%Y-%m-%d %H:%M}",
        f"TIME en formato CWR (yyyyddd.ddd), {etiqueta_hora}; paso de {cfg['modelo_marea']['paso_minutos']} min",
        "HEIGHT en m respecto del nivel medio del mar (batimetría GEBCO, mismo datum): marea astronómica",
        f"  de FES2022b (pyTMD, {len(cfg['modelo_marea']['constituyentes'])} constituyentes + menores inferidas)",
        "  sin variaciones del nivel medio (no incluye la marea meteorológica)",
    ] + notas, {"HEIGHT": 4})

    # --- Verificación -----------------------------------------------------------------------
    cob = {}
    for arch, n in [(archivo_met, d.shape[1]), (archivo_mar, h.shape[1])]:
        a, b = verificar_bc(arch, n)
        cob[arch.name] = (a, b)
        if a > m_ini or b < m_fin:
            log.warning(f"{arch.name} cubre {a} a {b} y no todo {mes} en hora del modelo "
                        f"({m_ini} a {m_fin}): aumentar aem3d.margen_horas y correr 04 y 07")
        else:
            log.info(f"[sanidad] {arch.name} cubre todo {mes} ({etiqueta_hora}) -> OK")
    log.info(f"[sanidad] rangos: viento {d.WIND_SPEED.min():.1f}–{d.WIND_SPEED.max():.1f} m/s, "
             f"HR {d.REL_HUM.min():.2f}–{d.REL_HUM.max():.2f}, presión {d.ATM_PRESS.min():.0f}–"
             f"{d.ATM_PRESS.max():.0f} Pa, marea {h.min().min():+.2f} a {h.max().max():+.2f} m")

    # --- Instrucciones -----------------------------------------------------------------------
    t_ini = fecha_cwr(pd.DatetimeIndex([m_ini]))[0]
    # Nivel de la marea en las fronteras al inicio: como nivel inicial evita el transitorio de arranque
    h_ini = float(h.mean(axis=1).reindex(h.index.union([m_ini])).interpolate("time").loc[m_ini])
    leeme = f"""Forzamiento de AEM3D para la bahía de Paracas, mes candidato {mes}
Generado por scripts/10_exportar_aem3d.py. Tiempos en {etiqueta_hora}.

Archivos (copiar a la carpeta run/infiles del caso):
  {archivo_met.name}   meteorología horaria, conjunto 0 (toda la superficie)
  {archivo_mar.name}   nivel en las fronteras abiertas cada {cfg['modelo_marea']['paso_minutos']} min
{chr(10).join(notas)}

Líneas para run_aem3d.dat (formato de Model/Lagoon/run/run_aem3d.dat):
  {archivo_met.name:<40} boundary_condition_file
  {archivo_mar.name:<40} boundary_condition_file
  {t_ini:<40} start_date_cwr
  ! nivel inicial = marea en la frontera al inicio (evita un transitorio de arranque)
  {h_ini:<40.3f} DEFAULT_HEIGHT
  iter_max = días x 86400 / del_t (p. ej. 14 días con del_t = 30 s -> 40320 iteraciones)
  (la versión demo de AEM3D permite como máximo 14 días; la latitud la toma de la batimetría)

Revisar antes de correr:
  * Los conjuntos de celdas {', '.join(f'{k} = {v}' for k, v in ae['bc_frontera'].items())} deben existir
    en el bc.dat del preprocesador (pre_aem3d), sobre las filas/columnas de las fronteras abiertas.
  * start_date_cwr está en la misma hora que los archivos ({etiqueta_hora}).
  * iheat_input = 1 usa SOLAR_RAD y {ae['onda_larga']}; irain = 1 si se quiere la lluvia (en Paracas
    es casi nula).
  * Cobertura: {', '.join(f'{k}: {a:%Y-%m-%d %H:%M} a {b:%Y-%m-%d %H:%M}' for k, (a, b) in cob.items())}.
"""
    (destino / "LEEME_aem3d.txt").write_text(leeme, encoding="utf-8")
    log.info(f"Instrucciones en {relativa(destino / 'LEEME_aem3d.txt')}")


if __name__ == "__main__":
    main()
