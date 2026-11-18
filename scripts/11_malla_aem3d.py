#!/usr/bin/env python
"""11 - Malla de AEM3D para la bahía de Paracas: batimetría, fronteras abiertas y caso de prueba.

Arma un caso completo con la estructura de ~/AEM3D/Model/Lagoon:

  <malla_aem3d.carpeta>/
    pre/run_pre.dat                    control del preprocesador
    pre/bathymetry/paracas_bathymetry.dat
    pre/bathymetry/paracas_bc.dat      conjuntos open_cell de cada frontera abierta
    run/run_aem3d.dat                  corrida de PRUEBA (barotrópica: marea y viento)
    run/infiles/datablock.xml          salidas (nivel y velocidades en superficie)
    run/infiles/*.dat                  forzamiento copiado de data/processed/aem3d (script 10)

Formatos según el manual (~/AEM3D/docs/AEM3D_User.pdf, cap. 2) y el ejemplo Lagoon:
  * bathymetry.dat: palabras clave, vectores dx/dy/dz y la matriz BATHYMETRY DATA con x_rows filas
    (i crece hacia abajo = hacia el SUR) y y_columns columnas (j crece hacia el ESTE); norte arriba,
    north_x = -1, north_y = 0. Cotas en m respecto del nivel medio, positivas hacia arriba (GEBCO).
    Alrededor del agua va un anillo de celdas de tierra (land_value) o de frontera abierta (open_value).
  * bc.dat: "<n> open_cell <nombre>" seguido de líneas "i j k" con rangos (a:b) o ':' (todas).

Decisiones (elegidas por el usuario el 2026-11-18): celda de 250 m, borde sur abierto y capas
estiradas hasta el fondo real.

Uso:
    python scripts/11_malla_aem3d.py              # arma el caso
    python scripts/11_malla_aem3d.py --correr     # además corre pre_aem3d y la prueba corta
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy import ndimage

from utils.config import RAIZ, cargar_config, carpeta, parser_base, ruta
from utils.graficos import guardar, plt
from utils.log import obtener_logger

log = obtener_logger("11_malla_aem3d")
TIERRA, ABIERTA = 999.0, 888.0
M_POR_GRADO = 111320.0
LADOS = {"norte": "N", "sur": "S", "oeste": "O", "este": "E"}


def relativa(p):
    return p.relative_to(RAIZ) if p.is_relative_to(RAIZ) else p


# ---------------------------------------------------------------------------
# Malla horizontal y batimetría
# ---------------------------------------------------------------------------
def malla(cfg):
    """Centros de celda (norte arriba) e interpolación bilineal de GEBCO."""
    d, ma = cfg["dominio"], cfg["malla_aem3d"]
    lat0 = (d["lat_min"] + d["lat_max"]) / 2
    dlat = ma["dx_m"] / M_POR_GRADO
    dlon = ma["dx_m"] / (M_POR_GRADO * np.cos(np.deg2rad(lat0)))
    lats = np.arange(d["lat_max"] - dlat / 2, d["lat_min"], -dlat)  # filas: de norte a sur
    lons = np.arange(d["lon_min"] + dlon / 2, d["lon_max"], dlon)   # columnas: de oeste a este
    g = xr.open_dataset(ruta(cfg["batimetria"]["gebco_nc"]))
    var = next(v for v in ("elevation", "z", "Band1") if v in g.data_vars)
    z = g[var].interp(lat=lats, lon=lons).values.astype(float)
    if np.isnan(z).any():
        raise SystemExit("GEBCO no cubre todo el dominio: descarga un recorte más grande")
    return lats, lons, dlat, dlon, z


def mascara_agua(z, abiertos: set[str], prof_min: float):
    """Agua = cota < 0, conectada a alguna frontera abierta (se quitan lagunas aisladas)."""
    agua = z < 0
    etiquetas, n = ndimage.label(agua)  # conectividad de 4 vecinos, como las caras de AEM3D
    bordes = set()
    if "N" in abiertos:
        bordes |= set(etiquetas[0][agua[0]])
    if "S" in abiertos:
        bordes |= set(etiquetas[-1][agua[-1]])
    if "O" in abiertos:
        bordes |= set(etiquetas[:, 0][agua[:, 0]])
    if "E" in abiertos:
        bordes |= set(etiquetas[:, -1][agua[:, -1]])
    conectada = np.isin(etiquetas, list(bordes - {0}))
    quitadas = int(agua.sum() - conectada.sum())
    somera = conectada & (z > -prof_min)
    zz = np.where(conectada, np.minimum(z, -prof_min), TIERRA)
    return conectada, zz, quitadas, int(somera.sum())


def anillo(zz, agua, abiertos: set[str]):
    """Agrega el anillo exterior: frontera abierta frente a agua en los lados abiertos; tierra en el resto."""
    nf, nc = zz.shape
    b = np.full((nf + 2, nc + 2), TIERRA)
    b[1:-1, 1:-1] = zz
    if "N" in abiertos:
        b[0, 1:-1] = np.where(agua[0], ABIERTA, TIERRA)
    if "S" in abiertos:
        b[-1, 1:-1] = np.where(agua[-1], ABIERTA, TIERRA)
    if "O" in abiertos:
        b[1:-1, 0] = np.where(agua[:, 0], ABIERTA, TIERRA)
    if "E" in abiertos:
        b[1:-1, -1] = np.where(agua[:, -1], ABIERTA, TIERRA)
    return b


def segmentos(v: np.ndarray) -> list[tuple[int, int]]:
    """Tramos contiguos de True como pares (inicio, fin) en índices de AEM3D (desde 1)."""
    d = np.diff(np.r_[0, v.astype(int), 0])
    return [(a + 1, b) for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0])]


def conjuntos_bc(b, bc_frontera: dict[str, int]) -> list[tuple[int, str, list[str]]]:
    """Líneas de bc.dat para cada frontera abierta: celdas open_value del anillo, todas las capas."""
    nf, nc = b.shape
    salida = []
    for nombre, ref in bc_frontera.items():
        lado = LADOS[nombre]
        if lado == "N":
            lin = [f"1 {a}:{z} :" for a, z in segmentos(b[0] == ABIERTA)]
        elif lado == "S":
            lin = [f"{nf} {a}:{z} :" for a, z in segmentos(b[-1] == ABIERTA)]
        elif lado == "O":
            lin = [f"{a}:{z} 1 :" for a, z in segmentos(b[:, 0] == ABIERTA)]
        else:
            lin = [f"{a}:{z} {nc} :" for a, z in segmentos(b[:, -1] == ABIERTA)]
        if not lin:
            raise SystemExit(f"La frontera '{nombre}' no tiene celdas de agua en el borde")
        salida.append((int(ref), nombre, lin))
    return salida


# ---------------------------------------------------------------------------
# Capas verticales
# ---------------------------------------------------------------------------
def capas(prof_max: float, c: dict) -> list[float]:
    """Espesores de arriba hacia abajo: constantes hasta 'hasta_m' y luego crecientes."""
    total = prof_max + c["margen_superior_m"]
    dz, z = [], -c["margen_superior_m"]  # z = profundidad del fondo de la capa respecto del nivel medio
    while sum(dz) < total:
        e = c["dz_superficie_m"] if z < c["hasta_m"] else min(dz[-1] * c["factor"], c["dz_max_m"])
        dz.append(round(e, 3))
        z += e
    return dz


# ---------------------------------------------------------------------------
# Escritura de archivos
# ---------------------------------------------------------------------------
def linea_comentario(texto=""):
    # Cada línea debe tener al menos dos ítems (manual, 2.2.1): "!" + texto + "!"
    return f"! {texto:<66} !"


def escribir_batimetria(destino, b, dx, dz, lat_esq, lon_esq, cfg):
    nf, nc = b.shape
    L = [linea_comentario("-" * 66), linea_comentario(),
         f" FILE           {destino.name}",
         linea_comentario("Batimetria de la bahia de Paracas para AEM3D (GEBCO_2026)"),
         linea_comentario("Generado por scripts/11_malla_aem3d.py; no editar a mano"),
         linea_comentario("N.B. all lines must have at least two items"),
         linea_comentario("-" * 66),
         " 'Bahia de Paracas'             TITLE",
         " 'scripts/11_malla_aem3d.py'    ANALYST",
         " 'UNMSM'                        ORGANIZATION",
         f" 'GEBCO 15s a {dx:g} m'          COMMENT",
         linea_comentario(),
         " yes      overwrite files",
         linea_comentario(),
         f"{nf:<15} x_rows",
         f"{nc:<15} y_columns",
         f"{len(dz):<15} z_layers",
         f"{0:<15} n_max",
         linea_comentario("value used to represent land and open boundaries"),
         f"{TIERRA:<15.2f} land_value",
         f"{ABIERTA:<15.2f} open_value",
         linea_comentario("vector pointing north: north up (i increases southward)"),
         f"{-1.0:<15} north_x",
         f"{0.0:<15} north_y",
         linea_comentario("geographic position of the top-left corner"),
         f"{lat_esq:<15.5f} latitude (degrees +north -south)",
         f"{lon_esq:<15.5f} longitude (degrees +east -west)",
         f"{0:<15} altitude (m above sea level)",
         linea_comentario("grid spacing"),
         f"{dx:<15.1f} x_grid_size",
         f"{dx:<15.1f} y_grid_size",
         linea_comentario("dx vector starting from top going down")]
    L += [f"{dx:<15.1f} dx"] * nf
    L += [linea_comentario("dy vector starting from left going right")]
    L += [f"{dx:<15.1f} dy"] * nc
    L += [linea_comentario("dz vector starting from top going down")]
    L += [f"{e:<15.3f} dz" for e in dz]
    L += [linea_comentario("x increases with rows (southward), y with columns (eastward)"),
          "BATHYMETRY DATA"]
    L += [" ".join(f"{v:8.2f}" for v in fila) for fila in b]
    destino.write_text("\n".join(L) + "\n", encoding="ascii")


def escribir_bc(destino, conjuntos):
    L = ["! - - -", "! - bc.dat: fronteras abiertas de la bahia de Paracas",
         "! - generado por scripts/11_malla_aem3d.py", "! - comment lines need three items"]
    for ref, nombre, lin in conjuntos:
        L.append(f"{ref} open_cell {nombre}")
        L += [f"\t{x}" for x in lin]
    destino.write_text("\n".join(L) + "\n", encoding="ascii")


def escribir_run_pre(destino, bati, bc):
    sep = "! " + "-" * 75 + " !"
    L = [sep, " FILE                                   run_pre.dat", sep,
         "! input files                                                                 !",
         f" 'bathymetry/{bati}'       BATHYMETRY_FILE",
         f" 'bathymetry/{bc}'               BOUNDARY_CONDITION_FILE",
         "! output files                                                                !",
         " 'bathymetry/paracas_bathymetry_out.txt'    DATAOUT_FILE",
         " '../run/infiles/usedata.unf'            AEM3D_USEDATA_FILE",
         " '../run/infiles/sparsedata.unf'         AEM3D_SPARSEDATA_FILE",
         "! permission to overwrite existing files                                      !",
         " yes                                     overwrite", sep]
    destino.write_text("\n".join(L) + "\n", encoding="ascii")


def escribir_run_aem3d(destino, cfg, met, mar, inicio, del_t, iteraciones, h_ini):
    """run_aem3d.dat de PRUEBA, con las palabras clave del ejemplo Lagoon (misma versión de AEM3D).

    Barotrópica: sin temperatura, salinidad ni calor (no hay perfiles iniciales medidos de la bahía).
    Sirve para verificar la malla, la marea en las fronteras y el viento."""
    lat = cfg["mareografo"]["lat"]
    v = [("'Bahia de Paracas - prueba'", "TITLE"), ("'scripts/11_malla_aem3d.py'", "ANALYST"),
         ("'UNMSM'", "ORGANIZATION"), ("'Prueba barotropica: marea FES2022 y viento SPSO'", "COMMENT"),
         ("'NONE'", "CASE_KEYWORD"),
         ("!", "Time controls"),
         (inicio, "start_date_cwr"), (f"{del_t:.1f}", "del_t"), (iteraciones, "iter_max"),
         ("!", "Simulation module controls"),
         (0, "iheat_input"), (0, "iatmstability"), (0, "iice"), (0, "irain"), (1, "iflow"),
         (0, "iunderflow"), (0, "ibubbler"), (0, "itemperature"), (0, "isalinity"), (0, "idensity"),
         (0, "ijet"), (0, "idrafttube"), (0, "icaedym"), (0, "inonhydrostatic"), (1, "icoriolis"),
         (0, "ilaketide"), (1, "allow_neumann"), (0, "iretention"), (0, "ntracer"), (0, "ndrifters"),
         (3, "idatablock"),
         ("!", "Model settings and controls"),
         (0.08, "mean_albedo"), (0.0, "time_zone"), (0.0013, "wind_cd"), (0.005, "drag_btm_cd"),
         (1.0, "model_grav_damp_x"), (1.0, "model_grav_damp_y"), (2, "ic_dist_weighting"),
         (lat, "latitude"),
         ("!", "Default (uniformly distributed) values"),
         (f"{h_ini:.3f}", "DEFAULT_HEIGHT"), (0.0, "DEFAULT_WIND_SPEED"), (0.0, "DEFAULT_WIND_DIR"),
         (0.4, "DEFAULT_PAR_EXTINCTION"), (1.0, "DEFAULT_NIR_EXTINCTION"), (1.0, "DEFAULT_UVA_EXTINCTION"),
         (2.5, "DEFAULT_UVB_EXTINCTION"), (2.5, "DEFAULT_RFLCT_EXTINCTION"), (18.0, "DEFAULT_WTR_TEMP"),
         (35.0, "DEFAULT_SALINITY"), (9, "DEFAULT_BC"), (0.0, "DEFAULT_TRACER_1"), (0.0, "DEFAULT_FILTER"),
         (0.0, "DEFAULT_DIFFUSIVITY"),
         ("!", "Scalar filtering controls"), (0, "IFILTER"),
         ("!", "Initalization and update options"),
         (0, "irestart"), (0, "user_init_u_vel"), (0, "user_init_v_vel"), (0, "user_init_w_vel"),
         (0, "user_init_temperature"), (0, "user_init_salinity"), (0, "user_init_tracer"),
         (0, "user_init_height"), (0, "user_init_extinction"),
         ("!", "Meterological sensor heights"),
         (10.0, "WIND_SPEED_HEIGHT"), (10.0, "SCALAR_HEIGHT"), (0.9, "SEDIMENT_REFLECTIVITY"),
         (0.0013, "SURF_HEAT_TRANSF_COEFF"),
         # Sin densidad AEM3D exige iclosure = 0 ("If idensity == 0 iclosure must be 0");
         # en una corrida baroclínica usar 6 (capa de mezcla por viento, recomendado por el manual)
         ("!", "Turbulence modelling controls"), (0, "iclosure"),
         ("!", "Iterative (conjugate gradient method) solution controls"),
         ("1.0e-16", "CGM_TOL"), (30, "CGM_MIN"), (1000, "CGM_MAX"),
         ("!", "Input file names"),
         ("'infiles'", "infile_dir"), ("sparsedata.unf", "3D_data_file"), ("usedata.unf", "preprocessor_file"),
         ("datablock.xml", "datablock_file"), (met, "boundary_condition_file"), (mar, "boundary_condition_file"),
         ("!", "Output controls"),
         ("'ncfiles'", "outfile_nc_dir"), ("'unffiles'", "outfile_unf_dir"), ("'txtfiles'", "outfile_txt_dir"),
         (0, "start_output_monitor"), (0, "start_output_save"), (120, "iter_out_monitor"),
         (0, "iter_out_txtfiles"), (iteraciones, "iter_out_save"), (iteraciones, "iter_out_restart"),
         ("save", "restart_save_file"), ("restart_final", "restart_out_file"),
         ("!", "Debugging controls"),
         (0, "iquiet"), (1, "debug_check"), (0, "debug_print"), (0, "debug_print_start"), (0, "debug_point"),
         (0, "debug_timestep_file"), (10, "debug_timestep_reset"), (1, "debug_baroclinic_x"),
         (1, "debug_baroclinic_y"), (0, "ihardlimit")]
    sep = "! " + "-" * 63 + " !"
    L = [sep, "!  AEM3D Configuration file - bahia de Paracas (PRUEBA)            !",
         "!  Generado por scripts/11_malla_aem3d.py a partir de Model/Lagoon !", sep]
    for a, b in v:
        L.append(f"! {b:<63} !" if a == "!" else f"  {str(a):<50} {b}")
    L += [sep, "!  End                                                            !", sep]
    destino.write_text("\n".join(L) + "\n", encoding="ascii")


DATABLOCK = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<AEM3DOutput>
  <GlobalTime>
    <interval>{intervalo}</interval>
  </GlobalTime>
  <globalVariable>U_VELOCITY</globalVariable>
  <globalVariable>V_VELOCITY</globalVariable>
  <Sheet>
    <filename>sheet_top.nc</filename>
    <type>SURFACE</type>
    <value>0.0</value>
  </Sheet>
  <Sheet>
    <filename>sheet_avg.nc</filename>
    <type>AVERAGE</type>
    <value>0.0</value>
  </Sheet>
</AEM3DOutput>
"""


# ---------------------------------------------------------------------------
def fig_malla(b, lats, lons, dz, conjuntos, figdir):
    agua = b < ABIERTA
    fig, axs = plt.subplots(1, 2, figsize=(14, 6.5), gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axs[0]
    prof = np.where(agua, -b, np.nan)[1:-1, 1:-1]
    ext = [lons[0], lons[-1], lats[-1], lats[0]]
    im = ax.imshow(prof, extent=ext, cmap="viridis_r", origin="upper", aspect="equal")
    for i, (nombre, c) in enumerate([("norte", "tab:red"), ("oeste", "tab:orange"), ("sur", "magenta"),
                                     ("este", "tab:pink")]):
        sel = {"norte": (0, slice(1, -1)), "sur": (-1, slice(1, -1)), "oeste": (slice(1, -1), 0),
               "este": (slice(1, -1), -1)}[nombre]
        m = b[sel] == ABIERTA
        if not m.any():
            continue
        if nombre in ("norte", "sur"):
            ax.plot(lons[m], np.full(m.sum(), lats[0] if nombre == "norte" else lats[-1]), "s", color=c, ms=3,
                    label=f"Frontera {nombre}")
        else:
            ax.plot(np.full(m.sum(), lons[0] if nombre == "oeste" else lons[-1]), lats[m], "s", color=c, ms=3,
                    label=f"Frontera {nombre}")
    fig.colorbar(im, ax=ax, label="Profundidad (m)")
    ax.set_xlabel("Longitud (°)")
    ax.set_ylabel("Latitud (°)")
    ax.set_title(f"Malla de AEM3D: {b.shape[0]}×{b.shape[1]} celdas (con anillo exterior)")
    ax.legend(fontsize=8, loc="lower right")
    ax = axs[1]
    fondo = np.cumsum(dz)
    ax.barh(fondo - np.array(dz) / 2, dz, height=np.array(dz) * 0.9, color="tab:blue")
    ax.invert_yaxis()
    ax.set_xlabel("Espesor de la capa (m)")
    ax.set_ylabel("Profundidad del centro de la capa desde el tope de la malla (m)")
    ax.set_title(f"{len(dz)} capas verticales")
    guardar(fig, figdir, "aem3d_malla", log)


def main():
    p = parser_base("Malla y caso de prueba de AEM3D para la bahía de Paracas")
    p.add_argument("--correr", action="store_true", help="correr pre_aem3d y la prueba corta de AEM3D")
    args = p.parse_args()
    cfg = cargar_config(args.config)
    ma, ae = cfg["malla_aem3d"], cfg["aem3d"]
    figdir = carpeta(cfg, "carpeta_figuras")
    caso = ruta(ma["carpeta"])
    for sub in ("pre/bathymetry", "run/infiles", "run/ncfiles", "run/unffiles", "run/txtfiles"):
        (caso / sub).mkdir(parents=True, exist_ok=True)

    abiertos = {LADOS[k] for k in ae["bc_frontera"]}
    lats, lons, dlat, dlon, z = malla(cfg)
    agua, zz, quitadas, somera = mascara_agua(z, abiertos, ma["profundidad_minima_m"])
    log.info(f"Malla de {ma['dx_m']} m: {len(lats)} filas x {len(lons)} columnas; {int(agua.sum())} columnas de agua; "
             f"{quitadas} celdas de agua aisladas eliminadas; {somera} celdas profundizadas a "
             f"{ma['profundidad_minima_m']} m")
    # Bordes con agua que no son frontera abierta quedan como pared: avisar
    for lado, fila in (("N", agua[0]), ("S", agua[-1]), ("O", agua[:, 0]), ("E", agua[:, -1])):
        if fila.any() and lado not in abiertos:
            log.warning(f"El borde {lado} tiene {int(fila.sum())} celdas de agua y no es frontera abierta: "
                        "queda como pared")
    b = anillo(zz, agua, abiertos)
    prof_max = float(-zz[agua].min())
    dz = capas(prof_max, ma["capas"])
    log.info(f"Profundidad máxima {prof_max:.1f} m; {len(dz)} capas: {dz[0]:g} m en superficie hasta "
             f"{ma['capas']['hasta_m']:g} m, luego hasta {max(dz):g} m; total {sum(dz):.1f} m")
    conjuntos = conjuntos_bc(b, ae["bc_frontera"])
    borde = {"norte": b[0], "sur": b[-1], "oeste": b[:, 0], "este": b[:, -1]}
    n_abiertas = {}
    for ref, nombre, lin in conjuntos:
        n = n_abiertas[nombre] = int((borde[nombre] == ABIERTA).sum())
        log.info(f"Frontera {nombre}: conjunto {ref}, {n} celdas abiertas ({'; '.join(lin)})")

    # Esquina superior izquierda del arreglo con anillo (centro de la celda (1,1))
    lat_esq, lon_esq = lats[0] + dlat, lons[0] - dlon
    pre = caso / "pre"
    escribir_batimetria(pre / "bathymetry" / "paracas_bathymetry.dat", b, float(ma["dx_m"]), dz, lat_esq, lon_esq, cfg)
    escribir_bc(pre / "bathymetry" / "paracas_bc.dat", conjuntos)
    escribir_run_pre(pre / "run_pre.dat", "paracas_bathymetry.dat", "paracas_bc.dat")

    # Caso de prueba: forzamiento del script 10 y run_aem3d.dat barotrópico
    with open(carpeta(cfg, "carpeta_procesados") / "mes_candidato.yaml", encoding="utf-8") as f:
        mes = yaml.safe_load(f)["mes"]
    origen = ruta(ae["carpeta"])
    met, mar = f"met_paracas_{mes}.dat", f"marea_paracas_{mes}.dat"
    for a in (met, mar):
        if not (origen / a).exists():
            raise SystemExit(f"Falta {a}: corre antes scripts/10_exportar_aem3d.py")
        shutil.copy2(origen / a, caso / "run" / "infiles" / a)
    texto = (origen / "LEEME_aem3d.txt").read_text(encoding="utf-8")
    inicio = next(l.split()[0] for l in texto.splitlines() if l.strip().endswith("start_date_cwr"))
    h_ini = float(next(l.split()[0] for l in texto.splitlines() if l.strip().endswith("DEFAULT_HEIGHT")))
    pr = ma["prueba"]
    iteraciones = int(pr["horas"] * 3600 / pr["del_t_s"])
    escribir_run_aem3d(caso / "run" / "run_aem3d.dat", cfg, met, mar, inicio, pr["del_t_s"], iteraciones, h_ini)
    (caso / "run" / "infiles" / "datablock.xml").write_text(
        DATABLOCK.format(intervalo=int(3600 / pr["del_t_s"])), encoding="ascii")
    log.info(f"Caso escrito en {relativa(caso)} (prueba: {pr['horas']} h con del_t = {pr['del_t_s']} s)")
    fig_malla(b, lats, lons, dz, conjuntos, figdir)

    resumen = {"dx_m": ma["dx_m"], "filas": int(b.shape[0]), "columnas": int(b.shape[1]),
               "columnas_agua": int(agua.sum()), "capas": len(dz), "prof_max_m": round(prof_max, 1),
               "celdas_abiertas": n_abiertas}
    if args.correr:
        resumen.update(correr(caso, pr, cfg))
    with open(carpeta(cfg, "carpeta_procesados") / "malla_aem3d_resumen.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(resumen, f, allow_unicode=True, sort_keys=False)


def correr(caso, pr, cfg) -> dict:
    """pre_aem3d y la corrida de prueba; revisa que termine y que el nivel siga la marea."""
    for prog, cwd in (("pre_aem3d", caso / "pre"), ("aem3d", caso / "run")):
        exe = shutil.which(prog) or str(RAIZ.parent / "bin" / prog)
        log.info(f"Ejecutando {prog} en {relativa(cwd)}")
        r = subprocess.run([exe], cwd=cwd, capture_output=True, text=True)
        (cwd / f"salida_{prog}.txt").write_text(r.stdout + r.stderr, encoding="utf-8")
        fin = (r.stdout + r.stderr).strip().splitlines()[-6:]
        if r.returncode != 0 or (prog == "aem3d" and "FINISHED" not in r.stdout):
            raise SystemExit(f"{prog} falló (código {r.returncode}); ver {relativa(cwd)}/salida_{prog}.txt:\n"
                             + "\n".join(fin))
        log.info(f"{prog} terminó bien")
    ds = xr.open_dataset(caso / "run" / "ncfiles" / "sheet_top.nc", decode_times=False)
    h = ds.HEIGHT.values
    h = np.where(np.abs(h) > 1e3, np.nan, h)
    serie = np.nanmean(h.reshape(h.shape[0], -1), axis=1)
    u = np.where(np.abs(ds.U_VELOCITY.values) > 1e3, np.nan, ds.U_VELOCITY.values)
    v = np.where(np.abs(ds.V_VELOCITY.values) > 1e3, np.nan, ds.V_VELOCITY.values)
    vmax = float(np.nanmax(np.hypot(u, v)))
    log.info(f"Prueba: nivel medio del dominio {[round(float(x), 3) for x in serie]} m; velocidad superficial máxima "
             f"{vmax:.2f} m/s")
    if not np.isfinite(vmax) or vmax > 3:
        log.warning("[sanidad] velocidades fuera de lo esperable (> 3 m/s): revisar del_t o la batimetría")
    else:
        log.info("[sanidad] velocidades razonables -> OK")
    # Nivel en la celda del mareógrafo frente a FES2022 en el mismo punto (misma hora del modelo)
    t = pd.DatetimeIndex(pd.to_datetime(dict(year=ds.Year.values, month=ds.Month.values, day=ds.Day.values,
                                             hour=ds.Hour.values, minute=ds.Minute.values,
                                             second=ds.Second.values))).round("min")
    mg = cfg["mareografo"]
    lat, lon = ds.LATITUDE.values, ds.LONGITUDE.values
    d2 = (lat - mg["lat"]) ** 2 + ((lon - mg["lon"]) * np.cos(np.deg2rad(mg["lat"]))) ** 2
    d2 = np.where(np.isnan(h[0]), np.inf, d2)
    i, j = np.unravel_index(np.argmin(d2), d2.shape)
    fes = pd.read_csv(carpeta(cfg, "carpeta_procesados") / "marea_frontera_mes_candidato.csv",
                      index_col=0, parse_dates=True).mareografo
    if cfg["aem3d"]["hora"] == "local":
        fes.index = fes.index + pd.Timedelta(hours=cfg["proyecto"]["zona_horaria_local"])
    fes_t = fes.reindex(fes.index.union(t)).interpolate("time").reindex(t)  # FES a la hora exacta de salida
    dif = pd.Series(h[:, i, j], index=t) - fes_t.values
    tras = dif[t >= t[0] + pd.Timedelta(hours=pr.get("arranque_h", 3))].dropna()
    dmax = float(tras.abs().max()) if len(tras) else float("nan")
    log.info(f"Mareógrafo (celda a {np.sqrt(d2[i, j]) * 111:.2f} km): AEM3D − FES2022 tras el arranque = "
             f"{(100 * tras).round(1).tolist()} cm")
    if np.isfinite(dmax) and dmax < 0.05:
        log.info("[sanidad] AEM3D reproduce la marea de FES2022 en el mareógrafo (< 5 cm) -> OK")
    else:
        log.warning("[sanidad] AEM3D se aleja de FES2022 en el mareógrafo más de 5 cm: revisar fronteras")
    return {"prueba_horas": pr["horas"], "nivel_medio_m": [round(float(x), 3) for x in serie],
            "vel_max_superficie_ms": round(vmax, 3),
            "dif_max_mareografo_cm": round(100 * dmax, 1) if np.isfinite(dmax) else None}


if __name__ == "__main__":
    main()
