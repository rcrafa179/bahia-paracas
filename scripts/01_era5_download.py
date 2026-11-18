#!/usr/bin/env python
"""01 - Descarga de ERA5 horario de niveles únicos (CDS) en bloques mensuales.

Las credenciales se leen de ~/.cdsapirc (nunca del repositorio). Antes de la
primera descarga hay que aceptar la licencia de ERA5 en la página del dataset.

Uso:
    python scripts/01_era5_download.py                       # periodo del config
    python scripts/01_era5_download.py --inicio 2025-01-01 --fin 2025-01-31
"""
from __future__ import annotations

import calendar
import shutil
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
import xarray as xr

from utils.config import cargar_config, parser_base, periodo, ruta
from utils.log import obtener_logger

log = obtener_logger("01_era5_download")


def area_descarga(cfg: dict) -> list[float]:
    """Área [N, O, S, E] con margen, como la pide el CDS."""
    d, m = cfg["dominio"], cfg["era5"]["margen_grados"]
    return [d["lat_max"] + m, d["lon_min"] - m, d["lat_min"] - m, d["lon_max"] + m]


def peticion(cfg: dict, anio: int, mes: int) -> dict:
    """Arma la petición según la sintaxis actual del CDS (cdsapi >= 0.7.7)."""
    ndias = calendar.monthrange(anio, mes)[1]
    return {
        "product_type": ["reanalysis"],
        "variable": cfg["era5"]["variables"],
        "year": [f"{anio}"],
        "month": [f"{mes:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndias + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": area_descarga(cfg),
    }


def normalizar(archivo_descargado: Path, destino: Path) -> None:
    """Deja un único NetCDF por mes con coordenada 'time'.

    El CDS a veces entrega un ZIP con dos NetCDF (variables instantáneas y
    acumuladas). En ese caso se extraen y se unen.
    """
    with tempfile.TemporaryDirectory() as tmp:
        if zipfile.is_zipfile(archivo_descargado):
            with zipfile.ZipFile(archivo_descargado) as z:
                z.extractall(tmp)
            partes = sorted(Path(tmp).glob("*.nc"))
            log.info(f"  ZIP con {len(partes)} archivos: {[p.name for p in partes]}")
        else:
            partes = [archivo_descargado]
        dsets = []
        for p in partes:
            ds = xr.open_dataset(p).load()
            if "valid_time" in ds.dims:
                ds = ds.rename({"valid_time": "time"})
            ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables])
            dsets.append(ds)
        ds = xr.merge(dsets, compat="override", join="outer")
        ds.attrs["fuente"] = "ERA5 hourly single levels, Copernicus CDS"
        enc = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
        ds.to_netcdf(destino, encoding=enc)


def main():
    p = parser_base("Descarga ERA5 horario para el dominio de Paracas")
    args = p.parse_args()
    cfg = cargar_config(args.config)
    ini, fin = periodo(cfg, args)
    carpeta = ruta(cfg["era5"]["carpeta_raw"])
    carpeta.mkdir(parents=True, exist_ok=True)

    import cdsapi  # se importa aquí para que --help funcione sin credenciales

    if not Path("~/.cdsapirc").expanduser().exists():
        raise SystemExit("No existe ~/.cdsapirc. Ver README (credenciales del CDS).")
    cliente = cdsapi.Client(quiet=True)

    meses = pd.period_range(ini, fin, freq="M")
    log.info(f"Dominio de descarga [N, O, S, E] = {area_descarga(cfg)}")
    log.info(f"Meses a descargar: {len(meses)} ({meses[0]} a {meses[-1]})")
    for per in meses:
        destino = carpeta / f"era5_{per.year}{per.month:02d}.nc"
        if destino.exists() and not args.forzar:
            log.info(f"{per}: ya existe {destino.name}, se omite")
            continue
        log.info(f"{per}: solicitando al CDS ...")
        with tempfile.TemporaryDirectory() as tmp:
            bruto = Path(tmp) / "descarga.bin"
            cliente.retrieve(cfg["era5"]["dataset"], peticion(cfg, per.year, per.month), str(bruto))
            normalizar(bruto, Path(tmp) / "mes.nc")
            shutil.move(Path(tmp) / "mes.nc", destino)
        with xr.open_dataset(destino) as ds:
            log.info(f"{per}: {destino.name} ({destino.stat().st_size / 1e6:.1f} MB), "
                     f"{ds.sizes['time']} horas, {ds.sizes['latitude']}x{ds.sizes['longitude']} puntos, "
                     f"variables {list(ds.data_vars)}")
    log.info("Descarga ERA5 terminada")


if __name__ == "__main__":
    main()
