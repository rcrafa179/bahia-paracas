#!/usr/bin/env python
"""00 - Descarga de FES2022b (elevaciones extrapoladas) desde el FTP de AVISO.

Baja SOLO las constituyentes de modelo_marea.constituyentes (config.yaml), no las 34,
y descomprime cada archivo .xz al vuelo (no queda el comprimido en disco).

Requisitos (una sola vez):
  1. Cuenta en AVISO+ con acceso a "FES (Finite Element Solution - Oceanic Tides Heights)":
     https://www.aviso.altimetry.fr/en/data/data-access/registration-form.html
  2. Credenciales en ~/.netrc (NUNCA dentro del repositorio), una línea:
        machine ftp-access.aviso.altimetry.fr login TU_CORREO password TU_CONTRASEÑA
     y permisos privados:  chmod 600 ~/.netrc

Servidor y carpeta (verificados en la página de AVISO y en pyTMD.datasets.fetch_aviso_fes):
  ftp://ftp-access.aviso.altimetry.fr/auxiliary/tide_model/fes2022b/ocean_tide_extrapolated/

Salida: <modelo_marea.directorio>/fes2022b/ocean_tide_extrapolated/<constituyente>_fes2022.nc

Uso:
    python scripts/00_fes2022_download.py --listar     # solo lista archivos y tamaños
    python scripts/00_fes2022_download.py              # descarga lo que falte
    python scripts/00_fes2022_download.py --forzar     # vuelve a bajar todo
"""
from __future__ import annotations

import ftplib
import lzma
import netrc
import posixpath
import shutil
import time
from pathlib import Path

from utils.config import RAIZ, cargar_config, parser_base, ruta
from utils.log import obtener_logger

log = obtener_logger("00_fes2022_download")
MB = 1024 ** 2


def credenciales(host: str) -> tuple[str, str]:
    """Lee usuario y contraseña de ~/.netrc. Nunca se piden por línea de comandos."""
    archivo = Path.home() / ".netrc"
    if not archivo.exists():
        raise SystemExit(f"No existe {archivo}. Créalo con la línea:\n"
                         f"  machine {host} login TU_CORREO password TU_CONTRASEÑA\n"
                         "y luego: chmod 600 ~/.netrc")
    try:
        datos = netrc.netrc(archivo).authenticators(host)
    except netrc.NetrcParseError as e:
        raise SystemExit(f"Error de formato en {archivo}: {e}")
    if datos is None:
        raise SystemExit(f"{archivo} no tiene una entrada para 'machine {host}'")
    usuario, _, clave = datos
    if (archivo.stat().st_mode & 0o077) != 0:
        log.warning(f"{archivo} es legible por otros usuarios: ejecuta 'chmod 600 ~/.netrc'")
    return usuario, clave


def constituyente(nombre: str) -> str:
    """'m2_fes2022.nc.xz' -> 'm2'."""
    return nombre.split("_", 1)[0].lower()


def listar_remotos(ftp: ftplib.FTP, carpeta: str, buscadas: list[str]) -> dict[str, tuple[str, int]]:
    """Devuelve {constituyente: (nombre_remoto, tamaño_bytes)} solo para las buscadas."""
    try:
        nombres = [posixpath.basename(n) for n in ftp.nlst(carpeta)]
    except ftplib.error_perm as e:
        raise SystemExit(f"No se pudo listar {carpeta}: {e}\n"
                         "¿Tu cuenta AVISO tiene acceso a FES? Las carpetas sin suscripción aparecen vacías.")
    if not nombres:
        raise SystemExit(f"La carpeta {carpeta} aparece vacía: tu cuenta aún no tiene acceso a FES2022")
    ftp.voidcmd("TYPE I")  # modo binario, necesario para SIZE
    encontrados = {}
    for n in sorted(nombres):
        c = constituyente(n)
        if c in buscadas and (n.endswith(".nc.xz") or n.endswith(".nc")):
            encontrados[c] = (n, ftp.size(posixpath.join(carpeta, n)) or 0)
    log.info(f"Archivos en el servidor: {len(nombres)}; se usarán {len(encontrados)} de {len(buscadas)} pedidos")
    faltan = [c for c in buscadas if c not in encontrados]
    if faltan:
        log.warning(f"No están en el servidor: {faltan} (revisa los nombres en config.yaml)")
    return encontrados


def descargar(ftp: ftplib.FTP, remoto: str, destino: Path, tam: int):
    """Baja un archivo y, si es .xz, lo descomprime mientras llega. Escribe en .part y renombra al final."""
    parcial = destino.with_suffix(destino.suffix + ".part")
    descompresor = lzma.LZMADecompressor() if remoto.endswith(".xz") else None
    recibido = 0
    ultimo = time.time()
    t0 = ultimo

    with open(parcial, "wb") as f:
        def bloque(datos: bytes):
            nonlocal recibido, ultimo
            recibido += len(datos)
            f.write(descompresor.decompress(datos) if descompresor else datos)
            if time.time() - ultimo > 15:  # avance cada 15 s
                pct = 100 * recibido / tam if tam else float("nan")
                log.info(f"    {recibido / MB:7.1f} / {tam / MB:.1f} MB ({pct:.0f} %)")
                ultimo = time.time()

        ftp.retrbinary(f"RETR {remoto}", bloque, blocksize=1024 * 1024)

    if descompresor is not None and not descompresor.eof:
        parcial.unlink(missing_ok=True)
        raise RuntimeError(f"{remoto}: el archivo .xz llegó incompleto")
    if tam and recibido != tam:
        parcial.unlink(missing_ok=True)
        raise RuntimeError(f"{remoto}: se recibieron {recibido} bytes de {tam}")
    parcial.rename(destino)
    seg = max(time.time() - t0, 1e-3)
    log.info(f"    listo: {destino.name} ({destino.stat().st_size / MB:.0f} MB en disco, "
             f"{recibido / MB / seg:.1f} MB/s)")


def verificar(archivo: Path):
    """Chequeo de sanidad: el NetCDF abre y tiene amplitud y fase."""
    import xarray as xr

    with xr.open_dataset(archivo) as ds:
        variables = set(ds.data_vars)
    if not {"amplitude", "phase"} <= variables:
        log.warning(f"    {archivo.name}: variables inesperadas {sorted(variables)}")


def main():
    p = parser_base("Descarga FES2022b (elevaciones extrapoladas) desde el FTP de AVISO")
    p.add_argument("--listar", action="store_true", help="solo listar archivos y tamaños, sin descargar")
    args = p.parse_args()
    cfg = cargar_config(args.config)
    mm = cfg["modelo_marea"]
    dsc = mm["descarga"]
    buscadas = [c.lower() for c in mm["constituyentes"]]

    base = ruta(mm["directorio"])
    destino = base / Path(mm["archivos"]).parent  # .../fes2022b/ocean_tide_extrapolated
    usuario, clave = credenciales(dsc["host"])

    log.info(f"Conectando a {dsc['host']} como {usuario}")
    try:
        ftp = ftplib.FTP(dsc["host"], timeout=dsc.get("timeout_s", 120))
        ftp.login(usuario, clave)
    except ftplib.error_perm as e:
        raise SystemExit(f"AVISO rechazó el inicio de sesión: {e}")
    except OSError as e:
        raise SystemExit(f"No se pudo conectar a {dsc['host']}: {e}")

    try:
        remotos = listar_remotos(ftp, dsc["carpeta_remota"], buscadas)
        total = sum(t for _, t in remotos.values())
        for c, (n, t) in remotos.items():
            log.info(f"  {c:>4}: {n} ({t / MB:.0f} MB comprimido)")
        log.info(f"Total a bajar: {total / MB:.0f} MB comprimidos")
        if args.listar:
            return

        destino.mkdir(parents=True, exist_ok=True)
        libre = shutil.disk_usage(destino).free
        # Descomprimido ocupa más que el .xz; se exige un margen holgado
        necesario = 3 * total + 2 * 1024 ** 3
        log.info(f"Espacio libre en disco: {libre / 1024 ** 3:.1f} GiB")
        if libre < necesario:
            raise SystemExit(f"Poco espacio: hay {libre / 1024 ** 3:.1f} GiB y se piden al menos "
                             f"{necesario / 1024 ** 3:.1f} GiB (3 x comprimido + 2 GiB)")

        for i, (c, (n, t)) in enumerate(remotos.items(), 1):
            local = destino / n.removesuffix(".xz")
            if local.exists() and not args.forzar:
                log.info(f"[{i}/{len(remotos)}] {local.name} ya existe, se omite")
                continue
            log.info(f"[{i}/{len(remotos)}] Bajando {n} ({t / MB:.0f} MB)")
            descargar(ftp, posixpath.join(dsc["carpeta_remota"], n), local, t)
            verificar(local)
    finally:
        try:
            ftp.quit()
        except Exception:
            ftp.close()

    archivos = sorted(destino.glob("*_fes2022.nc"))
    tam = sum(a.stat().st_size for a in archivos)
    log.info(f"FES2022b listo: {len(archivos)} archivos, {tam / 1024 ** 3:.2f} GiB en "
             f"{destino.relative_to(RAIZ) if destino.is_relative_to(RAIZ) else destino}")
    log.info("Siguiente paso: python scripts/07_tide_model_pytmd.py "
             "(después se pueden borrar los globales: 07 guarda un recorte regional)")


if __name__ == "__main__":
    main()
