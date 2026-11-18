#!/usr/bin/env python
"""00 - Descarga de FES2022b (elevaciones extrapoladas) desde el FTP de AVISO.

Baja SOLO las constituyentes de modelo_marea.constituyentes (config.yaml), no las 34.
Cada .xz se baja a <archivo>.nc.xz.part; si la conexión se corta (el FTP de AVISO es lento,
~15 min por archivo) se reconecta y se REANUDA desde el último byte recibido (comando REST).
Al completarse se descomprime a .nc y se borra el comprimido. Volver a correr el script
también reanuda: los .part que queden se aprovechan.

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
    python scripts/00_fes2022_download.py              # descarga lo que falte (reanuda cortes)
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

# Errores de red tras los que vale la pena reconectar y reanudar
ERRORES_RED = (OSError, EOFError, ftplib.error_temp, ftplib.error_reply)


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


class Sesion:
    """Conexión FTP que se rehace cuando se cae (un corte a mitad de archivo la deja inservible)."""

    def __init__(self, host: str, usuario: str, clave: str, timeout: float):
        self.host, self.usuario, self.clave, self.timeout = host, usuario, clave, timeout
        self._ftp: ftplib.FTP | None = None

    @property
    def ftp(self) -> ftplib.FTP:
        if self._ftp is None:
            ftp = ftplib.FTP(self.host, timeout=self.timeout)
            try:
                ftp.login(self.usuario, self.clave)
            except ftplib.error_perm as e:
                ftp.close()
                raise SystemExit(f"AVISO rechazó el inicio de sesión: {e}")
            ftp.voidcmd("TYPE I")  # modo binario, necesario para SIZE y REST
            self._ftp = ftp
        return self._ftp

    def cerrar(self, cortada: bool = False):
        """Cierra la conexión; si se cayó, sin esperar respuesta del servidor."""
        if self._ftp is None:
            return
        try:
            if cortada:
                self._ftp.close()
            else:
                self._ftp.quit()
        except Exception:
            self._ftp.close()
        self._ftp = None


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
    # nlst deja la conexión en modo ASCII y AVISO rechaza SIZE en ese modo ("550 SIZE not allowed")
    ftp.voidcmd("TYPE I")
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


def bajar_reanudable(sesion: Sesion, remoto: str, parcial: Path, tam: int,
                     reintentos: int, espera: float) -> bool:
    """Baja `remoto` en `parcial`, continuando desde lo que ya haya en disco.

    Devuelve True si el archivo quedó completo (tamaño == tam), False si se agotaron los reintentos."""
    fallos = 0
    while True:
        hecho = parcial.stat().st_size if parcial.exists() else 0
        if tam and hecho == tam:
            return True
        if tam and hecho > tam:  # sobrante raro: se empieza de nuevo
            log.warning(f"    {parcial.name} es más grande que el remoto; se reinicia")
            parcial.unlink()
            hecho = 0
        if fallos > reintentos:
            return False
        if hecho:
            log.info(f"    reanudando desde {hecho / MB:.1f} MB de {tam / MB:.1f} MB")

        recibido = hecho
        ultimo = t0 = time.time()
        try:
            with open(parcial, "ab") as f:
                def bloque(datos: bytes):
                    nonlocal recibido, ultimo
                    f.write(datos)
                    recibido += len(datos)
                    if time.time() - ultimo > 15:  # avance cada 15 s
                        pct = 100 * recibido / tam if tam else float("nan")
                        vel = (recibido - hecho) / MB / max(time.time() - t0, 1e-3)
                        log.info(f"    {recibido / MB:7.1f} / {tam / MB:.1f} MB ({pct:.0f} %, {vel:.2f} MB/s)")
                        ultimo = time.time()

                sesion.ftp.retrbinary(f"RETR {remoto}", bloque, blocksize=1024 * 1024, rest=hecho or None)
            if tam and recibido < tam:  # el servidor cerró antes de tiempo sin avisar error
                raise EOFError(f"llegaron {recibido} de {tam} bytes")
        except ftplib.error_perm as e:
            # Respuesta 5xx: si fue por REST (servidor sin reanudación) se baja completo, una sola vez
            sesion.cerrar(cortada=True)
            if hecho and fallos == 0:
                log.warning(f"    el servidor no aceptó reanudar ({e}); se baja completo")
                parcial.unlink(missing_ok=True)
                fallos += 1
                continue
            raise SystemExit(f"{remoto}: el servidor rechazó la descarga: {e}")
        except ERRORES_RED as e:
            fallos += 1
            sesion.cerrar(cortada=True)
            if fallos > reintentos:
                log.warning(f"    corte de conexión ({type(e).__name__}: {e}); sin reintentos restantes")
                continue
            log.warning(f"    corte de conexión ({type(e).__name__}: {e}); "
                        f"intento {fallos}/{reintentos}, reconecto en {espera:.0f} s")
            time.sleep(espera)


def descomprimir(comprimido: Path, destino: Path):
    """Descomprime un .xz a `destino` (vía .part) y borra el comprimido."""
    parcial = destino.with_suffix(destino.suffix + ".part")
    descompresor = lzma.LZMADecompressor()
    with open(comprimido, "rb") as fi, open(parcial, "wb") as fo:
        while bloque := fi.read(8 * MB):
            fo.write(descompresor.decompress(bloque))
    if not descompresor.eof:
        parcial.unlink(missing_ok=True)
        comprimido.unlink(missing_ok=True)
        raise RuntimeError(f"{comprimido.name}: el .xz está incompleto o dañado; se borró, vuelve a correr el script")
    parcial.rename(destino)
    comprimido.unlink()


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
    reintentos, espera = dsc.get("reintentos", 8), dsc.get("espera_reintento_s", 20)

    log.info(f"Conectando a {dsc['host']} como {usuario}")
    sesion = Sesion(dsc["host"], usuario, clave, dsc.get("timeout_s", 120))
    fallidos = []
    try:
        try:
            remotos = listar_remotos(sesion.ftp, dsc["carpeta_remota"], buscadas)
        except ERRORES_RED as e:
            raise SystemExit(f"No se pudo conectar a {dsc['host']}: {e}")
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
            parcial = destino / (n + ".part")
            if args.forzar:
                local.unlink(missing_ok=True)
                parcial.unlink(missing_ok=True)
            if local.exists():
                log.info(f"[{i}/{len(remotos)}] {local.name} ya existe, se omite")
                continue
            # Restos de la versión anterior (descompresión al vuelo): no sirven para reanudar
            local.with_suffix(local.suffix + ".part").unlink(missing_ok=True)
            log.info(f"[{i}/{len(remotos)}] Bajando {n} ({t / MB:.0f} MB)")
            t0 = time.time()
            if not bajar_reanudable(sesion, posixpath.join(dsc["carpeta_remota"], n), parcial, t,
                                    reintentos, espera):
                log.error(f"    {n}: se agotaron los {reintentos} reintentos; queda {parcial.name} para reanudar")
                fallidos.append(n)
                continue
            if n.endswith(".xz"):
                descomprimir(parcial, local)
            else:
                parcial.rename(local)
            log.info(f"    listo: {local.name} ({local.stat().st_size / MB:.0f} MB en disco, "
                     f"{(time.time() - t0) / 60:.1f} min)")
            verificar(local)
    finally:
        sesion.cerrar()

    archivos = sorted(destino.glob("*_fes2022.nc"))
    tam = sum(a.stat().st_size for a in archivos)
    log.info(f"FES2022b en disco: {len(archivos)} archivos, {tam / 1024 ** 3:.2f} GiB en "
             f"{destino.relative_to(RAIZ) if destino.is_relative_to(RAIZ) else destino}")
    if fallidos:
        raise SystemExit(f"No se completaron {len(fallidos)} archivos: {fallidos}. "
                         "Vuelve a correr el script: reanuda donde quedó.")
    log.info("Siguiente paso: python scripts/07_tide_model_pytmd.py "
             "(después se pueden borrar los globales: 07 guarda un recorte regional)")


if __name__ == "__main__":
    main()
