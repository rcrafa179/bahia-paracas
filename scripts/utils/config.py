"""Carga de config.yaml y argumentos comunes de línea de comandos."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

# Raíz del repositorio: dos niveles arriba de scripts/utils/
RAIZ = Path(__file__).resolve().parents[2]


def ruta(rel: str | Path) -> Path:
    """Convierte una ruta relativa del config en absoluta respecto de la raíz."""
    p = Path(rel).expanduser()
    return p if p.is_absolute() else RAIZ / p


def cargar_config(archivo: str | Path | None = None) -> dict:
    """Lee config.yaml (por defecto el de la raíz del repositorio)."""
    archivo = ruta(archivo) if archivo else RAIZ / "config.yaml"
    with open(archivo, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parser_base(descripcion: str) -> argparse.ArgumentParser:
    """Parser con los argumentos comunes a todos los scripts."""
    p = argparse.ArgumentParser(description=descripcion)
    p.add_argument("--config", default=None, help="ruta a config.yaml (por defecto, el de la raíz)")
    p.add_argument("--inicio", default=None, help="fecha de inicio AAAA-MM-DD (sobrescribe el config)")
    p.add_argument("--fin", default=None, help="fecha de fin AAAA-MM-DD (sobrescribe el config)")
    p.add_argument("--forzar", action="store_true", help="rehacer descargas/salidas aunque ya existan")
    return p


def periodo(cfg: dict, args) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Devuelve (inicio, fin) en UTC; fin incluye el día completo."""
    ini = pd.Timestamp(args.inicio or cfg["periodo"]["inicio"])
    fin = pd.Timestamp(args.fin or cfg["periodo"]["fin"])
    fin = fin + pd.Timedelta(hours=23)  # hasta las 23:00 UTC del último día
    if fin < ini:
        raise ValueError(f"Periodo inválido: {ini} > {fin}")
    return ini, fin


def carpeta(cfg: dict, clave: str) -> Path:
    """Crea (si hace falta) y devuelve una carpeta de salidas."""
    p = ruta(cfg["salidas"][clave])
    p.mkdir(parents=True, exist_ok=True)
    return p
