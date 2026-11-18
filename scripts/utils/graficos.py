"""Estilo común de figuras (rótulos en español, PNG a 300 dpi)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin ventana: los scripts solo guardan PNG
import matplotlib.pyplot as plt  # noqa: E402

MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"]
ESTACIONES = {"DEF": "Verano (DEF)", "MAM": "Otoño (MAM)", "JJA": "Invierno (JJA)", "SON": "Primavera (SON)"}

plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.formatter.use_locale": False,
    "axes.titleweight": "bold",
})


def guardar(fig, carpeta: Path, nombre: str, log=None, dpi: int = 300) -> Path:
    """Guarda la figura en PNG y la cierra."""
    carpeta.mkdir(parents=True, exist_ok=True)
    salida = carpeta / f"{nombre}.png"
    fig.savefig(salida, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    if log:
        log.info(f"Figura guardada: {salida.relative_to(carpeta.parent)}")
    return salida


# --- Rosas de viento -------------------------------------------------------
BINS_VIENTO = [0, 2, 4, 6, 8, 10, 12]  # m/s
# windrose usa ángulos matemáticos (0° = este, antihorario): etiquetas en ese orden
ETIQ_RUMBOS = ["E", "NE", "N", "NO", "O", "SO", "S", "SE"]


def eje_rosa(fig, pos):
    """Crea un eje de rosa de vientos en la posición (filas, columnas, índice)."""
    from windrose import WindroseAxes  # noqa: F401  (registra la proyección 'windrose')

    return fig.add_subplot(*pos, projection="windrose")


def rosa(ax, direccion, velocidad, titulo, bins=BINS_VIENTO, leyenda=True):
    """Dibuja una rosa de vientos (dirección de donde viene el viento) en %."""
    import numpy as np

    ok = np.isfinite(direccion) & np.isfinite(velocidad)
    ax.bar(np.asarray(direccion)[ok], np.asarray(velocidad)[ok], normed=True, opening=0.85,
           edgecolor="white", linewidth=0.4, bins=bins, cmap=plt.cm.viridis, nsector=16)
    ax.set_thetagrids(range(0, 360, 45), ETIQ_RUMBOS)
    ax.set_title(titulo, pad=14)
    # Frecuencias radiales en %
    ax.set_yticklabels([f"{t:.0f} %" for t in ax.get_yticks()], fontsize=7)
    if leyenda:
        ax.set_legend(title="Velocidad (m/s)", loc="lower left", bbox_to_anchor=(1.02, 0.0),
                      fontsize=7, title_fontsize=8, decimal_places=0)
    return ax
