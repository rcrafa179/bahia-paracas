"""Verificaciones de sanidad de los datos. Avisan en el log o detienen el script."""
from __future__ import annotations

import numpy as np


class ErrorSanidad(RuntimeError):
    pass


def rango(nombre, x, vmin, vmax, log, fraccion_max=0.0, detener=True):
    """Comprueba que los valores de x estén en [vmin, vmax].

    fraccion_max: fracción tolerada de valores fuera de rango antes de fallar.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        msg = f"[sanidad] {nombre}: no hay datos válidos"
        if detener:
            raise ErrorSanidad(msg)
        log.warning(msg)
        return False
    fuera = np.mean((x < vmin) | (x > vmax))
    txt = f"[sanidad] {nombre}: min={x.min():.3g}, max={x.max():.3g}, fuera de [{vmin}, {vmax}] = {100 * fuera:.2f} %"
    if fuera > fraccion_max:
        if detener:
            raise ErrorSanidad(txt)
        log.warning(txt)
        return False
    log.info(txt + " -> OK")
    return True


def rango_marea(nombre, h, log, esperado=(0.5, 3.0)):
    """Rango de marea (máximo - mínimo de un periodo largo) del orden de 1-2 m en Pisco."""
    h = np.asarray(h, dtype=float)
    h = h[np.isfinite(h)]
    if h.size == 0:
        raise ErrorSanidad(f"[sanidad] {nombre}: serie de marea vacía")
    # Se usan percentiles para no depender de un pico aislado
    r = np.percentile(h, 99.9) - np.percentile(h, 0.1)
    txt = f"[sanidad] {nombre}: rango (p99.9 - p0.1) = {r:.2f} m (esperado entre {esperado[0]} y {esperado[1]} m)"
    if not (esperado[0] <= r <= esperado[1]):
        log.warning(txt + " -> REVISAR")
        return False
    log.info(txt + " -> OK")
    return True
