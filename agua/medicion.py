"""Cronómetro de pasos de simulación (Rol 1).

Guarda cuánto tarda cada paso en milisegundos y da el promedio, el percentil 95,
el máximo y un promedio móvil (EMA) que sirve para mostrar en el HUD sin que el número salte.

    crono = Cronometro()
    with crono:
        hacer_un_paso()
    print(crono.resumen())
"""
import time
from collections import deque

import numpy as np

ALFA_EMA = 0.05             # peso del último paso en el promedio móvil


class Cronometro:
    def __init__(self, max_muestras: int = 20000):
        self.muestras = deque(maxlen=max_muestras)
        self.reiniciar()

    def reiniciar(self):
        self.muestras.clear()
        self.pasos = 0
        self.total_ms = 0.0
        self.maximo_ms = 0.0
        self.ultimo_ms = 0.0
        self.ema_ms = None
        self._inicio = None

    # --- medir -----------------------------------------------------------------
    def __enter__(self):
        self._inicio = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.registrar((time.perf_counter() - self._inicio) * 1000.0)
        return False

    def registrar(self, ms: float):
        self.pasos += 1
        self.total_ms += ms
        self.maximo_ms = max(self.maximo_ms, ms)
        self.ultimo_ms = ms
        self.ema_ms = ms if self.ema_ms is None else self.ema_ms + ALFA_EMA * (ms - self.ema_ms)
        self.muestras.append(ms)

    # --- resultados ------------------------------------------------------------
    @property
    def media_ms(self) -> float:
        return self.total_ms / self.pasos if self.pasos else 0.0

    def percentil_ms(self, p: float) -> float:
        return float(np.percentile(self.muestras, p)) if self.muestras else 0.0

    def resumen(self) -> str:
        return (f"{self.pasos} pasos, media {self.media_ms:.3f} ms, "
                f"p95 {self.percentil_ms(95):.3f} ms, máximo {self.maximo_ms:.3f} ms")
