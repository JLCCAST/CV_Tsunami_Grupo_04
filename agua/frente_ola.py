"""Simulación del agua, semana 2 (Rol 1): frente de ola simple.

Modelo cinemático por filas (una fila = un valor de iz):
  * Cada fila tiene la posición `xf` (en metros) del frente de la ola. El frente entra
    por el borde del mar (ix = 0) y avanza hacia la tierra a VEL_FRENTE_MS.
  * Detrás del frente la superficie del agua sube desde la altura del "muro" del frente
    (ALTURA_FRENTE_M) hasta ALTURA_OLA_M a lo largo de LARGO_COLA_M.
  * Una fila deja de avanzar cuando el terreno que tiene delante es más alto que el muro
    del frente (límite de alcance o "run-up"). En la semana 3 esto se reemplaza por un
    frenado según la pendiente y se agrega la máscara de obstáculos del rol 2.
  * Después de T_RETIRO_S la altura de la ola baja hasta cero en DURACION_RETIRO_S y el
    agua vuelve al mar (velocidad negativa).

Publica en `campos` (ver CONTRATO.md): W, U, V, t y alarma. Solo LEE H.
Se usa igual que DatosFalsos: `ola = FrenteOla(campos)` y en cada cuadro `ola.actualizar(t)`.
Si el tiempo retrocede (tecla R) la simulación se reinicia sola.
"""
import numpy as np

import config as cfg
from agua.alarma import calcular_alarma

# --- Parámetros de la ola ----------------------------------------------------
T_LLEGADA_S = 3.0           # segundo en que el frente entra por el borde del mar
VEL_FRENTE_MS = 10.0        # m/s a la que avanza el frente
ALTURA_OLA_M = 6.0          # altura máxima de la superficie (sobre el nivel del mar) detrás del frente
ALTURA_FRENTE_M = 3.0       # altura del muro de agua justo en el frente
LARGO_COLA_M = 200.0        # distancia detrás del frente en la que se llega a ALTURA_OLA_M
CURVATURA_M = 40.0          # el frente entra curvado: los extremos van esta distancia detrás del centro
T_RETIRO_S = 40.0           # segundo en que la ola empieza a bajar
DURACION_RETIRO_S = 20.0    # segundos que tarda en bajar del todo
FROUDE = 1.0                # rapidez del flujo = FROUDE * sqrt(g * profundidad)
FACTOR_RETIRO = 0.6         # el flujo de retirada es más lento que el de entrada
VEL_MAX_MS = 12.0           # tope de la rapidez del flujo
G = 9.81
DT_MAX_S = 0.1              # el frente se integra en subpasos de como máximo este tamaño
VENTANA_SUAVIZADO = 9       # filas que promedia el frente para calcular la dirección del flujo
PENDIENTE_MAX = 1.0         # |dxf/dz| máximo: el flujo se desvía como mucho 45° de la dirección x


def _suavizar(xf: np.ndarray) -> np.ndarray:
    """Promedio móvil de VENTANA_SUAVIZADO filas, repitiendo los bordes."""
    m = VENTANA_SUAVIZADO // 2
    ext = np.pad(xf, m, mode="edge")
    return np.convolve(ext, np.ones(VENTANA_SUAVIZADO) / VENTANA_SUAVIZADO, mode="valid")


class FrenteOla:
    def __init__(self, campos):
        self.campos = campos
        g = cfg.GRID
        self.x_m = np.arange(g, dtype=np.float32) * cfg.CELL_M          # [G]  centro de cada columna
        z = np.arange(g, dtype=np.float32)
        centro = (g - 1) / 2.0
        self.xf0 = (-CURVATURA_M * ((z - centro) / centro) ** 2).astype(np.float32)   # [G] frente inicial
        self.filas = np.arange(g)
        self.reiniciar()

    # --- estado -----------------------------------------------------------------
    def reiniciar(self):
        """Vuelve al instante anterior a la llegada de la ola."""
        self.xf = self.xf0.copy()                       # posición del frente por fila (m)
        self.avanzando = np.ones(cfg.GRID, bool)        # filas cuyo frente aún se mueve
        self.t_sim = 0.0
        self.base = np.maximum(0.0, cfg.NIVEL_MAR_M - self.campos.H).astype(np.float32)   # mar en calma

    def amplitud(self, t: float) -> float:
        """Factor de 0 a 1 que escala la altura de la ola (1 durante la llegada, baja en el retiro)."""
        if t < T_LLEGADA_S:
            return 0.0
        return float(np.clip(1.0 - (t - T_RETIRO_S) / DURACION_RETIRO_S, 0.0, 1.0))

    def en_retiro(self, t: float) -> bool:
        return t >= T_RETIRO_S

    # --- avance del frente --------------------------------------------------------
    def _avanzar_frente(self, t0: float, t1: float):
        """Integra la posición del frente entre t0 y t1 en subpasos de DT_MAX_S."""
        H = self.campos.H
        ultimo = cfg.GRID - 1
        t = t0
        while t < t1 - 1e-9:
            dt = min(DT_MAX_S, t1 - t)
            # solo se mueve entre la llegada y el retiro
            ini, fin = max(t, T_LLEGADA_S), min(t + dt, T_RETIRO_S)
            if fin > ini and self.avanzando.any():
                nuevo = self.xf + VEL_FRENTE_MS * (fin - ini)
                ix = np.clip((nuevo / cfg.CELL_M).astype(np.int64), 0, ultimo)
                bloqueado = H[self.filas, ix] >= cfg.NIVEL_MAR_M + ALTURA_FRENTE_M
                mover = self.avanzando & ~bloqueado
                self.xf = np.where(mover, np.minimum(nuevo, ultimo * cfg.CELL_M), self.xf).astype(np.float32)
                self.avanzando &= ~bloqueado
            t += dt
        self.t_sim = t1

    # --- campos ---------------------------------------------------------------------
    def actualizar(self, t: float):
        t = float(t)
        if t < self.t_sim - 1e-9:
            self.reiniciar()
        self._avanzar_frente(self.t_sim, t)

        c = self.campos
        a = self.amplitud(t)
        detras = self.x_m[None, :] <= self.xf[:, None]                                 # [G,G]
        cola = np.clip((self.xf[:, None] - self.x_m[None, :]) / LARGO_COLA_M, 0.0, 1.0)
        eta = a * (ALTURA_FRENTE_M + (ALTURA_OLA_M - ALTURA_FRENTE_M) * cola)           # sobre el nivel del mar
        ola = np.where(detras, np.maximum(0.0, cfg.NIVEL_MAR_M + eta - c.H), 0.0)
        c.W[:] = np.maximum(self.base, ola)

        # Rapidez del flujo solo donde hay agua de la ola (el mar en calma está quieto).
        con_ola = detras & (ola > cfg.UMBRAL_MOJADO_M)
        rapidez = np.minimum(FROUDE * np.sqrt(G * c.W), VEL_MAX_MS)
        if self.en_retiro(t):
            rapidez = -FACTOR_RETIRO * rapidez * (a > 0)
        rapidez = np.where(con_ola, rapidez, 0.0)

        # Dirección: normal a la línea del frente, x = xf(z), suavizada para que las filas
        # detenidas junto a filas que siguen no produzcan un flujo casi lateral.
        pendiente = np.clip(np.gradient(_suavizar(self.xf), cfg.CELL_M), -PENDIENTE_MAX, PENDIENTE_MAX)
        norma = np.sqrt(1.0 + pendiente ** 2)
        c.U[:] = rapidez * (1.0 / norma)[:, None]
        c.V[:] = rapidez * (-pendiente / norma)[:, None]

        c.t = t
        c.alarma = calcular_alarma(c)

    # --- consultas útiles para el HUD y las pruebas --------------------------------
    def frente_ix(self) -> np.ndarray:
        """Columna del frente en cada fila (int, recortada a la grilla)."""
        return np.clip((self.xf / cfg.CELL_M).astype(np.int64), 0, cfg.GRID - 1)
