"""Datos FALSOS (momentaneo) para que todos trabajen

Rellena todos los campos del contrato con valores razonables:
  * un terreno en pendiente,
  * unas casas de prueba,
  * un frente de ola que avanza y retrocede.

Cuando tu módulo real esté listo, deja de llamar a la parte que reemplaza.
"""
import numpy as np

import config as cfg
from agua.alarma import calcular_alarma

# Parámetros de la ola de prueba
VEL_FRENTE_MS = 12.0      # m/s a la que avanza el frente
ALTURA_OLA_M = 6.0        # altura de la superficie del agua detrás del frente
LARGO_COLA_M = 250.0      # distancia detrás del frente en la que la altura sube a su máximo
PERIODO_S = 45.0          # la ola se repite cada PERIODO_S segundos
PLAYA_X_M = 120.0         # donde el terreno cruza el nivel del mar


class DatosFalsos:
    def __init__(self, campos):
        self.campos = campos
        g = cfg.GRID
        self.x_m = (np.arange(g, dtype=np.float32) * cfg.CELL_M)[None, :].repeat(g, axis=0)
        self.z_m = (np.arange(g, dtype=np.float32) * cfg.CELL_M)[:, None].repeat(g, axis=1)
        self._terreno_falso()
        self._casas_falsas()
        self.actualizar(0.0)

    # Terreno y casas de prueba 
    def _terreno_falso(self):
        c = self.campos
        c.H[:] = ((self.x_m - PLAYA_X_M) * 0.03).astype(np.float32)   # de -3.6 m a +15.6 m

    def _casas_falsas(self):
        c = self.campos
        plano = [(48, 20, 3, 3), (56, 34, 4, 3), (50, 52, 3, 4),
                 (60, 70, 4, 4), (52, 88, 3, 3), (64, 104, 4, 3)]
        c.casas = np.array([[x, z, x + w, z + h] for x, z, w, h in plano], np.int32)
        c.dano = np.zeros(len(plano), np.uint8)
        c.obstaculo[:] = False
        for x0, z0, x1, z1 in c.casas:
            c.obstaculo[z0:z1, x0:x1] = True

    # Ola de prueba 
    def actualizar(self, t: float):
        c = self.campos
        ciclo = t % PERIODO_S
        frente = 20.0 + VEL_FRENTE_MS * ciclo        
        detras = np.clip(1.0 - (frente - self.x_m) / LARGO_COLA_M, 0.0, 1.0)
        superficie = np.where(self.x_m <= frente, ALTURA_OLA_M * detras, 0.0)
        c.W[:] = np.maximum(0.0, superficie - c.H).astype(np.float32)
        c.W[:] = np.maximum(c.W, np.maximum(0.0, cfg.NIVEL_MAR_M - c.H)).astype(np.float32)

        mojado_en_tierra = (c.W > cfg.UMBRAL_MOJADO_M) & (self.x_m > PLAYA_X_M) & (self.x_m <= frente)
        c.U[:] = np.where(mojado_en_tierra, 8.0 * np.clip(c.W, 0.0, 1.0), 0.0).astype(np.float32)
        c.V[:] = (0.4 * np.sin(self.z_m * 0.05 + t) * c.U).astype(np.float32)
        dentro = frente <= (cfg.GRID - 1) * cfg.CELL_M
        c.frente_x[:] = frente if dentro else cfg.SIN_FRENTE      # frente recto: igual en todas las filas

        c.t = float(t)
        c.alarma = calcular_alarma(c)
