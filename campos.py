"""Campos compartidos entre todos los módulos.

Todos los módulos leen y escriben SOBRE ESTE MISMO objeto. Nadie crea arreglos
propios con otra forma o con otras unidades.

Convención de ejes (importante, no cambiar sin avisar):
  * Los arreglos se indexan [iz, ix]  (fila = z, columna = ix).
  * ix crece hacia la tierra: el MAR está en ix pequeño (borde izquierdo).
  * iz crece "hacia abajo" en el mapa 2D.
  * En 3D, y es la altura. El punto (x, z) en metros es (ix * CELL_M, iz * CELL_M).
"""
from dataclasses import dataclass, field

import numpy as np

import config as cfg


@dataclass
class Campos:
    # --- Terreno --------------------------------------
    H: np.ndarray = field(default=None)          # float32 [G,G] altura del terreno en m (<0 bajo el mar)
    obstaculo: np.ndarray = field(default=None)  # bool    [G,G] True = celda ocupada por una casa
    casas: np.ndarray = field(default=None)      # int32   [N,4] cajas (ix0, iz0, ix1, iz1), ix1/iz1 exclusivos
    dano: np.ndarray = field(default=None)       # uint8   [N]   0 intacta, 1 dañada, 2 destruida

    # --- Agua -----------------------------------------
    W: np.ndarray = field(default=None)          # float32 [G,G] profundidad del agua sobre el terreno, m (>= 0)
    U: np.ndarray = field(default=None)          # float32 [G,G] velocidad en x, m/s
    V: np.ndarray = field(default=None)          # float32 [G,G] velocidad en z, m/s
    alarma: float = 0.0                          # 0.0 (calma) a 1.0 (crítica)

    # --- Reloj de la simulación ---------------------------------------------
    t: float = 0.0                               # segundos de simulación

    @staticmethod
    def vacios() -> "Campos":
        g = cfg.GRID
        return Campos(
            H=np.zeros((g, g), np.float32),
            obstaculo=np.zeros((g, g), bool),
            casas=np.zeros((0, 4), np.int32),
            dano=np.zeros((0,), np.uint8),
            W=np.zeros((g, g), np.float32),
            U=np.zeros((g, g), np.float32),
            V=np.zeros((g, g), np.float32),
            alarma=0.0,
            t=0.0,
        )

    def validar(self) -> None:
        """Lanza AssertionError si algún campo rompe el contrato.

        Úsala en tus pruebas: `campos.validar()` antes de publicar tu semana.
        """
        g = cfg.GRID
        for nombre, dtype in (("H", np.float32), ("W", np.float32), ("U", np.float32),
                              ("V", np.float32), ("obstaculo", np.bool_)):
            a = getattr(self, nombre)
            assert isinstance(a, np.ndarray), f"{nombre} debe ser np.ndarray"
            assert a.shape == (g, g), f"{nombre} debe ser ({g},{g}), es {a.shape}"
            assert a.dtype == dtype, f"{nombre} debe ser {np.dtype(dtype)}, es {a.dtype}"
        assert (self.W >= 0).all(), "W (profundidad) no puede ser negativa"
        assert np.isfinite(self.H).all() and np.isfinite(self.W).all(), "H y W deben ser finitos"
        assert np.isfinite(self.U).all() and np.isfinite(self.V).all(), "U y V deben ser finitos"
        assert 0.0 <= self.alarma <= 1.0, "alarma debe estar entre 0 y 1"
        assert self.casas.ndim == 2 and self.casas.shape[1] == 4, "casas debe ser [N,4]"
        assert self.casas.dtype == np.int32, "casas debe ser int32"
        assert self.dano.shape == (self.casas.shape[0],), "dano debe tener una entrada por casa"
        assert self.dano.dtype == np.uint8, "dano debe ser uint8"
        assert (self.dano <= cfg.DANO_DESTRUIDA).all(), "dano solo admite 0, 1 o 2"
        if len(self.casas):
            assert (self.casas[:, 0] < self.casas[:, 2]).all() and (self.casas[:, 1] < self.casas[:, 3]).all()
            assert self.casas.min() >= 0 and self.casas.max() <= g, "casas fuera de la grilla"


def celda_de_metros(x_m: float, z_m: float) -> tuple:
    """(x, z) en metros -> (ix, iz) de la grilla, recortado a los bordes."""
    ix = int(np.clip(x_m / cfg.CELL_M, 0, cfg.GRID - 1))
    iz = int(np.clip(z_m / cfg.CELL_M, 0, cfg.GRID - 1))
    return ix, iz
