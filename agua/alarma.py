"""Estado de alarma. Lo leen el HUD y el shader."""
import numpy as np

import config as cfg


def nivel_max_en_tierra(campos) -> float:
    """Mayor profundidad de agua en celdas de tierra firme."""
    tierra = campos.H >= cfg.NIVEL_MAR_M
    if not tierra.any():
        return 0.0
    return float(campos.W[tierra].max())


def calcular_alarma(campos) -> float:
    """Valor de 0 a 1: la profundidad máxima en tierra dividida por el nivel crítico."""
    return float(np.clip(nivel_max_en_tierra(campos) / cfg.NIVEL_CRITICO_M, 0.0, 1.0))
