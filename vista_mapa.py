"""Imágenes 2D del mundo a partir de los campos (NumPy puro, sin OpenGL).

Se usan para la vista de depuración de la ventana y para el mini-mapa de riesgo del HUD.
"""
import numpy as np

import config as cfg

_ARENA = np.array([225, 205, 150], np.float32)
_VERDE = np.array([96, 150, 80], np.float32)
_ROCA = np.array([130, 110, 90], np.float32)
_AGUA = np.array([35, 110, 200], np.float32)
_COLOR_DANO = {
    cfg.DANO_INTACTA: (235, 235, 235),
    cfg.DANO_DANADA: (235, 150, 40),
    cfg.DANO_DESTRUIDA: (130, 30, 30),
}


def rampa_riesgo(x):
    """Rampa verde -> amarillo -> rojo para x en [0, 1] (acepta escalares o arreglos)."""
    x = np.clip(np.asarray(x, np.float32), 0.0, 1.0)
    verde = np.array([46, 204, 113], np.float32)
    amarillo = np.array([241, 196, 15], np.float32)
    rojo = np.array([231, 76, 60], np.float32)
    x = x[..., None]
    mitad_baja = verde + (amarillo - verde) * (x / 0.5)
    mitad_alta = amarillo + (rojo - amarillo) * ((x - 0.5) / 0.5)
    return np.where(x < 0.5, mitad_baja, mitad_alta)


def _terreno_rgb(H):
    t = np.clip(H / 15.0, 0.0, 1.0)[..., None]
    tierra = np.where(t < 0.25,
                      _ARENA + (_VERDE - _ARENA) * (t / 0.25),
                      _VERDE + (_ROCA - _VERDE) * ((t - 0.25) / 0.75))
    fondo = np.clip(-H / 6.0, 0.0, 1.0)[..., None]
    mar = np.array([60, 130, 190], np.float32) * (1.0 - 0.6 * fondo)
    return np.where((H >= cfg.NIVEL_MAR_M)[..., None], tierra, mar)


def imagen_mapa(campos) -> np.ndarray:
    """Terreno + agua + casas coloreadas por daño. uint8 [GRID, GRID, 3] (fila = z)."""
    base = _terreno_rgb(campos.H)
    opacidad = np.clip(campos.W / 4.0, 0.0, 1.0)[..., None] * 0.85
    opacidad = np.where((campos.W > cfg.UMBRAL_MOJADO_M)[..., None], np.maximum(opacidad, 0.25), 0.0)
    img = base * (1.0 - opacidad) + _AGUA * opacidad
    img = np.clip(img, 0, 255).astype(np.uint8)
    for (x0, z0, x1, z1), d in zip(campos.casas, campos.dano):
        img[z0:z1, x0:x1] = _COLOR_DANO[int(d)]
    return img


def imagen_riesgo(campos) -> np.ndarray:
    """Mapa de riesgo para el HUD: tierra en gris, agua en tierra firme en la rampa de riesgo."""
    H, W = campos.H, campos.W
    gris = np.clip(60 + H * 4.0, 40, 140).astype(np.float32)[..., None].repeat(3, axis=2)
    mar = np.array([25, 55, 100], np.float32)
    img = np.where((H < cfg.NIVEL_MAR_M)[..., None], mar, gris)
    en_tierra_mojada = (H >= cfg.NIVEL_MAR_M) & (W > cfg.UMBRAL_MOJADO_M)
    riesgo = rampa_riesgo(W / cfg.NIVEL_CRITICO_M)
    img = np.where(en_tierra_mojada[..., None], riesgo * 0.85 + gris * 0.15, img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    for (x0, z0, x1, z1), d in zip(campos.casas, campos.dano):
        img[z0:z1, x0:x1] = _COLOR_DANO[int(d)]
    return img
