import numpy as np

import config as cfg
from modulos.brep import techo_dos_aguas
from modulos.csg import (
    MAT_CIMIENTO,
    MAT_ESCOMBRO,
    MAT_MADERA,
    MAT_PARED,
    MAT_PARED_DANADA,
    MAT_TECHO,
    Solido,
)

ALTO_PISO = 3.0
ESPESOR = 0.4
Y0 = 0.15
ALERO = 0.4             # cuánto sobresale la losa del techo de las paredes
ESPESOR_LOSA = 0.3
PENDIENTE_TECHO = 0.5   # alto del techo a dos aguas = PENDIENTE_TECHO * medio ancho de la casa

TIPOS = [(w, d, p) for w in (2, 3, 4) for d in (2, 3, 4) for p in (1, 2)]


def _dims(w, d, pisos, cell_m):
    sx, sz = w * cell_m - 1.0, d * cell_m - 1.0
    return sx, sz, sx / 2, sz / 2, ALTO_PISO * pisos


def solido_simple(w, d, pisos, estado, cell_m=None):
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    sx, sz, hx, hz, alto = _dims(w, d, pisos, cell_m)
    s = Solido()
    if estado == cfg.DANO_INTACTA:
        s.sumar((-hx, -0.6, -hz), (hx, alto, hz), MAT_PARED)
    elif estado == cfg.DANO_DANADA:
        s.sumar((-hx, -0.6, -hz), (hx, alto, hz), MAT_PARED_DANADA)
    else:
        s.sumar((-hx, -0.6, -hz), (hx, 0.6, hz), MAT_ESCOMBRO)
    return s


def solido_casa(w, d, pisos, estado, cell_m=None):
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    sx, sz, hx, hz, alto = _dims(w, d, pisos, cell_m)
    t = ESPESOR
    y0 = Y0
    ytech = y0 + alto
    rng = np.random.default_rng([cfg.SEMILLA, w, d, pisos, estado])

    s = Solido()
    s.sumar((-hx, -0.6, -hz), (hx, y0, hz), MAT_CIMIENTO)
    s.sumar((-hx + 0.1, y0, -hz + 0.1), (hx - 0.1, ytech, hz - 0.1), MAT_PARED)
    s.restar((-hx + 0.1 + t, y0 + 0.2, -hz + 0.1 + t), (hx - 0.1 - t, ytech - 0.3, hz - 0.1 - t))
    if pisos == 2:
        s.sumar(
            (-hx + 0.1 + t, y0 + ALTO_PISO, -hz + 0.1 + t),
            (hx - 0.1 - t, y0 + ALTO_PISO + 0.25, hz - 0.1 - t),
            MAT_PARED,
        )
    s.sumar((-hx - ALERO, ytech, -hz - ALERO), (hx + ALERO, ytech + ESPESOR_LOSA, hz + ALERO), MAT_TECHO)

    s.restar((-hx - 0.2, y0 + 0.2, -0.7), (-hx + 0.1 + t + 0.1, y0 + 2.4, 0.7))

    nx = max(1, int(sx // 3.6))
    nz = max(1, int(sz // 3.6))
    for p in range(pisos):
        yv0 = y0 + p * ALTO_PISO + 0.9
        yv1 = yv0 + 1.1
        for k in range(nx):
            xc = -hx + (k + 0.5) * sx / nx
            s.restar((xc - 0.6, yv0, -hz - 0.2), (xc + 0.6, yv1, -hz + 0.1 + t + 0.1))
            s.restar((xc - 0.6, yv0, hz - 0.1 - t - 0.1), (xc + 0.6, yv1, hz + 0.2))
            s.sumar((xc - 0.7, yv0 - 0.12, -hz - 0.05), (xc + 0.7, yv0, -hz + 0.2), MAT_MADERA)
            s.sumar((xc - 0.7, yv0 - 0.12, hz - 0.2), (xc + 0.7, yv0, hz + 0.05), MAT_MADERA)
        for k in range(nz):
            zc = -hz + (k + 0.5) * sz / nz
            s.restar((hx - 0.1 - t - 0.1, yv0, zc - 0.6), (hx + 0.2, yv1, zc + 0.6))
            s.sumar((hx - 0.2, yv0 - 0.12, zc - 0.7), (hx + 0.05, yv0, zc + 0.7), MAT_MADERA)
            if p > 0 or abs(zc) > 1.6:
                s.restar((-hx - 0.2, yv0, zc - 0.6), (-hx + 0.1 + t + 0.1, yv1, zc + 0.6))
                s.sumar((-hx - 0.05, yv0 - 0.12, zc - 0.7), (-hx + 0.2, yv0, zc + 0.7), MAT_MADERA)

    if estado == cfg.DANO_INTACTA:
        return s

    if estado == cfg.DANO_DANADA:
        s.pintar((-hx - 1, y0, -hz - 1), (hx + 1, ytech, hz + 1), MAT_PARED_DANADA)
        ancho = float(np.clip(0.4 * sz, 2.5, max(2.5, sz - 1.5)))
        zc = float(rng.uniform(-0.25 * hz, 0.25 * hz))
        s.restar(
            (-hx - 0.5, y0 + 0.2, zc - ancho / 2),
            (-hx + 0.1 + t + 0.2, y0 + min(alto - 0.6, 2.9), zc + ancho / 2),
        )
        s.restar((-hx - 1, ytech - 0.3, -hz - 1), (-hx + 0.45 * sx, ytech + 3, -hz + 0.5 * sz))
        return s

    n = 3
    xs = np.linspace(-hx - 0.6, hx + 0.6, n + 1)
    zs = np.linspace(-hz - 0.6, hz + 0.6, n + 1)
    for i in range(n):
        for j in range(n):
            corte = float(rng.uniform(0.35, 1.7))
            s.restar((xs[i], y0 + corte, zs[j]), (xs[i + 1], ytech + 3, zs[j + 1]))
    s.pintar((-hx - 1, y0, -hz - 1), (hx + 1, ytech + 3, hz + 1), MAT_ESCOMBRO)
    return s


def techo_casa(w, d, pisos, estado, cell_m=None):
    """Techo a dos aguas en B-rep (ver brep.py), apoyado sobre la losa del CSG.

    Intacta: techo completo. Dañada: solo la mitad de atrás, porque el CSG ya le quitó la
    esquina de adelante a la losa. Destruida: sin techo (devuelve None).
    """
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    _sx, _sz, hx, hz, alto = _dims(w, d, pisos, cell_m)
    if estado == cfg.DANO_DESTRUIDA:
        return None
    y_base = Y0 + alto + ESPESOR_LOSA
    alto_techo = PENDIENTE_TECHO * hx
    if estado == cfg.DANO_INTACTA:
        return techo_dos_aguas(hx, hz, y_base, alto_techo, ALERO)
    return techo_dos_aguas(hx, hz / 2, y_base, alto_techo, ALERO).trasladar((0.0, 0.0, hz / 2))


def malla_casa(w, d, pisos, estado, cell_m=None, simple=False):
    if simple:
        return solido_simple(w, d, pisos, estado, cell_m).malla()
    m = solido_casa(w, d, pisos, estado, cell_m).malla()
    techo = techo_casa(w, d, pisos, estado, cell_m)
    return m if techo is None else np.concatenate([m, techo.malla()])


def piezas_casa(w, d, pisos, cell_m=None, semilla=0, cortes=(3, 2, 3)):
    c = cortes if pisos == 1 else (cortes[0], cortes[1] + 1, cortes[2])
    return solido_casa(w, d, pisos, cfg.DANO_INTACTA, cell_m).partir(cortes=c, semilla=semilla)
