"""Pruebas del B-rep (Rol 2): fórmula de Euler, cierre, orientación y volumen. Sin ventana."""
import numpy as np
import pytest

import config as cfg
from modulos.brep import BRep, prisma, techo_dos_aguas
from modulos.casas import TIPOS, _dims, malla_casa, solido_casa, techo_casa
from modulos.csg import volumen_firmado

CUADRADO = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]


def _normales_hacia_afuera(b):
    centro = b.vertices.mean(axis=0)
    for k, c in enumerate(b.caras):
        cara = b.vertices[list(c)].mean(axis=0)
        assert np.dot(b.normal_cara(k), cara - centro) > 0, f"la cara {k} mira hacia adentro"


def test_cubo_cumple_euler():
    cubo = prisma(CUADRADO, 0.0, 1.0)
    assert (len(cubo.vertices), len(cubo.aristas()), len(cubo.caras)) == (8, 12, 6)
    assert cubo.euler() == 2
    assert cubo.es_cerrado()
    assert cubo.volumen() == pytest.approx(1.0)
    _normales_hacia_afuera(cubo)


@pytest.mark.parametrize("n", [3, 5, 6, 12])
def test_prisma_de_n_lados(n):
    ang = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    perfil = [(np.cos(a), np.sin(a)) for a in ang]
    p = prisma(perfil, -2.0, 3.0)
    assert (len(p.vertices), len(p.aristas()), len(p.caras)) == (2 * n, 3 * n, n + 2)
    assert p.euler() == 2 and p.es_cerrado()
    area = 0.5 * n * np.sin(2 * np.pi / n)
    assert p.volumen() == pytest.approx(area * 5.0)


def test_detecta_frontera_abierta():
    cubo = prisma(CUADRADO, 0.0, 1.0)
    abierto = BRep(cubo.vertices, cubo.caras[:-1], cubo.materiales[:-1])
    assert not abierto.es_cerrado()
    assert abierto.euler() == 1


def test_detecta_cara_mal_orientada():
    cubo = prisma(CUADRADO, 0.0, 1.0)
    caras = list(cubo.caras)
    caras[0] = tuple(reversed(caras[0]))
    assert not BRep(cubo.vertices, caras, cubo.materiales).es_cerrado()


def test_techo_dos_aguas():
    t = techo_dos_aguas(4.0, 6.0, 3.0, 2.0, alero=0.5)
    assert (len(t.vertices), len(t.aristas()), len(t.caras)) == (6, 9, 5)
    assert t.euler() == 2 and t.es_cerrado()
    assert t.volumen() == pytest.approx(0.5 * 9.0 * 2.0 * 13.0)
    _normales_hacia_afuera(t)
    assert t.vertices[:, 1].min() == pytest.approx(3.0) and t.vertices[:, 1].max() == pytest.approx(5.0)


@pytest.mark.parametrize("tipo", TIPOS)
def test_techo_de_cada_casa(tipo):
    for estado in (cfg.DANO_INTACTA, cfg.DANO_DANADA):
        t = techo_casa(*tipo, estado)
        assert t.euler() == 2 and t.es_cerrado() and t.volumen() > 0
    assert techo_casa(*tipo, cfg.DANO_DESTRUIDA) is None
    assert techo_casa(*tipo, cfg.DANO_DANADA).volumen() < techo_casa(*tipo, cfg.DANO_INTACTA).volumen()


@pytest.mark.parametrize("estado", [cfg.DANO_INTACTA, cfg.DANO_DANADA, cfg.DANO_DESTRUIDA])
def test_malla_de_la_casa_suma_paredes_y_techo(estado):
    w, d, p = 3, 4, 2
    m = malla_casa(w, d, p, estado)
    esperado = solido_casa(w, d, p, estado).volumen()
    techo = techo_casa(w, d, p, estado)
    if techo is not None:
        esperado += techo.volumen()
    assert volumen_firmado(m) == pytest.approx(esperado, rel=1e-4)


def test_el_techo_se_apoya_sobre_la_losa():
    w, d, p = 3, 3, 1
    _sx, _sz, hx, hz, alto = _dims(w, d, p, cfg.CELL_M)
    t = techo_casa(w, d, p, cfg.DANO_INTACTA)
    losa_lo, losa_hi = solido_casa(w, d, p, cfg.DANO_INTACTA).caja_envolvente()
    assert t.vertices[:, 1].min() == pytest.approx(losa_hi[1])
    assert t.vertices[:, 0].max() == pytest.approx(losa_hi[0])
    assert t.vertices[:, 2].min() == pytest.approx(losa_lo[2])
