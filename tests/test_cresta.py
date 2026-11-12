"""Pruebas de la cresta de la ola (Rol 2): superficie de Bézier sobre frente_x. Sin ventana."""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import config as cfg
from agua import frente_ola as fo
from agua.frente_ola import FrenteOla
from campos import Campos
from modulos import cresta as cr
from modulos.cresta import Cresta, bernstein, factor_cresta, malla_cresta, parche_bezier
from modulos.terreno import Terreno

G = cfg.GRID


def _mar(profundidad=10.0):
    return np.full((G, G), -profundidad, np.float32)


def _frente(x):
    return np.full(G, x, np.float32)


def test_bernstein_suma_uno():
    B = bernstein(np.linspace(0, 1, 7))
    assert np.allclose(B.sum(axis=1), 1.0)


def test_parche_pasa_por_las_esquinas():
    P = np.random.default_rng(3).normal(size=(4, 4, 3))
    S = parche_bezier(P, 5, 6)
    assert S.shape == (5, 6, 3)
    for (i, j), (u, v) in {(0, 0): (0, 0), (3, 0): (-1, 0), (0, 3): (0, -1), (3, 3): (-1, -1)}.items():
        assert np.allclose(S[u, v], P[i, j])


def test_sin_frente_no_hay_cresta():
    assert len(malla_cresta(_frente(cfg.SIN_FRENTE), _mar())) == 0


def test_cresta_completa_en_agua_profunda():
    m = malla_cresta(_frente(200.0), _mar())
    assert m.shape[1] == 10 and len(m) > 0 and len(m) % 3 == 0
    y = m[:, 1]
    assert cr.NIVEL_ESPALDA_M + 0.5 * cr.ALTO_EXTRA_M <= y.max() <= cr.NIVEL_ESPALDA_M + 1.5 * cr.ALTO_EXTRA_M
    assert y.min() >= cfg.NIVEL_MAR_M - 1e-3
    assert m[:, 0].min() >= 200.0 - 35.0 - 1e-3 and m[:, 0].max() <= 200.0 + 10.0 + 1e-3
    assert m[:, 2].min() == pytest.approx(0.0) and m[:, 2].max() == pytest.approx((G - 1) * cfg.CELL_M)
    assert np.isfinite(m).all()


def test_la_espalda_nace_transparente_del_agua():
    m = malla_cresta(_frente(200.0), _mar())
    atras = m[:, 0] <= 200.0 - 34.0
    assert atras.any() and m[atras, 9].max() < 0.05          # se funde con el agua, sin borde visible
    assert m[:, 9].max() == pytest.approx(cr.ALFA, abs=0.01)


def test_la_altura_cambia_a_lo_largo_del_frente():
    m = malla_cresta(_frente(200.0), _mar())
    zs = np.unique(np.round(m[:, 2], 2))
    cimas = [m[np.isclose(m[:, 2], z, atol=0.01), 1].max() for z in zs]
    assert max(cimas) - min(cimas) > 0.5                      # no es un muro parejo


def test_la_cresta_baja_al_llegar_a_la_costa():
    profunda = malla_cresta(_frente(200.0), _mar(10.0))[:, 1].max()
    bajando = malla_cresta(_frente(200.0), _mar(2.0))[:, 1].max()
    assert bajando < profunda - 0.8
    assert len(malla_cresta(_frente(200.0), _mar(-1.0))) == 0          # frente sobre tierra: ya rompió
    assert factor_cresta(_frente(200.0), _mar(2.0)) == pytest.approx(np.full(G, 0.5))


def test_sigue_la_curva_del_frente():
    x = np.linspace(150.0, 250.0, G).astype(np.float32)
    m = malla_cresta(x, _mar())
    desfases = []
    for iz in (0, G // 2, G - 1):
        en_fila = np.isclose(m[:, 2], iz * cfg.CELL_M)
        cima = m[en_fila][np.argmax(m[en_fila, 1])]
        desfases.append(cima[0] - x[iz])
    assert all(-12.0 <= d <= 3.0 for d in desfases)        # la cima va justo en el frente
    assert max(desfases) - min(desfases) < 1.0             # en todas las filas a la misma distancia


@pytest.fixture(scope="module")
def terreno_real():
    c = Campos.vacios()
    Terreno().iniciar(c, None)
    return c


def test_con_la_ola_real(terreno_real):
    c = Campos.vacios()
    c.H[:] = terreno_real.H
    ola, cresta = FrenteOla(c), Cresta()
    ola.actualizar(fo.T_LLEGADA_S + 6.0)          # frente todavía en el mar
    cresta.actualizar(c, 0.1)
    assert len(cresta.vertices) > 0
    ola.actualizar(fo.T_RETIRO_S + 1.0)           # retiro: no hay frente publicado
    cresta.actualizar(c, 0.1)
    assert len(cresta.vertices) == 0


def test_ciclo_gl_con_contexto_simulado(monkeypatch):
    falso = types.ModuleType("OpenGL")
    gl = MagicMock()
    gl.glGetShaderiv.return_value = 1
    gl.glGetProgramiv.return_value = 1
    falso.GL = gl
    monkeypatch.setitem(sys.modules, "OpenGL", falso)
    monkeypatch.setitem(sys.modules, "OpenGL.GL", gl)
    sys.modules.pop("gl_util", None)
    c = Campos.vacios()
    c.H[:] = _mar()
    ctx = types.SimpleNamespace(vp=np.eye(4, dtype=np.float32))
    cresta = Cresta()
    cresta.iniciar(c, ctx)
    cresta.actualizar(c, 0.1)
    cresta.dibujar(c, ctx)
    assert gl.glDrawArrays.call_count == 0             # sin frente no dibuja nada
    c.frente_x[:] = 200.0
    cresta.actualizar(c, 0.1)
    cresta.dibujar(c, ctx)
    assert gl.glDrawArrays.call_count == 1
    cresta.cerrar()
    sys.modules.pop("gl_util", None)
