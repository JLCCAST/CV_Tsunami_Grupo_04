"""Pruebas del horizonte (Rol 2): franjas de terreno y mar fuera del mapa. Sin ventana."""
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import config as cfg
from campos import Campos
from modulos import horizonte as hz
from modulos.horizonte import AguaHorizonte, Bordes, Horizonte, caida_ola, promedio_movil

G = cfg.GRID


@pytest.fixture(scope="module")
def bordes():
    return Bordes()


def test_promedio_movil():
    v = np.arange(10, dtype=float)
    assert np.array_equal(promedio_movil(v, 1), v)
    p = promedio_movil(v, 5)
    assert len(p) == 10 and p[5] == pytest.approx(5.0)
    assert np.allclose(promedio_movil(np.full(7, 3.0), 3), 3.0)


def test_el_primer_anillo_es_el_borde_del_mapa(bordes):
    H = np.random.default_rng(0).normal(size=(G, G))
    vals = bordes.valores(H)
    for s, (iz, ix, base, _d) in enumerate(bordes.lados):
        ini = s * len(bordes.R) * G
        assert np.allclose(vals[ini:ini + G], H[iz, ix])
        assert np.allclose(bordes.xz[ini:ini + G], base)


def test_las_franjas_cosen_sin_saltos(bordes):
    H = np.random.default_rng(1).normal(size=(G, G)) * 5.0
    vals = bordes.valores(H)
    clave = np.round(bordes.xz, 3)
    _u, inv = np.unique(clave, axis=0, return_inverse=True)
    inv = inv.ravel()
    for grupo in range(inv.max() + 1):
        mismos = vals[inv == grupo]
        assert np.ptp(mismos) < 1e-9          # vértices repetidos en las costuras tienen el mismo valor


def test_el_agua_levantada_baja_al_nivel_del_mar(bordes):
    H = np.full((G, G), -10.0)
    W = np.full((G, G), 16.0)                  # el mar subió 6 m
    terreno = bordes.alto_terreno(H)
    y, prof = bordes.agua(H, W, terreno)
    assert y[bordes.dist == 0].min() == pytest.approx(6.0)
    assert np.abs(y[bordes.dist >= hz.CAIDA_OLA_M]).max() < 1e-9
    assert (prof > 0).all()
    assert caida_ola(0.0) == 1.0 and caida_ola(hz.CAIDA_OLA_M) == 0.0


def test_en_tierra_seca_no_hay_agua(bordes):
    H = np.full((G, G), 5.0)
    terreno = bordes.alto_terreno(H)
    _y, prof = bordes.agua(H, np.zeros((G, G)), terreno)
    assert (prof <= 1e-9).all()
    assert np.allclose(terreno[bordes.dist == 0], 5.0)                     # nace al alto del borde
    lejos = terreno[bordes.dist >= hz.BAJA_TIERRA_M]
    assert np.allclose(lejos, 5.0 * (1 - hz.BAJA_TIERRA))                  # y baja suave a lo lejos


def test_el_fondo_marino_sigue_bajando(bordes):
    terreno = bordes.alto_terreno(np.full((G, G), -10.0))
    assert terreno[bordes.dist == 0].max() == pytest.approx(-10.0)
    assert terreno.min() == pytest.approx(hz.FONDO_MIN_M)


def test_el_agua_de_fuera_usa_el_mismo_reflejo_que_la_del_mapa():
    from modulos import materiales
    from modulos.iluminacion import armar_fragment
    for frag in (materiales.FRAG_AGUA, hz.FRAG_AGUA):
        assert materiales.REFLEJO_CIELO_GLSL in frag
        assert "reflejo_cielo(col, N, V, uCielo)" in frag
        assert armar_fragment(frag).count("vec3 reflejo_cielo(") == 1


def test_indices_validos(bordes):
    idx = bordes.indices
    assert len(idx) % 3 == 0 and idx.max() < len(bordes.xz)


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
    c.H[:, :40] = -10.0
    c.W[:, :40] = 10.0
    ctx = types.SimpleNamespace(vp=np.eye(4, dtype=np.float32), cam_pos=np.array([0.0, 300.0, 0.0]))
    h = Horizonte()
    h.iniciar(c, ctx)
    h.actualizar(c, 0.1)
    h.dibujar(c, ctx)
    assert gl.glDrawElements.call_count == 1          # terreno (opaco)
    AguaHorizonte(h).dibujar(c, ctx)
    assert gl.glDrawElements.call_count == 2          # y después el agua (transparente)
    h.cerrar()
    sys.modules.pop("gl_util", None)
