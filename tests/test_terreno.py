import hashlib
import sys
import types
from unittest.mock import MagicMock

import numpy as np
import pytest

import config as cfg
from agua.datos_falsos import DatosFalsos
from campos import Campos
from modulos.casas import TIPOS, solido_casa
from modulos.csg import Solido, volumen_firmado
from modulos.dano import ModeloDano
from modulos.terreno import Terreno
from modulos.terreno_gen import altura_en, bezier, diamond_square, generar_escenario, malla_terreno


def _hash(a):
    return hashlib.sha1(np.ascontiguousarray(a).tobytes()).hexdigest()


def test_determinismo_semilla():
    a, b = generar_escenario(), generar_escenario()
    assert _hash(a.H) == _hash(b.H)
    assert _hash(a.casas) == _hash(b.casas)
    assert generar_escenario(semilla=7).H.tobytes() != a.H.tobytes()


def test_forma_y_perfil_de_costa():
    e = generar_escenario()
    g = cfg.GRID
    assert e.H.shape == (g, g) and e.H.dtype == np.float32
    assert (e.H[:, :3] < cfg.NIVEL_MAR_M).all()
    assert (e.H[:, -3:] > cfg.NIVEL_CRITICO_M).all()
    for iz in range(g):
        fila = e.H[iz]
        ix_c = int(np.argmax(fila >= cfg.NIVEL_MAR_M))
        assert 10 < ix_c < 60
        assert (fila[ix_c:] >= -0.5).all()


def test_diamond_square_rango_y_semilla():
    a = diamond_square(64, 0.85, np.random.default_rng(1))
    b = diamond_square(64, 0.85, np.random.default_rng(1))
    assert a.shape == (65, 65) and np.array_equal(a, b)
    assert abs(a).max() == 1.0


def test_bezier_extremos():
    c = [(0, 0), (1, 2), (3, 2), (4, 0)]
    p = bezier(c, [0, 1])
    assert np.allclose(p[0], c[0]) and np.allclose(p[1], c[-1])


def test_costa_en_forma_de_bahia():
    e = generar_escenario()
    g = cfg.GRID
    centro, puntas = e.x_costa[g // 2], (e.x_costa[0], e.x_costa[-1])
    assert centro > max(puntas) + 60.0                  # el centro de la costa entra tierra adentro
    assert np.allclose(e.x_costa, e.x_costa[::-1], atol=cfg.CELL_M)      # simétrica


def test_calles_en_bezier_y_sin_casas_encima():
    from modulos.terreno_gen import CTRL_COSTA_X, D_AVENIDAS, calles_bezier
    e = generar_escenario()
    assert len(e.calles) == 5 and e.mascara_calles.any()
    ancho = cfg.GRID * cfg.CELL_M
    for d, avenida in zip(D_AVENIDAS, calles_bezier()[:2]):
        assert np.allclose(avenida[0], (CTRL_COSTA_X[0] + d, 0.0))     # Bézier pasa por sus extremos
        assert np.allclose(avenida[-1], (CTRL_COSTA_X[-1] + d, ancho))
    for x0, z0, x1, z1 in e.casas:
        assert not e.mascara_calles[z0:z1, x0:x1].any()
    v, _ = malla_terreno(e.H, calles=e.mascara_calles)
    gris = v[e.mascara_calles.ravel(), 6:9]
    assert np.allclose(gris[:, 0], gris[:, 1], atol=0.08) and np.allclose(gris[:, 1], gris[:, 2], atol=0.08)


def test_muelles_sobre_el_mar():
    from modulos.terreno_gen import malla_muelles
    e = generar_escenario()
    assert len(e.muelles) == 3
    for m in e.muelles:
        assert altura_en(e.H, m["x0"], m["z"]) < cfg.NIVEL_MAR_M - 1.0      # la punta está sobre el agua
        assert altura_en(e.H, m["x1"], m["z"]) > cfg.NIVEL_MAR_M             # el otro extremo toca tierra
    malla = malla_muelles(e.muelles, e.H)
    assert len(malla) > 0 and volumen_firmado(malla) > 0


def test_casas_validas():
    e = generar_escenario()
    assert 30 <= len(e.casas) <= 120
    ocupado = np.zeros((cfg.GRID, cfg.GRID), int)
    for (ix0, iz0, ix1, iz1), (w, d, p), y in zip(e.casas, e.tipos, e.base_y):
        assert 2 <= ix1 - ix0 <= 4 and 2 <= iz1 - iz0 <= 4
        assert (ix1 - ix0, iz1 - iz0) == (w, d)
        assert (e.H[iz0 : iz1 + 1, ix0 : ix1 + 1] == y).all() and y >= 0.9
        ocupado[iz0:iz1, ix0:ix1] += 1
    assert ocupado.max() == 1


def test_mallas_cerradas_y_orientadas():
    for tipo in TIPOS:
        for estado in (cfg.DANO_INTACTA, cfg.DANO_DANADA, cfg.DANO_DESTRUIDA):
            s = solido_casa(*tipo, estado)
            m = s.malla()
            assert len(m) > 0 and len(m) % 3 == 0
            assert np.isclose(volumen_firmado(m), s.volumen(), rtol=1e-4), (tipo, estado)


def test_estados_progresivos():
    v = [solido_casa(3, 3, 1, e).volumen() for e in (0, 1, 2)]
    assert v[0] > v[1] > v[2] > 0


def test_csg_resta():
    s = Solido().sumar((0, 0, 0), (2, 2, 2), 2).restar((1, 1, 1), (3, 3, 3))
    assert np.isclose(s.volumen(), 7)


def test_piezas():
    for pisos in (1, 2):
        piezas = solido_casa(4, 4, pisos, 0).partir(cortes=(3, 1 + pisos, 3), semilla=1)
        assert 6 <= len(piezas) <= 40
        for p in piezas:
            assert np.isclose(volumen_firmado(p["malla"]), p["volumen"], rtol=1e-3)


def test_terreno_malla():
    e = generar_escenario()
    v, idx = malla_terreno(e.H)
    g = cfg.GRID
    assert v.shape == (g * g, 9) and len(idx) == (g - 1) ** 2 * 6
    assert (v[:, 4] > 0).all()
    assert np.isclose(altura_en(e.H, 0.0, 0.0), e.H[0, 0])
    assert np.isclose(altura_en(e.H, 10 * cfg.CELL_M, 20 * cfg.CELL_M), e.H[20, 10])


def test_dano_umbrales_y_persistencia():
    g = cfg.GRID
    casas = np.array([[10, 10, 13, 13], [30, 30, 33, 33]], np.int32)
    m = ModeloDano(casas, g)
    W, U, V = (np.zeros((g, g), np.float32) for _ in range(3))
    dano = np.zeros(2, np.uint8)
    W[9:14, 9:14] = cfg.NIVEL_CRITICO_M + 0.5
    U[9:14, 9:14] = 3.0
    assert m.actualizar(W, U, V, dano, 0.3) == [] and dano[0] == 0
    cambios = m.actualizar(W, U, V, dano, 0.4)
    assert cambios == [0] and dano.tolist() == [2, 0]
    W[:] = 0
    m.actualizar(W, U, V, dano, 1.0)
    assert dano[0] == 2
    W[29:35, 29:35], U[29:35, 29:35] = 0.8, 4.0
    m.actualizar(W, U, V, dano, 1.0)
    assert dano[1] == 1
    W[:] = 0
    U[:] = 0
    W[29:35, 29:35] = 1.2
    m2 = ModeloDano(casas, g)
    d2 = np.zeros(2, np.uint8)
    m2.actualizar(W, U, V, d2, 1.0)
    assert d2[1] == 0


def _campos_con_terreno(**kw):
    campos = Campos.vacios()
    falsos = DatosFalsos(campos)
    terreno = Terreno(**kw)
    terreno.iniciar(campos, None)
    campos.validar()
    return campos, falsos, terreno


@pytest.mark.parametrize("simple", [False, True])
def test_publica_en_campos_y_cumple_contrato(simple):
    campos, falsos, terreno = _campos_con_terreno(modo_simple=simple)
    assert len(campos.casas) == len(terreno.escenario.casas) > 0
    esperado = np.zeros((cfg.GRID, cfg.GRID), bool)
    for x0, z0, x1, z1 in campos.casas:
        esperado[z0:z1, x0:x1] = True
    assert np.array_equal(campos.obstaculo, esperado)
    assert np.array_equal(campos.H, terreno.escenario.H)
    for t in np.arange(0.0, 30.0, 0.5):
        falsos.actualizar(t)
        terreno.actualizar(campos, 0.5)
        campos.validar()


def test_ola_falsa_destruye_y_libera_obstaculo():
    campos, falsos, terreno = _campos_con_terreno()
    for t in np.arange(0.0, 40.0, 0.1):
        falsos.actualizar(t)
        terreno.actualizar(campos, 0.1)
    cuenta = np.bincount(campos.dano, minlength=3)
    assert cuenta[cfg.DANO_DESTRUIDA] > 0
    asignado = sum((c[2] - c[0]) * (c[3] - c[1]) for c, d in zip(campos.casas, campos.dano) if d < 2)
    assert campos.obstaculo.sum() == asignado
    campos.validar()


def test_reinicio_del_reloj_repara_casas():
    campos, falsos, terreno = _campos_con_terreno()
    for t in np.arange(0.0, 40.0, 0.1):
        falsos.actualizar(t)
        terreno.actualizar(campos, 0.1)
    assert campos.dano.max() > 0
    falsos.actualizar(0.0)
    terreno.actualizar(campos, 0.0)
    assert campos.dano.max() == 0
    esperado = np.zeros((cfg.GRID, cfg.GRID), bool)
    for x0, z0, x1, z1 in campos.casas:
        esperado[z0:z1, x0:x1] = True
    assert np.array_equal(campos.obstaculo, esperado)


def test_ciclo_gl_con_contexto_simulado(monkeypatch):
    falso = types.ModuleType("OpenGL")
    gl = MagicMock()
    gl.glGetShaderiv.return_value = 1
    gl.glGetProgramiv.return_value = 1
    falso.GL = gl
    monkeypatch.setitem(sys.modules, "OpenGL", falso)
    monkeypatch.setitem(sys.modules, "OpenGL.GL", gl)
    sys.modules.pop("gl_util", None)
    campos = Campos.vacios()
    DatosFalsos(campos)
    terreno = Terreno()
    ctx = types.SimpleNamespace(ancho_fb=1280, alto_fb=720, vp=np.eye(4, dtype=np.float32))
    terreno.iniciar(campos, ctx)
    terreno.dibujar(campos, ctx)
    assert gl.glDrawArrays.call_count == len(campos.casas) + 1      # una por casa y una para los muelles
    gl.glDrawArrays.reset_mock()
    terreno.dibujar_fragmentos(ctx, 0, 1.5)
    assert gl.glDrawArrays.call_count >= 6
    terreno.dibujar(campos, types.SimpleNamespace())
    terreno.cerrar()
    sys.modules.pop("gl_util", None)
