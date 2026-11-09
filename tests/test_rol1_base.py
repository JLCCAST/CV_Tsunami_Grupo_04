"""Pruebas del Rol 1 (no necesitan ventana ni GPU).

Cubren el contrato de campos, la alarma de 0 a 1, los algoritmos propios del HUD
(Bresenham y punto medio) y el panel lateral.
"""
import math

import numpy as np
import pytest

import config as cfg
from agua.alarma import calcular_alarma, nivel_max_en_tierra
from agua.datos_falsos import DatosFalsos
from campos import Campos, celda_de_metros
from hud.lienzo import Lienzo, puntos_bresenham, puntos_circulo_punto_medio
from hud.panel import PanelHUD


# ---------------------------------------------------------------------------
# Contrato de campos
# ---------------------------------------------------------------------------
def test_campos_vacios_cumplen_contrato():
    Campos.vacios().validar()


@pytest.mark.parametrize("romper", [
    lambda c: setattr(c, "W", -np.ones_like(c.W)),                         # profundidad negativa
    lambda c: setattr(c, "U", np.zeros((cfg.GRID, cfg.GRID), np.float64)),   # tipo equivocado
    lambda c: setattr(c, "H", np.zeros((10, 10), np.float32)),               # forma equivocada
    lambda c: setattr(c, "alarma", 1.5),                                     # fuera de [0, 1]
    lambda c: c.V.__setitem__((0, 0), np.nan),                               # no finito
    lambda c: c.frente_x.__setitem__(0, -5.0),                               # frente fuera de rango
    lambda c: setattr(c, "frente_x", np.zeros(cfg.GRID, np.float64)),        # tipo equivocado
])
def test_validar_detecta_campos_rotos(romper):
    c = Campos.vacios()
    romper(c)
    with pytest.raises(AssertionError):
        c.validar()


def test_celda_de_metros_recorta_a_la_grilla():
    assert celda_de_metros(0.0, 0.0) == (0, 0)
    assert celda_de_metros(cfg.CELL_M * 3.5, cfg.CELL_M * 7.2) == (3, 7)
    assert celda_de_metros(-50.0, 1e9) == (0, cfg.GRID - 1)


@pytest.mark.parametrize("t", [0.0, 5.0, 17.3, 30.0, 44.9, 90.0])
def test_datos_falsos_respetan_contrato(t):
    c = Campos.vacios()
    DatosFalsos(c).actualizar(t)
    c.validar()
    assert c.t == pytest.approx(t)


def test_datos_falsos_publican_un_frente_recto():
    from agua import datos_falsos as df
    c = Campos.vacios()
    DatosFalsos(c).actualizar(5.0)
    assert np.allclose(c.frente_x, 20.0 + df.VEL_FRENTE_MS * 5.0)


# ---------------------------------------------------------------------------
# Alarma
# ---------------------------------------------------------------------------
def _campos_rampa():
    """Mar en las primeras 20 columnas, tierra firme en el resto."""
    c = Campos.vacios()
    c.H[:, :20] = -5.0
    c.H[:, 20:] = 2.0
    return c


def test_alarma_cero_si_tierra_seca():
    c = _campos_rampa()
    c.W[:, :20] = 5.0                      # el mar tiene agua, pero no cuenta
    assert nivel_max_en_tierra(c) == 0.0
    assert calcular_alarma(c) == 0.0


def test_alarma_proporcional_al_nivel_en_tierra():
    c = _campos_rampa()
    c.W[10, 40] = cfg.NIVEL_CRITICO_M / 2
    assert calcular_alarma(c) == pytest.approx(0.5)


def test_alarma_satura_en_uno():
    c = _campos_rampa()
    c.W[10, 40] = cfg.NIVEL_CRITICO_M * 4
    assert calcular_alarma(c) == 1.0


def test_alarma_sin_tierra_firme():
    c = Campos.vacios()
    c.H[:] = -1.0
    c.W[:] = 1.0
    assert calcular_alarma(c) == 0.0


# ---------------------------------------------------------------------------
# Bresenham
# ---------------------------------------------------------------------------
DESTINOS = [(x, y) for x in range(-17, 18, 4) for y in range(-17, 18, 4)]


@pytest.mark.parametrize("x1,y1", DESTINOS)
def test_bresenham_extremos_y_largo(x1, y1):
    p = puntos_bresenham(0, 0, x1, y1)
    assert p[0] == (0, 0) and p[-1] == (x1, y1)
    assert len(p) == max(abs(x1), abs(y1)) + 1          # un píxel por paso del eje mayor


@pytest.mark.parametrize("x1,y1", DESTINOS)
def test_bresenham_continua_y_cerca_de_la_recta(x1, y1):
    p = puntos_bresenham(0, 0, x1, y1)
    for (xa, ya), (xb, yb) in zip(p[:-1], p[1:]):
        assert max(abs(xb - xa), abs(yb - ya)) == 1      # 8-conectada, sin huecos ni repetidos
    largo = math.hypot(x1, y1) or 1.0
    for x, y in p:
        assert abs(x * y1 - y * x1) / largo <= 0.5 + 1e-9  # error menor a medio píxel


def test_bresenham_punto_unico():
    assert puntos_bresenham(3, 4, 3, 4) == [(3, 4)]


# ---------------------------------------------------------------------------
# Círculo por punto medio
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("r", [1, 2, 5, 10, 23, 46, 60])
def test_punto_medio_error_menor_a_medio_pixel(r):
    for dx, dy in puntos_circulo_punto_medio(r):
        assert abs(math.hypot(dx, dy) - r) < 0.5


@pytest.mark.parametrize("r", [3, 10, 46])
def test_punto_medio_simetria_de_ocho_octantes(r):
    pts = set(puntos_circulo_punto_medio(r))
    for x, y in pts:
        for s in ((y, x), (-x, y), (x, -y), (-x, -y), (-y, x), (y, -x), (-y, -x)):
            assert s in pts


@pytest.mark.parametrize("r", [4, 17, 46])
def test_punto_medio_sin_huecos(r):
    """Recorriendo el círculo por ángulo, cada píxel toca al siguiente (8-conexo)."""
    pts = sorted(set(puntos_circulo_punto_medio(r)), key=lambda p: math.atan2(p[1], p[0]))
    for (xa, ya), (xb, yb) in zip(pts, pts[1:] + pts[:1]):
        assert max(abs(xb - xa), abs(yb - ya)) == 1


def test_arco_del_medidor_deja_libre_la_parte_de_abajo():
    """El medidor va de 225° a -45° (270°): el cuarto inferior queda vacío."""
    lz = Lienzo(120, 120)
    lz.limpiar((0, 0, 0, 255))
    cx = cy = 60
    lz.arco(cx, cy, 40, (255, 255, 255), 225, -45, grosor=1)
    ys, xs = np.nonzero(lz.px[:, :, 0])
    assert len(xs) > 0
    ang = np.degrees(np.arctan2(-(ys - cy), xs - cx))      # 0° a la derecha, antihorario
    assert not ((ang < -45 - 2) & (ang > -135 + 2)).any()


def test_lienzo_recorta_lo_que_sale_del_borde():
    lz = Lienzo(50, 30)
    lz.linea(-100, -100, 200, 200, (255, 0, 0), grosor=3)
    lz.circulo(0, 0, 80, (0, 255, 0), grosor=2)
    assert lz.px.shape == (30, 50, 4)


# ---------------------------------------------------------------------------
# Panel del HUD
# ---------------------------------------------------------------------------
def test_panel_dibuja_rgba_del_tamano_del_hud():
    c = Campos.vacios()
    DatosFalsos(c).actualizar(20.0)
    panel = PanelHUD()
    px = panel.dibujar(c)
    assert px.shape == (cfg.VENTANA_ALTO, cfg.HUD_ANCHO, 4) and px.dtype == np.uint8
    assert panel.ultimo_ms > 0


def test_panel_solo_redibuja_si_cambian_los_valores():
    c = Campos.vacios()
    falsos = DatosFalsos(c)
    falsos.actualizar(20.0)
    panel = PanelHUD()
    panel.dibujar(c)
    assert not panel.necesita_redibujo(c)
    falsos.actualizar(26.0)
    assert panel.necesita_redibujo(c)
