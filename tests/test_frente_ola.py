"""Pruebas del frente de ola (Rol 1, semana 2). Usan el terreno real del rol 2, sin ventana."""
import numpy as np
import pytest

import config as cfg
from agua import frente_ola as fo
from agua.frente_ola import FrenteOla
from campos import Campos
from modulos.terreno import Terreno

PASO = 0.1
T_FIN = fo.T_RETIRO_S + fo.DURACION_RETIRO_S


@pytest.fixture(scope="module")
def terreno_real():
    c = Campos.vacios()
    Terreno().iniciar(c, None)          # sin contexto OpenGL: solo genera H, casas y obstáculos
    return c


def _campos(terreno_real):
    c = Campos.vacios()
    c.H[:] = terreno_real.H
    c.casas, c.dano = terreno_real.casas.copy(), terreno_real.dano.copy()
    c.obstaculo[:] = terreno_real.obstaculo
    return c


def _correr(c, hasta, paso=PASO, ola=None):
    ola = ola or FrenteOla(c)
    for t in np.arange(0.0, hasta + 1e-9, paso):
        ola.actualizar(float(t))
    return ola


def test_respeta_el_contrato_en_toda_la_simulacion(terreno_real):
    c = _campos(terreno_real)
    ola = FrenteOla(c)
    for t in np.arange(0.0, T_FIN + 5.0, 0.5):
        ola.actualizar(float(t))
        c.validar()
        assert c.t == pytest.approx(t)


def test_mar_en_calma_antes_de_la_llegada(terreno_real):
    c = _campos(terreno_real)
    FrenteOla(c).actualizar(fo.T_LLEGADA_S - 0.5)
    assert np.allclose(c.W, np.maximum(0.0, cfg.NIVEL_MAR_M - c.H))
    assert not c.U.any() and not c.V.any()
    assert c.alarma == 0.0


def test_el_frente_avanza_y_nunca_retrocede(terreno_real):
    c = _campos(terreno_real)
    ola = FrenteOla(c)
    previo = ola.xf.copy()
    for t in np.arange(0.0, T_FIN, 0.5):
        ola.actualizar(float(t))
        assert (ola.xf >= previo - 1e-6).all()
        previo = ola.xf.copy()
    assert ola.xf.mean() > ola.xf0.mean() + 100.0     # recorrió más de 100 m


def test_el_frente_se_detiene_donde_el_terreno_supera_el_muro(terreno_real):
    c = _campos(terreno_real)
    ola = _correr(c, fo.T_RETIRO_S)
    detenidas = ~ola.avanzando
    assert detenidas.any()
    ix = np.clip(ola.frente_ix() + 1, 0, cfg.GRID - 1)
    delante = c.H[ola.filas, ix]
    # en cada fila detenida, la celda siguiente al frente es más alta que el muro
    assert (delante[detenidas] >= cfg.NIVEL_MAR_M + fo.ALTURA_FRENTE_M - 1e-3).all()


def test_frente_x_vacio_antes_de_la_llegada_y_en_el_retiro(terreno_real):
    c = _campos(terreno_real)
    ola = FrenteOla(c)
    ola.actualizar(fo.T_LLEGADA_S - 0.5)
    assert (c.frente_x == cfg.SIN_FRENTE).all()
    ola.actualizar(fo.T_RETIRO_S + 1.0)
    assert (c.frente_x == cfg.SIN_FRENTE).all()


def test_frente_x_sigue_al_frente_mientras_entra(terreno_real):
    c = _campos(terreno_real)
    ola = FrenteOla(c)
    previo = c.frente_x.copy()
    for t in np.arange(fo.T_LLEGADA_S, fo.T_RETIRO_S, 0.5):
        ola.actualizar(float(t))
        dentro = ola.xf >= 0.0
        assert np.array_equal(c.frente_x[dentro], ola.xf[dentro])
        assert (c.frente_x[~dentro] == cfg.SIN_FRENTE).all()
        publicado = previo != cfg.SIN_FRENTE
        assert (c.frente_x[publicado] >= previo[publicado] - 1e-6).all()     # nunca retrocede
        previo = c.frente_x.copy()
    assert (c.frente_x != cfg.SIN_FRENTE).all()       # antes del retiro el frente ya entró en todas las filas


def test_tierra_seca_delante_del_frente(terreno_real):
    c = _campos(terreno_real)
    ola = _correr(c, 28.0)
    x = np.arange(cfg.GRID) * cfg.CELL_M
    delante = x[None, :] > ola.xf[:, None]
    tierra = c.H >= cfg.NIVEL_MAR_M
    assert (c.W[delante & tierra] == 0.0).all()
    assert (c.U[delante] == 0.0).all() and (c.V[delante] == 0.0).all()


def test_la_ola_inunda_y_sube_la_alarma(terreno_real):
    c = _campos(terreno_real)
    ola = FrenteOla(c)
    maxima = 0.0
    for t in np.arange(0.0, fo.T_RETIRO_S, PASO):
        ola.actualizar(float(t))
        maxima = max(maxima, c.alarma)
    assert maxima > 0.5
    mojado_tierra = (c.H >= cfg.NIVEL_MAR_M) & (c.W > cfg.UMBRAL_MOJADO_M)
    assert mojado_tierra.sum() > 500


def test_el_flujo_entra_y_luego_sale(terreno_real):
    c = _campos(terreno_real)
    ola = _correr(c, 30.0)
    mojado = c.W > cfg.UMBRAL_MOJADO_M
    assert c.U[mojado].max() > 1.0 and c.U.min() >= 0.0       # entrada: hacia la tierra (+x)
    _correr(c, fo.T_RETIRO_S + 5.0, ola=ola)
    assert c.U.min() < -1.0 and c.U.max() <= 0.0               # retiro: hacia el mar (-x)


def test_la_velocidad_es_finita_y_acotada(terreno_real):
    c = _campos(terreno_real)
    ola = FrenteOla(c)
    for t in np.arange(0.0, T_FIN, 0.5):
        ola.actualizar(float(t))
        rapidez = np.hypot(c.U, c.V)
        assert rapidez.max() <= fo.VEL_MAX_MS + 1e-4
        assert np.abs(c.V).max() <= np.abs(c.U).max() + 1e-4   # el flujo va sobre todo en x


def test_al_final_todo_vuelve_a_la_calma(terreno_real):
    c = _campos(terreno_real)
    _correr(c, T_FIN + 2.0)
    assert np.allclose(c.W, np.maximum(0.0, cfg.NIVEL_MAR_M - c.H))
    assert not c.U.any() and not c.V.any()
    assert c.alarma == 0.0


def test_reiniciar_al_retroceder_el_tiempo(terreno_real):
    a = _campos(terreno_real)
    ola = _correr(a, 35.0)
    ola.actualizar(12.0)                  # como si se apretara R y se volviera a t = 12
    W_reinicio = a.W.copy()

    b = _campos(terreno_real)
    FrenteOla(b).actualizar(12.0)         # simulación nueva directamente hasta t = 12
    assert np.allclose(W_reinicio, b.W)


def test_cada_paso_queda_medido(terreno_real):
    c = _campos(terreno_real)
    ola = _correr(c, 10.0, paso=0.5)                  # 21 pasos: t = 0, 0.5, ..., 10
    crono = ola.cronometro
    assert crono.pasos == 21 and len(crono.muestras) == 21
    assert 0.0 < crono.media_ms <= crono.maximo_ms
    assert crono.ema_ms > 0.0
    ola.actualizar(2.0)                               # reiniciar con R no borra la medición
    assert crono.pasos == 22


def test_cronometro_estadisticas():
    from agua.medicion import Cronometro
    c = Cronometro()
    for ms in (1.0, 2.0, 3.0, 10.0):
        c.registrar(ms)
    assert c.media_ms == pytest.approx(4.0)
    assert c.maximo_ms == 10.0 and c.ultimo_ms == 10.0
    assert c.percentil_ms(50) == pytest.approx(2.5)
    assert 1.0 < c.ema_ms < 10.0
    assert "4 pasos" in c.resumen()


def test_no_depende_del_tamano_del_paso(terreno_real):
    a, b = _campos(terreno_real), _campos(terreno_real)
    ola_a = _correr(a, 32.0, paso=1.0 / 60)        # 60 fps
    ola_b = _correr(b, 32.0, paso=0.5)             # 2 fps
    assert np.abs(ola_a.xf - ola_b.xf).max() <= cfg.CELL_M
