"""Pruebas unitarias para el Rol 3: PilaMatrices y Arrastre v1."""
import numpy as np

import config as cfg
from campos import Campos
from modulos.arrastre import Arrastre, PilaMatrices


def test_pila_matrices_push_pop():
    pila = PilaMatrices()
    identidad = np.eye(4, dtype=np.float32)
    assert np.allclose(pila.top(), identidad)

    pila.push()
    pila.translate(10.0, 5.0, -3.0)
    pila.rotate_y(np.pi / 4.0)
    pila.scale(2.0, 2.0, 2.0)
    assert not np.allclose(pila.top(), identidad)

    pila.pop()
    assert np.allclose(pila.top(), identidad)


def test_arrastre_respeta_umbral_profundidad():
    campos = Campos.vacios()
    mod = Arrastre(num_piezas=30)
    mod.iniciar(campos, None)

    pos_auto_inicial = mod.pos_auto.copy()
    pos_piezas_inicial = mod.piezas_pos.copy()

    # Agua con velocidad alta, pero profundidad W por debajo del umbral
    campos.W.fill(cfg.UMBRAL_ARRASTRE_M * 0.5)
    campos.U.fill(8.0)
    campos.V.fill(2.0)

    mod.actualizar(campos, dt=0.5)

    # No debe desplazarse en X ni en Z
    assert np.allclose(mod.pos_auto[[0, 2]], pos_auto_inicial[[0, 2]])
    assert np.allclose(mod.piezas_pos[:, [0, 2]], pos_piezas_inicial[:, [0, 2]])


def test_arrastre_desplaza_auto_y_piezas_sobre_umbral():
    campos = Campos.vacios()
    mod = Arrastre(num_piezas=35)
    mod.iniciar(campos, None)

    pos_auto_inicial = mod.pos_auto.copy()
    pos_piezas_inicial = mod.piezas_pos.copy()

    # Agua supera el umbral con flujo positivo hacia tierra (+U)
    campos.W.fill(cfg.UMBRAL_ARRASTRE_M + 1.2)
    campos.U.fill(6.0)
    campos.V.fill(1.5)

    mod.actualizar(campos, dt=0.5)

    assert mod.pos_auto[0] > pos_auto_inicial[0]
    assert mod.pos_auto[2] > pos_auto_inicial[2]
    assert np.all(mod.piezas_pos[:, 0] > pos_piezas_inicial[:, 0])