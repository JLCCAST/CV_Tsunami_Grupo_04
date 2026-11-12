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


def _campos_con_costa(ix_costa=40):
    campos = Campos.vacios()
    campos.H[:, :ix_costa] = -5.0                  # mar
    campos.H[:, ix_costa:] = 1.0                   # tierra
    campos.W[:, :ix_costa] = 5.0                   # mar en calma
    return campos


def test_auto_y_piezas_empiezan_en_tierra():
    campos = _campos_con_costa(40)
    mod = Arrastre(num_piezas=30)
    mod.iniciar(campos, None)
    costa = 40 * cfg.CELL_M
    assert mod.pos_auto[0] >= costa
    assert np.all(mod.piezas_pos[:, 0] >= costa)
    assert np.all(mod.piezas_pos[:, 0] <= costa + 25.0)        # en la franja de playa


def test_flota_sobre_la_superficie():
    campos = Campos.vacios()
    mod = Arrastre(num_piezas=20)
    mod.iniciar(campos, None)
    campos.W.fill(2.0)                             # 2 m de agua quieta
    mod.actualizar(campos, dt=0.1)
    alto = mod.tamano_auto[1]
    assert mod.pos_auto[1] + alto / 2 > 2.0        # el techo del auto asoma sobre el agua
    assert mod.pos_auto[1] < 2.0                   # y una parte queda hundida
    tapas = mod.piezas_pos[:, 1] + mod.piezas_escala[:, 1] / 2
    assert np.all(tapas > 2.0)


def test_casa_destruida_suelta_fragmentos():
    from modulos import arrastre as ar
    campos = Campos.vacios()
    campos.casas = np.array([[60, 60, 63, 63]], np.int32)
    campos.dano = np.zeros(1, np.uint8)
    mod = Arrastre(num_piezas=20)
    mod.iniciar(campos, None)
    mod.actualizar(campos, dt=0.1)
    assert len(mod.frag_pos) == 0                  # sin daño no hay fragmentos
    campos.dano[0] = cfg.DANO_DESTRUIDA
    mod.actualizar(campos, dt=0.1)
    assert len(mod.frag_pos) == ar.FRAG_POR_CASA
    x, z = mod.frag_pos[:, 0] / cfg.CELL_M, mod.frag_pos[:, 2] / cfg.CELL_M
    assert np.all((x >= 60) & (x <= 63) & (z >= 60) & (z <= 63))
    mod.actualizar(campos, dt=0.1)
    assert len(mod.frag_pos) == ar.FRAG_POR_CASA   # no se repiten


def test_al_retroceder_el_reloj_todo_vuelve_al_inicio():
    campos = Campos.vacios()
    campos.casas = np.array([[60, 60, 63, 63]], np.int32)
    campos.dano = np.zeros(1, np.uint8)
    mod = Arrastre(num_piezas=20)
    mod.iniciar(campos, None)
    inicio_auto, inicio_piezas = mod.pos_auto.copy(), mod.piezas_pos.copy()
    campos.t = 20.0
    campos.W.fill(2.0)
    campos.U.fill(5.0)
    campos.dano[0] = cfg.DANO_DESTRUIDA
    mod.actualizar(campos, dt=0.5)
    assert len(mod.frag_pos) > 0 and mod.pos_auto[0] > inicio_auto[0]
    campos.t = 0.0                                 # tecla R
    campos.W.fill(0.0)
    campos.dano[0] = cfg.DANO_INTACTA
    mod.actualizar(campos, dt=0.0)
    assert np.allclose(mod.pos_auto, inicio_auto)
    assert np.allclose(mod.piezas_pos, inicio_piezas)
    assert len(mod.frag_pos) == 0


def test_cubo_con_normales_hacia_afuera():
    from modulos.arrastre import cubo_con_normales
    v = cubo_con_normales()
    assert v.shape == (36, 6)
    assert np.allclose(np.linalg.norm(v[:, 3:], axis=1), 1.0)
    assert np.all(np.einsum("ij,ij->i", v[:, :3], v[:, 3:]) > 0)      # cada cara mira hacia afuera