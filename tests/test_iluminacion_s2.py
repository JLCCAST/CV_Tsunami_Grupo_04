"""Pruebas del Rol 5, semana 2: especular mojado, tinte de alarma y rampa de color (sin GPU)."""
import numpy as np
import pytest

from modulos.iluminacion import leer_glsl
from modulos.iluminacion_cpu import phong, phong_mojado, rampa_color, tinte_alarma
from vista_mapa import rampa_riesgo

# Escena de prueba: suelo horizontal y sol a 45 grados.
N = np.array([0.0, 1.0, 0.0], np.float32)
L = np.array([1.0, 1.0, 0.0], np.float32) / np.sqrt(2.0)
R = -L + 2.0 * float(N @ L) * N                      # dirección del reflejo perfecto
BASE = np.array([0.50, 0.40, 0.30], np.float32)
LUZ = np.array([1.0, 0.96, 0.88], np.float32)
KA, KD, KS, BRILLO = 0.18, 0.90, 0.35, 24.0


def luminancia(c):
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


# ---------------------------------------------------------------- rampa de color
@pytest.mark.parametrize("t", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_rampa_color_igual_a_la_del_hud(t):
    esperado = np.asarray(rampa_riesgo(t), np.float32) / 255.0
    assert np.allclose(rampa_color(t), esperado, atol=1e-5)


def test_rampa_color_fuera_de_rango_se_recorta():
    assert np.allclose(rampa_color(-3.0), rampa_color(0.0))
    assert np.allclose(rampa_color(9.0), rampa_color(1.0))


# ---------------------------------------------------------------- tinte de alarma
@pytest.mark.parametrize("alarma", [0.0, 0.3, 0.6])
def test_tinte_no_cambia_el_color_con_alarma_baja(alarma):
    c = np.array([0.7, 0.6, 0.4], np.float32)
    assert np.allclose(tinte_alarma(c, alarma), c, atol=1e-6)


def test_tinte_maximo_oscurece_y_vira_a_rojizo():
    c = np.array([0.7, 0.6, 0.4], np.float32)
    t = tinte_alarma(c, 1.0)
    assert luminancia(t) < luminancia(c)
    assert t[0] / t[1] > c[0] / c[1]                 # más rojo relativo al verde


def test_tinte_oscurece_cada_vez_mas_al_subir_la_alarma():
    c = np.array([0.7, 0.6, 0.4], np.float32)
    lums = [luminancia(tinte_alarma(c, a)) for a in np.linspace(0.0, 1.0, 11)]
    assert all(b <= a + 1e-6 for a, b in zip(lums, lums[1:]))


# ---------------------------------------------------------------- Phong y mojado
def test_phong_solo_ambiente_si_la_visibilidad_es_cero():
    c = phong(N, R, L, BASE, LUZ, KA, KD, KS, BRILLO, visibilidad=0.0)
    assert np.allclose(c, KA * BASE, atol=1e-6)


def test_phong_solo_ambiente_de_noche():
    c = phong(N, R, L, BASE, np.zeros(3), KA, KD, KS, BRILLO)
    assert np.allclose(c, KA * BASE, atol=1e-6)


def test_phong_sin_especular_si_la_luz_viene_por_debajo():
    c = phong(N, R, -L, BASE, LUZ, KA, KD, KS, BRILLO)
    assert np.allclose(c, KA * BASE, atol=1e-6)


def test_humedad_cero_equivale_a_phong_seco():
    seco = phong(N, R, L, BASE, LUZ, KA, KD, KS, BRILLO)
    mojado0 = phong_mojado(N, R, L, BASE, LUZ, KA, KD, KS, BRILLO, humedad=0.0)
    assert np.allclose(seco, mojado0, atol=1e-6)


def test_mojado_brilla_mas_en_el_reflejo():
    """V alineado con el reflejo perfecto: el especular fuerte hace que lo mojado brille más."""
    seco = phong(N, R, L, BASE, LUZ, KA, KD, KS, BRILLO)
    mojado = phong_mojado(N, R, L, BASE, LUZ, KA, KD, KS, BRILLO, humedad=1.0)
    assert luminancia(mojado) > luminancia(seco)


def test_mojado_se_ve_mas_oscuro_fuera_del_reflejo():
    """Fuera del reflejo manda el color base, que se oscurece al mojarse."""
    V = np.array([0.0, 1.0, 0.0], np.float32)
    seco = phong(N, V, L, BASE, LUZ, KA, KD, KS, BRILLO)
    mojado = phong_mojado(N, V, L, BASE, LUZ, KA, KD, KS, BRILLO, humedad=1.0)
    assert luminancia(mojado) < luminancia(seco)


def test_el_reflejo_mojado_es_mas_estrecho_que_el_seco():
    """Brillo 128 frente a 24: al apartarse un poco del reflejo, lo mojado cae más rápido."""
    a = 0.25                                         # giro de V respecto del reflejo, en radianes
    V = np.array([R[0] * np.cos(a) - R[1] * np.sin(a),
                  R[0] * np.sin(a) + R[1] * np.cos(a), 0.0], np.float32)
    pico_seco = luminancia(phong(N, R, L, BASE, LUZ, 0.0, 0.0, KS, BRILLO))
    pico_moj = luminancia(phong(N, R, L, BASE, LUZ, 0.0, 0.0, 0.90, 128.0))
    lado_seco = luminancia(phong(N, V, L, BASE, LUZ, 0.0, 0.0, KS, BRILLO))
    lado_moj = luminancia(phong(N, V, L, BASE, LUZ, 0.0, 0.0, 0.90, 128.0))
    assert lado_moj / pico_moj < lado_seco / pico_seco


# ------------------------------------------- el espejo en Python sigue al GLSL
@pytest.mark.parametrize("fragmento", [
    "smoothstep(0.6, 1.0, alarma)",
    "vec3(1.25, 0.82, 0.78)",
    "color * 0.80",
    "mix(1.0, 0.55, humedad)",
    "mix(ks, 0.90, humedad)",
    "mix(brillo, 128.0, humedad)",
])
def test_constantes_del_espejo_siguen_en_el_glsl(fragmento):
    assert fragmento in leer_glsl(), f"cambió el GLSL: actualiza modulos/iluminacion_cpu.py ({fragmento})"
