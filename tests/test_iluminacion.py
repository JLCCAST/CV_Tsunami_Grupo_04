"""Pruebas del Rol 5 (no necesitan ventana ni GPU)."""
import re
from types import SimpleNamespace

import numpy as np
import pytest

from campos import Campos
from modulos.iluminacion import (RUTA_GLSL, Iluminacion, armar_fragment, direccion_sol,
                                 intensidad_sol, leer_glsl)
from vista_mapa import rampa_riesgo

FIRMAS = [
    r"vec3\s+direccion_sol\s*\(\s*float\s+\w+\s*\)",
    r"float\s+intensidad_sol\s*\(\s*vec3\s+\w+\s*\)",
    r"vec3\s+phong\s*\(",
    r"vec3\s+phong_mojado\s*\(",
    r"float\s+sombra\s*\(\s*sampler2D\s+\w+\s*,\s*vec4\s+\w+\s*,\s*float\s+\w+\s*\)",
    r"vec3\s+tinte_alarma\s*\(\s*vec3\s+\w+\s*,\s*float\s+\w+\s*\)",
    r"vec3\s+rampa_color\s*\(\s*float\s+\w+\s*\)",
    r"vec3\s+rampa_profundidad\s*\(\s*float\s+\w+\s*,\s*float\s+\w+\s*\)",
    r"vec3\s+rgb2hsv\s*\(",
    r"vec3\s+hsv2rgb\s*\(",
]


def test_glsl_existe_y_tiene_las_firmas_acordadas():
    assert RUTA_GLSL.exists()
    texto = leer_glsl()
    for patron in FIRMAS:
        assert re.search(patron, texto), f"falta la firma: {patron}"


def test_glsl_llaves_y_parentesis_balanceados():
    t = leer_glsl()
    assert t.count("{") == t.count("}")
    assert t.count("(") == t.count(")")


def test_el_glsl_no_trae_version_ni_main():
    t = leer_glsl()
    assert "#version" not in t, "el #version lo pone armar_fragment()"
    assert not re.search(r"\bvoid\s+main\s*\(", t), "es una biblioteca: sin main()"


def test_armar_fragment_pone_version_primero():
    fs = armar_fragment("out vec4 c; void main(){ c = vec4(1.0); }")
    assert fs.startswith("#version 330 core\n")
    assert fs.count("#version") == 1


def test_armar_fragment_rechaza_cuerpo_con_version():
    with pytest.raises(AssertionError):
        armar_fragment("#version 330 core\nvoid main(){}")


def test_sol_unitario_y_ciclo_del_dia():
    for h in (0, 3, 6, 9, 12, 15, 18, 21):
        assert abs(np.linalg.norm(direccion_sol(h)) - 1.0) < 1e-5
    assert direccion_sol(12)[1] > 0.9                    # mediodía: casi vertical
    assert abs(direccion_sol(6)[1]) < 1e-5               # amanecer: en el horizonte
    assert direccion_sol(0)[1] < 0.0                     # medianoche: bajo el horizonte
    assert intensidad_sol(direccion_sol(0)) == 0.0
    assert intensidad_sol(direccion_sol(12)) == 1.0


def test_el_sol_sale_por_la_tierra_y_se_pone_por_el_mar():
    # El mar está en x pequeño y la tierra en x grande (CONTRATO.md).
    assert direccion_sol(7)[0] > 0 > direccion_sol(17)[0]


def test_modulo_publica_el_sol_en_el_contexto():
    ctx = SimpleNamespace()
    campos = Campos.vacios()
    m = Iluminacion(hora=11.0, ciclo_dia_s=0.0)
    m.iniciar(campos, ctx)
    assert ctx.sol_dir.shape == (3,) and abs(np.linalg.norm(ctx.sol_dir) - 1) < 1e-5
    assert ctx.hora == 11.0 and "phong" in ctx.glsl_iluminacion
    m.actualizar(campos, 1.0)
    assert ctx.hora == 11.0                              # ciclo 0 = sol fijo


def test_modulo_avanza_el_ciclo_del_dia():
    ctx = SimpleNamespace()
    campos = Campos.vacios()
    m = Iluminacion(hora=6.0, ciclo_dia_s=240.0)         # 240 s reales = 24 h
    m.iniciar(campos, ctx)
    m.actualizar(campos, 60.0)                           # un cuarto de día = 6 h
    assert abs(ctx.hora - 12.0) < 1e-6
    m.actualizar(campos, 180.0)
    assert abs(ctx.hora - 6.0) < 1e-6                    # da la vuelta


def test_modulo_funciona_sin_contexto():
    m = Iluminacion()
    m.iniciar(Campos.vacios(), None)
    m.actualizar(Campos.vacios(), 0.1)


def test_uniforms_incluyen_la_alarma_del_contrato():
    campos = Campos.vacios()
    campos.alarma = 0.7
    u = Iluminacion().uniforms(campos)
    assert u["uAlarma"] == pytest.approx(0.7)
    assert set(u) == {"uSolDir", "uSolColor", "uHora", "uAlarma"}


def test_rampa_color_glsl_coincide_con_la_del_hud():
    """Los tres colores de la rampa en GLSL deben ser los de vista_mapa.rampa_riesgo (Rol 1)."""
    t = leer_glsl()
    patron = r"COLOR_(VERDE|AMARILLO|ROJO)\s*=\s*vec3\(\s*([\d.]+),\s*([\d.]+),\s*([\d.]+)\)\s*/\s*255"
    encontrados = {n: tuple(float(v) for v in (r, g, b)) for n, r, g, b in re.findall(patron, t)}
    assert set(encontrados) == {"VERDE", "AMARILLO", "ROJO"}
    for nombre, x in (("VERDE", 0.0), ("AMARILLO", 0.5), ("ROJO", 1.0)):
        esperado = tuple(float(v) for v in rampa_riesgo(x))
        assert np.allclose(encontrados[nombre], esperado, atol=1.0)
