import numpy as np

import config as cfg
from campos import Campos
from modulos.materiales import Materiales


def crear_campos_prueba():
    campos = Campos.vacios()

    # Simular terreno por encima del nivel del mar.
    campos.H[:] = 2.0

    return campos


def test_humedad_inicia_en_cero():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    assert materiales.humedad.shape == campos.H.shape
    assert np.all(materiales.humedad == 0.0)


def test_humedad_aumenta_con_el_agua():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    campos.W[10, 10] = cfg.UMBRAL_MOJADO_M + 1.0

    materiales.actualizar(campos, 1.0)

    assert materiales.humedad[10, 10] > 0.0


def test_humedad_disminuye_al_secarse():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    materiales.humedad[10, 10] = 1.0
    campos.W[:] = 0.0

    materiales.actualizar(campos, 1.0)

    assert materiales.humedad[10, 10] < 1.0


def test_humedad_se_mantiene_entre_cero_y_uno():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    campos.W[:] = 10.0

    materiales.actualizar(campos, 10.0)

    assert np.all(materiales.humedad >= 0.0)
    assert np.all(materiales.humedad <= 1.0)


def test_datos_textura_humedad_tiene_formato_gpu():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    materiales.humedad[10, 10] = 1.0

    datos = materiales.datos_textura_humedad()

    assert datos.dtype == np.uint8
    assert datos.shape == (*campos.H.shape, 1)
    assert datos[10, 10, 0] == 255
    assert datos[0, 0, 0] == 0


def test_shader_terreno_reutiliza_funciones_de_rol5():
    from modulos.iluminacion import armar_fragment
    from modulos.materiales import FRAG_TERRENO

    fuente = armar_fragment(FRAG_TERRENO)

    assert fuente.startswith("#version 330 core")
    assert "phong_mojado(" in FRAG_TERRENO
    # La definición de phong_mojado viene solo del archivo de Rol 5.
    assert "vec3 phong_mojado(" not in FRAG_TERRENO
    assert fuente.count("vec3 phong_mojado(") == 1


def test_humedad_se_reinicia_cuando_el_reloj_vuelve_atras():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    campos.t = 30.0
    materiales.humedad[10, 10] = 1.0
    materiales.actualizar(campos, 0.0)
    assert materiales.humedad[10, 10] > 0.0

    campos.t = 0.0
    materiales.actualizar(campos, 0.0)
    assert np.all(materiales.humedad == 0.0)


def test_shader_agua_reutiliza_funciones_de_rol5():
    from modulos.iluminacion import armar_fragment
    from modulos.materiales import FRAG_AGUA

    fuente = armar_fragment(FRAG_AGUA)

    assert fuente.startswith("#version 330 core")
    assert "rampa_profundidad(" in FRAG_AGUA
    assert "phong_mojado(" in FRAG_AGUA
    # Las definiciones vienen solo del archivo de Rol 5.
    assert "vec3 rampa_profundidad(" not in FRAG_AGUA
    assert "vec3 phong_mojado(" not in FRAG_AGUA
    assert fuente.count("vec3 rampa_profundidad(") == 1
    assert fuente.count("vec3 phong_mojado(") == 1


def test_opacidad_del_agua_es_valida():
    from modulos import materiales as mod

    assert 0.0 < mod.ALFA_MIN < mod.ALFA_MAX <= 1.0
    assert mod.PROF_MAX_AGUA > 0.0


def test_dibujar_sin_gpu_no_hace_nada():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    class CtxVacio:
        pass

   
    materiales.dibujar(campos, CtxVacio())

def test_casa_junto_al_agua_se_moja():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    # Una casa ocupa 2x2 celdas y el agua toca solo una celda vecina, por fuera.
    campos.obstaculo[20:22, 20:22] = True
    campos.W[20, 19] = cfg.UMBRAL_MOJADO_M + 1.0

    materiales.actualizar(campos, 1.0)

    assert materiales.humedad[20, 20] > 0.0     # celda de la casa pegada al agua
    assert materiales.humedad[21, 21] == 0.0    # celda de la casa sin agua cerca


def test_fijar_origen_sin_gpu_no_hace_nada():
    campos = crear_campos_prueba()
    materiales = Materiales()

    materiales.iniciar(campos, None)

    materiales.fijar_origen(1.0, 2.0, 3.0)

def test_textura_detalle_tiene_formato_gpu_y_es_determinista():
    from modulos.materiales import generar_textura_detalle

    a = generar_textura_detalle()
    b = generar_textura_detalle()

    assert a.dtype == np.uint8
    assert a.shape == (128, 128, 1)
    assert np.array_equal(a, b)


def test_textura_detalle_tiene_variacion():
    from modulos.materiales import generar_textura_detalle

    datos = generar_textura_detalle()

    assert datos.std() > 10.0       # no es una textura plana
    assert datos.min() < 60 and datos.max() > 195


def test_shader_terreno_usa_la_textura_de_detalle():
    from modulos.materiales import FRAG_TERRENO

    assert "texture(uDetalle" in FRAG_TERRENO

def test_shader_agua_tiene_espuma_en_el_borde():
    from modulos.materiales import FRAG_AGUA

    assert "uAnchoEspuma" in FRAG_AGUA
    assert "texture(uDetalle" in FRAG_AGUA