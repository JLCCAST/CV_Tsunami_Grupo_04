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