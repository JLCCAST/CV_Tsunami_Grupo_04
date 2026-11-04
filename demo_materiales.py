"""Banco de pruebas visual del Rol 4: terreno con humedad y agua transparente.

    python demo_materiales.py
    python demo_materiales.py --t-inicio 15 --velocidad 2
    python demo_materiales.py --con-alarma

Teclas: izquierda/derecha giran la cámara, arriba/abajo acercan y alejan,
ESPACIO pausa, R reinicia, ESC sale.
"""
import argparse
import ctypes
import math
import sys

import numpy as np

from agua.datos_falsos import DatosFalsos
from campos import Campos
from modulos.iluminacion import Iluminacion
from modulos.materiales import Materiales
from modulos.terreno_gen import generar_escenario, malla_terreno


def perspectiva(fov_grados, aspecto, cerca, lejos):
    f = 1.0 / math.tan(math.radians(fov_grados) / 2.0)
    return np.array([
        [f / aspecto, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (lejos + cerca) / (cerca - lejos), 2.0 * lejos * cerca / (cerca - lejos)],
        [0.0, 0.0, -1.0, 0.0],
    ], np.float32)


def mirar_a(ojo, objetivo):
    ojo = np.asarray(ojo, np.float64)
    objetivo = np.asarray(objetivo, np.float64)
    f = objetivo - ojo
    f /= np.linalg.norm(f)
    s = np.cross(f, [0.0, 1.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ ojo, -u @ ojo, f @ ojo
    return m.astype(np.float32)


def crear_vao(gl, vertices, indices):
    """Mismo formato que Rol 2: 9 floats por vértice (pos, normal, color)."""
    vertices = np.ascontiguousarray(vertices, np.float32)
    indices = np.ascontiguousarray(indices, np.uint32)
    vao = gl.glGenVertexArrays(1)
    gl.glBindVertexArray(vao)
    vbo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
    gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
    for loc in range(3):
        gl.glEnableVertexAttribArray(loc)
        gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 36, ctypes.c_void_p(loc * 12))
    ebo = gl.glGenBuffers(1)
    gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
    gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, gl.GL_STATIC_DRAW)
    gl.glBindVertexArray(0)
    return vao


class Contexto:
    ancho_fb = 1100
    alto_fb = 700


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t-inicio", type=float, default=0.0, help="segundo de la simulación con el que empieza")
    ap.add_argument("--velocidad", type=float, default=1.0, help="multiplicador del tiempo de simulación")
    ap.add_argument("--frames", type=int, default=0, help="salir después de N cuadros (0 = sin límite)")
    ap.add_argument("--con-alarma", action="store_true",
                    help="deja activo el tinte rojo de alarma de Rol 5 (por defecto se apaga en el demo)")
    args = ap.parse_args()

    import glfw
    from OpenGL import GL as gl

    if not glfw.init():
        sys.exit("No se pudo iniciar GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    ventana = glfw.create_window(1100, 700, "Rol 4 - terreno con humedad y agua", None, None)
    if not ventana:
        glfw.terminate()
        sys.exit("No se pudo crear la ventana OpenGL 3.3")
    glfw.make_context_current(ventana)
    glfw.swap_interval(1)

    ctx = Contexto()
    ctx.ancho_fb, ctx.alto_fb = glfw.get_framebuffer_size(ventana)

    # Mismo orden que main.py: datos falsos primero, luego el terreno y los módulos.
    campos = Campos.vacios()
    falsos = DatosFalsos(campos)
    escenario = generar_escenario(simple=False)
    campos.H[:] = escenario.H

    materiales, iluminacion = Materiales(), Iluminacion()
    modulos = [materiales, iluminacion]
    for m in modulos:
        m.iniciar(campos, ctx)

    vertices, indices = malla_terreno(escenario.H)
    vao = crear_vao(gl, vertices, indices)
    n_idx = len(indices)

    estado = {"pausa": False, "reiniciar": False}

    def al_teclear(win, tecla, _scan, accion, _mods):
        if accion != glfw.PRESS:
            return
        if tecla == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif tecla == glfw.KEY_SPACE:
            estado["pausa"] = not estado["pausa"]
        elif tecla == glfw.KEY_R:
            estado["reiniciar"] = True

    glfw.set_key_callback(ventana, al_teclear)

    cam = {"az": math.pi, "dist": 700.0}      # az = pi: mirando desde el lado del mar
    centro = np.array([320.0, 0.0, 320.0])

    t_sim = args.t_inicio
    anterior = t_titulo = glfw.get_time()
    cuadros = 0
    while not glfw.window_should_close(ventana):
        ahora = glfw.get_time()
        dt, anterior = ahora - anterior, ahora

        if estado["reiniciar"]:
            t_sim, estado["reiniciar"] = args.t_inicio, False
            materiales.humedad[:] = 0.0

        dt_sim = 0.0 if estado["pausa"] else dt * args.velocidad
        t_sim += dt_sim
        if not estado["pausa"]:
            falsos.actualizar(t_sim)
        if not args.con_alarma:
            campos.alarma = 0   # solo en el demo: sin el tinte rojo de Rol 5 se ve la humedad y el agua
        for m in modulos:
            m.actualizar(campos, dt_sim)

        if glfw.get_key(ventana, glfw.KEY_LEFT) == glfw.PRESS:
            cam["az"] -= 1.2 * dt
        if glfw.get_key(ventana, glfw.KEY_RIGHT) == glfw.PRESS:
            cam["az"] += 1.2 * dt
        if glfw.get_key(ventana, glfw.KEY_UP) == glfw.PRESS:
            cam["dist"] = max(150.0, cam["dist"] - 300.0 * dt)
        if glfw.get_key(ventana, glfw.KEY_DOWN) == glfw.PRESS:
            cam["dist"] = min(2000.0, cam["dist"] + 300.0 * dt)

        ojo = centro + np.array([
            cam["dist"] * math.cos(cam["az"]),
            cam["dist"] * 0.45,
            cam["dist"] * math.sin(cam["az"]),
        ])
        aspecto = ctx.ancho_fb / ctx.alto_fb
        vp = perspectiva(45.0, aspecto, 1.0, 4000.0) @ mirar_a(ojo, centro)
        ctx.vp = vp
        ctx.cam_pos = ojo

        gl.glViewport(0, 0, ctx.ancho_fb, ctx.alto_fb)
        gl.glClearColor(0.55, 0.72, 0.90, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        gl.glEnable(gl.GL_DEPTH_TEST)

        # 1) Lo opaco primero: el terreno con humedad.
        if materiales.preparar_terreno(campos, ctx, vp, ojo):
            gl.glBindVertexArray(vao)
            gl.glDrawElements(gl.GL_TRIANGLES, n_idx, gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
        # 2) Lo transparente al final: el agua.
        materiales.dibujar(campos, ctx)

        cuadros += 1
        if ahora - t_titulo >= 0.5:
            h = materiales.consultar_humedad()
            glfw.set_window_title(
                ventana,
                f"Rol 4 | t = {t_sim:5.1f} s | humedad máx = {h['maxima']:.2f} "
                f"| promedio = {h['promedio']:.3f}" + (" | PAUSA" if estado["pausa"] else ""),
            )
            t_titulo = ahora

        if args.frames and cuadros >= args.frames:
            break
        glfw.swap_buffers(ventana)
        glfw.poll_events()

    for m in modulos:
        m.cerrar()
    glfw.terminate()


if __name__ == "__main__":
    main()