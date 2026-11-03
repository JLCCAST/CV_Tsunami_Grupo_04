import math
import os
import sys
import time
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
from agua.datos_falsos import DatosFalsos
from campos import Campos
from modulos.terreno import Terreno


def perspectiva(fovy, aspecto, cerca, lejos):
    t = 1.0 / math.tan(fovy / 2)
    return np.array(
        [
            [t / aspecto, 0, 0, 0],
            [0, t, 0, 0],
            [0, 0, (lejos + cerca) / (cerca - lejos), 2 * lejos * cerca / (cerca - lejos)],
            [0, 0, -1, 0],
        ],
        np.float32,
    )


def mirar(ojo, objetivo, arriba=(0, 1, 0)):
    z = np.asarray(ojo, np.float64) - np.asarray(objetivo, np.float64)
    z /= np.linalg.norm(z)
    x = np.cross(arriba, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    M = np.eye(4)
    M[0, :3], M[1, :3], M[2, :3] = x, y, z
    M[:3, 3] = -M[:3, :3] @ np.asarray(ojo, np.float64)
    return M.astype(np.float32)


def main():
    import glfw
    from OpenGL import GL as gl

    simple = "--simple" in sys.argv
    if not glfw.init():
        sys.exit("No se pudo iniciar GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    ventana = glfw.create_window(1200, 760, "Visor de terreno (rol 2)", None, None)
    if not ventana:
        glfw.terminate()
        sys.exit("No se pudo crear la ventana OpenGL 3.3")
    glfw.make_context_current(ventana)
    glfw.swap_interval(1)

    campos = Campos.vacios()
    falsos = DatosFalsos(campos)
    terreno = Terreno(modo_simple=simple)
    ctx = SimpleNamespace(ancho_fb=1200, alto_fb=760, vp=None)
    inicio = time.perf_counter()
    terreno.iniciar(campos, ctx)
    print(f"iniciar: {time.perf_counter() - inicio:.2f} s, {len(campos.casas)} casas")

    est = SimpleNamespace(yaw=0.9, pitch=0.55, dist=520.0, arrastre=False, x=0.0, y=0.0,
                          pausa=False, sel=0, frag=False, sep=1.0, t=0.0)
    centro = np.array([cfg.WORLD_M / 2, 5.0, cfg.WORLD_M / 2], np.float32)

    def al_mover(_w, x, y):
        if est.arrastre:
            est.yaw -= (x - est.x) * 0.006
            est.pitch = float(np.clip(est.pitch + (y - est.y) * 0.006, 0.05, 1.5))
        est.x, est.y = x, y

    def al_boton(_w, boton, accion, _m):
        if boton == glfw.MOUSE_BUTTON_LEFT:
            est.arrastre = accion == glfw.PRESS

    def al_rueda(_w, _dx, dy):
        est.dist = float(np.clip(est.dist * (0.9 ** dy), 30.0, 1500.0))

    def al_tecla(_w, tecla, _sc, accion, _m):
        if accion != glfw.PRESS:
            return
        n = len(campos.casas)
        if tecla == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(ventana, True)
        elif tecla == glfw.KEY_SPACE:
            est.pausa = not est.pausa
        elif tecla == glfw.KEY_R:
            est.t = 0.0
        elif tecla in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3) and n:
            campos.dano[est.sel] = tecla - glfw.KEY_1
        elif tecla == glfw.KEY_RIGHT_BRACKET and n:
            est.sel = (est.sel + 1) % n
        elif tecla == glfw.KEY_LEFT_BRACKET and n:
            est.sel = (est.sel - 1) % n
        elif tecla == glfw.KEY_F:
            est.frag = not est.frag

    glfw.set_cursor_pos_callback(ventana, al_mover)
    glfw.set_mouse_button_callback(ventana, al_boton)
    glfw.set_scroll_callback(ventana, al_rueda)
    glfw.set_key_callback(ventana, al_tecla)

    previo = time.perf_counter()
    while not glfw.window_should_close(ventana):
        glfw.poll_events()
        ahora = time.perf_counter()
        dt, previo = ahora - previo, ahora
        if not est.pausa:
            est.t += dt
            falsos.actualizar(est.t)
        terreno.actualizar(campos, dt)
        if est.frag and glfw.get_key(ventana, glfw.KEY_F) == glfw.PRESS:
            est.sep = min(est.sep + dt * 1.5, 3.0)
        else:
            est.sep = max(est.sep - dt * 3.0, 1.0)

        ancho, alto = glfw.get_framebuffer_size(ventana)
        ctx.ancho_fb, ctx.alto_fb = ancho, alto
        gl.glViewport(0, 0, ancho, max(alto, 1))
        gl.glClearColor(0.55, 0.72, 0.90, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        ojo = centro + est.dist * np.array(
            [math.cos(est.pitch) * math.sin(est.yaw), math.sin(est.pitch), math.cos(est.pitch) * math.cos(est.yaw)],
            np.float32,
        )
        ctx.vp = perspectiva(math.radians(50), ancho / max(alto, 1), 5.0, 4000.0) @ mirar(ojo, centro)
        terreno.dibujar(campos, ctx)
        if est.frag and len(campos.casas):
            terreno.dibujar_fragmentos(ctx, est.sel, est.sep)

        cuenta = np.bincount(campos.dano, minlength=3)
        glfw.set_window_title(
            ventana,
            f"Terreno | intactas {cuenta[0]} · dañadas {cuenta[1]} · destruidas {cuenta[2]}"
            f" | casa {est.sel} | t = {est.t:5.1f} s" + (" | PAUSA" if est.pausa else ""),
        )
        glfw.swap_buffers(ventana)

    terreno.cerrar()
    glfw.terminate()


if __name__ == "__main__":
    main()
