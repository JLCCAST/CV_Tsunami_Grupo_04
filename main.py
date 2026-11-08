"""Punto de entrada: ventana GLFW + OpenGL 3.3 core, bucle principal y módulos.

    python main.py                  abre la ventana (ESC sale, ESPACIO pausa, R reinicia)
    python main.py --3d             vista 3D del terreno, agua y casas (flechas mueven la cámara)
    python main.py --frames 200 --captura salida.png    corre 200 cuadros, guarda una captura y sale

Por defecto se ve la vista de depuración de los campos (terreno, agua y casas) y el HUD
a la derecha, todo con datos falsos.
"""
import argparse
import sys
import time
import numpy as np

import config as cfg
from agua.datos_falsos import DatosFalsos
from campos import Campos
from hud.panel import PanelHUD
from modulos.arrastre import Arrastre
from modulos.base import Modulo
from modulos.iluminacion import Iluminacion
from modulos.materiales import Materiales
from modulos.terreno import Terreno
from vista_mapa import imagen_mapa


class ModuloHUD(Modulo):
    """Panel lateral HUD."""
    nombre = "hud"

    def __init__(self):
        self.panel = PanelHUD(cfg.HUD_ANCHO, cfg.VENTANA_ALTO)
        self.tex = None
        self._ultimo = -1e9
        self.redibujos = 0

    def iniciar(self, campos, ctx):
        from gl_util import Textura
        self.tex = Textura(cfg.HUD_ANCHO, cfg.VENTANA_ALTO, 4)
        self.tex.subir(self.panel.dibujar(campos))

    def actualizar(self, campos, dt):
        ahora = time.perf_counter()
        if ahora - self._ultimo >= 1.0 / cfg.HUD_HZ and self.panel.necesita_redibujo(campos):
            self.tex.subir(self.panel.dibujar(campos))
            self._ultimo = ahora
            self.redibujos += 1

    def dibujar(self, campos, ctx):
        ancho_hud = ctx.ancho_fb * cfg.HUD_ANCHO // cfg.VENTANA_ANCHO
        ctx.quad.dibujar(self.tex, ctx.ancho_fb - ancho_hud, 0, ancho_hud, ctx.alto_fb)


class VistaMapa(Modulo):
    """Vista de depuración 2D."""
    nombre = "vista de mapa"

    def iniciar(self, campos, ctx):
        from gl_util import Textura
        self.tex = Textura(cfg.GRID, cfg.GRID, 3)

    def actualizar(self, campos, dt):
        self.tex.subir(imagen_mapa(campos))

    def dibujar(self, campos, ctx):
        zona = ctx.ancho_fb - ctx.ancho_fb * cfg.HUD_ANCHO // cfg.VENTANA_ANCHO
        lado = min(zona, ctx.alto_fb)
        ctx.quad.dibujar(self.tex, (zona - lado) // 2, (ctx.alto_fb - lado) // 2, lado, lado)


class Contexto:
    """Lo que los módulos necesitan para dibujar en pantalla."""
    ancho_fb = cfg.VENTANA_ANCHO
    alto_fb = cfg.VENTANA_ALTO
    quad = None
    vp = None
    cam_pos = None


def crear_modulos(vista3d=False):
    modulos = [
        Terreno(),          # rol 2
        Arrastre(),         # rol 3
        Materiales(),       # rol 4
        Iluminacion(),      # rol 5
        ModuloHUD(),        # rol 1 (siempre al final: se dibuja encima)
    ]
    if not vista3d:
        modulos.insert(0, VistaMapa())
    return modulos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=0, help="salir después de N cuadros (0 = sin límite)")
    ap.add_argument("--captura", default="", help="guardar una captura PNG antes de salir")
    ap.add_argument("--t-inicio", type=float, default=0.0, help="segundo de inicio de simulación")
    ap.add_argument("--3d", dest="vista3d", action="store_true", help="activar vista 3D")
    args = ap.parse_args()

    import glfw
    from OpenGL import GL as gl

    if not glfw.init():
        sys.exit("No se pudo iniciar GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    ventana = glfw.create_window(cfg.VENTANA_ANCHO, cfg.VENTANA_ALTO, "Simulación de tsunami", None, None)
    if not ventana:
        glfw.terminate()
        sys.exit("No se pudo crear la ventana OpenGL")
    glfw.make_context_current(ventana)
    glfw.swap_interval(0)       # sin vsync para medir fps reales; ponlo en 1 para la demo
    print("OpenGL:", gl.glGetString(gl.GL_VERSION).decode(), "|", gl.glGetString(gl.GL_RENDERER).decode())

    from gl_util import QuadTextura
    ctx = Contexto()
    ctx.ancho_fb, ctx.alto_fb = glfw.get_framebuffer_size(ventana)
    ctx.quad = QuadTextura()

    campos = Campos.vacios()
    falsos = DatosFalsos(campos)
    modulos = crear_modulos(args.vista3d)

    # Identificar el módulo del Rol 3 para el control de la cámara
    modulo_arrastre = next((m for m in modulos if isinstance(m, Arrastre)), None)

    # Callback para scroll del ratón gestionado por el Rol 3
    def al_hacer_scroll(win, xoffset, yoffset):
        if modulo_arrastre and args.vista3d:
            modulo_arrastre.camara.scroll_callback(xoffset, yoffset)

    glfw.set_scroll_callback(ventana, al_hacer_scroll)

    for m in modulos:
        m.iniciar(campos, ctx)

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

    t_sim = args.t_inicio
    inicio_total = anterior = glfw.get_time()
    t_fps, cuadros_fps, cuadros = anterior, 0, 0

    gl.glEnable(gl.GL_DEPTH_TEST)

    while not glfw.window_should_close(ventana):
        ahora = glfw.get_time()
        dt, anterior = ahora - anterior, ahora
        if estado["reiniciar"]:
            t_sim, estado["reiniciar"] = args.t_inicio, False
        if not estado["pausa"]:
            t_sim += dt
            falsos.actualizar(t_sim)

        # Entrada y cálculo de cámara desde el Rol 3
        if args.vista3d and modulo_arrastre:
            modulo_arrastre.camara.procesar_entrada(glfw, ventana, dt)
            zona = ctx.ancho_fb - ctx.ancho_fb * cfg.HUD_ANCHO // cfg.VENTANA_ANCHO
            ctx.vp, ctx.cam_pos = modulo_arrastre.camara.matrices(zona / ctx.alto_fb)

        for m in modulos:
            m.actualizar(campos, dt)

        if args.vista3d:
            zona = ctx.ancho_fb - ctx.ancho_fb * cfg.HUD_ANCHO // cfg.VENTANA_ANCHO
            gl.glViewport(0, 0, zona, ctx.alto_fb)
            gl.glClearColor(0.55, 0.72, 0.90, 1.0)
        else:
            gl.glViewport(0, 0, ctx.ancho_fb, ctx.alto_fb)
            gl.glClearColor(0.08, 0.09, 0.12, 1.0)

        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        for m in modulos:
            m.dibujar(campos, ctx)

        cuadros += 1
        cuadros_fps += 1
        if ahora - t_fps >= 0.5:
            fps = cuadros_fps / (ahora - t_fps)
            glfw.set_window_title(ventana, f"Simulación de tsunami | {fps:5.1f} fps | t = {t_sim:5.1f} s"
                                  + (" | PAUSA" if estado["pausa"] else ""))
            t_fps, cuadros_fps = ahora, 0

        if args.frames and cuadros >= args.frames:
            if args.captura:
                gl.glReadBuffer(gl.GL_BACK)
                datos = gl.glReadPixels(0, 0, ctx.ancho_fb, ctx.alto_fb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
                img = np.frombuffer(datos, np.uint8).reshape(ctx.alto_fb, ctx.ancho_fb, 3)[::-1]
                from PIL import Image
                Image.fromarray(np.ascontiguousarray(img)).save(args.captura)
                print("captura guardada en", args.captura)
            break
        glfw.swap_buffers(ventana)
        glfw.poll_events()

    total = glfw.get_time() - inicio_total
    print(f"{cuadros} cuadros en {total:.1f} s = {cuadros / max(total, 1e-9):.0f} fps promedio "
          f"(ojo: en la nube el render es por software; mide en tu computadora)")
    for m in modulos:
        m.cerrar()
    glfw.terminate()


if __name__ == "__main__":
    main()