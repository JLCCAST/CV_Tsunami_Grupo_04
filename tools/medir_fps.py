"""
medir_fps.py  -  Banco de pruebas del Rol 5 (iluminación) + medición de fps base

Qué hace:
  1. Abre una ventana GLFW (OpenGL 3.3 core).
  2. Dibuja un terreno de prueba de GRID x GRID celdas (datos FALSOS, sin depender de nadie).
  3. Compila el fragment shader concatenando iluminacion.glsl, así que sirve para
     probar phong(), phong_mojado(), tinte_alarma() y direccion_sol().
  4. Mide fps durante N segundos y guarda el resultado en metricas_fps.csv.

Ubicación: tools/medir_fps.py (se ejecuta desde la raíz del repositorio).
Instalación:   python -m pip install -r requirements.txt
Ejemplos:      python tools/medir_fps.py                       (interactivo, sin límite)
               python tools/medir_fps.py --duracion 10 --nota "semana1 base"
               python tools/medir_fps.py --grid 256 --duracion 10 --nota "prueba 256"

Para ver la iluminación sobre el terreno y las casas REALES del Rol 2 usa tools/visor_terreno.py.

Controles (modo interactivo):
  FLECHAS ARRIBA/ABAJO -> hora del día (mueve el sol)
  FLECHAS IZQ./DER.    -> nivel de alarma (0 a 1)
  M                    -> activa/desactiva la "mancha" de mojado que avanza
  ESC                  -> salir
"""

import argparse
import csv
import ctypes
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glfw
import numpy as np
from OpenGL.GL import *

import config as cfg
from modulos.iluminacion import RUTA_GLSL, armar_fragment

# ---------------------------------------------------------------------
#  Shaders (el fragment shader se arma con iluminacion.glsl del Rol 5)
# ---------------------------------------------------------------------
VERTEX_SRC = """#version 330 core
layout(location = 0) in vec3 pos;
layout(location = 1) in vec3 nrm;
uniform mat4 mvp;
out vec3 v_pos;
out vec3 v_nrm;
void main() {
    v_pos = pos;            // el modelo es la identidad: mundo = objeto
    v_nrm = nrm;
    gl_Position = mvp * vec4(pos, 1.0);
}
"""

FRAGMENT_CUERPO = """
in vec3 v_pos;
in vec3 v_nrm;
uniform vec3  cam_pos;
uniform float hora;      // 0..24
uniform float alarma;    // 0..1 (FALSA: la publicará el Rol 1)
uniform float frente;    // posición z del frente de mojado (FALSA: vendrá de la humedad del Rol 4)
out vec4 color;

void main() {
    vec3 N = normalize(v_nrm);
    vec3 V = normalize(cam_pos - v_pos);
    vec3 L = direccion_sol(hora);
    vec3 color_luz = vec3(1.0, 0.96, 0.88) * intensidad_sol(L);

    vec3 arena = vec3(0.78, 0.70, 0.50);
    float humedad = 1.0 - smoothstep(frente - 1.5, frente + 1.5, v_pos.z);

    vec3 c = phong_mojado(N, V, L, arena, color_luz,
                          0.18, 0.90, 0.35, 24.0, humedad, 1.0);
    c = tinte_alarma(c, alarma);
    color = vec4(c, 1.0);
}
"""


def compilar(tipo, fuente):
    sh = glCreateShader(tipo)
    glShaderSource(sh, fuente)
    glCompileShader(sh)
    if not glGetShaderiv(sh, GL_COMPILE_STATUS):
        raise RuntimeError("Error de compilación GLSL:\n" + glGetShaderInfoLog(sh).decode())
    return sh


def crear_programa(ruta_glsl):
    fs_src = armar_fragment(FRAGMENT_CUERPO, ruta_glsl)     # misma ruta que usará el Rol 4
    vs = compilar(GL_VERTEX_SHADER, VERTEX_SRC)
    fs = compilar(GL_FRAGMENT_SHADER, fs_src)
    prog = glCreateProgram()
    glAttachShader(prog, vs)
    glAttachShader(prog, fs)
    glLinkProgram(prog)
    if not glGetProgramiv(prog, GL_LINK_STATUS):
        raise RuntimeError("Error de enlace:\n" + glGetProgramInfoLog(prog).decode())
    return prog


# ---------------------------------------------------------------------
#  Terreno de prueba (datos falsos) y matrices
# ---------------------------------------------------------------------
def construir_terreno(n, tam=16.0):
    """Playa que sube hacia +x con ondulaciones. Devuelve vértices (pos+normal) e índices."""
    xs = np.linspace(-tam / 2, tam / 2, n, dtype=np.float32)
    X, Z = np.meshgrid(xs, xs)                      # X[i,j]=xs[j], Z[i,j]=xs[i]
    Y = 0.08 * (X + tam / 2) + 0.25 * np.sin(X * 0.9) * np.cos(Z * 0.7)
    Y = Y.astype(np.float32)

    paso = float(xs[1] - xs[0])
    dydz, dydx = np.gradient(Y, paso)               # eje 0 = z, eje 1 = x
    nrm = np.stack([-dydx, np.ones_like(Y), -dydz], axis=-1)
    nrm /= np.linalg.norm(nrm, axis=-1, keepdims=True)

    verts = np.concatenate([np.stack([X, Y, Z], axis=-1), nrm], axis=-1)
    verts = verts.reshape(-1, 6).astype(np.float32)

    i, j = np.meshgrid(np.arange(n - 1), np.arange(n - 1), indexing="ij")
    a = (i * n + j).ravel()
    b, c = a + 1, a + n
    d = c + 1
    idx = np.stack([a, c, b, b, c, d], axis=1).ravel().astype(np.uint32)
    return verts, idx


def perspectiva(fov_deg, aspecto, cerca, lejos):
    t = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), np.float32)
    m[0, 0], m[1, 1] = t / aspecto, t
    m[2, 2] = (lejos + cerca) / (cerca - lejos)
    m[2, 3] = 2 * lejos * cerca / (cerca - lejos)
    m[3, 2] = -1.0
    return m


def mirar_a(ojo, objetivo):
    ojo, objetivo = np.array(ojo, np.float32), np.array(objetivo, np.float32)
    f = objetivo - ojo
    f /= np.linalg.norm(f)
    s = np.cross(f, [0, 1, 0]).astype(np.float32)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, np.float32)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ ojo, -u @ ojo, f @ ojo
    return m


# ---------------------------------------------------------------------
#  Estadísticas y registro en CSV
# ---------------------------------------------------------------------
def resumen(tiempos):
    t = np.array(tiempos)
    return {
        "frames": len(t),
        "fps_prom": round(len(t) / t.sum(), 1),
        "ms_prom": round(1000 * t.mean(), 2),
        "ms_p95": round(1000 * np.percentile(t, 95), 2),
        "fps_1pct_low": round(1.0 / np.percentile(t, 99), 1),   # peor 1 % de los cuadros
    }


def guardar_csv(ruta, fila):
    existe = Path(ruta).exists()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if not existe:
            w.writeheader()
        w.writerow(fila)


# ---------------------------------------------------------------------
#  Programa principal
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=cfg.GRID, help="celdas por lado (por defecto cfg.GRID)")
    ap.add_argument("--duracion", type=float, default=0, help="segundos a medir (0 = interactivo)")
    ap.add_argument("--ancho", type=int, default=cfg.VENTANA_ANCHO)
    ap.add_argument("--alto", type=int, default=cfg.VENTANA_ALTO)
    ap.add_argument("--vsync", action="store_true", help="activar vsync (limita a 60 fps)")
    ap.add_argument("--msaa", type=int, default=0, help="muestras MSAA (0 = sin MSAA)")
    ap.add_argument("--glsl", default=str(RUTA_GLSL))
    ap.add_argument("--nota", default="", help="etiqueta, p. ej. 'semana1 base'")
    ap.add_argument("--csv", default="metricas_fps.csv")
    args = ap.parse_args()

    if not glfw.init():
        raise SystemExit("No se pudo iniciar GLFW")
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)      # necesario en macOS
    if args.msaa:
        glfw.window_hint(glfw.SAMPLES, args.msaa)
    ventana = glfw.create_window(args.ancho, args.alto, "Banco de pruebas - Rol 5", None, None)
    if not ventana:
        glfw.terminate()
        raise SystemExit("No se pudo crear la ventana (¿OpenGL 3.3 disponible?)")
    glfw.make_context_current(ventana)
    glfw.swap_interval(1 if args.vsync else 0)               # sin vsync = fps reales

    gpu = glGetString(GL_RENDERER).decode()
    prog = crear_programa(args.glsl)
    verts, idx = construir_terreno(args.grid)

    vao = glGenVertexArrays(1)
    glBindVertexArray(vao)
    vbo = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, verts.nbytes, verts, GL_STATIC_DRAW)
    ebo = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(0))
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 24, ctypes.c_void_p(12))
    glEnableVertexAttribArray(1)

    glEnable(GL_DEPTH_TEST)
    if args.msaa:
        glEnable(GL_MULTISAMPLE)
    glClearColor(0.55, 0.75, 0.90, 1.0)

    loc = {n: glGetUniformLocation(prog, n) for n in ("mvp", "cam_pos", "hora", "alarma", "frente")}
    estado = {"hora": 11.0, "alarma": 0.0, "mojado": True}

    def teclado(_v, tecla, _sc, accion, _mod):
        if accion == glfw.RELEASE:
            return
        if tecla == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(ventana, True)
        elif tecla == glfw.KEY_UP:
            estado["hora"] = min(24.0, estado["hora"] + 1.0)
        elif tecla == glfw.KEY_DOWN:
            estado["hora"] = max(0.0, estado["hora"] - 1.0)
        elif tecla == glfw.KEY_RIGHT:
            estado["alarma"] = min(1.0, estado["alarma"] + 0.1)
        elif tecla == glfw.KEY_LEFT:
            estado["alarma"] = max(0.0, estado["alarma"] - 0.1)
        elif tecla == glfw.KEY_M:
            estado["mojado"] = not estado["mojado"]
    glfw.set_key_callback(ventana, teclado)

    P = perspectiva(55.0, args.ancho / args.alto, 0.1, 100.0)
    tiempos, t0, t_prev, CALENTAMIENTO = [], glfw.get_time(), glfw.get_time(), 1.0
    print(f"GPU: {gpu} | GRID {args.grid}x{args.grid} | {len(idx)//3} triángulos")

    while not glfw.window_should_close(ventana):
        ahora = glfw.get_time()
        dt, t_prev = ahora - t_prev, ahora
        t = ahora - t0

        # Cámara orbital automática (la cámara real es del Rol 3)
        ang = 0.25 * t
        ojo = (11.0 * math.cos(ang), 7.0, 11.0 * math.sin(ang))
        mvp = P @ mirar_a(ojo, (0.0, 0.0, 0.0))
        frente = (-9.0 + 18.0 * ((t / 8.0) % 1.0)) if estado["mojado"] else -99.0

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glUseProgram(prog)
        glUniformMatrix4fv(loc["mvp"], 1, GL_TRUE, mvp)
        glUniform3f(loc["cam_pos"], *ojo)
        glUniform1f(loc["hora"], estado["hora"])
        glUniform1f(loc["alarma"], estado["alarma"])
        glUniform1f(loc["frente"], frente)
        glBindVertexArray(vao)
        glDrawElements(GL_TRIANGLES, len(idx), GL_UNSIGNED_INT, None)

        glfw.swap_buffers(ventana)
        glfw.poll_events()

        if t > CALENTAMIENTO:                       # se descarta el primer segundo
            tiempos.append(dt)
        if args.duracion and t > CALENTAMIENTO + args.duracion:
            break

    if len(tiempos) > 10:
        r = resumen(tiempos)
        print(f"FPS prom: {r['fps_prom']} | ms/cuadro: {r['ms_prom']} | "
              f"p95: {r['ms_p95']} ms | 1% low: {r['fps_1pct_low']} fps")
        guardar_csv(args.csv, {
            "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "gpu": gpu, "grid": args.grid, "ventana": f"{args.ancho}x{args.alto}",
            "msaa": args.msaa, "vsync": args.vsync, **r, "nota": args.nota,
        })
        print(f"Guardado en {args.csv}")
    glfw.terminate()


if __name__ == "__main__":
    main()
