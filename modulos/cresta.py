"""Cresta de la ola (Rol 2): superficie de Bézier bicúbica que sigue al frente de ola.

Es decorado: no cambia W, U ni V. Lee del contrato `frente_x` (la x del frente en cada fila
mientras la ola entra) y `H`. Cuando el agua bajo el frente se vuelve poco profunda, la cresta
baja hasta desaparecer, como una ola que rompe al llegar a la costa.

La superficie se arma con parches bicúbicos de Bézier de 4 x 4 puntos de control:
  * en u (de atrás hacia adelante) va el perfil de la ola en tres parches: la espalda que sube
    hasta la cima, el labio que se enrosca hacia adelante y la cara que baja hasta el mar;
  * en v (a lo largo del frente) cada parche toma cuatro filas de la grilla, así sigue la
    curva del frente. Los parches vecinos comparten su borde, así que la superficie es continua.
El material del agua y la espuma lo pone el rol 4; aquí se dibuja con un color simple.
"""
import ctypes

import numpy as np

import config as cfg
from modulos.base import Modulo

ALTO_CRESTA_M = 8.0        # alto de la cima sobre el nivel del mar cuando el frente va por agua profunda
NIVEL_ESPALDA_M = 3.0      # la espalda se junta con el agua de la ola (ALTURA_FRENTE_M del rol 1)
PROF_ROMPE_M = 4.0         # con menos profundidad que esto bajo el frente, la cresta empieza a bajar
TRAMOS = 8                 # parches a lo largo del frente
MUESTRAS = 9               # puntos evaluados por parche en cada dirección
ALFA = 0.85

# Perfil completo (s = x - frente_x en m, y en m sobre el mar) y el perfil "roto", ya sin cresta.
# Cada fila es un parche en u con sus 4 puntos de control.
PERFIL = np.array([
    [(-45.0, 3.0), (-25.0, 3.0), (-10.0, 8.0), (0.0, 8.0)],     # espalda
    [(0.0, 8.0), (8.0, 8.4), (14.0, 6.4), (10.0, 4.2)],         # labio que se enrosca
    [(10.0, 4.2), (6.0, 3.2), (5.0, 1.2), (9.0, 0.0)],          # cara que baja hasta el mar
])
PERFIL_ROTO = PERFIL.copy()
PERFIL_ROTO[:2, :, 1] = NIVEL_ESPALDA_M
PERFIL_ROTO[2, :, 1] = (NIVEL_ESPALDA_M, NIVEL_ESPALDA_M, 1.0, 0.0)
# Fracción de espuma de cada parche (0 = agua, 1 = blanco), de abajo hacia la cima.
ESPUMA = (0.35, 0.9, 0.25)

COLOR_AGUA = np.array([0.16, 0.45, 0.66])
COLOR_ESPUMA = np.array([0.93, 0.96, 0.98])


def bernstein(t):
    """Base cúbica de Bernstein: [n, 4] para n valores de t en [0, 1]."""
    t = np.asarray(t, np.float64)[:, None]
    return np.hstack([(1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t ** 2 * (1 - t), t ** 3])


def parche_bezier(P, nu=MUESTRAS, nv=MUESTRAS):
    """Evalúa un parche bicúbico: P [4, 4, 3] -> puntos [nu, nv, 3].

    S(u, v) = sum_i sum_j B_i(u) B_j(v) P_ij
    """
    Bu, Bv = bernstein(np.linspace(0, 1, nu)), bernstein(np.linspace(0, 1, nv))
    return np.einsum("ui,ijk,vj->uvk", Bu, np.asarray(P, np.float64), Bv)


def factor_cresta(frente_x, H, cell_m=None):
    """De 0 a 1 por fila: 1 con el frente sobre agua profunda, 0 sin frente o ya en la costa."""
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    g = H.shape[0]
    hay = frente_x != cfg.SIN_FRENTE
    ix = np.clip((np.maximum(frente_x, 0.0) / cell_m).astype(int), 0, H.shape[1] - 1)
    prof = cfg.NIVEL_MAR_M - H[np.arange(g), ix]
    return np.where(hay, np.clip(prof / PROF_ROMPE_M, 0.0, 1.0), 0.0)


def malla_cresta(frente_x, H, cell_m=None):
    """Triángulos de la cresta: [n, 9] (posición, normal, color). Vacío si no hay cresta."""
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    g = H.shape[0]
    f = factor_cresta(frente_x, H, cell_m)
    tramo = (g - 1) / TRAMOS
    trozos = []
    for k in range(TRAMOS):
        filas = np.rint(k * tramo + np.array([0, 1, 2, 3]) * tramo / 3).astype(int)
        fv = f[filas]
        if fv.max() < 0.02 or (frente_x[filas] == cfg.SIN_FRENTE).any():
            continue
        x0 = frente_x[filas].astype(np.float64)
        z = filas * cell_m
        for p in range(len(PERFIL)):
            P = np.empty((4, 4, 3))
            P[:, :, 0] = x0[None, :] + PERFIL[p, :, 0][:, None]
            P[:, :, 1] = (cfg.NIVEL_MAR_M + PERFIL_ROTO[p, :, 1][:, None]
                          + fv[None, :] * (PERFIL[p, :, 1] - PERFIL_ROTO[p, :, 1])[:, None])
            P[:, :, 2] = z[None, :]
            trozos.append(_triangulos(parche_bezier(P), ESPUMA[p]))
    if not trozos:
        return np.zeros((0, 9), np.float32)
    return np.concatenate(trozos).astype(np.float32)


def _triangulos(S, espuma):
    """Grilla [nu, nv, 3] -> triángulos con normal suave y color de agua a espuma según el alto."""
    du = np.gradient(S, axis=0)
    dv = np.gradient(S, axis=1)
    n = np.cross(du, dv)
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    y = S[..., 1]
    k = espuma * np.clip((y - y.min()) / max(float(np.ptp(y)), 1e-6), 0.0, 1.0)
    col = COLOR_AGUA * (1 - k[..., None]) + COLOR_ESPUMA * k[..., None]
    v = np.concatenate([S, n, col], axis=-1)
    a, b, c, d = v[:-1, :-1], v[1:, :-1], v[:-1, 1:], v[1:, 1:]
    return np.stack([a, b, c, c, b, d], axis=2).reshape(-1, 9)


VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uVP;
out vec3 vNormal;
out vec3 vColor;
void main() {
    vNormal = aNormal;
    vColor = aColor;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

FRAG = """#version 330 core
in vec3 vNormal;
in vec3 vColor;
uniform vec3 uSol;
uniform float uAlfa;
out vec4 fragColor;
void main() {
    float luz = abs(dot(normalize(vNormal), normalize(uSol)));   // la lámina se ve por los dos lados
    fragColor = vec4(vColor * (0.45 + 0.55 * luz), uAlfa);
}
"""


class Cresta(Modulo):
    nombre = "cresta (rol 2)"

    def __init__(self):
        self.vertices = np.zeros((0, 9), np.float32)
        self._gl = None

    def iniciar(self, campos, ctx=None):
        if ctx is not None:
            self._subir_a_gpu()

    def actualizar(self, campos, dt):
        self.vertices = malla_cresta(campos.frente_x, campos.H)

    def dibujar(self, campos, ctx):
        vp = getattr(ctx, "vp", None)
        if self._gl is None or vp is None or not len(self.vertices):
            return
        gl = self._gl["gl"]
        datos = np.ascontiguousarray(self.vertices, np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._gl["vbo"])
        gl.glBufferData(gl.GL_ARRAY_BUFFER, datos.nbytes, datos, gl.GL_DYNAMIC_DRAW)
        prueba_prof = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)                  # transparente: después de lo opaco y sin escribir profundidad
        gl.glUseProgram(self._gl["prog"])
        gl.glUniformMatrix4fv(self._gl["u_vp"], 1, gl.GL_TRUE, np.ascontiguousarray(vp, np.float32))
        gl.glUniform3f(self._gl["u_sol"], -0.4, 0.8, -0.3)
        gl.glUniform1f(self._gl["u_alfa"], ALFA)
        gl.glBindVertexArray(self._gl["vao"])
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, len(datos))
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)
        if not prueba_prof:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def cerrar(self):
        if self._gl is None:
            return
        gl = self._gl["gl"]
        gl.glDeleteVertexArrays(1, [self._gl["vao"]])
        gl.glDeleteBuffers(1, [self._gl["vbo"]])
        gl.glDeleteProgram(self._gl["prog"])
        self._gl = None

    def _subir_a_gpu(self):
        from OpenGL import GL as gl

        from gl_util import compilar_programa

        prog = compilar_programa(VERT, FRAG)
        vao, vbo = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        for loc in range(3):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 36, ctypes.c_void_p(loc * 12))
        gl.glBindVertexArray(0)
        self._gl = {
            "gl": gl, "prog": prog, "vao": vao, "vbo": vbo,
            "u_vp": gl.glGetUniformLocation(prog, "uVP"),
            "u_sol": gl.glGetUniformLocation(prog, "uSol"),
            "u_alfa": gl.glGetUniformLocation(prog, "uAlfa"),
        }
