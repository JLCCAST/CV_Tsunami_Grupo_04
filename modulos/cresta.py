"""Cresta de la ola (Rol 2): superficie de Bézier bicúbica que sigue al frente de ola.

Es decorado: no cambia W, U ni V. Lee del contrato `frente_x` (la x del frente en cada fila
mientras la ola entra), `H` y `t`. Cubre el escalón que deja la simulación en el frente con una
ola redondeada: una espalda que nace del agua de la ola y una cara de espuma que baja hasta el mar,
como el frente turbulento de un tsunami. Cuando el agua bajo el frente se vuelve poco profunda,
la cresta baja hasta desaparecer, como si rompiera en la costa.

La superficie se arma con parches bicúbicos de Bézier de 4 x 4 puntos de control:
  * en u (de atrás hacia adelante) el perfil, en dos parches: la espalda y la cara;
  * en v (a lo largo del frente) cada parche toma cuatro filas de la grilla, así sigue la curva
    del frente. Los parches vecinos comparten su borde, así que la superficie es continua.
La altura cambia a lo largo del frente y la espuma forma vetas, para que no se vea como un muro
parejo. El material definitivo del agua lo pone el rol 4.
"""
import ctypes

import numpy as np

import config as cfg
from modulos.base import Modulo

NIVEL_ESPALDA_M = 3.0      # alto del agua de la ola justo detrás del frente (ALTURA_FRENTE_M del rol 1)
ALTO_EXTRA_M = 2.5         # cuánto sube la cima sobre esa agua cuando el frente va por agua profunda
PROF_ROMPE_M = 4.0         # con menos profundidad que esto bajo el frente, la cresta empieza a bajar
TRAMOS = 16                # parches a lo largo del frente
MUESTRAS = 9               # puntos evaluados por parche en cada dirección
ALFA = 0.9

# Perfil de cada parche en u: desfase s = x - frente_x (m) y alto de cada punto de control como
# (a, b): y = nivel del mar + a * NIVEL_ESPALDA_M + b * alto de la cima sobre la espalda.
S_PERFIL = np.array([[-35.0, -20.0, -10.0, -3.0],      # espalda
                     [-3.0, 2.0, 5.0, 10.0]])          # cara, termina en el mar
A_PERFIL = np.array([[1.0, 1.0, 1.0, 1.0],
                     [1.0, 1.0, 0.6, 0.0]])
B_PERFIL = np.array([[0.0, 0.0, 1.0, 1.0],
                     [1.0, 1.0, 0.3, 0.0]])

COLOR_AGUA = np.array([0.17, 0.47, 0.64])
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


def ondulacion(z, t):
    """Factor de alto a lo largo del frente (entre 0.5 y 1.5): la cresta no es pareja."""
    r = 1.0 + 0.25 * np.sin(z / 41.0 + 0.8) + 0.15 * np.sin(z / 17.0 + 0.9 * t) + 0.08 * np.sin(z / 7.0 - 1.4 * t)
    return np.clip(r, 0.5, 1.5)


def vetas(z, t):
    """Cuánta espuma hay a lo largo del frente (entre 0.3 y 1): forma vetas que se mueven."""
    return 0.65 + 0.35 * np.sin(z / 7.3 + 1.7 * t) * np.sin(z / 3.1 - 0.9 * t)


def malla_cresta(frente_x, H, t=0.0, cell_m=None):
    """Triángulos de la cresta: [n, 10] (posición, normal, color, alfa). Vacío si no hay cresta."""
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
        extra = ALTO_EXTRA_M * fv * ondulacion(z, t)
        for p in range(len(S_PERFIL)):
            P = np.empty((4, 4, 3))
            P[:, :, 0] = x0[None, :] + S_PERFIL[p][:, None]
            P[:, :, 1] = (cfg.NIVEL_MAR_M + A_PERFIL[p][:, None] * NIVEL_ESPALDA_M
                          + B_PERFIL[p][:, None] * extra[None, :])
            P[:, :, 2] = z[None, :]
            trozos.append(_triangulos(parche_bezier(P), p, t))
    if not trozos:
        return np.zeros((0, 10), np.float32)
    return np.concatenate(trozos).astype(np.float32)


def _suave(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _triangulos(S, parche, t):
    """Grilla [nu, nv, 3] -> triángulos con normal suave, color de agua a espuma y alfa."""
    n = np.cross(np.gradient(S, axis=0), np.gradient(S, axis=1))
    n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
    u = np.linspace(0.0, 1.0, S.shape[0])[:, None] * np.ones(S.shape[1])[None, :]
    veta = vetas(S[..., 2], t)
    if parche == 0:      # espalda: nace transparente del agua y se cubre de espuma cerca de la cima
        espuma = 0.85 * _suave((u - 0.55) / 0.45) * veta
        alfa = ALFA * _suave(u / 0.6)
    else:                # cara: espuma turbulenta que se abre al llegar al mar
        espuma = (0.95 - 0.45 * u) * veta
        alfa = ALFA * (1.0 - 0.5 * _suave((u - 0.7) / 0.3))
    col = COLOR_AGUA * (1 - espuma[..., None]) + COLOR_ESPUMA * espuma[..., None]
    v = np.concatenate([S, n, col, alfa[..., None]], axis=-1)
    a, b, c, d = v[:-1, :-1], v[1:, :-1], v[:-1, 1:], v[1:, 1:]
    return np.stack([a, b, c, c, b, d], axis=2).reshape(-1, 10)


VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in float aAlfa;
uniform mat4 uVP;
out vec3 vPos;
out vec3 vNormal;
out vec3 vColor;
out float vAlfa;
void main() {
    vPos = aPos;
    vNormal = aNormal;
    vColor = aColor;
    vAlfa = aAlfa;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

FRAG = """#version 330 core
in vec3 vPos;
in vec3 vNormal;
in vec3 vColor;
in float vAlfa;
uniform vec3 uSol;
uniform vec3 uCam;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCam - vPos);
    if (dot(N, V) < 0.0) N = -N;                 // la lámina se ve por los dos lados
    vec3 L = normalize(uSol);
    float difuso = max(dot(N, L), 0.0);
    float brillo = pow(max(dot(N, normalize(L + V)), 0.0), 60.0);
    fragColor = vec4(vColor * (0.5 + 0.5 * difuso) + vec3(0.45 * brillo), vAlfa);
}
"""

FLOTANTES = 10        # por vértice: posición, normal, color y alfa


class Cresta(Modulo):
    nombre = "cresta (rol 2)"

    def __init__(self):
        self.vertices = np.zeros((0, FLOTANTES), np.float32)
        self._gl = None

    def iniciar(self, campos, ctx=None):
        if ctx is not None:
            self._subir_a_gpu()

    def actualizar(self, campos, dt):
        self.vertices = malla_cresta(campos.frente_x, campos.H, campos.t)

    def dibujar(self, campos, ctx):
        vp = getattr(ctx, "vp", None)
        if self._gl is None or vp is None or not len(self.vertices):
            return
        gl = self._gl["gl"]
        cam = getattr(ctx, "cam_pos", None)
        cam = (cfg.WORLD_M / 2, 400.0, cfg.WORLD_M / 2) if cam is None else cam
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
        gl.glUniform3f(self._gl["u_cam"], *[float(c) for c in cam])
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
        paso = FLOTANTES * 4
        for loc, (n, desde) in enumerate(((3, 0), (3, 3), (3, 6), (1, 9))):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, n, gl.GL_FLOAT, gl.GL_FALSE, paso, ctypes.c_void_p(desde * 4))
        gl.glBindVertexArray(0)
        self._gl = {
            "gl": gl, "prog": prog, "vao": vao, "vbo": vbo,
            "u_vp": gl.glGetUniformLocation(prog, "uVP"),
            "u_sol": gl.glGetUniformLocation(prog, "uSol"),
            "u_cam": gl.glGetUniformLocation(prog, "uCam"),
        }
