"""Horizonte (Rol 2): el terreno y el mar siguen más allá del borde del mapa.

La simulación cubre 640 x 640 m. Sin esto el mapa se ve como una baldosa flotando, y el mar
que levanta la ola se corta en el borde como una caja. Aquí se agregan franjas alrededor del
mapa que nacen exactamente en el borde (mismo alto y mismo color) y se alejan hasta 3,2 km:
  * terreno: la tierra sigue plana y el fondo marino sigue bajando;
  * agua: nace al alto del agua del borde (H + W) y baja suave hasta el nivel del mar.
A lo lejos todo se funde con el color del cielo, así no se ve dónde termina.

Cada franja sale de un lado del mapa como una grilla (anillos x muestras del borde); en las
esquinas hay un parche que une los dos lados. Los valores del borde se suavizan más cuanto
más lejos del mapa, para que no queden rayas. Es solo dibujo: lee H y W, no escribe nada.
"""
import ctypes

import numpy as np

import config as cfg
from modulos.base import Modulo
from modulos.materiales import REFLEJO_CIELO_GLSL
from modulos.terreno_gen import colores_terreno

ANILLOS_M = (0.0, 15.0, 45.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
CAIDA_FONDO = 0.05          # el fondo marino sigue bajando 5 cm por metro fuera del mapa
FONDO_MIN_M = -40.0
CAIDA_OLA_M = 600.0         # distancia en la que el agua levantada por la ola baja al nivel del mar
BAJA_TIERRA = 0.8           # la tierra lejana baja este tanto de su alto en el borde...
BAJA_TIERRA_M = 1200.0      # ...a esta distancia, para que los cerros no sigan como una meseta
COLOR_LEJANIA = np.array([0.40, 0.52, 0.36])   # vegetación vista de lejos: la tierra lejana va a este color
LEJANIA_M = 700.0
NIEBLA_M = (200.0, 2600.0)  # niebla extra del mar según la distancia al borde (se suma a la de la cámara)
NIEBLA_TIERRA_M = (40.0, 500.0)   # la tierra lejana se pierde antes, como cerros en la bruma


def promedio_movil(v, ancho):
    """Promedio de `ancho` muestras vecinas (impar), repitiendo los extremos."""
    if ancho <= 1:
        return np.asarray(v, np.float64).copy()
    m = ancho // 2
    ext = np.pad(np.asarray(v, np.float64), m, mode="edge")
    c = np.concatenate([[0.0], np.cumsum(ext)])
    return (c[ancho:] - c[:-ancho]) / ancho


def caida_ola(r):
    """1 en el borde del mapa y 0 a CAIDA_OLA_M: el agua levantada vuelve al nivel del mar."""
    x = np.clip(np.asarray(r) / CAIDA_OLA_M, 0.0, 1.0)
    return 1.0 - x * x * (3 - 2 * x)


class Bordes:
    """Geometría de las franjas: posiciones fijas y cómo tomar los valores del borde de un campo."""

    def __init__(self, grid=None, cell_m=None):
        G = cfg.GRID if grid is None else grid
        cell = cfg.CELL_M if cell_m is None else cell_m
        self.G, self.cell = G, cell
        R = np.asarray(ANILLOS_M, np.float64)
        self.R = R
        c = (G - 1) * cell
        i = np.arange(G)
        pos = i * cell
        # Lados: (iz, ix) de las celdas del borde, punto base (x, z) y dirección hacia afuera.
        self.lados = [
            (i, np.zeros(G, int), np.stack([np.zeros(G), pos], 1), np.array([-1.0, 0.0])),     # x = 0 (mar)
            (i, np.full(G, G - 1), np.stack([np.full(G, c), pos], 1), np.array([1.0, 0.0])),  # x máximo
            (np.zeros(G, int), i, np.stack([pos, np.zeros(G)], 1), np.array([0.0, -1.0])),     # z = 0
            (np.full(G, G - 1), i, np.stack([pos, np.full(G, c)], 1), np.array([0.0, 1.0])),  # z máximo
        ]
        # Esquinas: (lado en x, muestra en ese lado, lado en z, muestra en ese lado, signo x, signo z).
        self.esquinas = [(0, 0, 2, 0, -1, -1), (0, G - 1, 3, 0, -1, 1),
                         (1, 0, 2, G - 1, 1, -1), (1, G - 1, 3, G - 1, 1, 1)]
        self.anchos = [1 + 2 * min(15, int(r / (4 * cell))) for r in R]

        xz, dist, formas = [], [], []
        for _iz, _ix, base, d in self.lados:
            p = base[None, :, :] + R[:, None, None] * d[None, None, :]
            xz.append(p.reshape(-1, 2))
            dist.append(np.repeat(R, G))
            formas.append((len(R), G))
        for lx, jx, lz, jz, sx, sz in self.esquinas:
            esquina = np.array([self.lados[lx][2][jx][0], self.lados[lz][2][jz][1]])
            p = np.empty((len(R), len(R), 2))
            p[..., 0] = esquina[0] + sx * R[:, None]
            p[..., 1] = esquina[1] + sz * R[None, :]
            xz.append(p.reshape(-1, 2))
            dist.append(np.hypot(R[:, None], R[None, :]).ravel())
            formas.append((len(R), len(R)))
        self.xz = np.concatenate(xz)
        self.dist = np.concatenate(dist)
        self.formas = formas
        self.indices = self._indices()

    def _indices(self):
        idx, o = [], 0
        for a, b in self.formas:
            k = np.arange(a * b).reshape(a, b) + o
            p, q, r, s = k[:-1, :-1], k[1:, :-1], k[:-1, 1:], k[1:, 1:]
            idx.append(np.stack([p, q, r, r, q, s], -1).ravel())
            o += a * b
        return np.concatenate(idx).astype(np.uint32)

    def valores(self, campo):
        """Valor de `campo` [G,G] en cada vértice de las franjas, más suave cuanto más lejos."""
        suaves = []
        for iz, ix, _b, _d in self.lados:
            borde = campo[iz, ix]
            suaves.append(np.stack([promedio_movil(borde, w) for w in self.anchos]))
        partes = [s.ravel() for s in suaves]
        for lx, jx, lz, jz, _sx, _sz in self.esquinas:
            a, b = suaves[lx][:, jx], suaves[lz][:, jz]
            partes.append((a[:, None] + b[None, :] - a[0]).ravel())     # coincide con los dos lados vecinos
        return np.concatenate(partes)

    def alto_terreno(self, H):
        h = self.valores(H)
        fondo = np.maximum(h - CAIDA_FONDO * self.dist, FONDO_MIN_M)
        x = np.clip(self.dist / BAJA_TIERRA_M, 0.0, 1.0)
        tierra = h * (1.0 - BAJA_TIERRA * x * x * (3 - 2 * x))
        return np.where(h >= cfg.NIVEL_MAR_M, tierra, fondo)

    def agua(self, H, W, terreno):
        """Alto de la superficie y profundidad del agua en las franjas (profundidad < 0: no hay agua)."""
        s = self.valores(H + W)
        y = cfg.NIVEL_MAR_M + (s - cfg.NIVEL_MAR_M) * caida_ola(self.dist)
        return y, y - terreno

    def normales(self, P):
        partes, o = [], 0
        for a, b in self.formas:
            g = P[o:o + a * b].reshape(a, b, 3)
            n = np.cross(np.gradient(g, axis=0), np.gradient(g, axis=1))
            n *= np.where(n[..., 1:2] < 0, -1.0, 1.0)
            n /= np.maximum(np.linalg.norm(n, axis=-1, keepdims=True), 1e-9)
            partes.append(n.reshape(-1, 3))
            o += a * b
        return np.concatenate(partes)


VERT_TERRENO = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
layout(location = 3) in float aDist;
uniform mat4 uVP;
out vec3 vPos;
out vec3 vNormal;
out vec3 vColor;
out float vDist;
void main() {
    vPos = aPos; vNormal = aNormal; vColor = aColor; vDist = aDist;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

# Mismo material que el terreno del rol 4 (sin humedad ni textura de detalle) más la niebla.
FRAG_TERRENO = """
in vec3 vPos;
in vec3 vNormal;
in vec3 vColor;
in float vDist;
uniform vec3 uSolDir;
uniform vec3 uSolColor;
uniform vec3 uCamPos;
uniform float uAlarma;
uniform vec3 uCielo;
uniform vec2 uNiebla;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vPos);
    vec3 L = normalize(uSolDir);
    vec3 col = phong_mojado(N, V, L, vColor, uSolColor, 0.35, 0.65, 0.10, 16.0, 0.0, 1.0);
    col = mix(col, vec3(0.10, 0.25, 0.35), clamp(-vPos.y / 6.0, 0.0, 0.6));
    col = tinte_alarma(col, uAlarma);
    // Niebla de la cámara (la misma de todo el mapa) más una propia que crece al alejarse del borde.
    // La propia es solo para la tierra: el fondo marino se ve a través del agua y, si se aclarara,
    // el agua de fuera quedaría más clara que la del mapa.
    float propia = vPos.y >= 0.0 ? smoothstep(uNiebla.x, uNiebla.y, vDist) : 0.0;
    float f = max(factor_niebla(vPos, uCamPos), propia);
    fragColor = vec4(con_niebla(col, f, uCielo, uAlarma), 1.0);
}
"""

VERT_AGUA = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aProf;
layout(location = 2) in float aDist;
uniform mat4 uVP;
out vec3 vPos;
out float vProf;
out float vDist;
void main() {
    vPos = aPos; vProf = aProf; vDist = aDist;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

# Mismo color, reflejo del cielo y transparencia que el agua del rol 4 (sin la espuma de la
# orilla), así no queda costura con el agua del mapa. A lo lejos se funde con el cielo.
FRAG_AGUA = REFLEJO_CIELO_GLSL + """
in vec3 vPos;
in float vProf;
in float vDist;
uniform vec3 uSolDir;
uniform vec3 uSolColor;
uniform vec3 uCamPos;
uniform float uAlarma;
uniform float uUmbral;
uniform float uDMax;
uniform float uAlfaMin;
uniform float uAlfaMax;
uniform vec3 uCielo;
uniform vec2 uNiebla;
out vec4 fragColor;
void main() {
    if (vProf < uUmbral) discard;
    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 V = normalize(uCamPos - vPos);
    vec3 L = normalize(uSolDir);
    vec3 col = phong_mojado(N, V, L, rampa_profundidad(vProf, uDMax), uSolColor,
                            0.45, 0.55, 0.90, 64.0, 1.0, 1.0);
    col = reflejo_cielo(col, N, V, uCielo);
    col = tinte_alarma(col, uAlarma);
    float alfa = mix(uAlfaMin, uAlfaMax, clamp(vProf / uDMax, 0.0, 1.0));
    alfa *= smoothstep(uUmbral, uUmbral + 0.3, vProf);
    float niebla = max(factor_niebla(vPos, uCamPos), smoothstep(uNiebla.x, uNiebla.y, vDist));
    fragColor = vec4(con_niebla(col, niebla, uCielo, uAlarma), mix(alfa, 1.0, niebla));
}
"""

UNIFORMS = ("uVP", "uSolDir", "uSolColor", "uCamPos", "uAlarma", "uCielo", "uNiebla",
            "uUmbral", "uDMax", "uAlfaMin", "uAlfaMax")


class Horizonte(Modulo):
    nombre = "horizonte (rol 2)"

    def __init__(self):
        self.bordes = None
        self.terreno = None         # [n, 10] posición, normal, color, distancia al mapa
        self.agua = None            # [n, 5]  posición, profundidad, distancia al mapa
        self._alto_terreno = None
        self._gl = None

    def iniciar(self, campos, ctx=None):
        b = Bordes()
        self.bordes = b
        h = b.alto_terreno(campos.H)
        self._alto_terreno = h
        P = np.column_stack([b.xz[:, 0], h, b.xz[:, 1]])
        x = np.clip(b.dist / LEJANIA_M, 0.0, 1.0)[:, None]
        lejos = np.where(h[:, None] >= cfg.NIVEL_MAR_M, x * x * (3 - 2 * x), 0.0)   # solo la tierra
        col = colores_terreno(h, np.zeros_like(h)) * (1 - lejos) + COLOR_LEJANIA * lejos
        self.terreno = np.column_stack([P, b.normales(P), col, b.dist]).astype(np.float32)
        self.agua = np.zeros((len(P), 5), np.float32)
        self.agua[:, 0], self.agua[:, 2], self.agua[:, 4] = b.xz[:, 0], b.xz[:, 1], b.dist
        self.actualizar(campos, 0.0)
        if ctx is not None:
            self._subir_a_gpu()

    def actualizar(self, campos, dt):
        if self.bordes is None:
            return
        from modulos.materiales import ELEVACION_AGUA
        y, prof = self.bordes.agua(campos.H, campos.W, self._alto_terreno)
        self.agua[:, 1] = y + ELEVACION_AGUA
        self.agua[:, 3] = prof

    def _comunes(self, ctx):
        from modulos.materiales import SOL_COLOR_DEFECTO, SOL_DIR_DEFECTO
        return {
            "uSolDir": getattr(ctx, "sol_dir", SOL_DIR_DEFECTO),
            "uSolColor": getattr(ctx, "sol_color", SOL_COLOR_DEFECTO),
            "uCamPos": ctx.cam_pos, "uCielo": cfg.COLOR_CIELO,
        }

    def dibujar(self, campos, ctx):
        """Terreno de fuera del mapa. Es opaco: va con lo opaco, ANTES del agua transparente del
        rol 4. Si se dibujara después, taparía el agua del mapa (el agua no escribe profundidad)."""
        vp, cam = getattr(ctx, "vp", None), getattr(ctx, "cam_pos", None)
        if self._gl is None or vp is None or cam is None:
            return
        from modulos.materiales import _alarma
        gl = self._gl["gl"]
        prueba_prof = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)
        u = self._usar(gl, "t", vp, self._comunes(ctx), _alarma(campos))
        gl.glUniform2f(u["uNiebla"], *NIEBLA_TIERRA_M)
        gl.glBindVertexArray(self._gl["vao_t"])
        gl.glDrawElements(gl.GL_TRIANGLES, self._gl["n_idx"], gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        if not prueba_prof:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def dibujar_agua(self, campos, ctx):
        """Agua de fuera del mapa: transparente, va después del agua del rol 4 (ver AguaHorizonte)."""
        vp, cam = getattr(ctx, "vp", None), getattr(ctx, "cam_pos", None)
        if self._gl is None or vp is None or cam is None:
            return
        from modulos.materiales import ALFA_MAX, ALFA_MIN, PROF_MAX_AGUA, _alarma
        gl = self._gl["gl"]
        comunes = self._comunes(ctx)
        prueba_prof = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)

        datos = np.ascontiguousarray(self.agua, np.float32)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._gl["vbo_a"])
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, datos.nbytes, datos)
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)
        u = self._usar(gl, "a", vp, comunes, _alarma(campos))
        gl.glUniform1f(u["uUmbral"], float(cfg.UMBRAL_MOJADO_M))
        gl.glUniform1f(u["uDMax"], float(PROF_MAX_AGUA))
        gl.glUniform1f(u["uAlfaMin"], float(ALFA_MIN))
        gl.glUniform1f(u["uAlfaMax"], float(ALFA_MAX))
        gl.glBindVertexArray(self._gl["vao_a"])
        gl.glDrawElements(gl.GL_TRIANGLES, self._gl["n_idx"], gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)
        if not prueba_prof:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def _usar(self, gl, cual, vp, comunes, alarma):
        gl.glUseProgram(self._gl["prog_" + cual])
        u = self._gl["u_" + cual]
        gl.glUniformMatrix4fv(u["uVP"], 1, gl.GL_TRUE, np.ascontiguousarray(vp, np.float32))
        for nombre, valor in comunes.items():
            gl.glUniform3f(u[nombre], *[float(v) for v in valor])
        gl.glUniform1f(u["uAlarma"], float(alarma))
        gl.glUniform2f(u["uNiebla"], *NIEBLA_M)
        return u

    def cerrar(self):
        if self._gl is None:
            return
        gl = self._gl["gl"]
        gl.glDeleteVertexArrays(2, [self._gl["vao_t"], self._gl["vao_a"]])
        gl.glDeleteBuffers(3, [self._gl["vbo_t"], self._gl["vbo_a"], self._gl["ebo"]])
        gl.glDeleteProgram(self._gl["prog_t"])
        gl.glDeleteProgram(self._gl["prog_a"])
        self._gl = None

    def _subir_a_gpu(self):
        from OpenGL import GL as gl

        from gl_util import compilar_programa
        from modulos.iluminacion import armar_fragment

        prog_t = compilar_programa(VERT_TERRENO, armar_fragment(FRAG_TERRENO))
        prog_a = compilar_programa(VERT_AGUA, armar_fragment(FRAG_AGUA))
        idx = np.ascontiguousarray(self.bordes.indices, np.uint32)
        ebo = gl.glGenBuffers(1)

        def vao(datos, atributos, uso):
            datos = np.ascontiguousarray(datos, np.float32)
            v, b = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
            gl.glBindVertexArray(v)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, b)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, datos.nbytes, datos, uso)
            paso, desde = datos.shape[1] * 4, 0
            for loc, n in enumerate(atributos):
                gl.glEnableVertexAttribArray(loc)
                gl.glVertexAttribPointer(loc, n, gl.GL_FLOAT, gl.GL_FALSE, paso, ctypes.c_void_p(desde * 4))
                desde += n
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            return v, b

        vao_t, vbo_t = vao(self.terreno, (3, 3, 3, 1), gl.GL_STATIC_DRAW)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        vao_a, vbo_a = vao(self.agua, (3, 1, 1), gl.GL_DYNAMIC_DRAW)
        gl.glBindVertexArray(0)
        self._gl = {
            "gl": gl, "prog_t": prog_t, "prog_a": prog_a, "vao_t": vao_t, "vao_a": vao_a,
            "vbo_t": vbo_t, "vbo_a": vbo_a, "ebo": ebo, "n_idx": len(idx),
            "u_t": {n: gl.glGetUniformLocation(prog_t, n) for n in UNIFORMS},
            "u_a": {n: gl.glGetUniformLocation(prog_a, n) for n in UNIFORMS},
        }


class AguaHorizonte(Modulo):
    """Segundo paso del horizonte: el agua de fuera del mapa, con lo transparente.

    En crear_modulos() va después de Materiales (rol 4), y Horizonte va antes, con lo opaco.
    """
    nombre = "horizonte, agua (rol 2)"

    def __init__(self, horizonte):
        self.horizonte = horizonte

    def dibujar(self, campos, ctx):
        self.horizonte.dibujar_agua(campos, ctx)
