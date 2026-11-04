"""Materiales, humedad del terreno, agua transparente y efectos visuales (Rol 4)."""

import numpy as np

import config as cfg
from modulos.base import Modulo


# Velocidades del efecto de humedad.
ACUMULACION = 0.8
SECADO = 0.12

# Agua transparente.
PROF_MAX_AGUA = 8.0      # metros de profundidad que llegan al color "profundo"
ALFA_MIN = 0.35          # opacidad del agua somera
ALFA_MAX = 0.85          # opacidad del agua profunda
ELEVACION_AGUA = 0.03    # el agua flota un poco sobre H+W para evitar parpadeos

# Respaldo si Iluminacion (Rol 5) todavía no publicó el sol en el contexto.
SOL_DIR_DEFECTO = (0.26, 0.93, 0.34)
SOL_COLOR_DEFECTO = (1.0, 0.96, 0.88)

NOMBRES_UNIFORMS = (
    "uVP", "uOrigen", "uCelda", "uTamMundo",
    "uSolDir", "uSolColor", "uCamPos", "uAlarma", "uMapaHumedad",
)

NOMBRES_UNIFORMS_AGUA = (
    "uVP", "uSolDir", "uSolColor", "uCamPos", "uAlarma",
    "uUmbral", "uDMax", "uAlfaMin", "uAlfaMax",
)

VERT_TERRENO = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uVP;
uniform vec3 uOrigen;
uniform float uCelda;      // metros por celda (CELL_M)
uniform float uTamMundo;   // metros por lado cubiertos por la textura (GRID * CELL_M)
out vec3 vPos;
out vec3 vNormal;
out vec3 vColor;
out vec2 vUvHumedad;
void main() {
    vec3 p = aPos + uOrigen;
    vPos = p;
    vNormal = aNormal;
    vColor = aColor;
    // Medio texel de desfase: cada vértice cae en el centro de su celda.
    vUvHumedad = (p.xz + 0.5 * uCelda) / uTamMundo;
    gl_Position = uVP * vec4(p, 1.0);
}
"""

# Cuerpo del fragment shader, SIN '#version' y SIN funciones de iluminación:
# armar_fragment() (Rol 5) antepone ambas cosas.
FRAG_TERRENO = """
in vec3 vPos;
in vec3 vNormal;
in vec3 vColor;
in vec2 vUvHumedad;

uniform sampler2D uMapaHumedad;
uniform vec3 uSolDir;
uniform vec3 uSolColor;
uniform vec3 uCamPos;
uniform float uAlarma;

out vec4 fragColor;

void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vPos);
    vec3 L = normalize(uSolDir);

    float humedad = texture(uMapaHumedad, vUvHumedad).r;

    // Material del terreno: ka, kd, ks, brillo (seco). Con humedad, phong_mojado
    // oscurece el color y sube el brillo especular.
    vec3 col = phong_mojado(N, V, L, vColor, uSolColor,
                            0.35, 0.65, 0.10, 16.0,
                            humedad, 1.0);

    // Mismo tinte de fondo marino que usa el shader de Rol 2.
    col = mix(col, vec3(0.10, 0.25, 0.35), clamp(-vPos.y / 6.0, 0.0, 0.6));

    col = tinte_alarma(col, uAlarma);
    fragColor = vec4(col, 1.0);
}
"""

# Agua: cada vértice trae su posición (x, y = H + W, z) y la profundidad W.
VERT_AGUA = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in float aProf;
uniform mat4 uVP;
out vec3 vPos;
out float vProf;
void main() {
    vPos = aPos;
    vProf = aProf;
    gl_Position = uVP * vec4(aPos, 1.0);
}
"""

FRAG_AGUA = """
in vec3 vPos;
in float vProf;

uniform vec3 uSolDir;
uniform vec3 uSolColor;
uniform vec3 uCamPos;
uniform float uAlarma;
uniform float uUmbral;
uniform float uDMax;
uniform float uAlfaMin;
uniform float uAlfaMax;

out vec4 fragColor;

void main() {
    // Donde casi no hay agua no se dibuja nada.
    if (vProf < uUmbral) discard;

    vec3 N = vec3(0.0, 1.0, 0.0);
    vec3 V = normalize(uCamPos - vPos);
    vec3 L = normalize(uSolDir);

    // Color según profundidad (Rol 5) e iluminación con brillo de superficie mojada.
    vec3 base = rampa_profundidad(vProf, uDMax);
    vec3 col = phong_mojado(N, V, L, base, uSolColor,
                            0.45, 0.55, 0.90, 64.0,
                            1.0, 1.0);
    col = tinte_alarma(col, uAlarma);

    // Opacidad: más profunda = más opaca. En el borde del agua se desvanece.
    float t = clamp(vProf / uDMax, 0.0, 1.0);
    float alfa = mix(uAlfaMin, uAlfaMax, t);
    alfa *= smoothstep(uUmbral, uUmbral + 0.3, vProf);

    fragColor = vec4(col, alfa);
}
"""


class Materiales(Modulo):
    nombre = "materiales (rol 4)"

    def __init__(self):
        self.humedad = None
        self.textura_humedad = None
        self.programa = None
        self._gl = None
        self._u = {}
        self._t_prev = None
        # Agua
        self._prog_agua = None
        self._u_agua = {}
        self._vao_agua = None
        self._vbo_agua = None
        self._ebo_agua = None
        self._buf_agua = None
        self._n_idx_agua = 0

    def iniciar(self, campos, ctx):
        # Una celda de humedad por cada celda del terreno.
        self.humedad = np.zeros(campos.H.shape, dtype=np.float32)
        if ctx is not None:
            ctx.materiales = self
            self._iniciar_gpu()

    def _iniciar_gpu(self):
        from OpenGL import GL as gl

        from gl_util import Textura, compilar_programa
        from modulos.iluminacion import armar_fragment

        self._gl = gl

        alto, ancho = self.humedad.shape
        self.textura_humedad = Textura(ancho, alto, canales=1, suave=True)
        self.textura_humedad.subir(self.datos_textura_humedad())

        fuente_fs = armar_fragment(FRAG_TERRENO)
        self.programa = compilar_programa(VERT_TERRENO, fuente_fs)
        self._u = {
            nombre: gl.glGetUniformLocation(self.programa, nombre)
            for nombre in NOMBRES_UNIFORMS
        }

        self._iniciar_agua(gl, alto, ancho)

    def _iniciar_agua(self, gl, alto, ancho):
        """Malla del agua: una cuadrícula con un vértice por celda (x, y, z, profundidad)."""
        from gl_util import compilar_programa
        from modulos.iluminacion import armar_fragment

        self._prog_agua = compilar_programa(VERT_AGUA, armar_fragment(FRAG_AGUA))
        self._u_agua = {
            nombre: gl.glGetUniformLocation(self._prog_agua, nombre)
            for nombre in NOMBRES_UNIFORMS_AGUA
        }

        # Buffer CPU: 4 floats por vértice = x, y, z, profundidad. x y z no cambian.
        xs = np.arange(ancho, dtype=np.float32) * cfg.CELL_M
        zs = np.arange(alto, dtype=np.float32) * cfg.CELL_M
        gx, gz = np.meshgrid(xs, zs)            # forma [iz, ix], igual que campos.H
        self._buf_agua = np.zeros((alto * ancho, 4), np.float32)
        self._buf_agua[:, 0] = gx.ravel()
        self._buf_agua[:, 2] = gz.ravel()

        # Dos triángulos por cada cuadro de la cuadrícula.
        idx = np.arange(alto * ancho, dtype=np.uint32).reshape(alto, ancho)
        a, b = idx[:-1, :-1], idx[:-1, 1:]
        c, d = idx[1:, :-1], idx[1:, 1:]
        tri = np.stack([a, c, b, b, c, d], axis=-1).reshape(-1).astype(np.uint32)
        self._n_idx_agua = len(tri)

        self._vao_agua = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self._vao_agua)
        self._vbo_agua = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_agua)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self._buf_agua.nbytes, None, gl.GL_DYNAMIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 1, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(12))
        self._ebo_agua = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._ebo_agua)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, tri.nbytes, tri, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)

    def actualizar(self, campos, dt: float):
        if self.humedad is None:
            return

        if self._t_prev is not None and campos.t < self._t_prev - 1e-9:
            self.humedad[:] = 0.0
        self._t_prev = campos.t

        dt = max(float(dt), 0.0)

        # Detectar las celdas de tierra alcanzadas por el agua.
        tierra = campos.H > cfg.NIVEL_MAR_M
        agua = campos.W > cfg.UMBRAL_MOJADO_M
        mojado = tierra & agua

        # Acumulación gradual: se aproxima progresivamente a 1.
        factor_acumulacion = 1.0 - np.exp(-ACUMULACION * dt)
        self.humedad[mojado] += (
            1.0 - self.humedad[mojado]
        ) * factor_acumulacion

        # Secado gradual: la humedad disminuye cuando no hay agua.
        seco = tierra & ~mojado
        self.humedad[seco] *= np.exp(-SECADO * dt)

        # Mantener los valores dentro del intervalo permitido.
        np.clip(self.humedad, 0.0, 1.0, out=self.humedad)

        # Copiar la humedad actual a la GPU (si existe la textura).
        if self.textura_humedad is not None:
            self.textura_humedad.subir(self.datos_textura_humedad())

    def datos_textura_humedad(self):
        """Convierte la humedad (float 0..1) a uint8 (0..255) con 1 canal."""
        datos = (self.humedad * 255.0).astype(np.uint8)
        return datos[..., np.newaxis]

    def preparar_terreno(self, campos, ctx, vp, cam_pos):
        """Activa el shader de terreno con humedad y deja listos sus uniforms.

        No dibuja: quien llama hace después el glDrawElements de la malla.
        Devuelve False si todavía no hay recursos de GPU.
        """
        if self.programa is None:
            return False

        gl = self._gl
        u = self._u

        gl.glUseProgram(self.programa)
        gl.glUniformMatrix4fv(
            u["uVP"], 1, gl.GL_TRUE, np.ascontiguousarray(vp, np.float32)
        )
        gl.glUniform3f(u["uOrigen"], 0.0, 0.0, 0.0)
        gl.glUniform1f(u["uCelda"], float(cfg.CELL_M))
        gl.glUniform1f(u["uTamMundo"], float(cfg.GRID * cfg.CELL_M))

        sol_dir = getattr(ctx, "sol_dir", SOL_DIR_DEFECTO)
        sol_color = getattr(ctx, "sol_color", SOL_COLOR_DEFECTO)
        gl.glUniform3f(u["uSolDir"], *[float(v) for v in sol_dir])
        gl.glUniform3f(u["uSolColor"], *[float(v) for v in sol_color])
        gl.glUniform3f(u["uCamPos"], *[float(v) for v in cam_pos])
        gl.glUniform1f(u["uAlarma"], float(campos.alarma))

        # La humedad va en la unidad de textura 1 (la 0 la usa el HUD).
        gl.glActiveTexture(gl.GL_TEXTURE1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.textura_humedad.id)
        gl.glUniform1i(u["uMapaHumedad"], 1)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        return True

    def dibujar(self, campos, ctx):
        """Dibuja el agua transparente. Va después del terreno y las casas (lo opaco)."""
        vp = getattr(ctx, "vp", None)
        cam_pos = getattr(ctx, "cam_pos", None)
        if self._prog_agua is None or vp is None or cam_pos is None:
            return

        gl = self._gl
        u = self._u_agua

        # Posiciones y profundidades de este cuadro: y = H + W, profundidad = W.
        self._buf_agua[:, 1] = (campos.H + campos.W).ravel() + ELEVACION_AGUA
        self._buf_agua[:, 3] = campos.W.ravel()
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo_agua)
        gl.glBufferSubData(gl.GL_ARRAY_BUFFER, 0, self._buf_agua.nbytes, self._buf_agua)

        prueba_prof = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDisable(gl.GL_CULL_FACE)           # el agua se ve por ambos lados
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glDepthMask(gl.GL_FALSE)             # transparente: no escribe profundidad

        gl.glUseProgram(self._prog_agua)
        gl.glUniformMatrix4fv(
            u["uVP"], 1, gl.GL_TRUE, np.ascontiguousarray(vp, np.float32)
        )
        sol_dir = getattr(ctx, "sol_dir", SOL_DIR_DEFECTO)
        sol_color = getattr(ctx, "sol_color", SOL_COLOR_DEFECTO)
        gl.glUniform3f(u["uSolDir"], *[float(v) for v in sol_dir])
        gl.glUniform3f(u["uSolColor"], *[float(v) for v in sol_color])
        gl.glUniform3f(u["uCamPos"], *[float(v) for v in cam_pos])
        gl.glUniform1f(u["uAlarma"], float(campos.alarma))
        gl.glUniform1f(u["uUmbral"], float(cfg.UMBRAL_MOJADO_M))
        gl.glUniform1f(u["uDMax"], PROF_MAX_AGUA)
        gl.glUniform1f(u["uAlfaMin"], ALFA_MIN)
        gl.glUniform1f(u["uAlfaMax"], ALFA_MAX)

        gl.glBindVertexArray(self._vao_agua)
        gl.glDrawElements(gl.GL_TRIANGLES, self._n_idx_agua, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

        # Dejar el estado de OpenGL como estaba para los demás módulos.
        gl.glDepthMask(gl.GL_TRUE)
        gl.glDisable(gl.GL_BLEND)
        if not prueba_prof:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def consultar_humedad(self):
        return {
            "minima": float(self.humedad.min()),
            "maxima": float(self.humedad.max()),
            "promedio": float(self.humedad.mean()),
        }

    def cerrar(self):
        if self.textura_humedad is not None:
            self.textura_humedad.borrar()
            self.textura_humedad = None
        if self._gl is not None and self.programa is not None:
            self._gl.glDeleteProgram(self.programa)
            self.programa = None
        if self._gl is not None and self._prog_agua is not None:
            self._gl.glDeleteProgram(self._prog_agua)
            self._gl.glDeleteVertexArrays(1, [self._vao_agua])
            self._gl.glDeleteBuffers(2, [self._vbo_agua, self._ebo_agua])
            self._prog_agua = None