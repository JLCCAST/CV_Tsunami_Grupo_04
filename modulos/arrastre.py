"""Cámara orbital, pila de matrices propia y arrastre de objetos/fragmentos (Rol 3).

Objetos que arrastra el agua:
  * un auto en la avenida de la costa;
  * piezas sueltas (cajas, tablas) en la franja de playa, que ya están antes de la ola;
  * fragmentos de cada casa que el rol 2 marca como destruida (dano = 2): aparecen en ese
    momento dentro de la casa y el agua se los lleva.
Si el agua supera UMBRAL_ARRASTRE_M el objeto flota sobre la superficie (H + W) y se mueve con
el flujo; si no, reposa sobre el terreno. Si el reloj retrocede (tecla R) todo vuelve al inicio.
"""
import numpy as np

import config as cfg
from camara3d import CamaraOrbital
from campos import celda_de_metros
from modulos.base import Modulo
from modulos.csg import MAT_ESCOMBRO, MAT_PARED_DANADA, MAT_TECHO, PALETA

DIST_AUTO_COSTA_M = 25.0          # el auto arranca en la avenida de la costa (rol 2)
PLAYA_M = (3.0, 22.0)             # franja de playa, desde la costa, donde están las piezas sueltas
FRAG_POR_CASA = 5                 # fragmentos que suelta cada casa destruida
MAX_FRAGMENTOS = 40               # decenas, no cientos
HUNDIMIENTO = 0.3                 # fracción del alto que queda bajo el agua al flotar
SECADO = 0.15                     # cuánto se seca un objeto por segundo fuera del agua

COLOR_AUTO = (0.85, 0.16, 0.16)
COLOR_PIEZA = (0.65, 0.52, 0.36)
COLORES_FRAGMENTO = [tuple(float(c) for c in PALETA[m]) for m in (MAT_PARED_DANADA, MAT_TECHO, MAT_ESCOMBRO)]

# Cubo unitario con normales (36 vértices: posición y normal).
_CARAS = [((1, 0, 0), (0, 1, 0), (0, 0, 1)), ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
          ((0, 1, 0), (0, 0, 1), (1, 0, 0)), ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
          ((0, 0, 1), (1, 0, 0), (0, 1, 0)), ((0, 0, -1), (0, 1, 0), (1, 0, 0))]


def cubo_con_normales():
    filas = []
    for n, a, b in _CARAS:
        n, a, b = (np.array(v, np.float32) for v in (n, a, b))
        c = n * 0.5
        esq = [c - a * 0.5 - b * 0.5, c + a * 0.5 - b * 0.5, c + a * 0.5 + b * 0.5, c - a * 0.5 + b * 0.5]
        for k in (0, 1, 2, 0, 2, 3):
            filas.append(np.concatenate([esq[k], n]))
    return np.asarray(filas, np.float32)


VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
uniform mat4 uMVP;
uniform mat4 uModelo;
uniform mat3 uNormal;
out vec3 vPos;
out vec3 vNormal;
void main() {
    vPos = (uModelo * vec4(aPos, 1.0)).xyz;
    vNormal = uNormal * aNormal;
    gl_Position = uMVP * vec4(aPos, 1.0);
}
"""

# Cuerpo sin '#version': armar_fragment() del rol 5 le antepone phong_mojado() y tinte_alarma().
FRAG = """
in vec3 vPos;
in vec3 vNormal;
uniform vec3 uColor;
uniform vec3 uSolDir;
uniform vec3 uSolColor;
uniform vec3 uCamPos;
uniform float uAlarma;
uniform float uHumedad;
uniform vec3 uCielo;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vNormal);
    vec3 V = normalize(uCamPos - vPos);
    vec3 L = normalize(uSolDir);
    vec3 col = phong_mojado(N, V, L, uColor, uSolColor, 0.35, 0.65, 0.15, 16.0, uHumedad, 1.0);
    col = con_niebla(tinte_alarma(col, uAlarma), factor_niebla(vPos, uCamPos), uCielo, uAlarma);
    fragColor = vec4(col, 1.0);
}
"""


class PilaMatrices:
    """Pila de matrices 4x4 manual en NumPy (sin depender de glPushMatrix/glPopMatrix)."""

    def __init__(self):
        self.pila = [np.eye(4, dtype=np.float32)]

    def top(self) -> np.ndarray:
        return self.pila[-1]

    def push(self):
        self.pila.append(self.pila[-1].copy())

    def pop(self):
        if len(self.pila) > 1:
            self.pila.pop()

    def translate(self, tx: float, ty: float, tz: float):
        T = np.eye(4, dtype=np.float32)
        T[0:3, 3] = [tx, ty, tz]
        self.pila[-1] = self.pila[-1] @ T

    def rotate_y(self, angulo_rad: float):
        c, s = np.cos(angulo_rad), np.sin(angulo_rad)
        R = np.array([
            [ c, 0, s, 0],
            [ 0, 1, 0, 0],
            [-s, 0, c, 0],
            [ 0, 0, 0, 1]
        ], dtype=np.float32)
        self.pila[-1] = self.pila[-1] @ R

    def scale(self, sx: float, sy: float, sz: float):
        S = np.diag([sx, sy, sz, 1.0]).astype(np.float32)
        self.pila[-1] = self.pila[-1] @ S


class Arrastre(Modulo):
    """Módulo principal del Rol 3: maneja la cámara 3D, transformaciones y física de arrastre."""
    nombre = "camara y arrastre (rol 3)"

    def __init__(self, num_piezas: int = cfg.CANTIDAD_PIEZAS_S2):
        super().__init__()
        self.camara = CamaraOrbital()
        self.stack = PilaMatrices()
        self.num_piezas = num_piezas

        # Auto: ancho, alto y largo de un auto real. Su x se fija en iniciar(), según la costa.
        self.pos_auto = np.array([180.0, 0.0, 310.0], dtype=np.float32)
        self.rot_auto = 0.0
        self.tamano_auto = np.array([1.9, 1.5, 4.5], dtype=np.float32)
        self.humedad_auto = 0.0

        # Piezas sueltas reproducibles (20 a 50 según la guía S2). Su x final se fija en iniciar().
        rng = np.random.default_rng(cfg.SEMILLA)
        self.piezas_pos = np.zeros((num_piezas, 3), dtype=np.float32)
        self.piezas_pos[:, 0] = rng.uniform(140.0, 320.0, size=num_piezas)
        self.piezas_pos[:, 2] = rng.uniform(80.0, 560.0, size=num_piezas)   # z a lo largo de la costa
        self.piezas_rot = rng.uniform(0.0, 2.0 * np.pi, size=num_piezas).astype(np.float32)
        self.piezas_escala = rng.uniform(1.2, 3.2, size=(num_piezas, 3)).astype(np.float32)
        self.piezas_escala[:, 1] *= 0.5  # piezas más planas tipo tablas/escombros
        self.piezas_humedad = np.zeros(num_piezas, np.float32)

        # Fragmentos de casas destruidas: aparecen durante la simulación.
        self._rng_frag = np.random.default_rng(cfg.SEMILLA + 3)
        self._vaciar_fragmentos()
        self._dano_prev = None
        self._t_prev = None
        self._inicial = None

        # Recursos OpenGL
        self.vao = None
        self.vbo = None
        self.shader_program = None
        self._u = {}

    # --- estado -----------------------------------------------------------------------
    def _vaciar_fragmentos(self):
        self.frag_pos = np.zeros((0, 3), np.float32)
        self.frag_rot = np.zeros(0, np.float32)
        self.frag_escala = np.zeros((0, 3), np.float32)
        self.frag_color = np.zeros((0, 3), np.float32)
        self.frag_humedad = np.zeros(0, np.float32)

    def _ubicar_en_terreno(self, campos):
        """Pone el auto en la avenida de la costa y las piezas en la playa, todo sobre H."""
        H = campos.H

        def costa_x(z):
            _ix, iz = celda_de_metros(0.0, z)
            tierra = H[iz] >= cfg.NIVEL_MAR_M
            return float(np.argmax(tierra)) * cfg.CELL_M if tierra.any() else 0.0

        def libre(x, z):
            for _ in range(20):                       # no empezar dentro de una casa
                ix, iz = celda_de_metros(x, z)
                if not campos.obstaculo[iz, ix]:
                    break
                x += cfg.CELL_M
            return x

        z = float(self.pos_auto[2])
        self.pos_auto[0] = libre(costa_x(z) + DIST_AUTO_COSTA_M, z)
        ix, iz = celda_de_metros(float(self.pos_auto[0]), z)
        self.pos_auto[1] = H[iz, ix] + self.tamano_auto[1] * 0.5

        lo, hi = PLAYA_M
        for i in range(self.num_piezas):
            z = float(self.piezas_pos[i, 2])
            fraccion = (float(self.piezas_pos[i, 0]) - 140.0) / 180.0      # el azar de la x original
            x = libre(costa_x(z) + lo + fraccion * (hi - lo), z)
            pix, piz = celda_de_metros(x, z)
            self.piezas_pos[i, 0] = x
            self.piezas_pos[i, 1] = H[piz, pix] + self.piezas_escala[i, 1] * 0.5

        self._inicial = (self.pos_auto.copy(), self.rot_auto, self.piezas_pos.copy(), self.piezas_rot.copy())

    def reiniciar(self):
        """Vuelve al estado del inicio (tecla R): auto y piezas en su lugar, sin fragmentos."""
        if self._inicial is None:
            return
        auto, rot, piezas, rots = self._inicial
        self.pos_auto, self.rot_auto = auto.copy(), rot
        self.piezas_pos, self.piezas_rot = piezas.copy(), rots.copy()
        self.humedad_auto = 0.0
        self.piezas_humedad[:] = 0.0
        self._vaciar_fragmentos()
        self._rng_frag = np.random.default_rng(cfg.SEMILLA + 3)
        self._dano_prev = None

    def iniciar(self, campos, ctx):
        """Posiciona objetos sobre el terreno e inicializa OpenGL si hay ventana."""
        if campos is not None:
            self._ubicar_en_terreno(campos)
            self._dano_prev = campos.dano.copy()
        if ctx is None:
            return
        try:
            from OpenGL import GL as gl
        except ImportError:
            return
        import ctypes

        from gl_util import compilar_programa
        from modulos.iluminacion import armar_fragment

        self.shader_program = compilar_programa(VERT, armar_fragment(FRAG))
        self._u = {n: gl.glGetUniformLocation(self.shader_program, n)
                   for n in ("uMVP", "uModelo", "uNormal", "uColor", "uSolDir", "uSolColor",
                             "uCamPos", "uAlarma", "uHumedad", "uCielo")}
        vertices = cubo_con_normales()
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 24, ctypes.c_void_p(12))
        gl.glEnableVertexAttribArray(1)
        gl.glBindVertexArray(0)

    # --- física -------------------------------------------------------------------------
    def _mover_cuerpo(self, campos, pos: np.ndarray, rot: float, alto: float,
                      factor_arrastre: float, dt: float) -> tuple[np.ndarray, float]:
        """Aplica física de arrastre v1 si W > UMBRAL_ARRASTRE_M."""
        ix, iz = celda_de_metros(float(pos[0]), float(pos[2]))
        w_val = float(campos.W[iz, ix])
        h_val = float(campos.H[iz, ix])

        # Si el agua supera el umbral de profundidad, se aplica fuerza de arrastre y flotación
        if w_val > cfg.UMBRAL_ARRASTRE_M:
            u_val = float(campos.U[iz, ix])
            v_val = float(campos.V[iz, ix])

            # La fuerza crece con la profundidad relativa y la velocidad del flujo
            intensidad = min(1.5, w_val / cfg.NIVEL_CRITICO_M)
            vx = u_val * factor_arrastre * (0.5 + intensidad)
            vz = v_val * factor_arrastre * (0.5 + intensidad)

            nx = float(np.clip(pos[0] + vx * dt, 0.0, cfg.WORLD_M - 1.0))
            nz = float(np.clip(pos[2] + vz * dt, 0.0, cfg.WORLD_M - 1.0))

            # Evitar atravesar casas en pie (obstaculo)
            nix, niz = celda_de_metros(nx, nz)
            if not campos.obstaculo[niz, nix]:
                pos[0] = nx
                pos[2] = nz
                h_val = float(campos.H[niz, nix])
                w_val = float(campos.W[niz, nix])

            # Giro proporcional a la rapidez del flujo
            rapidez = float(np.hypot(vx, vz))
            rot += rapidez * 0.12 * dt

        # Flota sobre la superficie del agua (H + W) con parte del cuerpo hundida; si hay poca
        # agua, reposa sobre el terreno.
        flotando = h_val + w_val - HUNDIMIENTO * alto if w_val > cfg.UMBRAL_ARRASTRE_M else -1e9
        pos[1] = max(h_val + alto * 0.5, flotando)
        return pos, rot

    @staticmethod
    def _mojar(campos, pos, humedad, dt):
        """1 dentro del agua; fuera se seca de a poco (lo usa phong_mojado al dibujar)."""
        ix, iz = celda_de_metros(float(pos[0]), float(pos[2]))
        if campos.W[iz, ix] > cfg.UMBRAL_MOJADO_M:
            return 1.0
        return float(humedad) * float(np.exp(-SECADO * dt))

    def _soltar_fragmentos(self, campos):
        """Cada casa que pasa a destruida suelta FRAG_POR_CASA fragmentos dentro de su caja."""
        dano = campos.dano
        if self._dano_prev is None or len(self._dano_prev) != len(dano):
            self._dano_prev = dano.copy()
            return
        nuevas = np.where((dano == cfg.DANO_DESTRUIDA) & (self._dano_prev != cfg.DANO_DESTRUIDA))[0]
        self._dano_prev = dano.copy()
        rng = self._rng_frag
        for i in nuevas:
            libres = MAX_FRAGMENTOS - len(self.frag_pos)
            if libres <= 0:
                break
            n = min(FRAG_POR_CASA, libres)
            ix0, iz0, ix1, iz1 = (int(v) for v in campos.casas[i])
            x = rng.uniform(ix0, ix1, n) * cfg.CELL_M
            z = rng.uniform(iz0, iz1, n) * cfg.CELL_M
            esc = np.column_stack([rng.uniform(1.0, 2.6, n), rng.uniform(0.3, 0.8, n), rng.uniform(1.0, 2.6, n)])
            celdas = [celda_de_metros(a, b) for a, b in zip(x, z)]
            y = np.array([campos.H[iz, ix] for ix, iz in celdas]) + esc[:, 1] * 0.5
            colores = np.array([COLORES_FRAGMENTO[k % len(COLORES_FRAGMENTO)] for k in range(n)])
            self.frag_pos = np.vstack([self.frag_pos, np.column_stack([x, y, z])]).astype(np.float32)
            self.frag_rot = np.concatenate([self.frag_rot, rng.uniform(0, 2 * np.pi, n)]).astype(np.float32)
            self.frag_escala = np.vstack([self.frag_escala, esc]).astype(np.float32)
            self.frag_color = np.vstack([self.frag_color, colores]).astype(np.float32)
            self.frag_humedad = np.concatenate([self.frag_humedad, np.ones(n)]).astype(np.float32)

    def actualizar(self, campos, dt: float):
        """Actualiza el auto, las piezas y los fragmentos según W, U, V."""
        if self._t_prev is not None and campos.t < self._t_prev - 1e-9:
            self.reiniciar()
        self._t_prev = campos.t
        if dt <= 0.0:
            return

        self._soltar_fragmentos(campos)

        self.pos_auto, self.rot_auto = self._mover_cuerpo(
            campos, self.pos_auto, self.rot_auto,
            float(self.tamano_auto[1]), cfg.FACTOR_ARRASTRE_AUTO, dt
        )
        self.humedad_auto = self._mojar(campos, self.pos_auto, self.humedad_auto, dt)

        for i in range(self.num_piezas):
            self.piezas_pos[i], self.piezas_rot[i] = self._mover_cuerpo(
                campos, self.piezas_pos[i], float(self.piezas_rot[i]),
                float(self.piezas_escala[i, 1]), cfg.FACTOR_ARRASTRE_PIEZA, dt
            )
            self.piezas_humedad[i] = self._mojar(campos, self.piezas_pos[i], self.piezas_humedad[i], dt)

        for i in range(len(self.frag_pos)):
            self.frag_pos[i], self.frag_rot[i] = self._mover_cuerpo(
                campos, self.frag_pos[i], float(self.frag_rot[i]),
                float(self.frag_escala[i, 1]), cfg.FACTOR_ARRASTRE_PIEZA, dt
            )
            self.frag_humedad[i] = self._mojar(campos, self.frag_pos[i], self.frag_humedad[i], dt)

    # --- dibujo -----------------------------------------------------------------------------
    def _dibujar_caja(self, gl, vp, pos, rot, escala, color, humedad):
        self.stack.push()
        self.stack.translate(float(pos[0]), float(pos[1]), float(pos[2]))
        self.stack.rotate_y(float(rot))
        self.stack.scale(float(escala[0]), float(escala[1]), float(escala[2]))

        modelo = self.stack.top()
        u = self._u
        gl.glUniformMatrix4fv(u["uMVP"], 1, gl.GL_TRUE, np.ascontiguousarray(vp @ modelo, np.float32))
        gl.glUniformMatrix4fv(u["uModelo"], 1, gl.GL_TRUE, np.ascontiguousarray(modelo, np.float32))
        normal = np.linalg.inv(modelo[:3, :3]).T          # así la escala no deforma las normales
        gl.glUniformMatrix3fv(u["uNormal"], 1, gl.GL_TRUE, np.ascontiguousarray(normal, np.float32))
        gl.glUniform3f(u["uColor"], *[float(c) for c in color])
        gl.glUniform1f(u["uHumedad"], float(humedad))
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
        self.stack.pop()

    def dibujar(self, campos, ctx):
        """Dibuja el auto, las piezas y los fragmentos con la luz del sol y el tinte de alarma."""
        if not hasattr(ctx, "vp") or ctx.vp is None or self.shader_program is None:
            return
        try:
            from OpenGL import GL as gl
        except ImportError:
            return
        from modulos.materiales import SOL_COLOR_DEFECTO, SOL_DIR_DEFECTO, _alarma

        u = self._u
        gl.glUseProgram(self.shader_program)
        gl.glUniform3f(u["uSolDir"], *[float(v) for v in getattr(ctx, "sol_dir", SOL_DIR_DEFECTO)])
        gl.glUniform3f(u["uSolColor"], *[float(v) for v in getattr(ctx, "sol_color", SOL_COLOR_DEFECTO)])
        cam = getattr(ctx, "cam_pos", None)
        cam = (cfg.WORLD_M / 2, 400.0, cfg.WORLD_M / 2) if cam is None else cam
        gl.glUniform3f(u["uCamPos"], *[float(v) for v in cam])
        gl.glUniform1f(u["uAlarma"], float(_alarma(campos)))
        gl.glUniform3f(u["uCielo"], *[float(c) for c in cfg.COLOR_CIELO])
        gl.glBindVertexArray(self.vao)

        self._dibujar_caja(gl, ctx.vp, self.pos_auto, self.rot_auto, self.tamano_auto,
                           COLOR_AUTO, self.humedad_auto)
        for i in range(self.num_piezas):
            self._dibujar_caja(gl, ctx.vp, self.piezas_pos[i], self.piezas_rot[i], self.piezas_escala[i],
                               COLOR_PIEZA, self.piezas_humedad[i])
        for i in range(len(self.frag_pos)):
            self._dibujar_caja(gl, ctx.vp, self.frag_pos[i], self.frag_rot[i], self.frag_escala[i],
                               self.frag_color[i], self.frag_humedad[i])

        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

    def cerrar(self):
        """Libera recursos OpenGL."""
        try:
            from OpenGL import GL as gl
        except ImportError:
            return

        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
        if self.shader_program:
            gl.glDeleteProgram(self.shader_program)
