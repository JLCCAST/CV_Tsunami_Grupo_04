"""Cámara orbital, pila de matrices propia y arrastre de objetos/fragmentos (Rol 3)."""
import numpy as np

import config as cfg
from camara3d import CamaraOrbital
from campos import celda_de_metros
from modulos.base import Modulo


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

        # Auto principal en zona costera (x cercano a la playa donde entra la ola)
        self.pos_auto = np.array([180.0, 0.0, 310.0], dtype=np.float32)
        self.rot_auto = 0.0
        self.tamano_auto = np.array([4.5, 2.0, 8.0], dtype=np.float32)

        # Generación reproducible de piezas/escombros (20 a 50 según guía S2)
        rng = np.random.default_rng(cfg.SEMILLA)
        self.piezas_pos = np.zeros((num_piezas, 3), dtype=np.float32)
        self.piezas_pos[:, 0] = rng.uniform(140.0, 320.0, size=num_piezas)  # x en zona de impacto
        self.piezas_pos[:, 2] = rng.uniform(80.0, 560.0, size=num_piezas)   # z a lo largo de la costa
        self.piezas_rot = rng.uniform(0.0, 2.0 * np.pi, size=num_piezas).astype(np.float32)
        self.piezas_escala = rng.uniform(1.2, 3.2, size=(num_piezas, 3)).astype(np.float32)
        self.piezas_escala[:, 1] *= 0.5  # piezas más planas tipo tablas/escombros

        # Recursos OpenGL
        self.vao = None
        self.vbo = None
        self.shader_program = None

    def _ubicar_en_terreno(self, campos):
        """Ajusta la altura inicial Y de auto y piezas sobre el terreno H."""
        ix, iz = celda_de_metros(float(self.pos_auto[0]), float(self.pos_auto[2]))
        self.pos_auto[1] = campos.H[iz, ix] + self.tamano_auto[1] * 0.5

        for i in range(self.num_piezas):
            px, pz = float(self.piezas_pos[i, 0]), float(self.piezas_pos[i, 2])
            pix, piz = celda_de_metros(px, pz)
            self.piezas_pos[i, 1] = campos.H[piz, pix] + self.piezas_escala[i, 1] * 0.5

    def iniciar(self, campos, ctx):
        """Posiciona objetos sobre el terreno e inicializa OpenGL si hay ventana."""
        if campos is not None:
            self._ubicar_en_terreno(campos)

        if ctx is None:
            return

        try:
            from OpenGL import GL as gl
        except ImportError:
            return

        # Vértices de un cubo simple 3D
        vertices = np.array([
            # x,    y,    z
            -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5,  0.5, -0.5,
             0.5,  0.5, -0.5,  -0.5,  0.5, -0.5,  -0.5, -0.5, -0.5,

            -0.5, -0.5,  0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,
             0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,  -0.5, -0.5,  0.5,

            -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,  -0.5, -0.5, -0.5,
            -0.5, -0.5, -0.5,  -0.5, -0.5,  0.5,  -0.5,  0.5,  0.5,

             0.5,  0.5,  0.5,   0.5,  0.5, -0.5,   0.5, -0.5, -0.5,
             0.5, -0.5, -0.5,   0.5, -0.5,  0.5,   0.5,  0.5,  0.5,

            -0.5, -0.5, -0.5,   0.5, -0.5, -0.5,   0.5, -0.5,  0.5,
             0.5, -0.5,  0.5,  -0.5, -0.5,  0.5,  -0.5, -0.5, -0.5,

            -0.5,  0.5, -0.5,   0.5,  0.5, -0.5,   0.5,  0.5,  0.5,
             0.5,  0.5,  0.5,  -0.5,  0.5,  0.5,  -0.5,  0.5, -0.5,
        ], dtype=np.float32)

        # Shaders mínimos para visualizar el objeto de prueba
        vert_code = """
        #version 330 core
        layout(location = 0) in vec3 aPos;
        uniform mat4 uMVP;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
        """
        frag_code = """
        #version 330 core
        uniform vec3 uColor;
        out vec4 FragColor;
        void main() {
            FragColor = vec4(uColor, 1.0);
        }
        """

        vert = gl.glCreateShader(gl.GL_VERTEX_SHADER)
        gl.glShaderSource(vert, vert_code)
        gl.glCompileShader(vert)

        frag = gl.glCreateShader(gl.GL_FRAGMENT_SHADER)
        gl.glShaderSource(frag, frag_code)
        gl.glCompileShader(frag)

        self.shader_program = gl.glCreateProgram()
        gl.glAttachShader(self.shader_program, vert)
        gl.glAttachShader(self.shader_program, frag)
        gl.glLinkProgram(self.shader_program)

        gl.glDeleteShader(vert)
        gl.glDeleteShader(frag)

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW)

        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 3 * 4, None)
        gl.glEnableVertexAttribArray(0)
        gl.glBindVertexArray(0)

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

            # Flotación sobre la superficie del agua (H + W)
            pos[1] = h_val + w_val * 0.75 + alto * 0.35
        else:
            # En seco o agua somera, reposa sobre el terreno H
            pos[1] = h_val + alto * 0.5

        return pos, rot

    def actualizar(self, campos, dt: float):
        """Actualiza el auto y las 20-50 piezas según W, U, V."""
        if dt <= 0.0:
            return

        self.pos_auto, self.rot_auto = self._mover_cuerpo(
            campos, self.pos_auto, self.rot_auto,
            float(self.tamano_auto[1]), cfg.FACTOR_ARRASTRE_AUTO, dt
        )

        for i in range(self.num_piezas):
            self.piezas_pos[i], self.piezas_rot[i] = self._mover_cuerpo(
                campos, self.piezas_pos[i], float(self.piezas_rot[i]),
                float(self.piezas_escala[i, 1]), cfg.FACTOR_ARRASTRE_PIEZA, dt
            )

    def _dibujar_caja(self, gl, vp, loc_mvp, loc_color, pos, rot, escala, color):
        self.stack.push()
        self.stack.translate(float(pos[0]), float(pos[1]), float(pos[2]))
        self.stack.rotate_y(float(rot))
        self.stack.scale(float(escala[0]), float(escala[1]), float(escala[2]))

        mvp = vp @ self.stack.top()
        gl.glUniformMatrix4fv(loc_mvp, 1, gl.GL_TRUE, mvp)
        gl.glUniform3f(loc_color, *color)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
        self.stack.pop()

    def dibujar(self, campos, ctx):
        """Dibuja el auto y todas las piezas reutilizando el VAO y la pila de matrices."""
        if not hasattr(ctx, "vp") or ctx.vp is None or self.shader_program is None:
            return

        try:
            from OpenGL import GL as gl
        except ImportError:
            return

        gl.glUseProgram(self.shader_program)
        loc_mvp = gl.glGetUniformLocation(self.shader_program, "uMVP")
        loc_color = gl.glGetUniformLocation(self.shader_program, "uColor")
        gl.glBindVertexArray(self.vao)

        # 1. Dibujar auto (rojo intenso)
        self._dibujar_caja(
            gl, ctx.vp, loc_mvp, loc_color,
            self.pos_auto, self.rot_auto, self.tamano_auto, (0.90, 0.18, 0.18)
        )

        # 2. Dibujar piezas/escombros (tono madera/concreto)
        for i in range(self.num_piezas):
            self._dibujar_caja(
                gl, ctx.vp, loc_mvp, loc_color,
                self.piezas_pos[i], self.piezas_rot[i], self.piezas_escala[i], (0.65, 0.52, 0.36)
            )

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