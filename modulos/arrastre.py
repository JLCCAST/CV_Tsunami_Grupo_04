"""Cámara orbital, pila de matrices propia y arrastre de objetos/fragmentos (Rol 3)."""
import numpy as np
from OpenGL import GL as gl

import config as cfg
from camara3d import CamaraOrbital
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

    def __init__(self):
        super().__init__()
        self.camara = CamaraOrbital()
        self.stack = PilaMatrices()

        # Objeto de prueba (Auto / Caja en coordenadas del mundo)
        self.pos_auto = np.array([100.0, 2.0, 100.0], dtype=np.float32)
        self.tamano_auto = np.array([6.0, 3.0, 10.0], dtype=np.float32)

        # Buffers de renderizado OpenGL para el objeto dummy
        self.vao = None
        self.vbo = None
        self.shader_program = None

    def iniciar(self, campos, ctx):
        """Inicializa los buffers OpenGL para dibujar el objeto de prueba en 3D."""
        if ctx is None:
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
        out vec4 FragColor;
        void main() {
            FragColor = vec4(0.9, 0.2, 0.2, 1.0); // Red auto dummy
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

    def actualizar(self, campos, dt: float):
        """Muestra U y V desde 'campos' y actualiza la posición del auto de prueba."""
        ix = int(np.clip(self.pos_auto[0] / cfg.CELL_M, 0, cfg.GRID - 1))
        iz = int(np.clip(self.pos_auto[2] / cfg.CELL_M, 0, cfg.GRID - 1))

        # Muestreo del flujo de velocidad (Contrato)
        u_val = campos.U[iz, ix]
        v_val = campos.V[iz, ix]

        # Traslación progresiva impulsada por el agua simulada/falsa
        self.pos_auto[0] += u_val * dt
        self.pos_auto[2] += v_val * dt

    def dibujar(self, campos, ctx):
        """Aplica transformaciones de la pila de matrices y renderiza el objeto dummy."""
        if not hasattr(ctx, "vp") or ctx.vp is None or self.shader_program is None:
            return

        self.stack.push()
        # Posicionar y escalar la caja de prueba usando nuestra PilaMatrices
        self.stack.translate(self.pos_auto[0], self.pos_auto[1], self.pos_auto[2])
        self.stack.scale(self.tamano_auto[0], self.tamano_auto[1], self.tamano_auto[2])

        matriz_modelo = self.stack.top()
        mvp = ctx.vp @ matriz_modelo

        gl.glUseProgram(self.shader_program)
        loc_mvp = gl.glGetUniformLocation(self.shader_program, "uMVP")
        gl.glUniformMatrix4fv(loc_mvp, 1, gl.GL_TRUE, mvp)

        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, 36)
        gl.glBindVertexArray(0)
        gl.glUseProgram(0)

        self.stack.pop()

    def cerrar(self):
        """Libera los recursos de OpenGL."""
        if self.vao:
            gl.glDeleteVertexArrays(1, [self.vao])
        if self.vbo:
            gl.glDeleteBuffers(1, [self.vbo])
        if self.shader_program:
            gl.glDeleteProgram(self.shader_program)