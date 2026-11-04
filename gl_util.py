"""Utilidades mínimas de OpenGL moderno (3.3 core) para dibujar texturas 2D."""
import ctypes

import numpy as np
from OpenGL import GL as gl

VERT_QUAD = """#version 330 core
layout(location = 0) in vec2 pos;
layout(location = 1) in vec2 uv;
out vec2 vuv;
void main() {
    vuv = vec2(uv.x, 1.0 - uv.y);          // fila 0 del arreglo NumPy = arriba de la imagen
    gl_Position = vec4(pos, 0.0, 1.0);
}
"""

FRAG_QUAD = """#version 330 core
in vec2 vuv;
out vec4 color;
uniform sampler2D tex;
void main() {
    color = texture(tex, vuv);
}
"""


def compilar_programa(fuente_vertice: str, fuente_fragmento: str) -> int:
    def compilar(tipo, fuente):
        sh = gl.glCreateShader(tipo)
        gl.glShaderSource(sh, fuente)
        gl.glCompileShader(sh)
        if not gl.glGetShaderiv(sh, gl.GL_COMPILE_STATUS):
            raise RuntimeError(gl.glGetShaderInfoLog(sh).decode())
        return sh

    vs = compilar(gl.GL_VERTEX_SHADER, fuente_vertice)
    fs = compilar(gl.GL_FRAGMENT_SHADER, fuente_fragmento)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    if not gl.glGetProgramiv(prog, gl.GL_LINK_STATUS):
        raise RuntimeError(gl.glGetProgramInfoLog(prog).decode())
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    return prog


class Textura:
    """Textura 2D que se actualiza desde un arreglo NumPy uint8 [alto, ancho, canales].

    Admite 1 canal (GL_RED), 3 canales (GL_RGB) o 4 canales (GL_RGBA).
    """

    def __init__(self, ancho: int, alto: int, canales: int = 4, suave: bool = False):
        self.ancho, self.alto, self.canales = ancho, alto, canales

        formatos = {
            1: gl.GL_RED,
            3: gl.GL_RGB,
            4: gl.GL_RGBA,
        }

        if canales not in formatos:
            raise ValueError("La textura debe tener 1, 3 o 4 canales.")

        self.formato = formatos[canales]

        self.id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)

        filtro = gl.GL_LINEAR if suave else gl.GL_NEAREST
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, filtro)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, filtro)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D, 0, self.formato,
            ancho, alto, 0,
            self.formato, gl.GL_UNSIGNED_BYTE, None
        )

    def subir(self, arreglo: np.ndarray):
        assert arreglo.shape == (
            self.alto, self.ancho, self.canales
        ) and arreglo.dtype == np.uint8

        gl.glBindTexture(gl.GL_TEXTURE_2D, self.id)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        gl.glTexSubImage2D(
            gl.GL_TEXTURE_2D, 0, 0, 0,
            self.ancho, self.alto,
            self.formato, gl.GL_UNSIGNED_BYTE,
            np.ascontiguousarray(arreglo)
        )

    def borrar(self):
        gl.glDeleteTextures([self.id])


class QuadTextura:
    """Dibuja una textura llenando el viewport actual."""

    def __init__(self):
        self.programa = compilar_programa(VERT_QUAD, FRAG_QUAD)
        datos = np.array([
            -1, -1, 0, 0,
            1, -1, 1, 0,
            1, 1, 1, 1,
            -1, 1, 0, 1,
        ], np.float32)
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, datos.nbytes, datos, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, ctypes.c_void_p(8))
        gl.glBindVertexArray(0)

    def dibujar(self, textura: Textura, x: int, y: int, ancho: int, alto: int):
        """Dibuja `textura` en el rectángulo de la ventana (x, y, ancho, alto); y=0 es abajo."""
        gl.glViewport(x, y, ancho, alto)
        gl.glUseProgram(self.programa)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, textura.id)
        gl.glUniform1i(gl.glGetUniformLocation(self.programa, "tex"), 0)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_FAN, 0, 4)
        gl.glBindVertexArray(0)