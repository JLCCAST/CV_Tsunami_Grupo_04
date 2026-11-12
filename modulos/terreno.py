import ctypes

import numpy as np

import config as cfg
from modulos.base import Modulo
from modulos.casas import malla_casa, piezas_casa
from modulos.dano import ModeloDano
from modulos.terreno_gen import generar_escenario, malla_muelles, malla_terreno

MODO_SIMPLE = False

VERT = """#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;
layout(location = 2) in vec3 aColor;
uniform mat4 uVP;
uniform vec3 uOrigen;
out vec3 vNormal;
out vec3 vColor;
out float vAltura;
void main() {
    vec3 p = aPos + uOrigen;
    vNormal = aNormal;
    vColor = aColor;
    vAltura = p.y;
    gl_Position = uVP * vec4(p, 1.0);
}
"""

FRAG = """#version 330 core
in vec3 vNormal;
in vec3 vColor;
in float vAltura;
uniform vec3 uSol;
out vec4 fragColor;
void main() {
    vec3 N = normalize(vNormal);
    float lambert = max(dot(N, normalize(uSol)), 0.0);
    vec3 col = vColor * (0.35 + 0.65 * lambert);
    col = mix(col, vec3(0.10, 0.25, 0.35), clamp(-vAltura / 6.0, 0.0, 0.6));
    fragColor = vec4(col, 1.0);
}
"""


class Terreno(Modulo):
    nombre = "terreno (rol 2)"

    def __init__(self, modo_simple=MODO_SIMPLE):
        self.modo_simple = modo_simple
        self.escenario = None
        self.modelo_dano = None
        self.origenes = None
        self.destruidas_cuadro = []
        self._dano_prev = None
        self._t_prev = None
        self._gl = None
        self._piezas_gpu = {}

    def generar(self, campos):
        esc = generar_escenario(simple=self.modo_simple)
        self.escenario = esc
        campos.H[:] = esc.H
        campos.casas = esc.casas.astype(np.int32).copy()
        campos.dano = np.zeros(len(esc.casas), np.uint8)
        cx = (esc.casas[:, 0] + esc.casas[:, 2]) / 2.0 * cfg.CELL_M
        cz = (esc.casas[:, 1] + esc.casas[:, 3]) / 2.0 * cfg.CELL_M
        self.origenes = np.stack([cx, esc.base_y, cz], -1).astype(np.float32)
        self.modelo_dano = ModeloDano(esc.casas, cfg.GRID)
        self._dano_prev = None
        self._t_prev = None
        self.destruidas_cuadro = []
        self._sincronizar_obstaculos(campos)

    def _sincronizar_obstaculos(self, campos):
        clave = campos.dano.tobytes()
        if clave == self._dano_prev:
            return
        m = campos.obstaculo
        m[:] = False
        for (ix0, iz0, ix1, iz1), d in zip(campos.casas, campos.dano):
            if d < cfg.DANO_DESTRUIDA:
                m[iz0:iz1, ix0:ix1] = True
        self._dano_prev = clave

    def reiniciar_dano(self, campos):
        campos.dano[:] = cfg.DANO_INTACTA
        self.modelo_dano.reiniciar()
        self._sincronizar_obstaculos(campos)

    def iniciar(self, campos, ctx=None):
        self.generar(campos)
        if ctx is not None:
            self._subir_a_gpu()

    def actualizar(self, campos, dt):
        self.destruidas_cuadro = []
        if self.modelo_dano is None or len(campos.casas) != self.modelo_dano.n:
            return
        if self._t_prev is not None and campos.t < self._t_prev - 1e-9:
            self.reiniciar_dano(campos)
        self._t_prev = campos.t
        cambiadas = self.modelo_dano.actualizar(campos.W, campos.U, campos.V, campos.dano, float(dt))
        self.destruidas_cuadro = [i for i in cambiadas if campos.dano[i] == cfg.DANO_DESTRUIDA]
        self._sincronizar_obstaculos(campos)

    def dibujar(self, campos, ctx):
        vp = getattr(ctx, "vp", None)
        if self._gl is None or vp is None:
            return
        gl = self._gl["gl"]
        prueba_prof = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

        # Si Rol 4 publicó su material (con humedad), terreno y casas se dibujan con ese shader;
        # si no, se usa el shader propio de siempre.
        materiales = getattr(ctx, "materiales", None)
        cam_pos = getattr(ctx, "cam_pos", None)
        con_humedad = (materiales is not None and cam_pos is not None
                       and materiales.preparar_terreno(campos, ctx, vp, cam_pos))
        if not con_humedad:
            gl.glUseProgram(self._gl["prog"])
            gl.glUniformMatrix4fv(self._gl["u_vp"], 1, gl.GL_TRUE, np.ascontiguousarray(vp, np.float32))
            gl.glUniform3f(self._gl["u_sol"], -0.4, 0.8, -0.3)
            gl.glUniform3f(self._gl["u_origen"], 0.0, 0.0, 0.0)
        gl.glBindVertexArray(self._gl["vao_terreno"])
        gl.glDrawElements(gl.GL_TRIANGLES, self._gl["n_idx"], gl.GL_UNSIGNED_INT, None)

        gl.glBindVertexArray(self._gl["vao_casas"])
        rangos = self._gl["rangos"]
        for i in range(min(len(self.origenes), len(campos.dano))):
            first, count = rangos[(self.escenario.tipos[i], int(campos.dano[i]))]
            if con_humedad:
                materiales.fijar_origen(*self.origenes[i])
            else:
                gl.glUniform3f(self._gl["u_origen"], *self.origenes[i])
            gl.glDrawArrays(gl.GL_TRIANGLES, first, count)

        if self._gl["n_muelles"]:                         # los muelles ya están en coordenadas del mundo
            if con_humedad:
                materiales.fijar_origen(0.0, 0.0, 0.0)
            else:
                gl.glUniform3f(self._gl["u_origen"], 0.0, 0.0, 0.0)
            gl.glBindVertexArray(self._gl["vao_muelles"])
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, self._gl["n_muelles"])
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_CULL_FACE)
        if not prueba_prof:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def dibujar_fragmentos(self, ctx, indice=0, separacion=1.0):
        vp = getattr(ctx, "vp", None)
        if self._gl is None or vp is None:
            return
        gl = self._gl["gl"]
        if indice not in self._piezas_gpu:
            lista = []
            for p in self.piezas_de_casa(indice):
                vao, vbo, _ = self._crear_vao(gl, p["malla"])
                lista.append((vao, vbo, len(p["malla"]), p["centroide"]))
            self._piezas_gpu[indice] = lista
        prueba_prof = gl.glIsEnabled(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glUseProgram(self._gl["prog"])
        gl.glUniformMatrix4fv(self._gl["u_vp"], 1, gl.GL_TRUE, np.ascontiguousarray(vp, np.float32))
        gl.glUniform3f(self._gl["u_sol"], -0.4, 0.8, -0.3)
        base = self.origenes[indice]
        for vao, _vbo, n, cen in self._piezas_gpu[indice]:
            gl.glUniform3f(self._gl["u_origen"], *(base + np.asarray(cen, np.float32) * separacion))
            gl.glBindVertexArray(vao)
            gl.glDrawArrays(gl.GL_TRIANGLES, 0, n)
        gl.glBindVertexArray(0)
        gl.glDisable(gl.GL_CULL_FACE)
        if not prueba_prof:
            gl.glDisable(gl.GL_DEPTH_TEST)

    def cerrar(self):
        if self._gl is None:
            return
        gl = self._gl["gl"]
        bufs = [b for k in ("t", "c", "m") for b in self._gl[k][1:] if b is not None]
        for lista in self._piezas_gpu.values():
            for vao, vbo, _n, _c in lista:
                bufs.append(vbo)
                gl.glDeleteVertexArrays(1, [vao])
        gl.glDeleteVertexArrays(3, [self._gl["vao_terreno"], self._gl["vao_casas"], self._gl["vao_muelles"]])
        gl.glDeleteBuffers(len(bufs), bufs)
        gl.glDeleteProgram(self._gl["prog"])
        self._gl = None
        self._piezas_gpu = {}

    def piezas_de_casa(self, i):
        w, d, p = self.escenario.tipos[i]
        return piezas_casa(w, d, p, semilla=i)

    def _subir_a_gpu(self):
        from OpenGL import GL as gl

        from gl_util import compilar_programa

        prog = compilar_programa(VERT, FRAG)
        v, idx = malla_terreno(self.escenario.H, calles=self.escenario.mascara_calles)
        vao_t, vbo_t, ebo_t = self._crear_vao(gl, v, idx)
        muelles = (malla_muelles(self.escenario.muelles, self.escenario.H) if self.escenario.muelles
                   else np.zeros((0, 9), np.float32))
        vao_m, vbo_m, _ = self._crear_vao(gl, muelles)

        trozos, rangos, pos = [], {}, 0
        for tipo in sorted(set(self.escenario.tipos)):
            for estado in (cfg.DANO_INTACTA, cfg.DANO_DANADA, cfg.DANO_DESTRUIDA):
                m = malla_casa(*tipo, estado=estado, simple=self.modo_simple)
                rangos[(tipo, estado)] = (pos, len(m))
                trozos.append(m)
                pos += len(m)
        todo = np.concatenate(trozos) if trozos else np.zeros((0, 9), np.float32)
        vao_c, vbo_c, _ = self._crear_vao(gl, todo)

        self._gl = {
            "gl": gl,
            "prog": prog,
            "u_vp": gl.glGetUniformLocation(prog, "uVP"),
            "u_origen": gl.glGetUniformLocation(prog, "uOrigen"),
            "u_sol": gl.glGetUniformLocation(prog, "uSol"),
            "vao_terreno": vao_t,
            "vao_casas": vao_c,
            "vao_muelles": vao_m,
            "t": (vao_t, vbo_t, ebo_t),
            "c": (vao_c, vbo_c, None),
            "m": (vao_m, vbo_m, None),
            "n_idx": len(idx),
            "n_muelles": len(muelles),
            "rangos": rangos,
        }

    @staticmethod
    def _crear_vao(gl, verts, idx=None):
        verts = np.ascontiguousarray(verts, np.float32)
        vao, vbo = gl.glGenVertexArrays(1), gl.glGenBuffers(1)
        gl.glBindVertexArray(vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_STATIC_DRAW)
        for loc in range(3):
            gl.glEnableVertexAttribArray(loc)
            gl.glVertexAttribPointer(loc, 3, gl.GL_FLOAT, gl.GL_FALSE, 36, ctypes.c_void_p(loc * 12))
        ebo = None
        if idx is not None:
            idx = np.ascontiguousarray(idx, np.uint32)
            ebo = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, ebo)
            gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, idx.nbytes, idx, gl.GL_STATIC_DRAW)
        gl.glBindVertexArray(0)
        return vao, vbo, ebo