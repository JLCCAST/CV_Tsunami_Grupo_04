"""Rol 5: iluminación, color y sombras.

Semana 1: carga shaders/iluminacion.glsl (biblioteca de funciones: phong(), sombra(),
tinte_alarma(), rampa_color()...), mantiene el estado del sol (hora del día) y lo publica
en el contexto para que el Rol 4 lo use en su shader de materiales:

    ctx.sol_dir      np.float32 [3]  vector unitario HACIA el sol, en mundo (y = arriba)
    ctx.sol_color    np.float32 [3]  color de la luz (ya multiplicado por la intensidad)
    ctx.hora         float           hora del día, 0 a 24
    ctx.glsl_iluminacion  str        texto de iluminacion.glsl (para concatenar)

Cómo lo usa el Rol 4 (en su shader de materiales):

    from modulos.iluminacion import armar_fragment
    fuente_fs = armar_fragment(CUERPO_FRAGMENT)   # pone '#version 330 core' + iluminacion.glsl + su cuerpo

Los uniforms recomendados están en Iluminacion.uniforms(campos).
"""
import math
from pathlib import Path

import numpy as np

import config as cfg
from modulos.base import Modulo

RUTA_GLSL = Path(__file__).resolve().parent.parent / "shaders" / "iluminacion.glsl"

# Constantes del sol. Si el grupo lo aprueba, moverlas a config.py (regla: una sola fuente).
HORA_INICIAL = getattr(cfg, "HORA_INICIAL", 11.0)     # hora del día al arrancar
CICLO_DIA_S = getattr(cfg, "CICLO_DIA_S", 0.0)        # segundos reales por día; 0 = sol fijo
COLOR_SOL = (1.0, 0.96, 0.88)                         # luz cálida de día


def leer_glsl(ruta=RUTA_GLSL) -> str:
    return Path(ruta).read_text(encoding="utf-8")


def armar_fragment(cuerpo: str, ruta=RUTA_GLSL) -> str:
    """'#version 330 core' + iluminacion.glsl + cuerpo del fragment shader (sin #version)."""
    assert "#version" not in cuerpo, "el cuerpo no debe traer #version: ya lo pone armar_fragment()"
    return "#version 330 core\n" + leer_glsl(ruta) + "\n" + cuerpo


def direccion_sol(hora: float) -> np.ndarray:
    """Espejo en Python de direccion_sol() de GLSL: vector unitario hacia el sol."""
    ang = (hora - 6.0) / 12.0 * math.pi
    v = np.array([math.cos(ang), math.sin(ang), 0.35], np.float32)
    return v / np.linalg.norm(v)


def intensidad_sol(direccion) -> float:
    """Espejo de intensidad_sol() de GLSL: 0 con el sol bajo el horizonte, 1 a pleno día."""
    return float(np.clip(direccion[1] * 3.0, 0.0, 1.0))


class Iluminacion(Modulo):
    nombre = "iluminacion (rol 5)"

    def __init__(self, hora: float = HORA_INICIAL, ciclo_dia_s: float = CICLO_DIA_S):
        self.hora = float(hora)
        self.ciclo_dia_s = float(ciclo_dia_s)
        self.ctx = None
        self.sol_dir = direccion_sol(self.hora)
        self.sol_color = np.array(COLOR_SOL, np.float32) * intensidad_sol(self.sol_dir)

    def iniciar(self, campos, ctx):
        self.ctx = ctx
        self._publicar()

    def actualizar(self, campos, dt: float):
        if self.ciclo_dia_s > 0.0:
            self.hora = (self.hora + 24.0 * dt / self.ciclo_dia_s) % 24.0
        self.sol_dir = direccion_sol(self.hora)
        self.sol_color = np.array(COLOR_SOL, np.float32) * intensidad_sol(self.sol_dir)
        self._publicar()

    def _publicar(self):
        """Deja el estado del sol en el contexto (los módulos lo leen en dibujar())."""
        if self.ctx is None:
            return
        self.ctx.sol_dir = self.sol_dir
        self.ctx.sol_color = self.sol_color
        self.ctx.hora = self.hora
        self.ctx.glsl_iluminacion = leer_glsl()

    def uniforms(self, campos) -> dict:
        """Valores listos para subir al shader de materiales (nombres con prefijo u, como en terreno.py)."""
        return {
            "uSolDir": self.sol_dir,
            "uSolColor": self.sol_color,
            "uHora": self.hora,
            "uAlarma": float(campos.alarma),
        }
