"""Shader de materiales, texturas, transparencia del agua y efecto mojado.

Primero shader básico con texturas, plano de agua translúcido con el orden de
dibujo correcto (opaco primero, agua después con escritura de profundidad
desactivada) y humedad falsa. Tus archivos GLSL van en shaders/.
"""
from modulos.base import Modulo


class Materiales(Modulo):
    nombre = "materiales (rol 4)"
