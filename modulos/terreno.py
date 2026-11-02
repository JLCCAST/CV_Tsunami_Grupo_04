"""Terreno, casas, máscara de obstáculos y estados de daño.

Primero terreno simple publicado como H, casas como cajas (2 a 4 celdas),
máscara `campos.obstaculo`. Mientras no esté listo, agua/datos_falsos.py rellena H.
"""
from modulos.base import Modulo


class Terreno(Modulo):
    nombre = "terreno (rol 2)"
