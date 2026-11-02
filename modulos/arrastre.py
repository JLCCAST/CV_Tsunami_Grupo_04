"""Cámara, pila de matrices propia, arrastre de objetos y fragmentos.

Primero cámara orbital (rotar y zoom), pila de matrices propia y un objeto de
prueba que se mueve con un campo de velocidad falso (campos.U, campos.V).
"""
from modulos.base import Modulo


class Arrastre(Modulo):
    nombre = "camara y arrastre (rol 3)"
