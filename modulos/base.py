"""Interfaz mínima que cumple cada módulo para engancharse a main.py.

Cada integrante crea su clase en su archivo, hereda de Modulo y la agrega a la lista
MODULOS de main.py. No hace falta tocar nada más.
"""


class Modulo:
    nombre = "modulo"

    def iniciar(self, campos, ctx):
        """Se llama una vez, con el contexto OpenGL ya creado (ctx puede ser None sin ventana)."""

    def actualizar(self, campos, dt: float):
        """Se llama en cada cuadro. Lee y escribe en `campos` solo lo que te corresponde."""

    def dibujar(self, campos, ctx):
        """Se llama en cada cuadro después de actualizar. Aquí va el código OpenGL."""

    def cerrar(self):
        """Libera recursos."""
