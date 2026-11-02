"""Lienzo 2D sobre un arreglo NumPy.

Las FORMAS (líneas, círculos, rectángulos) se dibujan con algoritmos propios:
  * líneas  -> algoritmo de Bresenham
  * círculos y arcos -> algoritmo del punto medio
No se usa ninguna función de dibujo de OpenGL ni de Pillow para ellas.
Pillow se usa solo para rasterizar el TEXTO de las etiquetas.

Convención: (0, 0) es la esquina superior izquierda, y crece hacia abajo.
El arreglo `px` es uint8 [alto, ancho, 4] (RGBA) y se sube tal cual como textura.
"""
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_RUTAS_FUENTE = {
    False: ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "DejaVuSans.ttf", "arial.ttf", "Arial.ttf"],
    True: ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf", "arialbd.ttf", "Arial Bold.ttf"],
}
_cache_fuentes = {}


def _fuente(tam: int, negrita: bool):
    clave = (tam, negrita)
    if clave not in _cache_fuentes:
        fuente = None
        for ruta in _RUTAS_FUENTE[negrita]:
            try:
                fuente = ImageFont.truetype(ruta, tam)
                break
            except OSError:
                continue
        if fuente is None:
            fuente = ImageFont.load_default(tam)
        _cache_fuentes[clave] = fuente
    return _cache_fuentes[clave]


# ---------------------------------------------------------------------------
# Algoritmos (funciones puras, fáciles de probar)
# ---------------------------------------------------------------------------
def puntos_bresenham(x0: int, y0: int, x1: int, y1: int) -> list:
    """Píxeles de la línea (x0, y0)-(x1, y1) con Bresenham, válido en los 8 octantes."""
    puntos = []
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        puntos.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy
    return puntos


def puntos_circulo_punto_medio(r: int) -> list:
    """Desplazamientos (dx, dy) del círculo de radio r centrado en (0, 0), por punto medio.

    Se calcula un octante y se replica por simetría en los 8.
    """
    puntos = []
    x, y = 0, int(r)
    d = 1 - int(r)
    while x <= y:
        for px, py in ((x, y), (y, x), (-x, y), (-y, x), (x, -y), (y, -x), (-x, -y), (-y, -x)):
            puntos.append((px, py))
        if d < 0:
            d += 2 * x + 3
        else:
            d += 2 * (x - y) + 5
            y -= 1
        x += 1
    return puntos


# ---------------------------------------------------------------------------
# Lienzo
# ---------------------------------------------------------------------------
class Lienzo:
    def __init__(self, ancho: int, alto: int):
        self.ancho = ancho
        self.alto = alto
        self.px = np.zeros((alto, ancho, 4), np.uint8)

    # --- básicos -----------------------------------------------------------
    def limpiar(self, color=(0, 0, 0, 255)):
        self.px[:, :] = _rgba(color)

    def _estampar(self, x: int, y: int, color, grosor: int):
        """Pinta un cuadrado de `grosor` píxeles centrado en (x, y), recortado al lienzo."""
        mitad = grosor // 2
        x0, y0 = x - mitad, y - mitad
        xa, ya = max(x0, 0), max(y0, 0)
        xb, yb = min(x0 + grosor, self.ancho), min(y0 + grosor, self.alto)
        if xa < xb and ya < yb:
            self.px[ya:yb, xa:xb] = color

    def pixel(self, x: int, y: int, color):
        if 0 <= x < self.ancho and 0 <= y < self.alto:
            self.px[y, x] = _rgba(color)

    # --- primitivas con algoritmos propios -----------------------------------
    def linea(self, x0, y0, x1, y1, color, grosor: int = 1):
        c = _rgba(color)
        for x, y in puntos_bresenham(int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))):
            self._estampar(x, y, c, grosor)

    def polilinea(self, puntos, color, grosor: int = 1):
        for (xa, ya), (xb, yb) in zip(puntos[:-1], puntos[1:]):
            self.linea(xa, ya, xb, yb, color, grosor)

    def circulo(self, cx, cy, r, color, grosor: int = 1):
        c = _rgba(color)
        cx, cy = int(round(cx)), int(round(cy))
        for dx, dy in puntos_circulo_punto_medio(int(round(r))):
            self._estampar(cx + dx, cy + dy, c, grosor)

    def arco(self, cx, cy, r, color, ang_desde: float, ang_hasta: float, grosor: int = 1):
        """Arco de círculo (punto medio) entre dos ángulos en grados.

        0° es la derecha y los ángulos crecen en sentido antihorario sobre la pantalla.
        Se dibuja desde `ang_desde` hasta `ang_hasta` en sentido horario (ang_desde > ang_hasta).
        """
        c = _rgba(color)
        cx, cy = int(round(cx)), int(round(cy))
        largo = ang_desde - ang_hasta
        for dx, dy in puntos_circulo_punto_medio(int(round(r))):
            ang = math.degrees(math.atan2(-dy, dx))          # (-180, 180]
            while ang > ang_desde:
                ang -= 360.0
            while ang < ang_desde - 360.0:
                ang += 360.0
            # recorrido horario desde ang_desde hasta este punto: ang_desde - ang
            if ang_desde - ang <= largo + 1e-9:
                self._estampar(cx + dx, cy + dy, c, grosor)

    def rect_relleno(self, x, y, ancho, alto, color):
        xa, ya = max(int(x), 0), max(int(y), 0)
        xb, yb = min(int(x + ancho), self.ancho), min(int(y + alto), self.alto)
        if xa < xb and ya < yb:
            self.px[ya:yb, xa:xb] = _rgba(color)

    def rect_borde(self, x, y, ancho, alto, color, grosor: int = 1):
        x1, y1 = x + ancho - 1, y + alto - 1
        self.linea(x, y, x1, y, color, grosor)
        self.linea(x1, y, x1, y1, color, grosor)
        self.linea(x1, y1, x, y1, color, grosor)
        self.linea(x, y1, x, y, color, grosor)

    # --- imágenes y texto ---------------------------------------------------
    def pegar_rgb(self, x: int, y: int, imagen_rgb: np.ndarray):
        """Copia una imagen RGB uint8 [h, w, 3] con su esquina superior izquierda en (x, y)."""
        h, w = imagen_rgb.shape[:2]
        xa, ya = max(x, 0), max(y, 0)
        xb, yb = min(x + w, self.ancho), min(y + h, self.alto)
        if xa >= xb or ya >= yb:
            return
        self.px[ya:yb, xa:xb, :3] = imagen_rgb[ya - y:yb - y, xa - x:xb - x]
        self.px[ya:yb, xa:xb, 3] = 255

    def texto(self, x, y, cadena: str, color=(255, 255, 255), tam: int = 14,
              negrita: bool = False, ancla: str = "izq") -> int:
        """Escribe texto (rasterizado con Pillow). `ancla`: 'izq', 'centro' o 'der' sobre x.

        Devuelve el ancho en píxeles del texto escrito.
        """
        fuente = _fuente(tam, negrita)
        caja = fuente.getbbox(cadena)
        w, h = caja[2] + 2, caja[3] + 2
        mascara = Image.new("L", (w, h), 0)
        ImageDraw.Draw(mascara).text((0, 0), cadena, font=fuente, fill=255)
        m = np.asarray(mascara, dtype=np.float32) / 255.0
        if ancla == "centro":
            x = x - w // 2
        elif ancla == "der":
            x = x - w
        x, y = int(x), int(y)
        xa, ya = max(x, 0), max(y, 0)
        xb, yb = min(x + w, self.ancho), min(y + h, self.alto)
        if xa < xb and ya < yb:
            m = m[ya - y:yb - y, xa - x:xb - x, None]
            region = self.px[ya:yb, xa:xb, :3].astype(np.float32)
            nuevo = region * (1.0 - m) + np.array(color[:3], np.float32) * m
            self.px[ya:yb, xa:xb, :3] = nuevo.astype(np.uint8)
        return w


def _rgba(color):
    if len(color) == 3:
        return (color[0], color[1], color[2], 255)
    return tuple(color)

