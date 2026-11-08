"""Panel lateral del HUD (Rol 1): alarma, dos medidores, gráfica de nivel y mini-mapa.

Solo LEE los campos del contrato (W, U, V, H, alarma, casas, dano). No escribe nada.
Dibuja sobre un lienzo NumPy; main.py lo sube como textura a unos 10 Hz y solo
cuando cambian los valores que se muestran.

Primeros: valores falsos. La gráfica recorta por clamp; luego se reemplaza
por recorte real de líneas contra la ventana de la gráfica (Cohen-Sutherland o Liang-Barsky).
"""
from collections import deque
import time

import numpy as np

import config as cfg
from agua.alarma import nivel_max_en_tierra
from hud.lienzo import Lienzo
from vista_mapa import imagen_riesgo, rampa_riesgo

FONDO = (24, 28, 38, 255)
TARJETA = (36, 42, 56, 255)
TEXTO = (235, 238, 245)
TEXTO_SUAVE = (150, 158, 175)
PISTA = (62, 70, 88)
AVISO = (241, 196, 15)

NIVEL_MAX_GRAFICA_M = 6.0
MUESTRAS_GRAFICA = 120            # a 2 muestras por segundo = últimos 60 s
PERIODO_MUESTRA_S = 0.5


def velocidad_max_en_tierra(campos) -> float:
    mojado = (campos.W > cfg.UMBRAL_MOJADO_M) & (campos.H >= cfg.NIVEL_MAR_M)
    if not mojado.any():
        return 0.0
    return float(np.hypot(campos.U, campos.V)[mojado].max())


def _color(frac):
    return tuple(int(v) for v in rampa_riesgo(frac))


class PanelHUD:
    def __init__(self, ancho: int = cfg.HUD_ANCHO, alto: int = cfg.VENTANA_ALTO, datos_falsos: bool = True):
        self.lienzo = Lienzo(ancho, alto)
        self.datos_falsos = datos_falsos
        self.historial = deque(maxlen=MUESTRAS_GRAFICA)
        self._t_muestra = -1e9
        self._firma = None
        self.ultimo_ms = 0.0          # milisegundos que tardó el último dibujo
        self.ms_agua = None           # ms por paso de la simulación del agua (lo pone main.py)

    # --- cuándo hace falta redibujar ----------------------------------------
    def _valores(self, campos):
        return nivel_max_en_tierra(campos), velocidad_max_en_tierra(campos), campos.alarma

    def necesita_redibujo(self, campos) -> bool:
        nivel, vel, alarma = self._valores(campos)
        mojadas = int((campos.W > cfg.UMBRAL_MOJADO_M).sum())
        firma = (round(nivel, 2), round(vel, 1), round(alarma, 2), mojadas, int(campos.t))
        return firma != self._firma

    # --- dibujo ---------------------------------------------------------------
    def dibujar(self, campos) -> np.ndarray:
        inicio = time.perf_counter()
        nivel, vel, alarma = self._valores(campos)
        self._firma = (round(nivel, 2), round(vel, 1), round(alarma, 2),
                       int((campos.W > cfg.UMBRAL_MOJADO_M).sum()), int(campos.t))
        self._actualizar_historial(campos.t, nivel)

        lz = self.lienzo
        lz.limpiar(FONDO)
        lz.texto(16, 10, "MONITOREO DE TSUNAMI", TEXTO, 17, negrita=True)
        self._barra_alarma(lz, alarma)
        self._medidor(lz, 85, 172, 46, nivel, NIVEL_MAX_GRAFICA_M, "Nivel de agua", "m")
        self._medidor(lz, 235, 172, 46, vel, 12.0, "Velocidad", "m/s")
        self._grafica(lz)
        self._minimapa(lz, campos)
        lz.texto(16, 684, f"t = {campos.t:6.1f} s", TEXTO, 13)
        if self.datos_falsos:
            lz.texto(304, 684, "DATOS DE PRUEBA", AVISO, 12, negrita=True, ancla="der")
        elif self.ms_agua is not None:
            lz.texto(304, 685, f"agua: {self.ms_agua:.2f} ms/paso", TEXTO_SUAVE, 12, ancla="der")
        self.ultimo_ms = (time.perf_counter() - inicio) * 1000.0
        return lz.px

    # --- piezas -----------------------------------------------------------------
    def _actualizar_historial(self, t, nivel):
        if t < self._t_muestra:                       # se reinició la simulación
            self.historial.clear()
            self._t_muestra = -1e9
        if t - self._t_muestra >= PERIODO_MUESTRA_S:
            self.historial.append(nivel)
            self._t_muestra = t

    def _barra_alarma(self, lz, alarma):
        lz.texto(16, 40, "ALARMA", TEXTO_SUAVE, 12, negrita=True)
        x, y, ancho, alto = 16, 58, 288, 24
        lz.rect_relleno(x, y, ancho, alto, PISTA)
        lz.rect_relleno(x, y, int(ancho * alarma), alto, _color(alarma))
        lz.rect_borde(x, y, ancho, alto, TEXTO_SUAVE)
        estado = "CALMA" if alarma < 0.34 else ("PRECAUCIÓN" if alarma < 0.67 else "CRÍTICA")
        lz.texto(x + ancho // 2, y + 3, f"{estado}  {alarma * 100:.0f}%", (10, 10, 10), 14, negrita=True, ancla="centro")

    def _medidor(self, lz, cx, cy, r, valor, vmax, etiqueta, unidad):
        frac = float(np.clip(valor / vmax, 0.0, 1.0))
        lz.rect_relleno(cx - 70, cy - r - 12, 140, 2 * r + 44, TARJETA)
        lz.arco(cx, cy, r, PISTA, 225, -45, grosor=7)                       # pista
        if frac > 0:
            lz.arco(cx, cy, r, _color(frac), 225, 225 - 270 * frac, grosor=7)   # tramo lleno
        ang = np.radians(225 - 270 * frac)
        lz.linea(cx, cy, cx + (r - 12) * np.cos(ang), cy - (r - 12) * np.sin(ang), TEXTO, 2)   # aguja
        lz.circulo(cx, cy, 4, TEXTO, 3)
        lz.texto(cx, cy + 14, f"{valor:.1f}", TEXTO, 20, negrita=True, ancla="centro")
        lz.texto(cx, cy + 40, unidad, TEXTO_SUAVE, 12, ancla="centro")
        lz.texto(cx, cy + r + 14, etiqueta, TEXTO, 13, negrita=True, ancla="centro")

    def _grafica(self, lz):
        lz.texto(16, 258, "Nivel de agua en tierra firme, últimos 60 s", TEXTO_SUAVE, 12, negrita=True)
        x0, y0, ancho, alto = 44, 278, 256, 96          # recuadro de la gráfica
        x1, y1 = x0 + ancho, y0 + alto
        lz.rect_relleno(x0, y0, ancho, alto, TARJETA)
        lz.linea(x0, y0, x0, y1, TEXTO_SUAVE)            # eje y
        lz.linea(x0, y1, x1, y1, TEXTO_SUAVE)            # eje x
        for v in (0, 3, 6):                              # marcas del eje y
            y = y1 - int(v / NIVEL_MAX_GRAFICA_M * alto)
            lz.linea(x0 - 4, y, x0, y, TEXTO_SUAVE)
            lz.texto(x0 - 7, y - 7, f"{v}", TEXTO_SUAVE, 11, ancla="der")
        y_crit = y1 - int(cfg.NIVEL_CRITICO_M / NIVEL_MAX_GRAFICA_M * alto)    # nivel crítico (punteada)
        for xs in range(x0 + 2, x1, 10):
            lz.linea(xs, y_crit, min(xs + 5, x1 - 1), y_crit, _color(1.0))
        if len(self.historial) >= 2:
            paso = ancho / (MUESTRAS_GRAFICA - 1)
            ys = np.clip(y1 - np.array(self.historial) / NIVEL_MAX_GRAFICA_M * alto, y0, y1)   # clamp (semana 3: recorte real)
            puntos = [(x0 + i * paso, ys[i]) for i in range(len(ys))]
            lz.polilinea(puntos, (90, 190, 255), 2)

    def _minimapa(self, lz, campos):
        lz.texto(16, 388, "Mapa de riesgo  (mar a la izquierda)", TEXTO_SUAVE, 12, negrita=True)
        mapa = imagen_riesgo(campos)
        factor = max(1, 256 // cfg.GRID)
        mapa = np.repeat(np.repeat(mapa, factor, axis=0), factor, axis=1)   # 128 -> 256 px
        x, y = (lz.ancho - mapa.shape[1]) // 2, 410
        lz.pegar_rgb(x, y, mapa)
        lz.rect_borde(x - 1, y - 1, mapa.shape[1] + 2, mapa.shape[0] + 2, TEXTO_SUAVE)
