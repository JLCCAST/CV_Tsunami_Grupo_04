"""Genera capturas PNG de la ventana (vista de mapa + HUD) SIN abrir ventana ni OpenGL.

Sirve para ver el HUD en cualquier computadora y para medir cuánto tarda el dibujo.
Uso (desde la carpeta del proyecto):
    python tools/captura_hud.py                 # guarda capturas/ventana_t*.png
    python tools/captura_hud.py --tiempos 5 20  # solo esos instantes (segundos)
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from agua.datos_falsos import DatosFalsos
from campos import Campos
from hud.panel import PanelHUD
from vista_mapa import imagen_mapa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiempos", type=float, nargs="*", default=[0.0, 8.0, 14.0, 22.0])
    ap.add_argument("--salida", default="capturas")
    args = ap.parse_args()
    os.makedirs(args.salida, exist_ok=True)

    campos = Campos.vacios()
    falsos = DatosFalsos(campos)
    panel = PanelHUD()
    dt = 0.1
    t = 0.0
    pendientes = sorted(args.tiempos)
    tiempos_dibujo = []
    while pendientes:
        falsos.actualizar(t)
        campos.validar()
        if panel.necesita_redibujo(campos):
            panel.dibujar(campos)
            tiempos_dibujo.append(panel.ultimo_ms)
        if t + 1e-9 >= pendientes[0]:
            panel.dibujar(campos)
            mapa = Image.fromarray(imagen_mapa(campos)).resize((720, 720), Image.NEAREST)
            ventana = Image.new("RGB", (cfg.VENTANA_ANCHO, cfg.VENTANA_ALTO), (20, 24, 32))
            ventana.paste(mapa, ((cfg.VENTANA_ANCHO - cfg.HUD_ANCHO - 720) // 2, 0))
            ventana.paste(Image.fromarray(panel.lienzo.px[:, :, :3]), (cfg.VENTANA_ANCHO - cfg.HUD_ANCHO, 0))
            ruta = os.path.join(args.salida, f"ventana_t{int(pendientes[0]):02d}.png")
            ventana.save(ruta)
            print("guardado", ruta, f"alarma={campos.alarma:.2f}")
            pendientes.pop(0)
        t += dt
    if tiempos_dibujo:
        print(f"HUD: {len(tiempos_dibujo)} redibujos, promedio {np.mean(tiempos_dibujo):.1f} ms, "
              f"máximo {np.max(tiempos_dibujo):.1f} ms (a 10 Hz el presupuesto es 100 ms)")


if __name__ == "__main__":
    main()
