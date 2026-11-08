"""Mide los milisegundos por paso de la simulación del agua (Rol 1), SIN abrir ventana.

Corre el evento completo del frente de ola (llegada, inundación y retiro) sobre el terreno
real del rol 2, con pasos del tamaño de un cuadro, y mide solo FrenteOla.actualizar().
Con --grid prueba otras resoluciones manteniendo el mundo de 640 m (cambia los metros por celda).

Uso (desde la carpeta del proyecto):
    python tools/medir_agua.py                                 # GRID 128 a 60 pasos por segundo
    python tools/medir_agua.py --grid 64 128 256               # comparación por resolución
    python tools/medir_agua.py --grid 128 256 --csv metricas_agua.csv --nota "semana 2"
"""
import argparse
import csv
import os
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from agua import frente_ola as fo

PRESUPUESTO_MS = 1000.0 / cfg.FPS_OBJETIVO        # tiempo de un cuadro a los fps objetivo


def medir(grid: int, pasos_por_s: float, segundos: float):
    """Devuelve el cronómetro de una corrida completa con la grilla `grid`."""
    grid_orig, cell_orig = cfg.GRID, cfg.CELL_M
    cfg.GRID, cfg.CELL_M = grid, cfg.WORLD_M / grid        # mismo mundo de 640 m
    try:
        from campos import Campos
        from modulos.terreno import Terreno
        campos = Campos.vacios()
        Terreno().iniciar(campos, None)                     # terreno real, sin OpenGL
        ola = fo.FrenteOla(campos)
        for t in np.arange(0.0, segundos, 1.0 / pasos_por_s):
            ola.actualizar(float(t))
        campos.validar()
        return ola.cronometro
    finally:
        cfg.GRID, cfg.CELL_M = grid_orig, cell_orig


def guardar_csv(ruta, fila):
    existe = Path(ruta).exists()
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if not existe:
            w.writeheader()
        w.writerow(fila)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, nargs="+", default=[cfg.GRID])
    ap.add_argument("--pasos-por-s", type=float, default=60.0, help="pasos por segundo simulado (un paso por cuadro)")
    ap.add_argument("--segundos", type=float, default=fo.T_RETIRO_S + fo.DURACION_RETIRO_S + 5.0)
    ap.add_argument("--csv", default="", help="si se indica, agrega una fila por resolución a este CSV")
    ap.add_argument("--nota", default="")
    args = ap.parse_args()

    print(f"Frente de ola: {args.segundos:.0f} s simulados a {args.pasos_por_s:.0f} pasos/s; "
          f"presupuesto de un cuadro a {cfg.FPS_OBJETIVO} fps = {PRESUPUESTO_MS:.0f} ms")
    print(f"{'GRID':>6} {'m/celda':>8} {'pasos':>7} {'media ms':>9} {'p95 ms':>8} {'máx ms':>8} {'% cuadro':>9}")
    for g in args.grid:
        c = medir(g, args.pasos_por_s, args.segundos)
        p95 = c.percentil_ms(95)
        print(f"{g:>6} {cfg.WORLD_M / g:>8.2f} {c.pasos:>7} {c.media_ms:>9.3f} {p95:>8.3f} "
              f"{c.maximo_ms:>8.3f} {100 * c.media_ms / PRESUPUESTO_MS:>8.2f}%")
        if args.csv:
            guardar_csv(args.csv, {
                "fecha": datetime.now().isoformat(timespec="seconds"),
                "equipo": platform.node(), "cpu": platform.processor(),
                "grid": g, "m_por_celda": cfg.WORLD_M / g, "pasos": c.pasos,
                "media_ms": round(c.media_ms, 4), "p95_ms": round(p95, 4), "max_ms": round(c.maximo_ms, 4),
                "nota": args.nota,
            })
    if args.csv:
        print("Guardado en", args.csv)


if __name__ == "__main__":
    main()
