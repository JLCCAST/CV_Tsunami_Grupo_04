"""
medir_app.py - fps de la APLICACIÓN COMPLETA (todos los módulos juntos), Rol 5.

Ejecuta `main.py --3d --frames N` varias veces, lee la línea final que imprime main.py
("N cuadros en T s = X fps promedio") y agrega una fila por corrida a metricas_fps.csv,
con las mismas columnas que tools/medir_fps.py. Así el banco de pruebas (terreno falso) y la
app real quedan en la misma tabla, distinguidos por la columna `nota`.

Se ejecuta desde la raíz del repositorio:
    python tools/medir_app.py --nota "semana2 todo junto"
    python tools/medir_app.py --repeticiones 5 --frames 900 --t-inicio 20 --nota "semana2 ola en tierra"

Notas para interpretar los resultados:
  - main.py trabaja sin vsync (glfw.swap_interval(0)), así que mide lo que realmente da tu equipo.
  - main.py solo informa el PROMEDIO: las columnas ms_p95 y fps_1pct_low quedan vacías.
  - --t-inicio 20 empieza con la ola ya llegando a la costa (hay agua, humedad y alarma). Con
    --t-inicio 0 las primeras escenas son de mar en calma y salen más fps.
  - Compara solo corridas hechas en el mismo equipo y con la misma ventana.
"""
import argparse
import csv
import os
import re
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))

import config as cfg  # noqa: E402

PAT_GPU = re.compile(r"OpenGL:\s*(?P<version>.+?)\s*\|\s*(?P<gpu>.+)")
PAT_FPS = re.compile(r"(?P<cuadros>\d+)\s+cuadros en\s+(?P<segundos>[\d.]+)\s*s")
COLUMNAS = ["fecha", "gpu", "grid", "ventana", "msaa", "vsync", "frames",
            "fps_prom", "ms_prom", "ms_p95", "fps_1pct_low", "nota"]


def parsear_salida(texto: str):
    """Devuelve {'gpu', 'frames', 'segundos'} a partir de lo que imprime main.py, o None."""
    m_fps = PAT_FPS.search(texto)
    if not m_fps:
        return None
    m_gpu = PAT_GPU.search(texto)
    return {
        "gpu": m_gpu.group("gpu").strip() if m_gpu else "desconocida",
        "frames": int(m_fps.group("cuadros")),
        "segundos": float(m_fps.group("segundos")),
    }


def ejecutar_main(main_py: Path, frames: int, t_inicio: float, vista3d: bool, timeout: int):
    orden = [sys.executable, str(main_py)]
    if vista3d:
        orden.append("--3d")
    orden += ["--frames", str(frames), "--t-inicio", str(t_inicio)]
    entorno = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(orden, cwd=str(main_py.parent), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout, env=entorno)
    return proc.returncode, proc.stdout + "\n" + proc.stderr


def fila_csv(datos: dict, nota: str) -> dict:
    fps = datos["frames"] / max(datos["segundos"], 1e-9)
    return {
        "fecha": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "gpu": datos["gpu"],
        "grid": cfg.GRID,
        "ventana": f"{cfg.VENTANA_ANCHO}x{cfg.VENTANA_ALTO}",
        "msaa": 0,
        "vsync": False,
        "frames": datos["frames"],
        "fps_prom": round(fps, 1),
        "ms_prom": round(1000.0 / fps, 2),
        "ms_p95": "",
        "fps_1pct_low": "",
        "nota": nota,
    }


def guardar_csv(ruta: Path, fila: dict):
    existe = ruta.exists() and ruta.stat().st_size > 0
    with open(ruta, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNAS)
        if not existe:
            w.writeheader()
        w.writerow(fila)


def main():
    ap = argparse.ArgumentParser(description="Mide los fps de main.py con todos los módulos.")
    ap.add_argument("--frames", type=int, default=600, help="cuadros por corrida")
    ap.add_argument("--repeticiones", type=int, default=3, help="número de corridas")
    ap.add_argument("--t-inicio", type=float, default=20.0, help="segundo de inicio de la simulación")
    ap.add_argument("--mapa2d", action="store_true", help="medir la vista de mapa 2D en vez de la 3D")
    ap.add_argument("--nota", default="semana2 todo junto")
    ap.add_argument("--csv", default=str(RAIZ / "metricas_fps.csv"))
    ap.add_argument("--main", default=str(RAIZ / "main.py"), help="ruta de main.py (para pruebas)")
    ap.add_argument("--timeout", type=int, default=300, help="segundos máximos por corrida")
    args = ap.parse_args()

    fps_corridas = []
    for i in range(1, args.repeticiones + 1):
        print(f"Corrida {i}/{args.repeticiones}: {args.frames} cuadros...", flush=True)
        try:
            codigo, salida = ejecutar_main(Path(args.main), args.frames, args.t_inicio,
                                           not args.mapa2d, args.timeout)
        except subprocess.TimeoutExpired:
            print("  Se acabó el tiempo. Baja --frames o revisa que la ventana abra.")
            continue
        datos = parsear_salida(salida)
        if codigo != 0 or datos is None:
            print(f"  main.py terminó con código {codigo} y no se pudo leer el resultado.")
            print("  Últimas líneas de la salida:\n  " + "\n  ".join(salida.strip().splitlines()[-8:]))
            continue
        fila = fila_csv(datos, args.nota)
        guardar_csv(Path(args.csv), fila)
        fps_corridas.append(fila["fps_prom"])
        print(f"  {fila['fps_prom']} fps ({fila['ms_prom']} ms por cuadro) en {datos['gpu']}")

    if fps_corridas:
        print(f"\nPromedio de {len(fps_corridas)} corridas: {statistics.mean(fps_corridas):.1f} fps "
              f"(mín {min(fps_corridas)}, máx {max(fps_corridas)}). Guardado en {args.csv}")
        objetivo = cfg.FPS_OBJETIVO
        estado = "cumple" if statistics.mean(fps_corridas) >= objetivo else "NO cumple"
        print(f"Objetivo del proyecto: {objetivo} fps -> {estado}.")
    else:
        sys.exit("Ninguna corrida produjo datos.")


if __name__ == "__main__":
    main()
