"""Pruebas de tools/medir_app.py (no abren ventana: leen texto y escriben un CSV temporal)."""
import csv
import importlib.util
from pathlib import Path

import pytest

RUTA = Path(__file__).resolve().parent.parent / "tools" / "medir_app.py"
spec = importlib.util.spec_from_file_location("medir_app", RUTA)
medir_app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(medir_app)

SALIDA = (
    "OpenGL: 3.3.0 - Build 32.0.101.7076 | Intel(R) UHD Graphics\n"
    "600 cuadros en 20.0 s = 30 fps promedio (ojo: en la nube el render es por software; "
    "mide en tu computadora)\n"
    "agua (frente de ola): 0.18 ms\n"
)


def test_parsear_salida_lee_gpu_cuadros_y_segundos():
    d = medir_app.parsear_salida(SALIDA)
    assert d == {"gpu": "Intel(R) UHD Graphics", "frames": 600, "segundos": 20.0}


def test_parsear_salida_sin_resultado_devuelve_none():
    assert medir_app.parsear_salida("Traceback (most recent call last): ...") is None


def test_fila_csv_calcula_fps_y_ms():
    fila = medir_app.fila_csv({"gpu": "X", "frames": 600, "segundos": 20.0}, "prueba")
    assert fila["fps_prom"] == 30.0
    assert fila["ms_prom"] == pytest.approx(33.33, abs=0.01)
    assert fila["nota"] == "prueba" and fila["ms_p95"] == ""
    assert list(fila.keys()) == medir_app.COLUMNAS


def test_guardar_csv_escribe_encabezado_una_sola_vez(tmp_path):
    ruta = tmp_path / "m.csv"
    fila = medir_app.fila_csv({"gpu": "X", "frames": 100, "segundos": 5.0}, "a")
    medir_app.guardar_csv(ruta, fila)
    medir_app.guardar_csv(ruta, fila)
    filas = list(csv.reader(open(ruta, encoding="utf-8")))
    assert filas[0] == medir_app.COLUMNAS
    assert len(filas) == 3
