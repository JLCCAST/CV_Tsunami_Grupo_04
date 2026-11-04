"""Configuración única del proyecto.

Regla del equipo: ningún módulo escribe números "mágicos" propios.
Si necesitas una constante compartida, agrégala aquí y avisa al grupo.
"""

GRID = 128            # celdas por lado. Probar 256 solo en la semana 5 si hay fps.
CELL_M = 5.0          # metros por celda: una casa de 10 a 20 m ocupa de 2 a 4 celdas.
WORLD_M = GRID * CELL_M   # 640 m por lado
NIVEL_MAR_M = 0.0     # la altura H se mide respecto al nivel del mar

SEMILLA = 42

NIVEL_CRITICO_M = 3.0     # profundidad en tierra firme que equivale a alarma = 1.0
UMBRAL_MOJADO_M = 0.05    # por debajo de esta profundidad la celda se considera seca

VENTANA_ANCHO = 1280
VENTANA_ALTO = 720
HUD_ANCHO = 320           # panel lateral derecho, en píxeles de la ventana de 1280
HUD_HZ = 10               # el lienzo del HUD se redibuja como máximo a 10 Hz
FPS_OBJETIVO = 20         # referencia para el control de la semana

DANO_INTACTA = 0
DANO_DANADA = 1
DANO_DESTRUIDA = 2

HORA_INICIAL = 11.0
CICLO_DIA_S = 0.0