# Simulación de un tsunami sobre un terreno costero (Computación Visual, Grupo 4)

Python + OpenGL 3.3 (PyOpenGL + GLFW). Entrega de la **semana 1, rol 1 (integra)**: esqueleto del
proyecto, configuración, contrato de datos, HUD base y datos falsos para que todos trabajen desde ya.

## Cómo correrlo

```bash
python -m pip install -r requirements.txt
python main.py                       # ESC sale, ESPACIO pausa, R reinicia la ola
python -m pytest                     # 31 pruebas, no necesitan ventana
python tools/captura_hud.py          # genera capturas PNG del HUD sin abrir ventana
```

Si `python main.py` dice "No se pudo crear la ventana OpenGL 3.3", actualiza los drivers de video.
En macOS puede hacer falta `pip install glfw` desde el Python oficial (no el del sistema).

## Estructura

| Archivo o carpeta | Qué es | Dueño |
|---|---|---|
| `config.py` | constantes compartidas (GRID, metros por celda, umbrales) | Rol 1 |
| `campos.py` | objeto `Campos` con H, W, U, V, alarma, casas, daño, máscara + `validar()` | Rol 1 (el contrato lo acordamos todos) |
| `CONTRATO.md` | forma, unidad y dueño de cada campo; ejes; cómo engancharse | todos |
| `agua/datos_falsos.py` | terreno, casas y ola falsos (se reemplazan por los módulos reales) | Rol 1 |
| `agua/alarma.py` | alarma de 0 a 1 | Rol 1 |
| `hud/lienzo.py` | lienzo NumPy con Bresenham y punto medio propios | Rol 1 |
| `hud/panel.py` | panel: alarma, medidores, gráfica, mini-mapa | Rol 1 |
| `vista_mapa.py` | imagen 2D de terreno, agua y casas | Rol 1 |
| `modulos/terreno.py` | módulo vacío del terreno | Rol 2 |
| `modulos/arrastre.py` | módulo vacío de cámara y arrastre | Rol 3 |
| `modulos/materiales.py` | módulo vacío de texturas y mojado | Rol 4 |
| `modulos/iluminacion.py` | módulo vacío de luz y color | Rol 5 |
| `shaders/` | aquí van los GLSL (Rol 4 el de materiales, Rol 5 `iluminacion.glsl`) | Roles 4 y 5 |
| `main.py` | ventana, bucle y lista `MODULOS` | integra el rol de la semana |

## Qué hace cada rol esta semana

Ver el cuadro "Entregables por semana y rol" del documento del grupo. Para engancharte:
crea tu clase en `modulos/` heredando de `Modulo`, agrégala a `crear_modulos()` en `main.py` y
usa `Campos` tal como dice `CONTRATO.md`.

## Notas

- El HUD dibuja las formas con algoritmos propios (Bresenham, punto medio). Pillow solo rasteriza el texto.
- El lienzo del HUD se sube como textura como máximo a 10 Hz y solo si cambian los valores mostrados.
- Los fps que imprime `main.py` en la nube son de render por software y no representan tu computadora.
