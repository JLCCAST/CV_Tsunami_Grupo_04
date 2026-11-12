# Simulación de un tsunami sobre un terreno costero (Computación Visual, Grupo 4)

Python + OpenGL 3.3 (PyOpenGL + GLFW). Cada rol trabaja en su módulo y todos comparten un solo
objeto `Campos` (ver `CONTRATO.md`).

Estado del Rol 1 en la semana 2: el agua ya no es de prueba. El frente de ola (`agua/frente_ola.py`)
corre sobre el terreno real del rol 2 y publica W, U, V, la alarma de 0 a 1 y la posición del frente
en cada fila (`frente_x`, que el rol 2 usa para la cresta de la ola). El HUD muestra la
alarma, los medidores circulares (punto medio), la gráfica (Bresenham) y los milisegundos por paso.

## Cómo correrlo

```bash
python -m pip install -r requirements.txt
python -m pytest                     # 330 pruebas, no necesitan ventana
python main.py                       # vista de mapa 2D + HUD. ESC sale, ESPACIO pausa, R reinicia la ola
python main.py --3d                  # vista 3D (flechas y ratón mueven la cámara, la rueda hace zoom)
```

Opciones de `main.py`:

| Opción | Qué hace |
|---|---|
| `--3d` | vista 3D del terreno, el agua y las casas en lugar del mapa plano |
| `--t-inicio 30` | empieza la simulación en ese segundo (la ola llega a la costa hacia t = 17 s) |
| `--frames 200 --captura salida.png` | corre 200 cuadros, guarda una captura y sale |
| `--ola-falsa` | usa la ola de prueba de la semana 1 (`agua/datos_falsos.py`) en vez del frente de ola |

Herramientas sin ventana:

```bash
python tools/captura_hud.py --tiempos 0 22 30 45 60   # capturas/frente_t*.png (mapa + HUD)
python tools/medir_agua.py --grid 64 128 256          # ms por paso del agua por resolución
python tools/medir_agua.py --csv metricas_agua.csv --nota "semana 2"
```

Si `python main.py` dice "No se pudo crear la ventana OpenGL", actualiza los drivers de video.
En macOS puede hacer falta `pip install glfw` desde el Python oficial (no el del sistema).

## La ola (Rol 1, semana 2)

| Segundo | Qué pasa |
|---|---|
| 0 a 3 | mar en calma |
| 3 | el frente entra por el borde del mar (ix = 0), curvado, a 10 m/s |
| ~17 | llega a la costa; la alarma sube a 1 cuando el agua en tierra pasa los 3 m |
| 17 a 40 | inunda la playa; cada fila se detiene donde el terreno supera el muro de agua (3 m) |
| 40 a 60 | la ola baja y el agua vuelve al mar (U negativa) |
| 60 en adelante | calma otra vez; R reinicia todo (agua y daño de las casas) |

Rapidez del flujo: `sqrt(g * W)` con tope de 12 m/s, en dirección normal al frente.
Los parámetros están al inicio de `agua/frente_ola.py`. En la semana 3 el frente se frena según la
pendiente y respeta la máscara de obstáculos del rol 2.

Milisegundos por paso (laptop con Ryzen y Radeon 780M, 60 pasos por segundo, evento completo):

| GRID | m/celda | media | p95 | % de un cuadro a 20 fps |
|---|---|---|---|---|
| 64 | 10.0 | 0.10 ms | 0.12 ms | 0.2 % |
| 128 | 5.0 | 0.18 ms | 0.22 ms | 0.4 % |
| 256 | 2.5 | 0.50 ms | 0.69 ms | 1.0 % |

## Estructura

| Archivo o carpeta | Qué es | Dueño |
|---|---|---|
| `config.py` | constantes compartidas (GRID, metros por celda, umbrales) | Rol 1 |
| `campos.py` | objeto `Campos` con H, W, U, V, alarma, casas, daño, máscara + `validar()` | Rol 1 (el contrato lo acordamos todos) |
| `CONTRATO.md` | forma, unidad y dueño de cada campo; ejes; cómo engancharse | todos |
| `agua/frente_ola.py` | simulación del agua: frente de ola que publica W, U, V, frente_x, t y alarma | Rol 1 |
| `agua/alarma.py` | alarma de 0 a 1 | Rol 1 |
| `agua/medicion.py` | cronómetro de ms por paso | Rol 1 |
| `agua/datos_falsos.py` | ola de prueba de la semana 1 (solo con `--ola-falsa`) | Rol 1 |
| `hud/lienzo.py` | lienzo NumPy con Bresenham y punto medio propios | Rol 1 |
| `hud/panel.py` | panel: alarma, medidores, gráfica, mini-mapa, ms por paso | Rol 1 |
| `vista_mapa.py` | imagen 2D de terreno, agua y casas | Rol 1 |
| `modulos/terreno.py`, `terreno_gen.py`, `casas.py`, `csg.py`, `dano.py` | terreno diamond-square en forma de bahía, costa y calles en Bézier, muelles, casas CSG y estados de daño | Rol 2 |
| `modulos/brep.py`, `modulos/cresta.py` | techos a dos aguas en B-rep (fórmula de Euler) y cresta de la ola con parches de Bézier sobre `frente_x` | Rol 2 |
| `modulos/arrastre.py`, `camara3d.py` | cámara orbital, pila de matrices y arrastre | Rol 3 |
| `modulos/materiales.py`, `demo_materiales.py` | texturas, humedad, agua transparente y espuma | Rol 4 |
| `modulos/iluminacion.py`, `shaders/iluminacion.glsl` | sol, Phong, mojado, tinte de alarma y rampas de color | Rol 5 |
| `main.py` | ventana, bucle y `crear_modulos()` | integra el rol de la semana |
| `tools/` | `captura_hud.py`, `medir_agua.py` (Rol 1), `visor_terreno.py` (Rol 2), `medir_fps.py` (Rol 5) | cada rol |
| `tests/` | pruebas sin ventana de cada rol | cada rol |

## Qué hace cada rol esta semana

Ver el cuadro "Entregables por semana y rol" del documento del grupo. Para engancharte:
crea tu clase en `modulos/` heredando de `Modulo`, agrégala a `crear_modulos()` en `main.py` y
usa `Campos` tal como dice `CONTRATO.md`.

## Notas

- El HUD dibuja las formas con algoritmos propios (Bresenham, punto medio). Pillow solo rasteriza el texto.
- El lienzo del HUD se sube como textura como máximo a 10 Hz y solo si cambian los valores mostrados.
- Los fps que imprime `main.py` en la nube son de render por software y no representan tu computadora.
