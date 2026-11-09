# CONTRATO de datos compartidos

Todos los módulos leen y escriben sobre **un solo objeto** `Campos` (archivo `campos.py`).
No se cambia un campo (nombre, forma, unidad) sin avisar a todo el grupo.
Los valores de `config.py` son la única fuente de constantes compartidas.

## Mundo

| Constante | Valor | Significado |
|---|---|---|
| `GRID` | 128 | celdas por lado (parámetro único; 256 solo se prueba en la semana 5) |
| `CELL_M` | 5.0 m | metros por celda. Una casa de 10 a 20 m ocupa de 2 a 4 celdas |
| `WORLD_M` | 640 m | `GRID * CELL_M` |

## Ejes (no cambiar)

- Los arreglos se indexan `[iz, ix]` (fila = z, columna = x).
- El **mar está en `ix` pequeño** (borde izquierdo). `ix` crece hacia la tierra.
- En 3D, `y` es la altura. El punto en metros `(x, z)` es `(ix * CELL_M, iz * CELL_M)`.

## Campos

| Campo | Tipo y forma | Unidad / rango | Quién escribe | Quién lee |
|---|---|---|---|---|
| `H` | `float32 [G,G]` | m. Altura del terreno; `< 0` bajo el mar | Rol 2 | todos |
| `obstaculo` | `bool [G,G]` | `True` = celda ocupada por una casa en pie | Rol 2 | Rol 1 (agua) |
| `casas` | `int32 [N,4]` | cajas `(ix0, iz0, ix1, iz1)`, `ix1` e `iz1` exclusivos | Rol 2 | todos |
| `dano` | `uint8 [N]` | `0` intacta, `1` dañada, `2` destruida (una entrada por casa) | Rol 2 | Roles 3, 4 y HUD |
| `W` | `float32 [G,G]` | m. Profundidad del agua sobre el terreno, `>= 0` | Rol 1 | Roles 3, 4 y HUD |
| `U` | `float32 [G,G]` | m/s. Velocidad en x | Rol 1 | Roles 3 y 4 |
| `V` | `float32 [G,G]` | m/s. Velocidad en z | Rol 1 | Roles 3 y 4 |
| `alarma` | `float` | de 0.0 a 1.0 | Rol 1 | HUD, Roles 4 y 5 (tinte) |
| `t` | `float` | s. Reloj de la simulación | Rol 1 | todos |

Superficie del agua = `H + W`. Una celda está mojada si `W > UMBRAL_MOJADO_M` (0.05 m).
La alarma es la profundidad máxima en tierra firme (`H >= 0`) dividida entre `NIVEL_CRITICO_M` (3 m).

## Simulación del agua (desde la semana 2)

`agua/frente_ola.py` es la simulación real del rol 1. Lee `H` y escribe `W`, `U`, `V`, `t` y
`alarma`. Necesita el terreno del rol 2, así que se crea **después** de `Terreno.iniciar()`:

```python
from campos import Campos
from agua.frente_ola import FrenteOla
from modulos.terreno import Terreno

campos = Campos.vacios()
Terreno().iniciar(campos, None)   # sin OpenGL: genera H, casas, dano y obstaculo
ola = FrenteOla(campos)
ola.actualizar(t)         # t en segundos; modifica campos in situ. Si t retrocede, se reinicia
campos.validar()          # lanza AssertionError si algo rompe el contrato
```

Lo que conviene saber al leer el agua:

- El mar en calma tiene `W = -H` donde `H < 0` y velocidad cero. Solo el agua de la ola se mueve.
- `U > 0` mientras la ola entra (hacia la tierra) y `U < 0` cuando se retira (desde t = 40 s).
- `|U, V| <= 12 m/s`. Delante del frente, `W = 0` en tierra y `U = V = 0`.
- En `main.py` el agua se actualiza **antes** que los módulos en cada cuadro, así que en
  `actualizar()` cada módulo ya ve el agua del cuadro actual.

## Datos falsos (semana 1)

`agua/datos_falsos.py` rellena todos los campos (terreno en pendiente, 6 casas y una ola que
avanza y se repite cada 45 s). Sirve para pruebas que no quieren depender de nadie, y la app
lo usa con `python main.py --ola-falsa`:

```python
from agua.datos_falsos import DatosFalsos

falsos = DatosFalsos(campos)   # OJO: también reemplaza H y casas por las de prueba
falsos.actualizar(t)
```

## Cómo se engancha un módulo

1. Crea tu clase en `modulos/<tu_archivo>.py` heredando de `modulos.base.Modulo`
   (`iniciar`, `actualizar`, `dibujar`, `cerrar`; todos opcionales).
2. Agrégala a `crear_modulos()` en `main.py`.
3. Tus shaders GLSL van en `shaders/`. Dueños: Rol 4 el shader de materiales,
   Rol 5 `iluminacion.glsl`.

## Luz y sol (Rol 5)

`Iluminacion` publica en `ctx` (no en `Campos`) en cada `actualizar()`:

| Atributo | Tipo | Significado |
|---|---|---|
| `ctx.sol_dir` | float32 [3] | vector unitario HACIA el sol, en mundo (y arriba) |
| `ctx.sol_color` | float32 [3] | color de la luz ya multiplicado por la intensidad (0 de noche) |
| `ctx.hora` | float | hora del día, de 0 a 24 |
| `ctx.glsl_iluminacion` | str | texto de `shaders/iluminacion.glsl` |

Los shaders no definen su propio sol. La alarma se aplica al final del color con `tinte_alarma(color, alarma)`.

## Reglas de integración

- Cada semana integra un rol distinto (S1 rol 1, S2 rol 3, S3 rol 4, S4 rol 5, S5 rol 2, S6 rol 5).
- Lo que el viernes no esté en la rama principal no cuenta como hecho.
- Antes de subir: `python -m pytest` y `python main.py` deben funcionar.
