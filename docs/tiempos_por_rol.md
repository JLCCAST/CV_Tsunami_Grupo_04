# Tiempo estimado frente a tiempo real por rol

Grupo 4 · Computación Visual 2026-2. Esta tabla se lleva al control del final de la semana 2,
cuando el equipo decide si sigue con el proyecto o lo cambia. Sirve también para el informe.

## Cómo se llena

- **Horas estimadas:** lo que pensabas gastar en la tarea ANTES de empezarla. Si ya la terminaste
  sin anotarlo, escribe tu mejor recuerdo y márcalo con (aprox.) en la observación.
- **Horas reales:** el tiempo efectivo de trabajo, sin contar esperas ni pausas largas.
- **Diferencia:** reales menos estimadas. Positivo = tomó más de lo previsto.
- **Observación:** por qué se desvió (algo que no se sabía, una dependencia de otro rol, un error
  difícil de encontrar, etc.).
- Cada integrante completa **solo sus filas**, en su propio PR, para evitar conflictos.

## Semana 1

| Rol | Tarea | Horas estimadas | Horas reales | Diferencia | Observación |
|---|---|---|---|---|---|
| 1 | Repositorio, configuración y CONTRATO.md; ventana GLFW con HUD base (lienzo NumPy y Bresenham) | | | | |
| 2 | Terreno simple como campo H, casas como cajas y máscara de obstáculos | | | | |
| 3 | Cámara orbital, pila de matrices propia y objeto de prueba con campo de velocidad falso | 4h (aprox.) | 4h | 0h | Implementación de PilaMatrices en NumPy y control orbital con ratón y teclado. |
| 4 | Shader de materiales básico, plano de agua translúcido y humedad falsa | | | | |
| 5 | iluminacion.glsl con phong() y luz direccional, firmas acordadas, fps base y plantilla del informe |4 h (aprox.)|4 h|+1 h|Ajustar las firmas del shader y sincronizar el formato con los contratos iniciales tomó un poco más de lo previsto.|

## Semana 2

| Rol | Tarea | Horas estimadas | Horas reales | Diferencia | Observación |
|---|---|---|---|---|---|
| 1 | Frente de ola que publica W, U, V; ms por paso; alarma de 0 a 1; medidor circular con punto medio | | | | |
| 2 | Terreno diamond-square, costa en Bézier, casas con CSG y 3 estados; prototipo de casa partida | | | | |
| 3 | Arrastre v1 (muestrea W, U, V, umbral y fuerza) con un auto y de 20 a 50 piezas; fps | 5h | 5h | 0h | Arrastre v1 con flotabilidad, 40 piezas en escena y pruebas unitarias. En llvmpipe (Codespaces) rinde 13.0 fps promedio. |
| 4 | Humedad real calculada desde W y agua coloreada según la profundidad | | | | |
| 5 | Especular mojado, tinte_alarma() y rampa_color(); fps con todo junto; esta tabla |5 h|4 h|-1 h|El espejo en CPU y las pruebas salieron fluidas porque la matemática ya estaba probada en el GLSL de la semana anterior.|

## Resumen por rol

| Rol | Estimadas (S1 + S2) | Reales (S1 + S2) | Desviación | Qué haríamos distinto |
|---|---|---|---|---|
| 1 | | | | |
| 2 | | | | |
| 3 | 9h | 9h | 0h | Reutilizar desde el inicio el VAO compartido para lotes de escombros y probar temprano el entorno OpenGL. |
| 4 | | | | |
| 5 |9 h|9 h|0 h|Definir los contratos de uniforms y variables en el contexto (ctx) desde el primer día para evitar dependencias cruzadas entre los shaders de materiales y los de iluminación.|

## Conclusión del equipo para el control de la semana 2

- ¿Las estimaciones fueron razonables? [escribir]
- ¿Qué rol o tarea resultó más cargada de lo previsto y qué se recorta o se redistribuye? [escribir]
- Decisión: [seguir con el proyecto / cambiarlo / seguir con ajustes]
