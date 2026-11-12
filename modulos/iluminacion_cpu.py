"""Espejo en NumPy de las funciones de shaders/iluminacion.glsl (Rol 5).

Para qué sirve:
  - Probar SIN GPU la matemática de Phong, el efecto mojado, el tinte de alarma y la rampa
    de color (tests/test_iluminacion_s2.py).
  - Generar cifras y figuras para el informe sin abrir una ventana.

Las fórmulas deben ser idénticas a las del GLSL. Si cambias una, cambia la otra:
las pruebas comprueban que los números clave sigan en ambos archivos.
Convenciones (igual que en el GLSL): N, V y L normalizados, espacio de mundo, L hacia el sol.
"""
import numpy as np

COLOR_VERDE = np.array((46.0, 204.0, 113.0), np.float32) / 255.0
COLOR_AMARILLO = np.array((241.0, 196.0, 15.0), np.float32) / 255.0
COLOR_ROJO = np.array((231.0, 76.0, 60.0), np.float32) / 255.0


def _mix(a, b, t):
    return a + (b - a) * t


def _smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def rampa_color(t) -> np.ndarray:
    """Riesgo 0..1 -> RGB 0..1 (verde, amarillo, rojo). Igual que rampa_color() del GLSL."""
    t = float(np.clip(t, 0.0, 1.0))
    if t < 0.5:
        return _mix(COLOR_VERDE, COLOR_AMARILLO, t / 0.5)
    return _mix(COLOR_AMARILLO, COLOR_ROJO, (t - 0.5) / 0.5)


def tinte_alarma(color, alarma: float) -> np.ndarray:
    """Oscurece y vira a rojizo desde alarma 0.6 hasta el máximo en 1.0."""
    color = np.asarray(color, np.float32)
    oscuro = color * 0.80
    rojizo = oscuro * np.array((1.25, 0.82, 0.78), np.float32)
    return _mix(color, rojizo, _smoothstep(0.6, 1.0, alarma))


def phong(N, V, L, color_base, color_luz, ka, kd, ks, brillo, visibilidad=1.0) -> np.ndarray:
    """I = ka*Cb + visibilidad * ( kd*max(N.L,0)*Cb*Cl + ks*max(R.V,0)^n*Cl ),  R = reflect(-L, N)."""
    N, V, L = (np.asarray(v, np.float32) for v in (N, V, L))
    color_base = np.asarray(color_base, np.float32)
    color_luz = np.asarray(color_luz, np.float32)
    n_dot_l = max(float(N @ L), 0.0)
    R = -L + 2.0 * float(N @ L) * N                    # reflect(-L, N)
    r_dot_v = max(float(R @ V), 0.0)
    ambiente = ka * color_base
    difuso = kd * n_dot_l * color_base * color_luz
    especular = ks * (r_dot_v ** brillo) * color_luz if n_dot_l > 0.0 else np.zeros(3, np.float32)
    return ambiente + visibilidad * (difuso + especular)


def phong_mojado(N, V, L, color_base, color_luz, ka, kd, ks, brillo,
                 humedad, visibilidad=1.0) -> np.ndarray:
    """Mojado = color base más oscuro + especular más fuerte + brillo más concentrado."""
    base_mojada = np.asarray(color_base, np.float32) * _mix(1.0, 0.55, humedad)
    ks_m = _mix(ks, 0.90, humedad)
    brillo_m = _mix(brillo, 128.0, humedad)
    return phong(N, V, L, base_mojada, color_luz, ka, kd, ks_m, brillo_m, visibilidad)
