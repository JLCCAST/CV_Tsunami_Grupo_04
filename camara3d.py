"""Manejo de matriz de vista y proyección 3D con control por teclado y ratón (Rol 3)."""
import math
import numpy as np
import config as cfg


def perspectiva(fov_grados: float, aspecto: float, cerca: float, lejos: float) -> np.ndarray:
    """Calcula la matriz de proyección perspectiva 4x4."""
    f = 1.0 / math.tan(math.radians(fov_grados) / 2.0)
    return np.array([
        [f / aspecto, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (lejos + cerca) / (cerca - lejos), 2.0 * lejos * cerca / (cerca - lejos)],
        [0.0, 0.0, -1.0, 0.0],
    ], dtype=np.float32)


def mirar_a(ojo: np.ndarray, objetivo: np.ndarray) -> np.ndarray:
    """Calcula la matriz de vista (LookAt) 4x4."""
    ojo = np.asarray(ojo, dtype=np.float64)
    objetivo = np.asarray(objetivo, dtype=np.float64)
    f = objetivo - ojo
    norm_f = np.linalg.norm(f)
    if norm_f > 0:
        f /= norm_f

    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    s = np.cross(f, up)
    norm_s = np.linalg.norm(s)
    if norm_s > 0:
        s /= norm_s

    u = np.cross(s, f)

    m = np.identity(4, dtype=np.float64)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ ojo, -u @ ojo, f @ ojo
    return m.astype(np.float32)


class CamaraOrbital:
    """Cámara orbital 3D con soporte para controles por ratón (arrastre y scroll) y teclado."""

    def __init__(self):
        self.az = math.pi              # Azimut inicial (mirando desde el mar)
        self.el = math.radians(25.0)   # Elevación inicial
        self.dist = 700.0             # Distancia al centro en metros

        mitad = cfg.WORLD_M / 2.0
        self.centro = np.array([mitad, 0.0, mitad], dtype=np.float32)

        # Estado para el control con ratón
        self._last_mouse_x = 0.0
        self._last_mouse_y = 0.0
        self._dragging = False

    def procesar_entrada(self, glfw, ventana, dt: float):
        """Maneja teclado (Flechas) y ratón para controlar la órbita."""
        # 1. Controles por teclado (flechas)
        if glfw.get_key(ventana, glfw.KEY_LEFT) == glfw.PRESS:
            self.az -= 1.2 * dt
        if glfw.get_key(ventana, glfw.KEY_RIGHT) == glfw.PRESS:
            self.az += 1.2 * dt
        if glfw.get_key(ventana, glfw.KEY_UP) == glfw.PRESS:
            self.el = min(math.radians(85.0), self.el + 1.0 * dt)
        if glfw.get_key(ventana, glfw.KEY_DOWN) == glfw.PRESS:
            self.el = max(math.radians(5.0), self.el - 1.0 * dt)

        # 2. Controles por ratón (Botón Izquierdo = Rotar)
        state = glfw.get_mouse_button(ventana, glfw.MOUSE_BUTTON_LEFT)
        xpos, ypos = glfw.get_cursor_pos(ventana)

        if state == glfw.PRESS:
            if not self._dragging:
                self._dragging = True
                self._last_mouse_x = xpos
                self._last_mouse_y = ypos
            else:
                dx = xpos - self._last_mouse_x
                dy = ypos - self._last_mouse_y
                self.az += dx * 0.005
                self.el = max(math.radians(5.0), min(math.radians(85.0), self.el - dy * 0.005))
                self._last_mouse_x = xpos
                self._last_mouse_y = ypos
        else:
            self._dragging = False

    def scroll_callback(self, xoffset: float, yoffset: float):
        """Ajusta el zoom mediante la rueda del ratón."""
        self.dist = max(50.0, min(2500.0, self.dist - yoffset * 40.0))

    def matrices(self, aspecto: float):
        """Devuelve (VP, posición_ojo) para el pipeline de renderizado 3D."""
        eye_x = self.dist * math.cos(self.el) * math.cos(self.az)
        eye_y = self.dist * math.sin(self.el)
        eye_z = self.dist * math.cos(self.el) * math.sin(self.az)

        ojo = self.centro + np.array([eye_x, eye_y, eye_z], dtype=np.float32)
        v = mirar_a(ojo, self.centro)
        p = perspectiva(45.0, aspecto, 1.0, 4000.0)
        vp = p @ v
        return vp, ojo