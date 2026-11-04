"""Cámara orbital simple para la vista 3D (python main.py --3d)."""
import math

import numpy as np

import config as cfg


def perspectiva(fov_grados, aspecto, cerca, lejos):
    f = 1.0 / math.tan(math.radians(fov_grados) / 2.0)
    return np.array([
        [f / aspecto, 0.0, 0.0, 0.0],
        [0.0, f, 0.0, 0.0],
        [0.0, 0.0, (lejos + cerca) / (cerca - lejos), 2.0 * lejos * cerca / (cerca - lejos)],
        [0.0, 0.0, -1.0, 0.0],
    ], np.float32)


def mirar_a(ojo, objetivo):
    ojo = np.asarray(ojo, np.float64)
    objetivo = np.asarray(objetivo, np.float64)
    f = objetivo - ojo
    f /= np.linalg.norm(f)
    s = np.cross(f, [0.0, 1.0, 0.0])
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3], m[1, :3], m[2, :3] = s, u, -f
    m[0, 3], m[1, 3], m[2, 3] = -s @ ojo, -u @ ojo, f @ ojo
    return m.astype(np.float32)


class CamaraOrbital:
    """Gira alrededor del centro del mundo. Izquierda/derecha giran, arriba/abajo acercan."""

    def __init__(self):
        self.az = math.pi            # mirando desde el lado del mar
        self.dist = 700.0
        mitad = cfg.WORLD_M / 2.0
        self.centro = np.array([mitad, 0.0, mitad])

    def actualizar(self, glfw, ventana, dt):
        if glfw.get_key(ventana, glfw.KEY_LEFT) == glfw.PRESS:
            self.az -= 1.2 * dt
        if glfw.get_key(ventana, glfw.KEY_RIGHT) == glfw.PRESS:
            self.az += 1.2 * dt
        if glfw.get_key(ventana, glfw.KEY_UP) == glfw.PRESS:
            self.dist = max(150.0, self.dist - 300.0 * dt)
        if glfw.get_key(ventana, glfw.KEY_DOWN) == glfw.PRESS:
            self.dist = min(2000.0, self.dist + 300.0 * dt)

    def matrices(self, aspecto):
        """Devuelve (vp, posición del ojo)."""
        ojo = self.centro + np.array([
            self.dist * math.cos(self.az),
            self.dist * 0.45,
            self.dist * math.sin(self.az),
        ])
        vp = perspectiva(45.0, aspecto, 1.0, 4000.0) @ mirar_a(ojo, self.centro)
        return vp, ojo