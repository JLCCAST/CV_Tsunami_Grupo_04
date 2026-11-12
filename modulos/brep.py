"""Representación de fronteras (B-rep) explícita: vértices, aristas y caras (Rol 2).

Un sólido se guarda como su frontera:
  * vertices: arreglo [V, 3] de puntos en metros.
  * caras: lista de polígonos planos y convexos; cada uno es una tupla de índices de vértice
    en orden antihorario visto desde fuera (así la normal apunta hacia afuera).
  * aristas: se derivan de las caras (cada lado de una cara, sin orientación).

En un poliedro cerrado sin agujeros pasantes se cumple la fórmula de Euler V - E + F = 2.
Con g agujeros pasantes (género g) la fórmula es V - E + F = 2 - 2g.

Las casas se arman con CSG de cajas (csg.py), que solo da caras alineadas a los ejes. El techo a
dos aguas tiene caras inclinadas, así que se modela aquí como B-rep: un prisma triangular.
"""
import numpy as np

from modulos.csg import MAT_MADERA, MAT_PARED, MAT_TECHO, PALETA


class BRep:
    def __init__(self, vertices, caras, materiales=None):
        self.vertices = np.asarray(vertices, np.float64).reshape(-1, 3)
        self.caras = [tuple(int(i) for i in c) for c in caras]
        assert all(len(c) >= 3 for c in self.caras), "cada cara necesita al menos 3 vértices"
        if materiales is None:
            materiales = [MAT_PARED] * len(self.caras)
        assert len(materiales) == len(self.caras), "un material por cara"
        self.materiales = [int(m) for m in materiales]

    # --- topología ------------------------------------------------------------------
    def aristas(self):
        """Aristas sin orientación, como pares (i, j) con i < j."""
        pares = set()
        for c in self.caras:
            for a, b in zip(c, c[1:] + c[:1]):
                pares.add((min(a, b), max(a, b)))
        return sorted(pares)

    def euler(self):
        """V - E + F. Vale 2 en un sólido cerrado sin agujeros pasantes."""
        return len(self.vertices) - len(self.aristas()) + len(self.caras)

    def es_cerrado(self):
        """True si cada arista orientada aparece una sola vez y su opuesta también.

        Es decir: la frontera no tiene huecos, cada arista separa exactamente dos caras y
        todas las caras están orientadas de forma coherente.
        """
        dirigidas = {}
        for c in self.caras:
            for a, b in zip(c, c[1:] + c[:1]):
                dirigidas[(a, b)] = dirigidas.get((a, b), 0) + 1
        return all(n == 1 and dirigidas.get((b, a), 0) == 1 for (a, b), n in dirigidas.items())

    # --- geometría --------------------------------------------------------------------
    def normal_cara(self, k):
        """Normal unitaria de la cara k por el método de Newell (sirve para cualquier polígono plano)."""
        P = self.vertices[list(self.caras[k])]
        Q = np.roll(P, -1, axis=0)
        n = np.array([
            ((P[:, 1] - Q[:, 1]) * (P[:, 2] + Q[:, 2])).sum(),
            ((P[:, 2] - Q[:, 2]) * (P[:, 0] + Q[:, 0])).sum(),
            ((P[:, 0] - Q[:, 0]) * (P[:, 1] + Q[:, 1])).sum(),
        ])
        return n / np.linalg.norm(n)

    def volumen(self):
        """Volumen encerrado (teorema de la divergencia). Es positivo si las caras miran hacia afuera."""
        v = 0.0
        for c in self.caras:
            P = self.vertices[list(c)]
            for i in range(1, len(c) - 1):
                v += float(np.dot(P[0], np.cross(P[i], P[i + 1])))
        return v / 6.0

    def trasladar(self, d):
        return BRep(self.vertices + np.asarray(d, np.float64), self.caras, self.materiales)

    def malla(self, paleta=PALETA):
        """Triángulos en abanico, con la normal plana de cada cara: [n, 9] (posición, normal, color).

        Es el mismo formato que devuelve csg.Solido.malla(), así que se dibuja con el mismo shader.
        """
        filas = []
        for k, c in enumerate(self.caras):
            n = self.normal_cara(k).tolist()
            col = [float(x) for x in paleta[self.materiales[k]]]
            P = self.vertices[list(c)]
            for i in range(1, len(c) - 1):
                for p in (P[0], P[i], P[i + 1]):
                    filas.append(p.tolist() + n + col)
        return np.asarray(filas, np.float32).reshape(-1, 9)


def prisma(perfil, z0, z1, mat_tapas=MAT_PARED, mat_lados=None):
    """Extruye un polígono convexo del plano xy entre z0 y z1 (z0 < z1).

    `perfil` va en orden antihorario visto desde +z. Para n lados quedan V = 2n, E = 3n y
    F = n + 2, así que V - E + F = 2. `mat_lados[i]` es el material del lado que une el
    vértice i con el i + 1.
    """
    n = len(perfil)
    assert n >= 3 and z0 < z1
    if mat_lados is None:
        mat_lados = [mat_tapas] * n
    vertices = [(x, y, z1) for x, y in perfil] + [(x, y, z0) for x, y in perfil]
    caras = [tuple(range(n)), tuple(n + i for i in reversed(range(n)))]   # tapas en z1 (+z) y z0 (-z)
    materiales = [mat_tapas, mat_tapas]
    for i in range(n):
        j = (i + 1) % n
        caras.append((j, i, n + i, n + j))
        materiales.append(mat_lados[i])
    return BRep(vertices, caras, materiales)


def techo_dos_aguas(medio_ancho, medio_largo, y_base, alto, alero=0.5):
    """Techo a dos aguas: prisma triangular con la cumbrera a lo largo de z.

    Las dos caras inclinadas son de tejas, los hastiales (los triángulos de los extremos) del
    color de la pared y la cara de abajo de madera. V = 6, E = 9, F = 5.
    """
    a = medio_ancho + alero
    perfil = [(-a, y_base), (a, y_base), (0.0, y_base + alto)]
    lados = [MAT_MADERA, MAT_TECHO, MAT_TECHO]
    return prisma(perfil, -medio_largo - alero, medio_largo + alero, MAT_PARED, lados)
