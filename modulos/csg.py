import numpy as np

MAT_CIMIENTO, MAT_PARED, MAT_TECHO, MAT_MADERA, MAT_PARED_DANADA, MAT_ESCOMBRO = 1, 2, 3, 4, 5, 6

PALETA = np.array(
    [
        (0.00, 0.00, 0.00),
        (0.55, 0.55, 0.57),
        (0.93, 0.88, 0.78),
        (0.72, 0.30, 0.22),
        (0.45, 0.30, 0.18),
        (0.62, 0.55, 0.47),
        (0.36, 0.33, 0.30),
    ],
    np.float32,
)


def _r(v):
    return float(np.round(v, 4))


class Solido:
    def __init__(self):
        self._ops = []

    def copia(self):
        s = Solido()
        s._ops = list(self._ops)
        return s

    def sumar(self, lo, hi, material):
        self._ops.append(("+", tuple(lo), tuple(hi), int(material)))
        return self

    def restar(self, lo, hi):
        self._ops.append(("-", tuple(lo), tuple(hi), 0))
        return self

    def pintar(self, lo, hi, material):
        self._ops.append(("p", tuple(lo), tuple(hi), int(material)))
        return self

    def _rejilla(self, extra=None):
        pts = [[], [], []]
        for _, lo, hi, _m in self._ops:
            for a in range(3):
                pts[a] += [_r(lo[a]), _r(hi[a])]
        if extra is not None:
            for a in range(3):
                pts[a] += [_r(v) for v in extra[a]]
        coords = [np.unique(np.asarray(p, np.float64)) for p in pts]
        M = np.zeros(tuple(len(c) - 1 for c in coords), np.uint8)
        for op, lo, hi, mat in self._ops:
            sl = tuple(
                slice(int(np.searchsorted(coords[a], _r(lo[a]))), int(np.searchsorted(coords[a], _r(hi[a]))))
                for a in range(3)
            )
            if op == "+":
                M[sl] = mat
            elif op == "-":
                M[sl] = 0
            else:
                sub = M[sl]
                sub[sub > 0] = mat
        return coords, M

    @staticmethod
    def _vol_celdas(coords):
        d = [np.diff(c) for c in coords]
        return d[0][:, None, None] * d[1][None, :, None] * d[2][None, None, :]

    def volumen(self):
        coords, M = self._rejilla()
        return float((self._vol_celdas(coords) * (M > 0)).sum())

    def caja_envolvente(self):
        coords, M = self._rejilla()
        ocup = np.argwhere(M > 0)
        lo = [coords[a][ocup[:, a].min()] for a in range(3)]
        hi = [coords[a][ocup[:, a].max() + 1] for a in range(3)]
        return np.array(lo), np.array(hi)

    def malla(self):
        coords, M = self._rejilla()
        return _extraer(coords, M)

    def partir(self, cortes=(3, 2, 3), semilla=0, volumen_min=0.3):
        assert all(1 <= c <= 9 for c in cortes)
        rng = np.random.default_rng(semilla)
        coords0, _ = self._rejilla()
        planos = []
        for a in range(3):
            lo, hi, n = coords0[a][0], coords0[a][-1], cortes[a]
            paso = (hi - lo) / n
            base = np.linspace(lo, hi, n + 1)[1:-1]
            planos.append(np.round(base + rng.uniform(-0.2, 0.2, len(base)) * paso, 4))
        coords, M = self._rejilla(extra=planos)
        centros = [(c[:-1] + c[1:]) / 2 for c in coords]
        bins = [np.digitize(centros[a], planos[a]) for a in range(3)]
        etiqueta = bins[0][:, None, None] * 100 + bins[1][None, :, None] * 10 + bins[2][None, None, :]
        dvol = self._vol_celdas(coords)
        piezas = []
        for lab in np.unique(etiqueta[M > 0]):
            mascara = (etiqueta == lab) & (M > 0)
            vol = float((dvol * mascara).sum())
            if vol < volumen_min:
                continue
            cen = np.array(
                [
                    float((dvol * mascara * centros[0][:, None, None]).sum() / vol),
                    float((dvol * mascara * centros[1][None, :, None]).sum() / vol),
                    float((dvol * mascara * centros[2][None, None, :]).sum() / vol),
                ]
            )
            malla = _extraer(coords, np.where(mascara, M, 0).astype(np.uint8))
            malla[:, :3] -= cen.astype(np.float32)
            piezas.append({"malla": malla, "centroide": cen, "volumen": vol})
        return piezas


def _rectangulos(S):
    S = S.copy()
    nb, nc = S.shape
    out = []
    for b in range(nb):
        c = 0
        while c < nc:
            m = S[b, c]
            if m == 0:
                c += 1
                continue
            c1 = c + 1
            while c1 < nc and S[b, c1] == m:
                c1 += 1
            b1 = b + 1
            while b1 < nb and np.all(S[b1, c:c1] == m):
                b1 += 1
            S[b:b1, c:c1] = 0
            out.append((b, b1, c, c1, int(m)))
            c = c1
    return out


def _extraer(coords, M, paleta=PALETA):
    ocup = M > 0
    pad = np.pad(ocup, 1)
    verts = []
    for a in range(3):
        b, c = (a + 1) % 3, (a + 2) % 3
        for s in (+1, -1):
            sl = [slice(1, -1)] * 3
            sl[a] = slice(2, None) if s > 0 else slice(0, -2)
            vecino = pad[tuple(sl)]
            F = np.where(ocup & ~vecino, M, 0)
            F = np.transpose(F, (a, b, c))
            ca, cb, cc = coords[a], coords[b], coords[c]
            n = [0.0, 0.0, 0.0]
            n[a] = float(s)
            for i in range(F.shape[0]):
                S = F[i]
                if not S.any():
                    continue
                plano = ca[i + 1] if s > 0 else ca[i]
                for b0, b1, c0, c1, m in _rectangulos(S):
                    esq = [(b0, c0), (b1, c0), (b1, c1), (b0, c1)]
                    if s < 0:
                        esq = [(b0, c0), (b0, c1), (b1, c1), (b1, c0)]
                    P = []
                    for bi, ci in esq:
                        p = [0.0, 0.0, 0.0]
                        p[a], p[b], p[c] = plano, cb[bi], cc[ci]
                        P.append(p)
                    col = [float(x) for x in paleta[m]]
                    for tri in ((0, 1, 2), (0, 2, 3)):
                        for k in tri:
                            verts.append(P[k] + n + col)
    return np.asarray(verts, np.float32).reshape(-1, 9)


def volumen_firmado(verts):
    p = verts[:, :3].astype(np.float64).reshape(-1, 3, 3)
    return float(np.einsum("ij,ij->i", p[:, 0], np.cross(p[:, 1], p[:, 2])).sum() / 6.0)
