import numpy as np

import config as cfg

UMBRALES = {
    cfg.DANO_DANADA: ((1.0, 0.5), (0.5, 3.0)),
    cfg.DANO_DESTRUIDA: ((cfg.NIVEL_CRITICO_M, 2.0), (1.5, 5.0)),
}
T_PERSISTENCIA = 0.6


class ModeloDano:
    def __init__(self, casas, grid=None, halo=1, umbrales=UMBRALES, persistencia=T_PERSISTENCIA):
        grid = cfg.GRID if grid is None else grid
        self.n = len(casas)
        self.umbrales = umbrales
        self.persistencia = persistencia
        idx, offs, pos = [], [], 0
        for ix0, iz0, ix1, iz1 in np.asarray(casas).reshape(-1, 4):
            a0, a1 = max(iz0 - halo, 0), min(iz1 + halo, grid)
            b0, b1 = max(ix0 - halo, 0), min(ix1 + halo, grid)
            ii, jj = np.meshgrid(np.arange(a0, a1), np.arange(b0, b1), indexing="ij")
            flat = (ii * grid + jj).ravel()
            offs.append(pos)
            pos += flat.size
            idx.append(flat)
        self._idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        self._offs = np.asarray(offs, np.int64)
        self._t = np.zeros((self.n, 3), np.float32)

    def medir(self, W, U, V):
        if self.n == 0:
            return np.zeros(0), np.zeros(0)
        w = np.asarray(W).ravel()[self._idx]
        v = np.hypot(np.asarray(U).ravel()[self._idx], np.asarray(V).ravel()[self._idx])
        v = np.where(w >= cfg.UMBRAL_MOJADO_M, v, 0.0)
        return np.maximum.reduceat(w, self._offs), np.maximum.reduceat(v, self._offs)

    def actualizar(self, W, U, V, dano, dt):
        if self.n == 0:
            return []
        h, v = self.medir(W, U, V)
        antes = dano.copy()
        for nivel in (cfg.DANO_DANADA, cfg.DANO_DESTRUIDA):
            cond = np.zeros(self.n, bool)
            for hu, vu in self.umbrales[nivel]:
                cond |= (h >= hu) & (v >= vu)
            self._t[:, nivel] = np.where(cond, self._t[:, nivel] + dt, 0.0)
            alcanzado = cond & (self._t[:, nivel] >= self.persistencia)
            np.maximum(dano, np.where(alcanzado, nivel, 0).astype(np.uint8), out=dano)
        return np.flatnonzero(dano != antes).tolist()

    def reiniciar(self):
        self._t[:] = 0
