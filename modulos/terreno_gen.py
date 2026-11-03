from dataclasses import dataclass

import numpy as np

import config as cfg

CTRL_COSTA_X = (120.0, 310.0, 10.0, 200.0)
S_COSTA = 0.030
S_MAR = 0.070
L_MAR = 150.0
S_LLANO = 0.020
S_COLINA = 0.120
D_COLINA0, D_COLINA1 = 250.0, 400.0

AMP_D = np.array([-1000.0, -50.0, 0.0, 40.0, 100.0, 250.0, 400.0, 700.0])
AMP_M = np.array([0.8, 0.5, 0.0, 0.15, 0.8, 1.6, 6.0, 10.0])


def bezier(ctrl, t):
    P = np.asarray(ctrl, np.float64)
    t3 = np.atleast_1d(np.asarray(t, np.float64)).reshape(-1, 1, 1)
    pts = np.broadcast_to(P, (t3.shape[0],) + P.shape).copy()
    while pts.shape[1] > 1:
        pts = (1 - t3) * pts[:, :-1] + t3 * pts[:, 1:]
    return pts[:, 0, :]


def linea_costa(grid=None, cell_m=None, ctrl_x=CTRL_COSTA_X):
    grid = cfg.GRID if grid is None else grid
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    ancho = grid * cell_m
    ctrl = np.array([[x, ancho * k / (len(ctrl_x) - 1)] for k, x in enumerate(ctrl_x)])
    t = np.arange(grid) * cell_m / ancho
    return bezier(ctrl, t)[:, 0]


def diamond_square(n, rugosidad, rng):
    assert n >= 2 and (n & (n - 1)) == 0
    N = n + 1
    a = np.zeros((N, N), np.float64)
    a[[0, 0, n, n], [0, n, 0, n]] = rng.uniform(-1, 1, 4)
    paso, amp = n, 1.0
    while paso > 1:
        m = paso // 2
        k = n // paso
        ruido = rng.uniform(-amp, amp, (k, k))
        for ii, i in enumerate(range(m, N, paso)):
            for jj, j in enumerate(range(m, N, paso)):
                a[i, j] = (a[i - m, j - m] + a[i - m, j + m] + a[i + m, j - m] + a[i + m, j + m]) / 4 + ruido[ii, jj]
        for i in range(0, N, m):
            j0 = m if (i // m) % 2 == 0 else 0
            for j in range(j0, N, paso):
                v = []
                if i - m >= 0:
                    v.append(a[i - m, j])
                if i + m < N:
                    v.append(a[i + m, j])
                if j - m >= 0:
                    v.append(a[i, j - m])
                if j + m < N:
                    v.append(a[i, j + m])
                a[i, j] = sum(v) / len(v) + rng.uniform(-amp, amp)
        amp *= 2.0 ** (-rugosidad)
        paso = m
    return a / np.abs(a).max()


def _suave(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def _pendiente(d):
    d = np.asarray(d, np.float64)
    mar = S_COSTA + (S_MAR - S_COSTA) * (1 - np.exp(d / L_MAR))
    tierra = (
        S_LLANO
        + (S_COSTA - S_LLANO) * np.exp(-d / 40.0)
        + S_COLINA * _suave((d - D_COLINA0) / (D_COLINA1 - D_COLINA0))
    )
    return np.where(d < 0, mar, tierra)


_PASO = 0.5
_D_TAB = np.arange(-1000.0, 1000.0 + _PASO / 2, _PASO)
_H_TAB = np.cumsum(_pendiente(_D_TAB)) * _PASO
_H_TAB -= np.interp(0.0, _D_TAB, _H_TAB)


def perfil_base(d):
    return np.interp(d, _D_TAB, _H_TAB)


@dataclass
class Escenario:
    H: np.ndarray
    x_costa: np.ndarray
    casas: np.ndarray
    tipos: list
    base_y: np.ndarray


def generar_h(grid=None, cell_m=None, semilla=None, nivel_mar=None, simple=False):
    grid = cfg.GRID if grid is None else grid
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    semilla = cfg.SEMILLA if semilla is None else semilla
    nivel_mar = cfg.NIVEL_MAR_M if nivel_mar is None else nivel_mar
    x_m = np.arange(grid) * cell_m
    if simple:
        x_costa = np.full(grid, 160.0)
        d = x_m[None, :] - x_costa[:, None]
        return (nivel_mar + 0.03 * d).astype(np.float32), x_costa
    x_costa = linea_costa(grid, cell_m)
    d = x_m[None, :] - x_costa[:, None]
    n = 1 << int(np.ceil(np.log2(grid)))
    rng = np.random.default_rng(semilla)
    f = diamond_square(n, 0.85, rng)[:grid, :grid]
    H = perfil_base(d) + f * np.interp(d, AMP_D, AMP_M) + nivel_mar
    return H.astype(np.float32), x_costa


def colocar_casas(
    H,
    x_costa,
    cell_m=None,
    semilla=None,
    d_min=45.0,
    d_max=230.0,
    paso_d=40.0,
    paso_iz=8,
    prob=0.85,
    prob_2pisos=0.3,
    h_min=0.9,
):
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    semilla = cfg.SEMILLA if semilla is None else semilla
    G = H.shape[0]
    rng = np.random.default_rng(semilla + 1)
    ocupado = np.zeros(H.shape, bool)
    casas, tipos, base = [], [], []
    for iz_c in range(4, G - 4, paso_iz):
        d = d_min
        while d <= d_max:
            if rng.random() < prob:
                w, dd = int(rng.integers(2, 5)), int(rng.integers(2, 5))
                pisos = 2 if rng.random() < prob_2pisos else 1
                jx, jz = rng.uniform(-6.0, 6.0), int(rng.integers(-1, 2))
                x_c = x_costa[iz_c] + d + jx
                ix0 = int(round(x_c / cell_m - w / 2))
                iz0 = iz_c + jz - dd // 2
                ix1, iz1 = ix0 + w, iz0 + dd
                if ix0 >= 2 and iz0 >= 2 and ix1 <= G - 2 and iz1 <= G - 2:
                    sub = H[iz0 : iz1 + 1, ix0 : ix1 + 1]
                    libre = not ocupado[iz0 - 2 : iz1 + 2, ix0 - 2 : ix1 + 2].any()
                    if libre and sub.min() >= h_min and (sub.max() - sub.min()) <= 1.2:
                        z0 = float(np.median(sub))
                        anillo = H[iz0 - 1 : iz1 + 2, ix0 - 1 : ix1 + 2]
                        anillo[:] = 0.5 * (anillo + z0)
                        sub[:] = z0
                        ocupado[iz0:iz1, ix0:ix1] = True
                        casas.append((ix0, iz0, ix1, iz1))
                        tipos.append((w, dd, pisos))
                        base.append(z0)
            d += paso_d
    return (
        np.asarray(casas, np.int32).reshape(-1, 4),
        tipos,
        np.asarray(base, np.float32),
    )


def generar_escenario(grid=None, cell_m=None, semilla=None, nivel_mar=None, simple=False):
    H, x_costa = generar_h(grid, cell_m, semilla, nivel_mar, simple)
    casas, tipos, base = colocar_casas(H, x_costa, cell_m, semilla)
    return Escenario(H=H, x_costa=x_costa, casas=casas, tipos=tipos, base_y=base)


def altura_en(H, x_m, z_m, cell_m=None):
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    G = H.shape[0]
    fx = np.clip(np.asarray(x_m) / cell_m, 0, G - 1.001)
    fz = np.clip(np.asarray(z_m) / cell_m, 0, G - 1.001)
    ix, iz = fx.astype(int), fz.astype(int)
    tx, tz = fx - ix, fz - iz
    return (
        H[iz, ix] * (1 - tx) * (1 - tz)
        + H[iz, ix + 1] * tx * (1 - tz)
        + H[iz + 1, ix] * (1 - tx) * tz
        + H[iz + 1, ix + 1] * tx * tz
    )


def _mezcla(a, b, t):
    t = np.asarray(t)[..., None]
    return a * (1 - t) + b * t


def malla_terreno(H, cell_m=None):
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    G = H.shape[0]
    gx = np.gradient(H.astype(np.float64), cell_m, axis=1)
    gz = np.gradient(H.astype(np.float64), cell_m, axis=0)
    n = np.stack([-gx, np.ones_like(gx), -gz], -1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    pend = np.hypot(gx, gz)

    fondo = np.array([0.30, 0.36, 0.40])
    arena = np.array([0.86, 0.78, 0.58])
    pasto = np.array([0.42, 0.55, 0.30])
    roca = np.array([0.50, 0.46, 0.42])
    col = _mezcla(fondo, arena, _suave((H + 1.0) / 1.0))
    col = _mezcla(col, pasto, _suave((H - 1.0) / 1.5))
    col = _mezcla(col, roca, np.maximum(_suave((H - 14.0) / 8.0), _suave((pend - 0.35) / 0.35)))

    x = np.arange(G) * cell_m
    X, Z = np.meshgrid(x, x)
    v = np.concatenate([np.stack([X, H, Z], -1), n, col], -1).astype(np.float32).reshape(-1, 9)

    ii, jj = np.meshgrid(np.arange(G - 1), np.arange(G - 1), indexing="ij")
    a = (ii * G + jj).ravel()
    b, c, d = a + 1, a + G, a + G + 1
    idx = np.stack([a, c, b, b, c, d], -1).astype(np.uint32).ravel()
    return v, idx
