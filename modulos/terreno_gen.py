from dataclasses import dataclass, field

import numpy as np

import config as cfg
from modulos.csg import MAT_CIMIENTO, MAT_MADERA, Solido

# Bahía: los extremos de la costa (z = 0 y z = 640 m) salen hacia el mar como puntas y el
# centro se mete tierra adentro.
CTRL_COSTA_X = (90.0, 255.0, 255.0, 90.0)
# Cerros en las dos puntas de la bahía: (z centro, distancia a la costa, alto, radio en z, radio en d), m.
CERROS = ((20.0, 90.0, 22.0, 60.0, 90.0), (620.0, 90.0, 24.0, 60.0, 90.0))

# Calles. Las avenidas son la curva de la costa corrida tierra adentro (mover todos los puntos de
# control mueve igual la curva de Bézier). Las transversales suben desde la avenida costera.
D_AVENIDAS = (25.0, 105.0)
Z_TRANSVERSALES = (200.0, 320.0, 440.0)
LARGO_TRANSVERSAL = 195.0
ANCHO_CALLE = 8.0

# Muelles del puerto, en el centro de la bahía: z de cada uno, largo hacia el mar, ancho y alto del piso.
Z_MUELLES = (265.0, 320.0, 375.0)
LARGO_MUELLE = 45.0
ANCHO_MUELLE = 5.0
ALTO_MUELLE = 1.4
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
    calles: list = field(default_factory=list)          # polilíneas [n, 2] con (x, z) en metros
    mascara_calles: np.ndarray = None                   # bool [G,G] celdas cubiertas por una calle
    muelles: list = field(default_factory=list)         # dicts con x0, x1 (x1 en tierra), z, ancho, y


def _cerros(x_m, z_m, x_costa):
    d = x_m[None, :] - x_costa[:, None]
    z = z_m[:, None]
    total = np.zeros_like(d)
    for zc, dc, alto, rz, rd in CERROS:
        total += alto * np.exp(-(((z - zc) / rz) ** 2 + ((d - dc) / rd) ** 2))
    return total * _suave(d / 40.0)


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
    H = perfil_base(d) + f * np.interp(d, AMP_D, AMP_M) + _cerros(x_m, x_m, x_costa) + nivel_mar
    return H.astype(np.float32), x_costa


def calles_bezier(grid=None, cell_m=None, ctrl_x=CTRL_COSTA_X, muestras=129):
    """Calles como curvas de Bézier: dos avenidas paralelas a la costa y tres transversales."""
    grid = cfg.GRID if grid is None else grid
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    ancho = grid * cell_m
    zs = [ancho * k / (len(ctrl_x) - 1) for k in range(len(ctrl_x))]
    t = np.linspace(0.0, 1.0, muestras)
    calles = []
    for d in D_AVENIDAS:
        calles.append(bezier([(x + d, z) for x, z in zip(ctrl_x, zs)], t))
    x_costa = linea_costa(grid, cell_m, ctrl_x)
    for k, zc in enumerate(Z_TRANSVERSALES):
        x0 = float(np.interp(zc, np.arange(grid) * cell_m, x_costa)) + D_AVENIDAS[0]
        giro = 12.0 if k % 2 == 0 else -12.0          # curva suave en S, alternando el sentido
        L = LARGO_TRANSVERSAL
        ctrl = [(x0, zc), (x0 + L / 3, zc - giro), (x0 + 2 * L / 3, zc + giro), (x0 + L, zc + giro / 2)]
        calles.append(bezier(ctrl, t))
    return calles


def rasterizar_calles(calles, grid=None, cell_m=None, ancho=ANCHO_CALLE):
    """Marca las celdas cuyo centro queda a menos de ancho / 2 de alguna calle."""
    grid = cfg.GRID if grid is None else grid
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    m = np.zeros((grid, grid), bool)
    r = ancho / 2.0
    for poli in calles:
        largo = np.hypot(*np.diff(poli, axis=0).T).sum()
        n = max(2, int(np.ceil(largo / (cell_m / 4))))
        s = np.linspace(0.0, 1.0, n)
        u = np.linspace(0.0, 1.0, len(poli))
        px, pz = np.interp(s, u, poli[:, 0]), np.interp(s, u, poli[:, 1])
        for x, z in zip(px, pz):
            ix0, ix1 = int(np.floor((x - r) / cell_m)), int(np.ceil((x + r) / cell_m))
            iz0, iz1 = int(np.floor((z - r) / cell_m)), int(np.ceil((z + r) / cell_m))
            for iz in range(max(iz0, 0), min(iz1, grid - 1) + 1):
                for ix in range(max(ix0, 0), min(ix1, grid - 1) + 1):
                    if (ix * cell_m - x) ** 2 + (iz * cell_m - z) ** 2 <= r * r:
                        m[iz, ix] = True
    return m


def generar_muelles(x_costa, cell_m=None):
    """Muelles del puerto: un piso de madera sobre el mar que sale de la costa."""
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    z_m = np.arange(len(x_costa)) * cell_m
    muelles = []
    for z in Z_MUELLES:
        xc = float(np.interp(z, z_m, x_costa))
        muelles.append({"x0": xc - LARGO_MUELLE, "x1": xc + 4.0, "z": z, "ancho": ANCHO_MUELLE, "y": ALTO_MUELLE})
    return muelles


def malla_muelles(muelles, H, cell_m=None):
    """CSG de cajas: el piso de cada muelle y pilotes cada 9 m que bajan hasta el fondo."""
    cell_m = cfg.CELL_M if cell_m is None else cell_m
    s = Solido()
    for m in muelles:
        hz = m["ancho"] / 2
        s.sumar((m["x0"], m["y"] - 0.35, m["z"] - hz), (m["x1"], m["y"], m["z"] + hz), MAT_MADERA)
        for x in np.arange(m["x0"] + 1.0, m["x1"] - 2.0, 9.0):
            fondo = float(altura_en(H, x, m["z"], cell_m)) - 0.5
            if fondo > m["y"] - 0.6:          # ahí el muelle ya está sobre tierra: no lleva pilotes
                continue
            for dz in (-hz + 0.4, hz - 0.4):
                s.sumar((x - 0.25, fondo, m["z"] + dz - 0.25), (x + 0.25, m["y"] - 0.35, m["z"] + dz + 0.25),
                        MAT_CIMIENTO)
    return s.malla()


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
    reservado=None,
):
    """Coloca casas en filas paralelas a la costa. `reservado` (bool [G,G]) marca celdas donde
    no puede haber casa, como las calles; una casa puede quedar pegada a la vereda."""
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
                    if reservado is not None and reservado[iz0:iz1, ix0:ix1].any():
                        libre = False
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
    grid = cfg.GRID if grid is None else grid
    H, x_costa = generar_h(grid, cell_m, semilla, nivel_mar, simple)
    if simple:
        calles, mascara, muelles = [], np.zeros(H.shape, bool), []
    else:
        calles = calles_bezier(grid, cell_m)
        mascara = rasterizar_calles(calles, grid, cell_m)
        muelles = generar_muelles(x_costa, cell_m)
    casas, tipos, base = colocar_casas(H, x_costa, cell_m, semilla, reservado=mascara)
    return Escenario(H=H, x_costa=x_costa, casas=casas, tipos=tipos, base_y=base,
                     calles=calles, mascara_calles=mascara, muelles=muelles)


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


def malla_terreno(H, cell_m=None, calles=None):
    """Malla del terreno: [G*G, 9] (posición, normal, color) e índices. `calles` (bool [G,G])
    pinta de asfalto las celdas de las calles."""
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
    if calles is not None:
        asfalto = np.array([0.33, 0.34, 0.36])
        col = _mezcla(col, asfalto, np.asarray(calles, np.float64) * 0.9)

    x = np.arange(G) * cell_m
    X, Z = np.meshgrid(x, x)
    v = np.concatenate([np.stack([X, H, Z], -1), n, col], -1).astype(np.float32).reshape(-1, 9)

    ii, jj = np.meshgrid(np.arange(G - 1), np.arange(G - 1), indexing="ij")
    a = (ii * G + jj).ravel()
    b, c, d = a + 1, a + G, a + G + 1
    idx = np.stack([a, c, b, b, c, d], -1).astype(np.uint32).ravel()
    return v, idx
