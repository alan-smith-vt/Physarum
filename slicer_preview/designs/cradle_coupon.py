"""Gyroid cradle coupon (user 2026-10-31): calibrates the gyroid support interface under a curved, visible face before the
camera cartridges are reprinted face down on a cradle. Print in the cartridges' resin (clear blue + Tenacious, 2.3 s).

A 40 mm long slice of the cartridge's R30 face (3 mm shell, chord ~41 mm), lying face down on a sheet-gyroid bed that
follows the curve with a GAP to the face. 8 cells:
  along z (4 bands, 10 mm each):  GAP 0.05 / 0.10 / 0.15 / 0.20 mm   (from the notched end)
  along x (2 halves):             gyroid sheet 0.3 mm (notch side, -x) / 0.4 mm (+x)
Bed: PERIOD 5 mm (bumper_coupon's middle cell), 0.6 mm solid foot, face lifted LIFT above the plate.
Notch: a 2 mm corner notch in the shell at the -x, -z corner marks the 0.05 / 0.3 cell. Top face: engraved lines on
the cell borders and N dots per cell (N = cell number: 1..4 = sheet 0.3, gaps 0.05..0.20; 5..8 = sheet 0.4, same order).
Judge each cell: 1) does the face snap off cleanly (not welded, didn't fall off mid-print); 2) is the gyroid
pattern printed into the face (look at a raking light; does a light sand clear it); 3) is the curve still round
(R30 template / straight edge), no sag between contacts. Winner: the smallest gap that releases cleanly with no
visible pattern; it sets the cradle defaults for the cartridge reprint.
Viewer: python generate.py cradle_coupon."""
import numpy as np
from geometry import implicit_piece

SUPPORTS = False
EXPOSURE_S = 2.3                 # clear blue + Tenacious (the cartridges' resin)
BOTTOM_EXPOSURE_S = 20.0

R, SHELL, HALF_W, LENGTH = 30.0, 3.0, 20.0, 40.0
GAPS = (0.05, 0.10, 0.15, 0.20)
SHEETS = (0.3, 0.4)
PERIOD, FOOT, LIFT = 5.0, 0.6, 3.0
NOTCH = 2.0

YC = LIFT + R                                    # cylinder axis (print up = y), axis along z
H = YC - np.sqrt(R * R - HALF_W * HALF_W) + 1.0  # top of the part: just above the shell's edges
W = 2 * HALF_W + 2.0


def _gyr_dist(x, y, z):
    k = 2 * np.pi / PERIOD
    sx, cx, sy, cy, sz, cz = np.sin(k * x), np.cos(k * x), np.sin(k * y), np.cos(k * y), np.sin(k * z), np.cos(k * z)
    g = sx * cy + sy * cz + sz * cx
    gn = k * np.sqrt((cx * cy - sz * sx) ** 2 + (cy * cz - sx * sy) ** 2 + (cz * cx - sy * sz) ** 2)
    return np.abs(g) / np.maximum(gn, 1e-6)


LINE_W, MARK_DEPTH, DOT_R, DOT_PITCH = 0.5, 0.6, 0.45, 1.6


def _marks(X, Z):
    """Engraved on the shell's top (inner) face: lines on the cell borders, and in each cell N dots, N = its cell
    number: 1..4 = sheet 0.3 (-x half) with gaps 0.05..0.20, 5..8 = sheet 0.4 (+x half), same gap order."""
    bl = LENGTH / len(GAPS)
    m = np.abs(X) <= LINE_W / 2
    for i in range(1, len(GAPS)):
        m = m | (np.abs(Z - i * bl) <= LINE_W / 2)
    for j, sx in enumerate((-1, 1)):
        for i in range(len(GAPS)):
            n = j * len(GAPS) + i + 1
            cx, cz = sx * HALF_W / 2, (i + 0.5) * bl
            for k in range(n):                                    # two rows of up to 4
                row, col = k // 4, k % 4
                ncol = min(n - row * 4, 4)
                px = cx + (col - (ncol - 1) / 2) * DOT_PITCH * 1.5
                pz = cz + (row - (0.5 if n > 4 else 0.0)) * DOT_PITCH * 1.5
                m = m | ((X - px) ** 2 + (Z - pz) ** 2 <= DOT_R ** 2)
    return m


def solid(x, y, z):
    X, Z = x - W / 2, z                         # X centred, Z 0..LENGTH
    r = np.sqrt(X * X + (y - YC) ** 2)
    inside = (np.abs(X) <= HALF_W) & (Z >= 0) & (Z <= LENGTH)
    shell = inside & (r >= R - SHELL) & (r <= R) & (y <= YC)
    shell &= ~((X < -HALF_W + NOTCH) & (Z < NOTCH))                       # orientation notch
    shell &= ~(_marks(X, Z) & (r < R - SHELL + MARK_DEPTH))               # cell lines + ID dots, on the top face
    band = np.clip(np.floor(Z / (LENGTH / len(GAPS))).astype(int), 0, len(GAPS) - 1)
    gap = np.asarray(GAPS)[band]
    sheet = np.where(X < 0, SHEETS[0], SHEETS[1])
    below = inside & (r > R) & (y < YC)                                   # under the face (its shadow)
    bed = below & (r >= R + gap) & ((y <= FOOT) | (_gyr_dist(X, y, Z) <= sheet / 2))
    return shell | bed


PIECES = [implicit_piece(W, H, LENGTH, solid, x0=-W / 2, z0=-LENGTH / 2)]
