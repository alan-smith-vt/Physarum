"""Gyroid cradle coupon #2 (user 2026-11-01), after cradle_coupon (strips): there only gap 0.20 released, smaller gaps
welded, and the middle bands debonded mid-print and let the face deform. Period 10 gave the best finish.

12 separate cells (no shared face or bed, so no cell influences another), each a 40 (chord, +-41.8°) x 20 piece of the
cartridge's R30 face (3 mm shell) lying face down on its own sheet-gyroid bed (period 10, sheet 0.3, 0.6 foot).
Last time (uniform normal gap 0.05..0.20) the flat MIDDLE debonded mid-print and EVERY EDGE bonded solid (even 0.20
scarred). So the gap now depends on the face angle th (0 at the middle, 41.8° at the edges), measured normal to the face:
    gap(th) = G_MID + (G_EDGE - G_MID) * sin(th) / sin(41.8°)
  - G_MID 0.05 / 0.10 / 0.15: small, so overcure tacks the flat middle (printed first) to the bed;
  - G_EDGE 0.20 / 0.40: 0.20 = the last coupon's only releasing gap (lightly scarred; baseline, maybe ideal with 60 %
    gray), 0.40 = one step past it (user: don't jump so far that everything debonds);
  - interface gray: the top 0.3 mm of the bed under the face at 100 % or 60 % pixel gray (a weaker tack).
Layout: 3 columns = G_MID (rising with +x); rows (-z first): G_EDGE 0.20 @100 %, 0.40 @100 %, 0.20 @60 %, 0.40 @60 %.
ID: N dots on the shell's top, N = 1..12 in that order (row by row, left to right). Notch at each cell's -x, -z corner.
Judge: middle stayed attached through the print (no warp), edges release by hand, texture even across the width.
Clear blue + Tenacious, 2.3 s. Viewer: python generate.py cradle_squares (60 % pixels show darker).
"""
import numpy as np
import printer
from designs.cradle_coupon import _gyr_dist, R, SHELL, FOOT, LIFT

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

PERIOD, SHEET = 10.0, 0.3
G_MIDS = (0.05, 0.10, 0.15)
G_EDGES = (0.20, 0.40)
GRAYS = (255, 153)                     # 100 %, 60 %
INTERFACE = 0.3                        # mm of bed under the face printed at the cell's gray
HALF, SIDE = 20.0, 20.0                # cell: face chord 40 (x, the full +-41.8° of the cartridge face) x 20 long (z)
SPACING = 6.0
EDGE = 0.5
NOTCH, DOT_R, DOT_PITCH, MARK_DEPTH = 2.0, 0.45, 2.4, 0.6

YC = LIFT + R
H_TOP = YC - np.sqrt(R * R - HALF * HALF) + 1.0


def _cell(g_mid, g_edge, gray, n):
    def value(x, y, z):
        """uint8 (H, W) for local x (1, W), z (H, 1), print height y."""
        X, Z = x - HALF, z
        r = np.sqrt(X * X + (y - YC) ** 2)
        inside = (np.abs(X) <= HALF) & (Z >= 0) & (Z <= SIDE)
        shell = inside & (r >= R - SHELL) & (r <= R) & (y <= YC)
        shell &= ~((X < -HALF + NOTCH) & (Z < NOTCH))
        cx, cz = 0.0, SIDE / 2
        dots = np.zeros(np.broadcast_shapes(X.shape, Z.shape), bool)
        for k in range(n):                                   # 2 rows of up to 4
            row, col = k // 4, k % 4
            ncol = min(n - row * 4, 4)
            px = cx + (col - (ncol - 1) / 2) * DOT_PITCH
            pz = cz + (row - (0.5 if n > 4 else 0.0)) * DOT_PITCH
            dots |= (X - px) ** 2 + (Z - pz) ** 2 <= DOT_R ** 2
        shell &= ~(dots & (r < R - SHELL + MARK_DEPTH))
        sin_th = np.clip(np.abs(X) / np.maximum(r, 1e-6), 0.0, 1.0)            # face angle from the bottom line
        gap = g_mid + (g_edge - g_mid) * sin_th / (HALF / R)                   # normal gap, blended by angle
        below = (np.abs(X) <= HALF - EDGE) & (Z >= EDGE) & (Z <= SIDE - EDGE) & (y < YC) & (r > R)
        bed = below & (r >= R + gap) & ((y <= FOOT) | (_gyr_dist(X, y, Z, PERIOD) <= SHEET / 2))
        iface = bed & (r < R + gap + INTERFACE) & (y > FOOT)
        out = np.where(shell | (bed & ~iface), 255, 0).astype(np.uint8)
        return np.where(iface, gray, out).astype(np.uint8)
    return value


def _piece(fn, x0, z0):
    W, Hh = int(np.ceil(2 * HALF / printer.PX_MM)), int(np.ceil(SIDE / printer.PZ_MM))
    N = int(np.ceil(H_TOP / printer.LY_MM))
    xs = ((np.arange(W) + 0.5) * printer.PX_MM)[None, :]
    zs = ((np.arange(Hh) + 0.5) * printer.PZ_MM)[:, None]
    return dict(W=W, H=Hh, N_SLICES=N, OFFSET_X_MM=x0, OFFSET_Y_MM=0.0, OFFSET_Z_MM=z0,
                make_slice=lambda layer: np.broadcast_to(fn(xs, (layer + 0.5) * printer.LY_MM, zs), (Hh, W)).copy())


_NC, _ROWS = len(G_MIDS), [(ge, gr) for gr in GRAYS for ge in G_EDGES]
_X0 = -(_NC * 2 * HALF + (_NC - 1) * SPACING) / 2
_Z0 = -(len(_ROWS) * SIDE + (len(_ROWS) - 1) * SPACING) / 2
PIECES = [_piece(_cell(gm, ge, gr, j * _NC + i + 1), _X0 + i * (2 * HALF + SPACING), _Z0 + j * (SIDE + SPACING))
          for j, (ge, gr) in enumerate(_ROWS) for i, gm in enumerate(G_MIDS)]
