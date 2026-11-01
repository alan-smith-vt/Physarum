"""Gyroid cradle coupon #3 (user 2026-11-01). Coupon #2 (cradle_squares.py) found a cliff in the flat middle, which
prints first as a floating layer: middle gap <= 0.10 -> the face welds to the bed (whatever the edge gap); >= 0.15 ->
the face never attaches (lost). 60 % gray on the bed top made it worse (the thin sheets cure short = a bigger gap).
So the gap stays where the face reliably attaches, and the BOND STRENGTH is set by how much of each gyroid sheet
touches the face: each sheet tapers from 0.3 to CONTACT over the last TAPER mm under the face (thin gyroid lines, an
even bed-like pattern; no point tips, user).
9 separate cells, each a 40 (chord, +-41.8°) x 20 piece of the R30 face (3 mm shell) on its own gyroid bed (period
10, 0.6 foot), full exposure. Normal gap blended by face angle: G_MID at the middle -> G_EDGE 0.40 at 41.8°.
  - columns (+x): G_MID 0.05 / 0.08 / 0.10
  - rows (-z first): CONTACT 0.20 / 0.15 / 0.10 (sheet width at the face; 0.10 may not print at all)
ID: N dots on the shell's top, N = 1..9 row by row, left to right. Notch at each cell's -x, -z corner.
Judge: face printed and stayed attached; releases by hand; how even the texture is.
Clear blue + Tenacious, 2.3 s. Viewer: python generate.py cradle_squares3.
"""
import numpy as np
import printer
from designs.cradle_coupon import _gyr_dist, R, SHELL, FOOT, LIFT

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

PERIOD, SHEET = 10.0, 0.3
G_MIDS = (0.05, 0.08, 0.10)
G_EDGE = 0.40
CONTACTS = (0.20, 0.15, 0.10)
TAPER = 1.0                            # mm under the face over which each sheet thins to CONTACT
HALF, SIDE = 20.0, 20.0                # cell: face chord 40 (x, the full +-41.8° of the cartridge face) x 20 long (z)
SPACING = 6.0
EDGE = 0.5
NOTCH, DOT_R, DOT_PITCH, MARK_DEPTH = 2.0, 0.45, 2.4, 0.6

YC = LIFT + R
H_TOP = YC - np.sqrt(R * R - HALF * HALF) + 1.0


def _cell(g_mid, g_edge, contact, n):
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
        depth = r - (R + gap)                                                 # distance below the bed's top surface
        t = contact + (SHEET - contact) * np.clip(depth / TAPER, 0.0, 1.0)    # sheet thins toward the face
        bed = below & (depth >= 0) & ((y <= FOOT) | (_gyr_dist(X, y, Z, PERIOD) <= t / 2))
        return np.where(shell | bed, 255, 0).astype(np.uint8)
    return value


def _piece(fn, x0, z0):
    W, Hh = int(np.ceil(2 * HALF / printer.PX_MM)), int(np.ceil(SIDE / printer.PZ_MM))
    N = int(np.ceil(H_TOP / printer.LY_MM))
    xs = ((np.arange(W) + 0.5) * printer.PX_MM)[None, :]
    zs = ((np.arange(Hh) + 0.5) * printer.PZ_MM)[:, None]
    return dict(W=W, H=Hh, N_SLICES=N, OFFSET_X_MM=x0, OFFSET_Y_MM=0.0, OFFSET_Z_MM=z0,
                make_slice=lambda layer: np.broadcast_to(fn(xs, (layer + 0.5) * printer.LY_MM, zs), (Hh, W)).copy())


_NC, _NR = len(G_MIDS), len(CONTACTS)
_X0 = -(_NC * 2 * HALF + (_NC - 1) * SPACING) / 2
_Z0 = -(_NR * SIDE + (_NR - 1) * SPACING) / 2
PIECES = [_piece(_cell(gm, G_EDGE, c, j * _NC + i + 1), _X0 + i * (2 * HALF + SPACING), _Z0 + j * (SIDE + SPACING))
          for j, c in enumerate(CONTACTS) for i, gm in enumerate(G_MIDS)]
