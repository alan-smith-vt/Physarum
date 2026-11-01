"""Gyroid cradle coupon #4 (user 2026-11-01): a CRUSHABLE bed. Goal (user): a support that crumples by hand when the
part comes off, like Elegoo's lattice supports; the bed pattern printed into the face is fine, point dots are not.
Coupon #3 (cradle_squares3): thin contact lines (0.10, middle gap 0.05) gave clean edges (#7 nearly perfect), but the
flat middle welded in every cell and the 0.3 sheets were far too rigid to crush. A doubly curved sheet is much stiffer
than its thickness suggests (user), so the whole bed goes thin and denser:
  - gyroid period 5 (fixed);
  - rows (-z first): BODY sheet thickness 0.10 / 0.15 / 0.20 (the whole bed, so it crushes);
  - columns (+x): CONTACT 0.05 / 0.10 (each sheet tapers from BODY to CONTACT over the last TAPER mm under the face);
  - normal gap blended by face angle: G_MID 0.05 at the middle -> G_EDGE 0.40 at 41.8°; full exposure; 0.6 foot.
6 separate cells, each a 40 (chord, +-41.8°) x 20 piece of the R30 face (3 mm shell). ID: N dots, 1..6 row by row.
Judge: face printed + attached; does the bed crush by hand (fingers / rolling it off); texture, middle vs edges.
Clear blue + Tenacious, 2.3 s. Viewer: python generate.py cradle_squares4.
"""
import numpy as np
import printer
from designs.cradle_coupon import _gyr_dist, R, SHELL, FOOT, LIFT

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

PERIOD = 5.0
G_MID, G_EDGE = 0.05, 0.40
BODIES = (0.10, 0.15, 0.20)
CONTACTS = (0.05, 0.10)
TAPER = 1.0                            # mm under the face over which each sheet thins to CONTACT
HALF, SIDE = 20.0, 20.0                # cell: face chord 40 (x, the full +-41.8° of the cartridge face) x 20 long (z)
SPACING = 6.0
EDGE = 0.5
NOTCH, DOT_R, DOT_PITCH, MARK_DEPTH = 2.0, 0.45, 2.4, 0.6

YC = LIFT + R
H_TOP = YC - np.sqrt(R * R - HALF * HALF) + 1.0


def _cell(g_mid, g_edge, body, contact, n):
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
        t = contact + (body - contact) * np.clip(depth / TAPER, 0.0, 1.0)     # sheet thins toward the face
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


_NC, _NR = len(CONTACTS), len(BODIES)
_X0 = -(_NC * 2 * HALF + (_NC - 1) * SPACING) / 2
_Z0 = -(_NR * SIDE + (_NR - 1) * SPACING) / 2
PIECES = [_piece(_cell(G_MID, G_EDGE, b, c, j * _NC + i + 1), _X0 + i * (2 * HALF + SPACING), _Z0 + j * (SIDE + SPACING))
          for j, b in enumerate(BODIES) for i, c in enumerate(CONTACTS)]
