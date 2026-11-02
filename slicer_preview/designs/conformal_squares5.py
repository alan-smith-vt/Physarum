"""Cradle coupon #5 (user 2026-11-01): the CONFORMAL stack (designs/conformal_cradle.py) with a gray-tuned face.
Coupon #4 (cradle_squares4): crushable by hand achieved, but every failed cell broke at the foot-to-gyroid ROOT (so
contact 0.05 / body 0.10 were never really tested), the flat middle still welded (breaking left ~2 mm stubs: the
face/tip joint was stronger than the sheets), and the sides' finish was poor too.
Every cell: 40 (chord, +-41.8°) x 20 of the R30 face (3 mm) on: foot 0.6 -> egg-crate ribs (0.8 walls, 8 grid, arched
drains) -> perforated conformal shell 1.2 -> CRUSH layer (sheet gyroid period 5, CRUSH mm) -> the face.
Crush sheet thickness along the layer: ROOT 0.4 at the shell, tapering to BODY 0.15 over ROOT_L, BODY up to TAPER under
the face, then to CONTACT at the face. Normal gap blended by face angle: G_MID 0.10 (2 layers) -> G_EDGE 0.40 (41.8°).
FACE GRAY: the face's down-facing skin (normal depth SKIN = 2 layers) printed at a gray blended by face angle:
GRAY_MID at the flat middle -> GRAY_EDGE at 41.8° (less overcure into the tips = a weaker face/tip joint).
Cells (ID = N dots; 4 per row, row by row from -z, left to right):
  1..6: contact 0.10, GRAY_MID 100 / 85 / 70 (columns) x GRAY_EDGE 100 / 85 (rows)
  7..8: contact 0.05, GRAY_MID 85, GRAY_EDGE 100 / 85
Judge: printed + attached; the root held; the bed crushes off; the break is AT the face (no stubs); middle and side
finish. Clear blue + Tenacious, 2.3 s. Viewer: PREVIEW_SCALE=6 python generate.py conformal_squares5.
"""
import numpy as np
import printer
from designs.cradle_coupon import _gyr_dist
from designs import conformal_cradle as CC

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

R, FACE_T, HALF, LEN = 30.0, 3.0, 20.0, 20.0
FOOT, RIB_T, RIB_P, ARCH_R = CC.FOOT, CC.RIB_T, CC.RIB_P, CC.ARCH_R
SHELL_T, DRAIN_D, DRAIN_P = CC.SHELL_T, CC.DRAIN_D, CC.DRAIN_P
CRUSH, PERIOD = 5.0, 5.0
ROOT, ROOT_L, BODY, TAPER = 0.4, 1.0, 0.15, 1.0
G_MID, G_EDGE = 0.10, 0.40
SKIN = 0.10                                         # face skin depth printed at the reduced gray (2 layers)
EDGE, MIN_RIB_H = 0.5, 2.0
NOTCH, DOT_R, DOT_PITCH, MARK_DEPTH = 2.0, 0.45, 2.4, 0.6
SPACING = 6.0
LIFT = FOOT + MIN_RIB_H + SHELL_T + CRUSH
YC = LIFT + R
H_TOP = YC - np.sqrt(R * R - HALF * HALF) + 1.0
S_MAX = HALF / R                                    # sin of the edge angle (41.8°)

CELLS = [(0.10, gm, ge) for ge in (255, 217) for gm in (255, 217, 179)] + [(0.05, 217, ge) for ge in (255, 217)]
#        (contact, GRAY_MID, GRAY_EDGE); 217 = 85 %, 179 = 70 %


def _cell(contact, gray_mid, gray_edge, n):
    def value(x, y, z):
        X, Z = x - HALF, z
        r = np.sqrt(X * X + (y - YC) ** 2)
        inside = (np.abs(X) <= HALF) & (Z >= 0) & (Z <= LEN)
        under = inside & (y < YC) & (r > R)
        s = np.clip(np.abs(X) / np.maximum(r, 1e-6), 0.0, 1.0) / S_MAX              # 0 middle .. 1 edge
        # face + ID dots + notch
        face = inside & (r >= R - FACE_T) & (r <= R) & (y <= YC)
        face &= ~((X < -HALF + NOTCH) & (Z < NOTCH))
        dots = np.zeros(np.broadcast_shapes(X.shape, Z.shape), bool)
        for k in range(n):
            row, col = k // 4, k % 4
            ncol = min(n - row * 4, 4)
            dots |= (X - (col - (ncol - 1) / 2) * DOT_PITCH) ** 2 + (Z - LEN / 2 - (row - (0.5 if n > 4 else 0.0)) * DOT_PITCH) ** 2 <= DOT_R ** 2
        face &= ~(dots & (r < R - FACE_T + MARK_DEPTH))
        skin = face & (r > R - SKIN)
        gray = np.rint(gray_mid + (gray_edge - gray_mid) * np.clip(s, 0, 1)).astype(np.uint8)
        # crush layer: root -> body -> contact
        gap = G_MID + (G_EDGE - G_MID) * s
        top_r, bot_r = R + gap, R + CRUSH
        depth = r - top_r                                                           # below the crush top
        up = bot_r - r                                                              # above the shell
        t = np.where(depth < TAPER, contact + (BODY - contact) * np.clip(depth / TAPER, 0, 1),
                     BODY + (ROOT - BODY) * np.clip(1.0 - up / ROOT_L, 0, 1))
        keep = (np.abs(X) <= HALF - EDGE) & (Z >= EDGE) & (Z <= LEN - EDGE)
        crush = under & keep & (depth >= 0) & (r < bot_r + 0.05) & (_gyr_dist(X, y, Z, PERIOD) <= t / 2)
        # shell, ribs, foot (as conformal_cradle)
        shell = under & (r >= bot_r) & (r <= bot_r + SHELL_T)
        hx, hz = (X + HALF) % DRAIN_P - DRAIN_P / 2, (Z % DRAIN_P) - DRAIN_P / 2
        shell &= ~((hx * hx + hz * hz) <= (DRAIN_D / 2) ** 2)
        ribs_zone = under & (r > bot_r + SHELL_T - 0.2) & (y > FOOT - 0.01)
        wx = (np.abs(((X + HALF) % RIB_P) - RIB_P / 2) >= RIB_P / 2 - RIB_T / 2) | (np.abs(X) >= HALF - RIB_T)
        wz = (np.abs((Z % RIB_P) - RIB_P / 2) >= RIB_P / 2 - RIB_T / 2) | (Z <= RIB_T) | (Z >= LEN - RIB_T)
        yy = y - FOOT
        ax_, az_ = (Z % RIB_P) - RIB_P / 2, ((X + HALF) % RIB_P) - RIB_P / 2
        ribs = ribs_zone & ((wx & ~(ax_ * ax_ + yy * yy <= ARCH_R ** 2)) | (wz & ~(az_ * az_ + yy * yy <= ARCH_R ** 2)))
        foot = inside & (y <= FOOT)
        out = np.where(face | crush | shell | ribs | foot, 255, 0).astype(np.uint8)
        return np.where(skin, gray, out).astype(np.uint8)
    return value


def _piece(fn, x0, z0):
    W, Hh = int(np.ceil(2 * HALF / printer.PX_MM)), int(np.ceil(LEN / printer.PZ_MM))
    N = int(np.ceil(H_TOP / printer.LY_MM))
    xs = ((np.arange(W) + 0.5) * printer.PX_MM)[None, :]
    zs = ((np.arange(Hh) + 0.5) * printer.PZ_MM)[:, None]
    return dict(W=W, H=Hh, N_SLICES=N, OFFSET_X_MM=x0, OFFSET_Y_MM=0.0, OFFSET_Z_MM=z0,
                make_slice=lambda layer: np.broadcast_to(fn(xs, (layer + 0.5) * printer.LY_MM, zs), (Hh, W)).copy())


_NC = 4
_X0 = -(_NC * 2 * HALF + (_NC - 1) * SPACING) / 2
PIECES = [_piece(_cell(c, gm, ge, i + 1), _X0 + (i % _NC) * (2 * HALF + SPACING), (i // _NC - 1) * (LEN + SPACING))
          for i, (c, gm, ge) in enumerate(CELLS)]
