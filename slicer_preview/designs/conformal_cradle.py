"""Conformal cradle, first model for review (user 2026-11-01). The end goal for printing curved faces face down: a rigid
sacrificial SHELL floating a few mm under the face, and only that thin gap filled with a very fine, crushable gyroid
(the big contact area makes an even, bed-like texture; the rest of the support no longer wastes resin on fragile
gyroid). One cell: a 40 (chord, +-41.8°) x 24 piece of the cartridge's R30 face (3 mm shell). Stack, bottom to top:
  1. foot 0.6 on the plate;
  2. egg-crate RIB grid: vertical walls RIB_T thick on a RIB_P grid (x and z), arched drain cut-outs at their feet;
  3. conformal SHELL, SHELL_T thick, CRUSH below the face (offset along the normal), DRAIN_D holes on a DRAIN_P grid;
  4. CRUSH layer: sheet gyroid, period CR_PERIOD, body CR_BODY tapering to CR_CONTACT over the last TAPER under the
     face; gap to the face blended by face angle G_MID (middle) -> G_EDGE (41.8°) (coupon #3/#4 numbers, provisional);
  5. the face (R30, 3 mm shell).
Viewer: python generate.py conformal_cradle. Not for printing yet: crush numbers wait for coupon #4.
"""
import numpy as np
import printer
from designs.cradle_coupon import _gyr_dist

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

R, FACE_T = 30.0, 3.0
HALF, LEN = 20.0, 24.0
FOOT = 0.6
RIB_T, RIB_P, ARCH_R = 0.8, 8.0, 2.0
SHELL_T, DRAIN_D, DRAIN_P = 1.2, 2.0, 6.0
CRUSH = 3.0
CR_PERIOD, CR_BODY, CR_CONTACT, TAPER = 3.0, 0.15, 0.10, 1.0
G_MID, G_EDGE = 0.05, 0.40
EDGE = 0.5
MIN_RIB_H = 2.0                                    # rib height under the lowest point of the shell

LIFT = FOOT + MIN_RIB_H + SHELL_T + CRUSH           # face's lowest point above the plate
YC = LIFT + R
H_TOP = YC - np.sqrt(R * R - HALF * HALF) + 1.0


def cell(x, y, z):
    X, Z = x - HALF, z
    r = np.sqrt(X * X + (y - YC) ** 2)
    inside = (np.abs(X) <= HALF) & (Z >= 0) & (Z <= LEN)
    under = inside & (y < YC) & (r > R)                                # below the face
    face = inside & (r >= R - FACE_T) & (r <= R) & (y <= YC)

    sin_th = np.clip(np.abs(X) / np.maximum(r, 1e-6), 0.0, 1.0)
    gap = G_MID + (G_EDGE - G_MID) * sin_th / (HALF / R)
    depth = r - (R + gap)                                              # below the crush layer's top
    t = CR_CONTACT + (CR_BODY - CR_CONTACT) * np.clip(depth / TAPER, 0.0, 1.0)
    keep = (np.abs(X) <= HALF - EDGE) & (Z >= EDGE) & (Z <= LEN - EDGE)
    crush = under & keep & (depth >= 0) & (r < R + CRUSH) & (_gyr_dist(X, y, Z, CR_PERIOD) <= t / 2)

    shell = under & (r >= R + CRUSH) & (r <= R + CRUSH + SHELL_T)
    hx, hz = (X + HALF) % DRAIN_P - DRAIN_P / 2, (Z % DRAIN_P) - DRAIN_P / 2
    shell &= ~((hx * hx + hz * hz) <= (DRAIN_D / 2) ** 2)              # vertical drain holes

    ribs_zone = under & (r > R + CRUSH + SHELL_T - 0.2) & (y > FOOT - 0.01)
    wx = np.abs(((X + HALF) % RIB_P) - RIB_P / 2) >= RIB_P / 2 - RIB_T / 2   # walls at X = -HALF + k*RIB_P
    wz = np.abs((Z % RIB_P) - RIB_P / 2) >= RIB_P / 2 - RIB_T / 2
    wx |= np.abs(X) >= HALF - RIB_T                                    # closing walls on the cell edges
    wz |= (Z <= RIB_T) | (Z >= LEN - RIB_T)
    # arched drain cut-outs at each wall's foot, centred between crossings
    ax_ = ((Z % RIB_P) - RIB_P / 2)                                     # along a wall of constant X
    az_ = (((X + HALF) % RIB_P) - RIB_P / 2)                            # along a wall of constant Z
    yy = y - FOOT
    arch_x = (ax_ * ax_ + yy * yy <= ARCH_R ** 2)
    arch_z = (az_ * az_ + yy * yy <= ARCH_R ** 2)
    ribs = ribs_zone & ((wx & ~arch_x) | (wz & ~arch_z))

    foot = inside & (y <= FOOT)
    return face | crush | shell | ribs | foot


def _piece(fn, x0, z0):
    W, Hh = int(np.ceil(2 * HALF / printer.PX_MM)), int(np.ceil(LEN / printer.PZ_MM))
    N = int(np.ceil(H_TOP / printer.LY_MM))
    xs = ((np.arange(W) + 0.5) * printer.PX_MM)[None, :]
    zs = ((np.arange(Hh) + 0.5) * printer.PZ_MM)[:, None]
    return dict(W=W, H=Hh, N_SLICES=N, OFFSET_X_MM=x0, OFFSET_Y_MM=0.0, OFFSET_Z_MM=z0,
                make_slice=lambda layer: (np.broadcast_to(fn(xs, (layer + 0.5) * printer.LY_MM, zs), (Hh, W)) * 255).astype(np.uint8))


PIECES = [_piece(cell, -HALF, -LEN / 2)]
