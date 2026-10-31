"""v4 blue (turquoise) resin plate (user 2026-10-31): both camera cartridges + battery door A. The floor waits for
the uni adapter; print pi_box_v4_door_fit first (door rails vs the core grooves), and one face-down cartridge
(no separate cartridge test, user 2026-10-31).
- Cartridges: FACE DOWN, the R30 face on the plate (print up = -n, x = t, z = h), no supports except a few needles
  under the brow's top edge, which starts as an island ~1.1 mm up (16 x 0.3 sliver where the brow slope meets the
  curve).
- Door: STANDING on its grip (+Y) end, so the plate is a vertical wall; the port window's top edge (64 mm bridge) is
  held by branching trees growing from the window's bottom edge: 4 trunks, each splitting twice at 35° -> 16 tips
  4 mm apart, spiked at both ends (snap off, file the inside edges).
- Door fit strip (pi_box_v4_door_fit: wall stub + standing door corner) on the same plate (user 2026-10-31: if the
  fit fails, the core's joining side gets modified, so the door prints anyway).
Viewer: python generate.py pi_box_v4_blue_plate."""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
import numpy as np                                                              # noqa: E402
from geometry import implicit_piece                                             # noqa: E402
from sdf import solid                                                           # noqa: E402
from designs import pi_box_v4_cam_cartridge as K, pi_box_v4_door_a as D         # noqa: E402
from designs.pi_box_bat_rig import Y_MAX, PORT_WINDOW, hole                    # noqa: E402

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0
DIG = 0.25                                  # support tips sink this far into the part


def _seg(x, y, z, a, b, r0, r1):
    """Tapered rod a -> b (radius r0 at a, r1 at b) at print height y; x (1, W), z (H, 1)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    k = np.clip(((x - a[0]) * ab[0] + (y - a[1]) * ab[1] + (z - a[2]) * ab[2]) / (ab @ ab), 0.0, 1.0)
    dx, dy, dz = x - a[0] - k * ab[0], y - a[1] - k * ab[1], z - a[2] - k * ab[2]
    return dx * dx + dy * dy + dz * dz <= (r0 + (r1 - r0) * k) ** 2


# ── cartridge, face down ──
def _face_nmax():
    """The R30 face's outermost n (bisection over a t x h grid): it sits ON the plate (layer 0 must be part)."""
    ts, hs = np.arange(K.T0, K.T1, 0.1)[None, :], np.arange(0.0, K.H_TOP, 0.1)
    def hit(n):
        return any(np.any(solid(K.ROOT, np.full((1, 1), n), ts, float(h))) for h in hs)
    lo, hi = K.N0 + 20.0, K.N1 + 5.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if hit(mid) else (lo, mid)
    return lo


NMAX = _face_nmax() + 0.01                  # the face's outermost point is at the plate
CART_W, CART_H, CART_D = K.T1 - K.T0, NMAX - K.N0, K.H_TOP
NEEDLE_T, NEEDLE_R = (-6.0, 0.0, 6.0), 0.4


def _brow_edge():
    """(h, print height) of the island's lowest point: the brow's top edge at the cap's top (h ~ H_TOP - 0.5)."""
    hs, ns = np.arange(K.H_TOP - 1.5, K.H_TOP, 0.05), np.arange(NMAX - 3.0, NMAX, 0.02)
    best = None
    for h in hs:
        m = np.broadcast_to(solid(K.ROOT, ns[None, :], np.zeros((1, 1)), float(h)), (1, len(ns)))[0]
        if m.any():
            n = float(ns[m][-1])
            if best is None or n > best[1]:
                best = (float(h), n)
    return best[0], NMAX - best[1]


BROW_H, BROW_Y = _brow_edge()


def cart_face_down(x, y, z):
    n = NMAX - y
    out = np.zeros(np.broadcast_shapes(x.shape, z.shape), bool)
    for i, zi in enumerate(z[:, 0]):                  # h runs along z: row by row (solid takes a scalar h)
        out[i] = np.broadcast_to(solid(K.ROOT, np.full((1, 1), n), x + K.T0, float(zi)), (1, x.shape[1]))[0]
    if y <= BROW_Y + DIG:                             # needles under the brow edge island
        for t in NEEDLE_T:
            out |= (x - (t - K.T0)) ** 2 + (z - BROW_H) ** 2 <= (NEEDLE_R + max(0.6 - y, 0.0)) ** 2
    return out


# ── door, standing on its grip end: print up = Y_MAX - Yr, x = X + PLATE_X, z = d - D_TOP ──
TRUNK_X = (-24.0, -8.0, 8.0, 24.0)
SPLIT = (4.0, 2.0)                          # horizontal offsets of the two splits (tips at trunk +-2, +-6)
TAN = np.tan(np.radians(35.0))
R_TRUNK, R_MID, R_TIP, R_POINT = 0.8, 0.6, 0.45, 0.25
SPIKE_L = 1.5                               # trunk foot tapers to R_POINT on the window's bottom edge (user)
DC = (D.PLATE_D[0] + D.PLATE_D[1]) / 2 - D.D_TOP     # the trees lie in the plate's mid-plane


def _trees():
    """[(a, b, r0, r1)] rods in door print coords."""
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    y_bot, y_top = Y_MAX - wy1 - DIG, Y_MAX - wy0 + DIG      # window bottom edge (plate below) .. top edge (bridge)
    y2 = y_top - SPLIT[1] / TAN                              # second split
    y1 = y2 - SPLIT[0] / TAN                                 # first split
    x0 = D.PLATE_X
    rods = []
    for c in TRUNK_X:
        rods.append(((x0 + c, y_bot, DC), (x0 + c, y_bot + SPIKE_L, DC), R_POINT, R_TRUNK))   # spiked foot
        rods.append(((x0 + c, y_bot + SPIKE_L, DC), (x0 + c, y1, DC), R_TRUNK, R_TRUNK))
        for s1 in (-1, 1):
            m = c + s1 * SPLIT[0]
            rods.append(((x0 + c, y1, DC), (x0 + m, y2, DC), R_MID, R_MID))
            for s2 in (-1, 1):
                rods.append(((x0 + m, y2, DC), (x0 + m + s2 * SPLIT[1], y_top, DC), R_TIP, R_POINT))
    return rods


TREES = _trees()


def door_stand(x, y, z):
    X, Yr, d = x - D.PLATE_X, Y_MAX - y, z + D.D_TOP
    aX = np.abs(X)
    inY = (Yr <= Y_MAX) & (Yr >= D.Y_LO)
    web = (d < D.PLATE_D[0]) & (aX >= D.HX - D.WEB_T) & (aX <= D.HX)
    lead = max(D.TONGUE_LEAD - (Yr - D.Y_LO), 0.0)
    hook = (d >= D.TONGUE_D[0]) & (d <= D.TONGUE_D[1]) & (aX >= D.HX - D.WEB_T) & (aX <= D.TONGUE_X - lead)
    plate = (d >= D.PLATE_D[0]) & (d <= D.PLATE_D[1]) & (aX <= D.PLATE_X)
    grip = (d > D.PLATE_D[1]) & (d <= D.GRIP['d1']) & (aX <= D.GRIP['x']) & (Yr >= D.GRIP['y0'])
    s = (web | hook | plate | grip) & inY & (d >= D.D_TOP)
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    s &= ~((X >= wx0) & (X <= wx1) & (Yr >= wy0) & (Yr <= wy1))
    s &= ~hole(*D.SCREW_XY, D.CAPTIVE_D, X, Yr)
    for a, b, r0, r1 in TREES:
        if min(a[1], b[1]) - r0 <= y <= max(a[1], b[1]) + r0:
            s = s | _seg(x, y, z, a, b, r0, r1)
    return s


DOOR_W, DOOR_H, DOOR_D = 2 * D.PLATE_X, Y_MAX - D.Y_LO, D.GRIP['d1'] - D.D_TOP

# plate ~211 x 118 (x, z): the door along the front, the two face-down cartridges behind it, the fit strip at +x
from designs import pi_box_v4_door_fit as _F                                    # noqa: E402
PIECES = [{**_F.PIECES[0], 'OFFSET_X_MM': 56.0, 'OFFSET_Z_MM': -30.0},
          {**_F.PIECES[1], 'OFFSET_X_MM': 56.0, 'OFFSET_Z_MM': 0.0},
          implicit_piece(DOOR_W, DOOR_H, DOOR_D, door_stand, x0=-DOOR_W / 2, z0=-52.0),
          implicit_piece(CART_W, CART_H, CART_D, cart_face_down, x0=-CART_W - 4.0, z0=-32.0),
          implicit_piece(CART_W, CART_H, CART_D, cart_face_down, x0=4.0, z0=-32.0)]
