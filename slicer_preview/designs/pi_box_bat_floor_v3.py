"""v3 CONCEPT (2026-10-15): sliding floor for designs/pi_box_bat_core_v3.py. Slides in from -Y between the camera
walls, its two edge tongues in the walls' inner grooves, until the grooves stop it near +Y. 4 × M4 up into the
core's boss inserts (primary load path); the wall ledges under the grooves are the safety net.

Plate: sheet gyroid (the walls' 12 mm period, FILL_T sparser) outside a solid keep footprint (perimeter band,
tongues, M4 rings, well ring, port-window frame, spokes + diamond). No collar, rails or nubs (they'd hit the
tube / webs while sliding). Printed outside face down.
Rig -> print: (X, Yr, d) -> (x = Yr - Y_MIN, y = DEPTH + FLOOR_T3 - d, z = X + W/2).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import (X_IN, Y_MIN, Y_MAX, DEPTH, WELL_R, BOSSES, PORT_WINDOW, GYROID_PERIOD,
                                    RESIN_HOLE_COMP, hole)
from designs.pi_box_bat_core_v3 import FLOOR_T3, FLOOR_TONGUE, FLOOR_CLEAR, FLOOR_STOP_Y, M4_CLEAR

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

FILL_T = 0.27
OVERSIZE = RESIN_HOLE_COMP / 2           # external faces print ~0.35 oversize (diameter): per face
BODY_X = X_IN - FLOOR_CLEAR - OVERSIZE   # |X| of the plate between the walls
TONGUE_X = X_IN + FLOOR_TONGUE - FLOOR_CLEAR - OVERSIZE
TONGUE_Y1 = FLOOR_STOP_Y - FLOOR_CLEAR - OVERSIZE
TONGUE_T0, TONGUE_T1 = FLOOR_CLEAR, FLOOR_T3 - FLOOR_CLEAR    # tongue thickness band (print y): slack top and bottom
TONGUE_BEVEL = 0.5                       # 45° on both faces of the tongue's outer edge (also takes the elephant foot)
TONGUE_LEAD = 2.0                        # tongue tip (+Y) tapers in by this over its last 2 mm, and thins at its faces
FRONT_LEAD = 0.8                         # 45° on both faces of the plate's leading (+Y) edge
W = 2 * TONGUE_X


def _keep(step=0.1):
    from scipy.ndimage import gaussian_filter
    xs = np.arange(-TONGUE_X, TONGUE_X + step, step)
    ys = np.arange(Y_MIN, Y_MAX + step, step)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    b = 5.0
    req = (np.abs(XX) >= BODY_X - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)
    for bx, by in BOSSES:
        req |= np.hypot(XX - bx, YY - by) <= 6.0
    req |= np.hypot(XX, YY) <= WELL_R + 4.0
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    req |= (XX >= wx0 - 2) & (XX <= wx1 + 2) & (YY >= wy0 - 2) & (YY <= wy1 + 2)

    def seg(ax, ay, bx, by, w=3.0):
        vx, vy = bx - ax, by - ay
        t = np.clip(((XX - ax) * vx + (YY - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        return np.hypot(XX - ax - t * vx, YY - ay - t * vy) <= w / 2
    ye, xe = Y_MAX - b, BODY_X - b
    req |= seg(0, -ye, 0, -WELL_R) | seg(0, ye, 0, WELL_R)
    mids = [(0, -ye), (xe, 0), (0, ye), (-xe, 0)]
    for i in range(4):
        req |= seg(*mids[i], *mids[(i + 1) % 4])
    return step, gaussian_filter(req.astype(float), 2.0 / step) > 0.3


_KS, _KEEP = _keep()


def floor(x, y, z):
    X, Yr, d = z - TONGUE_X, x + Y_MIN, DEPTH + FLOOR_T3 - y
    aX = np.abs(X)
    s = (aX <= BODY_X) & (Yr <= Y_MAX)
    s &= Yr <= Y_MAX - max(FRONT_LEAD - min(y, FLOOR_T3 - y), 0.0)     # leading (+Y) edge of the plate
    if TONGUE_T0 <= y <= TONGUE_T1:
        ty = min(y - TONGUE_T0, TONGUE_T1 - y)                   # depth into the tongue from its nearer face
        ch = max(TONGUE_BEVEL - ty, 0.0)
        lead = np.maximum(TONGUE_LEAD - (TONGUE_Y1 - Yr), 0.0)  # tip tapers in over its last TONGUE_LEAD
        s |= (aX <= TONGUE_X - ch - lead) & (Yr <= TONGUE_Y1 - max(TONGUE_LEAD - ty, 0.0) * 0.5)
    s &= ~hole(0, 0, 2 * WELL_R, X, Yr)
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    s &= ~((X >= wx0) & (X <= wx1) & (Yr >= wy0) & (Yr <= wy1))
    for bx, by in BOSSES:
        s &= ~hole(bx, by, M4_CLEAR, X, Yr)
    keep = np.asarray(_KEEP)                           # np may be cupy under to_goo --gpu
    i = np.clip((z / _KS).astype(int), 0, keep.shape[0] - 1)
    j = np.clip((x / _KS).astype(int), 0, keep.shape[1] - 1)
    k = 2 * np.pi / GYROID_PERIOD
    g = np.sin(k * X) * np.cos(k * Yr) + np.sin(k * Yr) * np.cos(k * d) + np.sin(k * d) * np.cos(k * X)
    return s & (keep[i, j] | (np.abs(g) <= FILL_T))


PIECES = [implicit_piece(Y_MAX - Y_MIN, FLOOR_T3, W, floor, x0=-(Y_MAX - Y_MIN) / 2, z0=-W / 2)]
