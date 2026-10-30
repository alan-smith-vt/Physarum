"""v4 camera cartridge PROPOSAL (2026-10-29, not approved, don't print): the v3 cartridge re-done for the v4 core's
camera lands (designs/pi_box_v4_core.py). Two identical parts (the -X one is the mirror image: print one mirrored).

  - CAP: flush with the core's R30 round, with the 1 mm land in the middle; its inner side is the 45° countersunk
    seat (rectangular frustum ±FL_T × FL_D at the seat plane, narrowing to the body). It slides in along the land
    normal (no undercut) and centres itself in the core's countersink. 3 × M2 through the cap (counterbored) into
    M2 inserts set in the core's pad from inside (nuts) -> M2×5 [MEASURE].
  - BODY: the opening's prism, from the seat to a back face parallel to the board: the OV9281 board is pitched 24°
    up inside it (yaw 30° comes from the land). Board on 4 bosses (2 tall) at its holes, M2 screws from outside
    (counterbores in the cap) through the body and the board into an M2 insert threaded on behind it as a nut
    (as v3) -> ~M2×25 [MEASURE]. Lens Ø14 × 18 in a Ø19 bore, lens front LENS_RECESS behind the land; a 76°
    window cone in front of it (the 125° × 78° fisheye's corners are at ≤ 74.4° off axis).
  - Ribbon connector edge DOWN, behind the board: the back is open.
Print: standing on the cap's bottom face (rig d max) with rig 'up' as print up: every face is ≤ 45° off vertical or
faces up (the body's bottom is the countersink's 45° face continued; the lens bore has a 45° peaked roof).
Tree frame: (X', Y', d') = (n, t, h) in the +X camera's land frame, h = FL_D[1] - d (height off the plate).
"""
import numpy as np
from sdf import Box, Cyl, Half, Fn, union, inter, sub, sdf_piece, solid, INF
from designs.pi_box_bat_rig import RESIN_HOLE_COMP, CAM_BOSS_R, CAM_BOSS_H, CAM_WINDOW
from designs.pi_box_v4_core import (R_FRONT, LAND, SEAT, CSK, OPEN_T, OPEN_D, FL_T, FL_D, CART_SCREWS, F_ND, U_ND,
                                    LENS_Q, BOARD_B, RC, N2, T2, X_OUT, Y_MIN)

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

SCREW_CLEAR, HEAD_CB = 2.2, 4.2        # M2 socket cap (head Ø3.8)
BOARD_HOLES = [(sy * 10.5, sv * 6.25) for sy in (-1, 1) for sv in (-1, 1)]   # (t, along u from the lens axis)
BOARD_CB_C = CAM_BOSS_H + 4.0          # board-screw counterbore floor, along the axis from the board face
F_PX, WIN_MARGIN, PUPIL_R = 294.0, 15.0, 2.0   # px; margin ~3°; pupil + lens-centring slop (mm)
FC = np.array([F_ND[0], 0.0, F_ND[1]])   # axis in (n, t, d)
UC = np.array([U_ND[0], 0.0, U_ND[1]])
TC = np.array([0.0, 1.0, 0.0])
Q3 = np.array([LENS_Q[0], 0.0, LENS_Q[1]])
B3 = np.array([BOARD_B[0], 0.0, BOARD_B[1]])
H_TOP = FL_D[1] - (FL_D[0] - 0.5)


def H(an, at, ad, c):
    """Half-space an*n + at*t + ad*d <= c, in the tree frame (d = FL_D[1] - h)."""
    return Half(an, at, -ad, c - ad * FL_D[1])


def axis_cyl(o, a, r, c0, c1, peak=False):
    """Cylinder along unit a (n, t, d) from o; peak: 45° roof toward rig-up (-d) for printing."""
    o, a = np.asarray(o, float), np.asarray(a, float)
    up = np.array([0.0, 0.0, -1.0]) - a * (-a[2])
    up /= np.linalg.norm(up)
    side = np.cross(a, up)
    ends = [o + a * c0, o + a * c1]
    rr = r * (np.sqrt(2) if peak else 1.0)
    pts = np.array([e + k1 * rr * up + k2 * rr * side for e in ends for k1 in (-1, 1) for k2 in (-1, 1)])
    lo, hi = pts.min(0) - 0.5, pts.max(0) + 0.5

    def f(X, Y, h, xp):
        vn, vt, vd = X - o[0], Y - o[1], (FL_D[1] - h) - o[2]
        c = vn * a[0] + vt * a[1] + vd * a[2]
        pu = vn * up[0] + vt * up[1] + vd * up[2]
        ps = vn * side[0] + vt * side[1] + vd * side[2]
        rad = xp.sqrt(pu * pu + ps * ps) - r
        if peak:
            rad = xp.minimum(rad, xp.maximum((xp.abs(ps) + pu) / np.sqrt(2) - r, -pu))
        return xp.maximum(rad, xp.maximum(c0 - c, c - c1))
    return Fn(f, (lo[0], hi[0], lo[1], hi[1], FL_D[1] - hi[2], FL_D[1] - lo[2]))


def window_fov():
    """Exact view window: the points whose direction from the lens front maps onto the sensor (Kalibr equidistant
    f 294 px, 640 x 400, principal point 192 px from the top) plus WIN_MARGIN px, grown by the pupil radius.
    Value ~ angular excess (rad) x distance - pupil (a distance bound good enough for slicing)."""
    fz = np.array([F_ND[0], 0.0, -F_ND[1]])          # (n, t, z up)
    rt = np.array([0.0, -1.0, 0.0])                  # image right (+X camera; the -X one is its mirror)
    up = np.cross(rt, fz)
    q = np.array([LENS_Q[0], 0.0, -LENS_Q[1]])
    umax, vmax, vmin = 320.0 + WIN_MARGIN, 192.0 + WIN_MARGIN, -208.0 - WIN_MARGIN

    def f(X, Y, h, xp):
        vn, vt, vz = X - q[0], Y - q[1], -(FL_D[1] - h) - q[2]
        a = vn * fz[0] + vt * fz[1] + vz * fz[2]
        b = vn * rt[0] + vt * rt[1] + vz * rt[2]
        c = vn * up[0] + vt * up[1] + vz * up[2]
        rho = xp.sqrt(b * b + c * c) + 1e-9
        th = xp.arctan2(rho, a)
        pu, pv = F_PX * th * b / rho, F_PX * th * c / rho
        e = xp.maximum(xp.maximum(xp.abs(pu) - umax, pv - vmax), vmin - pv) / F_PX
        return e * xp.sqrt(rho * rho + a * a) - PUPIL_R
    return Fn(f, (SEAT - CSK - 1, LAND + 30, -60, 60, -10, H_TOP + 30))


# cap: the countersunk frustum, cut to the core's outside (R30 round, the land, the side wall X <= X_OUT, front Yr)
cap = inter(Box(SEAT - CSK, LAND, -FL_T, FL_T, 0.0, H_TOP),
            H(-1, 1, 0, FL_T - SEAT), H(-1, -1, 0, FL_T - SEAT), H(-1, 0, -1, -FL_D[0] - SEAT), H(-1, 0, 1, FL_D[1] - SEAT),
            Cyl('d', (0.0, 0.0), R_FRONT), H(N2[0], T2[0], 0, X_OUT - RC[0]), H(-N2[1], -T2[1], 0, RC[1] - Y_MIN))
# body: the opening prism from the seat back to the board-face plane (+ the countersink's bottom 45° face continued)
_cb = BOARD_B[0] * F_ND[0] + BOARD_B[1] * F_ND[1] + CAM_BOSS_H     # f . x >= this
body = inter(Box(BOARD_B[0] - 20, SEAT, -OPEN_T, OPEN_T, FL_D[1] - OPEN_D[1], FL_D[1] - OPEN_D[0]),
             H(-F_ND[0], 0, -F_ND[1], -_cb), H(-1, 0, 1, FL_D[1] - SEAT))
bosses = union(*[axis_cyl(B3 + t * TC + v * UC, FC, CAM_BOSS_R, 0.0, CAM_BOSS_H + 0.5) for t, v in BOARD_HOLES])
cuts = union(
    axis_cyl(Q3, FC, (CAM_WINDOW + RESIN_HOLE_COMP) / 2, -20.0, 0.01, peak=True),          # lens bore back to the board
    window_fov(),
    *[axis_cyl(B3 + t * TC + v * UC, FC, (SCREW_CLEAR + RESIN_HOLE_COMP) / 2, -1.0, 40.0) for t, v in BOARD_HOLES],
    *[axis_cyl(B3 + t * TC + v * UC, FC, (HEAD_CB + RESIN_HOLE_COMP) / 2, BOARD_CB_C, 40.0, peak=True) for t, v in BOARD_HOLES],
    *[axis_cyl((SEAT, t, d), (1, 0, 0), (SCREW_CLEAR + RESIN_HOLE_COMP) / 2, -10.0, 20.0) for t, d in CART_SCREWS],
    *[axis_cyl((SEAT, t, d), (1, 0, 0), (HEAD_CB + RESIN_HOLE_COMP) / 2, 0.0, 20.0, peak=True) for t, d in CART_SCREWS])
ROOT = sub(union(cap, body, bosses), cuts)

N0, N1 = BOARD_B[0] - 4.0, LAND + 0.5                  # part extent along n (tree X')
T0, T1 = -FL_T - 0.5, FL_T + 0.5
GAP = 6.0


def to_model(x, y, z):
    """Local print (x along n, z along t, y up) -> tree (n, t, h)."""
    return x + N0, z + T0, y


PIECES = [sdf_piece(N1 - N0, H_TOP, T1 - T0, ROOT, to_model, x0=-(N1 - N0) / 2, z0=z0)
          for z0 in (-(T1 - T0) - GAP / 2, GAP / 2)]


def _mc(vol, h, origin):
    from skimage import measure
    v = np.pad(vol, 1).astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(v, 0.5, spacing=(h, h, h))
    return (verts - h + np.asarray(origin))[faces]


def land_to_rig(p, s):
    """(..., 3) land (n, t, d) -> rig (X, Yr, d) for side s."""
    xy = RC + p[..., :1] * N2 + p[..., 1:2] * T2
    return np.stack([s * xy[..., 0], xy[..., 1], p[..., 2]], -1)


def meshes(h=0.3):
    """Cartridge, board and lens triangles in land coords (n, t, d)."""
    ns, ts = np.arange(N0, N1, h), np.arange(T0, T1, h)
    hs = np.arange(0.0, H_TOP + h, h)
    vol = np.stack([np.broadcast_to(solid(ROOT, ns[:, None], ts[None, :], float(z)), (len(ns), len(ts))) for z in hs])
    tri = _mc(vol, h, (0.0, N0, T0))
    cart = np.stack([tri[..., 1], tri[..., 2], FL_D[1] - tri[..., 0]], -1)

    def quad_box(o, a, b, c):                      # parallelepiped from o spanned by a, b, c -> 12 triangles
        o = np.asarray(o, float)
        P = [o + i * a + j * b + k * c for i in (0, 1) for j in (0, 1) for k in (0, 1)]
        F = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1), (2, 3, 7), (2, 7, 6),
             (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
        return np.array([[P[i], P[j], P[k]] for i, j, k in F])
    board = quad_box(B3 - 12.5 * TC - 15.6 * UC - 1.6 * FC, 25 * TC, 23.85 * UC, 1.6 * FC)
    q = np.linspace(0, 2 * np.pi, 33)
    ring = lambda c: [B3 + c * FC + 7 * (np.cos(a) * TC + np.sin(a) * UC) for a in q]
    r0, r1 = ring(0.0), ring(18.0 - LENS_GLASS)
    g0, g1 = ring(18.0 - LENS_GLASS), [B3 + 18.0 * FC + 5.5 * (np.cos(a) * TC + np.sin(a) * UC) for a in q]
    tip = B3 + 18.0 * FC
    lens = np.array([t for i in range(32) for t in ((r0[i], r0[i + 1], r1[i + 1]), (r0[i], r1[i + 1], r1[i]),
                                                   (g0[i], g0[i + 1], g1[i + 1]), (g0[i], g1[i + 1], g1[i]),
                                                   (g1[i], g1[i + 1], tip))])
    return cart, board, lens


LENS_GLASS = 2.0                       # the lens's front taper (render only)


def rig_meshes():
    """[(name, rig-frame tris, colour, label)] for the core's viewer overlays (both sides)."""
    cart, board, lens = meshes()
    out = []
    for name, t, col, lab in (('v4_cartridges', cart, '#3498db', 'camera cartridges (v4 proposal)'),
                              ('v4_cam_boards', board, '#145a32', 'OV9281 boards, pitched 24° (25 × 23.85)'),
                              ('v4_cam_lenses', lens, '#566573', 'lens Ø14 × 18')):
        tri = np.concatenate([land_to_rig(t, s)[:, ::s] for s in (-1, 1)]) * [1, 1, -1]   # z up; mirror flips winding
        out.append((name, tri, col, lab))
    return out
