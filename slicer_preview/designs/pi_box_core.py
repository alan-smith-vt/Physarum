"""Lidar rig Pi box, v1 core (one print): flange, well tube, webs, camera walls.

Prints flange-down (the face that mates with the aluminium base plate is on the
build plate) as an open-bottomed ring, so layers are thin walls. Vents through
the flange stop the pockets acting as suction cups. Print it on its own.
Geometry and frames: designs/pi_box_rig.py. Floor + panels: designs/pi_box_lids.py.

Core features (rig depth d = print Y):
  - flange 0..10: 4 stepped bores (Ø8 spacer seat 5.85 deep, Ø10.2 head pocket),
    recesses for the lidar's M3 heads, vents, 1 mm chamfer on the plate face
  - Ø52 well tube; webs from the tube to each camera wall (X axis)
  - ±X camera walls 4 mm with: panel grooves at both ends (open at the bottom),
    a foot round each web with 2 Ø3 dowel holes, floor-screw bosses with glued
    M2.5 insert holes, the OV9281 window + 4 M2 bosses + lens guard ring, exhaust slots
  - Pi 5 standoff posts from the tube (M2.5 self-tap)
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_rig import *          # noqa: F401,F403  (constants + helpers)
from designs import pi_box_rig as R

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

X0, Z0 = -(GUARD_X + 0.5), Y_MIN          # piece corner (print x, z)
SIZE_X, SIZE_Z = 2 * (GUARD_X + 0.5), Y_MAX - Y_MIN


def core(x, d, z):
    X = x + X0                            # (1, W)
    Y = z + Z0                            # (H, 1)  rig Yr
    aX = np.abs(X)
    r2 = X ** 2 + Y ** 2
    inside = (aX <= X_OUT) & (Y >= Y_MIN) & (Y <= Y_MAX)

    if d < FLANGE:
        # ── flange ──
        inset = max(CHAMFER - d, 0.0)
        s = (aX <= X_OUT - inset) & (Y >= Y_MIN + inset) & (Y <= Y_MAX - inset)
        s &= r2 > WELL_R ** 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = sx * POST, sy * POST
                s &= ~hole(cx, cy, SPACER_D if d < SPACER_SEAT else HEAD_D, X, Y)
                if d < M3_HEAD_RECESS:
                    s &= ~hole(sx * LIDAR_M3[0], sy * LIDAR_M3[1], M3_HEAD_D, X, Y)
        for vx, vy in VENTS:
            s &= ~hole(vx, vy, VENT_D, X, Y)
        return s

    # ── below the flange ──
    tube = (r2 >= WELL_R ** 2) & (r2 <= (WELL_R + TUBE_T) ** 2)
    walls = (aX >= X_IN) & inside
    webs = (aX >= WELL_R + 1) & (aX <= X_IN) & (np.abs(Y) <= WEB_T / 2)
    for x0, x1, d0, d1 in WEB_SLOTS:                 # 45° pointed top: no flat bridge
        roof = d1 + np.minimum(aX - x0, x1 - aX)
        webs &= ~((aX >= x0) & (aX <= x1) & (d >= d0) & (d <= roof))
    s = tube | walls | webs

    rise = d - (DEPTH - FOOT_H)                       # >= 0 once the foot is full size
    foot_in = max(FOOT_X0, X_IN - max(rise + (X_IN - FOOT_X0), 0.0))
    if foot_in < X_IN:                                 # 45° taper from the wall's inner face
        s |= (aX >= foot_in) & (aX <= X_OUT) & (np.abs(Y) <= FOOT_Y)
    for bx, by in BOSSES:                              # cone from the wall, 45°, full radius at the foot
        attach = abs(abs(bx) - X_IN) + 0.5
        r = min(BOSS_R, attach + max(rise + (BOSS_R - attach), 0.0)) if rise > -(BOSS_R - attach) else 0.0
        if r > 0:
            s |= (X - bx) ** 2 + (Y - by) ** 2 <= r * r
    # Pi standoff posts: horizontal (along Yr) from the tube to the PCB back
    rp = (PI_PILOT + RESIN_HOLE_COMP) / 2
    for px, pd in PI_HOLES:
        rr = (X - px) ** 2 + (d - pd) ** 2
        s |= (rr <= PI_POST_R ** 2) & (Y >= PCB_YR) & (Y <= 0) & (r2 >= WELL_R ** 2)
        # gusset: below the post, its far end recedes toward the tube at 45°
        y_tube = -np.sqrt(max(WELL_R ** 2 - px ** 2, 0.0))
        reach = pd - d                                 # how far below the post centre
        if 0 < reach <= PI_POST_R + (y_tube - PCB_YR):
            y_end = PCB_YR + max(reach - PI_POST_R, 0.0)
            s |= (np.abs(X - px) <= PI_POST_R) & (Y >= y_end) & (Y <= y_tube + 1) & (r2 >= WELL_R ** 2)
        s &= ~((rr <= rp ** 2) & (Y >= PCB_YR) & (Y <= PCB_YR + 8))
    # camera bosses (horizontal, along X) on the wall's inner face
    for side in (-1, 1):
        sX = side * X
        for cy, cd in cam_holes(side):
            if abs(d - cd) <= CAM_BOSS_R:
                w = np.sqrt(CAM_BOSS_R ** 2 - (d - cd) ** 2)
                s |= (np.abs(Y - cy) <= w) & (sX >= X_IN - CAM_BOSS_H) & (sX <= X_IN)
    # lens guard ring (outside the wall)
    rc = np.sqrt((Y - CAM_YR) ** 2 + (d - CAM_D) ** 2)
    s |= (aX >= X_OUT) & (aX <= GUARD_X) & (rc >= GUARD_R0) & (rc <= GUARD_R1)

    # ── cuts ──
    gw = GROOVE_W + RESIN_HOLE_COMP
    groove = (aX >= X_IN) & (aX <= X_IN + GROOVE_D) & ((Y <= Y_MIN + gw) | (Y >= Y_MAX - gw))
    s &= ~groove
    # the panel itself passes inside the wall ends: keep the region between the walls open
    s &= ~((aX < X_IN) & ((Y <= Y_MIN + gw) | (Y >= Y_MAX - gw)) & ~tube)
    sw = POWER_SWITCH                                  # power switch: teardrop hole through the +X wall
    rs = (sw['hole'] + RESIN_HOLE_COMP) / 2
    tear = ((Y - sw['y']) ** 2 + (d - sw['d']) ** 2 <= rs * rs) | (
        (d >= sw['d']) & (np.abs(Y - sw['y']) + (d - sw['d']) <= rs * 1.4142))
    s &= ~((X >= X_IN - 0.01) & (X <= X_OUT) & tear)
    win = CAM_WINDOW + RESIN_HOLE_COMP
    dy = np.abs(Y - CAM_YR)
    s &= ~((aX >= X_IN - 0.01) & (aX <= X_OUT) & (dy <= win / 2) & (d >= CAM_D - win / 2)
           & (d <= CAM_D + win / 2 + (win / 2 - dy)))    # pointed top: no 16 mm flat bridge
    rp = (CAM_PILOT + RESIN_HOLE_COMP) / 2
    for side in (-1, 1):
        sX = side * X
        for cy, cd in cam_holes(side):
            if abs(d - cd) < rp:
                w = np.sqrt(rp ** 2 - (d - cd) ** 2)
                s &= ~((np.abs(Y - cy) <= w) & (sX >= X_IN - CAM_BOSS_H) & (sX <= X_IN + 3))
    if EXHAUST_D[0] <= d <= EXHAUST_D[1]:
        for y0, y1 in EXHAUST:
            s &= ~((aX >= X_IN) & (Y >= y0) & (Y <= y1))
    if d > DEPTH - PIN_DEPTH:
        for px, py in PINS:
            s &= ~((X - px) ** 2 + (Y - py) ** 2 <= (PIN_SLIP / 2) ** 2)
    if d > DEPTH - INSERT_DEPTH:
        for bx, by in BOSSES:
            s &= ~hole(bx, by, INSERT_D, X, Y)
    return s


PIECES = [implicit_piece(SIZE_X, DEPTH, SIZE_Z, core, x0=X0, y0=0.0, z0=Z0)]
OVERLAYS = R.build_overlays()
