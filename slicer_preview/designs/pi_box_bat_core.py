"""BATTERY variant of pi_box_core.py (copy): taller, +Y pod with battery stop ribs, laid on
the plate turned 90° (print x = rig Yr) so its 140 mm length fits the long axis.

Lidar rig Pi box, v1 core (one print): flange, well tube, webs, camera walls.

Prints flange-down (the face that mates with the aluminium base plate is on the
build plate) as an open-bottomed ring, so layers are thin walls. Vents through
the flange stop the pockets acting as suction cups. Print it on its own.
Geometry and frames: designs/pi_box_bat_rig.py. Floor: designs/pi_box_bat_lids.py.

Core features (rig depth d = print Y):
  - flange 0..10: 4 stepped bores (Ø8 spacer seat 5.85 deep, Ø10.2 head pocket),
    recesses for the lidar's M3 heads, vents, 1 mm chamfer on the plate face
  - Ø52 well tube; webs from the tube to each camera wall (X axis)
  - ±X camera walls 4 mm with: panel grooves at both ends (open at the bottom),
    a foot round each web with 2 Ø3 dowel holes, floor-screw bosses with glued
    M3 insert holes, the OV9281 window + 4 M2-insert bosses + lens guard ring, exhaust slots
  - Pi 5 standoff posts from the tube (glued M2.5 inserts)
  Horizontal holes are drawn ~0.2 taller (hole_h) for print sag.
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import *      # noqa: F401,F403  (constants + helpers)
from designs import pi_box_bat_rig as R

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

X0, Z0 = Y_MIN, -(GUARD_X + 0.5)          # piece corner (print x = Yr, print z = -X)
SIZE_X, SIZE_Z = Y_MAX - Y_MIN, 2 * (GUARD_X + 0.5)


def cable_clip(X, Y, d, axis, base, along, d0, D, out=None):
    """Push-in C clip round a vertical cable (see CLIPS): two claws on a stem; 45° lead-in below (print)."""
    shape = np.broadcast_shapes(X.shape, Y.shape)
    if d < d0 or d > d0 + CLIP_H:
        return np.zeros(shape, bool)
    u, v = (X, Y) if axis == 'x' else (Y, X)          # u: across the surface (signed), v: along it
    h = -(out if out else -np.sign(base)) * (u - base)  # 0 at the surface, negative where the clip stands
    r_in = (D + RESIN_HOLE_COMP) / 2                   # vertical bore: prints ~0.35 undersize
    r_out = r_in + CLIP_T
    hc = -(r_out + CLIP_STEM)                          # cable centre
    r = np.sqrt((h - hc) ** 2 + (v - along) ** 2)
    ring = (r >= r_in) & (r <= r_out) & ~((h < hc) & (np.abs(v - along) < 0.85 * D / 2))
    stem = (np.abs(v - along) <= 1.5) & (h <= 0.5) & (h >= hc + r_in)
    return (ring | stem) & (h >= -(d - d0) - 1.0)      # grows off the surface at 45°


def cable_notch(X, Y, d):
    """Lidar cable notch up into the +Y rim from its bottom edge (the plate above stays solid)."""
    gw = GROOVE_W + RESIN_HOLE_COMP
    r = (CABLE_NOTCH_W + RESIN_HOLE_COMP) / 2
    return (d >= FLANGE_PLATE) & (np.abs(X) <= r) & (Y >= Y_MAX - PANEL_LIP - gw - 1.6)


def core(x, d, z):
    Y = x + X0                            # (1, W)  rig Yr
    X = -(z + Z0)                         # (H, 1)  rig X
    aX = np.abs(X)
    r2 = X ** 2 + Y ** 2
    inside = (aX <= X_OUT) & (Y >= Y_MIN) & (Y <= Y_MAX)

    if d < FLANGE_PLATE:
        # ── flange plate (4 mm) ──
        inset = max(CHAMFER - d, 0.0)
        s = (aX <= X_OUT - inset) & (Y >= Y_MIN + inset) & (Y <= Y_MAX - inset)
        s &= r2 > WELL_R ** 2
        for sx in (-1, 1):
            for sy in (-1, 1):
                cx, cy = sx * POST, sy * POST
                s &= ~hole(cx, cy, SPACER_D, X, Y)
                if d < M3_HEAD_RECESS:
                    s &= ~hole(sx * LIDAR_M3[0], sy * LIDAR_M3[1], M3_HEAD_D, X, Y)
        for vx, vy in VENTS:
            s &= ~hole(vx, vy, VENT_D, X, Y)
        return s

    # ── below the flange plate ──
    tube = (r2 >= WELL_R ** 2) & (r2 <= (WELL_R + TUBE_T) ** 2)
    if np.any(aX >= X_IN):                             # (row bands: skip the trig where there's no wall)
        walls = (aX >= X_IN) & inside & (wall_solid_zone(Y, d, X) | gyroid(X, Y, d))
    else:
        walls = np.zeros(np.broadcast_shapes(X.shape, Y.shape), bool)
    if d <= DEPTH - WALL_BOTTOM_SOLID:               # internal vertical ribs on the gyroid walls
        for r, top in WALL_RIBS:
            if d >= top:
                rib = (aX >= X_IN - 2.5) & (aX <= X_IN) & (np.abs(Y - r) <= 1.5)
                if d <= CAM_D + 13.5:                  # beside a camera board: keep the bulge 0.4 off its edge
                    rib = rib & (Y <= CAM_YR - 15.6 - 0.4)
                walls = walls | rib
    webs = (aX >= WELL_R + 1) & (aX <= X_IN) & (np.abs(Y) <= WEB_T / 2)
    for x0, x1, d0, d1 in WEB_SLOTS:                 # 45° pointed top: no flat bridge
        roof = d1 + np.minimum(aX - x0, x1 - aX)
        webs &= ~((aX >= x0) & (aX <= x1) & (d >= d0) & (d <= roof))
    s = tube | walls | webs
    if d < RIM_D:                                     # panel-stop rims (RIM_D deep) + bore bosses (FLANGE deep)
        gw = GROOVE_W + RESIN_HOLE_COMP
        s |= inside & ((Y <= Y_MIN + PANEL_LIP + gw + 1.5) | (Y >= Y_MAX - PANEL_LIP - gw - 1.5)) & ~cable_notch(X, Y, d)
        if d < FLANGE:
            for sx in (-1, 1):
                for sy in (-1, 1):
                    cx, cy = sx * POST, sy * POST
                    s |= (X - cx) ** 2 + (Y - cy) ** 2 <= CORNER_BOSS_R ** 2
                    s &= ~hole(cx, cy, SPACER_D if d < SPACER_SEAT else HEAD_D, X, Y)
        s &= inside
    # BATTERY: stop ribs between the tube bays and the bank, middle left open; below the built-in cord
    if d >= BAT_RAIL_D0:                              # stop ribs grow from the wall at 45°
        rib_in = max(BAT_RIB[0], X_IN - (d - BAT_RAIL_D0))
        s |= (aX >= rib_in) & (aX <= BAT_RIB[1]) & (Y >= BAT_RIB[2]) & (Y <= BAT_RIB[3])
    # cradle rails, flange to floor: 45° shoulder over the bank's top corners (narrows going down = no overhang)
    rail_in = min(BAT_RAIL_X, max(BAT_SHOULDER_X, BAT_RAIL_X - (BAT['d0'] + 0.5 - d)))
    for i, ry in enumerate(BAT_RAIL_YS):               # the first one joins the stop rib: one stepped flap
        y0 = BAT_RIB[3] if i == 0 else ry - BAT_RAIL_W / 2
        s |= (aX >= rail_in) & (aX <= X_IN + 0.5) & (Y >= y0) & (Y <= ry + BAT_RAIL_W / 2)
    # UE300 seat: a ledge off the web's Pi face over the adapter's top end (top face = seat; 45° below in print)
    a = ETH_ADAPTER
    seat_d = DEPTH - a['h'] - ETH_SEAT['clear']
    if seat_d - ETH_SEAT['reach'] <= d <= seat_d:
        reach = min(ETH_SEAT['reach'], d - (seat_d - ETH_SEAT['reach']))
        s |= (np.abs(X - a['x']) <= a['wx'] / 2) & (Y <= -WEB_T / 2 + 0.5) & (Y >= -WEB_T / 2 - reach)
    for clip in CLIPS:                                 # push-in cable clips on ribs and webs
        s |= cable_clip(X, Y, d, *clip)
    if d < FLANGE:                                     # re-cut the stepped bores through anything added above
        for sx in (-1, 1):
            for sy in (-1, 1):
                s &= ~hole(sx * POST, sy * POST, SPACER_D if d < SPACER_SEAT else HEAD_D, X, Y)

    rise = d - (DEPTH - FOOT_H)                       # >= 0 once the foot is full size
    foot_in = max(FOOT_X0, X_IN - max(rise + (X_IN - FOOT_X0), 0.0))
    if foot_in < X_IN:                                 # 45° taper from the wall's inner face
        s |= (aX >= foot_in) & (aX <= X_OUT) & (np.abs(Y) <= FOOT_Y) & ~((X > 0) & (Y < -WEB_T / 2))   # UE300 there
    for bx, by in BOSSES:                              # cone from the wall, 45°, full radius at the foot
        attach = abs(abs(bx) - X_IN) + 0.5
        r = min(BOSS_R, attach + max(rise + (BOSS_R - attach), 0.0)) if rise > -(BOSS_R - attach) else 0.0
        if r > 0:
            s |= (X - bx) ** 2 + (Y - by) ** 2 <= r * r
    # Pi standoff posts: horizontal (along Yr) from the tube to the PCB back
    for px, pd in PI_HOLES:
        rr = (X - px) ** 2 + (d - pd) ** 2
        s |= (rr <= PI_POST_R ** 2) & (Y >= PCB_YR) & (Y <= 0) & (r2 >= WELL_R ** 2)
        # gusset: below the post, its far end recedes toward the tube at 45°
        y_tube = -np.sqrt(max(WELL_R ** 2 - px ** 2, 0.0))
        reach = pd - d                                 # how far below the post centre
        if 0 < reach <= PI_POST_R + (y_tube - PCB_YR):
            y_end = PCB_YR + max(reach - PI_POST_R, 0.0)
            s |= (np.abs(X - px) <= PI_POST_R) & (Y >= y_end) & (Y <= y_tube + 1) & (r2 >= WELL_R ** 2)
        y_ins = PCB_YR + PI_INSERT_L                   # glued insert at the PCB end, screw clearance beyond
        s &= ~(hole_h(px, pd, PI_INSERT_D, X, d) & (Y >= PCB_YR) & (Y <= y_ins))
        s &= ~(hole_h(px, pd, PI_SCREW_CLEAR, X, d) & (Y >= PCB_YR) & (Y <= PCB_YR + PI_HOLE_DEPTH))
    # camera blocks: solid from the flange plate down to the board's far edge, board face at CAM_FACE
    for side in (-1, 1):
        sX = side * X
        y0, y1, d0, d1 = cam_board(side)
        if FLANGE_PLATE - 0.01 <= d <= d1:
            blk = (sX >= CAM_FACE) & (sX <= X_IN + 0.5) & (Y >= y0) & (Y <= y1)
            pw, pd = CAM_POCKET                        # connector pocket, open at the board's connector edge
            if CAM_CONN_DIR[side] == '-y':
                pocket = (Y <= y0 + pw) & (d >= d0)
            else:
                pocket = (d >= d1 - pw) & (Y >= y0)
            s |= blk & ~(pocket & (sX <= CAM_FACE + pd))
    # lens guard ring (outside the wall)
    rc = np.sqrt((Y - CAM_YR) ** 2 + (d - CAM_D) ** 2)
    s |= (aX >= X_OUT) & (aX <= GUARD_X) & (rc >= GUARD_R0) & (rc <= GUARD_R1)

    # ── cuts ──
    gw = GROOVE_W + RESIN_HOLE_COMP
    lo0, lo1 = Y_MIN + PANEL_LIP, Y_MIN + PANEL_LIP + gw        # -Y panel slot
    hi0, hi1 = Y_MAX - PANEL_LIP - gw, Y_MAX - PANEL_LIP        # +Y panel slot
    in_slot = ((Y >= lo0) & (Y <= lo1)) | ((Y >= hi0) & (Y <= hi1))
    if d >= RIM_D:                                     # above that the rims are the panels' top stop
        s &= ~((aX >= X_IN) & (aX <= X_IN + GROOVE_D) & in_slot)   # true slots, lip outside
        # the panel itself spans between the walls: keep that band (and the lips' inner side) open
        s &= ~((aX < X_IN) & (in_slot | (Y < lo0) | (Y > hi1)) & ~tube)
    # anti-rattle bumps on each slot's outer face, near the top and bottom of the panel
    for db in (RIM_D + 25.0, DEPTH - 25.0):
        if abs(d - db) <= 1.5:
            band = (aX >= X_IN + 0.5) & (aX <= X_IN + GROOVE_D - 0.5)
            s |= band & (((Y >= lo0) & (Y <= lo0 + 0.3)) | ((Y >= hi1 - 0.3) & (Y <= hi1)))
    sw = POWER_SWITCH                                  # power switch: round hole through its wall (-X, under the Pi)
    tear = hole_h(sw['y'], sw['d'], sw['hole'], Y, d)  # round (+ sag allowance): resin bridges the arch
    sX = sw['side'] * X
    s &= ~((sX >= X_IN - 0.01) & (sX <= X_OUT) & tear)
    win = CAM_WINDOW + RESIN_HOLE_COMP
    dy = np.abs(Y - CAM_YR)
    # round lens window, the same bore as the guard ring so the ring sits on wall all round
    window = dy ** 2 + (d - CAM_D) ** 2 <= (win / 2) ** 2
    for side in (-1, 1):                               # keep solid wall behind every camera boss
        keep = np.zeros_like(window)
        for cy, cd in cam_holes(side):
            keep = keep | ((Y - cy) ** 2 + (d - cd) ** 2 <= (CAM_BOSS_R + 1.0) ** 2)
        sX = side * X
        s &= ~((sX >= CAM_FACE - 0.01) & (sX <= X_OUT) & window & ~keep)   # through the block and wall
    face = CAM_FACE                                    # block face: glued M2 insert, screw clearance beyond
    for side in (-1, 1):
        sX = side * X
        for cy, cd in cam_holes(side):
            s &= ~(hole_h(cy, cd, CAM_INSERT_D, Y, d) & (sX >= face - 0.01) & (sX <= face + CAM_INSERT_L))
            s &= ~(hole_h(cy, cd, CAM_SCREW_CLEAR, Y, d) & (sX >= face - 0.01) & (sX <= face + CAM_HOLE_DEPTH))
    # (the gyroid walls are the vents now; WALL_VENTS / EXHAUST slots are no longer cut)
    if d > DEPTH - PIN_DEPTH:
        for px, py in PINS:
            s &= ~((X - px) ** 2 + (Y - py) ** 2 <= (PIN_SLIP / 2) ** 2)
    if d > DEPTH - INSERT_DEPTH:
        for bx, by in BOSSES:
            s &= ~hole(bx, by, INSERT_D, X, Y)
    return s


PIECES = [implicit_piece(SIZE_X, DEPTH, SIZE_Z, core, x0=X0, y0=0.0, z0=Z0)]
OVERLAYS = R.build_overlays()
