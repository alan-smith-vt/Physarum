"""v4 core PROPOSAL (2026-10-29, not approved, don't print): the v3 core (pi_box_bat_core_v3.py) with
  - back corners square; FRONT vertical edges rounded R_FRONT (R30, centres (±20, -32)), 4 mm shell;
  - a vertical flat LAND per camera on the round, facing yaw 30° forward; lens centred on it (±45.1, -46.5, d 35);
  - cartridge opening + 45° countersunk SEAT (rectangular frustum) in an inner pad frame: the cartridge's cap is
    flush with the round, slides in along the land normal, 3 × M2 from outside into inserts set from the pad's
    inner face (nuts). The board's 24° pitch lives in the cartridge (designs/pi_box_v4_cam_cartridge.py);
  - front M4 stubs at (±40, -40) cut to r 7.1, d 0..12, and a clear Ø HEAD_D driver path below each to the open
    bottom (it runs through the cartridge opening: clear with the cartridge out);
  - front panel L-slot channels where the rounds start (|X| = 20); back panel channels as v3;
  - interior: Pi lowered to PI_TOP_D 55 (v3: 14), PI_SHIFT +4 (v3: +6); power board top d 95 (v3: 80);
  - floor interface with home/v4-floor (battery door slides out +Y): rail rebates in both wall inner faces
    (closed grooves Yr 30.5..62, d 175.6..178.4, 1.6 deep) and the plunger boss in the -X back corner (Ø5.4 × 7 pin hole);
    the v3 door latch / hinge holes are dropped; the fixed floor's front screw bosses move inside the rounds.
Sketch: designs/sketches/pi_box_v4_core_sketch.png. Viewer: python generate.py pi_box_v4_core.

How: the v3 source is executed with a few constants patched and three hooks spliced in (see _V3_PATCHES), so the
v3 file stays untouched and v3 fixes flow through. Every patch must match exactly once (asserted).
"""
import os
import sys
from pathlib import Path
import numpy as np
from sdf import Box, Cyl, Half, Fn, union, inter, sub, sdf_piece, INF
from geometry import implicit_piece
from designs import pi_box_bat_rig as R

R_FRONT, LAND_OFF = 30.0, 0.0                    # no flat land: the cartridge is flush with the R30 curve
YAW, PITCH = np.radians(30.0), np.radians(24.0)
CAM_LENS_D = 35.0                                  # lens axis depth at the land
LENS_RECESS = 0.4 / np.cos(np.radians(24.0))   # lens front centre 0.4 behind the curve; its lowest barrel edge
#   (pitched 24°) still stands proud, reaching X ~49: set down on a side, the box rests on the flat (X 50), not the lens.                                  # lens front centre behind the land, along the axis: its top rim
                                                   # (pitched 24°) stays ~0.2 behind the land, the cap is its shroud
PI_TOP_D_V4, PI_SHIFT_V4 = 55.0, 2.0              # shift 2: the +X lower post clears the UE300
# ALTERNATIVE (user, 2026-10-29): PI_BOX_V4_PI=flip turns the Pi end for end (R.PI_FLIP): SD end at the floor
# (d 174), ports facing the flange (port end d 89), plugs pointing up beside the cartridges; switch to Yr -42.
FLIP = os.environ.get('PI_BOX_V4_PI') == 'flip'
if FLIP:
    PI_TOP_D_V4 = 89.0                             # the port end (R.PI_FLIP_TOP); SD end at 89 + 85 = 174 (user: back up; the switch moves)
BRD_Y0_V4, BRD_TOP_V4 = -43.0, 95.0                # power board Yr start (v3 -55.5): the -X round pulls the wall in
                                                   # forward of Yr -32; its +Y edge passes the web (notched). Top d 95.
# (a power board <-> UE300 side swap was modelled and reverted, user 2026-10-29 home:2102)

X_OUT, X_IN, Y_MIN, Y_MAX, DEPTH = R.X_OUT, R.X_IN, R.Y_MIN, R.Y_MAX, R.DEPTH
RC = np.array([X_OUT - R_FRONT, Y_MIN + R_FRONT])  # +X front round centre (X, Yr); -X mirrored
N2 = np.array([np.cos(YAW), -np.sin(YAW)])         # land normal (plan, +X side)
T2 = np.array([np.sin(YAW), np.cos(YAW)])          # land tangent (toward +Yr)
LAND = R_FRONT - LAND_OFF                          # land plane, n from RC
SEAT = 22.9                                        # cap seat plane (the countersink's outer edge)
CSK = 4.0                                          # 45° countersink depth: ±FL_T at SEAT .. ±OPEN_T at SEAT - CSK
PAD = SEAT - CSK - 1.0                             # pad frame inner face
OPEN_T, OPEN_D = 13.5, (25.0, 59.5)                # cartridge pod's shadow along n: half-width along the land, d range
FL_T, FL_D = OPEN_T + CSK, (OPEN_D[0] - CSK, OPEN_D[1] + CSK)
CART_CL = 0.2                                      # per side, + RESIN_HOLE_COMP / 2 in the core
CART_SCREWS = [(-(OPEN_T + 2.0), 42.6), (OPEN_T + 2.0, 42.6), (0.0, OPEN_D[1] + 2.0)]   # (t, d) on the countersink
STUB_R, STUB_D = 7.1, 12.0
CART_NUT_L = 2.5                                   # M2 insert, set from the pad's inner face, bottomed (a nut)

# camera frame in land coords (n, d): optical axis f (out and up), board up u
F_ND = np.array([np.cos(PITCH), -np.sin(PITCH)])
U_ND = np.array([-np.sin(PITCH), -np.cos(PITCH)])
LENS_P = np.array([LAND, CAM_LENS_D])              # axis meets the land here
LENS_Q = LENS_P - LENS_RECESS * F_ND               # lens front centre
BOARD_B = LENS_Q - 18.0 * F_ND                     # board front face centre (lens 18 from the board)


def lens_rig(s):
    """Lens centre on the land, rig (X, Yr, d)."""
    p = RC + LAND * N2
    return np.array([s * p[0], p[1], CAM_LENS_D])


def axis_rig(s):
    a = np.cos(PITCH) * N2
    return np.array([s * a[0], a[1], -np.sin(PITCH)])     # d down: up is -d


# ── land-frame helpers ──
def _nt(s, X, Y):
    xs = s * X - RC[0]
    yy = Y - RC[1]
    return xs * N2[0] + yy * N2[1], xs * T2[0] + yy * T2[1]


def _aabb(s, pts):
    """pts (n, t, d) -> rig box bounds."""
    P = np.array([[*(RC + n * N2 + t * T2), d] for n, t, d in pts])
    P[:, 0] *= s
    lo, hi = P.min(0) - 0.5, P.max(0) + 0.5
    return lo[0], hi[0], lo[1], hi[1], lo[2], hi[2]


def lpoly(s, planes, box):
    """Convex region in land coords: every plane (an, at, ad, c) means an*n + at*t + ad*d <= c.
    box = (n0, n1, t0, t1, d0, d1) bounds the region (for culling)."""
    hs = []
    for an, at, ad, c in planes:
        hs.append(Half(s * (an * N2[0] + at * T2[0]), an * N2[1] + at * T2[1], ad,
                       c + an * (RC @ N2) + at * (RC @ T2)))
    n0, n1, t0, t1, d0, d1 = box
    corners = [(n, t, d) for n in (n0, n1) for t in (t0, t1) for d in (d0, d1)]
    return inter(Box(*_aabb(s, corners)), *hs)


def lbox(s, n0, n1, t0, t1, d0, d1):
    return lpoly(s, [(-1, 0, 0, -n0), (1, 0, 0, n1), (0, -1, 0, -t0), (0, 1, 0, t1), (0, 0, -1, -d0), (0, 0, 1, d1)],
                 (n0, n1, t0, t1, d0, d1))


def lcyl(s, o, a, r, c0, c1):
    """Cylinder in land coords: origin o (n, t, d), unit axis a (n, t, d), radius r, along the axis c0..c1."""
    o, a = np.asarray(o, float), np.asarray(a, float) / np.linalg.norm(a)
    ends = [o + a * c0, o + a * c1]
    pts = [e + np.array(k) * r for e in ends for k in ((1, 1, 1), (-1, -1, -1), (1, -1, 1), (-1, 1, -1))]

    def f(X, Y, d, xp):
        n, t = _nt(s, X, Y)
        vn, vt, vd = n - o[0], t - o[1], d - o[2]
        c = vn * a[0] + vt * a[1] + vd * a[2]
        rad = xp.sqrt(xp.maximum(vn * vn + vt * vt + vd * vd - c * c, 0.0)) - r
        return xp.maximum(rad, xp.maximum(c0 - c, c - c1))
    return Fn(f, _aabb(s, [tuple(p) for p in pts]))


def frustum(s, grow, n_out):
    """The cartridge seat: rectangle ±(FL_T + grow) × FL_D (± grow) at SEAT, narrowing at 45° inward (-n);
    extends outward to n_out."""
    return lpoly(s, [(-1, 1, 0, FL_T + grow - SEAT), (-1, -1, 0, FL_T + grow - SEAT),
                     (-1, 0, -1, -(FL_D[0] - grow) - SEAT), (-1, 0, 1, FL_D[1] + grow - SEAT),
                     (1, 0, 0, n_out), (-1, 0, 0, -(SEAT - CSK - 0.01))],
                 (SEAT - CSK - 0.5, n_out, -FL_T - grow, FL_T + grow, FL_D[0] - grow, FL_D[1] + grow))


# ── outer envelope (square back, rounded front) and the round shells ──
def _envelope(r_in=0.0):
    r = R_FRONT - r_in
    return inter(Box(-X_OUT + r_in, X_OUT - r_in, Y_MIN + r_in, Y_MAX - r_in),
                 union(Box(-INF, INF, RC[1], INF), Box(-RC[0], RC[0], -INF, INF),
                       *[Cyl('d', (s * RC[0], RC[1]), r) for s in (-1, 1)]))


ENVELOPE = _envelope()
TOP, BOT = R.FLANGE_PLATE, DEPTH
GW = R.GROOVE_W + R.RESIN_HOLE_COMP
_lo0 = Y_MIN + R.PANEL_LIP
_lo1 = _lo0 + GW
XE = RC[0]                                         # front panel edge |X| (where the rounds start)

# imported lazily below from the v3 panel module (NECK, HOOK_T, HOOK, CL)
from designs import pi_box_bat_panel_v3 as PV      # noqa: E402

PILLAR_X1 = XE + PV.NECK + PV.HOOK_T + PV.CL + 1.5


def _shells():
    out = []
    for s in (-1, 1):
        quad = Box(*sorted((s * RC[0], s * INF)), -INF, RC[1], TOP, BOT)
        out.append(inter(quad, sub(Cyl('d', (s * RC[0], RC[1]), R_FRONT), Cyl('d', (s * RC[0], RC[1]), R_FRONT - 4.0))))
        # front pillar: solid round the panel channel
        out.append(Box(*sorted((s * (XE - 1.0), s * PILLAR_X1)), Y_MIN, _lo1 + PV.HOOK + 1.5, TOP, BOT))
    return union(*out)


def _front_channels():
    out = []
    for s in (-1, 1):
        x0, x1 = XE + PV.NECK - PV.CL, XE + PV.NECK + PV.HOOK_T + PV.CL
        out += [Box(*sorted((s * (XE - 1.01), s * (x0 + 0.01))), _lo0, _lo1, R.RIM_D, BOT + 1),
                Box(*sorted((s * x0, s * x1)), _lo0, _lo1 + PV.HOOK, R.RIM_D, BOT + 1)]
    out.append(Box(-XE, XE, Y_MIN - 1, _lo1, R.RIM_D, BOT + 1))     # the panel's own slot across the front
    return union(*out)


def _cam_core():
    """Pad frame (solid) and cuts (opening, countersunk seat, screw holes + insert pockets, outside of the land)."""
    pads, cuts = [], []
    g = CART_CL + R.RESIN_HOLE_COMP / 2
    for s in (-1, 1):
        pads.append(inter(lbox(s, PAD, SEAT + 0.5, -FL_T - 2.5, FL_T + 2.5, FL_D[0] - 2.5, FL_D[1] + 2.5),
                          _envelope()))
        # 45° underside on the pad (print: d is up; the pad hangs off the shell) -> grow it back to the wall
        cuts.append(frustum(s, g, LAND + 10.0))
        cuts.append(lbox(s, PAD - 1.0, SEAT, -OPEN_T - g, OPEN_T + g, OPEN_D[0] - g, OPEN_D[1] + g))
        for t, d in CART_SCREWS:
            cuts.append(lcyl(s, (SEAT, t, d), (1, 0, 0), (R.CAM_SCREW_CLEAR + R.RESIN_HOLE_COMP) / 2, -8.0, 8.0))
            cuts.append(lcyl(s, (SEAT, t, d), (1, 0, 0), (R.CAM_INSERT_D + R.RESIN_HOLE_COMP) / 2,
                             PAD - SEAT - 1.0, PAD - SEAT + CART_NUT_L))      # insert (nut) from the pad's inner face
    return union(*pads), union(*cuts)


def _pad_support(s):
    """Fill under each pad so it grows off the round wall at 45° (no flat overhang, print up = +d)... the pad's
    underside faces +d (down in the rig = UP in print: the core prints flange-down), so only its top face (-d) is
    an overhang: give it a 45° roof back to the wall."""
    return lpoly(s, [(-1, 0, 0, -PAD), (1, 0, 0, R_FRONT - 2.0), (0, -1, 0, FL_T + 2.5), (0, 1, 0, FL_T + 2.5),
                     (-1, 0, -1, -(FL_D[0] - 2.5) - PAD), (0, 0, 1, FL_D[0])],
                 (PAD, R_FRONT - 2.0, -FL_T - 2.5, FL_T + 2.5, FL_D[0] - 2.5 - (R_FRONT - PAD), FL_D[0]))


# floor interface (home/v4-floor, 2026-10-29 decision home:1997/1998)
DOOR_RAIL = dict(y0=30.5, d0=175.6, d1=178.4, depth=1.6)   # CLOSED groove: the 1.6 ledge below carries the door
DOOR_WEB_CLEAR = 1.8
PLUNGER = dict(x0=-46.0, x1=-38.0, y0=38.0, y1=50.0, d0=168.0, d1=180.0, hole=(-41.1, 44.0), dia=4.0, L=6.0)   # M3 insert (door thumb screw, user 2026-10-29)
FLOOR_BUTTON = (-34.0, 13.0)                       # floor power button (home:2092)
FLOOR_BOSSES_V4 = [(-42.0, -18.0), (45.0, 6.5)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]   # front pair moved off the
# rounds (wide front panel, user approved home:2136; floor home/v4-floor); the +X one stays X >= 41.5 (home:2135)


def _floor_iface():
    g = R.RESIN_HOLE_COMP / 2
    r = DOOR_RAIL
    cuts = [Box(*sorted((s * (X_IN - 0.01), s * (X_IN + r['depth'] + g))), r['y0'], Y_MAX + 1, r['d0'] - g, r['d1'] + g)
            for s in (-1, 1)]
    cuts += [inter(Box(*sorted((s * (X_IN - 0.01), s * (X_IN + r['depth'] + g + 1.0))), Y_MAX - 1.0, Y_MAX + 1,
                       r['d0'] - g - 1.0, r['d1'] + g + 1.0),
                   Half(s, -1, 0, X_IN + r['depth'] + g - (Y_MAX - 1.0)))  # 45° flare at the +Y mouth
             for s in (-1, 1)]
    p = PLUNGER
    boss = inter(Box(p['x0'], p['x1'], p['y0'], p['y1'], p['d0'] - (p['x1'] - p['x0']), p['d1']),
                 Half(1, 0, -1, -(X_IN + p['d0'] - (p['x1'] - p['x0']))))   # |X| + d >= 206: 45° top (print: underside)
    cuts += [Box(*sorted((s * (X_IN - DOOR_WEB_CLEAR), s * (X_IN + 0.01))), r['y0'], Y_MAX + 1, 175.5, BOT + 1)
             for s in (-1, 1)]                     # the door's webs ride up inside each wall face (home:2006)
    # ...which leaves the cradle-rail bottoms (and the bits by the back panel channel) hanging as thin slabs off
    # the wall (user, 2026-10-29: remove them). Everything inboard of the web slot below d 175.5 goes, except the
    # lock boss (+-X), which keeps its insert depth.
    for s in (-1, 1):
        cuts += [Box(*sorted((s * 30.0, s * (X_IN - DOOR_WEB_CLEAR + 0.01))), y0, y1, 175.5, BOT + 1)
                 for y0, y1 in ((r['y0'], Y_MAX + 1),) if s > 0]
        cuts += [Box(*sorted((s * 30.0, s * (X_IN - DOOR_WEB_CLEAR + 0.01))), y0, y1, 175.5, BOT + 1)
                 for y0, y1 in ((r['y0'], PLUNGER['y0']), (PLUNGER['y1'], Y_MAX + 1)) if s < 0]
        # the rail ends rise at 45° inboard from the web slot (cut d - |X| >= 131.3): brackets off the wall, not slabs
        x_s = X_IN - DOOR_WEB_CLEAR
        for y0, y1 in (((r['y0'], Y_MAX + 1),) if s > 0 else ((r['y0'], PLUNGER['y0']), (PLUNGER['y1'], Y_MAX + 1))):
            cuts.append(inter(Box(*sorted((s * 30.0, s * (x_s + 0.01))), y0, y1, 175.5 - (x_s - 30.0), 175.6),
                              Half(s, 0, -1, -(175.5 - x_s))))
        # where the groove/web cut passes the back-panel L-channel: a ~0.25 fin and a hook-rib strip (home:2051);
        # the groove ledge only needs Yr 30.5..54.5, so clear the channel corner inboard of its outer skin
        cuts.append(Box(*sorted((s * 44.0, s * 48.9)), 54.0, Y_MAX + 1, 175.5, BOT + 1))
    cuts.append(Cyl('d', p['hole'], (p['dia'] + R.RESIN_HOLE_COMP) / 2, p['d1'] - p['L'], p['d1'] + 1))
    return boss, union(*cuts)


def _front_bosses():
    """The fixed floor's front screw bosses (v3's at (±44, -51.5) fall outside the rounds): columns touching the
    round's inner face, grown off it at 45° (print up = +d)."""
    out = []
    for bx, by in FLOOR_BOSSES_V4[:2]:
        cx = np.sign(bx) * RC[0]
        r_wall = R_FRONT - 4.0
        d_on = DEPTH - 14.0

        def f(X, Y, d, xp, bx=bx, by=by, cx=cx):
            rho = xp.sqrt((X - cx) ** 2 + (Y - RC[1]) ** 2)
            cyl = xp.sqrt((X - bx) ** 2 + (Y - by) ** 2) - R.BOSS_R
            return xp.maximum(cyl, (r_wall - rho) - (d - d_on))
        out.append(Fn(f, (bx - R.BOSS_R, bx + R.BOSS_R, by - R.BOSS_R, by + R.BOSS_R, d_on - 10, BOT)))
    return union(*out)


def _web_notch():
    """Notch in the -X web for the power board's +Y edge and its parts (board X -43.5..-41.9, parts to ~-36)."""
    w = (-R.WEB_T / 2 - 0.01, R.WEB_T / 2 + 0.01)
    return union(Box(-X_IN - 0.01, -35.5, *w, BRD_TOP_V4 - 25.0, BRD_TOP_V4 + 62.0),   # J1/J2 plugs leave upward
                 Box(-X_IN - 0.01, -19.0, *w, BRD_TOP_V4 + 48.0, BRD_TOP_V4 + 64.0))   # J3 plug + wires


# Panel top-edge tongue groove under the back panel-stop rim (user approved home:2130; sketch
# designs/sketches/pi_box_v4_panel_tongue_sketch.py): 2.35 wide (tongue 1.6 + CL per side), 3.4 deep, 45° roof.
# Front: the centre strip's top edge (|X| <= FRONT_TONGUE_X) the same way.
TONGUE_T, TONGUE_H, TONGUE_GAP = 1.6, 3.0, 0.4
BACK_TONGUE_X = (6.2, X_IN - 2.5)                  # |X| range: gap at the lidar-cable notch
BACK_PANEL_YC = Y_MAX - R.PANEL_LIP - PV.CL - R.PANEL_T / 2
FRONT_TONGUE_X = XE - 2.5
FRONT_PANEL_YC = Y_MIN + R.PANEL_LIP + PV.CL + R.PANEL_T / 2


def _rim_grooves():
    g = TONGUE_T / 2 + PV.CL
    yc, top = BACK_PANEL_YC, R.RIM_D - TONGUE_H - TONGUE_GAP
    roof = [Half(0, 1, -1, yc - (top - g)), Half(0, -1, -1, -yc - (top - g))]
    x0, x1 = BACK_TONGUE_X[0] - PV.CL, BACK_TONGUE_X[1] + PV.CL
    back = [inter(Box(*sorted((s * x0, s * x1)), yc - g, yc + g, top - g, R.RIM_D + 0.01), *roof) for s in (-1, 1)]
    yc = FRONT_PANEL_YC
    roof = [Half(0, 1, -1, yc - (top - g)), Half(0, -1, -1, -yc - (top - g))]
    xf = FRONT_TONGUE_X + PV.CL
    return union(*back, inter(Box(-xf, xf, yc - g, yc + g, top - g, R.RIM_D + 0.01), *roof))


# ── wide front panel (user approved home:2136; sketch designs/sketches/pi_box_v4_front_panel_wide_sketch.py) ──
# Below STEP everything in front of the round/side-wall tangent (Yr < RC[1]) is the removable panel: front flat + both
# R30 rounds, wall PT, outer face on the envelope; L-hook edges (neck along each side wall's inner face, hook turning
# out into a channel in the wall); its top edge tongues into a groove in the fixed round shell at the step. Above
# STEP the centre strip (|X| <= XE, as before) runs up to the rim. It slides in from the floor end.
STEP = 72.0                                        # 6 below the cartridge pad frame (FL_D[1] + 2.5 = 66)
PT = 3.2
NECK_T, NECK_L, HOOK_W, HOOK_T = 1.6, 5.0, 2.0, 1.6
STEP_TONGUE_X0 = 26.0                              # step tongue on the rounds' |X| >= this (clear of the XE pillars)
R_MID = R_FRONT - PT + TONGUE_T / 2                # step tongue on the panel's inner half (R 26.8..28.4)
STEP_RIB = (24.0, 8.0)                             # fixed shell thickened inward above the step: to R 24, 8 deep


def _front_rad(X, Y, xp):
    """Distance from the front stadium's centre line (|X| <= XE at Yr RC[1]), for Yr <= RC[1]."""
    return xp.sqrt(xp.maximum(xp.abs(X) - XE, 0.0) ** 2 + xp.maximum(RC[1] - Y, 0.0) ** 2)


def _front_cuts():
    cl = PV.CL
    y_hook = RC[1] + NECK_L
    out = [Box(-X_OUT - 1, X_OUT + 1, Y_MIN - 1, RC[1], STEP, BOT + 1)]      # the panel's zone
    for s in (-1, 1):
        out.append(Box(*sorted((s * (X_IN - NECK_T - 2 * cl - 0.5), s * (X_IN + 0.01))), RC[1] - 1, y_hook + HOOK_W + cl,
                       STEP, BOT + 1))                                       # neck + hook clearance inside the wall
        out.append(Box(*sorted((s * (X_IN - 0.01), s * (X_IN + HOOK_T + cl))), y_hook - cl, y_hook + HOOK_W + cl,
                       STEP, BOT + 1))                                       # hook channel in the wall
    g = TONGUE_T / 2 + cl
    top = STEP - TONGUE_H - TONGUE_GAP

    def groove(X, Y, d, xp):                                                 # 45° roof (print +d up)
        e = xp.abs(_front_rad(X, Y, xp) - R_MID)
        f = xp.maximum(e - g, (top - g + e) - d)
        f = xp.maximum(f, d - (STEP + 0.01))
        f = xp.maximum(f, (STEP_TONGUE_X0 - cl) - xp.abs(X))
        return xp.maximum(f, Y - (RC[1] - 1.0))
    out.append(Fn(groove, (-X_OUT, X_OUT, Y_MIN, RC[1], top - g - 1, STEP + 1)))
    # floor switch nut (18.8 AC, r 9.4 + 0.35) clears the -X web foot corner (home:2144)
    out.append(Cyl('d', FLOOR_BUTTON, 9.75, 175.0, BOT + 0.01))
    # +X front floor boss (45, 6.5): kept at X >= 41.5 (the floor's +X rail end, home:2135)
    bx, by = FLOOR_BOSSES_V4[1]
    out.append(inter(Cyl('d', (bx, by), R.BOSS_R + 0.5, STEP, BOT + 1), Box(0.0, 41.5, by - 6, by + 6, STEP, BOT + 1)))
    return union(*out)


def _step_rib():
    """Inner rib on the fixed rounds just above the step (the groove leaves >= 1.2 outer skin); 45° top (print +d up)."""
    r0, h = STEP_RIB

    def f(X, Y, d, xp):
        rad = _front_rad(X, Y, xp)
        g = xp.maximum(r0 - rad, rad - (R_FRONT - 3.0))
        g = xp.maximum(g, (STEP - h + (R_FRONT - 4.0 - rad)) - d)   # 45° from the wall face
        g = xp.maximum(g, d - STEP)
        g = xp.maximum(g, (STEP_TONGUE_X0 - 2.0) - xp.abs(X))
        return xp.maximum(g, Y - (RC[1] - 1.0))
    return Fn(f, (-X_OUT, X_OUT, Y_MIN, RC[1], STEP - h - 3, STEP))


def front_panel_rig(X, Y, d):
    """The removable front panel, rig frame: X, Y arrays (broadcast), d scalar -> bool. Nominal size minus
    PRINT_GROW on its faces; the clearance is in the core's channels and grooves (CL)."""
    xp, G, cl = np, PV.PRINT_GROW, PV.CL
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    ax = np.abs(X)
    rad = _front_rad(X, Y, xp)
    front = Y <= RC[1] - 0.2                                                # 0.2 gap at the wall ends
    s = np.zeros(np.broadcast_shapes(X.shape, Y.shape), bool)
    if STEP + 0.1 <= d <= DEPTH - 0.1:
        x0, x1 = X_IN - cl - NECK_T + G, X_IN - cl - G                     # neck
        y_hook = RC[1] + NECK_L
        shell = front & (rad >= R_FRONT - PT + G) & (rad <= R_FRONT - G)
        lip = (ax >= x0) & (ax <= X_IN + 1.0) & (Y >= RC[1] - 4.0) & front & (rad <= R_FRONT - G)
        neck = (ax >= x0) & (ax <= x1) & (Y >= RC[1] - 0.01) & (Y <= y_hook + HOOK_W - G)
        hook = (ax >= x0) & (ax <= X_IN + HOOK_T - G) & (Y >= y_hook + G) & (Y <= y_hook + HOOK_W - G)
        s |= shell | lip | neck | hook
    e = np.abs(rad - R_MID)                                                  # step tongue, 45° tip
    tt = STEP + 0.1 - TONGUE_H
    if tt <= d < STEP + 0.2:
        s |= front & (Y <= RC[1] - 2.0) & (ax >= STEP_TONGUE_X0) & (e <= TONGUE_T / 2 - G) & (d >= tt + e)
    face = Y_MIN + R.PANEL_LIP + cl                                          # centre strip (+ 10 into the wide part)
    if d <= STEP + 10.0:
        y = Y - face
        s |= front_panel_mask(np.broadcast_to(X, s.shape), y, d)
        ty = np.abs(y - R.PANEL_T / 2)
        dt = PV.D0 - TONGUE_H                                                # its top tongue into the rim
        if dt <= d < PV.D0 + 1.0:
            s |= (ty <= TONGUE_T / 2 - G) & (d >= dt + ty) & (ax <= FRONT_TONGUE_X)
    return s


FRONT_PANEL_D0 = PV.D0 - TONGUE_H


def _driver_paths():
    return union(*[Cyl('d', (s * R.POST, -R.POST), (R.HEAD_D + R.RESIN_HOLE_COMP) / 2, R.SPACER_SEAT, BOT + 1)
                   for s in (-1, 1)])


def _stubs():
    return union(*[Cyl('d', (s * R.POST, -R.POST), STUB_R, TOP, STUB_D) for s in (-1, 1)])


# ── hooks spliced into the v3 source ──
def _v4_below(below, v3_cuts):
    pads, cam_cuts = _cam_core()
    boss, floor_cuts = _floor_iface()
    body = union(below, _shells(), pads, *[_pad_support(s) for s in (-1, 1)], boss, _stubs(), _step_rib())   # front bosses: v3 cones
    return sub(inter(body, ENVELOPE), v3_cuts, _front_channels(), cam_cuts, floor_cuts, _driver_paths(), _web_notch(),
               *[Cyl('d', (bx, by), (R.INSERT_D + R.RESIN_HOLE_COMP) / 2, DEPTH - R.INSERT_DEPTH, BOT + 1)
                 for bx, by in FLOOR_BOSSES_V4[:2]])


def _v4_root(root):
    pads, cam_cuts = _cam_core()
    _, floor_cuts = _floor_iface()
    return sub(inter(root, ENVELOPE), cam_cuts, _front_channels(), floor_cuts, _driver_paths(), _rim_grooves(), _front_cuts())


_V3_PATCHES = [
    ('PI_SHIFT = 6.0 ', f'PI_SHIFT = {PI_SHIFT_V4} '),
    # no wall switch (floor button): its pad and hole go
    ("    Box(*((-INF, 0.0) if _sw['side'] < 0 else (0.0, INF)), _sw['y'] - SWITCH_PAD_R, _sw['y'] + SWITCH_PAD_R,\n"
     "        _sw['d'] - SWITCH_PAD_R, _sw['d'] + SWITCH_PAD_R),\n", ""),
    ("    hhole('x', _sw['y'], _sw['d'], _sw['hole'], *sorted((_sw['side'] * (X_IN - 0.01), _sw['side'] * X_OUT))),\n", ""),
    ("y0=-55.5, top=80.0,", f"y0={BRD_Y0_V4}, top={BRD_TOP_V4},"),
    ("        bbox(32.0, 42.0, 50.0, 62.0, 0.0, 22.0),                                    # J3 + plug + wires",
     "        bbox(32.0, 42.0, 50.0, 62.0, 0.0, 11.0), bbox(42.0, 48.0, 52.0, 60.0, 0.0, 6.0),   # J3 plug; wires leave +Y (home:2127)"),
    # -X trunk turn lifted to d 136: clears the floor button (home:2105)
    ("TRUNK = dict(yr=(3.5, 28.5), d=(14.0, 166.0),", "TRUNK = dict(yr=(3.5, 28.5), d={-1: (14.0, 136.0), 1: (14.0, 166.0)},"),
    ("""    rb = (t['yr'][1] - t['yr'][0] - t['cord']) / 2
    ra = (t['d'][1] - t['d'][0] - t['cord']) / 2
    yc, dc = (t['yr'][0] + t['yr'][1]) / 2, (t['d'][0] + t['d'][1]) / 2
""", """    rb = (t['yr'][1] - t['yr'][0] - t['cord']) / 2
    yc = (t['yr'][0] + t['yr'][1]) / 2
"""),
    ("""    for sx, n in t['turns'].items():
        for i in range(n):""", """    for sx, n in t['turns'].items():
        ra = (t['d'][sx][1] - t['d'][sx][0] - t['cord']) / 2   # -X turn clears the floor button
        dc = (t['d'][sx][0] + t['d'][sx][1]) / 2
        for i in range(n):"""),
    ("f'M12 trunk slack: 3 ellipse turns, 25 mm U (~{3 * np.pi * (3 * (ra + rb) - np.sqrt((3 * ra + rb) * (ra + 3 * rb))) / 10:.0f} cm)'",
     "'M12 trunk slack: 3 ellipse turns, 25 mm U (-X d 14..136, +X d 14..166)'"),
    ('PI_PORT_D = 102.0 ', 'PI_PORT_D = PI_TOP_D + 88.0 '),
    ('FLOOR_BOSSES = _L2.FLOOR_BOSSES ', 'FLOOR_BOSSES = _V4.FLOOR_BOSSES_V4 '),
    ("HINGE_WALL_HOLES = [(HINGE['yc'] + dy, DEPTH - 7.0) for dy in (-5.0, 5.0)]",
     "HINGE_WALL_HOLES = []"),
    ("LATCH_HOLES = [(LATCH['yc'], 145.0), (LATCH['yc'], 157.0)]", "LATCH_HOLES = []"),
    ("              sub(bumps, cuts))\n", "              sub(bumps, cuts))\nbelow = _V4._v4_below(below, cuts)\n"),
    ("ROOT = union(plate, below, fillet)\n", "ROOT = _V4._v4_root(union(plate, below, fillet))\n"),
    ("R.OVERLAY_DIR.parent / 'overlays_battery_v3'", "R.OVERLAY_DIR.parent / 'overlays_battery_v4'"),
    ("OVERLAYS = _v3_overlays()", "OVERLAYS = []"),
    ("return out + _layout_overlays() + _part_overlays()", "return out + _layout_overlays()"),   # its imports re-run v3
    ("UE300 = dict(", "PI_PLUGS = [(a + PI_SHIFT - 6.0, b + PI_SHIFT - 6.0, c, e, L) for a, b, c, e, L in PI_PLUGS]\nUE300 = dict("),
    ("PCB_YR - 11.0, PCB_YR - 2.0, -31.0, -20.0)", "PCB_YR - 11.0, PCB_YR - 2.0, -(PI_TOP_D + 17.0), -(PI_TOP_D + 6.0))"),
]


_FLIP_PATCHES = [
    ('for bx in (3.5, 61.5)]', 'for bx in (85.0 - 61.5, 85.0 - 3.5)]'),
    ('PI_PORT_D = PI_TOP_D + 88.0 ', 'PI_PORT_D = PI_TOP_D - 3.0 '),
    ('PI_PLUGS = [(a + PI_SHIFT - 6.0, b + PI_SHIFT - 6.0, c, e, L) for a, b, c, e, L in PI_PLUGS]',
     'PI_PLUGS = [(-(b - 6.0) + PI_SHIFT, -(a - 6.0) + PI_SHIFT, c, e, L) for a, b, c, e, L in PI_PLUGS]'),
    ('-(pm + L), -pm) for x0', '-pm, -(pm - L)) for x0'),
    ("px = -28.0 + PI_SHIFT ", "px = 28.0 + PI_SHIFT + 18.0 "),
    ("-(PI_TOP_D + 17.0), -(PI_TOP_D + 6.0))", "-(PI_TOP_D + 85.0 - 6.0), -(PI_TOP_D + 85.0 - 17.0))"),
    ("'overlays_battery_v4'", "'overlays_battery_v4flip'"),
]


def _load_v3():
    src = (Path(__file__).with_name('pi_box_bat_core_v3.py')).read_text(encoding='utf-8')
    for a, b in _V3_PATCHES + (_FLIP_PATCHES if FLIP else []):
        assert src.count(a) == 1, f'v3 patch does not match exactly once: {a!r}'
        src = src.replace(a, b)
    R.PI_TOP_D = PI_TOP_D_V4                       # read by the v3 source (import *) and by rig.pi_pt
    R.PI_FLIP, R.PI_FLIP_TOP = FLIP, PI_TOP_D_V4
    g = {'__name__': __name__ + '._v3', '__file__': __file__, '_V4': sys.modules[__name__]}
    exec(compile(src, 'pi_box_bat_core_v3.py (v4-patched)', 'exec'), g)
    return g


_g = _load_v3()
for _k, _v in _g.items():                          # expose the v3 names (constants, helpers, ROOT, core, ...)
    if not _k.startswith('__') and _k not in ('R', 'np'):
        globals().setdefault(_k, _v)
ROOT = _g['ROOT']
PI_SHIFT, PI_HOLES, BRD = _g['PI_SHIFT'], _g['PI_HOLES'], _g['BRD']
X0, Z0, SIZE_X, SIZE_Z = _g['X0'], _g['Z0'], _g['SIZE_X'], _g['SIZE_Z']
to_model = _g['to_model']
SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.3, 20.0
PIECES = [sdf_piece(SIZE_X, DEPTH, SIZE_Z, ROOT, to_model, x0=X0, y0=0.0, z0=Z0, aa=_g['AA'])]


def core(x, d, z, xp=np):
    from sdf import solid
    X, Y, dd = to_model(x, d, z)
    return solid(ROOT, X, Y, dd, xp)


def _v4_overlays():
    out = [o for o in _g['_v3_overlays']() if not any(k in o['file'] for k in ('v2_floor', 'latches', 'hinge', 'pi_flip', 'power_switch'))]   # the v4 floor is home/v4-floor's
    if 'designs.pi_box_v4_cam_cartridge' in sys.modules and not hasattr(sys.modules['designs.pi_box_v4_cam_cartridge'], 'ROOT'):
        return out                                 # imported from the cartridge module (circular)
    from designs import pi_box_v4_cam_cartridge as K
    for name, tris, color, label in K.rig_meshes():
        path = R.OVERLAY_DIR / f'{name}.stl'
        R._write_stl(path, R.to_print(R._decimate(tris, 0.2)))
        out.append(dict(file=path.relative_to(R.ROOT).as_posix(), color=color, opacity=1.0, label=label))
    return out


# Power button on the fixed floor (user approved, home:2092; home/v4-power-button): Ø16 latching switch at (-34, +13),
# pointing down, bezel on the floor's outer face, body (40 incl. its plug [MEASURE]) up into the -X side bay. Overlay
# only: the Ø16.35 hole and nut seat are in the floor (home/v4-floor).


def _floor_button_overlay():
    from designs.pi_box_bat_lids import FLOOR_T
    x, y = FLOOR_BUTTON
    face = DEPTH + FLOOR_T                         # floor outer face (d); its top face is DEPTH
    tris = np.concatenate([R._cyl((x, y, 0.0), 'z', 8.0, -DEPTH, -(DEPTH - 40.0), 32),        # body + plug, d 140..180 (measured ~40)
                           R._cyl((x, y, 0.0), 'z', 9.4, -(face + 2.5), -(face + 0.05), 32)])  # bezel Ø18.8 x 2.5 [MEASURE height]
    nut = R._cyl((x, y, 0.0), 'z', 9.4, -DEPTH + 0.05, -(DEPTH - 2.5), 32)                    # M16 nut 18.8 AC (measured)
    out = []
    for name, t, op, label in (('power_switch', tris, 0.8, 'power switch, 16 mm latching (15 V line), floor (-34, 13)'),
                               ('power_switch_nut', nut, 0.35, 'nut 18.8 AC (measured)')):
        path = R.OVERLAY_DIR / f'{name}.stl'
        R._write_stl(path, R.to_print(t))
        out.append(dict(file=path.relative_to(R.ROOT).as_posix(), color='#e74c3c', opacity=op, label=label))
    return out


# Cable route (Ø3, rig (X, Yr, d)); overlay only. J4 -> floor button: PROVISIONAL (home/v4-power-button owns it)
CABLE_ROUTES = [
    ('J4 -> floor button', [(-29, -40, 141), (-40, -40, 141), (-40, 8, 140), (-34, 13, 140)]),
]


def _cable_route_overlays():
    tris = np.concatenate([R._tube([(x, y, -d) for x, y, d in pts], 1.5) for _, pts in CABLE_ROUTES])
    path = R.OVERLAY_DIR / 'cable_routes.stl'
    R._write_stl(path, R.to_print(tris))
    return [dict(file=path.relative_to(R.ROOT).as_posix(), color='#c0392b', opacity=0.9,
                 label='cable routes (Ø3): ' + '; '.join(n for n, _ in CABLE_ROUTES))]


OVERLAYS = _v4_overlays() + _floor_button_overlay() + _cable_route_overlays()


# ── v4 front panel (plain; the styling comes later): v3's L-hook edge on the narrow front, |X| <= XE + hook ──
FRONT_HALF_W = XE + PV.NECK + PV.HOOK_T - PV.PRINT_GROW
_FRONT_NECK_X = XE + PV.NECK + PV.PRINT_GROW


def front_panel_mask(X, y, d):
    """X rig, y off the outer face toward the interior (0 .. PANEL_T + HOOK), d rig depth (as PV.panel_mask)."""
    s = np.ones(np.broadcast_shapes(np.shape(X), np.shape(d)), bool)
    body = y <= R.PANEL_T - PV.PRINT_GROW
    hook = (y <= R.PANEL_T + PV.HOOK - PV.PRINT_GROW) & (np.abs(X) >= _FRONT_NECK_X)
    s &= (np.abs(X) <= FRONT_HALF_W) & (body | hook) & (y >= 0) & (d >= PV.D0) & (d <= PV.D1)
    return s


# the wide front panel as a printable piece (print up = toward the flange: floor edge on the plate, tongues up)
FRONT_PANEL_PIECES = [implicit_piece(2 * X_OUT, DEPTH - FRONT_PANEL_D0, RC[1] + NECK_L + HOOK_W + 1.0 - Y_MIN,
                                     lambda x, y, z: front_panel_rig(x - X_OUT, z + Y_MIN, DEPTH - y),
                                     x0=-X_OUT, z0=Y_MIN)]
