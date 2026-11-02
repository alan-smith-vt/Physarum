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
# 2026-11-01 (user, after core v3): board shifted 22 toward the bank, straddling the web through a window knocked
# out of it, so the measured lidar plug (21.5 off the PCB back) clears the stick tube (it reached X -22 at Yr -11..-1)
# and the plug wires run under the Pi, clear of the GPIO harness. Same -X wall, same orientation (wire lengths).
BRD_Y0_V4, BRD_TOP_V4 = -21.0, 95.0                # power board Yr start (v3 -55.5): the -X round pulls the wall in
                                                   # forward of Yr -32; its +Y edge passes the web (notched). Top d 95.
# (a power board <-> UE300 side swap was modelled and reverted, user 2026-10-29 home:2102)

X_OUT, X_IN, Y_MIN, Y_MAX, DEPTH = R.X_OUT, R.X_IN, R.Y_MIN, R.Y_MAX, R.DEPTH
RC = np.array([X_OUT - R_FRONT, Y_MIN + R_FRONT])  # +X front round centre (X, Yr); -X mirrored
N2 = np.array([np.cos(YAW), -np.sin(YAW)])         # land normal (plan, +X side)
T2 = np.array([np.sin(YAW), np.cos(YAW)])          # land tangent (toward +Yr)
LAND = R_FRONT - LAND_OFF                          # land plane, n from RC
SEAT = 22.9                                        # cap seat plane (the countersink's outer edge)
CSK = 6.5                                          # 45° countersink depth: ±FL_T at SEAT .. ±OPEN_T at SEAT - CSK
PAD = SEAT - CSK - 1.0                             # pad frame inner face
OPEN_T, OPEN_D = 13.5, (25.0, 59.5)                # cartridge pod's shadow along n: half-width along the land, d range
FL_T, FL_D = OPEN_T + CSK, (OPEN_D[0] - CSK, OPEN_D[1] + CSK)
CART_CL = 0.2                                      # per side, + RESIN_HOLE_COMP / 2 in the core
CART_SCREWS = [(sc * 17.25, d) for sc in (-1, 1) for d in (31.0, 55.0)]   # (t, d) on the widened flange (CSK 6.5;
# user home:2242): >= 1.6 wall to the opening; no bottom screw (the ribbon leaves there, home:2195)
STUB_R, STUB_D = 7.1, 12.0
CART_INSERT_HOLE = 3.2                             # M2 insert 2.8 OD (calipered 2026-11-01): pocket Ø3.2 + comp, teardrop
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


def lteardrop(s, o, r, c0, c1):
    """Horizontal hole along the land normal n (axis (1, 0, 0)) from o (n, t, d), radius r, n-range c0..c1, with a 45°
    point toward +d (the print roof; the core prints flange down) so the hole doesn't bridge and sag undersize."""
    def f(X, Y, d, xp):
        n, t = _nt(s, X, Y)
        vn, vt, vd = n - o[0], t - o[1], d - o[2]
        circ = xp.sqrt(vt * vt + vd * vd) - r
        wedge = xp.maximum((xp.abs(vt) + vd) / np.sqrt(2.0) - r, r / np.sqrt(2.0) - vd)   # from the 45° tangent points up
        return xp.maximum(xp.minimum(circ, wedge), xp.maximum(c0 - vn, vn - c1))
    pts = [(o[0] + c, o[1] + kt * r, o[2] + kd) for c in (c0, c1) for kt in (-1, 1) for kd in (-r, r * 1.5)]
    return Fn(f, _aabb(s, pts))


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
from designs import pi_box_bat_panel_v3 as PV      # noqa: E402
FRONT_FACE = Y_MIN + PV.PRINT_GROW                 # centre strip outer face: flush with the envelope like the wide
_lo0 = FRONT_FACE - PV.CL                          # part (no lip, no thickness step at the step, user home:2276)
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
        # open under the opening for the ribbon (n 14.9..23, t ±8.9, d 59..80; home/v4-core home:2226): the ribbon jogs in
        # to n 16 and drops; corner at R 24.65, inside the step rib (R 24.8)
        cuts.append(sub(lbox(s, PAD - 3.0, 23.0, -(8.5 + g), 8.5 + g, OPEN_D[1] - 0.5, 80.0),
                        _groove_keep()))      # the step rib + panel-tongue groove stay whole (user, home:2224)
        for t, d in CART_SCREWS:
            # horizontal in print: teardrops (user 2026-11-01: the 2.8 OD inserts didn't enter the Ø3.35 pockets of core
            # v3 on either side; horizontal holes print undersize/oval). Insert pocket Ø3.2 + comp.
            cuts.append(lteardrop(s, (SEAT, t, d), (R.CAM_SCREW_CLEAR + R.RESIN_HOLE_COMP) / 2, -8.0, 8.0))
            cuts.append(lteardrop(s, (SEAT, t, d), (CART_INSERT_HOLE + R.RESIN_HOLE_COMP) / 2,
                                  PAD - SEAT - 1.0, PAD - SEAT + CART_NUT_L))  # insert (nut) from the pad's inner face
    return union(*pads), union(*cuts)


def _groove_keep():
    """The step rib and its groove, + 0.5: never cut for the ribbon (rad >= STEP_RIB r - 0.5 from d 63.5)."""
    def f(X, Y, d, xp):
        rad = xp.sqrt(xp.maximum(xp.abs(X) - XE, 0.0) ** 2 + xp.maximum(RC[1] - Y, 0.0) ** 2)
        return xp.maximum((24.0 - 0.5) - rad, 63.5 - d)
    return Fn(f, (-X_OUT, X_OUT, Y_MIN, RC[1] + 1, 63.0, 77.0))


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
DOOR_EXTRA = {1: 0.15, -1: 0.0}   # extra per-face clearance by side (user 2026-11-01: door A very tight on the +X
#   thumb-screw side of core v3, likely warp/rocking that high in the print): groove deeper + taller, web slot wider
# 2026-11-01: door A was printed before the to_goo MIRROR_FIX, so its screw is at X +41.1 (door_a.SCREW_XY): the
# lock boss moves to +X to keep that door. The floor is REPRINTED as designed.
PLUNGER = dict(x0=38.0, x1=46.0, y0=38.0, y1=50.0, d0=168.0, d1=180.0, hole=(41.1, 44.0), dia=4.0, L=6.0)   # M3 insert (door thumb screw, user 2026-10-29)
# 2026-11-01 (user): the +X web half moves WEB_DY toward the bank so the GbE adapter's front face (Yr -23.95) clears
# the front panel's L-hook channel (to Yr -24.6); the adapter is GLUED to the web's Pi face (no floor cradle).
WEB_DY = 1.5                                      # +X web: Yr 0..3 (the -X half stays at +-1.5)
ETH_ADAPTER_V4 = dict(x=37.9, y=-12.1, wx=15.4, wy=23.7, h=53.0)   # uni USB-A GbE adapter, RJ45 down (measured 2026-10-31)
FLOOR_BUTTON = (33.0, 14.0)                       # floor power button (home:2092; moved for the 18.8 AC nut, home:2149)
FLOOR_BOSSES_V4 = [(-42.0, -18.0), (45.0, 6.5)] + [(sx * 44.0, 24.0) for sx in (-1, 1)]   # front pair moved off the
# rounds (wide front panel, user approved home:2136; floor home/v4-floor); the +X one stays X >= 41.5 (home:2135)


def _floor_iface():
    g = R.RESIN_HOLE_COMP / 2
    r = DOOR_RAIL
    e = DOOR_EXTRA
    cuts = [Box(*sorted((s * (X_IN - 0.01), s * (X_IN + r['depth'] + g + e[s]))), r['y0'], Y_MAX + 1,
                r['d0'] - g - e[s], r['d1'] + g + e[s])
            for s in (-1, 1)]
    cuts += [inter(Box(*sorted((s * (X_IN - 0.01), s * (X_IN + r['depth'] + g + e[s] + 1.0))), Y_MAX - 1.0, Y_MAX + 1,
                       r['d0'] - g - e[s] - 1.0, r['d1'] + g + e[s] + 1.0),
                   Half(s, -1, 0, X_IN + r['depth'] + g + e[s] - (Y_MAX - 1.0)))  # 45° flare at the +Y mouth
             for s in (-1, 1)]
    p = PLUNGER
    boss = inter(Box(p['x0'], p['x1'], p['y0'], p['y1'], p['d0'] - (p['x1'] - p['x0']), p['d1']),
                 Half(-1, 0, -1, -(X_IN + p['d0'] - (p['x1'] - p['x0']))))   # |X| + d >= 206: 45° top (print: underside)
    cuts += [Box(*sorted((s * (X_IN - DOOR_WEB_CLEAR - DOOR_EXTRA[s]), s * (X_IN + 0.01))), r['y0'], Y_MAX + 1, 175.5, BOT + 1)
             for s in (-1, 1)]                     # the door's webs ride up inside each wall face (home:2006)
    # ...which leaves the cradle-rail bottoms (and the bits by the back panel channel) hanging as thin slabs off
    # the wall (user, 2026-10-29: remove them). Everything inboard of the web slot below d 175.5 goes, except the
    # lock boss (+-X), which keeps its insert depth.
    for s in (-1, 1):
        cuts += [Box(*sorted((s * 30.0, s * (X_IN - DOOR_WEB_CLEAR + 0.01))), y0, y1, 175.5, BOT + 1)
                 for y0, y1 in ((r['y0'], Y_MAX + 1),) if s < 0]
        cuts += [Box(*sorted((s * 30.0, s * (X_IN - DOOR_WEB_CLEAR + 0.01))), y0, y1, 175.5, BOT + 1)
                 for y0, y1 in ((r['y0'], PLUNGER['y0']), (PLUNGER['y1'], Y_MAX + 1)) if s > 0]
        # the rail ends rise at 45° inboard from the web slot (cut d - |X| >= 131.3): brackets off the wall, not slabs
        x_s = X_IN - DOOR_WEB_CLEAR
        for y0, y1 in (((r['y0'], Y_MAX + 1),) if s < 0 else ((r['y0'], PLUNGER['y0']), (PLUNGER['y1'], Y_MAX + 1))):
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


# ── Web as horizontal strips (user 2026-11-01; sketch designs/sketches/web_strips_sketch.png + user's red boxes) ──
# The continuous web with its slots/notches is gone. Per half (tube skin |X| 23.5 .. wall X_IN), (d0, d1):
#   both: strip A d 48..72 (25 tall; just above the board's J1/J2 plugs at d 73); above it open (the flange stiffens)
#   -X:   solid d 158..180 under the power board (its J3 plug ends d 157)
#   +X:   solid d 48..180 (user 2026-11-01: backs the Pi's +X upper mounting boss, X 26.5 d 112.5, which floated;
#         the GbE adapter is glued against it), with the old cable slot X 28.5..45.5, d 76..98 cut back in
# Each strip's flange-side edge (print underside: the core prints flange down) is a semicircular arch spanning the
# half, springing from the tube and the wall: no flat bridge.
WEB_STRIPS = {-1: [(48.0, 72.0), (158.0, BOT)], 1: [(48.0, BOT)]}
WEB_HOLES = {1: [(31.0, 45.5, 76.0, 98.0)]}   # |X| range, d range; 45° pointed far (floor) end. X from 31: the Pi +X upper boss + gusset reach X 29.7 (user 2026-11-01)


def _web_strips():
    x0, x1 = R.WELL_R + 1.0, X_IN + 0.5                       # into the tube wall and the outer wall
    xm, half = (R.WELL_R + R.TUBE_T + X_IN) / 2, (X_IN - R.WELL_R - R.TUBE_T) / 2
    out = []
    for s, strips in WEB_STRIPS.items():
        yc = WEB_DY if s > 0 else 0.0
        for d0, d1 in strips:
            def f(X, Y, d, xp, s=s, d0=d0, d1=d1, yc=yc):
                ax = s * X
                u = xp.clip((ax - xm) / half, -1.0, 1.0)
                edge = d0 + half * xp.sqrt(1.0 - u * u)            # arch: d0 at the springings, d0 + half at the crown
                f = xp.maximum(xp.maximum(x0 - ax, ax - x1), xp.abs(Y - yc) - R.WEB_T / 2)
                f = xp.maximum(f, xp.maximum(edge - d, d - d1))
                return f * 0.7                                     # the arch term isn't 1-Lipschitz near the springing
            out.append(Fn(f, (*sorted((s * x0, s * x1)), yc - R.WEB_T / 2, yc + R.WEB_T / 2, d0, d1)))
    holes = [inter(Box(*sorted((s * hx0, s * hx1)), -INF, INF, hd0, hd1 + (hx1 - hx0) / 2),
                   Half(-s, 0, 1, hd1 - hx0), Half(s, 0, 1, hd1 + hx1))
             for s, hs in WEB_HOLES.items() for hx0, hx1, hd0, hd1 in hs]
    return sub(union(*out), *holes)


def _web_notch():
    """Window in the -X web for the power board, which straddles it (2026-11-01): wall to the tube's skin, from
    25 above the board (J1/J2 plugs leave upward) to 3 below it (J3 plug + wires)."""
    w = (-R.WEB_T / 2 - 0.01, R.WEB_T / 2 + 0.01)
    return Box(-X_IN - 0.01, -(R.WELL_R + R.TUBE_T + 0.5), *w, BRD_TOP_V4 - 25.0, BRD_TOP_V4 + 63.0)


# Panel top-edge tongue groove under the back panel-stop rim (user approved home:2130; sketch
# designs/sketches/pi_box_v4_panel_tongue_sketch.py): 2.35 wide (tongue 1.6 + CL per side), 3.4 deep, 45° roof.
# Front: the centre strip's top edge (|X| <= FRONT_TONGUE_X) the same way.
TONGUE_T, TONGUE_H, TONGUE_GAP = 1.6, 3.0, 0.4
BACK_TONGUE_X = (6.2, X_IN - 2.5)                  # |X| range: gap at the lidar-cable notch
BACK_PANEL_YC = Y_MAX - R.PANEL_LIP - PV.CL - R.PANEL_T / 2
FRONT_TONGUE_X = XE - 2.5
FRONT_PANEL_YC = FRONT_FACE + R.PANEL_T - TONGUE_T / 2   # tongue on the strip's inner half: groove keeps a 1.4 outer skin


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


# power board support after the wide panel took H1 (user approved home:2175; spec home/v4-layout home:2160/2171):
# H2 screw as before, a bottom-edge ledge and a top-edge hook on the flat -X wall. 45° faces for print (+d up).
BRD_LEDGE = dict(y=(-2.0, 10.0), d=155.2, x_lip=41.4)   # shifted with the board; short of the J3 plug
BRD_HOOK = dict(y=(-3.0, 8.0), d0=93.0, x_lip=40.9)


def _board_support():
    b = BRD_LEDGE
    ledge = inter(Box(-X_IN - 0.5, -b['x_lip'], *b['y'], b['d'] - 2.6, b['d'] + 2.0),
                  Half(1, 0, -1, -(43.6 + b['d'])))          # |X| + d >= 198.8: 45° off the wall; the board's back
    h = BRD_HOOK                                             # corner (X -43.5, d 155) rests on it
    slot = Box(-43.6, -41.8, *h['y'], BRD_TOP_V4 - 0.2, BRD_TOP_V4 + 2.4)   # board top edge goes 2 into the hook
    hook = sub(inter(Box(-X_IN - 0.5, -h['x_lip'], *h['y'], h['d0'] - 8.0, BRD_TOP_V4 + 2.2),
                     Half(1, 0, -1, -(h['d0'] + h['x_lip']))),   # |X| + d >= d0 + x_lip: 45° off the wall (print)
               slot)
    return union(ledge, hook)


def _front_cuts():
    cl = PV.CL
    y_hook = RC[1] + NECK_L
    out = [Box(-X_OUT - 1, X_OUT + 1, Y_MIN - 1, RC[1], STEP, BOT + 1)]      # the panel's zone
    for s in (-1, 1):
        out.append(Box(*sorted((s * (X_IN - NECK_T - 2 * cl - 0.5), s * (X_IN + 0.01))), RC[1] - 1, y_hook + HOOK_W + cl,
                       STEP, BOT + 1))                                       # neck + hook clearance inside the wall
        out.append(Box(*sorted((s * (X_IN - 0.01), s * (X_IN + HOOK_T + cl))), y_hook - cl, y_hook + HOOK_W + cl,
                       STEP, BOT + 1))                                       # hook channel in the wall

        def flare(X, Y, d, xp, s=s, y0=y_hook - cl, y1=y_hook + HOOK_W + cl, x1=X_IN + HOOK_T + cl):
            out_d = xp.maximum(xp.maximum(y0 - Y, Y - y1), s * X - x1)       # 45° lead-in at the floor end, 1 mm
            return xp.maximum(out_d - (d - (BOT - 1.0)), (BOT - 1.0) - d)
        out.append(Fn(flare, (*sorted((s * (X_IN - 0.5), s * (X_IN + HOOK_T + cl + 1.5))), y_hook - cl - 1.5,
                              y_hook + HOOK_W + cl + 1.5, BOT - 1.5, BOT + 1)))
    g = TONGUE_T / 2 + cl
    top = STEP - TONGUE_H - TONGUE_GAP

    def groove(X, Y, d, xp):                                                 # 45° roof (print +d up)
        e = xp.abs(_front_rad(X, Y, xp) - R_MID)
        f = xp.maximum(e - g, (top - g + e) - d)
        f = xp.maximum(f, d - (STEP + 0.01))
        f = xp.maximum(f, (STEP_TONGUE_X0 - cl) - xp.abs(X))
        return xp.maximum(f, Y - (RC[1] - 1.0))
    out.append(Fn(groove, (-X_OUT, X_OUT, Y_MIN, RC[1], top - g - 1, STEP + 1)))
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
    face = FRONT_FACE                                                        # centre strip (+ 10 into the wide part)
    if d <= STEP + 10.0:
        y = Y - face
        # inner face flush with the wide part's (PT 3.2 - 2G = 2.85 thick; PANEL_T - G was 3.0, a 0.15 ledge
        # at the join, user 2026-11-02); the hook zone in the channel keeps its full thickness
        flush = (Y <= RC[1] - (R_FRONT - PT + G)) | ((ax >= _FRONT_NECK_X - 0.01) & (d <= STEP))   # hooks end at
        # the step, where the core's channel ends (they ran 10 mm past it in the overlap, user 2026-11-02)
        s |= front_panel_mask(np.broadcast_to(X, s.shape), y, d) & flush
        ty = np.abs(y - (R.PANEL_T - TONGUE_T / 2))
        dt = PV.D0 - TONGUE_H                                                # its top tongue into the rim
        if dt <= d < PV.D0 + 1.0:
            s |= (ty <= TONGUE_T / 2 - G) & (d >= dt + ty) & (ax <= FRONT_TONGUE_X)
    return s


FRONT_PANEL_D0 = PV.D0 - TONGUE_H


DRIVER_D1 = 26.0                                   # driver bore runs down past the cartridge pad frame (d 13..24)


def _driver_paths():
    """Clear Ø HEAD_D path to each front M4 head (user 2026-10-31: the cartridge pad frame's top, d 13..24, blocked
    the screw; was stub-only per home:2181). Screw first, then the cartridge; the bore ends inside its slot."""
    return union(*[Cyl('d', (s * R.POST, -R.POST), (R.HEAD_D + R.RESIN_HOLE_COMP) / 2, R.SPACER_SEAT, DRIVER_D1)
                   for s in (-1, 1)])


def _stubs():
    return union(*[Cyl('d', (s * R.POST, -R.POST), STUB_R, TOP, STUB_D) for s in (-1, 1)])


# ── hooks spliced into the v3 source ──
def _v4_below(below, v3_cuts):
    pads, cam_cuts = _cam_core()
    boss, floor_cuts = _floor_iface()
    body = union(below, _shells(), pads, *[_pad_support(s) for s in (-1, 1)], boss, _stubs(), _step_rib(), _board_support())   # front bosses: v3 cones
    return sub(inter(body, ENVELOPE), v3_cuts, _front_channels(), cam_cuts, floor_cuts, _driver_paths(), _web_notch(),
               *[Cyl('d', (bx, by), (R.INSERT_D + R.RESIN_HOLE_COMP) / 2, DEPTH - R.INSERT_DEPTH, BOT + 1)
                 for bx, by in FLOOR_BOSSES_V4[:2]])


# ── Snap-off tree supports under the cartridge-slot floors (user 2026-11-01; they sagged 1-2 mm on core v3) ──
# Each slot floor (d ~67, ~19 x 33 per side, flat, facing the flange) is a 166 mm2 bridge in print. Trees as on the
# door (pi_box_v4_blue_plate): a spiked foot on the slot's upper edge (d ~17.4, straight below), a vertical trunk,
# one split into 4 branches at 35° off vertical, point tips dug 0.25 into the floor. Snapped out through the slot.
# Trunks on a (n, t) grid in each camera's land frame; a trunk is kept only if its whole column is clear of resin
# (2 mm margin) and both ends land on resin.
TREE = dict(grid_n=tuple(np.arange(14.0, 36.0, 4.0)), grid_t=tuple(np.arange(-14.0, 14.1, 4.0)), split=2.0,
            tan=np.tan(np.radians(35.0)), r_trunk=0.8, r_branch=0.6, r_tip=0.25, spike=1.5, dig=0.25, margin=1.0,
            floor_min=65.0)   # tips only on the slot floor (d ~67), not the pad underside (d 60.5)


def _rod(a, b, r0, r1):
    """Tapered rod a -> b in rig (X, Yr, d), radius r0 at a, r1 at b (an SDF bound: scaled for the taper)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ab = b - a
    L2 = float(ab @ ab)

    def f(X, Y, d, xp):
        k = xp.clip(((X - a[0]) * ab[0] + (Y - a[1]) * ab[1] + (d - a[2]) * ab[2]) / L2, 0.0, 1.0)
        dx, dy, dz = X - a[0] - k * ab[0], Y - a[1] - k * ab[1], d - a[2] - k * ab[2]
        return (xp.sqrt(dx * dx + dy * dy + dz * dz) - (r0 + (r1 - r0) * k)) * 0.9
    r = max(r0, r1) + 0.5
    return Fn(f, (min(a[0], b[0]) - r, max(a[0], b[0]) + r, min(a[1], b[1]) - r, max(a[1], b[1]) + r,
                  min(a[2], b[2]) - r, max(a[2], b[2]) + r))


def _slot_trees(body):
    def solid(x, y, d):
        return float(np.asarray(body(np.array([[x]]), np.array([[y]]), float(d), np, None, 0)).ravel()[0]) <= 0

    T = TREE
    rods, kept = [], []
    for sx in (-1, 1):
        for n in T['grid_n']:
            for t in T['grid_t']:
                xy = RC + n * N2 + t * T2
                X, Y = sx * xy[0], xy[1]
                top = next((d for d in np.arange(55.0, 75.0, 0.1) if solid(X, Y, d)), None)       # floor underside
                if top is None or top < T['floor_min']:
                    continue
                tips_ok = all(next((d for d in np.arange(55.0, 75.0, 0.1) if solid(X + ex * T['split'], Y + ey * T['split'], d)),
                                   0.0) >= T['floor_min'] for ex in (-1, 1) for ey in (-1, 1))
                if not tips_ok:
                    continue
                bot = next((d for d in np.arange(top - 1.0, 5.0, -0.1) if solid(X, Y, d)), None)  # resin below
                if bot is None or top - bot < 20.0:
                    continue
                # the column (incl. the 4 tips' spread) must be clear between the ends
                r = T['split'] + T['margin']
                ok = all(not solid(X + dx, Y + dy, d) for d in np.arange(bot + 2.0, top - 1.0, 2.0)
                         for dx, dy in ((0, 0), (r, 0), (-r, 0), (0, r), (0, -r)))
                if not ok:
                    continue
                kept.append((round(X, 1), round(Y, 1), round(bot, 1), round(top, 1)))
                split_d = top - T['split'] * np.sqrt(2) / T['tan']
                foot = (X, Y, bot - T['dig'])
                rods.append(_rod(foot, (X, Y, bot + T['spike']), T['r_tip'], T['r_trunk']))         # spiked foot
                rods.append(_rod((X, Y, bot + T['spike']), (X, Y, split_d), T['r_trunk'], T['r_trunk']))
                for ex, ey in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                    tip = (X + ex * T['split'], Y + ey * T['split'], top + T['dig'])
                    rods.append(_rod((X, Y, split_d), tip, T['r_branch'], T['r_tip']))
    global SLOT_TREES
    SLOT_TREES = kept
    return union(*rods)


SLOT_TREES = []


def _v4_root(root):
    pads, cam_cuts = _cam_core()
    _, floor_cuts = _floor_iface()
    body = sub(inter(root, ENVELOPE), cam_cuts, _front_channels(), floor_cuts, _driver_paths(), _rim_grooves(), _front_cuts())
    return union(body, _slot_trees(body))


_V3_PATCHES = [
    # v3 door latch / hinge wall pads: v4 has neither (thumb-screw door), so their solid squares in the wall
    # gyroid go (user 2026-10-31: a random solid square under the floor lock)
    ("    Box(-X_OUT, -X_IN + 0.5, LATCH['yc'] - LATCH['w'] / 2 - 3, LATCH['yc'] + LATCH['w'] / 2 + 3, LATCH['d0'] - 3, LATCH['d1'] + 3),\n"
     "    Box(X_IN - 0.5, X_OUT, HINGE['yc'] - 10, HINGE['yc'] + 10, DEPTH - HINGE['leaf'] - 3, BOT),   # latch / hinge pads\n", ''),
    ('PI_SHIFT = 6.0 ', f'PI_SHIFT = {PI_SHIFT_V4} '),
    # no wall switch (floor button): its pad and hole go
    ("    Box(*((-INF, 0.0) if _sw['side'] < 0 else (0.0, INF)), _sw['y'] - SWITCH_PAD_R, _sw['y'] + SWITCH_PAD_R,\n"
     "        _sw['d'] - SWITCH_PAD_R, _sw['d'] + SWITCH_PAD_R),\n", ""),
    ("    hhole('x', _sw['y'], _sw['d'], _sw['hole'], *sorted((_sw['side'] * (X_IN - 0.01), _sw['side'] * X_OUT))),\n", ""),
    ("y0=-55.5, top=80.0,", f"y0={BRD_Y0_V4}, top={BRD_TOP_V4},"),
    # board screw clearance blind, 5 deep (to X -48.5): the shared PI_HOLE_DEPTH 8 broke the outer skin (user home:2220)
    ("PI_SCREW_CLEAR, BRD['back'] - PI_HOLE_DEPTH, BRD['back'] + 0.01)", "PI_SCREW_CLEAR, BRD['back'] - 5.0, BRD['back'] + 0.01)"),
    # uni USB-A GbE adapter replaces the UE300 (user, home:2158; spec home/v4-layout home:2160): seat ledge + overlay
    ("_a = ETH_ADAPTER\n", "_a = _V4.ETH_ADAPTER_V4\n"),
    ("UE300 = dict(x0=29.9, x1=45.9, y0=-27.6, y1=-1.6, d0=DEPTH - 71.0)",
     "UE300 = dict(x0=29.9, x1=45.9, y0=-23.95, y1=-0.25, d0=DEPTH - 53.0)"),
    # +X web half shifted WEB_DY toward the bank (2026-11-01), with its seat ledge, foot cut, floor-collar notch
    ("webs = sub(union(*[sbox(s, WELL_R + 1, X_IN, -WEB_T / 2, WEB_T / 2) for s in (-1, 1)]),\n           *[web_slot(s, *ws) for s in (-1, 1) for ws in WEB_SLOTS])",
     "webs = _V4._web_strips()   # 2026-11-01: strips, no slots"),
    ("seat = inter(Box(_a['x'] - _a['wx'] / 2, _a['x'] + _a['wx'] / 2, -WEB_T / 2 - ETH_SEAT['reach'], -WEB_T / 2 + 0.5,",
     "seat = inter(Box(_a['x'] - _a['wx'] / 2, _a['x'] + _a['wx'] / 2, _V4.WEB_DY - WEB_T / 2 - ETH_SEAT['reach'], _V4.WEB_DY - WEB_T / 2 + 0.5,"),
    ("             Half(0, -1, -1, -(_seat_d - ETH_SEAT['reach'] - WEB_T / 2)))",
     "             Half(0, -1, -1, -(_seat_d - ETH_SEAT['reach'] - WEB_T / 2 + _V4.WEB_DY)))"),
    ("feet = union(*[inter(sbox(s, FOOT_X0, X_OUT, -WEB_T / 2 if s > 0 else -FOOT_Y, FOOT_Y,",
     "feet = union(*[inter(sbox(s, FOOT_X0, X_OUT, _V4.WEB_DY - WEB_T / 2 if s > 0 else -FOOT_Y, FOOT_Y,"),
    ("               Box(-X_IN, X_IN, -WEB_T / 2 - 0.5, WEB_T / 2 + 0.5))]))",
     "               Box(-X_IN, X_IN, -WEB_T / 2 - 0.5, WEB_T / 2 + _V4.WEB_DY + 0.5))]))"),
    ("'TP-Link UE300 (as v2)'", "'uni USB-A GbE adapter 53 x 23.7 x 15.4 (measured)'"),
    # solid wall (no gyroid) round the wide front panel's hook channels, 3+ mm margin (user, home:2148), and on up
    # past the camera cartridges to the flange (home:2179)
    ("    Box(-X_OUT, X_OUT, -WEB_PAD, WEB_PAD),\n",
     "    Box(-X_OUT, X_OUT, -WEB_PAD, WEB_PAD),\n"
     f"    Box(-X_OUT, X_OUT, {RC[1] - 3.0}, {RC[1] + 5.0 + 2.0 + 4.0}, TOP, BOT),\n"),
    ("        bbox(32.0, 42.0, 50.0, 62.0, 0.0, 22.0),                                    # J3 + plug + wires",
     "        bbox(32.0, 42.0, 50.0, 62.0, 0.0, 19.9),                                    # J3 lidar plug 19.5 + 2 (measured 2026-11-01)"),
    ("        bbox(7.0, 29.0, 52.0, 60.0, 0.0, 14.0),                                     # J5 + PH plug",
     "        bbox(7.0, 29.0, 52.0, 60.0, 0.0, 10.4),                                     # J5 GPIO plug 10 + 2 (measured)"),
    ("        bbox(0.5, 6.0, 34.0, 47.0, 0.0, 14.0),                                      # J4 + XH plug",
     "        bbox(0.5, 6.0, 34.0, 47.0, 0.0, 11.9),                                      # J4 button plug 11.5 + 2 (measured)"),
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
    # M12 slack loops dropped (user 2026-11-01: loop it outside the case if needed)
    ("    add('m12_trunk', np.concatenate(coils),", "    None and add('m12_trunk', np.concatenate(coils),"),
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
SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.8, 20.0   # smoky black + clear Tenacious 75/25: exposure_test_smoke winner 80 % of 3.5 s (2026-10-31)
PIECES = [sdf_piece(SIZE_X, DEPTH, SIZE_Z, ROOT, to_model, x0=X0, y0=0.0, z0=Z0, aa=_g['AA'])]


def core(x, d, z, xp=np):
    from sdf import solid
    X, Y, dd = to_model(x, d, z)
    return solid(ROOT, X, Y, dd, xp)


def _v4_overlays():
    out = [o for o in _g['_v3_overlays']() if not any(k in o['file'] for k in ('v2_floor', 'latches', 'hinge', 'pi_flip', 'power_switch', 'bank_socket', 'usbc_plug_envelope'))]   # the v4 floor is home/v4-floor's
    if 'designs.pi_box_v4_cam_cartridge' in sys.modules and not hasattr(sys.modules['designs.pi_box_v4_cam_cartridge'], 'ROOT'):
        return out                                 # imported from the cartridge module (circular)
    from designs import pi_box_v4_cam_cartridge as K
    for name, tris, color, label in K.rig_meshes():
        path = R.OVERLAY_DIR / f'{name}.stl'
        R._write_stl(path, R.to_print(R._decimate(tris, 0.2)))
        out.append(dict(file=path.relative_to(R.ROOT).as_posix(), color=color, opacity=1.0, label=label))
    return out


# Power button on the fixed floor (user approved, home:2092; home/v4-power-button): Ø16 latching switch at FLOOR_BUTTON,
# pointing down, bezel on the floor's outer face, body (40 incl. its plug [MEASURE]) up into the -X side bay. Overlay
# only: the Ø16.35 hole and nut seat are in the floor (home/v4-floor).


def _floor_button_overlay():
    from designs.pi_box_bat_lids import FLOOR_T
    x, y = FLOOR_BUTTON
    face = DEPTH + FLOOR_T                         # floor outer face (d); its top face is DEPTH
    tris = np.concatenate([R._cyl((x, y, 0.0), 'z', 8.0, -DEPTH, -(DEPTH - 40.0), 32),        # body + plug, d 140..180 (measured ~40)
                           R._cyl((x, y, 0.0), 'z', 8.75, -(face + 1.8), -(face + 0.05), 32)])  # bezel Ø17.5 x 1.8 (measured)
    nut = R._cyl((x, y, 0.0), 'z', 9.4, -DEPTH + 0.05, -(DEPTH - 2.5), 32)                    # M16 nut 18.8 AC (measured)
    out = []
    for name, t, op, label in (('power_switch', tris, 0.8, 'power switch, 16 mm latching (15 V line), floor (33, 14)'),
                               ('power_switch_nut', nut, 0.35, 'nut 18.8 AC (measured)')):
        path = R.OVERLAY_DIR / f'{name}.stl'
        R._write_stl(path, R.to_print(t))
        out.append(dict(file=path.relative_to(R.ROOT).as_posix(), color='#e74c3c', opacity=op, label=label))
    return out


# Cable route (Ø3, rig (X, Yr, d)); overlay only. J4 -> floor button: PROVISIONAL (home/v4-power-button owns it)
CABLE_ROUTES = [
    ('J4 -> floor button', [(-29, -12.3, 132), (-29, 0, 140), (-23, 12, 140), (-12, 26, 150), (12, 26, 150),
                            (30, 20, 150), (33, 14, 141)]),   # under the board, through the web window, round the
    # tube on the bank side (Yr 26: tube r 23.5, bank from 30.8) to the +X button (2026-11-01)
]


def _cable_route_overlays():
    tris = np.concatenate([R._tube([(x, y, -d) for x, y, d in pts], 1.5) for _, pts in CABLE_ROUTES])
    path = R.OVERLAY_DIR / 'cable_routes.stl'
    R._write_stl(path, R.to_print(tris))
    return [dict(file=path.relative_to(R.ROOT).as_posix(), color='#c0392b', opacity=0.9,
                 label='cable routes (Ø3): ' + '; '.join(n for n, _ in CABLE_ROUTES))]


# Pi power: flat 90° USB-C adapter (user home:2146; home/v4-layout home:2156), rig (X, Yr, d), all [MEASURE]
# MEASURED 2026-11-01 (user): the plugged-in 90° adapter reaches 11.5 past the Pi's board edge (X 28) and is 25 long,
# turning UP (toward the flange / lidar), d ~167 -> ~142; then the cable's own USB-C plug goes into its end [MEASURE].
USBC_PLUG = [(28.0, 39.5, -41.0, -34.0, 142.0, 167.0),   # adapter body
             (30.0, 38.0, -40.5, -34.5, 120.0, 142.0)]   # cable plug + boot [MEASURE: ~22 assumed]


# Pi 5 GPIO header (board bx 7..58, by 49.5..55.5) with the soldered harness: rigid female ends to 17 off the PCB
# back (user 2026-11-01) + 3 for the wires leaving them
GPIO_HARNESS = dict(bx=(7.0, 58.0), by=(49.5, 55.5), h=20.0)


def _gpio_overlay():
    g = GPIO_HARNESS
    (xa, da), (xb, db) = R.pi_pt(g['bx'][0], g['by'][0]), R.pi_pt(g['bx'][1], g['by'][1])
    tris = R._box(min(xa, xb), max(xa, xb), R.PCB_YR - g['h'], R.PCB_YR, -max(da, db), -min(da, db))
    path = R.OVERLAY_DIR / 'pi_gpio_harness.stl'
    R._write_stl(path, R.to_print(tris))
    return [dict(file=path.relative_to(R.ROOT).as_posix(), color='#8e44ad', opacity=0.8,
                 label='Pi GPIO harness: female ends 17 + wires 3 off the PCB back (measured)')]


def _usbc_overlay():
    path = R.OVERLAY_DIR / 'usbc_plug_envelope.stl'
    R._write_stl(path, R.to_print(np.concatenate([R._box(x0, x1, y0, y1, -d1, -d0) for x0, x1, y0, y1, d0, d1 in USBC_PLUG])))
    return [dict(file=path.relative_to(R.ROOT).as_posix(), color='#e74c3c', opacity=0.8,
                 label='Pi USB-C: 90° adapter 11.5 x 25 (measured), turns up; cable plug 22 [MEASURE]')]


# Ethernet adapter's USB-A plug (in the flipped Pi's USB3 port, pointing up) and its Ø4 braided cord (R >= 10 bends;
# ~126 mm needed of 142), home/v4-layout home:2160, [MEASURE]
ETH_USBA_PLUG = (-10.0, 2.0, -43.3, -38.8, 56.0, 86.0)
ETH_CORD = [(-4, -41, 56), (-4, -36, 44), (-4, -27, 52), (20, -27, 70), (33, -20, 100), (37.9, -15, 127)]


def _eth_overlays():
    x0, x1, y0, y1, d0, d1 = ETH_USBA_PLUG
    out = []
    for name, tris, color, label in (
            ('eth_usba_plug', R._box(x0, x1, y0, y1, -d1, -d0), '#7f8c8d', 'GbE adapter USB-A plug in the USB3 port [MEASURE]'),
            ('eth_cord', R._tube([(x, y, -d) for x, y, d in ETH_CORD], 2.0), '#2c3e50', 'GbE adapter cord Ø4, ~126 of 142 mm')):
        path = R.OVERLAY_DIR / f'{name}.stl'
        R._write_stl(path, R.to_print(tris))
        out.append(dict(file=path.relative_to(R.ROOT).as_posix(), color=color, opacity=0.9, label=label))
    return out


OVERLAYS = _v4_overlays() + _floor_button_overlay() + _cable_route_overlays() + _usbc_overlay() + _gpio_overlay() + _eth_overlays()

# legend groups (viewer: one click hides a whole group; 2026-11-01)
OVERLAY_GROUPS = {
    'Pi': ('pi5', 'cooler', 'pi_plugs', 'usbc_plug_envelope', 'pi_gpio_harness'),
    'Power board': ('power_board', 'power_board_parts'),
    'Ethernet': ('ue300', 'eth_usba_plug', 'eth_cord'),
    'Power button': ('power_switch', 'power_switch_nut', 'cable_routes'),
    'Power bank': ('battery',),
    'Lidar + mount': ('mid360s', 'base_plate', 'standoffs', 'box_spacers', 'coupler', 'monopod', 'm12_plug', 'm12_trunk'),
    'Cameras': ('v4_',),
    'Printed parts': ('asm_',),
}


def group_overlays(ovs):
    for ov in ovs:
        stem = ov['file'].rsplit('/', 1)[-1][:-4]
        for g, names in OVERLAY_GROUPS.items():
            if any(stem == n or (n.endswith('_') and stem.startswith(n)) for n in names):
                ov['group'] = g
                break
    return ovs


OVERLAYS = group_overlays(OVERLAYS)


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
