"""v3 CONCEPT (2026-10-15, copy of pi_box_bat_core.py). Changes from v2:
  - cameras back near the web (CAM_YR -16.5, behind the Pi's PCB plane) as CARTRIDGES inserted from outside
    (designs/pi_box_cam_cartridge.py): the core keeps only a pointed-top opening in each camera wall plus
    three M2 insert-nut pockets (from inside) for the cartridge's flange screws. No camera blocks, inserts or filed seats.
  - Pi shifted +X by PI_SHIFT: 24 mm from its edge to the -X wall for the right-angle USB-C plug + adapter.
  - removed: camera ribs, all cable clips, the UE300 seat ledge, the M4 bores through the camera blocks.
Not checked yet (clash, paths, layer areas): concept for review.

BATTERY variant of pi_box_core.py (copy): taller, +Y pod with battery stop ribs, laid on
the plate turned 90° (print x = rig Yr) so its 140 mm length fits the long axis.

Lidar rig Pi box, v1 core (one print): flange, well tube, webs, camera walls.

Prints flange-down (the face that mates with the aluminium base plate is on the
build plate) as an open-bottomed ring, so layers are thin walls. Vents through
the flange stop the pockets acting as suction cups. Print it on its own.
Geometry and frames: designs/pi_box_bat_rig.py. Floor: designs/pi_box_bat_lids.py.

Modelled as a signed-distance tree (sdf.py, 2026-10-13): every feature is a primitive with a bounding
box, so slices are evaluated tile by tile on only the features that reach each tile, on the CPU or
GPU (to_goo --gpu). PI_BOX_AA=1 turns on anti-aliased (grayscale) edges: off by default until gray
cure has been calibrated (gray edge pixels shift fits). `core(x, d, z)` still returns the solid mask
for the export / check / overhang scripts.

Core features (rig depth d = print Y):
  - flange 0..10: 4 stepped bores (Ø8 spacer seat 5.85 deep, Ø10.2 head pocket),
    recesses for the lidar's M3 heads, vents, 1 mm chamfer on the plate face
  - Ø52 well tube; webs from the tube to each camera wall (X axis)
  - ±X camera walls 4 mm with: panel grooves at both ends (open at the bottom),
    a foot round each web with dowel holes, floor-screw bosses with glued
    M3 insert holes, the OV9281 window + camera block with M2-insert holes + lens guard ring
  - Pi 5 standoff posts from the tube (glued M2.5 inserts)
  Horizontal holes are drawn ~0.2 taller (hole_h) for print sag.
"""
import os
import numpy as np
from sdf import Box, Cyl, Half, Fn, union, inter, sub, sdf_piece, solid, INF, TIE
from designs.pi_box_bat_rig import *      # noqa: F401,F403  (constants + helpers)
from designs import pi_box_bat_rig as R

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0
AA = os.environ.get('PI_BOX_AA', '0') == '1'

# ── v3 ──
PI_SHIFT = 6.0                                     # Pi +X: USB-C room 46 - (28 - 6) = 24 mm
CAM_YR = -16.5                                     # lens axis Yr (shadows the rig's v2 corner position)
PI_HOLES = [(sx * 24.5 + PI_SHIFT, PI_TOP_D + bx) for sx in (-1, 1) for bx in (3.5, 61.5)]
CART_HW = 13.5                                     # cartridge body half-width (Yr); board 25 wide
CART_D0, CART_D1 = CAM_D - 13.5, CAM_D + 15.6 + 0.5   # flat near edge; far edge past the board, then a 45° point
CART_CLEAR = 0.2                                   # per side, + RESIN_HOLE_COMP / 2 in the opening
CART_FL, CART_FL_T = 4.0, 1.5                      # flange margin past the body, thickness (outside the wall)
CART_SCREWS = [(CAM_YR - CART_HW - 2.0, CAM_D), (CAM_YR + CART_HW + 2.0, CAM_D), (CAM_YR, CART_D0 - 2.0)]
CART_SCREW_CLEAR = 2.2                             # M2 from outside through flange + wall...
CART_NUT_L = 2.5                                   # ...into an M2 insert set in a pocket from the INNER face: it bottoms on
                                                   # the wall's outer 1.5 mm, so the screw pulls it into resin (a nut; CA only
                                                   # locates it). Flange 1.5 + wall 1.5 + insert 2.5 → M2×5 / M2×6 [MEASURE]
CAM_PAD = (CART_HW + CART_FL + 1.0, CART_D1 + CART_HW + CART_FL + 1.0 - CAM_D)   # solid round the opening + flange


def cam_board(s):
    return CAM_YR - 12.5, CAM_YR + 12.5, CAM_D - 8.25, CAM_D + 15.6


def cam_holes(s):
    return [(CAM_YR + sy * 10.5, CAM_D + sd * 6.25) for sy in (-1, 1) for sd in (-1, 1)]


def cart_profile(hw, d0, d1, x0=-INF, x1=INF):
    """Pentagon in (Yr, d): flat at d0, sides at CAM_YR ± hw, 45° point past d1 (self-supporting in print)."""
    return inter(Box(x0, x1, CAM_YR - hw, CAM_YR + hw, d0, d1 + hw),
                 Half(0, 1, 1, d1 + hw + CAM_YR), Half(0, -1, 1, d1 + hw - CAM_YR))

X0, Z0 = Y_MIN, -(GUARD_X + 0.5)          # piece corner (print x = Yr, print z = -X)
SIZE_X, SIZE_Z = Y_MAX - Y_MIN, 2 * (GUARD_X + 0.5)
GW = GROOVE_W + RESIN_HOLE_COMP           # panel slot width as cut
TOP, BOT = FLANGE_PLATE, DEPTH            # below-plate features live in d = TOP..BOT


def to_model(x, y, z):
    """Local print coords -> rig (X, Yr, d)."""
    return -(z + Z0), x + X0, y


# ── helpers: sided boxes and the resin-compensated holes ──
def sbox(s, a0, a1, y0, y1, d0=TOP, d1=BOT):
    """Box over |X| in [a0, a1] on side s (+1 / -1)."""
    return Box(*sorted((s * a0, s * a1)), y0, y1, d0, d1)


def vhole(cx, cy, dia, d0, d1):
    """Vertical hole (along d), +RESIN_HOLE_COMP (see rig.hole)."""
    return Cyl('d', (cx, cy), (dia + RESIN_HOLE_COMP) / 2, d0, d1)


def hhole(axis, cu, cd, dia, h0, h1):
    """Horizontal hole along 'x' (centre Yr, d) or 'y' (centre X, d): +comp, and taller (rig.hole_h)."""
    ru = (dia + RESIN_HOLE_COMP) / 2
    rd = (dia + RESIN_HOLE_COMP + HORIZ_HOLE_EXTRA) / 2
    return Cyl(axis, (cu, cd), ru, h0, h1, r2=rd)


def taper(s, c):
    """|X| + d >= c on side s: a face growing off the wall at 45° (narrows toward the flange)."""
    return Half(-s, 0, -1, -c)


# ── wall fill: TPMS sheet distance (sign exact, magnitude a bound) ──
def wall_fill_dist(X, Y, d, xp):
    kind, a, b = R.WALL_FILL.split(':')
    a, b = float(a), float(b)
    k = 2 * np.pi / a
    t = min(max((d - (DEPTH - WALL_BOTTOM_SOLID - R.FILL_RAMP)) / R.FILL_RAMP, 0.0), 1.0)
    if kind == 'gyroid':
        sd, cd = np.sin(k * d), np.cos(k * d)
        g = xp.sin(k * X) * xp.cos(k * Y) + xp.sin(k * Y) * cd + sd * xp.cos(k * X)
        b = b + t * (1.6 - b)
    elif kind == 'schwarzp':
        g = xp.cos(k * X) + xp.cos(k * Y) + np.cos(k * d)
        b = b + t * (3.1 - b)
    elif kind == 'diamond':
        sx, sy, cx, cy = xp.sin(k * X), xp.sin(k * Y), xp.cos(k * X), xp.cos(k * Y)
        sz, cz = np.sin(k * d), np.cos(k * d)
        g = sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz
        b = b + t * (1.6 - b)
    else:
        raise NotImplementedError(f"PI_BOX_FILL={R.WALL_FILL}: the SDF core supports gyroid / schwarzp / diamond")
    return (xp.abs(g) - b) / (k * 1.8)


# ── flange plate (d 0..4) ──
plate = sub(
    inter(Box(-X_OUT, X_OUT, Y_MIN, Y_MAX, 0.0, FLANGE_PLATE),
          Half(1, 0, -1, X_OUT - CHAMFER), Half(-1, 0, -1, X_OUT - CHAMFER),     # 1 mm chamfer on the plate face
          Half(0, 1, -1, Y_MAX - CHAMFER), Half(0, -1, -1, -Y_MIN - CHAMFER)),
    Cyl('d', (0.0, 0.0), WELL_R),
    *[vhole(sx * POST, sy * POST, SPACER_D + SPACER_FIT, 0.0, FLANGE_PLATE) for sx in (-1, 1) for sy in (-1, 1)],
    *[vhole(sx * LIDAR_M3[0], sy * LIDAR_M3[1], M3_HEAD_D, 0.0, M3_HEAD_RECESS) for sx in (-1, 1) for sy in (-1, 1)])
# (the flange vents are gone: the open flange below has no closed pockets to act as suction cups)

# ── walls: solid zones + gyroid inside the 4 mm slabs ──
_sw = POWER_SWITCH
solid_zone = union(
    Box(-X_OUT, X_OUT, Y_MIN, Y_MAX, TOP, WALL_TOP_SOLID),
    Box(-X_OUT, X_OUT, Y_MIN, Y_MAX, DEPTH - WALL_BOTTOM_SOLID, BOT),
    Box(-X_OUT, X_OUT, Y_MIN, Y_MIN + PANEL_LIP + GW + 4.0),
    Box(-X_OUT, X_OUT, Y_MAX - PANEL_LIP - GW - 4.0, Y_MAX),
    Box(-X_OUT, X_OUT, CAM_YR - CAM_PAD[0], CAM_YR + CAM_PAD[0], TOP, CAM_D + CAM_PAD[1]),
    Box(*((-INF, 0.0) if _sw['side'] < 0 else (0.0, INF)), _sw['y'] - SWITCH_PAD_R, _sw['y'] + SWITCH_PAD_R,
        _sw['d'] - SWITCH_PAD_R, _sw['d'] + SWITCH_PAD_R),
    Box(-X_OUT, X_OUT, -WEB_PAD, WEB_PAD),
    *[Box(-X_OUT, X_OUT, ry - BAT_RAIL_W / 2, ry + BAT_RAIL_W / 2) for ry in BAT_RAIL_YS],
    Box(-X_OUT, X_OUT, BAT_RIB[2], BAT_RAIL_YS[0]))
gyroid = Fn(wall_fill_dist, (-X_OUT, X_OUT, Y_MIN, Y_MAX, TOP, BOT))
walls = union(*[inter(sbox(s, X_IN, X_OUT, Y_MIN, Y_MAX), union(solid_zone, gyroid)) for s in (-1, 1)])



def web_slot(s, x0, x1, d0, d1):
    """Pass-through with a 45° pointed far end (no flat bridge)."""
    return inter(sbox(s, x0, x1, -INF, INF, d0, d1 + (x1 - x0) / 2),
                 Half(-s, 0, 1, d1 - x0), Half(s, 0, 1, d1 + x1))


webs = sub(union(*[sbox(s, WELL_R + 1, X_IN, -WEB_T / 2, WEB_T / 2) for s in (-1, 1)]),
           *[web_slot(s, *ws) for s in (-1, 1) for ws in WEB_SLOTS])
tube = sub(Cyl('d', (0.0, 0.0), WELL_R + TUBE_T, TOP, BOT), Cyl('d', (0.0, 0.0), WELL_R))

# ── panel-stop rims (RIM_D deep) with the lidar cable notch, bore bosses (FLANGE deep) ──
_notch_r = (CABLE_NOTCH_W + RESIN_HOLE_COMP) / 2
rims = union(Box(-X_OUT, X_OUT, Y_MIN, Y_MIN + PANEL_LIP + GW + 1.5, TOP, RIM_D),
             sub(Box(-X_OUT, X_OUT, Y_MAX - PANEL_LIP - GW - 1.5, Y_MAX, TOP, RIM_D),
                 Box(-_notch_r, _notch_r, Y_MAX - PANEL_LIP - GW - 1.6, Y_MAX + 1, FLANGE_PLATE, RIM_D + 1)))
bosses = union(*[Cyl('d', (sx * POST, sy * POST), CORNER_BOSS_R, TOP, FLANGE) for sx in (-1, 1) for sy in (-1, 1)])
bores = union(*[vhole(sx * POST, sy * POST, SPACER_D + SPACER_FIT, TOP - 1, SPACER_SEAT) for sx in (-1, 1) for sy in (-1, 1)],
              *[vhole(sx * POST, sy * POST, HEAD_D, SPACER_SEAT, FLANGE) for sx in (-1, 1) for sy in (-1, 1)])

# ── battery: stop ribs merged with the first cradle rail (stepped flap), second rail; 45° shoulders ──
stop_ribs = union(*[inter(sbox(s, BAT_RIB[0], BAT_RIB[1], BAT_RIB[2], BAT_RIB[3], BAT_RAIL_D0, BOT),
                          taper(s, X_IN + BAT_RAIL_D0)) for s in (-1, 1)])
def _rail(s, y0, y1):
    """Rail beside the bank, plus a flat seat on its top face (d <= d0). Flat, not 45°: the bank's
    rounded corners let it sink ~5-10 mm past a sloped shoulder (core print 2026-10-14)."""
    return inter(sbox(s, BAT_SHOULDER_X, X_IN + 0.5, y0, y1),
                 union(Half(-s, 0, 0, -BAT_RAIL_X), Half(0, 0, 1, BAT['d0'])))


rails = union(*[_rail(s, BAT_RIB[3] if i == 0 else ry - BAT_RAIL_W / 2, ry + BAT_RAIL_W / 2)
                for s in (-1, 1) for i, ry in enumerate(BAT_RAIL_YS)])



# ── push-in cable clips ──
def clip(axis, base, along, d0, D, out=None):
    o = out if out else -np.sign(base)                 # the side the clip stands on
    r_in = (D + RESIN_HOLE_COMP) / 2
    r_out = r_in + CLIP_T
    uc = base + o * (r_out + CLIP_STEM)                # cable centre across the surface
    w = 0.85 * D / 2
    d1 = d0 + CLIP_H

    def at(u0, u1, v0, v1):                            # box in (u across, v along) -> (X, Y)
        u0, u1 = sorted((u0, u1))
        return Box(u0, u1, v0, v1, d0, d1) if axis == 'x' else Box(v0, v1, u0, u1, d0, d1)
    cen = (uc, along) if axis == 'x' else (along, uc)
    ring = sub(Cyl('d', cen, r_out, d0, d1), Cyl('d', cen, r_in),
               at(uc, uc + o * (r_out + 1), along - w, along + w))          # the push-in opening
    stem = at(base - o * 0.5, base + o * (CLIP_T + CLIP_STEM), along - 1.5, along + 1.5)
    lead = Half(o, 0, -1, o * base - d0 + 1) if axis == 'x' else Half(0, o, -1, o * base - d0 + 1)
    return inter(union(ring, stem), lead)



# ── wall feet round the webs (45° from the wall), floor-screw boss cones ──
feet = union(*[inter(sbox(s, FOOT_X0, X_OUT, -WEB_T / 2 if s > 0 else -FOOT_Y, FOOT_Y, DEPTH - FOOT_H - (X_IN - FOOT_X0), BOT),
                     taper(s, X_IN + DEPTH - FOOT_H - (X_IN - FOOT_X0)))     # |X| >= 212 - d
               for s in (-1, 1)])


def _cone(bx, by):
    d_on = DEPTH - FOOT_H - (BOSS_R - (abs(abs(bx) - X_IN) + 0.5))      # boss starts (radius 0+) here
    r_full = d_on - (abs(abs(bx) - X_IN) + 0.5)                          # radius = min(BOSS_R, d - r_full)

    def f(X, Y, d, xp):
        if d <= d_on:
            return INF
        return xp.sqrt((X - bx) ** 2 + (Y - by) ** 2) - min(BOSS_R, d - r_full)
    return Fn(f, (bx - BOSS_R, bx + BOSS_R, by - BOSS_R, by + BOSS_R, d_on, BOT))


cones = union(*[_cone(bx, by) for bx, by in BOSSES])

# ── Pi 5 standoff posts (along Yr from the tube to the PCB back) + 45° gussets below ──
_well = Cyl('d', (0.0, 0.0), WELL_R)
posts = []
for _px, _pd in PI_HOLES:
    _yt = -np.sqrt(max(WELL_R ** 2 - _px ** 2, 0.0))
    _L = PI_POST_R + (_yt - PCB_YR)
    posts += [sub(Cyl('y', (_px, _pd), PI_POST_R, PCB_YR, 0.0), _well),
              sub(inter(Box(_px - PI_POST_R, _px + PI_POST_R, PCB_YR, _yt + 1, max(_pd - _L, TOP), _pd),
                        Half(0, -1, -1, -(PCB_YR + _pd - PI_POST_R))), _well)]
posts = union(*posts)
pi_holes = union(*[hhole('y', px, pd, PI_INSERT_D, PCB_YR, PCB_YR + PI_INSERT_L) for px, pd in PI_HOLES],
                 *[hhole('y', px, pd, PI_SCREW_CLEAR, PCB_YR, PCB_YR + PI_HOLE_DEPTH) for px, pd in PI_HOLES])


# ── cuts ──
_lo0, _lo1 = Y_MIN + PANEL_LIP, Y_MIN + PANEL_LIP + GW
_hi0, _hi1 = Y_MAX - PANEL_LIP - GW, Y_MAX - PANEL_LIP
panel_cuts = union(*[sbox(s, X_IN, X_IN + GROOVE_D, y0, y1, RIM_D, BOT + 1)
                     for s in (-1, 1) for y0, y1 in ((_lo0, _lo1), (_hi0, _hi1))],
                   Box(-X_IN, X_IN, Y_MIN - 1, _lo1, RIM_D, BOT + 1),
                   Box(-X_IN, X_IN, _hi0, Y_MAX + 1, RIM_D, BOT + 1))
cuts = union(
    bores, pi_holes,
    hhole('x', _sw['y'], _sw['d'], _sw['hole'], *sorted((_sw['side'] * (X_IN - 0.01), _sw['side'] * X_OUT))),
    *[cart_profile(CART_HW + CART_CLEAR + RESIN_HOLE_COMP / 2, CART_D0 - CART_CLEAR - RESIN_HOLE_COMP / 2,
                   CART_D1 + CART_CLEAR, *sorted((s * (X_IN - 0.01), s * (X_OUT + 0.01)))) for s in (-1, 1)],   # cartridge openings
    *[hhole('x', cy, cd, CART_SCREW_CLEAR, *sorted((s * (X_IN - 0.01), s * (X_OUT + 0.01))))
      for s in (-1, 1) for cy, cd in CART_SCREWS],
    *[hhole('x', cy, cd, CAM_INSERT_D, *sorted((s * (X_IN - 0.01), s * (X_IN + CART_NUT_L))))
      for s in (-1, 1) for cy, cd in CART_SCREWS],
    *[Cyl('d', (px, py), PIN_SLIP / 2, DEPTH - PIN_DEPTH, BOT + 1) for px, py in PINS],
    *[vhole(bx, by, INSERT_D, DEPTH - INSERT_DEPTH, BOT + 1) for bx, by in BOSSES])
# anti-rattle bumps on each slot's outer face (added back after the slot cut)
bumps = union(*[sbox(s, X_IN + 0.5, X_IN + GROOVE_D - 0.5, y0, y1, db - 1.5, db + 1.5)
                for s in (-1, 1) for db in (RIM_D + 25.0, DEPTH - 25.0)
                for y0, y1 in ((_lo0, _lo0 + 0.3), (_hi1 - 0.3, _hi1))])

below = union(sub(union(tube, walls, webs, rims, bosses, stop_ribs, rails, feet, cones, posts), cuts, panel_cuts),
              sub(bumps, cuts))


# ── open flange (2026-10-14): the first core split ~5 mm up where the solid 4 mm plate (~9,700 mm²/layer)
#    met the walls (~3,800). The plate keeps only what the body below hangs off (plus rings round the M3
#    recesses, a perimeter band, a ring round the well and the panel-midpoint spokes + diamond ribs),
#    blurred (σ FLANGE_SIGMA) and thresholded into smooth windows (~6,400 mm²/layer). A quarter-round
#    fillet (FLANGE_FILLET) joins the body to the plate, only where there is plate under it. ──
def _flange_field(step=0.1):
    from scipy.ndimage import distance_transform_edt, gaussian_filter
    xs = np.arange(-X_OUT, X_OUT + step, step)
    ys = np.arange(Y_MIN, Y_MAX + step, step)
    XX, YY = np.meshgrid(xs, ys, indexing='ij')
    req = solid(below, xs[:, None], ys[None, :], FLANGE_PLATE + 0.5)
    for sx in (-1, 1):
        for sy in (-1, 1):
            req |= np.hypot(XX - sx * LIDAR_M3[0], YY - sy * LIDAR_M3[1]) <= M3_HEAD_D / 2 + 2.0
    b = FLANGE_BAND
    req |= (np.abs(XX) >= X_OUT - b) | (YY <= Y_MIN + b) | (YY >= Y_MAX - b)
    r = np.hypot(XX, YY)
    req |= (r >= WELL_R) & (r <= WELL_R + 3.0)
    ye = Y_MAX - PANEL_LIP - GW                         # panel edge (inner face of its slot)

    def seg(ax, ay, bx, by):
        vx, vy = bx - ax, by - ay
        t = np.clip(((XX - ax) * vx + (YY - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        return np.hypot(XX - ax - t * vx, YY - ay - t * vy) <= FLANGE_RIB_W / 2
    mids = [(0.0, -ye), (X_IN, 0.0), (0.0, ye), (-X_IN, 0.0)]
    req |= seg(0.0, -ye, 0.0, -WELL_R) | seg(0.0, ye, 0.0, WELL_R)              # panel midpoints -> tube
    for i in range(4):
        req |= seg(*mids[i], *mids[(i + 1) % 4])                                  # diamond
    keep = gaussian_filter(req.astype(float), FLANGE_SIGMA / step) > FLANGE_THRESH
    sd = (distance_transform_edt(~keep) - distance_transform_edt(keep)) * step
    return xs[0], ys[0], step, sd.astype(np.float32)


_fx0, _fy0, _fst, _fsd = _flange_field()
_fsd_dev = {}


def _flange_sd(X, Y, d, xp):
    """Signed distance to the flange's solid footprint (bilinear in the 0.1 mm field)."""
    g = _fsd if xp is np else _fsd_dev.setdefault(id(xp), xp.asarray(_fsd))
    u = xp.clip((X - _fx0) / _fst, 0, g.shape[0] - 1.001)
    v = xp.clip((Y - _fy0) / _fst, 0, g.shape[1] - 1.001)
    i, j = xp.floor(u).astype(int), xp.floor(v).astype(int)
    fu, fv = u - i, v - j
    i, j = xp.broadcast_arrays(i, j)
    fu, fv = xp.broadcast_arrays(fu, fv)
    return ((1 - fu) * (1 - fv) * g[i, j] + fu * (1 - fv) * g[i + 1, j]
            + (1 - fu) * fv * g[i, j + 1] + fu * fv * g[i + 1, j + 1])


_flange_bounds = (-X_OUT, X_OUT, Y_MIN, Y_MAX, -INF, INF)
plate = inter(plate, Fn(_flange_sd, _flange_bounds))
_fillet_keep_out = union(*[Cyl('d', (sx * POST, sy * POST), (HEAD_D + RESIN_HOLE_COMP) / 2 + 0.3)
                           for sx in (-1, 1) for sy in (-1, 1)],                 # head counterbores stay full size
                         Box(-_notch_r - 0.3, _notch_r + 0.3, Y_MAX - PANEL_LIP - GW - 2.5, INF))   # lidar cable notch


def _fillet(X, Y, d, xp):
    h = d - FLANGE_PLATE
    if not 0.0 < h < FLANGE_FILLET:
        return INF
    e = FLANGE_FILLET - np.sqrt(max(FLANGE_FILLET ** 2 - (FLANGE_FILLET - h) ** 2, 0.0))    # quarter-round reach
    grown = below(X, Y, d, xp) - e
    top = plate(X, Y, FLANGE_PLATE - 0.01, xp)                                  # only over plate
    return xp.maximum(xp.maximum(grown, top), TIE - _fillet_keep_out(X, Y, d, xp))


fillet = Fn(_fillet, (-X_OUT, X_OUT, Y_MIN, Y_MAX, FLANGE_PLATE, FLANGE_PLATE + FLANGE_FILLET))
ROOT = union(plate, below, fillet)


def core(x, d, z, xp=np):
    """Solid mask at layer depth d (x (1, W), z (H, 1) local print mm) - for export / checks."""
    X, Y, dd = to_model(x, d, z)
    return solid(ROOT, X, Y, dd, xp)


PIECES = [sdf_piece(SIZE_X, DEPTH, SIZE_Z, ROOT, to_model, x0=X0, y0=0.0, z0=Z0, aa=AA)]


def _v3_overlays():
    """The rig's overlays with the Pi shifted and the cameras at CAM_YR; stale routes dropped; plug envelope added."""
    R.OVERLAY_DIR = R.OVERLAY_DIR.parent / 'overlays_battery_v3'
    R.CAM_YR = CAM_YR
    _pi_pt = R.pi_pt
    R.pi_pt = lambda bx, by: (_pi_pt(bx, by)[0] + PI_SHIFT, _pi_pt(bx, by)[1])
    stale = ('eth_adapter', 'cable_usbc', 'cable_ribbons')
    out = [o for o in R.build_overlays() if not any(k in o['file'] for k in stale)]
    px = -28.0 + PI_SHIFT                          # right-angle USB-C plug ~10 + adapter allowance 8, toward -X
    path = R.OVERLAY_DIR / 'usbc_plug_envelope.stl'
    R._write_stl(path, R.to_print(R._box(px - 18.0, px, PCB_YR - 11.0, PCB_YR - 2.0, -31.0, -20.0)))
    out.append(dict(file=path.relative_to(R.ROOT).as_posix(), color='#e74c3c', opacity=0.8,
                    label='USB-C right-angle plug 10 + adapter 8 (envelope)'))
    return out


OVERLAYS = _v3_overlays()
