"""Shared geometry for the lidar rig's Pi box (core, floor, panels) + viewer overlays.

Box frame (mm): origin at the underside centre of the aluminium base plate,
X = across the camera walls, Yr = Pi (-) to buck bay (+), Zr = up. The box hangs
below the plate, so we mostly use depth d = -Zr. The box is turned 90° on the
square 80 × 80 standoff pattern so its buck bay faces the lidar's M12 connector:
the lidar's connector axis is box +Yr (LIDAR_YAW).

Print frame (Physarum: X/Z plate plane, Y up): the core prints flange-down on
the build plate, so  print (x, y, z) = (X, d, Yr).  That's a 180° turn about X
of the rig, not a mirror.

Concept and reasoning: D:\\Notes\\Reference\\lidar-photo-rig-cage-handoff.md
(Pi box section, 2026-10-13). Sizes marked [MEASURE] are guesses until the
parts are in hand.
"""
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # Physarum/
MODELS = ROOT / 'models'
OVERLAY_DIR = ROOT / 'slicer_preview' / 'out' / 'overlays'

RESIN_HOLE_COMP = 0.35        # holes/slots print ~0.35 undersize (cage v1 fit strip)

# ── Box envelope ──
X_OUT, X_IN = 50.0, 46.0      # camera wall outer / inner face (|X|)
Y_MIN, Y_MAX = -60.0, 47.0    # panel outer faces
DEPTH, FLANGE = 105.0, 10.0   # core bottom edge / flange thickness (depth below the plate)
CHAMFER = 1.0                 # flange plate-face edge: elephant foot + scraper entry

# ── Well (coupler + monopod platform pass through) ──
WELL_R, TUBE_T = 26.0, 2.0

# ── Box mount: M4×18 + 6 mm Al spacer (Ø8) per standoff, stepped bore ──
POST = 40.0
SPACER_D, SPACER_SEAT = 8.0, 5.85     # print 5.85 deep, sand until the spacer sits flush
HEAD_D = 10.2                         # M4 truss head ~9.5 + clearance [MEASURE]

# ── Lidar M3 button heads stand proud under the plate: (±24, ±18) in the lidar's frame
#    (24 along its connector axis), so (±18, ±24) in the box frame ──
LIDAR_YAW = 90.0                      # lidar frame -> box frame, degrees about Zr (connector -> +Yr)
LIDAR_M3 = (18.0, 24.0)
M3_HEAD_D, M3_HEAD_RECESS = 7.0, 2.5

# ── Panels slide up in grooves in the camera-wall ends ──
PANEL_T = 2.5
GROOVE_W = PANEL_T + 0.3              # + RESIN_HOLE_COMP applied where cut
GROOVE_D = 2.0                        # into the wall from its inner face

# ── Webs, wall feet, dowels, floor bosses ──
WEB_T = 3.0
WEB_SLOTS = [(30.0, 41.0, 24.0, 34.0), (30.0, 41.0, 82.0, 93.0)]   # |X| range, depth range: cable pass-throughs
FOOT_X0, FOOT_Y, FOOT_H = 42.0, 9.0, 10.0          # |X| 42..50, |Yr| < 9, bottom 10 mm
PINS = [(sx * 46.0, sy * 5.5) for sx in (-1, 1) for sy in (-1, 1)]
PIN_SLIP, PIN_PRESS, PIN_DEPTH = 3.4, 3.3, 4.0      # fit-strip values (no extra comp)
BOSSES = [(sx * 44.0, y) for sx in (-1, 1) for y in (-52.0, 39.5)]
BOSS_R, INSERT_D, INSERT_DEPTH = 4.5, 4.0, 6.0      # glued M2.5 brass insert [MEASURE]

# ── Flange vents (no suction cups when printed flange-down) ──
VENTS = [(0, -45), (-15, -45), (15, -45), (0, 38), (-15, 38), (15, 38),
         (37, 20), (37, -20), (-37, 20), (-37, -20)]
VENT_D = 4.0

# ── Pi 5 (stands on its short edge, ports down, HDMI edge toward -X) ──
PCB_YR = -31.0                # back (underside) of the PCB
PI_TOP_D = 14.0               # SD/PCIe end of the board; the card stops ~2 mm short of the flange
PI_HOLES = [(sx * 24.5, PI_TOP_D + bx) for sx in (-1, 1) for bx in (3.5, 61.5)]
PI_POST_R, PI_PILOT = 3.2, 2.2            # M2.5 self-tap (horizontal hole)

# ── OV9281 cameras (25 × 23.86 board, lens 18 mm from the board face) ──
# Lens axes just under the flange: the stiffest spot, ~55 mm from the lidar instead of ~90.
CAM_YR, CAM_D = -15.0, 23.0               # lens axis (both cameras)
CAM_WINDOW = 16.0                         # square opening for the lens holder [MEASURE]
CAM_BOSS_R, CAM_PILOT, CAM_BOSS_H = 2.25, 1.6, 2.0
# Connector edge direction per side. Board 25 × 23.86; the lens sits 8.25 from the edge opposite
# the connector and the 21 × 12.5 holes are symmetric about it. The -X (near) camera turns its
# connector down so its ribbon runs down to the Pi, clear of the Pi's USB-C plug ~25 mm down.
CAM_CONN_DIR = {-1: 'down', 1: '-y'}


def cam_holes(s):
    a, b = (6.25, 10.5) if CAM_CONN_DIR[s] == '-y' else (10.5, 6.25)
    return [(CAM_YR + sy * a, CAM_D + sd * b) for sy in (-1, 1) for sd in (-1, 1)]


def cam_board(s):
    """(Yr0, Yr1, d0, d1) of the board outline."""
    if CAM_CONN_DIR[s] == '-y':
        return CAM_YR - 15.6, CAM_YR + 8.25, CAM_D - 12.5, CAM_D + 12.5
    return CAM_YR - 12.5, CAM_YR + 12.5, CAM_D - 8.25, CAM_D + 15.6
GUARD_R0, GUARD_R1, GUARD_X = 9.5, 11.5, 57.0     # lens guard ring, stops ~5 mm behind the lens front

# ── Exhaust slots in the camera walls beside the Pi ──
EXHAUST = [(-38 - 4 * i, -36 - 4 * i) for i in range(4)]   # Yr ranges
EXHAUST_D = (20.0, 45.0)


def hole(cx, cy, d, a, b):
    """Disk test for a hole drawn at the wanted size, compensated for resin shrink."""
    r = (d + RESIN_HOLE_COMP) / 2
    return (a - cx) ** 2 + (b - cy) ** 2 <= r * r


def to_print(pts):
    """Rig (X, Yr, Zr) -> print (x, y, z)."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([pts[..., 0], -pts[..., 2], pts[..., 1]], axis=-1)


# ═════════════════════════════ overlays ═════════════════════════════
def _write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, tris
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + struct.pack('<I', len(tris)))
        f.write(rec.tobytes())


def _box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box in rig coords -> triangles."""
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c, d in q] + [[v[a], v[c], v[d]] for a, b, c, d in q])


def _cyl(c, axis, r, h0, h1, n=32):
    """Cylinder along rig axis 'x'|'y'|'z' through point c, from h0 to h1 along that axis."""
    t = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.stack([np.cos(t), np.sin(t)], 1) * r
    ax = 'xyz'.index(axis)
    oth = [i for i in range(3) if i != ax]

    def pt(u, w, h):
        p = np.array(c, dtype=float)
        p[oth[0]] += u
        p[oth[1]] += w
        p[ax] = h
        return p

    tris = []
    for i in range(n):
        a0, a1 = ring[i], ring[i + 1]
        p00, p01 = pt(*a0, h0), pt(*a1, h0)
        p10, p11 = pt(*a0, h1), pt(*a1, h1)
        tris += [[p00, p01, p11], [p00, p11, p10],
                 [pt(0, 0, h0), p01, p00], [pt(0, 0, h1), p10, p11]]
    return np.array(tris)


def _frames(pts, hint):
    """Per-segment direction and a perpendicular 'width' vector (from hint)."""
    pts = np.asarray(pts, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = (b - a) / np.linalg.norm(b - a)
        w = np.cross(t, hint)
        if np.linalg.norm(w) < 1e-6:
            w = np.cross(t, [1.0, 0, 0] if abs(t[0]) < 0.9 else [0, 1.0, 0])
        w /= np.linalg.norm(w)
        out.append((a, b, t, w, np.cross(w, t)))
    return out


def _tube(pts, r, n=10):
    """Round cable along a polyline (rig coords); joints get a short sleeve overlap."""
    tris = []
    ang = np.linspace(0, 2 * np.pi, n + 1)
    for a, b, t, u, v in _frames(pts, [0, 0, 1.0]):
        a, b = a - t * r * 0.5, b + t * r * 0.5
        ring = [np.cos(q) * u * r + np.sin(q) * v * r for q in ang]
        for i in range(n):
            tris += [[a + ring[i], a + ring[i + 1], b + ring[i + 1]], [a + ring[i], b + ring[i + 1], b + ring[i]]]
    return np.array(tris)


def _strip(pts, width, thick, hint):
    """Flat ribbon along a polyline; its width lies across the path, perpendicular to hint."""
    tris = []
    for a, b, t, w, n in _frames(pts, hint):
        tris.append(_quad_box(a, b, w * width / 2, n * thick / 2))
    return np.concatenate(tris)


def _quad_box(a, b, hw, hn):
    c = [a - hw - hn, a + hw - hn, a + hw + hn, a - hw + hn, b - hw - hn, b + hw - hn, b + hw + hn, b - hw + hn]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return np.array([[c[i], c[j], c[k]] for i, j, k, l in f] + [[c[i], c[k], c[l]] for i, j, k, l in f])


# ── Cable routes (rig X, Yr, Zr; Zr = -depth). Schematic: they show the path and the
#    space it needs, not the slack. Ribbons leave the Pi connectors along the board
#    (upward assumed), 22-pin FFC ~13 mm wide. ──
PI_CONN = [(-19.0, -35.5, -(PI_TOP_D + 49)), (-19.0, -35.5, -(PI_TOP_D + 57))]     # near, far
CAM_CONN = [(-(X_IN - CAM_BOSS_H - 3.6), CAM_YR, -(CAM_D + 15.0)),     # -X near: connector edge down
            (X_IN - CAM_BOSS_H - 3.6, CAM_YR - 15.0, -CAM_D)]          # +X far: connector edge toward -Yr
RIBBON_NEAR = [PI_CONN[0], (-26, -35.5, -60), (-26, -35.5, -50), (-36, -24, -44), CAM_CONN[0]]
RIBBON_FAR = [PI_CONN[1], (-26, -35.5, -68), (-26, -35.5, -42), (-29.8, -35.5, -40), (-29.8, -29.6, -40),
              (29.8, -29.6, -40), (37, -30, -30), CAM_CONN[1]]
# Pi 5 port end (bottom, ~99-104 mm down): Ethernet is the jack beside the micro-HDMI edge (-X here,
# board y ~2-18 -> X -26..-10); the two USB-A stacks are toward +X.
ETH = (-18.0, -40.0, -(PI_TOP_D + 88.5))
# Lidar cable: off the M12 connector (box +Yr), straight down outside the +Y panel and in through a
# notch at its top (the panel slides up round it, no plug threading), a flat coil
# in the buck bay, through the low web slot to the Pi side, and out of the floor notch to come up
# into the downward-facing Ethernet jack. The power pair splits off at the coil to the WAGOs.
PANEL_NOTCH_X, PANEL_NOTCH_D, PANEL_NOTCH_W = 30.0, 20.0, 8.0     # notch centre X, bottom depth, width
COIL = dict(c=(-5.0, 37.0, -82.0), r=17.0, turns=2.0, y0=34.0, y1=40.0)


def _coil_pts(n=40):
    cx, _, cz = COIL['c']
    t = np.linspace(0, 2 * np.pi * COIL['turns'], int(n * COIL['turns']) + 1)
    y = COIL['y0'] + (COIL['y1'] - COIL['y0']) * t / t[-1]
    return [(cx + COIL['r'] * np.cos(a), yy, cz + COIL['r'] * np.sin(a)) for a, yy in zip(t, y)]


_COIL = _coil_pts()
LIDAR_CABLE = ([(0, 40, 20), (0, 54, 20), (0, 54, 0), (PANEL_NOTCH_X, 54, -14), (PANEL_NOTCH_X, 53, -16),
                (PANEL_NOTCH_X, 40, -17), (24, 36, -50)]
               + _COIL
               + [(37, 20, -85), (37, -45, -95), (8, -52, -100), (5, -46, -118), (-18, -42, -118), ETH])
LIDAR_POWER = [_COIL[-1], (20, 36, -76), (24, 35, -74)]          # Livox splitter's power pair -> WAGO
BATTERY_LEAD = [(0, -16, -140), (-8, -52, -118), (-8, -52, -100), (-37, -45, -95), (-37, 20, -85),
                (-30, 34, -80), (18, 35, -74)]                     # up the pole, in at the floor notch, round the tube
BUCK_IN = [(14, 36, -38), (19, 36, -66)]
USBC_LEAD = [(-22, 36, -30), (-37, 20, -30), (-37, -20, -28), (-33, -34, -25)]   # right-angle plug at the Pi


def _decimate(tris, grid=0.25):
    """Vertex clustering: snap to a grid and drop collapsed triangles (overlay only)."""
    q = np.round(tris / grid) * grid
    a, b, c = q[:, 0], q[:, 1], q[:, 2]
    keep = ~(np.all(a == b, 1) | np.all(b == c, 1) | np.all(a == c, 1))
    return q[keep]


def _load(name):
    from stl_slicer import load_stl
    return load_stl(MODELS / name).astype(np.float64)


def build_overlays():
    """Write transformed overlay STLs (print frame) and return the viewer list.

    Reference models (gitignored, see models/README.md): rpi5.stl from the official
    Raspberry Pi 5 STEP (MIT), mid360s.stl from Livox's Mid-360S STEP.
    """
    out = []

    def emit(name, rig_tris, color, opacity=0.35, label=None):
        path = OVERLAY_DIR / f'{name}.stl'
        _write_stl(path, to_print(rig_tris))
        out.append(dict(file=str(path.relative_to(ROOT)).replace('\\', '/'), color=color, opacity=opacity,
                        label=label or name.replace('_', ' ')))

    # Pi 5: model board x 0..85 (SD end at 0), y 0..56 (HDMI edge at 0), z up (components)
    if (MODELS / 'rpi5.stl').exists():
        t = _decimate(_load('rpi5.stl'))
        bx, by, bz = t[..., 0], t[..., 1], t[..., 2]
        emit('pi5', np.stack([by - 28.0, PCB_YR - bz, -(PI_TOP_D + bx)], -1), '#2ecc71', 0.55, 'Pi 5 (official STEP)')
    # Active Cooler envelope [MEASURE]: over the SoC area, ~14 mm above the board
    emit('cooler', _box(-24, 18, PCB_YR - 1.6 - 14, PCB_YR - 1.6, -(PI_TOP_D + 62), -(PI_TOP_D + 12)), '#95a5a6', 0.35, 'Active Cooler envelope [MEASURE]')

    # Mid-360S: model Y up (base at -25.92), connector +X; sits on the 6.35 plate
    if (MODELS / 'mid360s.stl').exists():
        t = _decimate(_load('mid360s.stl'), 0.15)
        emit('mid360s', np.stack([t[..., 2], t[..., 0], t[..., 1] + 25.92 + 6.35], -1), '#bdc3c7', 0.5, 'Mid-360S (Livox STEP)')

    # Aluminium plate, standoffs + spacers, coupler, monopod platform + pole
    emit('base_plate', _box(-45, 45, -45, 45, 0, 6.35), '#7f8c8d', 0.3)
    posts = np.concatenate([_cyl((sx * POST, sy * POST, 0), 'z', 3, 6.35, 72.35)
                            for sx in (-1, 1) for sy in (-1, 1)])
    spacers = np.concatenate([_cyl((sx * POST, sy * POST, 0), 'z', SPACER_D / 2, -SPACER_D + 2, 0)
                              for sx in (-1, 1) for sy in (-1, 1)])
    emit('standoffs', posts, '#7f8c8d', 0.3)
    emit('box_spacers', spacers, '#e67e22', 0.7)
    emit('coupler', _cyl((0, 0, 0), 'z', 6.4, -20.3, 0, 6), '#7f8c8d', 0.6)
    emit('monopod', np.concatenate([_cyl((0, 0, 0), 'z', 22.5, -30.3, -20.3),
                                    _cyl((0, 0, 0), 'z', 15, -200, -30.3)]), '#34495e', 0.3)

    # OV9281 boards + lenses (board face on 2 mm bosses off the wall)
    cams = []
    for s in (-1, 1):
        xf = s * (X_IN - CAM_BOSS_H)                      # board front face
        y0, y1, d0, d1 = cam_board(s)
        cams.append(_box(min(xf, xf - s), max(xf, xf - s), y0, y1, -d1, -d0))
        cams.append(_cyl((0, CAM_YR, -CAM_D), 'x', 7, xf, xf + s * 18) if s > 0 else
                    _cyl((0, CAM_YR, -CAM_D), 'x', 7, xf + s * 18, xf))
    emit('cameras', np.concatenate(cams), '#8e44ad', 0.6, 'OV9281 boards + lenses')

    # 15→5 V buck and two WAGO 221-413 in the +Y bay [MEASURE]
    emit('buck_wago', np.concatenate([_box(-22, 14, 31, 41, -60, -15), _box(17, 22, 31, 40, -78, -66),
                                      _box(23.5, 28.5, 31, 40, -78, -66)]), '#f39c12', 0.5, 'buck + 2 WAGO [MEASURE]')

    # Cables (schematic routes, see the route constants above)
    emit('cable_ribbons', np.concatenate([_strip(RIBBON_NEAR[:-1], 13, 0.3, [0, 1.0, 0]),
                                          _strip(RIBBON_NEAR[-2:], 13, 0.3, [1.0, 0, 0]),
                                          _strip(RIBBON_FAR[:-1], 13, 0.3, [0, 1.0, 0]),
                                          _strip(RIBBON_FAR[-2:], 13, 0.3, [1.0, 0, 0])]), '#e67e22', 1.0,
         'camera ribbons (15→22-pin, 150 mm)')
    emit('cable_lidar', _tube(LIDAR_CABLE, 3.5), '#1f4e79', 1.0, 'lidar cable: M12 → RJ45 (Pi Ethernet)')
    emit('cable_lidar_power', _tube(LIDAR_POWER, 2.0), '#8e1b1b', 1.0, 'lidar power pair (Livox splitter) → WAGO')
    emit('cable_battery', _tube(BATTERY_LEAD, 2.0), '#e74c3c', 1.0, '15 V battery lead (up the pole) → WAGO')
    emit('cable_buck_in', _tube(BUCK_IN, 1.5), '#f1948a', 1.0, 'WAGO → buck input')
    emit('cable_usbc', _tube(USBC_LEAD, 2.0), '#111111', 1.0, 'buck USB-C → Pi power (right-angle plug)')
    return out
