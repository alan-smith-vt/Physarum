"""Lidar photo-rig cage, resin fit-check prototype v1 (aluminum is the real part).

Dimensions from "Lidar photo-rig handoff.md" (Oct 7, 2026) + its figs.
Checked 2026-10-11 against Livox mid-360-asm.stp (Mid-360, not the S — re-check
when the S drawing is available): M3 48 × 36 (48 along the connector axis),
body 65 × 65 (72.7 incl. connector), 60.1 tall. Dowel holes are NOT symmetric:
round Ø3 at Z = −16, slot Ø3 from Z = +22 to +24 (pin goes mid-slot, +23).
Still PROVISIONAL: anti-twist pin offset (fig 3, ~10 mm) — depends on the GoPro adapter.

Both plates print flat (holes vertical → full XY accuracy). Base plate's
countersinks open on the bottom (plate-side) face, as in the drawing.
Coordinates: X = connector direction (+X), Z = the other plate axis.

Resin-prototype deviations from the aluminum drawing:
  - 1/4-20 hub in the cross is a Ø5.2 tap-drill hole: tap it, or heat-set insert
  - holes print ~0.35 mm undersize (fit strip, 2026-10-11): clearance/tap holes get
    RESIN_HOLE_COMP added; dowel holes use the fit-strip value directly
  - 1 mm chamfer on bottom edges (elephant foot + scraper entry)
"""
import numpy as np
from geometry import implicit_piece

SUPPORTS = False
EXPOSURE_S = 2.4          # old Fast/Tenacious blend, 2026-10-11 exposure test
# v1 vat: old blend + 130 ml new (30 ml Tenacious + 100 ml Fast gray, ~23% Tenacious)
BOTTOM_EXPOSURE_S = 20.0  # template's 30 s welds big flat parts (Julia); ~79 cm² contact here

IN = 25.4
CSK_SLOPE = np.tan(np.radians(41))   # 82° countersink: radius change per mm of depth
CHAMFER = 1.0                        # also the scraper entry for plate removal
RESIN_HOLE_COMP = 0.35               # v1 fit strip: Ø3 pin slipped only in the 3.4 hole,
                                     # M3 thread slipped in 3.3/3.4, bit at 3.2, forced at 3.0/3.1

# ── Base plate: 1/4" 6061, 90 × 90 ──
BASE = 90.0
BASE_T = 0.25 * IN                  # 6.35
BASE_R = 5.0                        # corner radius
STANDOFF = 40.0                     # ±, 80 × 80 pattern
LIDAR_M3 = (24.0, 18.0)             # ±, 48 × 36 pattern (48 along X)
DOWEL_Z = (-16.0, 23.0)             # on the X = 0 centreline (STEP: round hole, slot centre)
M3_CLEAR, M3_HEAD = 3.4, 6.3        # Ø, flat-head countersunk from below
M4_CLEAR, M4_HEAD = 4.5, 8.4
DOWEL_D = 3.3                       # printed nominal: Ø3 pin press (3.4 = slip)
QTR_CLEAR, QTR_HEAD = 6.6, 13.0     # 1/4-20 flat-head from the top

# ── Cross top: 3/16" 6061 ──
CROSS_T = 3 / 16 * IN               # 4.76
ARM_W = 8.0
HUB_D = 30.0
PAD_R = 6.0
QTR_TAP = 5.2                       # 1/4-20 tap drill (resin)
PIN_D, PIN_Z = 3.3, 10.0            # anti-twist pin  [PROVISIONAL offset]

GAP = 6.0                           # between parts on the plate


def _rounded_rect(x, z, half, r):
    dx = np.maximum(np.abs(x) - (half - r), 0)
    dz = np.maximum(np.abs(z) - (half - r), 0)
    return dx ** 2 + dz ** 2 <= r ** 2


def _pin_hole(x, z, cx, cz, d):
    """Dowel hole: d is the printed nominal from the fit strip, no compensation."""
    return (x - cx) ** 2 + (z - cz) ** 2 <= (d / 2) ** 2


def _hole(x, z, cx, cz, d):
    """Clearance/tap hole: d is the size you want, compensated for resin shrink."""
    d = d + RESIN_HOLE_COMP
    return (x - cx) ** 2 + (z - cz) ** 2 <= (d / 2) ** 2


def _csk(x, z, cx, cz, clear_d, head_d, depth_from_face):
    """Countersunk hole; depth_from_face = distance from the face it opens on."""
    clear_d, head_d = clear_d + RESIN_HOLE_COMP, head_d + RESIN_HOLE_COMP
    r = max(clear_d / 2, head_d / 2 - CSK_SLOPE * depth_from_face)
    return (x - cx) ** 2 + (z - cz) ** 2 <= r ** 2


def _chamfered(x, z, y, outline, half):
    """Inset the outline on the bottom CHAMFER mm."""
    if y >= CHAMFER:
        return outline(x, z, half)
    return outline(x, z, half - (CHAMFER - y))


def base_plate(x, y, z):
    solid = _chamfered(x, z, y, lambda x, z, h: _rounded_rect(x, z, h, BASE_R), BASE / 2)
    for sx in (-1, 1):
        for sz in (-1, 1):
            solid &= ~_csk(x, z, sx * STANDOFF, sz * STANDOFF, M4_CLEAR, M4_HEAD, y)
            solid &= ~_csk(x, z, sx * LIDAR_M3[0], sz * LIDAR_M3[1], M3_CLEAR, M3_HEAD, y)
    for dz in DOWEL_Z:
        solid &= ~_pin_hole(x, z, 0, dz, DOWEL_D)
    solid &= ~_csk(x, z, 0, 0, QTR_CLEAR, QTR_HEAD, BASE_T - y)
    return solid


def cross_top(x, y, z):
    def outline(x, z, h):
        inset = BASE / 2 - h                              # chamfer inset (0 above CHAMFER)
        u, v = (x + z) / np.sqrt(2), (x - z) / np.sqrt(2)  # diagonal arm axes
        arm_half = ARM_W / 2 - inset
        arms = (np.abs(u) <= arm_half) | (np.abs(v) <= arm_half)
        arms &= (np.abs(x) <= STANDOFF) & (np.abs(z) <= STANDOFF)
        hub = x ** 2 + z ** 2 <= (HUB_D / 2 - inset) ** 2
        pads = np.zeros_like(arms)
        for sx in (-1, 1):
            for sz in (-1, 1):
                pads |= (x - sx * STANDOFF) ** 2 + (z - sz * STANDOFF) ** 2 <= (PAD_R - inset) ** 2
        return arms | hub | pads

    solid = _chamfered(x, z, y, outline, BASE / 2)
    for sx in (-1, 1):
        for sz in (-1, 1):
            solid &= ~_hole(x, z, sx * STANDOFF, sz * STANDOFF, M4_CLEAR)
    solid &= ~_hole(x, z, 0, 0, QTR_TAP)
    solid &= ~_pin_hole(x, z, 0, -PIN_Z, PIN_D)
    return solid


def _centered(fn, size):
    """Adapt a centred solid fn to implicit_piece's corner-origin coords."""
    return lambda x, y, z: fn(x - size / 2, y, z - size / 2)


CROSS_SPAN = 2 * (STANDOFF + PAD_R)   # 92
BASE_X = -(BASE / 2 + GAP / 2)
CROSS_X = CROSS_SPAN / 2 + GAP / 2

# ── Fit strip: which dowel hole gives the press/slip fit you want ──
FIT_DOWEL_D = [3.0, 3.1, 3.2, 3.3, 3.4]   # left → right
FIT_W, FIT_D, FIT_T = 40.0, 8.0, 3.0


def fit_strip(x, y, z):
    solid = np.ones(np.broadcast_shapes(x.shape, z.shape), dtype=bool)
    for i, d in enumerate(FIT_DOWEL_D):
        solid &= ~_pin_hole(x, z, (i - 2) * 7.5, 0, d)
    return solid


PIECES = [
    implicit_piece(BASE, BASE_T, BASE, _centered(base_plate, BASE),
                   x0=BASE_X - BASE / 2, z0=-BASE / 2),
    implicit_piece(CROSS_SPAN, CROSS_T, CROSS_SPAN, _centered(cross_top, CROSS_SPAN),
                   x0=CROSS_X - CROSS_SPAN / 2, z0=-CROSS_SPAN / 2),
    implicit_piece(FIT_W, FIT_T, FIT_D, lambda x, y, z: fit_strip(x - FIT_W / 2, y, z - FIT_D / 2),
                   x0=-FIT_W / 2, z0=BASE / 2 + GAP),
]
