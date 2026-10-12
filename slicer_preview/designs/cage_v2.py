"""Lidar photo-rig cage, resin fit-check prototype v2 (aluminum is the real part).

v2 vs v1: resin hole compensation (v1 fit strip), fit strip dropped, plus four
mock standoffs — swoopy-L corner posts, thicker than the real Ø7 aluminum ones
because thin resin posts would flex. Standoffs print lying on one flange
(18 mm tall instead of 65). Locating pins become printed nubs on the base top
(the aluminum plate gets pressed dowels in reamed holes instead).

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
# vat: old blend + 130 ml new (30 Tenacious + 100 Fast gray) before v1, + 100 ml (25 + 75) before v2
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
M3_CLEAR, M4_CLEAR = 3.4, 4.5
# Counterbores from below (resin mock uses the screws on hand, not flat-heads):
#   Livox M3: 6 mm shaft, Ø5 × 3 cap head → 3.9 deep leaves 2.45 of plate,
#             3.55 mm into the lidar's 5 mm holes (handoff: ≤ ~4)
#   M4 × 35 mushroom head Ø9: NO counterbore — at 5 mm from the R5 corner a Ø9.6
#             bore breaks out of the plate. Heads sit proud underneath (only the
#             centre mount touches there). 28.65 into the standoff + 30.2 = 58.9 < 65
M3_CB_D, M3_CB_DEPTH = 5.6, 3.9
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


def _cbore(x, z, cx, cz, clear_d, cb_d, cb_depth, y):
    """Counterbored from the bottom (plate-side) face."""
    return _hole(x, z, cx, cz, cb_d if y < cb_depth else clear_d)


def base_plate(x, y, z):
    solid = _chamfered(x, z, y, lambda x, z, h: _rounded_rect(x, z, h, BASE_R), BASE / 2)
    for sx in (-1, 1):
        for sz in (-1, 1):
            solid &= ~_hole(x, z, sx * STANDOFF, sz * STANDOFF, M4_CLEAR)
            solid &= ~_cbore(x, z, sx * LIDAR_M3[0], sz * LIDAR_M3[1], M3_CLEAR, M3_CB_D, M3_CB_DEPTH, y)
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

# ── Locating nubs on the base top (into the lidar's Ø3 round hole and slot) ──
NUB_D = 2.65              # external features print ~0.35 oversize (mirror of the hole shrink)
NUB_H = 2.0               # lidar hole depth unknown — keep it short
NUB_CHAMFER = 0.5


def nubs(x, y, z):
    r = NUB_D / 2 - max(y - (NUB_H - NUB_CHAMFER), 0)
    solid = np.zeros(np.broadcast_shapes(x.shape, z.shape), dtype=bool)
    for dz in DOWEL_Z:
        solid |= x ** 2 + (z - dz) ** 2 <= r ** 2
    return solid


# ── Mock standoffs, printed lying down ──
# Cross-section (a across, b up), outer corner of the L at (0, 0):
#   flat flange b ∈ [0, FLANGE_T], upright flange a ∈ [0, FLANGE_T],
#   Ø10 core around the screw axis at (5, 5), flanges flush with its outer faces.
# Along the length u, flanges reach FLANGE_END from the axis at the ends, FLANGE_MID mid-way.
STANDOFF_LEN = 65.0        # handoff: lidar 60.1 tall + ~5 mm over the dome
CORE_D = 10.0
FLANGE_T = 3.0
FLANGE_END, FLANGE_MID = 13.0, 6.0
AX = CORE_D / 2
SO_W = AX + FLANGE_END     # 18: footprint width and print height
M4_PILOT, PILOT_DEPTH = 3.4, 10.0   # self-tap only the last 10 mm; clearance beyond


def standoff(u, b, a):
    t = u / STANDOFF_LEN
    reach = AX + FLANGE_MID + (FLANGE_END - FLANGE_MID) * np.abs(2 * t - 1) ** 3
    inset = max(CHAMFER - b, 0)
    core = (a - AX) ** 2 + (b - AX) ** 2 <= AX ** 2
    corner = (a <= AX) & (b <= AX)
    flat = (b <= FLANGE_T) & (a <= reach)
    upright = (a <= FLANGE_T) & (b <= reach)
    solid = core | corner | flat | upright
    solid &= (a >= inset) & (a <= reach - inset) & (u >= inset) & (u <= STANDOFF_LEN - inset)
    end = (u < PILOT_DEPTH) | (u > STANDOFF_LEN - PILOT_DEPTH)
    bore = np.where(end, M4_PILOT, M4_CLEAR) + RESIN_HOLE_COMP
    solid &= (a - AX) ** 2 + (b - AX) ** 2 > (bore / 2) ** 2
    return solid


# ── Plate layout. Printable image is 211.7 × 118.4 mm (15120 × 14 µm, 6230 × 19 µm),
#    not the 218.9 × 128.9 marketing size. Plates in a row, 4th standoff along Z at the
#    right end, 3 standoffs in a row along +Z. 206 × 113 mm, ~2.8 mm edge margin. ──
LAY_GAP = 3.0
TOTAL_X = BASE + LAY_GAP + CROSS_SPAN + LAY_GAP + SO_W
TOTAL_Z = CROSS_SPAN + LAY_GAP + SO_W
X0, Z0 = -TOTAL_X / 2, -TOTAL_Z / 2
CROSS_X0 = X0 + BASE + LAY_GAP
PLATE_Z0 = Z0 + (CROSS_SPAN - BASE) / 2          # base centred on the cross
ROW_Z = Z0 + CROSS_SPAN + LAY_GAP
ROW_X0 = -(3 * STANDOFF_LEN + 2 * LAY_GAP) / 2

PIECES = [
    implicit_piece(BASE, BASE_T, BASE, _centered(base_plate, BASE), x0=X0, z0=PLATE_Z0),
    implicit_piece(BASE, NUB_H, BASE, _centered(nubs, BASE), x0=X0, y0=BASE_T, z0=PLATE_Z0),
    implicit_piece(CROSS_SPAN, CROSS_T, CROSS_SPAN, _centered(cross_top, CROSS_SPAN),
                   x0=CROSS_X0, z0=Z0),
]
for i in range(3):
    PIECES.append(implicit_piece(STANDOFF_LEN, SO_W, SO_W, standoff,
                                 x0=ROW_X0 + i * (STANDOFF_LEN + LAY_GAP), z0=ROW_Z))
PIECES.append(implicit_piece(SO_W, SO_W, STANDOFF_LEN, lambda x, y, z: standoff(z, y, x),
                             x0=CROSS_X0 + CROSS_SPAN + LAY_GAP, z0=Z0))
