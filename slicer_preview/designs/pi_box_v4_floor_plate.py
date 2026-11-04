"""One blue plate (user 2026-11-03): v4 floor (with the glue-in lip grooves) + its three standing lips.

Lips (designs/pi_box_v4_floor.py LIPS, sketches/pi_box_v4_floor_lips_sketch.png option B) print FOOT DOWN: the table
face on the plate, tongue up, so the shoulders and tongue are clean top faces and nothing overhangs. Each lip uses the
floor's own print frame (rig Yr / X in the plate plane, height = toward the floor), only rotated about the vertical,
never mirrored: the notches sit unevenly along the rails, so a mirrored lip would not fit.
  body   LIP_W x run x LIP_H, R NOTCH_R notch (full height) round each floor screw head it would cover,
         EF_CH x 45° bevel round the plate face
  tongue TONGUE_W x TONGUE_L on top, one per groove segment (floor.groove_runs), TONGUE_END_GAP short at each end
Glue: CA or thin epoxy; the shoulders sit on the floor face and set the height.
Export: python to_goo.py pi_box_v4_floor_plate  (CPU: the floor's gyroid keep-mask is numpy-only)
"""
import numpy as np
from geometry import implicit_piece
from designs import pi_box_v4_floor as F

SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, F.EXPOSURE_S, F.BOTTOM_EXPOSURE_S
EF_CH = 0.5                          # 45° plate-face bevel vs elephant's foot (user 2026-11-03: 0.5 is fine)
HEAD_R = 5.7 / 2                      # M3 button head (ISO 7380)
LIP_X0, LIP_GAP = 52.0, 4.0           # plate x of the first lip; gap between lips (floor spans x +-46)
H_TOT = F.LIP_H + F.TONGUE_L


def _lip_fn(lip):
    ax, (lo, hi), c, (r0, r1) = lip
    runs = [(a, b) for a, b, cc, aa in F.groove_runs() if cc == c and aa == ax]
    notches = []
    for bx, by in F.FLOOR_BOSSES_V4:
        a, r = (bx, by) if ax == 'X' else (by, bx)
        if lo - HEAD_R < a < hi + HEAD_R and r0 - HEAD_R < r < r1 + HEAD_R:
            notches.append((a, r))

    def solid(A, h, R):
        """A: across (rig X for 'X' lips, Yr for 'Y'), R: along the run, h: height above the table (mm)."""
        e = max(EF_CH - h, 0.0)
        if h <= F.LIP_H:
            s = (A >= lo + e) & (A <= hi - e) & (R >= r0 + e) & (R <= r1 - e)
            for a, r in notches:
                s = s & ((A - a) ** 2 + (R - r) ** 2 > (F.NOTCH_R + e) ** 2)
            return s
        s = np.zeros(np.broadcast_shapes(np.shape(A), np.shape(R)), bool)
        if h <= H_TOT:
            for a, b in runs:
                s = s | ((np.abs(A - c) <= F.TONGUE_W / 2) & (R >= a + F.TONGUE_END_GAP) & (R <= b - F.TONGUE_END_GAP))
        return s
    return solid


def _lip_piece(lip, x0):
    """Piece on the plate at x0. Floor frame: plate x = Yr, plate z = X. 'X' rails (run along Yr) are turned 90°
    about the vertical so the run lies along plate z: Yr = r0 + z, X = hi - x (a rotation: det +1)."""
    ax, (lo, hi), c, (r0, r1) = lip
    f = _lip_fn(lip)
    W, L = hi - lo, r1 - r0
    if ax == 'X':
        fn = lambda x, h, z: f(hi - x, h, r0 + z)                    # A = X, R = Yr
        return implicit_piece(W, H_TOT, L, fn, x0=x0, z0=-L / 2), W
    fn = lambda x, h, z: f(lo + x, h, r0 + z)                        # A = Yr (= plate x), R = X (= plate z)
    return implicit_piece(W, H_TOT, L, fn, x0=x0, z0=-L / 2), W


PIECES = list(F.PIECES)
_x = LIP_X0
for _lip in F.LIPS:
    _p, _w = _lip_piece(_lip, _x)
    PIECES.append(_p)
    _x += _w + LIP_GAP
assert _x - LIP_GAP <= 211.7 / 2 - 2, 'lips run off the plate'
