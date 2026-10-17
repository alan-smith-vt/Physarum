"""v3 back (+Y, battery-side) panel (2026-10-17, sketches/pi_box_v3_back_panel_sketch.py): L-hook edges as the
front panel (pi_box_bat_panel_v3.panel_mask), but an open frame instead of the gyroid: a border, a solid disc
under the phone with a MagSafe ring recess in its OUTER face, 4 mm braces from the disc to the frame, and a braced
window over the bank's display + button. The phone (portrait, centred) covers the window: it comes off to read the
battery. Display position is [MEASURE]: WIN is a placeholder round the bank's button (BUTTON, X -22, d 158).

Prints flat, outer face down (the recess is a shallow pocket on the plate side), hooks up.
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import X_IN, Y_MAX, PANEL_LIP, PANEL_T, RESIN_HOLE_COMP, BUTTON
from designs import pi_box_bat_panel_v3 as PV

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

FRAME = 5.0                                     # border inside the wall line (|X| >= X_IN - FRAME, top, bottom)
BRACE = 4.0
RING_OD, RING_ID, RING_DEPTH = 56.0, 46.0, 0.8  # MagSafe ring (sticker / plate) recess [MEASURE] ring + phone
RING_C = (0.0, 88.0)                            # rig X, d (phone centred, portrait)
DISC_R = RING_OD / 2 + 1.5                      # solid disc carrying the recess
WIN = (-40.0, 12.0, 142.0, PV.D1 - FRAME)       # display + button window: X0, X1, d0, d1 [MEASURE display]
IN_X = X_IN - FRAME                             # inner edge of the side frame


def _seg(X, d, a, b, w=BRACE):
    (ax, ad), (bx, bd) = a, b
    vx, vd = bx - ax, bd - ad
    t = np.clip(((X - ax) * vx + (d - ad) * vd) / (vx * vx + vd * vd), 0.0, 1.0)
    return np.hypot(X - ax - t * vx, d - ad - t * vd) <= w / 2


def back_mask(X, y, d):
    """X, d rig; y off the outer face toward the interior (-Y)."""
    s = PV.panel_mask(X, y, d)
    cx, cd = RING_C
    r = np.hypot(X - cx, d - cd)
    keep = (np.abs(X) >= IN_X) | (d <= PV.D0 + FRAME) | (d >= PV.D1 - FRAME) | (r <= DISC_R)
    top, bot = PV.D0 + FRAME, WIN[2] - BRACE / 2
    for a, b in [((0.0, cd - DISC_R + 1), (0.0, top)),                 # up to the top band
                 ((-DISC_R + 1, cd), (-IN_X, cd)), ((DISC_R - 1, cd), (IN_X, cd)),   # out to the sides
                 ((-0.7 * DISC_R, cd - 0.7 * DISC_R), (-IN_X, top)), ((0.7 * DISC_R, cd - 0.7 * DISC_R), (IN_X, top)),
                 ((-0.7 * DISC_R, cd + 0.7 * DISC_R), (-IN_X, bot)), ((0.7 * DISC_R, cd + 0.7 * DISC_R), (IN_X, bot)),
                 ((-IN_X, bot), (IN_X, bot)),                           # window head
                 ((WIN[1] + BRACE / 2, bot), (WIN[1] + BRACE / 2, PV.D1 - FRAME))]:   # window side
        keep |= _seg(X, d, a, b)
    s &= keep
    if y < RING_DEPTH:                                                  # MagSafe recess in the outer face
        s &= ~((r >= RING_ID / 2) & (r <= RING_OD / 2 + RESIN_HOLE_COMP / 2))
    return s


_FACE = Y_MAX - PANEL_LIP - PV.CL               # outer face Yr; the interior is toward -Y
PIECES = [implicit_piece(PV.D1 - PV.D0, PANEL_T + PV.HOOK, 2 * PV.HALF_W,
                         lambda x, y, z: back_mask(z - PV.HALF_W, y, x + PV.D0),
                         x0=-(PV.D1 - PV.D0) / 2, z0=-PV.HALF_W)]
