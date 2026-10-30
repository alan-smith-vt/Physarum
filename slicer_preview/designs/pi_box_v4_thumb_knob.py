"""v4 PROPOSAL, UNAPPROVED (2026-10-29, v4-floor): thumb knob for the battery door's lock screw. Do not print yet.

A stock M3 x 12 socket cap screw (head Ø5.5 x 3 [MEASURE the one used]) is pushed head-first into the pocket from the
knob's outer face and CA-glued; its thread leaves through the top face, goes up through the door's captive hole
and into the M3 insert in the core's lock boss (designs/pi_box_v4_door_a.py). Knurled Ø14 x 6. Print two (spare).
Print: top face (door side) down, pocket opening up.
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import RESIN_HOLE_COMP, hole

SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.3, 20.0
KNOB_D, KNOB_H = 14.0, 6.0
KNURL_N, KNURL_DEPTH = 16, 0.8           # grip flutes round the rim
HEAD_D, HEAD_H = 5.5 + 0.2, 3.0 + 0.3    # socket-cap head + glue gap (hole() adds the resin comp)
THREAD_D = 3.2
CHAMFER = 0.6
OVERSIZE = RESIN_HOLE_COMP / 2
R = KNOB_D / 2 + 1.0


def knob(x, y, z):
    X, Y = x - R, z - R
    r, a = np.hypot(X, Y), np.arctan2(Y, X)
    edge = min(y, KNOB_H - y)
    rim = KNOB_D / 2 - OVERSIZE - KNURL_DEPTH * (0.5 + 0.5 * np.cos(KNURL_N * a)) - max(CHAMFER - edge, 0.0)
    s = (r <= rim) & (y <= KNOB_H)
    s &= ~hole(0, 0, THREAD_D, X, Y)
    if y >= KNOB_H - HEAD_H:
        s &= ~hole(0, 0, HEAD_D, X, Y)
    return s


PIECES = [implicit_piece(2 * R, KNOB_H, 2 * R, knob, x0=-2 * R - 1, z0=-R),
          implicit_piece(2 * R, KNOB_H, 2 * R, knob, x0=1, z0=-R)]
