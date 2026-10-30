"""v4 PROPOSAL, UNAPPROVED (2026-10-29, v4-floor): sliding battery door for designs/pi_box_v4_core.py. Do not print.

Option A of sketches/pi_box_v4_floor_options_sketch.py, slide direction changed to +Y (board home:1999): the door
lives inside the walls under the bank (Yr 30.5..62, d 176..180). Its two edge tongues ride in closed grooves in the
±X wall inner faces (core DOOR_RAIL, d 175.6..178.4, 1.6 deep); the 1.6 mm wall ledge under each groove carries the
bank in shear. It slides in from +Y (under the back panel) until it butts the battery rib, and an off-the-shelf
M8 pull-ring index plunger [VERIFY: M8x1.25 body, Ø5 pin, ~6 throw, 13 AF nut] in a boss under the door shoots its
pin up into the core's PLUNGER hole at (-42, 44). Pull the ring, slide the door out by its grip lip.
The door's top face also closes the bottom of the back-panel channels (Yr 56..60.5), so it holds the panel in.
Fixed floor (stick hole, collar, UE300 cradle, FLOOR_BOSSES_V4, R30 front) not modelled yet.

Rig -> print (top face down): (X, Yr, d) -> (x = X + HX, y = d - D_TOP, z = Yr - Y_LO).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import X_IN, Y_MAX, DEPTH, PORT_WINDOW, RESIN_HOLE_COMP, hole

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

OVERSIZE = RESIN_HOLE_COMP / 2           # external faces print ~0.35 oversize (diameter): per face
CLEAR = 0.2                              # per face, on top of OVERSIZE
RIB_Y = 30.5                             # battery rib +Y face: the door's stop
GROOVE_D, GROOVE_DEPTH = (175.6, 178.4), 1.6   # core DOOR_RAIL (closed groove; ledge below to DEPTH)
D_TOP = 176.0                            # door top face (bank side); bottom flush with the wall ends at DEPTH
HX = X_IN - CLEAR - OVERSIZE             # body half-width between the walls
TONGUE_X = X_IN + GROOVE_DEPTH - CLEAR - OVERSIZE
TONGUE_D = (GROOVE_D[0] + 0.4, GROOVE_D[1] - 0.4)
TONGUE_LEAD = 2.0                        # tongue tip (-Y, the leading end) tapers in over 2 mm
Y_LO = RIB_Y + CLEAR + OVERSIZE
GRIP = dict(x=25.0, y0=Y_MAX - 4.0, d1=DEPTH + 6.0)     # finger lip under the +Y end, |X| <= x
PLUNGER_XY = (-42.0, 44.0)               # core PLUNGER hole
NUT_AF, NUT_H, M8_CLEAR = 13.0, 6.5, 8.5
BOSS_R, BOSS_D1 = 9.0, DEPTH + 8.0       # boss under the door: plunger body through, nut pocket from below


def door(x, y, z):
    X, d, Yr = x - TONGUE_X, y + D_TOP, z + Y_LO
    aX = np.abs(X)
    s = (aX <= HX) & (d <= DEPTH) & (Yr <= Y_MAX)
    if TONGUE_D[0] <= d <= TONGUE_D[1]:
        lead = np.maximum(TONGUE_LEAD - (Yr - Y_LO), 0.0)
        s |= (aX <= TONGUE_X - lead) & (Yr <= Y_MAX)
    if d > DEPTH:                         # below the box: grip lip + plunger boss
        s = (aX <= GRIP['x']) & (Yr >= GRIP['y0']) & (Yr <= Y_MAX) & (d <= GRIP['d1'])
        s |= hole(*PLUNGER_XY, 2 * BOSS_R - RESIN_HOLE_COMP, X, Yr) & (d <= BOSS_D1)
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    s &= ~((X >= wx0) & (X <= wx1) & (Yr >= wy0) & (Yr <= wy1))
    s &= ~hole(*PLUNGER_XY, M8_CLEAR, X, Yr)
    if d >= BOSS_D1 - NUT_H:                                     # hex nut pocket from below
        px, py = X - PLUNGER_XY[0], Yr - PLUNGER_XY[1]
        r = NUT_AF / 2 + RESIN_HOLE_COMP / 2
        hexin = np.ones_like(s)
        for a in (0.0, np.pi / 3, 2 * np.pi / 3):
            hexin &= np.abs(px * np.cos(a) + py * np.sin(a)) <= r
        s &= ~hexin
    return s


PIECES = [implicit_piece(2 * TONGUE_X, BOSS_D1 - D_TOP, Y_MAX - Y_LO, door, x0=-TONGUE_X, z0=0.0)]
