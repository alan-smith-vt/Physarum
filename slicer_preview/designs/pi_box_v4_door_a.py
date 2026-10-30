"""v4 PROPOSAL, UNAPPROVED (2026-10-29, v4-floor): sliding battery door for designs/pi_box_v4_core.py. Do not print.

Option A of sketches/pi_box_v4_floor_options_sketch.py, slide direction changed to +Y (board home:1999): the door
plate sits UNDER the wall ends (d 180.2..183.2, so the bank keeps the v3 height under the core's seat at d 22). Along
each ±X wall a thin web stands up inside the wall and turns out into a hook in the wall's closed groove (core
DOOR_RAIL, d 175.6..178.4, 1.6 deep): the door hangs from the 1.6 mm wall ledges under the grooves, which carry the
bank in shear. It slides in from +Y (under the back panel) until the webs butt the battery rib, and an off-the-shelf
M8 pull-ring index plunger [VERIFY: M8x1.25 body, Ø5 pin, ~6 throw, 13 AF nut] in a boss under the plate shoots its
pin up into the core's PLUNGER hole at (-42, 44) (boss bottom at d 180). Pull the ring, slide the door out by its lip.
The plate also closes the bottom of the back-panel channels (Yr 56..60.5), so it holds the panel in.
Fixed floor: designs/pi_box_v4_floor.py. Fit strip: designs/pi_box_v4_door_fit.py.

Rig -> print (web tops down): (X, Yr, d) -> (x = X + PLATE_X, y = d - D_TOP, z = Yr - Y_LO).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import X_IN, X_OUT, Y_MAX, DEPTH, PORT_WINDOW, RESIN_HOLE_COMP, hole

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

OVERSIZE = RESIN_HOLE_COMP / 2           # external faces print ~0.35 oversize (diameter): per face
CLEAR = 0.2                              # per face, on top of OVERSIZE
RIB_Y = 30.5                             # battery rib +Y face: the webs' stop
GROOVE_D, GROOVE_DEPTH = (175.6, 178.4), 1.6   # core DOOR_RAIL (closed groove; 1.6 ledge below to DEPTH)
PLATE_D = (DEPTH + CLEAR, DEPTH + CLEAR + 3.0)  # plate under the wall ends; the bank stands on its top
PLATE_X = X_OUT - OVERSIZE               # flush with the walls' outer faces
WEB_T = 1.4                              # web inside each wall: tension ~2 MPa at 50 g
HX = X_IN - CLEAR - OVERSIZE             # web outer face
TONGUE_X = X_IN + GROOVE_DEPTH - CLEAR - OVERSIZE
TONGUE_D = (GROOVE_D[0] + 0.4, GROOVE_D[1] - 0.4)
D_TOP = TONGUE_D[0]
TONGUE_LEAD = 2.0                        # hook tip (-Y, the leading end) tapers in over 2 mm
Y_LO = RIB_Y + CLEAR + OVERSIZE
GRIP = dict(x=25.0, y0=Y_MAX - 4.0, d1=PLATE_D[1] + 5.0)   # finger lip under the +Y end, |X| <= x
PLUNGER_XY = (-42.0, 44.0)               # core PLUNGER hole
NUT_AF, NUT_H, M8_CLEAR = 13.0, 6.5, 8.5
BOSS_R, BOSS_D1 = 9.0, PLATE_D[1] + 8.0  # boss under the plate: plunger body through, nut pocket from below


def door(x, y, z):
    X, d, Yr = x - PLATE_X, y + D_TOP, z + Y_LO
    aX = np.abs(X)
    inY = (Yr <= Y_MAX) & (Yr >= Y_LO) & (d >= D_TOP)
    if d < PLATE_D[0]:                    # inside the walls: webs + hooks
        s = (aX >= HX - WEB_T) & (aX <= HX) & inY
        if TONGUE_D[0] <= d <= TONGUE_D[1]:
            lead = np.maximum(TONGUE_LEAD - (Yr - Y_LO), 0.0)
            s |= (aX >= HX - WEB_T) & (aX <= TONGUE_X - lead) & inY
    elif d <= PLATE_D[1]:
        s = (aX <= PLATE_X) & inY
    else:                                 # below: grip lip + plunger boss
        s = (aX <= GRIP['x']) & (Yr >= GRIP['y0']) & inY & (d <= GRIP['d1'])
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


PIECES = [implicit_piece(2 * PLATE_X, BOSS_D1 - D_TOP, Y_MAX - Y_LO, door, x0=-PLATE_X, z0=0.0)]
