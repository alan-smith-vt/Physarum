"""v4 PROPOSAL, UNAPPROVED (2026-10-29, v4-floor): sliding battery door for designs/pi_box_v4_core.py. Do not print.

Option A of sketches/pi_box_v4_floor_options_sketch.py, slide direction changed to +Y (board home:1999): the door
plate sits UNDER the wall ends (d 180.2..183.2, so the bank keeps the v3 height under the core's seat at d 22). Along
each ±X wall a thin web stands up inside the wall and turns out into a hook in the wall's closed groove (core
DOOR_RAIL, d 175.6..178.4, 1.6 deep): the door hangs from the 1.6 mm wall ledges under the grooves, which carry the
bank in shear. It slides in from +Y (under the back panel) until the webs butt the battery rib, and is locked by a
thumb screw (user, 2026-10-29): a stock M3 x 12 socket cap with a printed knurled knob CA-glued on its head
(designs/pi_box_v4_thumb_knob.py) goes up through the plate into an M3 brass insert (Ø4 x 4, glued) in the core's
lock boss at SCREW_XY. The plate's hole is undersize (CAPTIVE_D): the thread is screwed through it once, so the screw
cannot drop out when undone. The screw only stops sliding; the bank load stays in the rails. Undo, slide out by the lip.
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
SCREW_XY = (-41.1, 44.0)                 # core lock boss X -44.2..-38: Ø4 insert hole centred, ~0.9 walls
CAPTIVE_D = 2.6                          # M3 thread cuts its way through once (hole() adds the resin comp)
KNOB_H = 6.0                             # printed knob under the plate (pi_box_v4_thumb_knob)


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
    else:                                 # below: grip lip
        s = (aX <= GRIP['x']) & (Yr >= GRIP['y0']) & inY & (d <= GRIP['d1'])
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    s &= ~((X >= wx0) & (X <= wx1) & (Yr >= wy0) & (Yr <= wy1))
    s &= ~hole(*SCREW_XY, CAPTIVE_D, X, Yr)
    return s


PIECES = [implicit_piece(2 * PLATE_X, GRIP['d1'] - D_TOP, Y_MAX - Y_LO, door, x0=-PLATE_X, z0=0.0)]
