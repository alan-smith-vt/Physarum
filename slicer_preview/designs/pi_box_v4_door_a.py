"""v4 battery door, APPROVED by the user 2026-10-29 (single-piece sliding door, one M3 screw): for
designs/pi_box_v4_core.py. Print the fit strip (pi_box_v4_door_fit.py) first. Lock screw for now: a stock M3 x 8
socket cap driven with a hex key (plate 3 + gap 0.2 + insert 4: M3 x 10 would bottom in the 6 deep hole); the
printed thumb knob comes later.

Option A of sketches/pi_box_v4_floor_options_sketch.py, slide direction changed to +Y (board home:1999): the door
plate sits UNDER the wall ends (d 180.2..183.2, so the bank keeps the v3 height under the core's seat at d 22). Along
each ±X wall a thin web stands up inside the wall and turns out into a hook in the wall's closed groove (core
DOOR_RAIL, d 175.6..178.4, 1.6 deep): the door hangs from the 1.6 mm wall ledges under the grooves, which carry the
bank in shear. It slides in from +Y (under the back panel) until the webs butt the battery rib, and is locked by a
screw (user, 2026-10-29): an M3 socket cap (later an M3 x 12 with the printed knob, designs/pi_box_v4_thumb_knob.py,
CA-glued on its head) goes up through the plate into an M3 brass insert (Ø4 x 4, glued) in the core's
lock boss at SCREW_XY. The plate's hole is a clearance hole (SCREW_D, 2026-11-03): the screw slides through (no longer captive). The screw only stops sliding; the bank load stays in the rails. Undo, slide out by the lip.
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
CLEAR = 0.1                              # per face, on top of OVERSIZE (0.2 until 2026-11-03: after the mirror fix +
#   new FEP the printed door was wiggly on the opened-up core v4 grooves; user: tighten a little)
RIB_Y = 30.5                             # battery rib +Y face: the webs' stop
GROOVE_D, GROOVE_DEPTH = (175.6, 178.4), 1.6   # core DOOR_RAIL (closed groove; 1.6 ledge below to DEPTH)
PLATE_D = (DEPTH + CLEAR, DEPTH + CLEAR + 3.0)  # plate under the wall ends; the bank stands on its top
PLATE_X = X_OUT - OVERSIZE               # flush with the walls' outer faces
WEB_T = 1.4                              # web inside each wall: tension ~2 MPa at 50 g
HX = X_IN - CLEAR - OVERSIZE             # web outer face
TONGUE_X = X_IN + GROOVE_DEPTH - CLEAR - OVERSIZE
# tongue height per side (2026-11-03): inset 0.2 from the modelled groove ends (was 0.4); the core opens its groove by
# comp/2 per face, +DOOR_EXTRA on +X, so the +X tongue grows by that too: up/down slop 0.75 total both sides.
CORE_DOOR_EXTRA = {1: 0.15, -1: 0.0}     # = pi_box_v4_core.DOOR_EXTRA
TONGUE_D = {sd: (GROOVE_D[0] + 0.2 - e, GROOVE_D[1] - 0.2 + e) for sd, e in CORE_DOOR_EXTRA.items()}
D_TOP = min(t[0] for t in TONGUE_D.values())
TONGUE_LEAD = 2.0                        # hook tip (-Y, the leading end) tapers in over 2 mm
Y_LO = RIB_Y + CLEAR + OVERSIZE
# standing lip under the +Y end (2026-11-03, sketches/pi_box_v4_floor_lips_sketch.png option B rear bar): full width,
# down to the same table plane as the floor's glue-in lips (floor face d 183.0 + LIP_H 7.7), 1.1 below the M3 thumb
# screw (6.5 proud of the plate). Was |X| <= 25, PLATE_D[1] + 5.0.
STAND_D = DEPTH + 3.0 + 7.7             # = pi_box_v4_floor: floor face + LIP_H
GRIP = dict(x=46.0, y0=Y_MAX - 4.0, d1=STAND_D)
# MIRROR (2026-11-01, see to_goo MIRROR_FIX): door A was printed before the fix, i.e. as the X-mirror of this design
# with the screw at -41.1. Physically its screw sits at X +41.1, so the design now says so (lock boss moved to +X in
# the core); the door is otherwise symmetric in X. Reprints from here match the printed door.
SCREW_XY = (41.1, 44.0)                  # core lock boss X 38..44.2: Ø4 insert hole centred, ~0.9 walls
SCREW_D = 3.2                            # M3 slides through (2026-11-03; was CAPTIVE_D 2.6, thread cut through once)
KNOB_H = 6.0                             # printed knob under the plate (pi_box_v4_thumb_knob, later)


# Rig name engraved in the outside face, -X strip beside the port window (opposite the thumb screw), running along
# Yr (user 2026-11-03; names: RIG_NAMES.md). NAME_FLIP mirrors it if it reads backwards from below.
NAME = dict(text='Alnilam', font=r'C:\Windows\Fonts\OLDENGL.TTF', depth=0.5, x=(-48.0, -34.0), y=(33.0, 56.0))
#   font shortlist (sketches/pi_box_v4_door_name_final_sketch.png): Old English (leading), Monotype Corsiva, Algerian
NAME_FLIP = True                          # 2026-11-03: False read mirrored from outside (viewer check)
_NAME_P = 0.05                            # raster pitch (mm)


def make_name_mask(font=None, text=None):
    """Mask function (X, Yr) -> True where the name is cut; array-safe under numpy or cupy (this module's np).
    Returns (fn, cap height mm, length mm)."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    from designs.print_post import text_inside
    import numpy as _n
    tp = TextPath((0, 0), text or NAME['text'], size=10.0, prop=FontProperties(fname=font or NAME['font']))
    v = tp.vertices
    u0, u1, v0, v1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
    (x0, x1), (y0, y1) = NAME['x'], NAME['y']
    k = min((y1 - y0) / (u1 - u0), (x1 - x0) / (v1 - v0))           # mm per text unit, fit the box
    XX, YY = _n.meshgrid(_n.arange(x0, x1, _NAME_P), _n.arange(y0, y1, _NAME_P), indexing='ij')
    u = (YY - (y0 + y1) / 2) / k * (-1 if NAME_FLIP else 1) + (u0 + u1) / 2     # text runs along Yr
    w = ((x0 + x1) / 2 - XX) / k + (v0 + v1) / 2                                # letter tops toward -X (outer edge)
    M = text_inside(tp, _n.c_[u.ravel(), w.ravel()]).reshape(XX.shape)

    def fn(X, Yr):
        m = np.asarray(M)
        i = np.floor((X - x0) / _NAME_P).astype(int)
        j = np.floor((Yr - y0) / _NAME_P).astype(int)
        ok = (i >= 0) & (i < m.shape[0]) & (j >= 0) & (j < m.shape[1])
        return ok & m[np.clip(i, 0, m.shape[0] - 1), np.clip(j, 0, m.shape[1] - 1)]
    return fn, (v1 - v0) * k, (u1 - u0) * k


name_mask, NAME_CAP, NAME_LEN = make_name_mask()


def door(x, y, z, name_fn=None):
    X, d, Yr = x - PLATE_X, y + D_TOP, z + Y_LO
    aX = np.abs(X)
    inY = (Yr <= Y_MAX) & (Yr >= Y_LO) & (d >= D_TOP)
    if d < PLATE_D[0]:                    # inside the walls: webs + hooks
        s = (aX >= HX - WEB_T) & (aX <= HX) & inY
        lead = np.maximum(TONGUE_LEAD - (Yr - Y_LO), 0.0)
        for sd, (t0, t1) in TONGUE_D.items():
            if t0 <= d <= t1:
                s |= (sd * X >= HX - WEB_T) & (sd * X <= TONGUE_X - lead) & inY
    elif d <= PLATE_D[1]:
        s = (aX <= PLATE_X) & inY
    else:                                 # below: grip lip
        s = (aX <= GRIP['x']) & (Yr >= GRIP['y0']) & inY & (d <= GRIP['d1'])
    wx0, wx1, wy0, wy1 = PORT_WINDOW
    s &= ~((X >= wx0) & (X <= wx1) & (Yr >= wy0) & (Yr <= wy1))
    s &= ~hole(*SCREW_XY, SCREW_D, X, Yr)
    if PLATE_D[1] - NAME['depth'] < d <= PLATE_D[1]:
        s &= ~(name_fn or name_mask)(X, Yr)
    return s


PIECES = [implicit_piece(2 * PLATE_X, GRIP['d1'] - D_TOP, Y_MAX - Y_LO, door, x0=-PLATE_X, z0=0.0)]
