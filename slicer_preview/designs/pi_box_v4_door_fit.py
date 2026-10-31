"""Fit strip for the v4 battery door (door APPROVED 2026-10-29): print this first, before the core and door.

Print this before the core/door: it tests the door's rail clearances and the thumb-screw lock at the real sizes.
  1. WALL: a -X wall stub (X -50..-46, Yr 28.5..52, d 165..180) with the battery rib stub, the closed door groove
     (core DOOR_RAIL: d 175.6..178.4, 1.6 deep, +comp) with its 45° mouth flare at +Y, the 1.8 web clearance, and
     the lock boss (X -46..-38, Yr 38..50, d 168..180) with its M3 insert hole Ø4 × 6 (+comp) at SCREW_XY.
  2. DOOR: the -X corner of designs/pi_box_v4_door_a.py (X <= -30, Yr <= 52): plate, web + hook, captive M3 hole.
Check: the door corner slides in from +Y with no force and no rattle, butts the rib, and an M3 x 8 socket cap
runs through the captive hole into the insert with a hex key. Clearances: 0.4 per side + RESIN_HOLE_COMP/2 on holes; external faces are pulled in by
OVERSIZE (= EXT_COMP/2 of ir_filter_clip).
Rig -> print: wall top down, y = d - 165; door corner standing on its grip end (y = Y_MAX - Yr).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import RESIN_HOLE_COMP, hole
from designs import pi_box_v4_door_a as D
from designs import pi_box_bat_rig as R

SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.3, 20.0
C = RESIN_HOLE_COMP / 2
WALL = dict(x0=-50.0, x1=-46.0, y0=28.5, y1=52.0, d0=165.0, d1=180.0)
RIB = (28.5, 30.5)
BOSS = dict(x0=-46.0, x1=-38.0, y0=38.0, y1=50.0, d0=168.0)
PIN = dict(x=D.SCREW_XY[0], y=D.SCREW_XY[1], dia=4.0, L=6.0)   # M3 insert Ø4 x 4 + 2 for the screw tip
FLARE = 1.0
WEB_CLEAR = (1.8, 175.5)                         # core: 1.8 clear inside the wall face below d 175.5

# the lock boss hangs 8 mm off the wall from d 168 (3 mm up in print): snap-off spikes from the plate under it
SPIKES = [(x, y) for x in (BOSS['x0'] + 4.0, BOSS['x1'] - 1.0) for y in (BOSS['y0'] + 1.5, BOSS['y1'] - 1.5)]   # 2 x 2 (user: 4 x 5 was overkill)
SPIKE_R, SPIKE_DIG = (0.6, 0.25), 0.25          # base radius on the plate -> tip radius, tip sunk into the boss

def wall(x, y, z):
    X, d, Yr = x + WALL['x0'], y + WALL['d0'], z + WALL['y0']
    s = (X <= WALL['x1']) & (Yr <= WALL['y1'])
    s |= (Yr <= RIB[1]) & (X <= -30.0)                                         # rib stub
    s |= (X >= BOSS['x0']) & (X <= BOSS['x1']) & (Yr >= BOSS['y0']) & (Yr <= BOSS['y1']) & (d >= BOSS['d0'])
    if d >= D.GROOVE_D[0] - C and d <= D.GROOVE_D[1] + C:                      # closed groove
        flare = np.maximum(FLARE - (WALL['y1'] - Yr), 0.0)
        s &= ~((X >= WALL['x1'] - D.GROOVE_DEPTH - C - flare) & (X <= WALL['x1'] + 0.01) & (Yr >= RIB[1]))
    if d >= WEB_CLEAR[1]:                                                      # web clearance in front of the wall
        s &= ~((X > WALL['x1']) & (X <= WALL['x1'] + WEB_CLEAR[0]) & (Yr >= RIB[1]))
    if d >= WALL['d1'] - PIN['L']:
        s &= ~hole(PIN['x'], PIN['y'], PIN['dia'], X, Yr)
    if d < BOSS['d0'] + SPIKE_DIG:                                             # spikes under the boss overhang (print
        h = d - WALL['d0']                                                     # only; user 2026-10-31)
        r = SPIKE_R[0] + (SPIKE_R[1] - SPIKE_R[0]) * h / (BOSS['d0'] + SPIKE_DIG - WALL['d0'])
        for sx, sy in SPIKES:
            s |= (X - sx) ** 2 + (Yr - sy) ** 2 <= r * r
    return s


def door_corner(x, y, z):
    """The door's -X corner STANDING on its grip end, as the door prints on pi_box_v4_blue_plate (user 2026-10-31)."""
    from designs.pi_box_v4_blue_plate import door_stand
    return door_stand(x, y, z) & (x - D.PLATE_X <= CORNER_X)           # stops before the port window (X -32) + trees


_W = 20.0                                                                     # X -50..-30
CORNER_X = -33.0
PIECES = [
    implicit_piece(_W, WALL['d1'] - WALL['d0'], WALL['y1'] - WALL['y0'], wall, x0=-_W - 2.0, z0=0.0),
    implicit_piece(20.0, R.Y_MAX - D.Y_LO, D.GRIP['d1'] - D.D_TOP, door_corner, x0=2.0, z0=0.0),
]
