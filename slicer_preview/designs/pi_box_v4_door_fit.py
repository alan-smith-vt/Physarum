"""v4 PROPOSAL, UNAPPROVED (2026-10-29, v4-floor): fit strip for the option-A battery door. Do not print yet.

Print this before the core/door: it tests the door's rail clearances and the thumb-screw lock at the real sizes.
  1. WALL: a -X wall stub (X -50..-46, Yr 28.5..52, d 165..180) with the battery rib stub, the closed door groove
     (core DOOR_RAIL: d 175.6..178.4, 1.6 deep, +comp) with its 45° mouth flare at +Y, the 1.8 web clearance, and
     the lock boss (X -46..-38, Yr 38..50, d 168..180) with its M3 insert hole Ø4 × 6 (+comp) at SCREW_XY.
  2. DOOR: the -X corner of designs/pi_box_v4_door_a.py (X <= -30, Yr <= 52): plate, web + hook, captive M3 hole.
Check: the door corner slides in from +Y with no force and no rattle, butts the rib, and the thumb screw
runs into the insert by hand. Clearances: 0.4 per side + RESIN_HOLE_COMP/2 on holes; external faces are pulled in by
OVERSIZE (= EXT_COMP/2 of ir_filter_clip).
Rig -> print (both pieces, top down): y = d - 165 (wall) or the door's own mapping.
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_bat_rig import RESIN_HOLE_COMP, hole
from designs import pi_box_v4_door_a as D

SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.3, 20.0
C = RESIN_HOLE_COMP / 2
WALL = dict(x0=-50.0, x1=-46.0, y0=28.5, y1=52.0, d0=165.0, d1=180.0)
RIB = (28.5, 30.5)
BOSS = dict(x0=-46.0, x1=-38.0, y0=38.0, y1=50.0, d0=168.0)
PIN = dict(x=D.SCREW_XY[0], y=D.SCREW_XY[1], dia=4.0, L=6.0)   # M3 insert Ø4 x 4 + 2 for the screw tip
FLARE = 1.0
WEB_CLEAR = (1.8, 175.5)                         # core: 1.8 clear inside the wall face below d 175.5


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
    return s


def door_corner(x, y, z):
    s = D.door(x, y, z)
    X, Yr = x - D.PLATE_X, z + D.Y_LO
    return s & (X <= -30.0) & (Yr <= WALL['y1'])


_W = 20.0                                                                     # X -50..-30
PIECES = [
    implicit_piece(_W, WALL['d1'] - WALL['d0'], WALL['y1'] - WALL['y0'], wall, x0=-_W - 2.0, z0=0.0),
    implicit_piece(2 * D.PLATE_X, D.GRIP['d1'] - D.D_TOP, WALL['y1'] - D.Y_LO, door_corner, x0=2.0 - (50 - D.PLATE_X), z0=0.0),
]
