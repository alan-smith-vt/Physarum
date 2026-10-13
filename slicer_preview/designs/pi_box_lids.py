"""Lidar rig Pi box, v1 floor + -Y (Pi) panel, printed flat on one plate.

Floor: outside face on the build plate; collar (round the well tube), web rails
and the floor top face up. Dowels (3×6) press into Ø3.3 holes and stand 3 mm
proud into the wall feet. 4 × M2.5 screws up into the core's glued inserts,
heads in shallow counterbores. Open notch under the Pi so its cables stay plugged in.
Panel: slides up into the camera-wall grooves; intake grille in front of the cooler.
The +Y panel is the same blank without the grille: designs/pi_box_panel.py.

Rig -> print for the floor: (X, Yr, d) -> (x = X, y = DEPTH + FLOOR_T - d, z = -Yr).
"""
import numpy as np
from geometry import implicit_piece
from designs.pi_box_rig import *          # noqa: F401,F403

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

FLOOR_T = 3.0
COLLAR_H = 4.0
COLLAR_R0 = WELL_R + TUBE_T + 0.15 + RESIN_HOLE_COMP / 2      # bore round the tube (prints undersize)
COLLAR_R1 = COLLAR_R0 + 1.8
RAIL_H, RAIL_W = 4.0, 1.5
RAIL_X = (WELL_R + 3.0, FOOT_X0 - 0.5)
NOTCH_X, NOTCH_Y = 29.0, PCB_YR - 0.5                        # open to the -Y edge
SCREW_CLEAR, HEAD_CB, HEAD_CB_DEPTH = 2.9, 5.4, 1.5

PANEL_W = 2 * (X_IN + GROOVE_D) - 0.8      # external faces print ~0.35 oversize
PANEL_H = DEPTH - FLANGE - 0.2
GRILLE = [(-20.0 + 5.5 * i, -20.0 + 5.5 * i + 2.6) for i in range(8)]
GRILLE_D = (25.0, 85.0)

GAP = 3.0
FLOOR_X0 = -(2 * X_OUT + GAP + PANEL_W) / 2
PANEL_X0 = FLOOR_X0 + 2 * X_OUT + GAP


def floor(x, y, z):
    X = x - X_OUT                          # local -> rig X
    Yr = -(z - Y_MAX)                      # print z grows toward -Yr
    aX = np.abs(X)
    r2 = X ** 2 + Yr ** 2
    if y < FLOOR_T:
        inset = max(CHAMFER - y, 0.0)
        s = (aX <= X_OUT - inset) & (Yr >= Y_MIN + inset) & (Yr <= Y_MAX - inset)
        s &= ~hole(0, 0, 2 * WELL_R, X, Yr)
        s &= ~((aX <= NOTCH_X) & (Yr <= NOTCH_Y))
        for bx, by in BOSSES:
            s &= ~hole(bx, by, HEAD_CB if y < HEAD_CB_DEPTH else SCREW_CLEAR, X, Yr)
        for px, py in PINS:
            s &= ~((X - px) ** 2 + (Yr - py) ** 2 <= (PIN_PRESS / 2) ** 2)
        return s
    h = y - FLOOR_T
    s = (r2 >= COLLAR_R0 ** 2) & (r2 <= COLLAR_R1 ** 2) & (h <= COLLAR_H)
    gap = WEB_T / 2 + 0.15 + RESIN_HOLE_COMP / 2
    rails = (aX >= RAIL_X[0]) & (aX <= RAIL_X[1]) & (np.abs(Yr) >= gap) & (np.abs(Yr) <= gap + RAIL_W)
    s |= rails & (h <= RAIL_H)
    return s


def panel(x, y, z, grille=True):
    X = x - PANEL_W / 2
    d = z + FLANGE + 0.1                   # print z runs down the panel
    s = np.ones(np.broadcast_shapes(X.shape, d.shape), bool)
    inset = max(CHAMFER * 0.5 - y, 0.0)
    s &= (np.abs(X) <= PANEL_W / 2 - inset)
    if grille:
        for x0, x1 in GRILLE:
            s &= ~((X >= x0) & (X <= x1) & (d >= GRILLE_D[0]) & (d <= GRILLE_D[1]))
    return s


PIECES = [
    implicit_piece(2 * X_OUT, FLOOR_T + COLLAR_H, Y_MAX - Y_MIN, floor, x0=FLOOR_X0, z0=-(Y_MAX - Y_MIN) / 2),
    implicit_piece(PANEL_W, PANEL_T, PANEL_H, panel, x0=PANEL_X0, z0=-PANEL_H / 2),
]
