"""v3 CONCEPT (2026-10-15): OV9281 camera cartridge, inserted from outside into the v3 core's wall opening
(designs/pi_box_bat_core_v3.py). Two identical parts on one plate, printed flange-down.

  - Body = the v2 camera block as its own part: board face at CAM_FACE (8 mm inset), Ø19 lens window,
    bosses at the board's holes, connector pocket at the board's far (ribbon) edge. Pointed far end matches
    the opening (45°, self-supporting in the core).
  - Board: M2 screws from the OUTSIDE (heads in counterbores in the flange face) through the bosses and the
    board, into an M2 brass insert threaded on behind the board as a nut. Lens-cap pushes put the board on
    the bosses / the flange on the wall (compression); no glued inserts.
  - Flange: 3 × M2 self-tapping screws into pilot holes in the core's solid wall pad.
Modelled in its own frame (SDF slicing needs print-up as the tree's d axis):
  tree (X', Y', d') = (rig d, rig Yr, h)   with  h = X_OUT + CART_FL_T - |X|  (0 = flange face, on the plate).
"""
import numpy as np
from sdf import Box, Cyl, Half, union, inter, sub, sdf_piece, INF
from designs.pi_box_bat_rig import (X_OUT, CAM_D, CAM_BOSS_R, CAM_BOSS_H, CAM_FACE, CAM_POCKET,
                                    CAM_WINDOW, RESIN_HOLE_COMP)
from designs.pi_box_bat_core_v3 import (CAM_YR, CART_HW, CART_D0, CART_D1, CART_FL, CART_FL_T, CART_SCREWS,
                                        cam_holes)

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

SCREW_CLEAR, HEAD_CB = 2.2, 4.2        # M2 socket cap (head Ø3.8)
CB_TO = CAM_FACE + CAM_BOSS_H + 2.0    # counterbore floor |X|: 4 mm of boss + body under the head → M2×8 [MEASURE]
H_TOP = X_OUT + CART_FL_T - CAM_FACE   # part height: flange face to the boss tops (board face)


def h(ax):                             # rig |X| -> print height
    return X_OUT + CART_FL_T - ax


def profile(hw, d0, d1, h0, h1):
    """The core's opening pentagon (flat at d0, 45° point past d1) in this frame."""
    return inter(Box(d0, d1 + hw, CAM_YR - hw, CAM_YR + hw, h0, h1),
                 Half(1, 1, 0, d1 + hw + CAM_YR), Half(1, -1, 0, d1 + hw - CAM_YR))


def xcyl(cy, cd, r, a0, a1):           # rig X-axis cylinder between |X| a0..a1
    return Cyl('d', (cd, cy), r, h(max(a0, a1)), h(min(a0, a1)))


body = profile(CART_HW, CART_D0, CART_D1, h(X_OUT), h(CAM_FACE + CAM_BOSS_H))
flange = profile(CART_HW + CART_FL, CART_D0 - CART_FL, CART_D1, 0.0, CART_FL_T)
bosses = union(*[xcyl(cy, cd, CAM_BOSS_R, CAM_FACE, CAM_FACE + CAM_BOSS_H + 0.5) for cy, cd in cam_holes(1)])
r_win = (CAM_WINDOW + RESIN_HOLE_COMP) / 2
window = sub(Cyl('d', (CAM_D, CAM_YR), r_win, -1, H_TOP + 1),
             *[xcyl(cy, cd, CAM_BOSS_R + 1.0, CAM_FACE - 1, CAM_FACE + CAM_BOSS_H + 0.5) for cy, cd in cam_holes(1)])
pw, pdp = CAM_POCKET
bd1 = CAM_D + 15.6                     # board's ribbon edge
pocket = Box(bd1 - pw, bd1 + 1, CAM_YR - 13.5, CAM_YR + 13.5, h(CAM_FACE + pdp), H_TOP + 1)
holes = union(*[xcyl(cy, cd, (SCREW_CLEAR + RESIN_HOLE_COMP) / 2, CAM_FACE - 1, X_OUT + 5) for cy, cd in cam_holes(1)],
              *[xcyl(cy, cd, (HEAD_CB + RESIN_HOLE_COMP) / 2, CB_TO, X_OUT + 5) for cy, cd in cam_holes(1)],
              *[Cyl('d', (cd, cy), (SCREW_CLEAR + RESIN_HOLE_COMP) / 2, -1, CART_FL_T + 1) for cy, cd in CART_SCREWS])
ROOT = sub(union(body, flange, bosses), window, pocket, holes)

D0, D1 = CART_D0 - CART_FL - 0.5, CART_D1 + CART_HW + CART_FL + 0.5     # part extent along rig d
Y0, Y1 = CAM_YR - CART_HW - CART_FL - 0.5, CAM_YR + CART_HW + CART_FL + 0.5
GAP = 6.0


def to_model(x, y, z):
    """Local print (x along rig d, z along rig Yr, y up) -> tree (X', Y', d')."""
    return x + D0, z + Y0, y


PIECES = [sdf_piece(D1 - D0, H_TOP, Y1 - Y0, ROOT, to_model, x0=-(D1 - D0) / 2, z0=z0)
          for z0 in (-(Y1 - Y0) - GAP / 2, GAP / 2)]
