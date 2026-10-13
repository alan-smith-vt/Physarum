"""Side panel blank (both sides identical): 1/8" (3.175) x 95.2 x 169.8 mm, no vents or holes.

Prototype: print two (e.g. tinted clear resin). Final: cut 1/8" clear polycarbonate from the DXF
written by designs/pi_box_panel_dxf.py. The vents are in the camera walls instead.
"""
from geometry import implicit_piece
from designs.pi_box_bat_lids import PANEL_W, PANEL_H, PANEL_T, panel

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

# 170 mm tall: lies with its height along the printer's long axis (x)
PIECES = [implicit_piece(PANEL_H, PANEL_T, PANEL_W, lambda x, y, z: panel(z, y, x, grille=False),
                         x0=-PANEL_H / 2, z0=-PANEL_W / 2)]
