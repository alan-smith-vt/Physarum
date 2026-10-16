"""Side panel (both sides identical): 1/8" (3.175) x 95.2 x 169.8 mm, open 12 mm sheet gyroid (the walls'
pattern) inside a PANEL_BORDER solid frame: cuts the flat-on-the-plate layer area to about a third.

Prototype: print two (e.g. tinted clear resin). Final: cut 1/8" clear polycarbonate from the DXF
written by designs/pi_box_panel_dxf.py. The vents are in the camera walls instead.
"""
from geometry import implicit_piece
from designs.pi_box_bat_lids import PANEL_W, PANEL_H, PANEL_T, panel
from designs.pi_box_bat_rig import Y_MIN, PANEL_LIP

SUPPORTS = False
EXPOSURE_S = 2.3   # ELEGOO ABS-like clear blue + Siraya Tenacious clear 75/25, exposure_test 2026-10-15 (90% cell, vat ~25 C)
BOTTOM_EXPOSURE_S = 20.0

# 170 mm tall: lies with its height along the printer's long axis (x)
PIECES = [implicit_piece(PANEL_H, PANEL_T, PANEL_W, lambda x, y, z: panel(z, y, x, grille=False, gyroid=(Y_MIN + PANEL_LIP, +1)),
                         x0=-PANEL_H / 2, z0=-PANEL_W / 2)]
