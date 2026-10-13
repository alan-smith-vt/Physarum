"""BATTERY variant -Y (Pi) panel with the intake grille; one per plate (170 × 95)."""
from geometry import implicit_piece
from designs.pi_box_bat_lids import PANEL_W, PANEL_H, PANEL_T, panel

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

# 170 mm tall: lies with its height along the printer's long axis (x)
PIECES = [implicit_piece(PANEL_H, PANEL_T, PANEL_W, lambda x, y, z: panel(z, y, x), x0=-PANEL_H / 2, z0=-PANEL_W / 2)]
