"""ARCHIVED 2026-10-13: pole-battery version (the battery-in-box version, pi_box_bat_*, was chosen).

Lidar rig Pi box, v1 +Y (buck side) panel: the -Y panel blank without the grille, with the
lidar-cable U-notch at the top of its +X end (the panel slides up round the cable)."""
from geometry import implicit_piece
from designs.archive.pole.pi_box_lids import PANEL_W, PANEL_H, PANEL_T, panel

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

PIECES = [implicit_piece(PANEL_W, PANEL_T, PANEL_H, lambda x, y, z: panel(x, y, z, grille=False, cable_notch=True),
                         x0=-PANEL_W / 2, z0=-PANEL_H / 2)]
