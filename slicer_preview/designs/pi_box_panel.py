"""Lidar rig Pi box, v1 +Y (buck side) panel: the -Y panel blank without the grille."""
from geometry import implicit_piece
from designs.pi_box_lids import PANEL_W, PANEL_H, PANEL_T, panel

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

PIECES = [implicit_piece(PANEL_W, PANEL_T, PANEL_H, lambda x, y, z: panel(x, y, z, grille=False),
                         x0=-PANEL_W / 2, z0=-PANEL_H / 2)]
