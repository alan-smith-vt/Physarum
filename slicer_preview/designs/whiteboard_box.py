"""Whiteboard box frame — printed 2026-03-31 as full_box_v2.goo, came out great.

Lattice box ~120 × 70 × 90 mm: sliced STL frame + vaulted base lattice +
quad-spiral wall columns with bridge struts, arched top beams, back-wall hole.
"""
import os
import numpy as np
from geometry import base_lattice, quad_spiral_column, bridge_struts, bridge_struts_x
from modifiers import arch_top_beams, vault_base
from stl_slicer import stl_piece

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STL_FILE = os.path.join(PROJECT_ROOT, 'whiteboard_box_frame_75mm.stl')

SUPPORTS = False

# ── Full box (final print settings) ──
HELIX_R = 2.0
COL_SPACING = 2 * np.pi + 2 * HELIX_R   # ≈ 10.283 mm
WALL_START = np.pi / 2   # front/back wall filaments start at ±Z not ±X
SIDE_N_COLS = int(65.0 / COL_SPACING) + 2          # columns along Z
col_z = [(i - (SIDE_N_COLS - 1) / 2) * COL_SPACING for i in range(SIDE_N_COLS)]
FRONT_N_COLS = int(115.0 / COL_SPACING) + 1        # 12 columns along X
col_x = [(i - (FRONT_N_COLS - 1) / 2) * COL_SPACING for i in range(FRONT_N_COLS)]
PIECES = []
_stl = stl_piece(STL_FILE, rot_x=-np.pi/2, rot_y=np.pi/2)
_side_z_pairs = [(col_z[i], col_z[i+1]) for i in range(1, len(col_z) - 2)]
_front_x_pairs = [(col_x[i], col_x[i+1]) for i in range(1, len(col_x) - 2)]
_back_x_pairs = ([(col_x[i], col_x[i+1]) for i in range(1, 3)] +
                 [(col_x[i], col_x[i+1]) for i in range(len(col_x) - 4, len(col_x) - 2)] +
                 [(col_x[10], 52.68), (13.59, col_x[8]),
                  (-52.68, col_x[1]), (col_x[3], -13.59)])
arch_top_beams(_stl,
    side_walls=[(-60.0, _side_z_pairs), (60.0, _side_z_pairs)],
    front_back_walls=[(-35.0, _front_x_pairs), (35.0, _back_x_pairs)],
    beam_bottom_y=95.0, arch_mm=1.0)
PIECES.append(_stl)
_lattice = base_lattice(col_x=col_x, col_z=col_z, grid_z_mm=65.0, spacing_mm=COL_SPACING)
vault_base(_lattice, col_x, col_z, vault_mm=2.0)
PIECES.append(_lattice)
for side_x in [-60.0, 60.0]:
    PIECES += [quad_spiral_column(offset_x_mm=side_x, offset_z_mm=z,
                                  base_taper_mm=3.5, base_helix_r=1.2, base_filament_r=0.75)
               for z in col_z[1:-1]]
    for i in range(len(col_z) - 1):
        if i == 0:
            PIECES.append(bridge_struts(col_z[i], col_z[i+1], offset_x_mm=side_x,
                                        taper_mm=4.0, taper_r=1.5, taper_end='a'))
        elif i == len(col_z) - 2:
            PIECES.append(bridge_struts(col_z[i], col_z[i+1], offset_x_mm=side_x,
                                        taper_mm=4.0, taper_r=1.5, taper_end='b'))
        else:
            PIECES.append(bridge_struts(col_z[i], col_z[i+1], offset_x_mm=side_x))
FRONT_Z = -35.0
PIECES += [quad_spiral_column(offset_x_mm=x, offset_z_mm=FRONT_Z,
                              start_angle=WALL_START,
                              base_taper_mm=3.5, base_helix_r=1.2, base_filament_r=0.75)
           for x in col_x]
PIECES += [bridge_struts_x(col_x[i], col_x[i+1], offset_z_mm=FRONT_Z,
                            start_angle=WALL_START)
           for i in range(len(col_x) - 1)]
BACK_Z = 35.0
HOLE_R = 16.0
HOLE_CY = 90.0 - 20.0
back_col_h = []
for x in col_x:
    if abs(x) >= HOLE_R:
        back_col_h.append(90.0)
    else:
        back_col_h.append(HOLE_CY - np.sqrt(HOLE_R ** 2 - x ** 2))
for x, h in zip(col_x, back_col_h):
    PIECES.append(quad_spiral_column(offset_x_mm=x, offset_z_mm=BACK_Z,
                                     wall_y_mm=h, start_angle=WALL_START,
                                     ref_wall_y_mm=90.0,
                                     base_taper_mm=3.5, base_helix_r=1.2,
                                     base_filament_r=0.75))
for i in range(len(col_x) - 1):
    h = min(back_col_h[i], back_col_h[i+1])
    PIECES.append(bridge_struts_x(col_x[i], col_x[i+1],
                                   offset_z_mm=BACK_Z, wall_y_mm=h,
                                   start_angle=WALL_START,
                                   ref_wall_y_mm=90.0))
