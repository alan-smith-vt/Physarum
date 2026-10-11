"""Calibration grid — 3×3 cells: strut diameter (rows) × grayscale exposure (cols).

Printed 2026-03-28 to dial in strut thickness and exposure for the resin.
"""
import numpy as np
from geometry import solid_frame, base_lattice
from modifiers import punch_dot_holes, apply_grayscale

SUPPORTS = False

# ── Calibration grid: 3×3, strut thickness (rows) × grayscale exposure (cols) ──
CELL_SIZE = 25.0
BAR_W = 3.0
CELL_GAP = 5.0
COL_SPACING = 2 * np.pi + 2 * 2.0   # same as full box

STRUT_DIAMETERS = [0.25, 0.35, 0.50]   # rows (Z axis)
GRAY_LEVELS = [180, 215, 255]           # columns (X axis)

PIECES = []
for row, strut_d in enumerate(STRUT_DIAMETERS):
    bar_h = strut_d / 2 + strut_d * 4   # match lattice total_h
    for col, gray in enumerate(GRAY_LEVELS):
        cx = (col - 1) * (CELL_SIZE + CELL_GAP)
        cz = (row - 1) * (CELL_SIZE + CELL_GAP)

        frame = solid_frame(outer_x=CELL_SIZE, outer_z=CELL_SIZE,
                            bar_w=BAR_W, bar_h=bar_h)
        frame['OFFSET_X_MM'] += cx
        frame['OFFSET_Z_MM'] += cz

        lattice = base_lattice(grid_x_mm=CELL_SIZE, grid_z_mm=CELL_SIZE,
                               spacing_mm=COL_SPACING, strut_d_mm=strut_d)
        lattice['OFFSET_X_MM'] += cx
        lattice['OFFSET_Z_MM'] += cz

        punch_dot_holes(frame, col + 1)

        if gray < 255:
            apply_grayscale(frame, gray)
            apply_grayscale(lattice, gray)

        PIECES.append(frame)
        PIECES.append(lattice)
