"""TPMS calibration coupon — 4×4 cells: solid fraction (rows) × cap clearance (cols).

Each cell: a TPMS lattice block standing on the plate, with a solid cap slab
floating above it, separated by a vertical air gap. After printing, check:
  - Rows (fill): does uncured resin drain out, or do pores close up?
  - Cols (gap): does the cap snap off cleanly, or did it fuse to the lattice?

ID holes are punched through each cap:
  dots along the front (-Z) edge = column (1 = smallest gap)
  dots along the left  (-X) edge = row    (1 = least solid)

Lateral clearance isn't tested here, only vertical.
Note: preview layers are 80 µm, so gaps under ~0.1 mm look closed in the
viewer even though they exist at the real 50 µm layer height.
"""
import numpy as np
from geometry import tpms_block, solid_frame
from printer import PX_MM, PZ_MM

SUPPORTS = False

THRESHOLDS = [0.15, 0.20, 0.30, 0.40]   # rows (Z): ~15/20/30/40% solid
GAPS_MM = [0.10, 0.20, 0.30, 0.50]      # cols (X): vertical clearance under the cap

CELL = 14.0          # mm, cell footprint
CELL_GAP = 4.0       # mm between cells
LATTICE_H = 10.0     # mm
PERIOD = 5.0         # mm, matches supports.py LATTICE_SPACING_MM
CAP_H = 1.5          # mm


def _id_holes(piece, n_front, n_left, r=0.6, spacing=1.8, inset=1.5):
    """Punch through-holes: n_front along the -Z edge, n_left along the -X edge."""
    px = np.arange(piece['W']) * PX_MM
    pz = np.arange(piece['H']) * PZ_MM
    w_mm, h_mm = piece['W'] * PX_MM, piece['H'] * PZ_MM
    centers = [(w_mm / 2 + (i - (n_front - 1) / 2) * spacing, inset) for i in range(n_front)]
    centers += [(inset, h_mm / 2 + (i - (n_left - 1) / 2) * spacing) for i in range(n_left)]
    mask = np.zeros((piece['H'], piece['W']), dtype=bool)
    for cx, cz in centers:
        mask |= (px[None, :] - cx) ** 2 + (pz[:, None] - cz) ** 2 <= r ** 2

    orig = piece['make_slice']
    def _punched(layer, _orig=orig, _mask=mask):
        img = _orig(layer).copy()
        img[_mask] = 0
        return img
    piece['make_slice'] = _punched


PIECES = []
for row, threshold in enumerate(THRESHOLDS):
    for col, gap in enumerate(GAPS_MM):
        cx = (col - (len(GAPS_MM) - 1) / 2) * (CELL + CELL_GAP)
        cz = (row - (len(THRESHOLDS) - 1) / 2) * (CELL + CELL_GAP)

        lattice = tpms_block(size_x=CELL, size_y=LATTICE_H, size_z=CELL,
                             period_mm=PERIOD, threshold=threshold)
        lattice['OFFSET_X_MM'] += cx
        lattice['OFFSET_Z_MM'] += cz

        # bar_w past half the size → no hole → solid slab
        cap = solid_frame(outer_x=CELL, outer_z=CELL, bar_w=CELL, bar_h=CAP_H)
        cap['OFFSET_X_MM'] += cx
        cap['OFFSET_Z_MM'] += cz
        cap['OFFSET_Y_MM'] = LATTICE_H + gap
        _id_holes(cap, n_front=col + 1, n_left=row + 1)

        PIECES += [lattice, cap]
