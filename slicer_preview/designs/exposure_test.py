"""Exposure test — 6 identical cells at 100/90/80/70/60/50% of EXPOSURE_S.

One file can only have one exposure time, so dose is varied per cell with
pixel gray level (gray 230 ≈ 90%). Gray → dose isn't perfectly linear on the
LCD, so treat the winner as a starting point and confirm it with a full
print at the matching EXPOSURE_S × fraction.

Each cell, on a 1.5 mm base that is always full exposure (so dim cells
still stick to the plate):
  pins     ⌀ 0.3 0.4 0.6 0.8 1.0 mm, 5 mm tall     — under-exposure: thin pins missing/bent
  slots    0.10 0.15 0.20 0.30 0.50 mm vertical    — over-exposure: narrow slots fused (XY bleed)
  holes    ⌀ 0.4 0.6 0.8 1.2 1.6 mm horizontal     — over-exposure: holes close from the roof (Z bleed)
  bridge   6 mm span, 0.6 mm thick, 5.4 mm up      — under: sags/breaks; over: thickens downward
  ID block engraved dots on top:
           front (-Z) edge = column, left (-X) edge = row

Cell layout (mm, top view):        grid: col 1   col 2   col 3
  z  3      pins (x 5…21)           row 1  100%    90%     80%
  z  7–9    slot wall               row 2   70%    60%     50%
  z 12–14   hole wall
  z 17–19   bridge (x 3–13) | ID block (x 15–24, z 16–25)
"""
import numpy as np
from geometry import implicit_piece
from modifiers import apply_grayscale

SUPPORTS = False
EXPOSURE_S = 2.5   # 100% cell. Old film ran 2.3 s; new film is clearer, so expect the winner lower

DOSE = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
N_COLS = 3

CELL = 26.0
CELL_GAP = 4.0
BASE_H = 1.5
FEATURE_XS = [5.0, 9.0, 13.0, 17.0, 21.0]

PIN_D = [0.3, 0.4, 0.6, 0.8, 1.0]
SLOT_W = [0.10, 0.15, 0.20, 0.30, 0.50]
HOLE_D = [0.4, 0.6, 0.8, 1.2, 1.6]


def _features(x, y, z, row, col):
    """Everything above the base, in cell-local mm (y measured from base top)."""
    def box(x0, x1, y0, y1, z0, z1):
        return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1) & (z >= z0) & (z <= z1)

    solid = np.zeros(np.broadcast_shapes(x.shape, z.shape), dtype=bool)

    # Pins
    for cx, d in zip(FEATURE_XS, PIN_D):
        solid |= ((x - cx) ** 2 + (z - 3.0) ** 2 <= (d / 2) ** 2) & (y <= 5.0)

    # Slot wall — slots run the full wall depth, from 1 mm above base to the top
    wall = box(2, 24, 0, 6, 7, 9)
    for cx, w in zip(FEATURE_XS, SLOT_W):
        wall &= ~((np.abs(x - cx) <= w / 2) & (y >= 1.0))
    solid |= wall

    # Hole wall — holes run along Z, centred 4 mm up
    wall = box(2, 24, 0, 8, 12, 14)
    for cx, d in zip(FEATURE_XS, HOLE_D):
        wall &= ~((x - cx) ** 2 + (y - 4.0) ** 2 <= (d / 2) ** 2)
    solid |= wall

    # Bridge: posts at x 3–5 and 11–13, beam on top
    solid |= box(3, 5, 0, 6, 17, 19) | box(11, 13, 0, 6, 17, 19)
    solid |= box(3, 13, 5.4, 6, 17, 19)

    # ID block, 3 mm tall, dots engraved 0.6 mm into the top
    block = box(15, 24, 0, 3, 16, 25)
    dots = [(15 + 1.5 + i * 1.8, 16 + 1.5) for i in range(col + 1)]       # front edge
    dots += [(15 + 1.5, 16 + 1.5 + 1.8 * (i + 1)) for i in range(row + 1)]  # left edge
    if y > 3 - 0.6:
        for cx, cz in dots:
            block &= ~((x - cx) ** 2 + (z - cz) ** 2 <= 0.6 ** 2)
    solid |= block

    return solid


PIECES = []
for i, dose in enumerate(DOSE):
    row, col = divmod(i, N_COLS)
    ox = (col - (N_COLS - 1) / 2) * (CELL + CELL_GAP) - CELL / 2
    oz = (row - 0.5) * (CELL + CELL_GAP) - CELL / 2

    PIECES.append(implicit_piece(CELL, BASE_H, CELL, lambda x, y, z: True,
                                 x0=ox, y0=0.0, z0=oz))

    features = implicit_piece(CELL, 8.0, CELL,
                              lambda x, y, z, r=row, c=col: _features(x, y, z, r, c),
                              x0=ox, y0=BASE_H, z0=oz)
    if dose < 1.0:
        apply_grayscale(features, round(255 * dose))
    PIECES.append(features)
