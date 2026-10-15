"""Flood-fill gyroid support coupon v2 (2026-10-14): interface finish and break-off force.

16 cells (4×4) plus one pocket cell, printed as one plate. Each cell, from the build plate up:
  0.0–0.6   solid foot (adhesion), 16 × 16
  0.6–9.0   graded gyroid (6 mm period): dense at the foot, sparse mid-height, dense under the band
  9.0–10.5  fine-lattice interface band (rows: period × sheet t)
  + gap     air gap (columns: 0 / 0.05 / 0.10 / 0.15 mm = 0–3 layers)
  cap       2 mm solid plate, 14 × 14 (stands in for the part's down-facing face), with a pull eyelet on top
ID holes through each cap: dots along the -Z edge = column (gap), along the -X edge = row (lattice).

Pocket cell (far +X): an arch whose 12 × 6 mm underside pocket is filled with the row-1 fine lattice at
gap 0. It tests whether lattice can be picked out of a confined space (an arm against a torso).

Break-off test: clamp the foot, hook a peak-hold luggage/spring scale into the eyelet, pull straight up slowly
until the cap lets go; record the peak and divide by the 196 mm² cap area. Then photograph and feel the cap's
underside. Note: preview layers are 160 µm, so the 0.05–0.15 mm gaps only show at print resolution.
"""
import numpy as np
from geometry import implicit_piece

SUPPORTS = False
EXPOSURE_S = 2.6
BOTTOM_EXPOSURE_S = 20.0

LATTICES = [(1.5, 0.30), (1.5, 0.60), (2.5, 0.30), (2.5, 0.60)]   # rows: fine period mm, sheet t
GAPS = [0.0, 0.05, 0.10, 0.15]                                       # cols: air gap mm under the cap

CELL, PITCH = 16.0, 20.0           # foot / block footprint, cell spacing
FOOT_H, BULK_TOP, BAND_H = 0.6, 9.0, 1.5
BULK_P, BULK_T_LO, BULK_T_HI = 6.0, 0.35, 0.9
CAP, CAP_H = 14.0, 2.0
EYE_W, EYE_T, EYE_H, EYE_HOLE = 10.0, 3.0, 7.0, 4.2
ID_R = 0.6

NX = len(GAPS) + 1                 # + the pocket column
NZ = len(LATTICES)
SIZE_X, SIZE_Z = NX * PITCH, NZ * PITCH
TOP = BULK_TOP + BAND_H + max(GAPS) + CAP_H + EYE_H


def _gyr(x, y, z, period):
    k = 2 * np.pi / period
    return (np.sin(k * x) * np.cos(k * y) + np.sin(k * y) * np.cos(k * z) + np.sin(k * z) * np.cos(k * x))


def solid(x, y, z):
    col = np.floor(x / PITCH)
    row = np.floor(z / PITCH)
    u = x - (col + 0.5) * PITCH                     # cell-local, centred
    v = z - (row + 0.5) * PITCH
    in_block = (np.abs(u) <= CELL / 2) & (np.abs(v) <= CELL / 2)
    in_cap = (np.abs(u) <= CAP / 2) & (np.abs(v) <= CAP / 2)

    # per-cell parameters (row -> lattice, col -> gap); the pocket column uses row 0 / gap 0
    per = np.zeros_like(row + col)
    t_f = np.zeros_like(row + col)
    for i, (p, t) in enumerate(LATTICES):
        per = np.where(row == i, p, per)
        t_f = np.where(row == i, t, t_f)
    gap = np.zeros_like(row + col)
    for j, g in enumerate(GAPS):
        gap = np.where(col == j, g, gap)
    pocket_col = col == len(GAPS)

    band_top = BULK_TOP + BAND_H
    cap0 = band_top + gap
    cap1 = cap0 + CAP_H

    if y < FOOT_H:
        s = in_block
    elif y < BULK_TOP:
        w = (y - FOOT_H) / (BULK_TOP - FOOT_H)              # dense → sparse → dense
        t = BULK_T_LO + (BULK_T_HI - BULK_T_LO) * (1 - np.sin(np.pi * w))
        s = in_block & (np.abs(_gyr(x, y, z, BULK_P)) <= t)
    elif y < band_top:
        s = in_block & (np.abs(_gyr(x, y, z, per)) <= t_f)
    else:
        s = in_cap & (y >= cap0) & (y < cap1)
        # ID holes through the cap: column count along -Z, row count along -X
        for n in range(1, len(GAPS) + 1):
            for i in range(n):
                cx = (i - (n - 1) / 2) * 1.8
                s &= ~((col + 1 == n) & ((u - cx) ** 2 + (v + CAP / 2 - 1.5) ** 2 <= ID_R ** 2))
        for n in range(1, NZ + 1):
            for i in range(n):
                cz = (i - (n - 1) / 2) * 1.8
                s &= ~((row + 1 == n) & ((v - cz) ** 2 + (u + CAP / 2 - 1.5) ** 2 <= ID_R ** 2))
        # pull eyelet: a bar across X standing on the cap, horizontal hole along Z
        hy = y - cap1
        bar = (np.abs(u) <= EYE_W / 2) & (np.abs(v) <= EYE_T / 2) & (hy >= 0) & (hy < EYE_H)
        hole = u ** 2 + (hy - EYE_H / 2) ** 2 <= (EYE_HOLE / 2) ** 2
        s |= bar & ~hole

    # pocket column: one arch per row-0 cell only (the others stay empty)
    pc = pocket_col & (row == 0)
    legs = (np.abs(v) <= 3.0) & (np.abs(u) >= 6.0) & (np.abs(u) <= 8.0)
    roof0, roof1 = 8.0, 11.0
    arch = legs & (y < roof1) | (np.abs(u) <= 8.0) & (np.abs(v) <= 3.0) & (y >= roof0) & (y < roof1)
    fill = (np.abs(u) < 6.0) & (np.abs(v) <= 3.0) & (y >= FOOT_H) & (y < roof0) & \
        (np.abs(_gyr(x, y, z, LATTICES[0][0])) <= LATTICES[0][1])
    foot = (np.abs(u) <= 8.0) & (np.abs(v) <= 3.0) & (y < FOOT_H)
    s = np.where(pocket_col, pc & (arch | fill | foot), s)
    return s


PIECES = [implicit_piece(SIZE_X, TOP, SIZE_Z, solid, x0=-SIZE_X / 2, z0=-SIZE_Z / 2)]
