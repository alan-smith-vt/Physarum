"""Reprint of the stick collar's printed screws alone (2026-10-17): the first set shared the plate with the collar
halves and all 8 came off at ~14 mm (the collar's layer-area jump there), 12 of the 18 mm thread. Same screw as
stick_collar_hardware (18 mm thread, fits nut set B), nothing else on the plate.
"""
from sdf import union, sdf_piece
from designs import stick_collar_hardware as hw

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0

N, PITCH = 8, 12.0
ROOT = union(*[hw.screw(6.0 + PITCH * (i % 4), 6.0 + PITCH * (i // 4)) for i in range(N)])
PIECES = [sdf_piece(12.0 + PITCH * 3, hw.HEAD_H + hw.SCREW_L, 12.0 + PITCH, ROOT,
                    lambda x, y, z: (x, z, y), x0=-(12.0 + PITCH * 3) / 2, z0=-(12.0 + PITCH) / 2)]
