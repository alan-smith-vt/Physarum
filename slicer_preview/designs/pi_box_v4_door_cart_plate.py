"""Blue plate 2 (user 2026-11-03): both camera cartridges (pi_box_v4_cart_tilt, standing on the picked face, with their
support columns) + battery door A reprint STANDING on its grip end (pi_box_v4_blue_plate.door_stand: trees under
the port window's top edge). Door changes 2026-11-03 (door_a): CLEAR 0.1, taller per-side tongues, Ø3.2 slide-through
screw hole, full-width standing lip down to the floor lips' table plane.
All tall and narrow, no big plate-to-thin step (user: avoid large area changes between layers).
Plate 211.7 x 118.4 (x, z): cartridges side by side in the middle (z -29.5..29.5), the door in the +z strip.
Export: python to_goo.py pi_box_v4_door_cart_plate  (CPU: the cartridge and door pieces are numpy-only)
"""
from geometry import implicit_piece
from designs import pi_box_v4_cart_tilt as C
from designs import pi_box_v4_blue_plate as B
from designs.print_post import bevel_piece

SUPPORTS, EXPOSURE_S, BOTTOM_EXPOSURE_S = False, 2.3, 20.0
DOOR_GAP = 4.0                       # door <-> cartridges on the plate
DOOR_EF = 0.5                        # 45° x 0.5 bevel round the door's plate face (user 2026-11-03)

_cz1 = max(p['OFFSET_Z_MM'] + p['H'] * __import__('geometry').PZ_MM for p in C.PIECES)
DOOR_Z0 = _cz1 + DOOR_GAP
assert DOOR_Z0 + B.DOOR_D <= 118.4 / 2 - 2, 'door runs off the plate'

PIECES = list(C.PIECES) + [bevel_piece(implicit_piece(B.DOOR_W, B.DOOR_H, B.DOOR_D, B.door_stand,
                                                   x0=-B.DOOR_W / 2, z0=DOOR_Z0), DOOR_EF)]
