"""Starter design — copy to designs/<name>.py and build PIECES.

Coordinates (mm): X/Z are the plate plane, Y is up (build direction).
Primitives live in geometry.py, carving/post-process in modifiers.py,
imported meshes via stl_slicer.stl_piece(path).

Preview:  python generate.py <name>   then  python server.py → http://localhost:5555
Print:    python to_goo.py <name>     → out/<name>.goo
"""
from geometry import solid_frame

SUPPORTS = False   # True → add TPMS lattice supports (supports.py)

PIECES = [
    solid_frame(outer_x=30.0, outer_z=30.0, bar_w=5.0, bar_h=5.0),
]
