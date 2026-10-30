"""v4 wide front panel (user approved home:2136): the full front incl. both R30 rounds below STEP (d 72), the centre
strip above it, L-hook side edges, tongues at the step and the rim. Geometry lives in pi_box_v4_core (front_panel_rig).
Viewer: python generate.py pi_box_v4_front_panel. Prints standing: floor edge down, tongues up (no overhangs)."""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
from designs import pi_box_v4_core as _C       # noqa: E402

SUPPORTS = False
EXPOSURE_S = 2.3
BOTTOM_EXPOSURE_S = 20.0
PIECES = _C.FRONT_PANEL_PIECES
