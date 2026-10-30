"""v4 core, flipped-Pi ALTERNATIVE (viewer: python generate.py pi_box_v4_core_flip): pi_box_v4_core with
PI_BOX_V4_PI=flip (SD end at the floor, ports facing the flange). Proposal for comparison only."""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
from designs.pi_box_v4_core import *        # noqa: F401,F403,E402
from designs.pi_box_v4_core import PIECES, OVERLAYS, ROOT, core   # noqa: F401,E402
