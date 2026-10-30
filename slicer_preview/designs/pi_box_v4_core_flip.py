"""v4 core, flipped Pi (the BASELINE, user 2026-10-29: Pi bottom-most is fixed; viewer: python generate.py
pi_box_v4_core_flip): pi_box_v4_core with PI_BOX_V4_PI=flip (SD end at the floor, ports facing the flange)."""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
from designs import pi_box_v4_core as _C     # noqa: E402

globals().update({k: v for k, v in vars(_C).items() if not k.startswith('__')})   # every name, _private ones too (pi_box_check)
