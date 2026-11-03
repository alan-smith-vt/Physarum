"""textile_patch with SQUASH 0.6: links 12.2 mm tall (original 20.3). Neighbours still locked, min clearance ~0.70 mm
(scan 2026-11-02: 100 escape directions, none free, at squash 1.0..0.5)."""
import os
os.environ['TEXTILE_SQUASH'] = '0.6'
from designs.textile_patch import *    # noqa: E402,F401,F403
