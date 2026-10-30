"""Sketch (for approval): camera ribbon exits and routes to the flipped Pi's CSI connectors (v4, 2026-10-29).

Rig frame: X across (cameras at ±X), Yr (forward = -Y), d down from the plate face. Numbers from pi_box_v4_core
(flip): ribbon leaves each board's connector at (±34.0, -40.1, d 58.2); Pi 5 CSI connectors (flipped Pi, shift +2)
at X 21, d 117 (CAM1) and 125 (CAM0), on its front (-Y) face, Yr ~ -34. Ribbon in stock: 15->22-pin, 150 mm
(pi_box_bat_rig 'camera ribbons'); FFC 1 mm pitch 15-pin end ~16 wide, 22-pin end ~13 wide; flat creases allowed
(handoff: crease once, >= 5 mm from the stiffened ends).

    python pi_box_v4_ribbon_sketch.py -> pi_box_v4_ribbon_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

EXIT = (34.0, -40.1, 58.2)
CSI = [(21.0, 117.0, 'CAM1'), (21.0, 125.0, 'CAM0')]
PI = dict(x0=-26.0, x1=32.0, d0=89.0, d1=174.0)
PLUGS = (56.0, 86.0)
COOLER = dict(x0=-16.0, x1=26.0, d0=112.0, d1=162.0, y0=-46.6)
CART = dict(d0=21.7, d1=63.5)
RIBBON_L = 150.0


def length(pts):
    p = np.asarray(pts, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


# routes in (X, Yr, d)
PX = [EXIT, (36.0, -40.5, 64.0), (37.0, -41.0, 117.0), (31.0, -36.0, 117.0), (24.0, -34.0, 117.0)]   # +X cam -> CAM1
MX = [(-EXIT[0], EXIT[1], EXIT[2]), (-36.0, -40.5, 64.0), (-37.0, -50.0, 100.0), (-30.0, -51.0, 125.0),
      (31.0, -51.0, 125.0), (31.0, -36.0, 125.0), (24.0, -34.0, 125.0)]                             # -X cam -> CAM0
L_PX, L_MX = length(PX) + 8.0, length(MX) + 8.0                       # + ~4 mm into each connector

fig, axs = plt.subplots(1, 3, figsize=(21, 10), gridspec_kw=dict(width_ratios=[1.2, 1, 0.9]))
ax = axs[0]                                         # front view (from -Y): X across, z = -d
ax.add_patch(Rectangle((-50, -180), 100, 180, fc='none', ec='k', lw=1.5))
for s in (-1, 1):
    ax.add_patch(Rectangle((s * 31 - 11, -CART['d1']), 22, CART['d1'] - CART['d0'], fc='#48c9b0', ec='#117864', alpha=0.7))
    ax.plot(s * EXIT[0], -EXIT[2], 'o', color='#f1c40f', ms=9, mec='k')
ax.add_patch(Rectangle((PI['x0'], -PI['d1']), PI['x1'] - PI['x0'], PI['d1'] - PI['d0'], fc='#2ecc71', alpha=0.35, ec='g'))
ax.add_patch(Rectangle((-25, -PLUGS[1]), 54, PLUGS[1] - PLUGS[0], fc='#e67e22', alpha=0.35))
ax.text(2, -72, 'Pi plugs (point up)', fontsize=7, ha='center')
ax.add_patch(Rectangle((COOLER['x0'], -COOLER['d1']), COOLER['x1'] - COOLER['x0'], COOLER['d1'] - COOLER['d0'], fc='#95a5a6', alpha=0.5))
ax.text(5, -140, 'Pi 5 (flipped)\ncooler', fontsize=8, ha='center')
for x, d, lab in CSI:
    ax.add_patch(Rectangle((x - 1, -d - 6.5), 4, 13, fc='k')); ax.text(x - 9, -d, lab, fontsize=7)
for pts, col, lab in ((PX, '#c0392b', f'+X camera -> CAM1: {L_PX:.0f} mm'), (MX, '#2471a3', f'-X camera -> CAM0: {L_MX:.0f} mm')):
    p = np.asarray(pts); ax.plot(p[:, 0], -p[:, 2], color=col, lw=5, alpha=0.8, label=lab, solid_capstyle='round')
ax.legend(loc='lower left', fontsize=8)
ax.text(0, -15, 'connectors (yellow) at d 58.2: ribbons leave DOWN', ha='center', fontsize=8)
ax.set_xlim(-60, 60); ax.set_ylim(-185, 5); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('X'); ax.set_ylabel('z = -d'); ax.set_title('FRONT view (from -Y, panel off): both ribbon routes')

ax = axs[1]                                         # side view (from +X): Yr across (forward = left), z = -d
ax.add_patch(Rectangle((-62, -180), 124, 180, fc='none', ec='k'))
ax.add_patch(Rectangle((-33.5, -PI['d1']), 1.6, PI['d1'] - PI['d0'], fc='g'))
ax.add_patch(Rectangle((-48.5, -PI['d1']), 15, PI['d1'] - PI['d0'], fc='#2ecc71', alpha=0.25))
ax.add_patch(Rectangle((COOLER['y0'], -COOLER['d1']), 14, COOLER['d1'] - COOLER['d0'], fc='#95a5a6', alpha=0.5))
ax.add_patch(Rectangle((-57.5, -180), 3.0, 166, fc='#f9e79f', ec='#b7950b'))
ax.text(-56, -100, 'front panel', rotation=90, fontsize=7)
for pts, col in ((PX, '#c0392b'), (MX, '#2471a3')):
    p = np.asarray(pts); ax.plot(p[:, 1], -p[:, 2], color=col, lw=5, alpha=0.8)
ax.text(-20, -150, '-X ribbon crosses in front of the\ncooler at Yr -51 (6 mm to the\npanel, 4 mm to the cooler)', fontsize=7)
ax.set_xlim(-65, 10); ax.set_ylim(-185, 5); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('Yr (forward = left)'); ax.set_title('SIDE view (+X): ribbons in front of the Pi')

ax = axs[2]; ax.axis('off')
txt = f"""RIBBON EXIT (both cartridges)
- The board's 15-pin connector is on its back
  face at the bottom edge; the ribbon leaves
  DOWN, along the board.
- Today the cartridge's 1.2 mm bottom wall is
  in the way: PROPOSED 17 x 4 slot under the
  connector (board sketch). Down then works,
  no 180° fold needed.
- After the slot: one flat crease to straight
  down, 6 mm below the cartridge.

ROUTES (flat creases at each corner)
+X camera -> CAM1 (X 21, d 117):
  down the inside of the +X round, past the
  Pi's +X edge, crease in to the connector.
  ~{L_PX:.0f} mm  (stock 150 mm: fits, ~{RIBBON_L - L_PX:.0f} slack)
-X camera -> CAM0 (X 21, d 125):
  down the -X round to d 100, crease across
  in front of the cooler at Yr -51, crease
  back to the connector.
  ~{L_MX:.0f} mm  (stock 150 mm: {'TOO SHORT' if L_MX > RIBBON_L - 10 else 'fits'})
  -> needs a 200 mm 15->22-pin cable
  (Raspberry Pi 'Camera Cable Standard-Mini'
  200 mm [VERIFY part no.]).

[MEASURE] which way the Pi 5 CSI connectors
take the ribbon (assumed: from the board's
HDMI edge, +X in the flipped Pi), and the
ribbons we have (150 mm assumed from the rig).
"""
ax.text(0, 1, txt, va='top', family='monospace', fontsize=9)
fig.suptitle('v4 camera ribbons: exit from the cartridge and routes to the flipped Pi 5 (for approval)', fontsize=14)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=95)
print(f'+X {L_PX:.0f} mm, -X {L_MX:.0f} mm')
