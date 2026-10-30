"""Styling sketch (for approval, not a model): the v4 core's front read as a fish face (2026-10-29).

Cameras = eyes (the flush cartridge caps on the R30 rounds), a soft brow over each, a mouth vent across the new
narrow front panel (in front of the Pi cooler), gill slots in the side walls behind the eyes (over the power board
on -X and the UE300 on +X). All below the lidar band (z <= 0; the band's -7° cone from z 45 is above z 0 out to r ~366)
and outside the camera views (ray test after modelling). Print rules: the core prints flange-down (up = +d).

    python pi_box_v4_fish_sketch.py -> pi_box_v4_fish_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Ellipse, Arc

X_OUT, Y_MIN, DEPTH = 50.0, -62.0, 180.0
R_FRONT = 30.0
EYE_D = 35.0
BROW = dict(d0=17.0, d1=21.0, out=3.0, half_t=19.0)        # ridge over the cap: top face 45°, bottom flat
MOUTH = dict(d=88.0, half_w=17.0, sag=7.0, slot_w=2.4, n=7, h=10.0)   # smile of short vertical slots
GILLS = dict(yr=(-28.0, -20.0, -12.0), d=(60.0, 140.0), w=2.4, bow=6.0)

fig = plt.figure(figsize=(18, 9.5))

# ── front elevation (from -Y): X across, z = -d ──
ax = fig.add_subplot(1, 3, 1)
ax.add_patch(Rectangle((-X_OUT, -DEPTH), 2 * X_OUT, DEPTH, fc='#2c3e50', ec='k', alpha=0.15))
for s in (-1, 1):
    ax.add_patch(Rectangle((s * 20 - (0 if s > 0 else 30), -DEPTH), 30, DEPTH, fc='#2c3e50', alpha=0.12))
    ax.add_patch(Ellipse((s * 41, -EYE_D), 15, 34, fc='#48c9b0', ec='#117864', alpha=0.9))      # cap (seen at 60° off)
    ax.add_patch(Ellipse((s * 41, -EYE_D - 1), 6, 9, fc='k'))                                   # lens
    ax.add_patch(Polygon([(s * 29, -BROW['d0']), (s * 49, -BROW['d0']), (s * 50.5, -BROW['d1']), (s * 30, -BROW['d1'] - 2)],
                         fc='#1b2631', ec='k'))                                                  # brow
ax.add_patch(Rectangle((-22.6, -DEPTH + 0.1), 45.2, DEPTH - 14, fc='#aed6f1', ec='#2471a3', alpha=0.6))   # front panel
xs = np.linspace(-MOUTH['half_w'], MOUTH['half_w'], MOUTH['n'])
for x in xs:
    zc = -(MOUTH['d'] + MOUTH['sag'] * (1 - (x / MOUTH['half_w']) ** 2))
    ax.add_patch(Rectangle((x - MOUTH['slot_w'] / 2, zc - MOUTH['h'] / 2), MOUTH['slot_w'], MOUTH['h'], fc='k'))
ax.text(0, -(MOUTH['d'] - 12), 'mouth: 7 slots 2.4 × 10\n(cooler vent, Pi d 55..140)', ha='center', fontsize=8)
ax.text(0, -12, 'brow ridges d 17..21, 3 proud', ha='center', fontsize=8)
ax.text(0, -60, 'eyes = cartridge caps\n(lens d 36, yaw 30°)', ha='center', fontsize=8)
ax.text(0, -170, 'front panel (v4, narrow): X ±20 + L-hooks, d 14..180', ha='center', fontsize=8, color='#1a5276')
ax.set_xlim(-60, 60); ax.set_ylim(-190, 10); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title('FRONT (from -Y): the face'); ax.set_xlabel('X'); ax.set_ylabel('z = -d')

# ── side elevation (+X): Yr across (forward = left), z = -d ──
ax = fig.add_subplot(1, 3, 2)
ax.add_patch(Rectangle((Y_MIN, -DEPTH), 124, DEPTH, fc='#2c3e50', alpha=0.15, ec='k'))
ax.add_patch(Ellipse((-46.5, -EYE_D), 26, 42, fc='#48c9b0', ec='#117864'))
ax.add_patch(Ellipse((-46.5, -EYE_D - 1), 9, 9, fc='k'))
ax.add_patch(Polygon([(-62, -BROW['d0']), (-31, -BROW['d0']), (-31, -BROW['d1']), (-62, -BROW['d1'] - 2)], fc='#1b2631'))
for y in GILLS['yr']:
    t = np.linspace(0, 1, 30)
    d = GILLS['d'][0] + t * (GILLS['d'][1] - GILLS['d'][0])
    yy = y - GILLS['bow'] * np.sin(np.pi * t)
    ax.plot(yy, -d, color='k', lw=4, solid_capstyle='round')
ax.text(-20, -150, 'gills: 3 bowed slots 2.4 wide, d 60..140\n(pointed ends: 45° in print)\nvent the power board (-X) / UE300 (+X)',
        ha='center', fontsize=8)
ax.annotate('', xy=(-75, -100), xytext=(-63, -100), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(-75, -96, 'FORWARD', fontsize=8, weight='bold')
ax.add_patch(Rectangle((30.8, -DEPTH + 1), 25, 157, fc='none', ec='grey', ls='--'))
ax.text(43, -100, 'battery', ha='center', fontsize=7, color='grey')
ax.set_xlim(-80, 70); ax.set_ylim(-190, 10); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title('SIDE (+X): eye, brow, gills'); ax.set_xlabel('Yr (forward = -Y, left)'); ax.set_ylabel('z = -d')

# ── brow section + rules ──
ax = fig.add_subplot(1, 3, 3)
ax.add_patch(Polygon([(26, -10), (30, -10), (30, -BROW['d0']), (30 + BROW['out'], -BROW['d0'] - BROW['out']),
                      (30 + BROW['out'], -BROW['d1']), (30, -BROW['d1']), (30, -60), (26, -60)], fc='#566573', ec='k'))
ax.add_patch(Rectangle((22.9, -63.5), 6.1, 41.8, fc='#48c9b0', alpha=0.6))
ax.plot([26.3, 26.3 + 30 * np.cos(np.radians(61.4))], [-36.2, -36.2 + 30 * np.sin(np.radians(61.4))], 'g--')
ax.text(31, -6, 'top view ray (61° elevation)\npasses outside the brow', fontsize=7, color='g')
ax.text(34, -24, 'brow: top face 45°\n(print: overhang 45°),\nbottom flat (faces print-up)', fontsize=7)
ax.set_xlim(15, 60); ax.set_ylim(-70, 5); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title('BROW section on the land normal (n →)')
fig.suptitle('v4 styling concept (secondary, for approval): fish face: eyes = cameras, brow, mouth vent, gills', fontsize=13)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=105)
