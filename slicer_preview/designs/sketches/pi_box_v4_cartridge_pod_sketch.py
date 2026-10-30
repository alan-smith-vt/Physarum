"""Sketch (for approval, not a model): v4 camera cartridge redone as a CAMERA-ALIGNED pod (2026-10-29).

The modelled cartridge is a box square to the land with the 24°-pitched camera carved into it: every camera
feature cuts it at an angle (fragments). Here the body is square to the optical axis instead: a rectangular pod,
board flat on its back, front face square to the axis with a plain lens hole + guard lip, and a flat flange where
the pod meets the land. It still slides in along the land normal (the core opening = the pod's shadow along n).

    python pi_box_v4_cartridge_pod_sketch.py -> pi_box_v4_cartridge_pod_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle

R_FRONT, LAND, SEAT = 30.0, 29.0, 22.9
PITCH = np.radians(24.0)
LENS = np.array([LAND, -35.0])                    # (n, z) lens front centre ON the land
f = np.array([np.cos(PITCH), np.sin(PITCH)])      # axis, out and up
u = np.array([-np.sin(PITCH), np.cos(PITCH)])     # board up
B = LENS - 18.0 * f                               # board front face centre
POD_U, POD_C, WALL = (-17.1, 10.75), (-4.6, 18.0), 1.5     # pod: along u (board -15.6..8.25 + wall), along the axis
GUARD = 1.0                                       # lip standing proud of the lens front (stays outside 62.5°)


def P(c, b):
    p = B + c * f + b * u
    return p[0], p[1]


fig, axs = plt.subplots(1, 2, figsize=(17, 9))
ax = axs[0]
# core round wall + plate + stub (schematic)
ax.add_patch(Rectangle((R_FRONT - 4, -100), 4, 100, fc='#d6eaf8', ec='k', alpha=0.6))
ax.add_patch(Rectangle((-5, 0), 45, 6.2, fc='#bdc3c7', ec='k'))
ax.add_patch(Rectangle((14.2 - 7.1, -12), 14.2, 12, fc='#d2b4de', ec='#7d3c98'))
ax.text(14.2, -9, 'M4 stub', ha='center', fontsize=7)
# pod
pod = [P(POD_C[0], POD_U[0]), P(POD_C[0], POD_U[1]), P(POD_C[1], POD_U[1]), P(POD_C[1], POD_U[0])]
ax.add_patch(Polygon(pod, fc='#48c9b0', ec='#117864', lw=1.5, alpha=0.9))
inner = [P(-1.6 - 0.3, -15.6 - 0.3), P(-1.6 - 0.3, 8.25 + 0.3), P(POD_C[1] - WALL, 8.25 + 0.3), P(POD_C[1] - WALL, -15.6 - 0.3)]
ax.add_patch(Polygon(inner, fc='white', ec='#117864', lw=0.8))
ax.plot(*zip(P(-1.6, -15.6), P(-1.6, 8.25)), color='#145a32', lw=3)
ax.text(*P(-12, -2), 'board 25 × 23.85\nflat on the back\n(M2 + captive nuts)', fontsize=7, color='#145a32')
ax.add_patch(Polygon([P(0, -7), P(0, 7), P(18, 7), P(18, -7)], fc='#566573', ec='k', alpha=0.9))
ax.add_patch(Polygon([P(18, 7.0), P(18, 9.7), P(18 + GUARD, 9.7), P(18 + GUARD, 7.0)], fc='#117864'))
ax.add_patch(Polygon([P(18, -7.0), P(18, -9.7), P(18 + GUARD, -9.7), P(18 + GUARD, -7.0)], fc='#117864'))
# flange: flat plate in the seat plane, where the pod crosses the land
z_lo, z_hi = -60.5, -21.5
ax.add_patch(Rectangle((SEAT, z_lo), 1.8, z_hi - z_lo, fc='#1abc9c', ec='#117864'))
ax.text(SEAT - 0.5, z_lo - 4, 'flange: a flat FRAME round the pod,\nin the seat plane; 3 × M2 into the core pad', fontsize=7, ha='right')
for da in (-40.5, 0, 37.4):
    a = PITCH + np.radians(da)
    e = np.array([np.cos(a), np.sin(a)]) * 28
    ax.plot([LENS[0], LENS[0] + e[0]], [LENS[1], LENS[1] + e[1]], color='#27ae60', ls='-' if da == 0 else '--')
ax.text(LENS[0] + 3, LENS[1] + 14, 'view: front face is square to the\naxis, so nothing of the pod is in\nfront of the lens (FOV < 180°)', fontsize=7, color='#1e8449')
top = P(POD_C[1], POD_U[1]); bot = P(POD_C[1], POD_U[0])
ax.annotate(f'top edge {top[0] - LAND:+.1f} (sits inside the land)', xy=top, xytext=(top[0] + 3, top[1] + 8), fontsize=7,
            arrowprops=dict(arrowstyle='->'))
ax.annotate(f'bottom edge {bot[0] - LAND:+.1f} past the land: a CHIN\n(can take a 45° chamfer below the lens)', xy=bot,
            xytext=(bot[0] + 3, bot[1] - 12), fontsize=7, arrowprops=dict(arrowstyle='->'))
ax.annotate('', xy=(R_FRONT + 9, -48), xytext=(R_FRONT + 2, -48), arrowprops=dict(arrowstyle='<-', lw=1.5))
ax.text(R_FRONT + 2, -52, 'slides in along n', fontsize=7)
ax.set_xlim(-5, 50); ax.set_ylim(-80, 10); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('n (outward along the land normal)'); ax.set_ylabel('z')
ax.set_title('SECTION on the land normal: pod square to the camera axis')

ax = axs[1]                                       # the face seen along the axis
ax.add_patch(Rectangle((-14.0, POD_U[0]), 28.0, POD_U[1] - POD_U[0], fc='#48c9b0', ec='#117864', lw=1.5))
ax.add_patch(plt.Circle((0, 0), 9.7, fc='#117864'))
ax.add_patch(plt.Circle((0, 0), 7.0, fc='#1c2833'))
ax.add_patch(plt.Circle((0, 0), 3.0, fc='#566573'))
for x in (-10.5, 10.5):
    for y in (-6.25, 6.25):
        ax.add_patch(plt.Circle((x, y), 1.1, fc='none', ec='#117864', ls=':'))
ax.text(0, POD_U[0] - 3, 'face: 28 × 27.9, Ø19.4 hole, 1 mm guard lip round the lens\n'
        'board screws (dotted) behind the face, not through it', ha='center', fontsize=8)
ax.set_xlim(-20, 20); ax.set_ylim(-25, 16); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title('FACE, looking down the camera axis')
fig.suptitle('v4 cartridge, proposed redo: camera-aligned pod + flat flange (replaces the land-aligned block)', fontsize=13)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
