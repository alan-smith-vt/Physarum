"""Sketch (for approval): where the camera ribbon leaves the cartridge and passes the core pad (v4, 2026-10-29).

Two views on the +X camera, from the PROPOSED dimensions (not the current solids): the section on the land normal
at t = 0 (cartridge bottom, pad, ribbon) and the view from inside the core onto the pad (both bottom flange screws
moved to t ±12, the 17-wide notch between them). Land coords: n out along the land normal, t along it, d down.

    python pi_box_v4_ribbon_exit_sketch.py -> pi_box_v4_ribbon_exit_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

PAD, SEAT, CSK, R = 17.9, 22.9, 4.0, 30.0
OPEN_D1, FL_D1, OPEN_T = 59.5, 63.5, 13.5
G = 0.375
PITCH = np.radians(24.0)
F = np.array([np.cos(PITCH), -np.sin(PITCH)]); U = np.array([-np.sin(PITCH), -np.cos(PITCH)])   # (n, d)
B = np.array([29.6, 35.18]) - 18.0 * F
CONN = B - 15.6 * U - 2.6 * F                    # ribbon at the connector (n 17.1, d 57.8)
EXIT = CONN - 9.0 * U                            # out of the cartridge slot (n 20.8, d 66.0)
NOTCH = dict(n=(PAD - 1.0, SEAT + 0.5), t=8.5 + G, d=(OPEN_D1 - 0.5, FL_D1 + 3.5))
SCREWS = [(-(OPEN_T + 2.0), 42.6), (OPEN_T + 2.0, 42.6), (-12.0, OPEN_D1 + 2.0), (12.0, OPEN_D1 + 2.0)]

fig, axs = plt.subplots(1, 2, figsize=(16, 9))
ax = axs[0]                                       # section at t = 0: n across, z = -d up
ax.add_patch(Rectangle((PAD, -(FL_D1 + 2.5)), SEAT - PAD, FL_D1 + 2.5 - OPEN_D1, fc='#aed6f1', ec='#2471a3'))
ax.text(PAD + 0.2, -(FL_D1 + 4.5), 'core pad (bottom bar)', fontsize=7, color='#1a5276')
ax.add_patch(Rectangle((R - 4, -100), 4, 100 - (FL_D1 + 2.5), fc='#d6eaf8', ec='k'))
ax.text(R - 3.6, -90, 'core round wall', rotation=90, fontsize=7)
ax.add_patch(Rectangle(NOTCH['n'][0:1] + (-NOTCH['d'][1],), NOTCH['n'][1] - NOTCH['n'][0], NOTCH['d'][1] - NOTCH['d'][0],
                       fc='none', ec='r', lw=2, ls='--'))
ax.text(NOTCH['n'][0] - 0.3, -(NOTCH['d'][1] + 1.6), 'NEW core notch n 16.9..23.4, d 59..67', color='r', fontsize=7)
pod = [B + c * F + b * U for c, b in ((-3, -17.1), (-3, 10.75), (18, 10.75), (18, -17.1))]
ax.add_patch(Polygon([(p[0], -p[1]) for p in pod], fc='#48c9b0', ec='#117864', alpha=0.6))
ax.add_patch(Polygon([(SEAT - CSK, -OPEN_D1), (R, -OPEN_D1), (R, -FL_D1), (SEAT, -FL_D1)], fc='#48c9b0', ec='#117864', alpha=0.6))
ax.text(23.5, -61.5, 'cap band', fontsize=7)
bd = [B + c * F + b * U for c, b in ((-1.6, -15.6), (-1.6, 8.25), (0, 8.25), (0, -15.6))]
ax.add_patch(Polygon([(p[0], -p[1]) for p in bd], fc='#1e8449'))
cn = [B + c * F + b * U for c, b in ((-1.6, -15.6), (-1.6, -11.1), (-4.1, -11.1), (-4.1, -15.6))]
ax.add_patch(Polygon([(p[0], -p[1]) for p in cn], fc='#f1c40f', ec='k'))
path = [CONN, EXIT, EXIT + np.array([0.0, 30.0])]
ax.plot([p[0] for p in path], [-p[1] for p in path], color='#e67e22', lw=4, solid_capstyle='round')
ax.annotate('ribbon: out along the board (24° bend\nat the slot), then straight down\nbetween the pad notch and the wall',
            xy=tuple((EXIT[0], -EXIT[1] - 8)), xytext=(4, -82), fontsize=7, arrowprops=dict(arrowstyle='->'))
ax.text(13.5, -52, 'slot in the cartridge\n(c -3.6..1, 17 wide)', fontsize=7, color='#117864')
ax.set_xlim(2, 34); ax.set_ylim(-100, -25); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('n (outward on the land normal)'); ax.set_ylabel('z = -d')
ax.set_title('SECTION at t = 0 (+X camera): ribbon exit')

ax = axs[1]                                       # from inside the core, onto the pad: t across, z = -d
ax.add_patch(Rectangle((-17.5 - 2.5, -(FL_D1 + 2.5)), 40, FL_D1 + 2.5 - 21.0 + 2.5, fc='#aed6f1', ec='#2471a3'))
ax.add_patch(Rectangle((-OPEN_T - G, -(OPEN_D1 + G)), 2 * (OPEN_T + G), OPEN_D1 - 25.0 + 2 * G, fc='white', ec='#2471a3'))
ax.text(0, -40, 'cartridge opening', ha='center', fontsize=8)
ax.add_patch(Rectangle((-NOTCH['t'], -NOTCH['d'][1]), 2 * NOTCH['t'], NOTCH['d'][1] - NOTCH['d'][0], fc='#fadbd8', ec='r', lw=2))
ax.text(0, -64.5, 'NEW notch 17.75 wide', ha='center', color='r', fontsize=8)
for t, d in SCREWS:
    ax.add_patch(Circle((t, -d), (4.2 + 0.35) / 2, fc='none', ec='k'))
    ax.add_patch(Circle((t, -d), (2.2 + 0.35) / 2, fc='#7f8c8d'))
ax.annotate('bottom flange screws: ONE at t 0 ->\nTWO at t ±12 (counterbores clear the\nnotch by 0.8)', xy=(12, -61.5), xytext=(4, -78),
            fontsize=7, arrowprops=dict(arrowstyle='->'))
ax.add_patch(Rectangle((-8.0, -(FL_D1 + 10)), 16.0, 10, fc='#e67e22', alpha=0.7))
ax.text(0, -(FL_D1 + 12), 'ribbon 16 wide, going down', ha='center', fontsize=7)
ax.set_xlim(-24, 24); ax.set_ylim(-82, -18); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('t (along the land)'); ax.set_ylabel('z = -d')
ax.set_title('FROM INSIDE THE CORE, onto the pad around the opening')
fig.suptitle('v4 camera ribbon exit (proposal): cartridge slot + core pad notch, bottom screws split to t ±12', fontsize=13)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
