"""Sketch (2026-11-03): separate glue-in lips for the v4 floor + door so the rig stands flat over the thumb screw.
Bottom view (looking up at the floor), rig X across, Yr up the page. Four layout options + a section through a slot.
Run: python designs/sketches/pi_box_v4_floor_lips_sketch.py -> pi_box_v4_floor_lips_sketch.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Polygon, Arc

XO, YMIN, SPLIT, YMAX = 50.0, -62.0, 30.1, 62.0
RC, RX, RY = 30.0, 20.0, -32.0
LIP = '#1f77b4'
DOOR_LIP_OLD = dict(x=25.0, y0=58.0)


def outline(ax):
    a = np.linspace(np.pi, 1.5 * np.pi, 30)
    left = np.c_[-RX + RC * np.cos(a), RY + RC * np.sin(a)]
    right = np.c_[RX + RC * np.cos(a[::-1] + np.pi / 2), RY + RC * np.sin(a[::-1] + np.pi / 2)]
    right = np.c_[RX - (left[:, 0] + RX), left[:, 1]][::-1]
    floor = np.vstack([[[-XO, SPLIT]], left, right, [[XO, SPLIT]]])
    ax.add_patch(Polygon(floor, closed=True, fc='#eeeeee', ec='k', lw=1.2))
    ax.add_patch(Rectangle((-XO, SPLIT + 0.8), 2 * XO, YMAX - SPLIT - 0.8, fc='#e4ecf4', ec='k', lw=1.2))
    ax.text(0, 50, 'DOOR (reprint)', ha='center', fontsize=7, color='#555')
    ax.text(0, -45, 'FLOOR', ha='center', fontsize=7, color='#555')
    # things that hang below / openings to keep clear
    ax.add_patch(Circle((0, 0), 21.5, fc='white', ec='#888', ls='--'))
    ax.text(0, 0, 'stick\nØ43', ha='center', va='center', fontsize=6, color='#888')
    ax.add_patch(Rectangle((37.9 - 6.75, -12.1 - 8), 13.5, 16, fc='#ffcccc', ec='r'))
    ax.text(37.9, -24, 'RJ45\nplug?', ha='center', va='top', fontsize=6, color='r')
    ax.add_patch(Circle((33, 14), 8, fc='#ffe0b0', ec='#d80'))
    ax.text(33, 14, 'btn', ha='center', va='center', fontsize=6, color='#a60')
    ax.add_patch(Circle((41.1, 44), 4, fc='#ffcccc', ec='r'))
    ax.text(41.1, 44, 'screw\n6.5', ha='center', va='center', fontsize=5, color='r')
    ax.add_patch(Rectangle((-32, 36.8), 64, 13, fc='white', ec='#888', ls=':'))
    for bx, by in [(-42, -18), (45, 6.5), (-44, 24), (44, 24)]:
        ax.add_patch(Circle((bx, by), 2.25, fc='white', ec='k', lw=0.6))


def lip(ax, xy, w, h, **kw):
    ax.add_patch(Rectangle(xy, w, h, fc=LIP, ec='k', lw=0.6, **kw))


def tri(ax, pts):
    pts = np.array(pts + [pts[0]])
    ax.plot(pts[:, 0], pts[:, 1], color='g', lw=1, ls='--')


fig, axs = plt.subplots(1, 5, figsize=(20, 6.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.9]))
titles = ['A  Front bar + rear bar\n(2 long lines)',
          'B  Side rails + rear bar  (CHOSEN)\nR3.5 notches round the M3 button heads',
          'C  Corner arcs + rear bar\n(follow the R30 rounds)',
          'D  3 short keyed pads\n(tripod: ignores floor bow)']
for ax, t in zip(axs[:4], titles):
    outline(ax)
    ax.set_title(t, fontsize=9)
    ax.set_xlim(-56, 56); ax.set_ylim(-66, 66); ax.set_aspect('equal'); ax.axis('off')

# A
lip(axs[0], (-18, -60.5), 36, 3)
lip(axs[0], (-46, 57.5), 92, 3)
tri(axs[0], [(-18, -59), (18, -59), (46, 59), (-46, 59)])
# B
NOTCH_R = 3.5                                   # M3 button head Ø5.7 + clearance / hex key
for sx in (-1, 1):
    lip(axs[1], (45 if sx > 0 else -50, -30), 5, 58)
    for bx, by in [(-42, -18), (45, 6.5), (-44, 24), (44, 24)]:
        if np.sign(bx) == sx:
            axs[1].add_patch(Circle((bx, by), NOTCH_R, fc='#eeeeee', ec='k', lw=0.6, zorder=3))
            axs[1].add_patch(Circle((bx, by), 2.85, fc='white', ec='k', lw=0.6, zorder=4))
            axs[1].plot([sx * 47.5] * 2, [by - NOTCH_R - 1, by + NOTCH_R + 1], color='w', lw=2.5, zorder=5)
lip(axs[1], (-18, -60.5), 36, 3)
lip(axs[1], (-46, 57.5), 92, 3)
tri(axs[1], [(-18, -59), (18, -59), (47.5, -30), (47.5, 28), (46, 59), (-46, 59), (-47.5, 28), (-47.5, -30)])
axs[1].text(54, -5, 'rails 5 wide, X 45..50\nR3.5 notch per head\n(slot breaks there)',
            fontsize=5, color='k', rotation=90, va='center')
# C
for sx in (-1, 1):
    a0, a1 = (200, 250) if sx < 0 else (290, 340)
    a = np.radians(np.linspace(a0, a1, 25))
    cx = sx * RX
    r_o, r_i = RC - 1.0, RC - 4.0
    poly = np.vstack([np.c_[cx + r_o * np.cos(a), RY + r_o * np.sin(a)],
                      np.c_[cx + r_i * np.cos(a[::-1]), RY + r_i * np.sin(a[::-1])]])
    axs[2].add_patch(Polygon(poly, fc=LIP, ec='k', lw=0.6))
lip(axs[2], (-46, 57.5), 92, 3)
tri(axs[2], [(-44, -45), (44, -45), (46, 59), (-46, 59)])
# D
for cx in (-30, 30):
    lip(axs[3], (cx - 8, -54), 16, 3)
    lip(axs[3], (cx - 1.5, -60), 3, 12)
lip(axs[3], (-12, 57.5), 24, 3)
lip(axs[3], (-1.5, 52), 3, 9)
tri(axs[3], [(-30, -53), (30, -53), (0, 59)])

# section through a slot
ax = axs[4]
ax.set_title('Section through a slot (any layout)\nfloor face down = table side down', fontsize=9)
ax.add_patch(Polygon([(-12, 0), (-2, 0), (-2, 1.5), (2, 1.5), (2, 0), (12, 0), (12, 3), (-12, 3)],
                     fc='#dddddd', ec='k'))
ax.text(8, 1.5, 'floor 3.0\n(solid here,\nno gyroid)', fontsize=6, va='center')
ax.add_patch(Polygon([(-1.8, 1.4), (1.8, 1.4), (1.8, 0), (2.5, 0), (2.5, -7.7), (-2.5, -7.7), (-2.5, 0), (-1.8, 0)],
                     fc=LIP, ec='k'))
ax.annotate('', (3.5, 0), (3.5, -7.7), arrowprops=dict(arrowstyle='<->'))
ax.text(4, -4, '7.7 proud\n(plane 1 mm below\nthe screw tip)', fontsize=6, va='center')
ax.text(-11.5, -2.5, 'tongue 3.6 wide x 1.4\nin a 4.0 x 1.5 slot\n(0.2 glue gap/face)\nCA or thin epoxy',
        fontsize=6)
ax.text(-11.5, -9.8, 'lip 5 wide, foot\nedges 0.5 chamfer\nprints standing,\nfoot face = top', fontsize=6)
ax.plot([-12, 12], [-7.7, -7.7], 'g--', lw=0.8)
ax.text(0, -8.6, 'table', ha='center', fontsize=6, color='g')
ax.set_xlim(-12.5, 12.5); ax.set_ylim(-12, 5); ax.set_aspect('equal'); ax.axis('off')

fig.text(0.01, 0.02, 'Bottom view, rig X across / Yr up (front = bottom of page). Blue = glue-in lips, all ending on one '
         'plane 7.7 below the floor face (7.5 below the door plate). Green dashed = support polygon. Red = hangs below '
         'or must stay clear (thumb screw, RJ45 plug if one is plugged in from below), orange = power button.',
         fontsize=8)
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=130, bbox_inches='tight')
print(out)
