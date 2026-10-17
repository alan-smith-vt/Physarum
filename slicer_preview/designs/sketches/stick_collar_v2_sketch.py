"""Concept sketch v2 (for approval, not a model): split cone collar with a clamp band below the floor, M4 screws
into M4 heat-set/glued inserts used as nuts (insert pressed in from the outside of the far half, against a
shoulder, so the screw pulls it into the resin: no glue in the load path).

    python stick_collar_v2_sketch.py -> stick_collar_v2_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle

STICK_D = 92 / np.pi
BORE = STICK_D - 0.3                    # squeeze
BORE_R, TUBE_T, DEPTH, FLOOR_T = 21.5, 2.0, 180.0, 3.0
SEAT = DEPTH + FLOOR_T
C_TOP_D, C_SEAT_D, C_LEN = 41.0, 43.0, 27.0   # cone: Ø41 top -> Ø43 at the floor underside (13 mm) -> Ø45 at 27 mm
c_bot_d = C_TOP_D + (C_SEAT_D - C_TOP_D) * C_LEN / 13.0
BAND_D, BAND_LEN = 56.0, 20.0           # clamp band below the cone
SPLIT = 1.0
M4_CLEAR, M4_HEAD_D, INS_D, INS_L = 4.5, 7.5, 5.6 + 0.35, 6.0   # insert OD [MEASURE], +RESIN_HOLE_COMP
SCREW_X = 20.5                          # screw axis offset from the stick axis (in the split plane's normal)
cone_top = SEAT - 13.0

fig = plt.figure(figsize=(17, 10))
ax = fig.add_axes([0.03, 0.06, 0.42, 0.88]); ax.set_aspect('equal'); ax.invert_yaxis()
ax.set_title('Section through the stick axis, bottom of the core (mm)', fontsize=13)
ax.add_patch(Rectangle((-19, 120), 38, 35, fc='#ddd', ec='k', hatch='..')); ax.text(0, 138, 'foam grip', ha='center', fontsize=8)
ax.add_patch(Rectangle((-STICK_D / 2, 155), STICK_D, 95, fc='#ccc', ec='k')); ax.text(0, 245, f'stick Ø{STICK_D:.1f}', ha='center', fontsize=9)
for s in (-1, 1):
    ax.add_patch(Rectangle((BORE_R if s > 0 else -BORE_R - TUBE_T, 120), TUBE_T, DEPTH - 120, fc='#7fb3d5', ec='k'))
    ax.add_patch(Rectangle((BORE_R if s > 0 else -BORE_R - 32, DEPTH), 32, FLOOR_T, fc='#a9cce3', ec='k'))
ax.text(BORE_R + 3, 150, 'well tube Ø43', fontsize=9); ax.text(BORE_R + 6, DEPTH - 2, 'floor', fontsize=8)
yt, ys, yb = cone_top, cone_top + C_LEN, cone_top + C_LEN + BAND_LEN
for s in (-1, 1):
    pts = [(s * BORE / 2, yt), (s * C_TOP_D / 2, yt), (s * c_bot_d / 2, ys), (s * BAND_D / 2, ys), (s * BAND_D / 2, yb),
           (s * BORE / 2, yb)]
    ax.add_patch(Polygon(pts, fc='#f5b041', ec='k', lw=1.2))
for z in (ys + 6, ys + 14):
    ax.plot([-BAND_D / 2 - 2, BAND_D / 2 + 2], [z, z], 'k--', lw=0.8)
ax.plot([-35, 35], [SEAT, SEAT], 'r:', lw=0.8)
ax.text(-62, SEAT - 1, 'seats at Ø43\n(floor underside)', color='r', fontsize=8, va='bottom')
dims = [(yt, f'Ø{C_TOP_D:.0f}'), (ys, f'Ø{c_bot_d:.0f} / band Ø{BAND_D:.0f}'), (yb, '')]
for y, t in dims:
    ax.text(-BAND_D / 2 - 3, y, t, ha='right', va='center', fontsize=9)
ax.annotate('', (BAND_D / 2 + 9, yt), (BAND_D / 2 + 9, ys), arrowprops=dict(arrowstyle='<->'))
ax.text(BAND_D / 2 + 11, (yt + ys) / 2, f'cone {C_LEN:.0f}\n(13 inside)', fontsize=8, va='center')
ax.annotate('', (BAND_D / 2 + 9, ys), (BAND_D / 2 + 9, yb), arrowprops=dict(arrowstyle='<->'))
ax.text(BAND_D / 2 + 11, (ys + yb) / 2, f'band {BAND_LEN:.0f}\n2 screw rows', fontsize=8, va='center')
ax.set_xlim(-65, 65); ax.set_ylim(255, 115)

bx = fig.add_axes([0.47, 0.30, 0.30, 0.62]); bx.set_aspect('equal'); bx.axis('off')
bx.set_title('Clamp band, cut through one screw row (from below)', fontsize=12)
th = np.linspace(0, np.pi, 120)
for s in (1, -1):
    o = s * SPLIT / 2
    outer = np.c_[BAND_D / 2 * np.cos(th), s * BAND_D / 2 * np.sin(th) + o]
    inner = np.c_[BORE / 2 * np.cos(th[::-1]), s * BORE / 2 * np.sin(th[::-1]) + o]
    bx.add_patch(Polygon(np.r_[outer, inner], fc='#f5b041', ec='k'))
bx.add_patch(Circle((0, 0), STICK_D / 2, fc='#ccc', ec='k'))
for x in (-SCREW_X, SCREW_X):
    ya = np.sqrt((BAND_D / 2) ** 2 - x ** 2)          # outer surface height at this x
    # screw from the top half (head in a counterbore at the outside) down through the split into the bottom half
    bx.add_patch(Rectangle((x - M4_HEAD_D / 2, ya - 4), M4_HEAD_D, 4 + 1, fc='w', ec='k'))           # head counterbore
    bx.add_patch(Rectangle((x - M4_CLEAR / 2, SPLIT / 2), M4_CLEAR, ya - 4 - SPLIT / 2, fc='w', ec='k', ls='--'))
    bx.add_patch(Rectangle((x - M4_CLEAR / 2, -3 - SPLIT / 2), M4_CLEAR, 3, fc='w', ec='k', ls='--'))      # shoulder (clearance)
    bx.add_patch(Rectangle((x - INS_D / 2, -ya), INS_D, ya - 3 - SPLIT / 2, fc='#aaa', ec='k'))          # insert pocket from outside
    bx.text(x, -ya - 4, 'insert\nin from\noutside', ha='center', va='top', fontsize=7)
bx.text(0, BAND_D / 2 + 3, 'M4 × 16 head side', ha='center', fontsize=8)
bx.text(0, 0, f'stick\nØ{STICK_D:.1f}', ha='center', va='center', fontsize=9)
bx.set_xlim(-34, 34); bx.set_ylim(-42, 36)

cx = fig.add_axes([0.78, 0.06, 0.21, 0.88]); cx.axis('off')
cx.text(0, 1, '\n'.join([
    'v2 changes vs v1:',
    '',
    '• Cone only 27 mm long:',
    '  Ø41 -> Ø45, seats at Ø43',
    '  at the floor underside',
    '  (13 mm inside the tube)',
    '',
    f'• Clamp band below it:',
    f'  Ø{BAND_D:.0f} × {BAND_LEN:.0f} mm, outside',
    '  the box, room for screws',
    '',
    '• 2 rows × 2 screws,',
    f'  M4 × 16 at ±{SCREW_X:.1f} mm',
    '  from the axis',
    '',
    '• Insert in the far half,',
    '  pressed in from outside',
    '  against a 3 mm shoulder:',
    '  screw pulls it INTO the',
    '  part (glue only locates)',
    '',
    f'• Bore Ø{BORE:.1f} (0.3 squeeze),',
    f'  {SPLIT:.0f} mm split gap',
    '',
    '• Halves print flat on',
    '  the split face',
    '',
    'Open:',
    '• M4 insert OD × length',
    '  (calipers; 5.6 × 6 assumed)',
    '• or M3 if you have',
    '  M3 × 16-20 screws',
    '  (band could be Ø52)',
]), va='top', fontsize=10, family='monospace')
plt.savefig(Path(__file__).with_suffix('.png'), dpi=100)
