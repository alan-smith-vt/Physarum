"""Concept sketch (for approval, not a model): v3 M12 trunk slack (~95 cm, Ø10) on the battery side, 2026-10-17.

Full-height ellipses standing in the Yr-depth plane in the side bays between the web and the bank (the gap behind
the tube itself is only ~7 mm). U-turns 25 mm outside (bend tested by the user). The tube limits the +X bay to
X 23.5..46 (22.5): 2 turns there, 1 turn in the -X bay. Breakout leads and their slack stay on the -Y (board) side.
Split point and splitter size not measured yet (packed in the v2 rig).

    python pi_box_v3_trunk_sketch.py -> pi_box_v3_trunk_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Ellipse

X_IN, X_OUT, Y_MIN, Y_MAX, DEPTH, FLANGE = 46.0, 50.0, -62.0, 62.0, 180.0, 10.0
WELL_R, TUBE_T, WEB_T = 21.5, 2.0, 3.0
BAT = dict(x=37.5, y0=30.8, y1=55.8, d0=22.0)
CORD = 10.0
E_YR = (3.5, 28.5)                     # trunk ellipse outer Yr band (web face 1.5 + 2 clear .. bank 30.8 - 2.3)
E_D = (14.0, 176.0)                    # outer depth band (flange 10 .. floor 180)
TURNS = {+1: 2, -1: 1}
R_TUBE = WELL_R + TUBE_T
x_tube = np.sqrt(R_TUBE ** 2 - E_YR[0] ** 2)          # tube surface at the coil's nearest Yr

b = (E_YR[1] - E_YR[0] - CORD) / 2     # centreline semi-axes
a = (E_D[1] - E_D[0] - CORD) / 2
per = np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b)))
total = sum(TURNS.values()) * per

fig = plt.figure(figsize=(17, 10.5))
ax = fig.add_axes([0.03, 0.08, 0.40, 0.84]); ax.set_aspect('equal')
ax.set_title('Section at depth 100, looking down (X across, Yr up)', fontsize=13)
ax.add_patch(Rectangle((-X_OUT, Y_MIN), 2 * X_OUT, Y_MAX - Y_MIN, fc='none', ec='k', lw=2))
ax.add_patch(Circle((0, 0), R_TUBE, fc='#bbb', ec='k')); ax.add_patch(Circle((0, 0), WELL_R, fc='w', ec='k'))
for s in (-1, 1):
    ax.add_patch(Rectangle((R_TUBE if s > 0 else -X_IN, -WEB_T / 2), X_IN - R_TUBE, WEB_T, fc='#bbb', ec='k'))
ax.add_patch(Rectangle((-BAT['x'], BAT['y0']), 2 * BAT['x'], BAT['y1'] - BAT['y0'], fc='#fd7', ec='k'))
ax.text(0, 43, 'battery bank', ha='center', va='center')
for s, n in TURNS.items():
    for i in range(n):
        xc = s * (X_IN - CORD / 2 - i * CORD)
        for yc in (E_YR[0] + CORD / 2, E_YR[1] - CORD / 2):
            ax.add_patch(Circle((xc, yc), CORD / 2, fc='#1f4e79', ec='k', alpha=0.85))
    ax.text(s * 35, 16, f'{n} turn{"s" if n > 1 else ""}', ha='center', va='center', fontsize=10, color='#1f4e79', weight='bold')
ax.add_patch(Rectangle((x_tube, E_YR[0]), X_IN - x_tube, E_YR[1] - E_YR[0], fc='none', ec='#1f4e79', ls=':'))
ax.annotate('', (x_tube, -8), (X_IN, -8), arrowprops=dict(arrowstyle='<->'))
ax.text((x_tube + X_IN) / 2, -12, f'{X_IN - x_tube:.1f} (tube limits)', ha='center', fontsize=8)
ax.annotate('', (-5, R_TUBE), (-5, BAT['y0']), arrowprops=dict(arrowstyle='<->', color='r'))
ax.text(-7, 27, f'{BAT["y0"] - R_TUBE:.1f}: no room\nbehind the tube', ha='right', fontsize=8, color='r')
ax.text(0, -40, '-Y side: Pi, power board, UE300,\nbreakout leads + their slack', ha='center', fontsize=10, color='#285')
ax.set_xlim(-55, 55); ax.set_ylim(Y_MIN - 3, Y_MAX + 3)

bx = fig.add_axes([0.45, 0.08, 0.22, 0.84]); bx.set_aspect('equal'); bx.invert_yaxis()
bx.set_title('+X bay seen from +X (Yr across, depth down)', fontsize=12)
bx.add_patch(Rectangle((-2, 0), 66, FLANGE, fc='#bbb', ec='k')); bx.text(31, 6, 'flange', ha='center', fontsize=8)
bx.add_patch(Rectangle((-WEB_T / 2, FLANGE), WEB_T, DEPTH - FLANGE, fc='#bbb', ec='k'))
bx.add_patch(Rectangle((BAT['y0'], BAT['d0']), BAT['y1'] - BAT['y0'], DEPTH - BAT['d0'], fc='#fd7', ec='k'))
bx.text((BAT['y0'] + BAT['y1']) / 2, 100, 'bank', ha='center', rotation=90)
bx.add_patch(Rectangle((Y_MAX - 1.5 - 3.2, FLANGE), 3.2, DEPTH - FLANGE, fc='#cde', ec='k'))
bx.add_patch(Rectangle((-2, DEPTH), 66, 3, fc='#9cf', ec='k'))
yc, dc = (E_YR[0] + E_YR[1]) / 2, (E_D[0] + E_D[1]) / 2
bx.add_patch(Ellipse((yc, dc), E_YR[1] - E_YR[0], E_D[1] - E_D[0], fc='none', ec='#1f4e79', lw=CORD * 1.6, alpha=0.5))
bx.add_patch(Ellipse((yc, dc), 2 * b, 2 * a, fc='none', ec='#1f4e79', lw=1, ls='--'))
bx.plot([58, 45, 30, yc + b], [9, 15, 16, 30], color='#1f4e79', lw=4, alpha=0.7)
bx.text(40, 26, 'from flange notch,\nover the bank top', fontsize=7, color='#1f4e79')
bx.annotate('to -Y side via\na web slot →\n(split point TBD)', (0, 87), (12, 120), fontsize=7, arrowprops=dict(arrowstyle='->'))
bx.text(yc, E_D[1] + 6, f'U-turn {E_YR[1] - E_YR[0]:.0f} outside', ha='center', fontsize=8)
bx.set_xlim(-4, 66); bx.set_ylim(DEPTH + 10, -4)

fig.text(0.70, 0.88, 'M12 trunk slack (v3)', fontsize=15, weight='bold')
fig.text(0.70, 0.30, '\n'.join([
    f'Ellipse turn (centreline {2 * b:.0f} x {2 * a:.0f}):',
    f'  {per:.0f} mm per turn',
    f'  +X bay {TURNS[1]} turns, -X bay {TURNS[-1]} turn',
    f'  holds ~{total / 10:.0f} cm (need ~95)',
    '',
    '• Only the side bays work: the gap',
    '  between the tube and the bank is',
    f'  {BAT["y0"] - R_TUBE:.1f} mm, under the Ø10 cord.',
    f'• +X bay is {X_IN - x_tube:.1f} wide (the tube',
    '  limits it): 2 turns, not 3.',
    '• Enters at the flange notch, crosses',
    '  over the bank top (12 mm gap,',
    '  shared with the bank cord root).',
    '• Leaves low through a web slot to',
    '  the -Y side, where the breakout',
    '  leads take their own slack.',
    '',
    'Open (v2 rig is packed):',
    '  - split point from the M12 plug',
    '  - splitter body size vs web slot',
    '  - bank cord route to the floor socket',
    '    (now shares the bays)',
]), fontsize=11, va='bottom', family='monospace')
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110)
print(out)
