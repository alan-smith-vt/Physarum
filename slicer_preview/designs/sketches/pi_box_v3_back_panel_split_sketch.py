"""Concept sketch (for approval, not a model): v3 back (+Y) panel split in two, 2026-10-18.

Both pieces slide up the same L-hook channels (open at the floor end): the TOP piece carries a 2.42" SSD1309 OLED
(no-phone backup display) inset from the outer face, the LOWER piece carries the MagSafe ring + bank window and
holds the top piece up against the rim. There is only ~1.2 mm between the panel's inner face and the bank (Y 55.8),
so the module is inset from OUTSIDE: PCB in a pocket, glass proud, back parts + wires through the pocket floor into
the gap, wires over the bank top (d 14..22, beside its cord) to the upper web slot (|X| 30..41, d 24..34) -> Pi.
Module sizes are [MEASURE] placeholders. The phone (portrait) moves down below the display and overhangs the floor.

    python pi_box_v3_back_panel_split_sketch.py -> pi_box_v3_back_panel_split_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

X_IN, D0, D1 = 46.0, 14.1, 179.9          # panel spans the wall gap, rim to the floor (pi_box_bat_panel_v3)
HOOK = 2.6                                # neck + hook past the wall line
FRAME = 5.0
SPLIT = 66.0                              # joint between the top and lower pieces
PCB = (71.0, 43.0)                        # 2.42" SSD1309 module [MEASURE]
GLASS = (60.5, 37.0)                      # glass outline [MEASURE]
ACTIVE = (55.0, 27.5)                     # active area [MEASURE]
MOD_C = (0.0, (D0 + SPLIT) / 2)           # module centre: X, d
HOLE_INSET = 2.5                          # M2 holes from the PCB corners [MEASURE]
PHONE = (72.0, 147.0)
RING_OD, RING_ID = 56.0, 46.0
RING_C = (0.0, SPLIT + FRAME + RING_OD / 2 + 1.5)   # disc as high as the lower piece allows
PHONE_TOP = RING_C[1] - 62.0              # same ring-to-phone-top offset as before (d 88 vs top d 26)
RAISE = 2.5                               # ring disc stands proud so the phone clears the display glass (~2 proud)
WIN = (-40.0, 12.0, 142.0, 174.9)         # bank display + button window [MEASURE display]
BUTTON = (-22.0, 158.0)
FLOOR_D = 183.0                           # floor underside (approx.)

fig = plt.figure(figsize=(17, 11))
ax = fig.add_axes([0.03, 0.05, 0.45, 0.88]); ax.set_aspect('equal'); ax.invert_yaxis(); ax.invert_xaxis()
ax.set_title('Back (+Y) panel from outside, split (rig X mirrored, depth down, mm)', fontsize=13)
for (a, b), fc, lab in (((D0, SPLIT), '#dfe9d8', 'TOP piece'), ((SPLIT, D1), '#e8eef7', 'LOWER piece')):
    ax.add_patch(Rectangle((-X_IN - HOOK, a), 2 * (X_IN + HOOK), b - a, fc=fc, ec='k', lw=1.5))
    ax.text(X_IN - 1, a + 4, lab, fontsize=9, weight='bold', va='top')
ax.plot([-X_IN - 8, X_IN + 8], [SPLIT, SPLIT], 'r--', lw=1.2)
ax.text(-X_IN - 9, SPLIT, f'joint d {SPLIT:.0f}', color='r', fontsize=8, va='center', ha='left')
ax.add_patch(Rectangle((-X_IN + FRAME, SPLIT + FRAME), 2 * (X_IN - FRAME), D1 - SPLIT - 2 * FRAME,
                       fc='w', ec='#888', ls=':'))
ax.text(-X_IN + FRAME + 1, SPLIT + 30, 'open frame +\nbraces (as now)', fontsize=7, color='#666', ha='right')

cx, cd = MOD_C
ax.add_patch(Rectangle((cx - PCB[0] / 2, cd - PCB[1] / 2), *PCB, fc='#2b5d9b', ec='k', alpha=0.85))
ax.add_patch(Rectangle((cx - GLASS[0] / 2, cd - GLASS[1] / 2), *GLASS, fc='#111', ec='k'))
ax.add_patch(Rectangle((cx - ACTIVE[0] / 2, cd - ACTIVE[1] / 2 - 2), *ACTIVE, fc='#0a2a40', ec='#5af'))
ax.text(cx, cd - 2, 'LOST TRACKING\n(+ mini map)', color='#9cf', ha='center', va='center', fontsize=9, family='monospace')
for sx in (-1, 1):
    for sd in (-1, 1):
        ax.add_patch(Circle((cx + sx * (PCB[0] / 2 - HOLE_INSET), cd + sd * (PCB[1] / 2 - HOLE_INSET)), 1.1, fc='w', ec='k'))
ax.text(cx, cd + PCB[1] / 2 + 3.5, '2.42" SSD1309: PCB in an outer-face pocket, 4x M2 self-tap', ha='center', fontsize=8)
ax.annotate('wires: through the pocket floor,\nover the bank top (d 14..22),\nweb slot d 24..34 -> Pi',
            xy=(-30, D0 + 4), xytext=(-75, 30), fontsize=8, color='#d60', arrowprops=dict(arrowstyle='->', color='#d60'))

ax.add_patch(Rectangle((WIN[0], WIN[2]), WIN[1] - WIN[0], WIN[3] - WIN[2], fc='#fd7', ec='#a80', lw=1.5, alpha=0.6))
ax.text((WIN[0] + WIN[1]) / 2, WIN[3] - 4, 'bank window [MEASURE]', ha='center', fontsize=8)
ax.add_patch(Circle(BUTTON, 5, fc='#f96', ec='k'))
ax.add_patch(Circle(RING_C, RING_OD / 2 + 1.5, fc='#e8eef7', ec='k', alpha=0.9))
ax.add_patch(Circle(RING_C, RING_OD / 2, fc='#bcd', ec='k', hatch='///'))
ax.add_patch(Circle(RING_C, RING_ID / 2, fc='#e8eef7', ec='k'))
ax.text(RING_C[0], RING_C[1], f'MagSafe ring\nd {RING_C[1]:.0f}', ha='center', va='center', fontsize=9)
ax.add_patch(Rectangle((-PHONE[0] / 2, PHONE_TOP), *PHONE, fc='none', ec='#c0c', ls='--', lw=2))
ax.text(0, PHONE_TOP + 3, 'phone covers the display when fitted', color='#c0c', fontsize=8, ha='center', va='top')
ax.text(-PHONE[0] / 2 + 1, PHONE_TOP + PHONE[1] - 3, f'phone 72 x 147\nd {PHONE_TOP:.0f}..{PHONE_TOP + PHONE[1]:.0f}',
        color='#c0c', fontsize=8, ha='right')
ax.add_patch(Rectangle((-X_IN - HOOK - 2, D1), 2 * (X_IN + HOOK + 2), FLOOR_D - D1, fc='#9cf', ec='k'))
ax.text(X_IN + 6, FLOOR_D + 3, 'floor', fontsize=8, va='top')
ax.add_patch(Rectangle((-X_IN - HOOK - 2, -6), 2 * (X_IN + HOOK + 2), D0 + 6, fc='#ccc', ec='k'))
ax.text(0, 4, 'flange / rim', ha='center', fontsize=8)
ax.set_xlim(85, -85); ax.set_ylim(PHONE_TOP + PHONE[1] + 10, -10)

# section through the module (Y across, depth down)
sx = fig.add_axes([0.50, 0.50, 0.22, 0.42]); sx.set_aspect('equal'); sx.invert_yaxis()
sx.set_title('Section at X 0 (Y across, depth down)', fontsize=10)
P_OUT, P_IN, BANK = 60.4, 57.2, 55.8                   # panel outer / inner face, bank face (approx.)
sx.add_patch(Rectangle((P_IN, D0), P_OUT - P_IN, SPLIT - D0, fc='#dfe9d8', ec='k'))
sx.add_patch(Rectangle((P_IN, SPLIT), P_OUT - P_IN, 30, fc='#e8eef7', ec='k'))
sx.add_patch(Rectangle((30.8, 22.0), BANK - 30.8, 74, fc='#34495e', ec='k')); sx.text(43, 60, 'bank', color='w', ha='center')
sx.add_patch(Rectangle((P_OUT - 1.6, cd - PCB[1] / 2), 1.6, PCB[1], fc='#2b5d9b', ec='k'))
sx.add_patch(Rectangle((P_OUT, cd - GLASS[1] / 2), 2.0, GLASS[1], fc='#111', ec='k'))
sx.add_patch(Rectangle((P_IN - 1.0, cd - 10), 2.6, 8, fc='#f90', ec='k')); sx.text(P_IN - 1.5, cd - 6, 'back parts\n-> 1.2 gap', fontsize=7, ha='right')
sx.plot([P_IN - 0.6, 40, 32], [cd - 14, D0 + 4, D0 + 4], color='#d60', lw=1.5)
sx.text(36, D0 + 2, 'wires', color='#d60', fontsize=7)
sx.add_patch(Rectangle((P_OUT, SPLIT + FRAME), RAISE, 30, fc='#bcd', ec='k'))
sx.text(P_OUT + RAISE + 0.5, SPLIT + 20, 'raised\nring disc', fontsize=7)
sx.plot([P_OUT + RAISE + 0.3] * 2, [PHONE_TOP, SPLIT + 30], color='#c0c', lw=2.5, ls='--')
sx.text(P_OUT + RAISE + 0.8, PHONE_TOP + 4, 'phone', color='#c0c', fontsize=7)
sx.set_xlim(28, 70); sx.set_ylim(SPLIT + 30, 5)

fig.text(0.50, 0.04, '\n'.join([
    'Back panel v3, split in two',
    '• Both pieces in the same L-hook channels: TOP slides in first up to the rim, LOWER behind it holds it up.',
    '  Removing the LOWER piece leaves the display in place; TOP comes out only for maintenance.',
    '• Only ~1.2 mm between panel and bank, so the module goes in from OUTSIDE: PCB pocket ~1.6 deep in the',
    '  3.2 mm panel, through-cuts for its back parts + wire pads, glass ~2 mm proud, 4x M2 self-tap screws.',
    '• Wires: solder to the pads, through the pocket floor, over the bank top (d 14..22, beside its cord),',
    '  upper web slot (|X| 30..41, d 24..34) to Pi 3V3/GND/SDA/SCL. I2C 0x3C, no clash with INA226 0x40.',
    f'• Phone REPLACES the display when fitted: ring as high as the lower piece allows (centre d {RING_C[1]:.1f},',
    f'  disc d {RING_C[1] - 29.5:.0f}..{RING_C[1] + 29.5:.0f}); phone d {PHONE_TOP:.0f}..{PHONE_TOP + PHONE[1]:.0f}, ~{PHONE_TOP + PHONE[1] - FLOOR_D:.0f} mm past the floor, covers the display.',
    f'  Ring disc raised {RAISE} mm so the phone clears the ~2 mm proud glass. Disc clears the bank window (d 142).',
    '',
    'To confirm / measure: module PCB, glass, holes, back-part height; bank display; phone model.',
]), fontsize=8.5, va='bottom', family='monospace')
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=105)
print(out)
