"""Concept sketch (for approval, not a model): split conical collar on the monopod stick, wedging into the core's
well tube mouth so the stick braces against the core (moment couple: Ø42 ring at the top of the tube + collar at
the bottom), taking bending out of the 1/4-20 screw into the base plate.

    python stick_collar_sketch.py  -> stick_collar_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch

STICK_D = 92 / np.pi            # 29.3
BORE_R, TUBE_T = 21.5, 2.0      # well tube Ø43, 2 mm wall, d 4..180
DEPTH, FLOOR_T = 180.0, 3.0
C_TOP_D, C_BOT_D, C_LEN = 41.0, 47.0, 40.0   # collar cone: Ø41 at the top -> Ø47 at the bottom over 40 mm
SPLIT = 1.0                     # gap between the halves before clamping
SEAT = DEPTH + FLOOR_T          # floor underside
# where the cone reaches Ø43 (wedges in the tube/floor mouth)
z43 = C_LEN * (43.0 - C_TOP_D) / (C_BOT_D - C_TOP_D)
cone_top = SEAT - z43            # cone top depth when seated (the Ø43 line sits at the floor's underside)

fig = plt.figure(figsize=(16, 10))
ax = fig.add_axes([0.04, 0.06, 0.52, 0.88])
ax.set_title('Section through the stick axis (rig depth d down; mm)', fontsize=13)
ax.set_aspect('equal'); ax.invert_yaxis()
# stick + monopod (schematic, from pi_box_bat_rig overlay)
ax.add_patch(Rectangle((-17.5, 20.3), 35, 8, fc='#bbb', ec='k'))           # cap Ø35
ax.add_patch(Rectangle((-21, 28.3), 42, 10, fc='#999', ec='k'))            # ring Ø42
ax.add_patch(Rectangle((-19, 38.3), 38, 155 - 38.3, fc='#ddd', ec='k', hatch='..'))  # foam grip
ax.add_patch(Rectangle((-STICK_D / 2, 155), STICK_D, 90, fc='#ccc', ec='k'))  # bare stick
ax.text(0, 100, 'foam grip\n< Ø40', ha='center', fontsize=9)
ax.text(0, 33.3, 'ring Ø42', ha='center', va='center', fontsize=8)
ax.text(0, 235, f'stick Ø{STICK_D:.1f}', ha='center', fontsize=9)
# core tube + flange + floor
for s in (-1, 1):
    ax.add_patch(Rectangle((s * BORE_R if s > 0 else -BORE_R - TUBE_T, 4), TUBE_T, DEPTH - 4, fc='#7fb3d5', ec='k'))
    ax.add_patch(Rectangle((s * BORE_R if s > 0 else -BORE_R - 30, DEPTH), 30, FLOOR_T, fc='#a9cce3', ec='k'))
ax.add_patch(Rectangle((-55, 0), 110, 4, fc='#7fb3d5', ec='k'))
ax.text(40, 2, 'core flange plate', fontsize=8, va='center')
ax.text(BORE_R + 3, 110, 'well tube\nØ43 bore\n2 mm wall', fontsize=9)
ax.text(BORE_R + 32, DEPTH + 1.5, 'floor (Ø43 hole)', fontsize=8, va='center')
# collar halves (cone)
yt, yb = cone_top, cone_top + C_LEN
for s in (-1, 1):
    pts = [(s * (STICK_D / 2), yt), (s * C_TOP_D / 2, yt), (s * C_BOT_D / 2, yb), (s * (STICK_D / 2), yb)]
    ax.add_patch(Polygon(pts, fc='#f5b041', ec='k', lw=1.2))
    # M3 screw across the split (shown as a dashed line through the wall, front half)
ax.plot([-20, 20], [yb - 8, yb - 8], 'k--', lw=1); ax.plot([-19, 19], [yt + 10, yt + 10], 'k--', lw=1)
ax.text(C_BOT_D / 2 + 3, yb - 8, '2 × M3 across the split\n(per side), nut traps', fontsize=8, va='center')
ax.annotate('', (C_TOP_D / 2, yt - 6), (C_BOT_D / 2, yt - 6), arrowprops=dict(arrowstyle='<->'))
ax.text(-C_BOT_D / 2 - 2, yt + 2, f'Ø{C_TOP_D:.0f}', ha='right', fontsize=9)
ax.text(-C_BOT_D / 2 - 2, yb, f'Ø{C_BOT_D:.0f}', ha='right', fontsize=9)
ax.annotate('', (-C_BOT_D / 2 - 14, yt), (-C_BOT_D / 2 - 14, yb), arrowprops=dict(arrowstyle='<->'))
ax.text(-C_BOT_D / 2 - 16, (yt + yb) / 2, f'{C_LEN:.0f}', ha='right', va='center', fontsize=9)
ax.plot([-30, 30], [SEAT, SEAT], color='r', lw=0.8, ls=':')
ax.text(-58, SEAT + 0.5, f'wedges where the cone = Ø43\n(at the floor underside, {z43:.0f} mm up the cone)', fontsize=8, color='r', va='top')
# load path
ax.add_patch(FancyArrowPatch((-45, 33), (-26, 33), arrowstyle='-|>', mutation_scale=15, color='g'))
ax.add_patch(FancyArrowPatch((45, yt + z43), (24, yt + z43), arrowstyle='-|>', mutation_scale=15, color='g'))
ax.text(-58, 26, 'top contact:\nring in tube', color='g', fontsize=9)
ax.text(30, yt + z43 + 8, 'bottom contact:\ncollar in tube mouth', color='g', fontsize=9)
ax.text(-58, 70, f'lever between\ncontacts ≈ {SEAT - 33:.0f} mm', color='g', fontsize=10)
ax.set_xlim(-62, 62); ax.set_ylim(250, -8)

# plan view of the collar
bx = fig.add_axes([0.6, 0.5, 0.36, 0.42])
bx.set_title('Plan (from below): two halves, split on one plane', fontsize=12)
bx.set_aspect('equal'); bx.axis('off')
th = np.linspace(0, np.pi, 100)
for s in (1, -1):
    o = SPLIT / 2 * s
    outer = np.c_[C_BOT_D / 2 * np.cos(th), s * C_BOT_D / 2 * np.sin(th) + o]
    inner = np.c_[STICK_D / 2 * np.cos(th[::-1]), s * STICK_D / 2 * np.sin(th[::-1]) + o]
    bx.add_patch(Polygon(np.r_[outer, inner], fc='#f5b041', ec='k'))
bx.add_patch(Circle((0, 0), STICK_D / 2 - 0.2, fc='#ccc', ec='k'))
for x in (-19, 19):
    bx.plot([x, x], [-12, 12], 'k--', lw=1)
    bx.add_patch(Rectangle((x - 3, 8), 6, 3, fc='w', ec='k'))     # nut trap
bx.text(0, 0, f'stick\nØ{STICK_D:.1f}', ha='center', va='center', fontsize=9)
bx.text(26, 0, f'{SPLIT:.0f} mm gap\nclosed by\nthe screws', fontsize=8, va='center')
bx.text(-34, -14, 'M3 (×2 per side,\nupper + lower)', fontsize=8)
bx.set_xlim(-30, 40); bx.set_ylim(-28, 28)

cx = fig.add_axes([0.6, 0.06, 0.36, 0.38]); cx.axis('off')
cx.text(0, 1, '\n'.join([
    'Concept (numbers to confirm)',
    f'• Bore Ø{STICK_D - 0.3:.1f} (stick Ø{STICK_D:.1f} - 0.3 squeeze) + {SPLIT:.0f} mm split gap',
    f'• Outer cone Ø{C_TOP_D:.0f} -> Ø{C_BOT_D:.0f} over {C_LEN:.0f} mm (~{np.degrees(np.arctan((C_BOT_D - C_TOP_D) / 2 / C_LEN)):.1f}° half-angle)',
    f'• Clamp on the stick, slide up until the cone seats in the floor/tube mouth (Ø43)',
    '• Ring at the top + collar at the bottom form a couple ~150 mm apart:',
    '  side loads go into the core, not into the 1/4-20 screw',
    '• Wedge load spreads the 2 mm tube mouth: keep the taper gentle,',
    '  only push up until snug (no hammering)',
    '• Resin creeps: re-snug the screws now and then',
    '',
    'To measure: the real bore Ø at the tube mouth (calipers),',
    'and how far up the stick the foam grip ends (bare-stick length).',
]), va='top', fontsize=10, family='monospace')
plt.savefig(Path(__file__).with_suffix('.png'), dpi=100)
print('cone seats with its Ø43 line at d', SEAT, '; cone top at d', round(cone_top, 1))
