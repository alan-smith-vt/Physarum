"""Concept sketch (for approval, not a model): v3 floor quick release (warm battery swap), 2026-10-17.

The v2 floor stays (nubs, rails, collar and the UE300 cradle locate it). Instead of 4 M3 screws: two off-the-shelf
mini stainless draw (toggle) latches on the outside of the ±X walls at Yr 0, where the web feet make the wall solid
at the bottom; the floor grows an ear up the wall's outer face at each latch to carry the catch. Latch size is a
generic ~30 x 12 mini draw latch [MEASURE the one bought].

    python pi_box_v3_floor_latch_sketch.py -> pi_box_v3_floor_latch_sketch.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle

X_IN, X_OUT, Y_MIN, Y_MAX, DEPTH = 46.0, 50.0, -62.0, 62.0, 180.0
FLOOR_T = 3.0
FOOT_Y, FOOT_H = 9.0, 10.0
EAR_T, EAR_H, EAR_W = 3.0, 16.0, 22.0          # floor ear up the wall's outer face (X 50..53)
LATCH = dict(w=12.0, base=18.0, arm=14.0)      # body on the wall (base along d), hook arm reaching down to the catch
CATCH_H = 8.0

fig = plt.figure(figsize=(17, 10))

# ── front view (from -Y), +X side, bottom of the box ──
ax = fig.add_axes([0.03, 0.08, 0.30, 0.84]); ax.set_aspect('equal'); ax.invert_yaxis()
ax.set_title('+X side from -Y (X across, depth down)', fontsize=12)
ax.add_patch(Rectangle((X_IN, 120), X_OUT - X_IN, DEPTH - 120, fc='#bbb', ec='k'))
ax.add_patch(Rectangle((0, DEPTH), X_OUT, FLOOR_T, fc='#e67e22', ec='k'))
ax.add_patch(Rectangle((X_OUT, DEPTH - EAR_H), EAR_T, EAR_H + FLOOR_T, fc='#e67e22', ec='k'))
ax.text(X_OUT + EAR_T + 1, DEPTH - 3, 'floor ear', fontsize=9, color='#a04000')
b0 = DEPTH - EAR_H - 4 - LATCH['base']
ax.add_patch(Rectangle((X_OUT, b0), 6, LATCH['base'], fc='#7f8c8d', ec='k'))
ax.text(X_OUT + 7, b0 + 6, 'latch body\n(2 × M3 into\ninserts)', fontsize=9)
ax.plot([X_OUT + 6, X_OUT + 8, X_OUT + EAR_T + 2.5], [b0 + LATCH['base'], DEPTH - CATCH_H - 2, DEPTH - CATCH_H], color='k', lw=2)
ax.add_patch(Rectangle((X_OUT + EAR_T, DEPTH - CATCH_H - 3), 3, 5, fc='#bdc3c7', ec='k'))
ax.text(X_OUT + EAR_T + 4, DEPTH - CATCH_H - 0.5, 'catch', fontsize=9)
ax.add_patch(Rectangle((X_IN - 16, DEPTH - 71), 16, 71, fc='none', ec='#36c', ls='--')); ax.text(X_IN - 15, DEPTH - 60, 'UE300\n(inside)', fontsize=8, color='#36c')
ax.add_patch(Rectangle((-23.5, 120), 2.0, DEPTH - 120, fc='#bbb', ec='k')); ax.text(-20, 125, 'tube', fontsize=8)
ax.set_xlim(-26, 70); ax.set_ylim(DEPTH + 8, 115)

# ── side view from +X: where the two latches sit ──
bx = fig.add_axes([0.35, 0.08, 0.32, 0.84]); bx.set_aspect('equal'); bx.invert_yaxis()
bx.set_title('+X wall from outside (Yr across, depth down)', fontsize=12)
bx.add_patch(Rectangle((Y_MIN, 100), Y_MAX - Y_MIN, DEPTH - 100, fc='#ddd', ec='k'))
bx.add_patch(Rectangle((-FOOT_Y, DEPTH - FOOT_H - 8), 2 * FOOT_Y, FOOT_H + 8, fc='#aaa', ec='k', hatch='..'))
bx.text(0, DEPTH - FOOT_H - 10, 'web foot (solid)', ha='center', fontsize=8)
bx.add_patch(Rectangle((Y_MIN, DEPTH), Y_MAX - Y_MIN, FLOOR_T, fc='#e67e22', ec='k'))
bx.add_patch(Rectangle((-EAR_W / 2, DEPTH - EAR_H), EAR_W, EAR_H, fc='#e67e22', ec='k', alpha=0.9))
bx.add_patch(Rectangle((-LATCH['w'] / 2, b0), LATCH['w'], LATCH['base'], fc='#7f8c8d', ec='k'))
for yy in (-3.5, 3.5):
    bx.add_patch(Circle((yy, b0 + 4), 1.2, fc='w', ec='k')); bx.add_patch(Circle((yy, DEPTH - CATCH_H), 1.0, fc='w', ec='k'))
for by in (Y_MIN + 10.5, Y_MAX - 10.5):
    bx.add_patch(Circle((by, DEPTH + 1.5), 2.2, fc='w', ec='k', ls='--'))
bx.text(Y_MIN + 10.5, DEPTH + 9, 'M3 bosses: kept,\nscrews optional', ha='center', fontsize=8)
bx.add_patch(Rectangle((-7, DEPTH - 37), 14, 4, fc='none', ec='#c33', ls=':')); bx.text(9, DEPTH - 34, 'solid pad in the gyroid wall', fontsize=8, color='#c33')
bx.set_xlim(Y_MIN - 4, Y_MAX + 4); bx.set_ylim(DEPTH + 14, 95)

fig.text(0.69, 0.92, 'Floor quick release (v3)', fontsize=15, weight='bold')
fig.text(0.69, 0.08, '\n'.join([
    'Recommended: 2 mini stainless draw latches',
    '  • one per ±X wall at Yr 0 (web foot = solid',
    '    wall there), body on the wall, catch on a',
    '    floor ear (3 thick, up the outer face)',
    '  • nubs + rails + collar still locate; the',
    '    latches only pull the floor up',
    '  • swap: flip 2 latches, drop the floor',
    '  • latch body: 2 × M3 into inserts in a solid',
    '    pad; catch: 2 × M2/M3 into the ear',
    '  • M3 boss inserts stay: 4 screws for',
    '    transport / backup',
    '',
    'Alternatives',
    '  B  4 captive M3 thumb screws in the existing',
    '     inserts: no design change, ~20 s, no tool',
    '  C  quarter-turn studs at the bosses: need',
    '     ~Ø10 receptacles, bigger than the bosses',
    '  D  4 latches (one per end + sides): stiffer',
    '     ends, more ears',
    '',
    'To check: latch model + size [MEASURE],',
    'ear vs the panel channels (Yr 0: clear),',
    'UE300 inside the +X wall (inserts 3 deep in',
    'a 4 mm wall: the inner face stays closed).',
]), fontsize=10.5, va='bottom', family='monospace')
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110)
print(out)
