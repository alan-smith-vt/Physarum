"""Concept sketch (for approval, not a model): v3 core layout assuming the power board works (2026-10-17).

Cameras dropped (cartridges, ribbons, shrouds gone). The 46 x 60 power board stands in the -Y bay under the Pi,
parallel to it, on an M2.5 post from the well tube and a bracket from the +X wall, parts facing the -Y panel. It replaces the PD trigger,
buck, WAGOs and slack bundle. Box, tube, webs, bank pod and Pi position (PI_SHIFT +6) are unchanged.

    python pi_box_v3_layout_sketch.py -> pi_box_v3_layout_sketch.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyArrowPatch

X_IN, X_OUT, Y_MIN, Y_MAX, DEPTH = 46.0, 50.0, -62.0, 62.0, 180.0
WELL_R, TUBE_T, FLANGE = 21.5, 2.0, 10.0
PI_SHIFT, PI_W, PI_L, PI_TOP, PCB_YR = 6.0, 56.0, 85.0, 14.0, -31.0
BAT = dict(x=37.5, y0=30.8, y1=55.8, d0=22.0)
BRD_W, BRD_H, BRD_TOP = 46.0, 60.0, 112.0          # board: 46 along X (centred), 60 along depth
BRD_X0 = -1.0                                      # against the +X wall: the switch body ends at X -6 (5 mm clear)
BRD_YR, POST_L = -33.0, 9.0                        # board back face Yr; post from the tube face (Yr -23.5)
SWITCH = dict(d=155.0, yr=-40.1)
UE = dict(x0=30.15, x1=46.15, y0=-27.6, y1=-1.6, d0=109.0)   # UE300 as in v2 (ETH_ADAPTER): on the floor, RJ45 down through it
COIL = dict(c=(0.0, 60.0), r=(42.5, 32.5), yr=(-56.0, -36.0))   # M12 slack (~95 cm, Ø10): 2 layers x 2 loops behind the Pi
SOCKET = dict(x=-20.0, yr=43.0, d=176.0)                     # bank USB-C: panel-mount socket at the chamber's floor end

fig = plt.figure(figsize=(18, 10.5))

# ── front view: looking at the -Y face (panel removed), X across, depth down ──
ax = fig.add_axes([0.03, 0.05, 0.47, 0.88]); ax.set_aspect('equal'); ax.invert_yaxis()
ax.set_title('Front view from -Y, panel off (X across, depth down, mm)', fontsize=13)
ax.add_patch(Rectangle((-X_OUT, 0), 2 * X_OUT, FLANGE, fc='#bbb', ec='k')); ax.text(0, 6, 'flange', ha='center', fontsize=9)
for s in (-1, 1):
    ax.add_patch(Rectangle((s * X_IN if s > 0 else -X_OUT, FLANGE), X_OUT - X_IN, DEPTH - FLANGE, fc='#bbb', ec='k'))
ax.add_patch(Rectangle((-X_OUT, DEPTH), 2 * X_OUT, 3, fc='#9cf', ec='k')); ax.text(0, DEPTH + 7, 'sliding floor + quick-release latches', ha='center', fontsize=9)
ax.add_patch(Rectangle((-WELL_R - TUBE_T, FLANGE), 2 * (WELL_R + TUBE_T), DEPTH - FLANGE, fc='none', ec='#888', ls='--'))
ax.text(0, 46, 'well tube\n(behind)', ha='center', fontsize=8, color='#666')
px0 = PI_SHIFT - PI_W / 2
ax.add_patch(Rectangle((px0, PI_TOP), PI_W, PI_L, fc='#7c7', ec='k', alpha=0.85))
ax.text(PI_SHIFT, PI_TOP + 40, 'Pi 5\n(unchanged, +6 X)', ha='center', fontsize=10)
ax.add_patch(Rectangle((px0 - 9, PI_TOP + 8), 9, 9, fc='#f96', ec='k')); ax.text(px0 - 10, PI_TOP + 5, 'R/A USB-C', ha='right', fontsize=8)
ax.add_patch(Rectangle((px0 + PI_W - 5, PI_TOP + 7), 5, 51, fc='#333', ec='k')); ax.text(px0 + PI_W + 1, PI_TOP + 30, 'GPIO', fontsize=8, rotation=90)
ax.add_patch(Rectangle((BRD_X0, BRD_TOP), BRD_W, BRD_H, fc='#285', ec='k'))
ax.text(BRD_X0 + BRD_W / 2, BRD_TOP + 30, 'POWER BOARD\n46 x 60', ha='center', color='w', fontsize=11, weight='bold')
for x, lab in ((BRD_X0 + 10, 'J2 → Pi'), (BRD_X0 + 36, 'J1 ← bank')):
    ax.add_patch(Rectangle((x - 4.5, BRD_TOP - 1.5), 9, 3.5, fc='#ccc', ec='k')); ax.text(x, BRD_TOP - 4, lab, ha='center', fontsize=8)
ax.add_patch(Rectangle((BRD_X0 + 30, BRD_TOP + BRD_H - 9), 9, 8, fc='#2a2', ec='k')); ax.text(BRD_X0 + 40, BRD_TOP + BRD_H + 5, 'J3 lidar 15 V', fontsize=8)
ax.add_patch(Rectangle((BRD_X0 + 7, BRD_TOP + BRD_H - 6), 21, 5, fc='w', ec='k')); ax.text(BRD_X0 + 2, BRD_TOP + BRD_H + 5, 'J5 → Pi GPIO', fontsize=8)
ax.add_patch(Rectangle((BRD_X0 + 1, BRD_TOP + 12), 5, 10, fc='w', ec='k')); ax.text(BRD_X0 - 1, BRD_TOP + 17, 'J4', ha='right', fontsize=8)
for hx, hy in ((3, BRD_H - 3), (BRD_W - 3, 47)):
    ax.add_patch(Circle((BRD_X0 + hx, BRD_TOP + hy), 1.6, fc='w', ec='k'))
ax.text(BRD_X0 + 46, BRD_TOP + 47, ' bracket', fontsize=7, va='center')
ax.add_patch(Rectangle((-X_OUT, SWITCH['d'] - 8), 4, 16, fc='#c33', ec='k')); ax.text(-X_IN + 1, SWITCH['d'] + 12, 'switch\n(kept)', fontsize=8)
arr = dict(arrowstyle='-|>', lw=1.6, mutation_scale=12)
ax.add_patch(FancyArrowPatch((BRD_X0 + 10, BRD_TOP - 2), (px0 - 5, PI_TOP + 17), connectionstyle='arc3,rad=-0.3', color='#d60', **arr))
ax.add_patch(FancyArrowPatch((BRD_X0 + 17, BRD_TOP + BRD_H - 4), (px0 + PI_W - 3, PI_TOP + 55), connectionstyle='arc3,rad=0.55', color='#06c', **arr))
ax.add_patch(FancyArrowPatch((40, 95), (BRD_X0 + 36, BRD_TOP - 2), connectionstyle='arc3,rad=0.2', color='#a0a', **arr))
ax.text(41, 88, 'bank cord\n(web slot)', fontsize=8, color='#a0a')
ax.add_patch(FancyArrowPatch((-X_IN + 4, SWITCH['d']), (BRD_X0 + 1, BRD_TOP + 17), connectionstyle='arc3,rad=-0.3', color='#c33', **arr))
ax.add_patch(FancyArrowPatch((BRD_X0 + 35, BRD_TOP + BRD_H), (8, 14), connectionstyle='arc3,rad=-0.9', color='#2a2', **arr))
ax.text(30, 30, 'lidar power pair\n(from flange notch)', fontsize=8, color='#2a2')
ax.add_patch(Rectangle((UE['x0'], UE['d0']), UE['x1'] - UE['x0'] - 0.15, DEPTH - UE['d0'], fc='none', ec='#36c', ls='--', lw=1.5))
ax.text(UE['x0'] + 1, UE['d0'] - 2, 'UE300 (behind board)', color='#36c', fontsize=8)
ax.add_patch(FancyArrowPatch((UE['x0'] + 8, UE['d0']), (PI_SHIFT + 10, PI_TOP + PI_L), connectionstyle='arc3,rad=0.3', color='#36c', **arr))
ax.text(UE['x0'] + 9, DEPTH + 11, 'RJ45 out the floor', color='#36c', fontsize=8)
from matplotlib.patches import Ellipse
for r in COIL['r']:
    ax.add_patch(Circle(COIL['c'], r, fc='none', ec='#1f4e79', lw=4, alpha=0.55))
ax.text(COIL['c'][0], COIL['c'][1] + 12, 'M12 slack coil\n(behind the Pi)', ha='center', color='#1f4e79', fontsize=9, weight='bold')
ax.set_xlim(-62, 70); ax.set_ylim(DEPTH + 14, -6)

# ── section at depth 150, looking down: Yr stacking ──
bx = fig.add_axes([0.53, 0.42, 0.45, 0.52]); bx.set_aspect('equal')
bx.set_title('Section at depth 150, looking down (X across, Yr up)', fontsize=13)
bx.add_patch(Rectangle((-X_OUT, Y_MIN), 2 * X_OUT, Y_MAX - Y_MIN, fc='none', ec='k', lw=2))
for yy in (Y_MIN + 1.5, Y_MAX - 1.5 - 3.2):
    bx.add_patch(Rectangle((-X_IN, yy), 2 * X_IN, 3.2, fc='#cde', ec='k'))
bx.text(0, Y_MIN + 6.5, '-Y panel (T-edge in T-channel)', ha='center', fontsize=8)
bx.add_patch(Circle((0, 0), WELL_R + TUBE_T, fc='#bbb', ec='k')); bx.add_patch(Circle((0, 0), WELL_R, fc='w', ec='k'))
bx.text(0, 0, 'stick', ha='center', va='center', fontsize=8)
for s in (-1, 1):
    bx.add_patch(Rectangle((s * (WELL_R + TUBE_T) if s > 0 else -X_IN, -1.5), X_IN - WELL_R - TUBE_T, 3, fc='#bbb', ec='k'))
bx.add_patch(Rectangle((-BAT['x'], BAT['y0']), 2 * BAT['x'], BAT['y1'] - BAT['y0'], fc='#fd7', ec='k'))
bx.text(0, (BAT['y0'] + BAT['y1']) / 2, 'battery bank', ha='center', va='center')
bx.add_patch(Rectangle((BRD_X0, BRD_YR - 1.6), BRD_W, 1.6, fc='#285', ec='k'))
bx.text(BRD_X0 + BRD_W / 2, BRD_YR - 6, 'power board, parts facing -Y\n(J3 plug + wires need ~20)', ha='center', va='top', fontsize=8)
bx.add_patch(Rectangle((BRD_X0 + 1.2, BRD_YR), 3.5, -BRD_YR - WELL_R - TUBE_T + 0.3, fc='#999', ec='k'))
bx.text(BRD_X0 + 5.5, -27, 'post from tube', fontsize=7)
bx.add_patch(Rectangle((BRD_X0 + 41, BRD_YR), X_IN - BRD_X0 - 41, 4, fc='#999', ec='k'))
bx.text(BRD_X0 + 26, BRD_YR + 6, 'bracket from +X wall', fontsize=7)
bx.annotate('', (BRD_X0 - 3, Y_MIN + 4.7), (BRD_X0 - 3, BRD_YR - 1.6), arrowprops=dict(arrowstyle='<->'))
bx.text(BRD_X0 - 5, (Y_MIN + BRD_YR) / 2, f'{BRD_YR - 1.6 - Y_MIN - 4.7:.0f}', ha='right', va='center')
bx.add_patch(Rectangle((-X_OUT, SWITCH['yr'] - 8), 40, 16, fc='none', ec='#c33', ls='--'))
bx.text(-X_IN + 2, SWITCH['yr'] + 9, 'switch body (d 147-163)', color='#c33', fontsize=7)
bx.add_patch(Rectangle((UE['x0'], UE['y0']), UE['x1'] - UE['x0'] - 0.15, UE['y1'] - UE['y0'], fc='#9bd', ec='#36c'))
bx.text((UE['x0'] + UE['x1']) / 2, (UE['y0'] + UE['y1']) / 2, 'UE300', ha='center', va='center', fontsize=8, rotation=90)
bx.add_patch(Rectangle((SOCKET['x'] - 6, SOCKET['yr'] - 4), 12, 8, fc='#f6c', ec='k'))
bx.text(SOCKET['x'], SOCKET['yr'] + 6, 'bank USB-C socket (floor end)', ha='center', fontsize=7)
bx.add_patch(Rectangle((-42.5, COIL['yr'][0]), 85, COIL['yr'][1] - COIL['yr'][0], fc='none', ec='#1f4e79', ls=':', lw=1.5))
bx.text(0, -48, 'coil zone (d 17-103, above this section)', ha='center', fontsize=7, color='#1f4e79')
bx.set_xlim(-60, 60); bx.set_ylim(Y_MIN - 4, Y_MAX + 4)

fig.text(0.53, 0.37, 'v3 changes in this sketch', fontsize=13, weight='bold')
fig.text(0.53, 0.05, '\n'.join([
    '• Cameras dropped: no cartridges, windows, ribbons, shrouds or Kalibr; plain ±X walls.',
    '• Power board under the Pi (d 112–172, X -1..45): a centred board hits the kept switch body',
    '  (ends at X -6). M2.5 post from the tube + bracket from the +X wall. Replaces PD trigger etc.',
    '• Short cables only: J2 → Pi USB-C (R/A, ~100 mm), J5 → Pi GPIO (PH → Dupont splice),',
    '  J4 → switch, J3 → lidar pair, J1 ← short lead from a panel-mount USB-C socket at the',
    '  chamber floor end (bank cord plugs in there: warm swap without reaching inside).',
    '• M12 slack (~95 cm, Ø10) coiled behind the Pi: 2 layers x 2 loops (Ø85 / Ø65), Yr -56..-36.',
    '• UE300 back as in v2: floor, +X / web corner, RJ45 out the floor; behind the board.',
    '• Kept: box 100 × 124 × 180, tube, webs, bank pod + flat seat, Pi at +6 X, switch hole.',
    '• Separate sketches next: T-edge panels / T-channels, floor latches, battery seat clash.',
    '• Fallback if the board fails: the same bay fits the v2 buck module + WAGOs.',
]), fontsize=10, va='bottom', family='monospace')
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110)
print(out)
