"""v4 power button on the fixed floor: top view of the -X bay (looking down, rig X/Yr) + section A-A. Sketch only."""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
from pathlib import Path

C0, C1 = (-36.0, 15.0), (-33.0, 14.0)          # purple candidate C, final (shift +3, -1; home:2149)
HOLE_R = (16.0 + 0.35) / 2                      # Ø16 + resin comp
BODY_R, NUT_R = 8.0, 9.4                        # body Ø16; nut 18.8 across corners (measured)
COLLAR_R1 = 21.5 + 2.0 + 0.15 + 0.175 + 1.8
BOSS, BOSS_R, SCREW_R = (-44.0, 24.0), 4.5, 2.25
fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7.5), gridspec_kw=dict(width_ratios=[1.25, 1]))
a.set_aspect('equal'); a.set_xlim(-54, 4); a.set_ylim(-34, 36)
a.add_patch(Rectangle((-50, -34), 4, 70, color='0.75'))                      # -X wall (X_OUT..X_IN)
a.add_patch(Rectangle((-50, 30.1), 54, 6, color='#c39bd3', alpha=.4)); a.text(-24, 32.5, 'battery door (Yr > 30.1)', fontsize=8)
a.add_patch(Rectangle((-50, -1.5), 54, 3, color='0.6')); a.text(-20, -0.8, 'web (Yr ±1.5)', fontsize=7)
for s in (1, -1):
    a.add_patch(Rectangle((-41.5, s * 1.65 if s > 0 else -3.15), 17, 1.5, color='#5d6d7e'))
a.text(-41, 4.0, 'floor rails h4 (X -41.5..-24.5, Yr ±1.65..3.15)', fontsize=7)
a.add_patch(Rectangle((-46, -9), 4, 18, color='0.45')); a.text(-53.5, -12, 'web foot |Yr|<9, |X|>42 (h10)', fontsize=7)
a.add_patch(Circle((0, 0), COLLAR_R1, fill=False, ls='--', color='k')); a.text(-12, -26, 'stick collar R%.1f h4' % COLLAR_R1, fontsize=7)
a.add_patch(Circle(BOSS, BOSS_R, color='0.4')); a.add_patch(Circle(BOSS, SCREW_R, color='w'))
a.text(-49, 26.5, 'core boss (-44,24) R4.5, M4', fontsize=7)
a.add_patch(Rectangle((-45.9, -27.6), 16, 26, color='#ecf0f1', ec='k', ls=':')); a.text(-45, -24, 'UE300 cradle\n(v4-layout draft)', fontsize=7)
a.add_patch(Circle(C0, BODY_R, color='#9b59b6', alpha=.35)); a.text(-41, 19.5, 'purple C\n(-36, 15)', fontsize=7, color='#6c3483')
a.add_patch(Circle(C1, HOLE_R, fill=False, lw=2, color='r')); a.add_patch(Circle(C1, NUT_R, fill=False, ls='--', color='r'))
a.plot(*C1, 'r+', ms=12)
a.annotate('', C1, C0, arrowprops=dict(arrowstyle='->', color='r'))
a.text(-30, 14.5, 'PROPOSED (-34, 13)\nshift dX +2, dYr -2\nhole Ø16.35, nut 18.8 [MEASURE]', fontsize=8, color='r')
cl = [('collar', np.hypot(*C1) - HOLE_R - COLLAR_R1, np.hypot(*C1) - NUT_R - COLLAR_R1),
      ('core boss', np.hypot(C1[0] - BOSS[0], C1[1] - BOSS[1]) - HOLE_R - BOSS_R, np.hypot(C1[0] - BOSS[0], C1[1] - BOSS[1]) - NUT_R - BOSS_R),
      ('wall X_IN 46', 46 + C1[0] - HOLE_R, 46 + C1[0] - NUT_R), ('rail Yr 3.15', C1[1] - HOLE_R - 3.15, C1[1] - NUT_R - 3.15),
      ('foot corner (-42,9)', np.hypot(C1[0] + 42, C1[1] - 9) - HOLE_R, np.hypot(C1[0] + 42, C1[1] - 9) - NUT_R),
      ('door edge 30.1', 30.1 - C1[1] - HOLE_R, 30.1 - C1[1] - NUT_R)]
txt = 'clearance (mm)      hole    nut 18.8\n' + '\n'.join(f'{n:<19}{h:5.1f}  {m:7.1f}' for n, h, m in cl)
txt += '\npurple (-36,15): hole hits the core boss (-0.6)'
a.text(-53, -33, txt, family='monospace', fontsize=7.5, va='bottom', bbox=dict(fc='w', ec='0.5'))
a.set_title('TOP (looking down, rig X / Yr), -X bay of the fixed floor'); a.set_xlabel('X'); a.set_ylabel('Yr'); a.grid(alpha=.2)
# section A-A through Yr 13 (X vs d, d down)
b.set_xlim(-54, 4); b.set_ylim(192, 80); b.set_aspect('equal')
b.add_patch(Rectangle((-50, 180), 54, 3, color='0.7')); b.text(-20, 182.5, 'fixed floor 3 (d 180..183)', fontsize=7)
b.add_patch(Rectangle((-50, 80), 4, 103, color='0.75'))
b.add_patch(Rectangle((-33 - 8, 140), 16, 40, color='#e74c3c', alpha=.5)); b.text(-31, 150, 'switch body Ø16 x 40\nincl. plug (d 140..180)', fontsize=7)
b.add_patch(Rectangle((-33 - 9.4, 177.5), 20, 2.5, color='k')); b.text(-24, 176, 'nut on floor top', fontsize=7)
b.add_patch(Rectangle((-33 - 9.4, 183), 18.8, 2.5, color='0.2')); b.text(-24, 188, 'bezel Ø19 outside (protrudes ~2.5) [MEASURE]', fontsize=7)
b.add_patch(Rectangle((-46, 115), 22, 21, fill=False, ls='--', color='#1f4e79')); b.text(-45, 113, 'M12 trunk -X turn lifted: d < 136', fontsize=7, color='#1f4e79')
b.plot([-33, -33, -38, -38], [140, 125, 115, 105], color='g', lw=2)
b.text(-36, 100, '5-wire pigtail up to d~105, through the old -X web notch,\nbehind the Pi PCB (Yr~-27) to J4 on the +X wall\n(X 28..42, Yr -42.5..-37, d 86..99): ~200 mm XH lead', fontsize=7, color='g')
b.axvline(-23.5, color='k', ls=':'); b.text(-22.5, 92, 'tube OD', fontsize=7)
b.set_title('SECTION A-A at Yr 14 (X vs d, d down)'); b.set_xlabel('X'); b.set_ylabel('d'); b.grid(alpha=.2)
fig.suptitle('v4 power button on the fixed floor: proposal (sketch, v4-power-button 2026-10-29)')
fig.tight_layout(); fig.savefig(Path(__file__).with_suffix('.png'), dpi=130)
print(txt)
