"""Concept sketch (for approval): power board support on the -X wall after the wide front panel removed H1 (v4-layout,
2026-10-29). H2 M2.5 screw + bottom-edge ledge + top-edge hook, all on the flat -X wall (Yr >= -31).
    python pi_box_v4_board_support_sketch.py -> .png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

B = dict(y0=-43.0, y1=3.0, top=95.0, bot=155.0, back=-43.5, face=-41.9)
LEDGE = dict(x=(-46.0, -41.4), y=(-31.0, -3.0), d=(155.2, 157.2))
HOOK = dict(x=(-46.0, -40.9), y=(-25.0, -14.0), d=(93.0, 95.2))
H2 = (0.0, 142.0)
fig, axs = plt.subplots(1, 3, figsize=(20, 9), gridspec_kw=dict(width_ratios=[1.3, 1, 1.2]))
fig.suptitle('Power board support on the -X wall (H1 standoff lost to the wide front panel)  [SKETCH, mm, rig frame]',
             fontsize=13, weight='bold')

# 1. wall elevation from inside (+X looking -X): Yr across, d down
ax = axs[0]; ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_title('1. -X wall from inside (Yr across, d down)')
ax.add_patch(Rectangle((-62, 72), 30, 110, fc='#d6eaf8', ec='#2874a6', ls='--'))
ax.text(-47, 178, 'wide front panel\n(Yr < -32)', ha='center', fontsize=8, color='#2874a6')
ax.add_patch(Rectangle((-1.5, 60), 3, 120, fc='#ccc', ec='k')); ax.text(1, 70, 'web', rotation=90, fontsize=8)
ax.add_patch(Rectangle((B['y0'], B['top']), 46, 60, fc='#1e8449', alpha=0.75, ec='k'))
ax.text(-20, 125, 'POWER BOARD 46 x 60\nUSB-C edge up', ha='center', color='w', weight='bold')
ax.add_patch(Rectangle((LEDGE['y'][0], LEDGE['d'][0]), 28, 2, fc='#c0392b', ec='k')); ax.text(-17, 162, 'bottom ledge Yr -31..-3, d 155.2..157.2', ha='center', fontsize=8, color='#c0392b')
ax.add_patch(Rectangle((HOOK['y'][0], HOOK['d'][0]), 11, 2.2, fc='#8e44ad', ec='k')); ax.text(-19.5, 89, 'top hook Yr -25..-14 (between J2 and J1)', ha='center', fontsize=8, color='#8e44ad')
ax.add_patch(Circle(H2, 2.5, fc='w', ec='k')); ax.text(3, 140, 'H2 M2.5 (kept)', fontsize=8)
ax.add_patch(Circle((-40, 152), 2.5, fc='none', ec='r', ls='--')); ax.text(-39, 150, 'H1 unused', fontsize=8, color='r')
ax.axvline(-32, color='#999', ls=':')
ax.annotate('', (-31, 168), (-3, 168), arrowprops=dict(arrowstyle='<->')); ax.text(-17, 172, '28', ha='center')
ax.set_xlim(-64, 10); ax.set_ylim(185, 80)

# 2. section through the hook / ledge (X across, d down) at Yr -28
ax = axs[1]; ax.set_aspect('equal'); ax.invert_yaxis(); ax.set_title('2. Section at Yr -20 (X across, d down)')
ax.add_patch(Rectangle((-50, 85), 4, 80, fc='#bbb', ec='k')); ax.text(-48, 83, 'wall', ha='center', fontsize=8)
ax.add_patch(Rectangle((B['back'], B['top']), 1.6, 60, fc='#1e8449', ec='k'))
ax.add_patch(Rectangle((-46, 93), 2.5, 2.2, fc='#8e44ad', ec='k')); ax.add_patch(Rectangle((-43.5 + 1.6, 93), 1.0, 4.0, fc='#8e44ad', ec='k'))
ax.add_patch(Rectangle((-46, 155.2), 4.6, 2.0, fc='#c0392b', ec='k'))
ax.plot([-46, -41.4], [157.2, 161.8], color='#c0392b'); ax.text(-38, 160, '45° underside', fontsize=8)
ax.text(-38, 95, 'hook lip 1.0 over\ncomponent face,\n2 mm down the board', fontsize=8)
ax.text(-38, 126, 'board back X -43.5\n(2.5 off the wall)', fontsize=8)
ax.add_patch(Rectangle((-41.9, 100), 6, 50, fc='#f1c40f', alpha=0.3)); ax.text(-38.9, 145, 'parts', fontsize=7, ha='center')
ax.set_xlim(-52, -28); ax.set_ylim(168, 82)

# 3. fitting sequence
ax = axs[2]; ax.axis('off'); ax.set_title('3. Fitting sequence')
steps = ['1. Plug J1/J2 USB-C leads and J3/J4/J5 onto the board first (top-edge USB-Cs leave upward).',
         '2. Hold the board flat against the -X wall, tilted ~10° (top edge in), and lower it from the flange side.',
         '3. Slip the top edge (between J2 and J1) behind the hook lip, then swing the bottom edge in.',
         '4. Let it drop the last 2 mm: the bottom edge sits on the ledge, the top edge stays 0.2 under the hook.',
         '   (board 60 tall; hook bottom d 95.2 .. ledge top d 155.2 = 60.0 + 0.2 play)',
         '5. Drive the H2 M2.5 screw (Yr 0, d 142) into the standoff insert: board locked.',
         'Removal: H2 out, lift 0.2 / tilt the bottom out, slide down & out.',
         '',
         'Why not move the board +Yr: H2 would land at Yr 9 in the -X trunk turn,',
         'and the bottom edge on the floor button (-33, 14). Why not a bracket at H1: the',
         'panel lip occupies X -46..-44 at Yr -36..-25.',
         'Hook sits in the free top-edge strip Yr -25..-14 (J2 plug Yr -39..-27, J1 plug Yr -13..-1).']
for i, s in enumerate(steps):
    ax.text(0, 1 - i * 0.065, s, fontsize=9.5, va='top', transform=ax.transAxes)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=110)
