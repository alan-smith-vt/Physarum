"""Concept sketch (for approval, not a model): v4 floor / battery door options, 2026-10-29 (v4-floor).

Box from the v3 numbers (the v4 core may change them): walls X ±50 (inner ±46), Y ±62, DEPTH 180 (d down from the plate).
Door = the strip under the bank, Yr 30.9..62 (opening ~92 x 31). The fixed floor (stick Ø43 hole + collar, UE300 cradle +
RJ45, nubs) stays screwed on as in v3. Load case: bank 0.36 kg (A1383 [MEASURE]) at 50 g = ~175 N on the door; design 350 N.

    python pi_box_v4_floor_options_sketch.py -> pi_box_v4_floor_options_sketch.png
"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

X_IN, X_OUT, Y_MIN, Y_MAX, DEPTH, FT = 46.0, 50.0, -62.0, 62.0, 180.0, 3.0
Y0, Y1 = 30.9, 62.0
WALL, DOOR, PART, BANK = '#bbb', '#e67e22', '#7f8c8d', '#9ecae1'


def plan(ax, title):
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    ax.add_patch(Rectangle((-X_OUT, Y_MIN), 2 * X_OUT, Y_MAX - Y_MIN, fc='#eee', ec='k'))
    ax.add_patch(Circle((0, 0), 21.5, fc='w', ec='k')); ax.text(0, 0, 'stick\nØ43', ha='center', va='center', fontsize=7)
    ax.add_patch(Rectangle((20, -40), 24, 30, fc='none', ec='#36c', ls='--'))
    ax.text(32, -25, 'UE300', ha='center', fontsize=6, color='#36c')
    ax.add_patch(Rectangle((-X_IN, Y0), 2 * X_IN, Y1 - Y0 - 0.1, fc=DOOR, ec='k', alpha=.85))
    ax.text(0, 46, 'door 92 × 31', ha='center', fontsize=7)
    ax.set_xlim(-68, 68); ax.set_ylim(-66, 72); ax.set_xlabel('X (seen from below)', fontsize=7); ax.tick_params(labelsize=6)


def section(ax, title):
    """Section at X = 0 through the door (Yr across, d down), bottom 35 mm."""
    ax.set_aspect('equal'); ax.set_title(title, fontsize=10)
    ax.add_patch(Rectangle((Y_MAX - 4, 150), 4, DEPTH - 150, fc=WALL, ec='k')); ax.text(Y_MAX + 1, 155, 'back\nwall', fontsize=7)
    ax.add_patch(Rectangle((26.5, 150), 4, DEPTH - 150, fc=WALL, ec='k')); ax.text(20, 155, 'rib', fontsize=7)
    ax.add_patch(Rectangle((10, DEPTH), 20.5, FT, fc='#ccc', ec='k')); ax.text(11, DEPTH + 7, 'fixed floor', fontsize=7)
    ax.add_patch(Rectangle((Y0 + 1.5, 152), 26, DEPTH - 152 - 0.5, fc=BANK, ec='k', alpha=.6)); ax.text(40, 168, 'bank', fontsize=8)
    ax.set_xlim(5, 75); ax.set_ylim(195, 140); ax.tick_params(labelsize=6)
    ax.set_xlabel('Yr', fontsize=7); ax.set_ylabel('d', fontsize=7)


def dim(ax, p, q, s, off=(0, 0)):
    ax.annotate('', p, q, arrowprops=dict(arrowstyle='<->', lw=.8))
    ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], s, fontsize=7, ha='center')


def door(ax):
    ax.add_patch(Rectangle((Y0 - 0.4, DEPTH), Y1 - 4 - Y0 + 0.4, FT, fc=DOOR, ec='k'))


fig = plt.figure(figsize=(20, 24))
gs = fig.add_gridspec(5, 2, height_ratios=[1, 1, 1, 1, .9], hspace=.35, wspace=.12)

# A: sliding door in wall grooves + pull-ring index plunger (as modelled: designs/pi_box_v4_door_a.py)
ax = fig.add_subplot(gs[0, 0]); plan(ax, 'A  SLIDING DOOR + INDEX PLUNGER (recommended): slides out +Y, under the back panel')
ax.annotate('', (0, 71), (0, 58), arrowprops=dict(arrowstyle='->', lw=2)); ax.text(3, 66, 'pull out (+Y)', fontsize=7)
ax.add_patch(Rectangle((-25, Y1 - 4), 50, 4, fc='#a04000', ec='k')); ax.text(-24, 52, 'grip lip', fontsize=6)
ax.add_patch(Circle((-42, 44), 2.7, fc=PART, ec='k')); ax.add_patch(Circle((-42, 44), 9, fc='none', ec='k', ls=':'))
ax.text(-66, 6, 'M8 pull-ring plunger\nin Ø18 boss (-42, 44)', fontsize=6)
for s in (-1, 1):
    ax.plot([s * 46.8] * 2, [30.5, 62], 'k--', lw=1)
ax.text(-45, 66, 'grooves in ±X wall inner faces, Yr 30.5..62; door butts the rib', fontsize=6)
ax = fig.add_subplot(gs[0, 1]); ax.set_aspect('equal')
ax.set_title('A  section across X at Yr 44 (-X side): plate under the wall, web + hook in a closed groove', fontsize=10)
ax.add_patch(Polygon([(-50, 165), (-46, 165), (-46, 175.6), (-44.4, 175.6), (-44.4, 178.4), (-46, 178.4), (-46, 180),
                      (-50, 180)], fc=WALL, ec='k')); ax.text(-49.5, 168, 'wall', fontsize=7)
ax.add_patch(Rectangle((-46, 168), 8, 7.8 + 4.2, fc=WALL, ec='k', alpha=.6)); ax.text(-41.5, 170, 'plunger\nboss', fontsize=7)
ax.add_patch(Rectangle((-42.5 - 0.2, 173), 5.4, 7, fc='w', ec='k', ls=':'))
ax.add_patch(Polygon([(-49.8, 180.2), (-30, 180.2), (-30, 183.2), (-49.8, 183.2)], fc=DOOR, ec='k'))
ax.add_patch(Polygon([(-45.6, 180.2), (-45.6, 178.0), (-47.2, 178.0), (-47.2, 176.0), (-44.2, 176.0), (-44.2, 180.2)], fc=DOOR, ec='k'))
ax.add_patch(Rectangle((-51, 183.2), 18, 8, fc=DOOR, ec='k', alpha=.7)); ax.text(-50.5, 189.5, 'boss + M8 nut', fontsize=7)
ax.add_patch(Rectangle((-44.5, 180.2), 5, 6 + 3, fc=PART, ec='k', alpha=.6)); ax.text(-38.5, 187, 'plunger body', fontsize=6)
ax.add_patch(Rectangle((-36, 160), 6, 20, fc=BANK, ec='k', alpha=.6)); ax.text(-35.5, 172, 'bank', fontsize=7)
dim(ax, (-27, 175.6), (-27, 178.4), 'groove 2.8 × 1.6', off=(6, 0))
dim(ax, (-27, 178.4), (-27, 180), 'ledge 1.6', off=(5, 1.5))
dim(ax, (-27, 180.2), (-27, 183.2), 'plate 3', off=(4, 1))
ax.text(-58, 160, 'Bank stands on the plate top (d 180.2): same height as v3 (core seat d 22).\n'
        'The bank load hangs on the 1.6 ledges (shear, ~2 MPa at 50 g), never on the plunger:\n'
        'the plunger pin (in hole Ø5.4 × 7 up from d 180) only stops sliding.', fontsize=7, va='top')
ax.set_xlim(-60, -15); ax.set_ylim(194, 152); ax.tick_params(labelsize=6); ax.set_xlabel('X', fontsize=7); ax.set_ylabel('d', fontsize=7)

# B: hook-in edge + quarter-turn compression cam latch
ax = fig.add_subplot(gs[1, 0]); plan(ax, 'B  HOOK-IN EDGE + QUARTER-TURN CAM LATCH (wing knob)')
ax.add_patch(Circle((0, 46), 6, fc=PART, ec='k')); ax.plot([-6, 6], [46, 46], 'k', lw=2)
ax.text(10, 36, 'wing knob, 1/4 turn (Southco E3-type\ncompression latch, Ø~16 head) [VERIFY]', fontsize=6)
ax = fig.add_subplot(gs[1, 1]); section(ax, 'B  section: hook under the rib, cam pawl into the back wall')
door(ax)
ax.add_patch(Polygon([(Y0 - .4, DEPTH), (Y0 - 3, DEPTH - 2), (Y0 - 3, DEPTH - 4), (Y0 - .4, DEPTH - 2)], fc=DOOR, ec='k'))
ax.text(12, 174, 'hook under\nrib lip', fontsize=7)
ax.add_patch(Rectangle((46, DEPTH - 14), 4, 14, fc=PART, ec='k')); ax.add_patch(Rectangle((46, DEPTH - 14), 14, 2.5, fc=PART, ec='k'))
ax.text(44, DEPTH - 17, 'pawl into slot', fontsize=7)
ax.text(7, 141, 'Door goes in rib edge first, swings up, quarter-turn: the pawl pulls it tight.\n'
        'Detent + compression: it does not back off under vibration. Pawl carries ~half the load.', fontsize=7, va='top')

# C: captive quarter-turn studs
ax = fig.add_subplot(gs[2, 0]); plan(ax, 'C  2 × CAPTIVE QUARTER-TURN STUDS (Dzus-style)')
for x in (-30, 30):
    ax.add_patch(Circle((x, 46), 5, fc=PART, ec='k')); ax.plot([x - 4, x + 4], [46, 46], 'k', lw=1.5)
ax.text(-45, 64, 'wing-head 1/4-turn studs Ø~10 (Southco 82/85 or Dzus wing) + clip receptacles [VERIFY]', fontsize=6)
ax = fig.add_subplot(gs[2, 1]); section(ax, 'C  section: stud into a receptacle in a boss')
door(ax)
ax.add_patch(Rectangle((44, DEPTH - 12), 6, 12, fc=WALL, ec='k')); ax.add_patch(Rectangle((46, DEPTH - 12), 2, 15, fc=PART, ec='k'))
ax.add_patch(Rectangle((44, DEPTH + 3), 6, 2, fc=PART, ec='k'))
ax.text(7, 141, 'Both studs carry the bank in tension (~90 N each at 50 g). Two turns, door loose in the\n'
        'other hand; the bosses sit in the bank chamber (lose bank room or need wall bosses).', fontsize=7, va='top')

# D: toggle latches with safety catch
ax = fig.add_subplot(gs[3, 0]); plan(ax, 'D  HOOK-IN EDGE + 2 TOGGLE LATCHES WITH SAFETY CATCH (±X walls)')
for x0 in (X_OUT, -X_OUT - 7):
    ax.add_patch(Rectangle((x0, 38), 7, 16, fc=PART, ec='k'))
ax.text(-45, 64, 'stainless mini toggle latch with safety catch, ~40 × 15 [VERIFY]', fontsize=6)
ax = fig.add_subplot(gs[3, 1]); section(ax, 'D  section: like B, latches on the side walls')
door(ax)
ax.text(7, 141, 'The v3 draw latch without the hinge, doubled, plus a second catch. Proven hardware, but\n'
        'it sticks out of the walls (snags; close to the v3 look the user dislikes) and the door needs ears.', fontsize=7, va='top')

ax = fig.add_subplot(gs[4, :]); ax.axis('off')
rows = [['', 'A slide + plunger', 'B hook + 1/4-turn cam', 'C 2 × 1/4-turn studs', 'D 2 toggles + catch'],
        ['tool-free', 'yes', 'yes (wing knob)', 'yes (wing heads)', 'yes'],
        ['retention 30-50 g, bank on door', '★★★ rails take the load in shear;\nplunger only stops sliding',
         '★★ hook + cam pawl in tension', '★★ 2 studs in tension', '★★ 2 latches in tension;\ncatch stops knock-open'],
        ['one-handed', '★★★ pull ring + slide', '★★★ one turn, door swings', '★ two turns, door loose', '★★ two latches'],
        ['print / assembly risk', '★★ rails must slide fit strip\nfirst (pi_box_v4_door_fit)', '★★ slot in back wall',
         '★★ bosses in bank chamber', '★★★ v3 ears/inserts exist'],
        ['sourcing', '★★★ M8 pull-ring index plunger\n(Kipp / McMaster type) ~$10', '★★ Southco E3-type\nwing latch ~$15',
         '★★ Southco/Dzus ~$8 each', '★★★ Amazon mini toggle\nwith safety catch ~$8/2'],
        ['outside profile', 'plate flush; grip lip + plunger\nboss 8 below the floor', 'flush knob under the floor', 'flush heads', 'proud of ±X walls ~7 mm']]
t = ax.table(cellText=rows, loc='center', cellLoc='center'); t.auto_set_font_size(False); t.set_fontsize(8); t.scale(1, 3.2)
ax.set_title('Rated (★★★ best). Bayonet twist floor rejected: the stick hole is in the floor centre and the bank strip is 92 × 31,'
             ' so a round twist door does not fit.', fontsize=9, y=1.12)
fig.suptitle('v4 Pi box floor / battery door: concept options (sketch for approval, v4-floor 2026-10-29). Recommend A.', fontsize=14)
fig.savefig(Path(__file__).with_suffix('.png'), dpi=80, bbox_inches='tight')
