"""Sketch (explanation + one proposed change): how the OV9281 board mates with the v4 camera cartridge (2026-10-29).

Section through the lens axis, the face seen along the axis, and the assembly order. Board frame: c along the optical
axis from the board's front face, b along the board's 'up'. Numbers from designs/pi_box_v4_cam_cartridge.py.

    python pi_box_v4_cartridge_board_sketch.py -> pi_box_v4_cartridge_board_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle

PITCH = np.radians(24.0)
POD_C, POD_B = (-3.0, 18.0), (-17.1, 10.75)
BOARD_T, BOARD_B = 1.6, (-15.6, 8.25)           # board 25 wide x 23.85 tall; lens axis 8.25 below its top edge
BOSS_H, NUT_C, NUT_T = 2.0, 5.0, 1.6
HOLDER_D, HOLDER_C, BARREL_D = 19.0, 7.0, 14.5
HOLES = [(t, v) for t in (-10.5, 10.5) for v in (-6.25, 6.25)]

fig = plt.figure(figsize=(20, 10))

# ── 1: section through the lens axis, drawn in the camera's own frame (axis horizontal), the land tilted 24° ──
ax = fig.add_subplot(1, 3, 1)
ax.add_patch(Rectangle((POD_C[0], POD_B[0]), POD_C[1] - POD_C[0], POD_B[1] - POD_B[0], fc='#48c9b0', ec='#117864', lw=1.5))
ax.add_patch(Rectangle((POD_C[0] - 0.01, BOARD_B[0] - 0.3), BOSS_H - POD_C[0], BOARD_B[1] - BOARD_B[0] + 0.6, fc='white', ec='none'))
ax.add_patch(Rectangle((-0.01, -HOLDER_D / 2), HOLDER_C, HOLDER_D, fc='white', ec='none'))
ax.add_patch(Rectangle((HOLDER_C - 0.01, -BARREL_D / 2), POD_C[1] - HOLDER_C + 0.02, BARREL_D, fc='white', ec='none'))
for v in (-6.25, 6.25):                                     # bosses + screw + nut slot (seen in section at t = ±10.5)
    ax.add_patch(Rectangle((0, v - 3.0), BOSS_H, 6.0, fc='#48c9b0', ec='#117864', ls='--'))
    ax.add_patch(Rectangle((-BOARD_T - 1.6, v - 1.1), 1.6, 2.2, fc='#7f8c8d'))                      # M2 head behind the board
    ax.add_patch(Rectangle((-BOARD_T, v - 0.5), NUT_C + NUT_T + BOARD_T, 1.0, fc='#95a5a6'))         # shank
    ax.add_patch(Rectangle((NUT_C, v - 2.2), NUT_T, 4.4, fc='#2c3e50'))                              # captive nut
ax.add_patch(Rectangle((-BOARD_T, BOARD_B[0]), BOARD_T, BOARD_B[1] - BOARD_B[0], fc='#1e8449'))
ax.add_patch(Rectangle((-BOARD_T - 2.5, BOARD_B[0]), 2.5, 4.5, fc='#f1c40f', ec='k'))                    # FFC connector (back, bottom)
ax.add_patch(Rectangle((0, -7.5), 6.5, 15.0, fc='#566573', alpha=0.8))                                   # lens holder
ax.add_patch(Rectangle((6.5, -7.0), 11.5, 14.0, fc='#34495e'))                                           # barrel
ax.add_patch(Polygon([(18.0, -5.5), (18.0, 5.5), (18.4, 4.5), (18.4, -4.5)], fc='#85c1e9'))             # front glass
ax.annotate('board front face rests on the 4 boss tops\n(bosses square to the axis: they set the 24° pitch)',
            xy=(BOSS_H, 6.25 + 3), xytext=(3, 15), fontsize=7, arrowprops=dict(arrowstyle='->'))
ax.annotate('M2×8 from behind, through the board,\ninto a hex nut slid in from the pod side', xy=(NUT_C, -6.25),
            xytext=(4, -24), fontsize=7, arrowprops=dict(arrowstyle='->'))
ax.annotate('FFC connector (back face, bottom edge):\nribbon leaves DOWN along the board', xy=(-BOARD_T - 1.2, BOARD_B[0]),
            xytext=(-14, -27), fontsize=7, color='#9a7d0a', arrowprops=dict(arrowstyle='->'))
ax.add_patch(Rectangle((POD_C[0], POD_B[0]), BOSS_H - POD_C[0] + 3, POD_B[0] - BOARD_B[0] + 0.0, fc='none', ec='r', lw=2))
ax.text(POD_C[0] - 1, POD_B[0] - 4.5, 'PROPOSED: 17 × 4 slot through this\n1.2 mm bottom wall for the ribbon', color='r', fontsize=7)
ax.annotate('lens front flush with the face;\ncountersunk 78° eye round it', xy=(18.2, 4), xytext=(9, 13), fontsize=7,
            arrowprops=dict(arrowstyle='->'))
ax.arrow(14, 0, 9, 0, width=0.3, color='g'); ax.text(19, 1.2, 'optical axis\n(24° up, 30° fwd)', color='g', fontsize=7)
ax.text(-3, 12.2, 'open back (board goes in from here)', fontsize=7)
ax.set_xlim(-18, 26); ax.set_ylim(-32, 20); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('c: along the optical axis from the board face (mm)'); ax.set_ylabel('b: board up (mm)')
ax.set_title('SECTION through the lens axis (camera frame)')

# ── 2: the board seen from behind, with the cartridge outline ──
ax = fig.add_subplot(1, 3, 2)
ax.add_patch(Rectangle((-13.5, POD_B[0]), 27, POD_B[1] - POD_B[0], fc='#48c9b0', ec='#117864', lw=1.5))
ax.add_patch(Rectangle((-12.8, BOARD_B[0] - 0.3), 25.6, BOARD_B[1] - BOARD_B[0] + 0.6, fc='white', ec='#117864', ls=':'))
ax.add_patch(Rectangle((-12.5, BOARD_B[0]), 25, BOARD_B[1] - BOARD_B[0], fc='#1e8449', alpha=0.85))
for t, v in HOLES:
    ax.add_patch(Circle((t, v), 1.9, fc='#7f8c8d'))
    ax.add_patch(Rectangle((t, v - 2.2), np.sign(t) * (13.5 - abs(t)), 4.4, fc='#2c3e50', alpha=0.35))
ax.add_patch(Rectangle((-8.0, BOARD_B[0]), 16.0, 4.5, fc='#f1c40f', ec='k'))
ax.add_patch(Rectangle((-8.5, POD_B[0]), 17.0, POD_B[0] - BOARD_B[0] + 1.5, fc='none', ec='r', lw=2))
ax.text(0, POD_B[0] - 3, 'proposed ribbon slot 17 wide', ha='center', color='r', fontsize=8)
ax.text(0, BOARD_B[0] + 6, '15-pin FFC connector', ha='center', fontsize=7)
ax.text(11, 9.5, 'nut slots open\nto the pod sides', fontsize=7)
ax.plot(0, 0, 'k+', ms=12); ax.text(0.8, 0.8, 'lens axis', fontsize=7)
ax.text(0, 13, 'board 25 × 23.85 in a 25.6 × 24.45 pocket (0.3 clearance);\nholes 21 × 12.5, lens axis 8.25 below the top edge',
        ha='center', fontsize=7)
ax.set_xlim(-17, 17); ax.set_ylim(-24, 17); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_title('BOARD seen from behind (open back of the cartridge)')

# ── 3: assembly order and what locates what ──
ax = fig.add_subplot(1, 3, 3); ax.axis('off')
txt = """ASSEMBLY ORDER (per camera)
1. Slide 4 × M2 hex nuts into the side slots of the cartridge.
2. Lens already focused in its holder on the board.
3. Plug the ribbon into the board's connector (back, bottom edge).
4. From the open back: lens through the Ø19 holder bore and Ø14.5
   barrel bore until the board's front face sits on the 4 bosses.
5. 4 × M2×8 [MEASURE] from behind through the board into the nuts.
   A lens-cap push loads the screws into the nuts (nothing glued,
   unlike the v2 inserts that pulled out).
6. Feed the ribbon through the core's opening from outside, then
   slide the cartridge in along the land normal until its 45°
   countersunk flange seats in the core (self-centring).
7. 3 × M2 through the flange counterbores into M2 inserts set from
   the core pad's inside (nuts).

WHAT LOCATES WHAT
- Board tilt (24° pitch): the 4 boss tops, square to the camera axis.
- Board in-plane: the 4 screws in its holes (pocket 0.3 clear/side).
- Yaw 30°: the core opening + countersink face the land normal.
- Cartridge depth/centre: the 45° countersink seat.

PROPOSED CHANGE (for approval): a 17 × 4 slot in the pod's bottom
wall under the connector. Today the 1.2 mm wall sits just past the
board edge, so the ribbon has no way out (see the ribbon sketch).
"""
ax.text(0, 1, txt, va='top', family='monospace', fontsize=9)
fig.suptitle('v4 camera cartridge: how the OV9281 board mates with it', fontsize=14)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=100)
