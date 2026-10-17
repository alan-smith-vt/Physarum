"""Concept sketch (for approval, not a model): v3 back (+Y, battery-side) panel, 2026-10-17.

Open frame like the v3 floor: border (T-edge on both long sides), a MagSafe ring recess centred under the phone
(portrait, centred; the phone comes off to read the battery), braces from the ring to the frame, and a framed
window over the bank's display + button. Display position and size are [MEASURE]; the phone is a generic 72 x 147.

    python pi_box_v3_back_panel_sketch.py -> pi_box_v3_back_panel_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

X_IN, D0, D1 = 46.0, 10.0, 180.0          # panel spans the wall gap, flange underside to the floor
T_EDGE = 3.0                              # T-edge band each side (sits in the walls' T-channels)
FRAME = 5.0                               # border width inside the T-edges
RING_OD, RING_ID, RING_C = 56.0, 46.0, (0.0, 88.0)    # MagSafe ring (recess ~0.8 deep)
PHONE = (72.0, 147.0)
WIN = (-40.0, 12.0, 142.0, 172.0)         # display + button window: X0, X1, d0, d1 [MEASURE display]
BUTTON = (-22.0, 158.0)
BRACE = 4.0

fig = plt.figure(figsize=(15, 10.5))
ax = fig.add_axes([0.04, 0.05, 0.5, 0.88]); ax.set_aspect('equal'); ax.invert_yaxis(); ax.invert_xaxis()
ax.set_title('Back (+Y) panel seen from outside (rig X mirrored, depth down, mm)', fontsize=13)

ax.add_patch(Rectangle((-X_IN - T_EDGE, D0), 2 * (X_IN + T_EDGE), D1 - D0, fc='#e8eef7', ec='k', lw=1.5))
inner = (-X_IN + FRAME, D0 + FRAME, 2 * (X_IN - FRAME), D1 - D0 - 2 * FRAME)
ax.add_patch(Rectangle(inner[:2], inner[2], inner[3], fc='w', ec='k'))          # open area
for s in (-1, 1):
    ax.add_patch(Rectangle((s * X_IN if s > 0 else -X_IN - T_EDGE, D0), T_EDGE, D1 - D0, fc='#9ab', ec='k'))
ax.text(X_IN + T_EDGE / 2, D0 + 6, 'T-edge', rotation=90, fontsize=8, ha='center', va='top')

def brace(p, q):
    p, q = np.array(p), np.array(q)
    n = np.array([-(q - p)[1], (q - p)[0]]); n = n / np.linalg.norm(n) * BRACE / 2
    ax.add_patch(Polygon([p + n, q + n, q - n, p - n], fc='#e8eef7', ec='k', lw=0.8))

cx, cy = RING_C
r = RING_OD / 2 + 1.5
for ang in (45, 135, 225, 315):
    a = np.radians(ang)
    p = (cx + r * np.cos(a), cy + r * np.sin(a))
    q = (np.sign(np.cos(a)) * (X_IN - FRAME), cy + np.sign(np.sin(a)) * (cy - D0 - FRAME) if np.sin(a) < 0 else WIN[2])
    brace(p, q)
brace((cx, cy - r), (cx, D0 + FRAME)); brace((cx - r, cy), (-X_IN + FRAME, cy)); brace((cx + r, cy), (X_IN - FRAME, cy))
ax.add_patch(Rectangle((-X_IN + FRAME, WIN[2] - BRACE), 2 * (X_IN - FRAME), BRACE, fc='#e8eef7', ec='k', lw=0.8))
ax.add_patch(Rectangle((WIN[1], WIN[2]), BRACE, WIN[3] - WIN[2], fc='#e8eef7', ec='k', lw=0.8))
ax.add_patch(Rectangle((WIN[0], WIN[2]), WIN[1] - WIN[0], WIN[3] - WIN[2], fc='#fd7', ec='#a80', lw=1.5, alpha=0.6))
ax.text((WIN[0] + WIN[1]) / 2, WIN[2] + 7, 'window: bank display\n[MEASURE position]', ha='center', fontsize=8)
ax.add_patch(Circle(BUTTON, 5, fc='#f96', ec='k')); ax.text(BUTTON[0], BUTTON[1] + 9, 'button', ha='center', fontsize=8)
ax.add_patch(Circle(RING_C, RING_OD / 2 + 1.5, fc='#e8eef7', ec='k'))
ax.add_patch(Circle(RING_C, RING_OD / 2, fc='#bcd', ec='k', hatch='///'))
ax.add_patch(Circle(RING_C, RING_ID / 2, fc='#e8eef7', ec='k'))
ax.text(cx, cy, 'MagSafe ring\nrecess Ø56', ha='center', va='center', fontsize=9)
ax.add_patch(Rectangle((-PHONE[0] / 2, cy - 62), PHONE[0], PHONE[1], fc='none', ec='#c0c', ls='--', lw=2))
ax.text(-PHONE[0] / 2 + 2, cy - 64, 'phone 72 x 147 (portrait, centred)', color='#c0c', fontsize=9)
ax.add_patch(Rectangle((-X_IN - T_EDGE, -6), 2 * (X_IN + T_EDGE), D0 + 6, fc='none', ec='#1f4e79', hatch='xx', alpha=0.5))
ax.text(0, 0, 'flange: M12 plug + loop stand off this end (outside)', ha='center', fontsize=8, color='#1f4e79')
ax.set_xlim(62, -62); ax.set_ylim(D1 + 8, -10)

fig.text(0.58, 0.88, 'Back panel (v3)', fontsize=15, weight='bold')
fig.text(0.58, 0.45, '\n'.join([
    '• Open frame like the v3 floor: 5 mm border,',
    '  T-edges both sides, 4 mm braces.',
    '• MagSafe ring: Ø56 recess (~0.8 deep) on a',
    '  solid annulus, centred under the phone;',
    '  braces to the corners, sides and top.',
    '• Phone portrait, centred: covers the window;',
    '  take it off to read the battery.',
    '• Window over the display, framed by a cross',
    '  brace; button reachable through it.',
    '• Phone top ~d 26: below the flange, so',
    '  clear of the lidar view and the M12 loop.',
    '',
    'To confirm / measure:',
    '  - bank display position + size',
    '  - phone model (ring centre height)',
    '  - ring sticker vs magnet ring (recess depth)',
]), fontsize=11, va='bottom', family='monospace')
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110)
print(out)
