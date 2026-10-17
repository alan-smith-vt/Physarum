"""Concept sketch (for approval, not a model): v3 panel edge captured in the wall, 2026-10-17.

v2: plain 3.2 panel in a 2 mm groove; a bowed panel's edges pull in (X) and pop out. v3: a hooked edge in a matching
channel, so the panel can only slide (along depth). A symmetric T doesn't fit: only 1.5 mm of wall (PANEL_LIP) is
left outside the panel, so the head turns inward (L-hook, toward the box interior). Section at any depth, -X/-Y corner.

    python pi_box_v3_panel_edge_sketch.py -> pi_box_v3_panel_edge_sketch.png
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

X_IN, X_OUT, Y_MIN, LIP = 46.0, 50.0, -62.0, 1.5
PT = 3.175                          # panel
CL = 0.2 + 0.175                    # per side: clearance + RESIN_HOLE_COMP / 2
NECK_X = 1.0                        # wall between the inner face and the channel (the slot's length)
HEAD_X = 1.6                        # hook thickness along X
HOOK = 2.0                          # hook reach past the panel's inner face (along +Yr)
y_out, y_in = Y_MIN + LIP, Y_MIN + LIP + PT                # panel faces
ch_x0, ch_x1 = X_IN + NECK_X, X_IN + NECK_X + HEAD_X       # head (panel), channel = head + CL

fig, axs = plt.subplots(1, 2, figsize=(16, 8.5))
for ax, v3 in zip(axs, (False, True)):
    ax.set_aspect('equal')
    ax.set_title('v3: L-hook edge in a channel' if v3 else 'v2: plain edge in a 2 mm groove', fontsize=13)
    ax.add_patch(Rectangle((X_IN, Y_MIN), X_OUT - X_IN, 12, fc='#bbb', ec='k'))            # wall (solid end zone)
    ax.add_patch(Rectangle((X_IN - 10, Y_MIN), 10, LIP, fc='#bbb', ec='k'))                  # end lip (rim / panel lip)
    if not v3:
        g = (X_IN, X_IN + 2.0, y_out - CL, y_in + CL)
        ax.add_patch(Rectangle((g[0], g[2]), g[1] - g[0], g[3] - g[2], fc='w', ec='k'))
        ax.add_patch(Rectangle((X_IN - 10, y_out), 10 + 2.0 - 0.4, PT, fc='#9cf', ec='k'))
        ax.annotate('bows → edge pulls\nout along -X', (X_IN - 4, y_in + 0.8), (X_IN - 9.5, y_in + 5), fontsize=10,
                    arrowprops=dict(arrowstyle='->', color='r'), color='r')
    else:
        slot = [(X_IN, y_out - CL), (ch_x0, y_out - CL), (ch_x0, y_out - CL), (ch_x1 + CL, y_out - CL),
                (ch_x1 + CL, y_in + HOOK + CL), (ch_x0 - CL, y_in + HOOK + CL), (ch_x0 - CL, y_in + CL), (X_IN, y_in + CL)]
        ax.add_patch(Polygon(slot, fc='w', ec='k'))
        pan = [(X_IN - 10, y_out), (ch_x1, y_out), (ch_x1, y_in + HOOK), (ch_x0, y_in + HOOK), (ch_x0, y_in), (X_IN - 10, y_in)]
        ax.add_patch(Polygon(pan, fc='#9cf', ec='k'))
        ax.annotate('', (ch_x1 + CL, Y_MIN + 0.4), (X_OUT, Y_MIN + 0.4), arrowprops=dict(arrowstyle='<->'))
        ax.text((ch_x1 + CL + X_OUT) / 2, Y_MIN - 0.6, f'{X_OUT - ch_x1 - CL:.2f} skin', ha='center', fontsize=9)
        ax.annotate('', (X_IN, y_in + HOOK + 2.2), (ch_x0, y_in + HOOK + 2.2), arrowprops=dict(arrowstyle='<->'))
        ax.text(X_IN + NECK_X / 2, y_in + HOOK + 2.8, f'neck {NECK_X}', ha='center', fontsize=9)
        ax.annotate('', (ch_x0, y_in + HOOK + 4.2), (ch_x1, y_in + HOOK + 4.2), arrowprops=dict(arrowstyle='<->'))
        ax.text((ch_x0 + ch_x1) / 2, y_in + HOOK + 4.8, f'hook {HEAD_X}', ha='center', fontsize=9)
        ax.annotate('', (ch_x1 + 0.8, y_in), (ch_x1 + 0.8, y_in + HOOK), arrowprops=dict(arrowstyle='<->'))
        ax.text(ch_x1 + 1.0, y_in + HOOK / 2, f'{HOOK}', fontsize=9, va='center')
        ax.annotate('pull-out along -X:\nhook catches the neck', (ch_x0 - 0.1, y_in + 1.0), (X_IN - 9.5, y_in + 6),
                    fontsize=10, arrowprops=dict(arrowstyle='->', color='g'), color='g')
    ax.text(X_IN - 5, y_out + PT / 2, 'panel 3.175', ha='center', va='center', fontsize=10)
    ax.text(X_IN + 2, Y_MIN + 10.5, 'wall X 46..50\n(solid near the ends)', ha='center', fontsize=9)
    ax.text(X_IN - 5, Y_MIN + 0.5, 'end lip 1.5', ha='center', fontsize=9)
    ax.set_xlim(X_IN - 11, X_OUT + 2); ax.set_ylim(Y_MIN - 2, Y_MIN + 13)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Yr (mm)')
fig.text(0.5, 0.02, '  '.join([
    f'Clearance {CL:.3f}/side incl. resin comp.',
    '· Channels open at the floor end with a 45° flare (like the floor grooves); the rim stops the top.',
    '· Panels print flat: the hook stands on the edge band, no overhang.',
    '· Same edge on both panels; keep PANEL_LEAD on the top edge.']), ha='center', fontsize=10)
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110)
print(out)
