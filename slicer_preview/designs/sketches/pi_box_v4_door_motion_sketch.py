"""Explanatory sketch (UNAPPROVED proposal): how the option-A battery door opens and closes, 2026-10-29 (v4-floor).

Left: section along Yr at X = -42 (through the plunger). Right: section across X at Yr 44 (-X rail: web + hook in the
wall groove, plunger pin in the core boss). Numbers from designs/pi_box_v4_door_a.py and pi_box_v4_core.py.

    python pi_box_v4_door_motion_sketch.py    -> pi_box_v4_door_motion_sketch.png (storyboard) + .gif (animation)
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.animation import FuncAnimation, PillowWriter

WALL, DOOR, PART, BANK, PANEL = '#bbb', '#e67e22', '#7f8c8d', '#9ecae1', '#c39bd3'
THROW = 6.0                                   # plunger pin travel [VERIFY]
SLIDE = 40.0                                  # door travel shown


def side(ax, s, p, bank_dy, bank_in=True):
    """s: door slide (+Y), p: pin up 0..1, bank_dy: bank drop below its seat."""
    ax.cla(); ax.set_aspect('equal'); ax.set_xlim(20, 108); ax.set_ylim(205, 140)
    ax.set_xlabel('Yr  (back of the box → right)', fontsize=8); ax.set_ylabel('d (down)', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title('Section along the door at X -42 (through the plunger)', fontsize=9)
    if bank_in:                               # the bank is not in this plane (|X| < 37): drawn dashed, behind
        ax.add_patch(Rectangle((32, 150 + bank_dy), 24, 30, fc=BANK, ec='k', ls='--', alpha=.5))
        ax.text(33, 156 + bank_dy, 'bank (behind,\n|X| < 37)', fontsize=7)
    ax.add_patch(Rectangle((28.5, 140), 2, 40, fc=WALL, ec='k')); ax.text(23, 145, 'battery\nrib', fontsize=7)
    ax.add_patch(Polygon([(38, 168), (50, 168), (50, 180), (46.7, 180), (46.7, 173), (41.3, 173), (41.3, 180), (38, 180)],
                         fc=WALL, ec='k')); ax.text(38.5, 166, 'core boss', fontsize=7)
    ax.add_patch(Rectangle((57, 140), 3.5, 40, fc=PANEL, ec='k')); ax.text(61, 146, 'back\npanel', fontsize=7)
    y0 = 30.9 + s
    ax.add_patch(Rectangle((y0, 180.2), 62 - 30.9, 3.0, fc=DOOR, ec='k'))
    ax.add_patch(Rectangle((58 + s, 183.2), 4, 5, fc=DOOR, ec='k')); ax.text(62.5 + s, 187.5, 'grip lip', fontsize=7)
    ax.add_patch(Rectangle((35 + s, 183.2), 18, 8, fc=DOOR, ec='k', alpha=.8))
    pin_top = 180.2 - THROW * p
    ax.add_patch(Rectangle((41.5 + s, pin_top), 5, 191.2 - pin_top, fc=PART, ec='k'))          # pin + body
    ax.add_patch(Rectangle((40 + s, 191.2 + THROW * (1 - p)), 8, 2, fc=PART, ec='k'))           # knob / ring stem
    ring_d = 193.2 + THROW * (1 - p)
    ax.add_patch(matplotlib.patches.Circle((44 + s, ring_d + 3.5), 3.5, fc='none', ec='k', lw=2))
    ax.text(49 + s, ring_d + 4, 'pull ring', fontsize=7)


def front(ax, p):
    ax.cla(); ax.set_aspect('equal'); ax.set_xlim(-58, -26); ax.set_ylim(196, 164)
    ax.set_xlabel('X', fontsize=8); ax.tick_params(labelsize=7)
    ax.set_title('Across the rail at Yr 44 (-X wall)', fontsize=9)
    ax.add_patch(Polygon([(-50, 164), (-46, 164), (-46, 175.6), (-47.6, 175.6), (-47.6, 178.4), (-46, 178.4), (-46, 180),
                          (-50, 180)], fc=WALL, ec='k')); ax.text(-49.6, 167, 'wall', fontsize=7)
    ax.add_patch(Polygon([(-44.2, 168), (-38, 168), (-38, 180), (-39.3, 180), (-39.3, 173), (-44.2, 173)],
                         fc=WALL, ec='k')); ax.text(-42, 166.5, 'core boss', fontsize=7)
    ax.annotate('pin hole opens into\nthe web slot (0.5)', (-44.4, 175), (-57, 171), fontsize=6,
                arrowprops=dict(arrowstyle='->', lw=.6))
    ax.add_patch(Polygon([(-49.8, 180.2), (-28, 180.2), (-28, 183.2), (-49.8, 183.2)], fc=DOOR, ec='k'))
    ax.add_patch(Polygon([(-45.6, 180.2), (-45.6, 178.0), (-47.2, 178.0), (-47.2, 176.0), (-44.2, 176.0), (-44.2, 180.2)],
                         fc=DOOR, ec='k')); ax.text(-57.5, 177, 'hook in\ngroove', fontsize=7)
    ax.add_patch(Rectangle((-51, 183.2), 18, 8, fc=DOOR, ec='k', alpha=.8))
    pin_top = 180.2 - THROW * p
    ax.add_patch(Rectangle((-44.5 + 0.0, pin_top), 5, 191.2 - pin_top, fc=PART, ec='k'))
    ax.text(-37, 177, 'pin ' + ('IN (locked)' if p > .95 else 'OUT' if p < .05 else 'moving'), fontsize=8,
            color='#c0392b' if p > .95 else 'k')


# key frames: (s, p, bank_dy, bank_in, caption)
STEPS = [
    (0, 1, 0, True, '1. Closed. The bank stands on the door; the door hangs on the wall ledges.\n'
                    'The plunger pin sits in the core boss, so the door cannot slide.'),
    (0, 0, 0, True, '2. Pull the ring down (~6 mm) and hold it: the pin leaves the boss.'),
    (12, 0, 0, True, '3. Slide the door out toward the back (+Y) by the grip lip.\n'
                     'Keep the rig upright; the back panel and rib keep the bank in place.'),
    (SLIDE, 0, 25, True, '4. The door clears the chamber: tip the bank out into your hand.\n'
                         'Note: the door also holds the back panel in its slots; with the door out, it can slide down.'),
    (SLIDE, 0, 0, True, '5. New bank in. Slide the door back in (ring pulled, or let the pin ride:\n'
                        'it drags on the boss underside until the hole).'),
    (0, 1, 0, True, '6. At the stop (rib) the spring snaps the pin into the boss: click = locked.\n'
                    'Tug the grip lip to check.'),
]


def storyboard():
    fig = plt.figure(figsize=(16, 26))
    for i, (s, p, dy, bi, cap) in enumerate(STEPS):
        y = 0.82 - i * 0.163
        a = fig.add_axes([0.04, y, 0.56, 0.12]); side(a, s, p, dy, bi)
        b = fig.add_axes([0.68, y, 0.28, 0.12]); front(b, p)
        fig.text(0.04, y + 0.145, cap, fontsize=10, va='top', ha='left',
                 bbox=dict(fc='#fff8e1', ec='#e0c060', boxstyle='round,pad=0.4'))
    fig.suptitle('v4 battery door A: opening and closing (sketch for approval, UNAPPROVED)', fontsize=14, y=0.995)
    fig.savefig(Path(__file__).with_suffix('.png'), dpi=70)


def animation():
    # timeline: hold, pull pin, slide out, drop bank, raise bank, slide in, pin snaps, hold
    seq = []
    seq += [(0, 1, 0, 'closed: pin locked')] * 8
    seq += [(0, 1 - t, 0, 'pull the ring') for t in np.linspace(0, 1, 10)]
    seq += [(SLIDE * t, 0, 0, 'slide the door out (+Y)') for t in np.linspace(0, 1, 20)]
    seq += [(SLIDE, 0, 25 * t, 'bank out into the hand') for t in np.linspace(0, 1, 12)]
    seq += [(SLIDE, 0, 25 * (1 - t), 'new bank in') for t in np.linspace(0, 1, 12)]
    seq += [(SLIDE * (1 - t), 0, 0, 'slide the door in, ring held') for t in np.linspace(0, 1, 20)]
    seq += [(0, t, 0, 'release: pin snaps in') for t in np.linspace(0, 1, 4)]
    seq += [(0, 1, 0, 'locked')] * 12
    fig = plt.figure(figsize=(12, 5))
    a = fig.add_axes([0.05, 0.12, 0.55, 0.75]); b = fig.add_axes([0.66, 0.12, 0.31, 0.75])
    title = fig.text(0.5, 0.95, '', ha='center', fontsize=13, weight='bold')

    def draw(i):
        s, p, dy, cap = seq[i]
        side(a, s, p, dy); front(b, p); title.set_text(cap)
    anim = FuncAnimation(fig, draw, frames=len(seq))
    anim.save(Path(__file__).with_suffix('.gif'), writer=PillowWriter(fps=8), dpi=70)


if __name__ == '__main__':
    storyboard()
    animation()
