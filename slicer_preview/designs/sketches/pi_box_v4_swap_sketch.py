"""Concept sketch (for approval, not a model): v4 flip core, power board <-> UE300 side swap (v4-layout, 2026-10-29).

Pi fixed (flip, bottom-most: X -24..32, d 89..174, PCB back Yr -31, parts toward -Y).
  - Power board moves to the +X wall, parts facing -X, USB-C edge DOWN (J2 lands 2 mm above the Pi's USB-C plug
    envelope): Yr -43..3 (crosses the web, as v4's -X board), d 73..133, back face X +43.5, standoffs H1/H2.
  - UE300 + floor cradle + RJ45 opening mirror to -X (X -45.9..-29.9, Yr -27.6..-1.6, d 109..180).
  - Power button on the floor at (-36, +15) (home/v4-power-button); the -X wall switch hole goes.
    python pi_box_v4_swap_sketch.py -> pi_box_v4_swap_sketch.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

X_IN, X_OUT, Y_MIN, Y_MAX, DEPTH, FL = 46.0, 50.0, -62.0, 62.0, 180.0, 10.0
RC, RF = (20.0, -32.0), 30.0                     # front round centre (|X|, Yr), radius
TUBE = 23.5
PI = dict(x0=-24.0, x1=32.0, d0=89.0, d1=174.0, yb=-31.0, yf=-48.5)
USBC = dict(x0=30.0, x1=48.0, y0=-42.0, y1=-33.0, d0=157.0, d1=168.0)
PLUGS = [(11.0, 29.0), (-7.0, 9.0), (-25.0, -9.0)]   # Pi port plugs, d 61..86 (pointing up)
B = dict(y0=-43.0, top=73.0, w=46.0, h=60.0, back=43.5, t=1.6)


def bd(bx, by):
    """Board coords (power_board/gen_board.py) -> (Yr, d), USB-C edge DOWN, parts facing -X."""
    return B['y0'] + bx, B['top'] + B['h'] - by


CONN = {  # name: (bx0, bx1, by0, by1, height off the board, colour)
    'J2': (4.0, 16.0, -22.0, 7.5, 7.5, '#c0392b'),
    'J1': (30.0, 42.0, -22.0, 7.5, 7.5, '#e67e22'),
    'J3': (32.0, 42.0, 50.0, 62.0, 15.0, '#2471a3'),
    'J5': (7.0, 29.0, 52.0, 60.0, 14.0, '#8e44ad'),
    'J4': (0.5, 6.0, 34.0, 47.0, 14.0, '#16a085'),
}
HOLES = [(3.0, 57.0), (43.0, 47.0)]
UE = dict(x0=-45.9, x1=-29.9, y0=-27.6, y1=-1.6, d0=109.0)
BTN = (-36.0, 15.0)
SOCK = (20.0, 25.0)
CSI = (21.0, -35.5, (117.0, 125.0))
GPIO = (-22.5, -41.0, (116.0, 167.0))

fig = plt.figure(figsize=(21, 11.5))
fig.suptitle('v4 flip: power board -> +X wall (USB-Cs down), UE300 -> -X   [SKETCH for approval, mm, rig frame]',
             fontsize=14, weight='bold')


def frame(ax, title):
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.2)


def conn_boxes_elev(ax):
    for n, (a0, a1, b0, b1, h, c) in CONN.items():
        ya, db = bd(a0, b1)
        yb_, da = bd(a1, b0)
        ax.add_patch(Rectangle((ya, db), yb_ - ya, da - db, fc=c, alpha=0.8, ec='k'))
        ax.text((ya + yb_) / 2, (da + db) / 2, n, ha='center', va='center', fontsize=8, color='w', weight='bold')


# ── 1. +X wall elevation, seen from inside (-X side): Yr across, d down ──
ax = fig.add_axes([0.02, 0.07, 0.31, 0.84]); frame(ax, '1. +X wall from inside (Yr across, d down)'); ax.invert_yaxis()
ax.add_patch(Rectangle((Y_MIN, 0), Y_MAX - Y_MIN, FL, fc='#bbb', ec='k'))
ax.add_patch(Rectangle((-1.5, FL), 3, DEPTH - FL, fc='#ccc', ec='k')); ax.text(0.5, 22, 'web', rotation=90, fontsize=8)
ax.axvline(-32, color='#999', ls=':'); ax.text(-31.5, 186, 'round starts Yr -32', fontsize=7, color='#666')
ax.add_patch(Rectangle((30.5, 165), 31.5, 15, fc='#fde', ec='#c69', ls='--')); ax.text(46, 175, 'battery door', ha='center', fontsize=7)
ax.add_patch(Rectangle((-58, 21), 23, 42, fc='#d7bde2', ec='#7d3c98', alpha=0.6)); ax.text(-46.5, 42, '+X\ncartridge', ha='center', fontsize=7)
ax.add_patch(Rectangle((3.5, 14), 25, 152, fc='none', ec='#1f4e79', lw=2))
ax.text(16, 90, 'M12 trunk\n2 turns\n(unchanged)', ha='center', fontsize=8, color='#1f4e79')
ax.add_patch(Rectangle((-62, PI['d0']), PI['yb'] + 62, PI['d1'] - PI['d0'], fc='#7c7', ec='k', alpha=0.25, ls='--'))
ax.text(-60, 150, 'Pi (X <= 32)\nbeyond', fontsize=7)
ax.add_patch(Rectangle((USBC['y0'], USBC['d0']), 9, 11, fc='#e74c3c', alpha=0.7, ec='k'))
ax.text(-37.5, 174, 'Pi USB-C\nplug env.', ha='center', fontsize=7, color='#900')
ax.add_patch(Rectangle((B['y0'], B['top']), B['w'], B['h'], fc='#1e8449', alpha=0.85, ec='k'))
ax.text(-20, 103, 'POWER BOARD 46 x 60\nparts face -X\nUSB-C edge down', ha='center', color='w', fontsize=9, weight='bold')
conn_boxes_elev(ax)
for hx, hy in HOLES:
    y, d = bd(hx, hy)
    ax.add_patch(Circle((y, d), 2.5, fc='w', ec='k')); ax.text(y + 3, d + 1, 'H', fontsize=7)
ax.annotate('', (-43, 66), (3, 66), arrowprops=dict(arrowstyle='<->')); ax.text(-20, 64.5, 'Yr -43 .. +3', ha='center', fontsize=8)
ax.annotate('', (-60, 73), (-60, 133), arrowprops=dict(arrowstyle='<->'))
ax.text(-61, 103, 'd 73..133', rotation=90, va='center', ha='right', fontsize=8)
ax.text(-62, 160, 'J2 plug ends d 155:\n2 above USB-C env.', fontsize=7, color='#900')
ax.plot([-34, -34, -36], [155, 160, 162], color='#c0392b', lw=3)
ax.plot([-7, -7, 6], [155, 172, 172], color='#e67e22', lw=3)
ax.text(-5, 168, 'J1 lead -> web notch d 168..175', fontsize=7, color='#a04000')
ax.plot([-1, 10], [77, 77], color='#2471a3', lw=3); ax.text(4, 74, 'lidar pwr', fontsize=7, color='#2471a3')
ax.set_xlim(-64, 64); ax.set_ylim(190, -2)

# ── 2. plan (X across, Yr up) ──
ax = fig.add_axes([0.35, 0.07, 0.31, 0.84]); frame(ax, '2. Plan, looking down from the flange (X across, Yr up)')
q1 = np.linspace(0, -np.pi / 2, 30)
q2 = np.linspace(-np.pi / 2, -np.pi, 30)


def outline(r, inset):
    return ([(-X_OUT + inset, Y_MAX - inset), (X_OUT - inset, Y_MAX - inset)]
            + [(RC[0] + r * np.cos(t), RC[1] + r * np.sin(t)) for t in q1]
            + [(-RC[0] + r * np.cos(t), RC[1] + r * np.sin(t)) for t in q2])


ax.add_patch(Polygon(outline(RF, 0), fc='#ddd', ec='k'))
ax.add_patch(Polygon(outline(RF - 4, 4), fc='w', ec='k'))
ax.add_patch(Rectangle((-X_IN, -1.5), 2 * X_IN, 3, fc='#ccc', ec='k'))
ax.add_patch(Rectangle((-X_IN, 30.5), 2 * X_IN, 1, fc='#999', ec='k')); ax.text(0, 45, 'battery bay (door)', ha='center', fontsize=8)
ax.add_patch(Circle((0, 0), TUBE, fc='#ddd', ec='k')); ax.text(0, 0, 'tube', ha='center', fontsize=8)
ax.add_patch(Rectangle((PI['x0'], PI['yf']), 56, PI['yb'] - PI['yf'], fc='#7c7', ec='k', alpha=0.6))
ax.text(4, -42, 'Pi (flip, fixed)\nparts -Y', ha='center', fontsize=8)
ax.add_patch(Rectangle((USBC['x0'], USBC['y0']), 18, 9, fc='#e74c3c', alpha=0.7, ec='k'))
ax.text(39, -45, 'USB-C', ha='center', fontsize=7, color='#900')
ax.add_patch(Rectangle((B['back'] - B['t'], B['y0']), B['t'], B['w'], fc='#1e8449', ec='k'))
ax.add_patch(Rectangle((36, B['y0'] + 2), B['back'] - B['t'] - 36, 42, fc='#f1c40f', alpha=0.4, ec='none'))
for n, (a0, a1, b0, b1, h, c) in CONN.items():
    ya, yb_ = bd(a0, 0)[0], bd(a1, 0)[0]
    ax.add_patch(Rectangle((B['back'] - B['t'] - h, ya), h, yb_ - ya, fc=c, alpha=0.55, ec='k', lw=0.6))
    ax.text(B['back'] - B['t'] - h - 0.5, (ya + yb_) / 2, n, ha='right', va='center', fontsize=7)
ax.add_patch(Rectangle((UE['x0'], UE['y0']), 16, 26, fc='#ecf0f1', ec='k'))
ax.text(-37.9, -14.6, 'UE300\n+cradle\nRJ45 down', ha='center', va='center', fontsize=7)
ax.add_patch(Circle(BTN, 8, fc='#9b59b6', alpha=0.7, ec='k')); ax.text(BTN[0], BTN[1] - 14, 'button\n(-36,+15)', ha='center', fontsize=7)
ax.add_patch(Rectangle((SOCK[0] - 6, SOCK[1] - 4), 12, 8, fc='#ff66cc', ec='k'))
ax.text(SOCK[0], SOCK[1] + 6, 'bank socket -> X +20', ha='center', fontsize=7)
for x0 in (35.5, 25.5, -45.5):
    ax.add_patch(Rectangle((x0, 3.5), 10, 25, fc='none', ec='#1f4e79', lw=1.5))
ax.text(-45, 24, 'trunk\nd<136', fontsize=6, color='#1f4e79')
for p in [(-31, -51), (31, -51), (-44, 24), (44, 24)]:
    ax.add_patch(Circle(p, 4.5, fc='#aaa', ec='k'))
ax.add_patch(Circle(CSI[:2], 2, fc='#d35400')); ax.text(CSI[0], CSI[1] + 2.5, 'CSI', fontsize=7, color='#d35400')
ax.add_patch(Rectangle((GPIO[0] - 2.5, GPIO[1]), 5, 8, fc='#8e44ad')); ax.text(GPIO[0], GPIO[1] - 4, 'GPIO', ha='center', fontsize=7, color='#8e44ad')
ax.plot([30, 30, -27, -27, -22.5], [-25, -27, -27, -38, -38], color='#8e44ad', lw=2.5, label='J5 -> GPIO: across at d ~80 (above the Pi), down its -X edge (~120 mm)')
ax.plot([35, 35, -27.5, -27.5, -36, -36], [-40, -28, -28, 0, 0, 7], color='#16a085', lw=2.5, ls='--',
        label='J4 -> button: Pi/tube gap Yr -28, -X web notch d ~105, down the bay (~200 mm)')
ax.plot([40, 40, 26, 26, 20], [-7, 0, 0, 22, 25], color='#e67e22', lw=2.5, label='J1 <- bank socket: floor end, +X web notch, under the trunk U')
ax.plot([30, 30, 36], [-24, -38, -38], color='#c0392b', lw=2.5, label='J2 -> Pi USB-C: ~40 mm right-angle to right-angle')
ax.plot([26, 26, 35], [10, 0, -6], color='#2471a3', lw=2.5, label='lidar pwr: trunk breakout -> +X web notch -> J3')
ax.plot([-4, -4, -16, -30], [-40, -30, -26, -20], color='#7f8c8d', lw=2.5, label='UE300 cord -> USB3 (X -4, mouth d 86): 142 mm "uni" cord')
ax.legend(loc='upper center', fontsize=7.5, bbox_to_anchor=(0.5, -0.01), ncol=1)
ax.set_xlim(-54, 54); ax.set_ylim(-66, 66)

# ── 3. front elevation from -Y (X across, d down) ──
ax = fig.add_axes([0.68, 0.07, 0.31, 0.84]); frame(ax, '3. From -Y, front panel off (X across, d down)'); ax.invert_yaxis()
ax.add_patch(Rectangle((-X_OUT, 0), 2 * X_OUT, FL, fc='#bbb', ec='k'))
for s in (-1, 1):
    ax.add_patch(Rectangle((X_IN if s > 0 else -X_OUT, FL), 4, DEPTH - FL, fc='#bbb', ec='k'))
    ax.add_patch(Rectangle((s * 45.1 - 13.5, 21), 27, 42, fc='#d7bde2', ec='#7d3c98', alpha=0.6))
ax.text(0, 40, 'cartridges (front corners, d 21..63)', ha='center', fontsize=8, color='#7d3c98')
ax.add_patch(Rectangle((PI['x0'], PI['d0']), 56, 85, fc='#7c7', ec='k', alpha=0.6)); ax.text(4, 150, 'Pi (flip)', ha='center')
for x0, x1 in PLUGS:
    ax.add_patch(Rectangle((x0, 61), x1 - x0, 25, fc='#e67e22', alpha=0.6, ec='k'))
ax.text(2, 58, 'port plugs (up)', ha='center', fontsize=7)
ax.add_patch(Rectangle((USBC['x0'], USBC['d0']), 18, 11, fc='#e74c3c', alpha=0.7, ec='k'))
ax.add_patch(Rectangle((B['back'] - B['t'], B['top']), B['t'], 60, fc='#1e8449', ec='k'))
ax.add_patch(Rectangle((36, B['top'] + 2), 5.9, 56, fc='#f1c40f', alpha=0.5, ec='k', lw=0.5))
for n, (a0, a1, b0, b1, h, c) in CONN.items():
    db, da = bd(0, b1)[1], bd(0, b0)[1]
    ax.add_patch(Rectangle((B['back'] - B['t'] - h, db), h, da - db, fc=c, alpha=0.55, ec='k', lw=0.6))
    ax.text(B['back'] - B['t'] - h - 0.5, (da + db) / 2, n, ha='right', va='center', fontsize=7)
ax.add_patch(Rectangle((UE['x0'], UE['d0']), 16, 71, fc='#ecf0f1', ec='k')); ax.text(-37.9, 145, 'UE300', ha='center', rotation=90)
ax.add_patch(Rectangle((-44, 140), 16, 40, fc='#9b59b6', alpha=0.35, ec='k', ls='--')); ax.text(-36, 137, 'button (behind web)', ha='center', fontsize=7)
ax.plot([CSI[0]] * 2, CSI[2], color='#d35400', lw=4)
ax.plot([40, 28, 24, 21], [60, 66, 88, 117], color='#d35400', lw=2, ls='-.')
ax.plot([-40, -20, 14, 21], [60, 64, 88, 125], color='#d35400', lw=2, ls='-.')
ax.text(-30, 100, 'ribbons pass in front of the\nport plugs (Yr ~ -52) to CSI', fontsize=7, color='#d35400')
ax.plot([GPIO[0]] * 2, GPIO[2], color='#8e44ad', lw=4)
ax.plot([30, -27, -27, -22.5], [78, 80, 120, 125], color='#8e44ad', lw=2)
ax.plot([30, -27.5, -27.5], [93, 100, 105], color='#16a085', lw=2, ls='--')
ax.plot([35, 38, 37], [155, 156, 160], color='#c0392b', lw=3)
ax.annotate('', (32, 128), (36, 128), arrowprops=dict(arrowstyle='<->')); ax.text(34, 126, '4', ha='center', fontsize=8)
ax.text(10, 188, 'board parts X 36..41.9: 4 mm to the Pi edge (X 32) where Yr < -31', ha='center', fontsize=7)
ax.set_xlim(-54, 54); ax.set_ylim(192, -2)

notes = ('OPEN / CLEARANCES: (a) J4 plug (X 28..42, Yr -42..-37, d 86..99) is ~0 mm from the Pi RJ45 port body (X <= 28): '
         'needs a low XH plug or the lead bent flat. (b) J3 wires must exit +Y: a straight 22 mm stack reaches the tube (X 22.7). '
         '(c) Bank socket mirrors to X +20; its lead passes under the +X trunk U (lift that U ~6 mm). '
         '(d) -X trunk turn lifted to d 136 for the button: ~6 cm less slack. (e) The UE300 cord (38 mm) cannot reach the '
         'flipped USB3 port: use the "uni" 142 mm adapter (already the plan). (f) The -X wall switch hole and the -X web notch '
         'go (keep a small notch at d ~105 for J4); a mirrored +X web notch serves J3 / lidar power.')
fig.text(0.02, 0.005, notes, fontsize=8.5, wrap=True)
fig.savefig(Path(__file__).with_suffix('.png'), dpi=110)
