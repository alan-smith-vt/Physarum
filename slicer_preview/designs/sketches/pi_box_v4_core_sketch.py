"""Concept sketch (for approval, not a model): v4 core with rounded front corners and camera lands (2026-10-29).

Plan section at the lens depth (d 35), a section along the +X camera's land normal, and the interior re-layout seen
from the front. Rig frame as the v3 core: X across (cameras at ±X), Yr (forward = -Y), d down from the plate face.

    python pi_box_v4_core_sketch.py -> pi_box_v4_core_sketch.png
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

X_IN, X_OUT, Y_MIN, Y_MAX, DEPTH = 46.0, 50.0, -62.0, 62.0, 180.0
WELL_R, TUBE_T = 21.5, 2.0
R_FRONT, LAND_OFF = 30.0, 1.0
C = np.array([X_OUT - R_FRONT, Y_MIN + R_FRONT])            # +X front round centre (X, Yr)
YAW, PITCH = np.radians(30), np.radians(24)
N = np.array([np.cos(YAW), -np.sin(YAW)])                     # land normal (plan)
T = np.array([np.sin(YAW), np.cos(YAW)])                      # land tangent
LAND = R_FRONT - LAND_OFF
P = C + LAND * N                                              # lens on the land (plan), d 35
SEAT = 22.9                                                   # cap seat plane, from C along N
PAD = SEAT - 3.0                                              # inner pad face round the opening
OPEN_T, OPEN_D = 13.5 + 0.375, (25.7 - 0.375, 59.5 + 0.375)   # opening (projection of the body along N)
FL_T, FL_D = 17.5, (21.7, 63.5)                               # cap / flange outline
PI_TOP, PI_SHIFT, PCB_YR = 55.0, 4.0, -31.0
BRD_TOP = 95.0

fig = plt.figure(figsize=(20, 11))


def mirror(pts):
    return [(-x, y) for x, y in pts]


def outline(r_in=0.0):
    """Core outline in plan (offset inward by r_in): square back, R30 front rounds."""
    pts = [(-(X_OUT - r_in), Y_MAX - r_in), (X_OUT - r_in, Y_MAX - r_in)]
    for a in np.linspace(0, -90, 31):
        pts.append(tuple(C + (R_FRONT - r_in) * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])))
    for a in np.linspace(-90, -180, 31):
        pts.append(tuple(C * [-1, 1] + (R_FRONT - r_in) * np.array([np.cos(np.radians(a)), np.sin(np.radians(a))])))
    return pts


def land_pt(n, t):
    return tuple(C + n * N + t * T)


# ── plan section at d 35 ──
ax = fig.add_subplot(1, 3, 1)
ax.add_patch(Polygon(outline(), closed=True, fc='#d6eaf8', ec='k', lw=1.5))
ax.add_patch(Polygon(outline(X_OUT - X_IN), closed=True, fc='white', ec='k', lw=1))
ax.add_patch(Circle((0, 0), WELL_R + TUBE_T, fc='#d6eaf8', ec='k'))
ax.add_patch(Circle((0, 0), WELL_R, fc='white', ec='k'))
for s in (-1, 1):
    m = (lambda pts: pts) if s > 0 else mirror
    ax.add_patch(Polygon(m([land_pt(PAD, -FL_T - 2), land_pt(PAD, FL_T + 2), land_pt(SEAT, FL_T + 2), land_pt(SEAT, -FL_T - 2)]),
                         fc='#aed6f1', ec='#2471a3', lw=1))                     # inner pad behind the seat
    cap = [land_pt(SEAT, -FL_T), land_pt(SEAT, FL_T)]
    for tt in np.linspace(FL_T, -FL_T, 25):
        cap.append(land_pt(min(np.sqrt(R_FRONT ** 2 - tt ** 2), LAND), tt))
    ax.add_patch(Polygon(m(cap), fc='#e74c3c', ec='#922b21', alpha=0.85))
    body = [land_pt(SEAT, -OPEN_T), land_pt(SEAT, OPEN_T), land_pt(SEAT - 18, OPEN_T), land_pt(SEAT - 18, -OPEN_T)]
    ax.add_patch(Polygon(m(body), fc='#f5b7b1', ec='#922b21', alpha=0.8))
    for da in (-62.5, 0, 62.5):                                                  # ±62.5° horizontal half-FOV
        a = YAW + np.radians(da)
        d = np.array([np.cos(a), -np.sin(a)]) * 40
        ax.plot(*zip(*m([tuple(P), tuple(P + d)])), color='#27ae60', lw=1 if da else 1.8, ls='--' if da else '-')
    ax.add_patch(Circle((s * 40, -40), 7.1, fc='none', ec='#7d3c98', lw=1.5))       # stub r 7.1 (d 0..12)
    ax.add_patch(Circle((s * 40, -40), 5.1, fc='none', ec='#7d3c98', ls=':', lw=1.5))  # Ø10.2 driver path
    ax.plot(*(P * [s, 1]), 'ko', ms=4)
ax.text(P[0] + 2, P[1] - 8, 'lens (±45.1, -46.5)\nd 35', fontsize=8)
ax.text(36, -22, 'M4 stub r7.1 (d 0..12)\n+ Ø10.2 driver path,\nclear with the cart out', fontsize=7, color='#7d3c98', ha='center')
ax.add_patch(Rectangle((-28 + PI_SHIFT, PCB_YR - 1.6), 56, 1.6, fc='#2ecc71', ec='g', alpha=0.4, ls='--'))
ax.text(PI_SHIFT, PCB_YR + 2, 'Pi PCB (below this section: d 55..140)', fontsize=7, ha='center', color='g')
ax.add_patch(Rectangle((-22.6, Y_MIN + 1.5), 45.2, 3.0, fc='#f9e79f', ec='#b7950b'))
ax.text(0, Y_MIN - 4, 'front panel X ±(20 + hook 2.6): L-slots where the rounds start', fontsize=7, ha='center', color='#7d6608')
ax.add_patch(Rectangle((-X_OUT + 1.5, Y_MAX - 4.5), 2 * X_OUT - 3, 3.0, fc='#f9e79f', ec='#b7950b'))
ax.text(0, Y_MAX + 3, 'back panel as v3 (square corners, L-slots)', fontsize=7, ha='center', color='#7d6608')
ax.annotate('', xy=(0, -77), xytext=(0, -67), arrowprops=dict(arrowstyle='->', lw=2))
ax.text(2, -76, 'FORWARD (-Y)', fontsize=9, weight='bold')
ax.text(C[0] - 6, C[1] + 3, f'R{R_FRONT:.0f}, centre ({C[0]:.0f}, {C[1]:.0f})', fontsize=7)
ax.text(-48, 8, 'square back corners\n(battery Yr 30.8..55.8)', fontsize=7)
ax.set_xlim(-70, 70); ax.set_ylim(-80, 70); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('X (mm)'); ax.set_ylabel('Yr (mm)')
ax.set_title('PLAN section at d 35 (lens depth), looking down; up = +Y (back)')

# ── section along the +X land normal through the lens axis (n outward, z = -d up) ──
ax = fig.add_subplot(1, 3, 2)


def nd(n, d):
    return (n, -d)


ax.add_patch(Polygon([nd(PAD, FL_D[0] - 4), nd(SEAT, FL_D[0] - 4), nd(SEAT, FL_D[1] + 4), nd(PAD, FL_D[1] + 4)],
                     fc='#aed6f1', ec='#2471a3'))                                # pad frame
ax.add_patch(Rectangle((R_FRONT - 4, -100), 4, 100 - (FL_D[1] + 4), fc='#d6eaf8', ec='k'))
ax.add_patch(Rectangle((R_FRONT - 4, -(FL_D[0] - 4)), 4, FL_D[0] - 4 - 4, fc='#d6eaf8', ec='k'))
ax.add_patch(Rectangle((SEAT, -(FL_D[0] - 0.01)), R_FRONT - SEAT, 0.01, ec='k'))
ax.add_patch(Rectangle((-20, 0), 60, 6.2, fc='#bdc3c7', ec='k'))
ax.text(-18, 2, 'Al base plate z 0..6.2', fontsize=7)
ax.add_patch(Rectangle((-20, -4), 50, 4, fc='#d6eaf8', ec='k'))
sn = (np.array([40, -40]) - C) @ N
ax.add_patch(Rectangle((sn - 7.1, -12), 14.2, 12, fc='#d2b4de', ec='#7d3c98', alpha=0.7))
ax.add_patch(Rectangle((sn - 5.1, -100), 10.2, 88, fc='none', ec='#7d3c98', ls=':'))
ax.text(sn, -10, 'M4 stub\nd 0..12', fontsize=7, ha='center')
ax.text(sn - 4, -95, 'driver\npath', fontsize=6, color='#7d3c98')
ax.add_patch(Polygon([nd(SEAT, FL_D[0]), nd(LAND, FL_D[0]), nd(LAND, FL_D[1]), nd(SEAT, FL_D[1])],
                     fc='#e74c3c', ec='#922b21', alpha=0.85))
f = np.array([np.cos(PITCH), -np.sin(PITCH)])                # (n, d): out and up
u = np.array([-np.sin(PITCH), -np.cos(PITCH)])
L = np.array([LAND, 35.0])
Bc = L - 18 * f


def q(c, b):
    p = Bc + c * f + b * u
    return (p[0], -p[1])


ax.add_patch(Polygon([q(-5, -16.6), q(-5, 9.25), (SEAT, -OPEN_D[0]), (SEAT, -OPEN_D[1])],
                     fc='#f5b7b1', ec='#922b21', alpha=0.8))
ax.plot(*zip(q(0, -15.6), q(0, 8.25)), color='#145a32', lw=3)
ax.add_patch(Polygon([q(0, -7), q(0, 7), q(18, 7), q(18, -7)], fc='#566573', ec='k'))
for da in (-39, 0, 39):                                       # vertical half-FOV 39°
    a = PITCH + np.radians(da)
    e = np.array([np.cos(a), np.sin(a)]) * 30
    ax.plot([L[0], L[0] + e[0]], [-L[1], -L[1] + e[1]], color='#27ae60', ls='-' if da == 0 else '--')
ax.text(L[0] + 12, -L[1] + 9, 'axis +24°', color='#27ae60', fontsize=8)
ax.text(Bc[0] - 22, -Bc[1] + 2, 'board 25×23.85\npitched 24°\n(ribbon edge down)', fontsize=7, color='#145a32')
ax.text(PAD - 4, -FL_D[1] - 10, f'cap seat n {SEAT}, pad face n {PAD}\n(the round wall alone is n 26..30)', fontsize=7)
ax.text(LAND + 0.5, -FL_D[0] + 3, f'land n {LAND:.0f}', fontsize=7)
ax.text(-4, -78, f'cap d {FL_D[0]:.1f}..{FL_D[1]:.1f}: top 9.7 below the stub\n'
        f'opening d {OPEN_D[0]:.1f}..{OPEN_D[1]:.1f}, ±{OPEN_T:.2f} along the land\n'
        'body reaches 24.8 in from the land; slides in along the\n'
        'land normal (straight walls, no undercut); cap prints flat-face down', fontsize=7)
ax.annotate('', xy=(R_FRONT + 7, -50), xytext=(R_FRONT + 1, -50), arrowprops=dict(arrowstyle='<-', lw=1.5))
ax.text(R_FRONT + 2, -54, 'insert', fontsize=7)
ax.set_xlim(-5, 45); ax.set_ylim(-82, 10); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('n from the round centre (mm), outward along the land normal (yaw 30° fwd)'); ax.set_ylabel('z = -d (mm)')
ax.set_title('SECTION along the +X land normal through the lens')

# ── front view (from -Y, panel off): interior re-layout ──
ax = fig.add_subplot(1, 3, 3)
ax.add_patch(Rectangle((-X_OUT, -DEPTH), 2 * X_OUT, DEPTH, fc='none', ec='k', lw=1.5))
ax.add_patch(Rectangle((-X_IN, -DEPTH), 2 * X_IN, DEPTH - 10, fc='white', ec='k', lw=0.8))
for s in (-1, 1):
    ax.add_patch(Rectangle((s * 32 - 13, -59.9), 26, 34.6, fc='#f5b7b1', ec='#922b21', alpha=0.8))
    ax.add_patch(Rectangle((s * 40 - 5.1, -DEPTH), 10.2, DEPTH - 12, fc='none', ec='#7d3c98', ls=':'))
    ax.add_patch(Rectangle((s * 40 - 7.1, -12), 14.2, 12, fc='#d2b4de', ec='#7d3c98'))
ax.text(32, -22, 'cartridges d 26..60', fontsize=7, ha='center')
ax.add_patch(Rectangle((-28 + PI_SHIFT, -(PI_TOP + 85)), 56, 85, fc='#2ecc71', ec='g', alpha=0.45))
ax.text(PI_SHIFT, -(PI_TOP + 40), f'Pi 5\nd {PI_TOP:.0f}..{PI_TOP + 85:.0f}\nPI_SHIFT +{PI_SHIFT:.0f}\n(v3: d 14, +6)',
        ha='center', fontsize=8)
ax.add_patch(Rectangle((-28 + PI_SHIFT - 18, -(PI_TOP + 17)), 18, 11, fc='#e74c3c', alpha=0.6))
ax.text(-44, -(PI_TOP + 3), 'USB-C R/A + adapter', fontsize=6)
pm = PI_TOP + 88
for x0, x1, L in [(-20, -4, 30), (0, 14, 25), (18, 32, 25)]:
    ax.add_patch(Rectangle((x0 + PI_SHIFT - 6, -(pm + L)), x1 - x0, L, fc='#e67e22', alpha=0.6))
ax.text(5, -(pm + 20), f'rigid plugs d {pm:.0f}..{pm + 30:.0f}', fontsize=7, ha='center')
ax.add_patch(Rectangle((-X_IN + 2.5, -(BRD_TOP + 60)), 1.6, 60, fc='#1e8449'))
ax.text(-X_IN + 5, -(BRD_TOP + 25), f'power board\n(-X wall)\nd {BRD_TOP:.0f}..{BRD_TOP + 60:.0f}\n(v3: 80..140)', fontsize=7)
ax.add_patch(Circle((-X_IN + 3, -168), 8, fc='#f7dc6f', ec='k'))
ax.text(-X_IN + 12, -170, 'switch d 168', fontsize=7)
ax.add_patch(Rectangle((29.9, -DEPTH), 16, 71, fc='#ecf0f1', ec='grey'))
ax.text(38, -150, 'UE300\n(Yr > -28)', fontsize=6, ha='center')
ax.text(0, -DEPTH - 9, 'floor: home/v4-floor (bottom edge d 180, rounded-front outline)', fontsize=7, ha='center')
ax.set_xlim(-60, 60); ax.set_ylim(-195, 10); ax.set_aspect('equal'); ax.grid(alpha=0.3)
ax.set_xlabel('X (mm)'); ax.set_ylabel('z = -d (mm)')
ax.set_title('FRONT view (from -Y, panel off): interior re-layout')

fig.suptitle('v4 core concept (for approval): R30 front rounds, vertical camera lands at yaw 30°, 24° board pitch '
             'inside a flush-capped cartridge; Pi lowered to d 55', fontsize=13)
fig.tight_layout()
fig.savefig(Path(__file__).with_suffix('.png'), dpi=110)
