"""v4 front panel covering the full front incl. the R30 rounds below the cameras (user request home:2125; PROPOSAL,
not modelled): dimensioned sketch. Plan section through the rounds, front elevation with the step, step section.

The panel is a prism along d (constant cross-section), so it slides along d like the flat panels:
  - wide part (d STEP..180): front flat + both rounds, edges at the round/side-wall tangent (Yr -32) with an L-hook
    (neck along the wall's inner face, hook turning +|X| into a channel in the wall);
  - centre strip (d 14..STEP): the existing narrow panel in the XE channels (|X| = 20), top tongue in the rim;
  - step: the wide part's top edge (d STEP, |X| 20..50) tongues into a groove in the fixed round shell above it.
Run: python -m designs.sketches.pi_box_v4_front_panel_wide_sketch -> .png next to this file
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Rectangle, Wedge
from designs import pi_box_bat_rig as R
from designs import pi_box_bat_panel_v3 as PV

X_OUT, X_IN, Y_MIN, Y_MAX, DEPTH = R.X_OUT, R.X_IN, R.Y_MIN, R.Y_MAX, R.DEPTH
R_FRONT, RC = 30.0, np.array([X_OUT - 30.0, Y_MIN + 30.0])     # (20, -32)
XE = RC[0]
FL_D = (21.0, 63.5)                    # cartridge flange d range (core)
PAD_D1 = FL_D[1] + 2.5                 # pad frame bottom (66)
STEP = 72.0                            # proposal: wide part from here to the floor (6 below the pad frame)
PT = 3.2                               # panel wall (outer face on the envelope: flush, no lip)
CL = PV.CL                             # 0.375 per side, as the L-hook channels
NECK_T, NECK_L, HOOK_W, HOOK_T = 1.6, 5.0, 2.0, 1.6
TONGUE_T, TONGUE_H = 1.6, 3.0
LENS = [(s * 45.1, -46.5) for s in (-1, 1)]
WELL_R, TUBE_T = 21.5, 2.0
PCB_YR = -31.0


def arc(c, r, a0, a1, n=40):
    a = np.radians(np.linspace(a0, a1, n))
    return np.c_[c[0] + r * np.cos(a), c[1] + r * np.sin(a)]


def panel_outline():
    """Wide part cross-section (X, Yr): outer = envelope front, inner = offset PT; ends at Yr -32 + the hooks."""
    outer = np.r_[arc((-RC[0], RC[1]), R_FRONT, 180, 270), arc(RC, R_FRONT, 270, 360)]
    inner = np.r_[arc(RC, R_FRONT - PT, 360, 270), arc((-RC[0], RC[1]), R_FRONT - PT, 270, 180)]
    return np.r_[outer, inner]


def hook(s):
    """Edge L-hook on side s: neck along the wall's inner face (X_IN - NECK_T .. X_IN), +Yr NECK_L, then the hook
    turning outward (+|X|) HOOK_T into the wall channel, HOOK_W along Yr."""
    y0 = RC[1]
    pts = [(X_IN - NECK_T, y0), (X_IN, y0), (X_IN, y0 + NECK_L), (X_IN + HOOK_T, y0 + NECK_L),
           (X_IN + HOOK_T, y0 + NECK_L + HOOK_W), (X_IN - NECK_T, y0 + NECK_L + HOOK_W)]
    return [(s * x, y) for x, y in pts]


def plan(ax):
    ax.set_title(f'Plan section at d 120 (looking from the flange toward the floor; front = -Yr down)', fontsize=9)
    # fixed core: side walls from the tangent back, back wall, tube
    for s in (-1, 1):
        ax.add_patch(Polygon([(s * X_IN, RC[1]), (s * X_OUT, RC[1]), (s * X_OUT, Y_MAX), (s * X_IN, Y_MAX)],
                             fc='#e8b04f', ec='k', lw=0.5))
        ch = [(s * (X_IN - 0.01), RC[1] + NECK_L - CL), (s * (X_IN + HOOK_T + CL), RC[1] + NECK_L - CL),
              (s * (X_IN + HOOK_T + CL), RC[1] + NECK_L + HOOK_W + CL), (s * (X_IN - 0.01), RC[1] + NECK_L + HOOK_W + CL)]
        ax.add_patch(Polygon(ch, fc='white', ec='k', lw=0.5))
    ax.add_patch(Wedge((0, 0), WELL_R + TUBE_T, 0, 360, width=TUBE_T, fc='#e8b04f', ec='k', lw=0.5))
    ax.plot([-28, 28], [PCB_YR, PCB_YR], color='#1e8449', lw=3)
    ax.text(0, PCB_YR - 2.5, 'Pi 5 PCB (Yr -31, X ±28; parts and plugs face the panel)', ha='center', fontsize=7, color='#1e8449')
    # panel
    ax.add_patch(Polygon(panel_outline(), fc='#5dade2', ec='k', lw=0.8, alpha=0.9))
    for s in (-1, 1):
        ax.add_patch(Polygon(hook(s), fc='#5dade2', ec='k', lw=0.8, alpha=0.9))
    # driver paths, old floor bosses
    for s in (-1, 1):
        ax.add_patch(Circle((s * 40, -40), (R.HEAD_D + R.RESIN_HOLE_COMP) / 2, fc='none', ec='#7d3c98', ls='--'))
        ax.add_patch(Circle((s * 31, -51), R.BOSS_R, fc='#e74c3c', ec='k', alpha=0.6))
        ax.add_patch(Circle((s * 42, -18), R.BOSS_R, fc='#2ecc71', ec='k', alpha=0.6))
    ax.text(40, -33.5, 'M4 driver path\n(±40, -40)', fontsize=6, ha='center', color='#7d3c98')
    ax.text(-31, -59.5 + 3, 'floor boss (±31,-51):\nloses its wall -> move', fontsize=6, ha='center', color='#c0392b')
    ax.text(-36, -11, 'proposed (±42,-18)\n[v4-floor]', fontsize=6, ha='center', color='#1e8449')
    # dims
    ax.annotate('', (RC[0], RC[1]), (RC[0] + R_FRONT * np.cos(np.radians(-45)), RC[1] + R_FRONT * np.sin(np.radians(-45))),
                arrowprops=dict(arrowstyle='<-', lw=0.6))
    ax.text(RC[0] + 6, RC[1] - 13, 'R30 outer\nR26.8 inner', fontsize=7)
    ax.text(0, Y_MIN - 2.6, f'panel wall {PT} (flush with the envelope; front flat |X| <= 20, rounds to the tangent Yr -32)',
            ha='center', fontsize=7)
    ax.text(X_OUT + 1, RC[1] + NECK_L, f'L-hook: neck {NECK_T} x {NECK_L} along the wall face,\nhook {HOOK_T} x {HOOK_W} into a channel '
            f'(CL {CL:.3f}),\nwall skin outside {X_OUT - X_IN - HOOK_T - CL:.2f}', fontsize=6.5, va='center')
    ax.text(0, 0, 'well tube', ha='center', fontsize=7)
    ax.set_xlim(-56, 84)
    ax.set_ylim(Y_MIN - 5, 10)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Yr')


def elevation(ax):
    ax.set_title('Front elevation (from -Y; d down). Blue = removable panel, orange = fixed core', fontsize=9)
    ax.add_patch(Rectangle((-X_OUT, 0), 2 * X_OUT, DEPTH, fc='#e8b04f', ec='k', lw=0.5))
    half_strip = XE + PV.NECK + PV.HOOK_T
    out = [(-half_strip, R.RIM_D), (half_strip, R.RIM_D), (half_strip, STEP), (X_OUT, STEP), (X_OUT, DEPTH),
           (-X_OUT, DEPTH), (-X_OUT, STEP), (-half_strip, STEP)]
    ax.add_patch(Polygon(out, fc='#5dade2', ec='k', lw=1.0, alpha=0.95))
    for x, _ in LENS:
        xs = np.sign(x)
        ax.add_patch(Rectangle((xs * 30 - 8 if xs > 0 else -X_OUT, FL_D[0]), 20 if xs > 0 else 20, FL_D[1] - FL_D[0],
                               fc='#2e86c1', ec='k', lw=0.5))
        ax.add_patch(Circle((xs * 45.1 if False else xs * 41.0, 35.0), 7.0, fc='#1b2631', ec='w', lw=0.5))
    ax.axhline(PAD_D1, color='#7d3c98', ls=':', lw=0.8)
    ax.text(X_OUT + 1, PAD_D1, f'cartridge pad frame ends d {PAD_D1:.0f}', fontsize=7, va='center', color='#7d3c98')
    ax.text(X_OUT + 1, STEP + 1.5, f'STEP d {STEP:.0f} (6 mm below it)', fontsize=7, va='center')
    ax.text(X_OUT + 1, R.RIM_D, f'centre strip top d {R.RIM_D:.0f} (tongue, home:2090)', fontsize=7, va='center')
    ax.text(0, 45, f'centre strip\n|X| <= {XE:.0f}\n(as now)', ha='center', fontsize=8)
    ax.text(0, 130, f'wide part: |X| <= {X_OUT:.0f} incl. both R30 rounds\nd {STEP:.0f}..{DEPTH:.0f}', ha='center', fontsize=8)
    ax.text(-40, 15, 'camera\ncartridge\n(fixed)', ha='center', fontsize=7, color='w')
    ax.set_xlim(-55, 110)
    ax.set_ylim(DEPTH + 5, -5)
    ax.set_aspect('equal')
    ax.set_xlabel('X (projected; the rounds wrap back to Yr -32)')
    ax.set_ylabel('d')


def step_section(ax):
    ax.set_title('Step: section across the round wall at |X| 35 (Yr-d), wide part top edge into the fixed shell', fontsize=9)
    y_out = RC[1] - np.sqrt(R_FRONT ** 2 - (35 - RC[0]) ** 2)
    y_in = RC[1] - np.sqrt((R_FRONT - PT) ** 2 - (35 - RC[0]) ** 2)
    ax.add_patch(Polygon([(y_out, STEP - 20), (y_in, STEP - 20), (y_in, STEP), (y_out, STEP)], fc='#e8b04f', ec='k', lw=0.6))
    yc = (y_out + y_in) / 2
    g = TONGUE_T / 2 + CL
    ax.add_patch(Polygon([(yc - g, STEP), (yc - g, STEP - TONGUE_H - 0.4), (yc, STEP - TONGUE_H - 0.4 - g),
                          (yc + g, STEP - TONGUE_H - 0.4), (yc + g, STEP)], fc='white', ec='k', lw=0.6))
    t = TONGUE_T / 2
    tt = STEP + 0.1 - TONGUE_H
    ax.add_patch(Polygon([(y_out, STEP + 15), (y_out, STEP + 0.1), (yc - t, STEP + 0.1), (yc - t, tt + t), (yc, tt),
                          (yc + t, tt + t), (yc + t, STEP + 0.1), (y_in, STEP + 0.1), (y_in, STEP + 15)],
                         fc='#5dade2', ec='k', lw=0.8))
    ax.text(yc, STEP - 12, 'fixed round shell\n(cartridge pad above)', ha='center', fontsize=7)
    ax.text(yc, STEP + 9, 'wide panel', ha='center', fontsize=7)
    ax.text(y_in + 1, STEP - 1.5, f'tongue {TONGUE_T} x {TONGUE_H}, groove {TONGUE_T + 2 * CL:.2f} x {TONGUE_H + 0.4}\n'
            '45° roof / tip, same as the rim tongue (home:2090)', fontsize=7)
    ax.set_xlim(y_out - 4, y_in + 22)
    ax.set_ylim(STEP + 16, STEP - 21)
    ax.set_aspect('equal')
    ax.set_xlabel('Yr')
    ax.set_ylabel('d')


if __name__ == '__main__':
    fig = plt.figure(figsize=(15, 13))
    plan(fig.add_subplot(2, 2, (1, 2)))
    elevation(fig.add_subplot(2, 2, 3))
    step_section(fig.add_subplot(2, 2, 4))
    fig.suptitle('v4 PROPOSAL: front panel over the full front incl. the R30 rounds below the cameras (slides in from '
                 'the floor end)', fontsize=11)
    plt.tight_layout()
    out = Path(__file__).with_suffix('.png')
    plt.savefig(out, dpi=100)
    print(out)
