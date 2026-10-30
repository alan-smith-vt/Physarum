"""v4 panel top edge slotted into the core (user request home:2054; PROPOSAL, not modelled): dimensioned sketch.

Each removable panel (front -Y, back +Y) gets a tongue on its top (leading) edge; the core's panel-stop rim gets a
matching groove in its underside. The panel still slides in from the floor end along d: the tongue enters the groove
over the last TONGUE_H of travel and the panel stops on the rim face (d RIM_D) as now, not on the tongue tip.
Run: python -m designs.sketches.pi_box_v4_panel_tongue_sketch  ->  designs/sketches/pi_box_v4_panel_tongue_sketch.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from designs import pi_box_bat_rig as R
from designs import pi_box_bat_panel_v3 as PV

# proposal values
TONGUE_T, TONGUE_H = 1.6, 3.0          # tongue thickness (Yr) and height above the panel's top edge
TIP = TONGUE_T / 2                     # 45° pointed tip (lead-in), part of TONGUE_H
GROOVE_CL = PV.CL                      # per side, as the L-hook channels (0.2 + RESIN_HOLE_COMP / 2)
GROOVE_W = TONGUE_T + 2 * GROOVE_CL
GROOVE_H = TONGUE_H + 0.4              # 0.4 end gap: the panel still stops on the rim face
GROOVE_PEAK = GROOVE_W / 2             # 45° pointed roof (no flat bridge; core prints flange down, +d up)
NOTCH_HALF = (R.CABLE_NOTCH_W + R.RESIN_HOLE_COMP) / 2   # back rim's lidar-cable notch
TONGUE_X_BACK = R.X_IN - 2.5           # tongue |X| extent, back panel (channels start at X_IN)
TONGUE_X_FRONT = 20.0 - 2.5            # front panel: XE 20 (where the rounds start)
TONGUE_GAP_BACK = NOTCH_HALF + 1.5     # tongue stops this far from X 0 (cable notch)

Y1 = R.Y_MAX - R.PANEL_LIP             # slot outer face (back)
Y0 = Y1 - (R.GROOVE_W + R.RESIN_HOLE_COMP)
FACE_OUT = Y1 - PV.CL                  # panel outer face (as built: hook edges set by CL)
FACE_IN = FACE_OUT - R.PANEL_T
YC = (FACE_OUT + FACE_IN) / 2
RIM_D, TOP = R.RIM_D, R.FLANGE_PLATE
RIM_Y0 = R.Y_MAX - R.PANEL_LIP - (R.GROOVE_W + R.RESIN_HOLE_COMP) - 1.5


def dim(ax, p, q, text, off=(0, 0), color='k'):
    ax.annotate('', p, q, arrowprops=dict(arrowstyle='<->', color=color, lw=0.8))
    ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, fontsize=8, color=color, ha='center', va='center',
            bbox=dict(fc='white', ec='none', pad=0.5))


def section(ax):
    ax.set_title('Section Yr-d through the back rim (front rim mirrored)  |  d down = toward the floor', fontsize=9)
    # core: flange plate + rim
    ax.add_patch(Rectangle((RIM_Y0 - 6, 0), R.Y_MAX - RIM_Y0 + 6, TOP, fc='#d9a441', ec='k', lw=0.6))
    rim = [(RIM_Y0, TOP), (R.Y_MAX, TOP), (R.Y_MAX, RIM_D), (RIM_Y0, RIM_D)]
    ax.add_patch(Polygon(rim, fc='#e8b04f', ec='k', lw=0.6))
    # groove (white), pointed roof
    g0, g1 = YC - GROOVE_W / 2, YC + GROOVE_W / 2
    gtop = RIM_D - GROOVE_H
    ax.add_patch(Polygon([(g0, RIM_D), (g0, gtop), (YC, gtop - GROOVE_PEAK), (g1, gtop), (g1, RIM_D)], fc='white', ec='k', lw=0.6))
    # wall below the rim: slot (only PANEL_LIP outside at the corners; mid-panel the slot is open to the outside)
    # panel + tongue
    tt = RIM_D + 0.1 - TONGUE_H                     # tongue tip d (panel top edge at D0 = RIM_D + 0.1)
    t0, t1 = YC - TONGUE_T / 2, YC + TONGUE_T / 2
    panel = [(FACE_IN, RIM_D + 14), (FACE_IN, RIM_D + 0.1), (t0, RIM_D + 0.1), (t0, tt + TIP), (YC, tt), (t1, tt + TIP),
             (t1, RIM_D + 0.1), (FACE_OUT, RIM_D + 0.1), (FACE_OUT, RIM_D + 14)]
    ax.add_patch(Polygon(panel, fc='#5dade2', ec='k', lw=0.8, alpha=0.9))
    ax.text(YC, RIM_D + 9, 'back panel\n3.175', ha='center', fontsize=8)
    ax.text(RIM_Y0 - 3, TOP / 2, 'flange plate', fontsize=7, ha='center')
    ax.text(RIM_Y0 + 1.6, (TOP + RIM_D) / 2 + 2.5, 'rim\n(solid)', fontsize=7, ha='center')
    ax.text(RIM_Y0 - 3, RIM_D + 6, 'interior', fontsize=7, ha='center')
    ax.text(R.Y_MAX + 0.6, RIM_D + 6, 'outside', fontsize=7, rotation=90, va='center')
    # dims
    dim(ax, (t0, tt + 2.0), (t1, tt + 2.0), '', color='#1f618d')
    ax.text(t0 - 0.3, tt + 2.0, f'tongue {TONGUE_T}', fontsize=8, color='#1f618d', ha='right', va='center')
    dim(ax, (FACE_IN - 1.0, RIM_D + 0.1), (FACE_IN - 1.0, tt), f'{TONGUE_H}', off=(-0.9, 0), color='#1f618d')
    dim(ax, (g0, gtop - 2.6), (g1, gtop - 2.6), f'groove {GROOVE_W:.2f}', off=(0, -0.7))
    dim(ax, (g1 + 1.2, RIM_D), (g1 + 1.2, gtop), f'{GROOVE_H}', off=(0.9, 0))
    dim(ax, (R.Y_MAX + 0.3, TOP), (R.Y_MAX + 0.3, RIM_D), f'rim {RIM_D - TOP:.0f}', off=(1.4, 2.5))
    ax.text(YC, gtop - GROOVE_PEAK - 0.6, '45° roof', ha='center', fontsize=7)
    ax.text(YC + 2.4, tt + 0.6, '45° tip\n= lead-in', fontsize=7, color='#1f618d')
    ax.text(RIM_Y0 - 5.6, RIM_D - 0.5, f'panel stops on the rim face, d {RIM_D:.0f} (unchanged); 0.4 end gap at the tongue tip',
            fontsize=7)
    ax.text(RIM_Y0 - 5.6, RIM_D + 13.4, f'clearance per side {GROOVE_CL:.3f} = L-hook channel CL (0.2 + RESIN_HOLE_COMP/2)',
            fontsize=7)
    ax.set_xlim(RIM_Y0 - 6.5, R.Y_MAX + 3.5)
    ax.set_ylim(RIM_D + 14.5, -0.5)
    ax.set_aspect('equal')
    ax.set_xlabel('Yr (mm)')
    ax.set_ylabel('d (mm)')


def elevation(ax):
    ax.set_title('Tongue / groove extent along X (looking from outside at the panel top edges)', fontsize=9)
    for row, (name, xt, gap) in enumerate((('back +Y', TONGUE_X_BACK, TONGUE_GAP_BACK), ('front -Y', TONGUE_X_FRONT, 0.0))):
        y = -row * 6
        half = R.X_IN + PV.NECK + PV.HOOK_T if row == 0 else 20.0 + PV.NECK + PV.HOOK_T
        ax.add_patch(Rectangle((-half, y - 1.5), 2 * half, 1.5, fc='#5dade2', ec='k', lw=0.5))
        segs = [(-xt, -gap), (gap, xt)] if gap else [(-xt, xt)]
        for a, b in segs:
            ax.add_patch(Rectangle((a, y), b - a, 1.2, fc='#1f618d', ec='k', lw=0.5))
        ax.text(-half - 2, y - 0.5, name, ha='right', va='center', fontsize=8)
        ax.text(xt + 1, y + 0.6, f'|X| <= {xt:.1f}', fontsize=7, va='center')
        if gap:
            ax.text(0, y + 2.2, f'gap |X| < {gap:.1f}: lidar cable notch Ø{R.CABLE_NOTCH_W:.0f}', ha='center', fontsize=7)
        ax.text(half + 1, y - 0.8, 'L-hook\nchannel', fontsize=6, va='center')
    ax.set_xlim(-62, 62)
    ax.set_ylim(-9, 5)
    ax.set_aspect('equal')
    ax.set_yticks([])
    ax.set_xlabel('X (mm)')


if __name__ == '__main__':
    fig, axs = plt.subplots(2, 1, figsize=(11, 12), gridspec_kw=dict(height_ratios=[3.2, 1]))
    section(axs[0])
    elevation(axs[1])
    fig.suptitle('v4 PROPOSAL: panel top edge tongue into a groove under the flange rim (both panels), slides in from the floor',
                 fontsize=10)
    plt.tight_layout()
    out = Path(__file__).with_suffix('.png')
    plt.savefig(out, dpi=110)
    print(out)
