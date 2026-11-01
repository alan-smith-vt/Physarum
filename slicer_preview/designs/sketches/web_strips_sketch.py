"""Web as horizontal strips (user 2026-11-01): elevation of the web plane (Yr ~ 0), seen from the Pi side (-Yr).
PROPOSAL for approval; numbers from pi_box_v4_core (flip). python web_strips_sketch.py -> web_strips_sketch.png"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

TUBE, XI, TOP, BOT = 23.5, 46.0, 4.0, 180.0
fig, ax = plt.subplots(figsize=(9, 11))
ax.add_patch(Rectangle((-TUBE, TOP), 2 * TUBE, BOT - TOP, fc='#ddd', ec='k'))
ax.text(0, 90, 'stick tube', ha='center', rotation=90)
web = dict(fc='#e67e22', ec='k', alpha=0.85)


def band(s, d0, d1, peak=True):
    """web band on side s from d0 (top) to d1; 45° roof on its top edge (print: the flange is down... top = small d)."""
    x0, x1 = sorted((s * TUBE, s * XI))
    if peak:   # chevron top: rises 45° from both ends toward the middle -> no flat bridge when printed flange-down
        xm, h = (x0 + x1) / 2, (x1 - x0) / 2
        ax.add_patch(Polygon([(x0, d1), (x0, d0 + h), (xm, d0), (x1, d0 + h), (x1, d1)], **web))
    else:
        ax.add_patch(Rectangle((x0, d0), x1 - x0, d1 - d0, **web))


# proposal
for s in (-1, 1):
    band(s, TOP, 66.0, peak=False)                          # A: top band, both halves (above the board's J1/J2 plugs)
band(-1, 158.0, 172.0)                                      # B: strip under the power board (J3 plug ends d 157)
band(1, 112.0, 180.0)                                       # C: +X solid from above the adapter's seat to the floor
band(-1, 176.0, 180.0, peak=False)                          # D: -X foot strip at the floor (collar/feet as now)
# parts
ax.add_patch(Rectangle((-43.5, 95), 1.6, 60, fc='#1e8449', ec='k'))
ax.add_patch(Rectangle((-41.9, 70), 19.9, 87, fc='none', ec='#1e8449', ls='--'))
ax.text(-33, 110, 'power board\n+ plugs\n(d 70..157)', ha='center', color='#1e8449')
ax.add_patch(Rectangle((30.2, 127), 15.4, 53, fc='none', ec='#2c3e50', lw=2))
ax.text(38, 150, 'GbE\nadapter\nglued\n(d 127..180)', ha='center', fontsize=8)
ax.plot([30.2, 45.6], [125.6, 125.6], 'k:', lw=1); ax.text(46.5, 125.6, 'seat ledge d 125.6', fontsize=7, va='center')
ax.plot([-29, -29, -27, -24], [132, 155, 157, 166], color='#c0392b', lw=2)
ax.text(-21, 168, 'button wire', color='#c0392b', fontsize=7)
for x0, x1, d0, d1 in [(30.0, 41.0, 24.0, 34.0), (28.5, 45.5, 76.0, 98.0)]:
    for s in (-1, 1):
        a, b = sorted((s * x0, s * x1))
        ax.add_patch(Rectangle((a, d0), b - a, d1 - d0, fc='none', ec='b', ls=':'))
ax.text(36, 30, 'old lidar\ncable slots', color='b', fontsize=7, ha='center')
for d in (66, 112, 158, 172, 176):
    ax.axhline(d, color='0.7', lw=0.5)
    ax.text(-58, d, f'd {d}', fontsize=7, va='center')
ax.text(-34.75, 35, 'A', fontsize=16, ha='center'); ax.text(34.75, 45, 'A', fontsize=16, ha='center')
ax.text(-34.75, 168, 'B', fontsize=16, ha='center'); ax.text(34.75, 118, 'C', fontsize=14, ha='center')
ax.text(-34.75, 179, 'D', fontsize=9, ha='center')
ax.set_xlim(-60, 60); ax.set_ylim(185, 0); ax.set_aspect('equal')
ax.set_xlabel('X (mm)   -X: power board side      +X: ethernet side'); ax.set_ylabel('depth d (flange 0 at top, floor 180)')
ax.set_title('Web as strips (PROPOSAL): orange = web, everything else open\nview from the Pi side; 45° chevron tops on strips that start mid-height')
fig.tight_layout(); fig.savefig(__file__.replace('.py', '.png'), dpi=110)
