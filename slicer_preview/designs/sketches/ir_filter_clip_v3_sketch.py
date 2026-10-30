"""Dimensioned sketch of IR filter clip v3 as modelled in designs/ir_filter_clip.py (2026-10-29, for approval
before reprint). All values are AS PRINTED unless marked 'drawn'."""
import numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge

RIB_PROUD, RIB_R, RIB_AT = 0.08, 0.25, 4.0
LIP, REC, KEY, LEAD, GUIDE = 0.30, 1.00, 4.0, 0.4, 0.25


def dim(ax, p0, p1, text, off=(0, 0), fs=8):
    ax.annotate("", p0, p1, arrowprops=dict(arrowstyle="<->", lw=0.8))
    ax.text((p0[0] + p1[0]) / 2 + off[0], (p0[1] + p1[1]) / 2 + off[1], text, ha="center", va="center",
            fontsize=fs, bbox=dict(fc="w", ec="none", pad=0.5))


fig = plt.figure(figsize=(18, 10))
fig.suptitle("IR filter clip v3 as MODELLED, for approval, NOT printed. mm, as printed unless 'drawn' "
             "(holes drawn +0.35 on dia, externals drawn 0.175/face small)", fontsize=11)

# 1: top view of one pocket (middle column)
ax = fig.add_subplot(1, 3, 1)
ax.set_title("TOP: one pocket, looking at the lens's rear face\nkey (red), crush rib on each fin face (dark red); middle column", fontsize=9)
ax.add_patch(Wedge((0, 0), 4.8, 0, 45, fc="#f4f4f4", ec="k", lw=0.6))
for a in (0, 45):
    ar = np.radians(a); r_ = np.array([np.cos(ar), np.sin(ar)]); n_ = np.array([-np.sin(ar), np.cos(ar)])
    ax.add_patch(Polygon([2.8 * r_ + .5 * n_, 4.8 * r_ + .5 * n_, 4.8 * r_ - .5 * n_, 2.8 * r_ - .5 * n_],
                         fc="#7cb87c", ec="k", lw=0.6))
g = 0.5 - 0.08                         # key face distance from the fin centre line (body fit -0.08)
rr = np.linspace(3.48, 4.58, 30)
side0 = [(np.sqrt(r * r - g * g), g) for r in rr]
side1 = [(r * np.cos(np.pi / 4 - np.arcsin(g / r)), r * np.sin(np.pi / 4 - np.arcsin(g / r))) for r in rr[::-1]]
ax.add_patch(Polygon(side0 + side1, fc="#e57373", ec="k", lw=0.6, alpha=0.9))
gc = g - RIB_PROUD + RIB_R
c0 = (np.sqrt(RIB_AT ** 2 - gc ** 2), gc)
ang = np.pi / 4 - np.arcsin(gc / RIB_AT); c1 = (RIB_AT * np.cos(ang), RIB_AT * np.sin(ang))
for c in (c0, c1):
    ax.add_patch(Circle(c, RIB_R, fc="#8b0000", ec="none"))
t = np.radians(np.linspace(0, 45, 100))
ax.plot(3.25 * np.cos(t), 3.25 * np.sin(t), "b", lw=1.2); ax.text(2.6, 1.05, "glass edge r 3.25", color="b", fontsize=7)
ax.plot(4.8 * np.cos(t), 4.8 * np.sin(t), "k--", lw=0.8)
dim(ax, (3.48, -0.3), (4.58, -0.3), "key r 3.48 .. 4.58", (0, -0.22))
dim(ax, (2.8, -0.95), (4.8, -0.95), "pocket: core face r 2.8 .. barrel wall r 4.8", (0, -0.22))
ax.annotate("key body overlaps each fin face by 0.08\n(cols 0.06 / 0.08 / 0.10; v2's 0.06 left ~0.03 play)",
            (4.3, 0.45), (4.4, -1.55), fontsize=7, arrowprops=dict(arrowstyle="->", lw=0.6), ha="center")
ax.annotate("crush rib dia 0.5 at r 4.0, full key length,\n0.08 proud of the key face\n"
            "-> 0.16 into the fin (cols 0.14 / 0.16 / 0.18)", c1, (1.7, 3.9), fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.text(1.7, 4.55, "fins assumed 1.0 thick, radial every 45 deg", fontsize=7)
ax.set_xlim(1.5, 5.4); ax.set_ylim(-1.9, 4.8); ax.set_aspect("equal"); ax.grid(alpha=0.3)

# 2: section through a key and the glass
ax = fig.add_subplot(1, 3, 2)
ax.set_title("SECTION through a key and the glass (print orientation: lip on the plate,\nkeys up into the lens)", fontsize=9)
rec_r = 6.45 / 2
ax.add_patch(Rectangle((2.8, 0), 4.6 - 2.8, LIP, fc="#bbb", ec="k", lw=0.6))
ax.add_patch(Polygon([(rec_r, LIP), (4.6, LIP), (4.6, LIP + REC), (rec_r + GUIDE, LIP + REC), (rec_r, LIP + REC - GUIDE)],
                     fc="#88a", ec="k", lw=0.6))
ax.add_patch(Rectangle((0, LIP), 3.25, 1.0, fc="#9ecae1", ec="b", lw=0.8, alpha=0.7))
ax.add_patch(Polygon([(3.48, LIP + REC), (4.58, LIP + REC), (4.58, LIP + REC + KEY - LEAD), (4.33, LIP + REC + KEY),
                      (3.73, LIP + REC + KEY), (3.48, LIP + REC + KEY - LEAD)], fc="#e57373", ec="k", lw=0.6))
ax.axhline(LIP + REC, color="g", ls=":", lw=1)
ax.text(0.05, LIP + REC + 0.1, "lens rear face (fin tops)", color="g", fontsize=7)
dim(ax, (5.1, 0), (5.1, LIP), "lip 0.30", (0.55, 0))
dim(ax, (5.1, LIP), (5.1, LIP + REC), "recess 1.00 drawn", (0.85, 0))
dim(ax, (5.1, LIP + REC), (5.1, LIP + REC + KEY), "key 4.0 (incl. 0.4 lead-in)", (1.1, 0))
dim(ax, (0, -0.3), (3.25, -0.3), "glass dia 6.5", (0, -0.22))
dim(ax, (0, -0.85), (rec_r, -0.85), "recess dia 6.40 / 6.45 / 6.50 (rows)", (0, -0.22))
dim(ax, (0, -1.4), (2.8, -1.4), "lip aperture dia 5.6", (0, -0.22))
ax.annotate("0.25 x 45 deg lead-in at the recess mouth\n(seats the glass square; no crush bumps,\n"
            "debris under the glass would tilt it)", (rec_r + 0.1, LIP + REC - 0.1), (0.1, 3.6), fontsize=7,
            arrowprops=dict(arrowstyle="->", lw=0.6))
ax.set_xlim(-0.1, 6.9); ax.set_ylim(-1.9, 6.0); ax.set_aspect("equal"); ax.grid(alpha=0.3)

# 3: table + open questions
ax = fig.add_subplot(1, 3, 3); ax.axis("off")
ax.text(0, 1, """PLATE: 3 x 3 clips, 48 x 48 mm, 5.3 tall

ROWS (-Z tab holes) = glass recess dia
  1 hole: 6.40   2: 6.45   3: 6.50     (glass 6.5)

COLUMNS (+X tab holes) = key fit per side
              body into fin   rib into fin
  1 hole:        0.06            0.14
  2 holes:       0.08            0.16
  3 holes:       0.10            0.18

KEYS: 8, r 3.48..4.58, 4.0 long incl. 0.4 lead-in

STILL OPEN BEFORE PRINTING
  Glass depth: recess drawn 1.00, you saw the glass
  sit only ~0.5 in. Calipers on a v2 clip:
   A. lip face to ring top (drawn 1.30)
   C. recess depth at the glass (drawn 1.00)
  -> recess = 1.00 + measured loss.
  E. pockets >= 4.4 deep behind the fin tops?
""", va="top", family="monospace", fontsize=9)
plt.tight_layout(rect=(0, 0, 1, 0.95))
plt.savefig("ir_filter_clip_v3_sketch.png", dpi=110)
