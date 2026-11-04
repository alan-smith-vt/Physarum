"""Sketch (2026-11-03): rig name engraved in door A's outside face, -X strip beside the port window (opposite the thumb
screw). Left: the face (bottom view as modelled). Right: font candidates at true size in the 14 x 23 mm strip; red =
strokes thinner than MIN_STROKE (would print poorly / fill in at 0.5 deep).
Run: python designs/sketches/pi_box_v4_door_name_sketch.py -> pi_box_v4_door_name_sketch.png
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.textpath import TextPath
from matplotlib.font_manager import FontProperties
from scipy.ndimage import binary_opening
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from designs.print_post import text_inside

TEXT = 'ALNILAM'
BOX_X, BOX_Y = (-48.0, -34.0), (33.0, 56.0)
MIN_STROKE, P = 0.2, 0.01                    # user 2026-11-03: accepts strokes down to 0.2 mm
F = r'C:\Windows\Fonts' + '\\'
# round 2 (user 2026-11-03): Stencil won round 1 (Arial/Bahnschrift/Agency/Impact/... ); scripts + fancier fonts.
# (name, file, text): scripts in mixed case.
MIX = 'Alnilam'
FONTS = [('Stencil (round 1 winner)', 'STENCIL.TTF', TEXT),
         ('Brush Script', 'BRUSHSCI.TTF', MIX), ('Edwardian Script', 'ITCEDSCR.TTF', MIX),
         ('Vladimir Script', 'VLADIMIR.TTF', MIX), ('Kunstler Script', 'KUNSTLER.TTF', MIX),
         ('Lucida Handwriting', 'LHANDW.TTF', MIX), ('Segoe Script Bold', 'segoescb.ttf', MIX),
         ('Script MT Bold', 'SCRIPTBL.TTF', MIX), ('Monotype Corsiva', 'MTCORSVA.TTF', MIX),
         ('Mistral', 'MISTRAL.TTF', MIX), ('Magneto', 'MAGNETOB.TTF', MIX),
         ('Harlow Solid', 'HARLOWSI.TTF', MIX), ('Lucida Calligraphy', 'LCALLIG.TTF', MIX),
         ('Old English', 'OLDENGL.TTF', MIX), ('Blackadder', 'ITCBLKAD.TTF', MIX),
         ('Algerian', 'ALGER.TTF', TEXT), ('Showcard Gothic', 'SHOWG.TTF', TEXT),
         ('Bernard MT Condensed', 'BERNHC.TTF', TEXT)]


def raster(font, text=TEXT):
    tp = TextPath((0, 0), text, size=10.0, prop=FontProperties(fname=F + font))
    v = tp.vertices
    u0, u1, v0, v1 = v[:, 0].min(), v[:, 0].max(), v[:, 1].min(), v[:, 1].max()
    L, Wd = BOX_Y[1] - BOX_Y[0], BOX_X[1] - BOX_X[0]
    k = min(L / (u1 - u0), Wd / (v1 - v0))
    us, ws = np.arange(0, L, P), np.arange(0, Wd, P)
    U, Wm = np.meshgrid(us, ws)
    m = text_inside(tp, np.c_[(U - L / 2).ravel() / k + (u0 + u1) / 2,
                                 (Wm - Wd / 2).ravel() / k + (v0 + v1) / 2]).reshape(U.shape)
    r = int(round(MIN_STROKE / 2 / P))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    thick = binary_opening(m, structure=xx ** 2 + yy ** 2 <= r * r)
    return m, m & ~thick, (v1 - v0) * k, (u1 - u0) * k


fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(6, 4, width_ratios=[1.6, 1, 1, 1])
ax = fig.add_subplot(gs[:, 0])
ax.add_patch(Rectangle((-50, 31.1), 100, 30.9, fc='#e4ecf4', ec='k'))
ax.add_patch(Rectangle((-32, 36.8), 64, 13, fc='white', ec='#888', ls=':'))
ax.text(0, 43.3, 'port window', ha='center', va='center', fontsize=8, color='#888')
ax.add_patch(Rectangle((-46, 58), 92, 4, fc='#1f77b4', alpha=0.5, ec='k'))
ax.text(0, 60, 'standing lip (7.6)', ha='center', va='center', fontsize=7)
ax.add_patch(Circle((41.1, 44), 1.6, fc='white', ec='r'))
ax.text(41.1, 40.5, 'thumb\nscrew', ha='center', va='top', fontsize=7, color='r')
m, thin, cap, ln = raster('STENCIL.TTF')
# draw the text in place (runs along Yr = up the page here: rotate the raster)
ax.imshow(np.where(np.rot90(m, 1)[:, ::-1], 0.0, np.nan), extent=[BOX_X[0], BOX_X[1], BOX_Y[0], BOX_Y[1]],
          cmap='gray', vmin=0, vmax=1, interpolation='nearest', zorder=3)
ax.add_patch(Rectangle((BOX_X[0], BOX_Y[0]), BOX_X[1] - BOX_X[0], BOX_Y[1] - BOX_Y[0], fill=False, ec='g', ls='--'))
ax.set_xlim(-52, 52); ax.set_ylim(29, 64); ax.set_aspect('equal')
ax.set_xlabel('rig X (mm)'); ax.set_ylabel('rig Yr (mm)')
ax.set_title(f'Door A outside face, bottom view (as modelled; reading direction TBC)\n'
             f'"{TEXT}" Stencil, cap {cap:.1f} mm x {ln:.1f} mm, 0.5 deep (green: name strip)', fontsize=10)

for i, (name, font, text) in enumerate(FONTS):
    a = fig.add_subplot(gs[i // 3, 1 + i % 3])
    m, thin, cap, ln = raster(font, text)
    img = np.ones(m.shape + (3,))
    img[m] = (0.15, 0.15, 0.15)
    img[thin] = (0.85, 0.1, 0.1)
    a.imshow(img, origin='lower', extent=[0, BOX_Y[1] - BOX_Y[0], 0, BOX_X[1] - BOX_X[0]], interpolation='nearest')
    a.set_title(f'{i + 1}. {name}   cap {cap:.1f}   thin {100 * thin.sum() / max(m.sum(), 1):.0f}%', fontsize=9)
    a.set_xticks([]); a.set_yticks([])
fig.text(0.5, 0.01, f'Red = strokes under {MIN_STROKE} mm (risk of filling in / breaking). All fitted to the 14 x 23 mm '
         'strip; wide fonts get shorter letters.', ha='center', fontsize=9)
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=110, bbox_inches='tight')
print(out)
