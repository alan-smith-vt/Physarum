"""Sketch (2026-11-03): final review of the rig-name font on door A's outside face (bottom view as modelled).
Contenders: Old English (round 2 winner), Stencil (round 1), Monotype Corsiva, Lucida Calligraphy, Algerian.
Left: whole door face; right: the 14 x 23 mm strip at true scale, red = strokes under 0.2 mm.
Run: python designs/sketches/pi_box_v4_door_name_final_sketch.py
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import importlib.util
_spec = importlib.util.spec_from_file_location('ns', Path(__file__).with_name('pi_box_v4_door_name_sketch.py'))
ns = importlib.util.module_from_spec(_spec)
import matplotlib.pyplot as _plt
_savefig = _plt.Figure.savefig
_plt.Figure.savefig = lambda *a, **k: None          # import the round-2 sketch for raster(); don't redraw it
_spec.loader.exec_module(ns)
_plt.Figure.savefig = _savefig
plt.close('all')

C = [('Old English (round 2 winner)', 'OLDENGL.TTF', ns.MIX), ('Stencil (round 1 winner)', 'STENCIL.TTF', ns.TEXT),
     ('Monotype Corsiva', 'MTCORSVA.TTF', ns.MIX), ('Lucida Calligraphy', 'LCALLIG.TTF', ns.MIX),
     ('Algerian', 'ALGER.TTF', ns.TEXT)]
BX, BY = ns.BOX_X, ns.BOX_Y
fig, axs = plt.subplots(len(C), 2, figsize=(16, 4.2 * len(C)), gridspec_kw=dict(width_ratios=[2.2, 1]))
for (name, font, text), (a0, a1) in zip(C, axs):
    m, thin, cap, ln = ns.raster(font, text)
    a0.add_patch(Rectangle((-50, 31.1), 100, 30.9, fc='#cfe0ef', ec='k'))
    a0.add_patch(Rectangle((-32, 36.8), 64, 13, fc='white', ec='#888'))
    a0.add_patch(Rectangle((-46, 58), 92, 4, fc='#1f77b4', alpha=0.45, ec='k'))
    a0.add_patch(Circle((41.1, 44), 1.6, fc='white', ec='k'))
    a0.text(0, 43.3, 'port window', ha='center', va='center', fontsize=8, color='#888')
    a0.imshow(np.where(np.rot90(m, 1)[:, ::-1], 0.0, np.nan), extent=[BX[0], BX[1], BY[0], BY[1]],
              cmap='gray', vmin=0, vmax=1, interpolation='nearest', zorder=3)
    a0.set_xlim(-52, 52); a0.set_ylim(29.5, 63.5); a0.set_aspect('equal'); a0.set_xticks([]); a0.set_yticks([])
    a0.set_title(f'{name}: cap {cap:.1f} mm, {100 * thin.sum() / m.sum():.0f}% under 0.2 mm', fontsize=11)
    img = np.ones(m.shape + (3,)); img[m] = (0.12, 0.12, 0.12); img[thin] = (0.85, 0.1, 0.1)
    a1.imshow(img, origin='lower', extent=[0, BY[1] - BY[0], 0, BX[1] - BX[0]], interpolation='nearest')
    a1.set_xticks([]); a1.set_yticks([]); a1.set_title('strip close-up (23 x 14 mm)', fontsize=9)
fig.suptitle('Door A outside face (table side), name opposite the thumb screw, 0.5 mm deep. Reading direction: '
             'check in the viewer', fontsize=12)
fig.tight_layout()
out = Path(__file__).with_suffix('.png')
fig.savefig(out, dpi=90, bbox_inches='tight')
print(out)
