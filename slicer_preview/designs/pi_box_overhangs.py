"""Overhang / island report for a Pi box core, in print orientation (flange on the plate).

    python -m designs.pi_box_overhangs [core_module]     (default pi_box_bat_core)

Slices the core at 0.2 mm and flags pixels with no resin under them within a 45° cone
(one pixel of growth per layer at this resolution). Clusters are reported by box-frame
position and depth, largest first, so each one can be named. Also writes a PNG per
cluster depth to out/<core>_overhangs/.
"""
import sys
import importlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
C = importlib.import_module('designs.' + (sys.argv[1] if len(sys.argv) > 1 else 'pi_box_bat_core'))

PX = 0.2                     # mm, square pixels and layers (45°: one pixel per layer)


def layer(d, xl, zl):
    return np.broadcast_to(C.core(xl, d, zl), (zl.shape[0], xl.shape[1]))


def dilate(m):
    out = m.copy()
    out[1:] |= m[:-1]; out[:-1] |= m[1:]
    out[:, 1:] |= m[:, :-1]; out[:, :-1] |= m[:, 1:]
    return out


def clusters(mask):
    """4-connected components (small images, plain BFS)."""
    lab = np.zeros(mask.shape, int)
    n = 0
    for r, c in zip(*np.nonzero(mask)):
        if lab[r, c]:
            continue
        n += 1
        stack = [(r, c)]
        lab[r, c] = n
        while stack:
            i, j = stack.pop()
            for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                if 0 <= a < mask.shape[0] and 0 <= b < mask.shape[1] and mask[a, b] and not lab[a, b]:
                    lab[a, b] = n
                    stack.append((a, b))
    return lab, n


def main():
    W, H = int(C.SIZE_X / PX), int(C.SIZE_Z / PX)
    xl = ((np.arange(W) + 0.5) * PX)[None, :]
    zl = ((np.arange(H) + 0.5) * PX)[:, None]
    prev = layer(PX / 2, xl, zl)
    found = []
    for k in range(1, int(C.DEPTH / PX)):
        d = (k + 0.5) * PX
        cur = layer(d, xl, zl)
        new = cur & ~dilate(prev)
        if new.sum() >= 4:
            lab, n = clusters(new)
            for i in range(1, n + 1):
                rr, cc = np.nonzero(lab == i)
                if len(rr) < 4:
                    continue
                px = np.stack([(cc + 0.5) * PX + C.X0, np.full(len(rr), d), (rr + 0.5) * PX + C.Z0], 1)
                rig = C.from_print(px)
                found.append((len(rr) * PX * PX, d, rig[:, 0].mean(), rig[:, 1].mean(),
                              np.ptp(rig[:, 0]) + PX, np.ptp(rig[:, 1]) + PX))
        prev = cur
    # merge reports from consecutive layers at the same spot
    found.sort(key=lambda f: (round(f[2] / 3), round(f[3] / 3), f[1]))
    merged = []
    for f in found:
        if merged and abs(merged[-1]['x'] - f[2]) < 3 and abs(merged[-1]['y'] - f[3]) < 3 and f[1] - merged[-1]['d1'] < 1.0:
            m = merged[-1]
            m['area'] += f[0]; m['d1'] = f[1]; m['w'] = max(m['w'], f[4]); m['t'] = max(m['t'], f[5])
        else:
            merged.append(dict(area=f[0], d0=f[1], d1=f[1], x=f[2], y=f[3], w=f[4], t=f[5]))
    merged.sort(key=lambda m: -m['area'])
    print(f"{'unsupported mm²':>16s}  {'depth (print height)':>20s}  {'box X':>7s} {'box Yr':>7s}  {'size X×Yr':>11s}")
    for m in merged[:25]:
        print(f"{m['area']:16.1f}  {m['d0']:8.1f} .. {m['d1']:7.1f}   {m['x']:7.1f} {m['y']:7.1f}  "
              f"{m['w']:5.1f}×{m['t']:<5.1f}")
    print(f"{len(merged)} overhang clusters in total")


if __name__ == '__main__':
    main()
