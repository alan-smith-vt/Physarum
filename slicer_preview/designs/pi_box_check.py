"""Clash check + section images for the Pi box core.

    python -m designs.pi_box_check [core_module]     (from slicer_preview/; default pi_box_bat_core)

Samples every overlay's vertices (and triangle centroids) against the core's
solid function and reports how many land inside resin, per overlay, with the
depth range of the hits. Writes section PNGs (rig depth d) to out/pi_box_sections/.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stl_slicer import load_stl                      # noqa: E402
import importlib                                     # noqa: E402
from designs.pi_box_bat_rig import ROOT              # noqa: E402

C = importlib.import_module('designs.' + (sys.argv[1] if len(sys.argv) > 1 else 'pi_box_bat_core'))

EXPECTED = {'box_spacers', 'monopod', 'standoffs', 'base_plate', 'coupler', 'mid360s'}   # outside the core or meant to sit in it


def inside(pts):
    """pts: (N, 3) print coords -> bool (N,) inside core resin."""
    out = np.zeros(len(pts), bool)
    d = pts[:, 1]
    ok = (d >= 0) & (d < C.DEPTH)
    for dv in np.unique(np.round(d[ok], 1)):
        sel = ok & (np.round(d, 1) == dv)
        p = pts[sel]
        xl = (p[:, 0] - C.X0)[None, :]
        zl = (p[:, 2] - C.Z0)[:, None]
        m = C.core(xl, float(dv), zl)
        out[np.where(sel)[0]] = np.diag(np.broadcast_to(m, (len(p), len(p))))
    return out


# ── Part vs part (2026-10-16): the printed floor and panels against the core, at nominal size and with both
#    sides grown by the print oversize. The overlay check above only ever saw bought parts, so the v2 floor's
#    collar went through the webs unnoticed. Each part: (name, rig-frame solid fn(X (nX,1), Yr (1,nY), d) -> bool,
#    X range, Yr range, d range). ──
GROW = (0.0, C.RESIN_HOLE_COMP / 2, C.RESIN_HOLE_COMP)    # per side: nominal, expected oversize, 2× margin


def _parts():
    name = C.__name__.split('.')[-1]
    L = importlib.import_module('designs.pi_box_bat_lids')
    out = []
    if name == 'pi_box_bat_core' or (name == 'pi_box_bat_core_v3' and not C.SLIDING_FLOOR):
        out.append(('floor (lids)', lambda X, Y, d: L._floor_solid(Y - C.Y_MIN, C.DEPTH + L.FLOOR_T - d, X + C.X_OUT),
                    (-C.X_OUT, C.X_OUT), (C.Y_MIN, C.Y_MAX), (C.DEPTH - 5.0, C.DEPTH - 0.05)))
    elif name == 'pi_box_bat_core_v3':
        F = importlib.import_module('designs.pi_box_bat_floor_v3')
        out.append(('floor (v3 sliding)', lambda X, Y, d: F.floor(Y - C.Y_MIN, C.DEPTH + F.FLOOR_T3 - d, X + F.TONGUE_X),
                    (-F.TONGUE_X - 1, F.TONGUE_X + 1), (C.Y_MIN, C.Y_MAX), (C.DEPTH + 0.05, C.DEPTH + F.FLOOR_T3 - 0.05)))
    if hasattr(C, 'PV'):                                           # v3: L-hook panels
        PV = C.PV
        for tag, face, sg in (('-Y', C._lo0 + PV.CL, 1), ('+Y', C._hi1 - PV.CL, -1)):
            fn = PV.panel_mask if sg > 0 else importlib.import_module('designs.pi_box_bat_back_panel_v3').back_mask
            out.append((f'panel {tag} (L-hook{"" if sg > 0 else ", back"})',
                        lambda X, Y, d, face=face, sg=sg, fn=fn: _panel_v3(fn, PV, X, Y, d, face, sg),
                        (-PV.HALF_W - 1, PV.HALF_W + 1), tuple(sorted((face - sg, face + sg * (L.PANEL_T + PV.HOOK + 1)))),
                        (PV.D0 + 0.05, PV.D1 - 0.05)))
    elif hasattr(C, '_lo0'):
        for tag, y0, y1 in (('-Y', C._lo0, C._lo1), ('+Y', C._hi0, C._hi1)):
            yp = (y0 + y1) / 2 - L.PANEL_T / 2                    # panel centred in its slot
            out.append((f'panel {tag}', lambda X, Y, d, yp=yp: _panel(L, X, Y, d, yp),
                        (-L.PANEL_W / 2 - 1, L.PANEL_W / 2 + 1), (y0 - 1, y1 + 1),
                        (C.RIM_D + 0.15, C.RIM_D + L.PANEL_H)))
    return out


def _panel_v3(fn, PV, X, Y, d, face, sg):
    s = np.zeros(np.broadcast_shapes(X.shape, Y.shape), bool)
    for j in range(Y.shape[1]):
        y = sg * (float(Y[0, j]) - face)
        if 0.0 <= y <= PV.PANEL_T + PV.HOOK:
            s[:, j] = np.broadcast_to(fn(X[:, 0], y, d), X.shape[:1])
    return s


def _panel(L, X, Y, d, yp):
    s = np.zeros(np.broadcast_shapes(X.shape, Y.shape), bool)
    for j in range(Y.shape[1]):
        y = float(Y[0, j]) - yp
        if 0.0 <= y <= L.PANEL_T:
            s[:, j] = np.broadcast_to(L.panel(X[:, 0] + L.PANEL_W / 2, y, np.asarray([d - C.RIM_D - 0.1]), grille=False), X.shape[:1])
    return s


def part_check(step=0.1, dd=0.5):
    from scipy.ndimage import binary_dilation
    parts = _parts()
    if not parts:
        return
    print(f"\npart vs part (mm² of overlap, worst layer; grown per side by {', '.join(f'{g:.3f}' for g in GROW)})")
    for name, fn, xr, yr, dr in parts:
        X = np.arange(*xr, step)[:, None]
        Y = np.arange(*yr, step)[None, :]
        worst = [(0.0, None)] * len(GROW)
        for d in np.arange(dr[0], dr[1], dd):
            p = np.broadcast_to(fn(X, Y, float(d)), (X.shape[0], Y.shape[1]))
            if not p.any():
                continue
            c = np.broadcast_to(np.asarray(C.core(Y - C.X0, float(d), -X - C.Z0), bool), p.shape)
            for i, g in enumerate(GROW):
                k = int(round(g / step))
                a = binary_dilation(p, iterations=k) if k else p
                b = binary_dilation(c, iterations=k) if k else c
                o = a & b
                area = o.sum() * step * step
                if area > worst[i][0]:
                    xs, ys = np.broadcast_to(X, o.shape)[o], np.broadcast_to(Y, o.shape)[o]
                    worst[i] = (area, f"d {d:.1f}, X {xs.min():.1f}..{xs.max():.1f}, Yr {ys.min():.1f}..{ys.max():.1f}")
        cells = '  '.join(f'{a:7.1f}' for a, _ in worst)
        flag = '  <-- CLASH' if worst[0][0] > 0 else ('  <-- tight' if worst[1][0] > 0 else '')
        where = next((w for a, w in worst if w), '')
        print(f"{name:20s} {cells}{flag}  {where}")


def main():
    print(f"{'overlay':14s} {'samples':>8s} {'in resin':>9s}  depth range of hits")
    for ov in C.OVERLAYS:
        name = Path(ov['file']).stem
        t = load_stl(ROOT / ov['file']).astype(np.float64)
        pts = np.concatenate([t.reshape(-1, 3), t.mean(1)])
        # plus points spread over the surface (area-weighted), so thin crossings aren't missed
        area = np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
        if area.sum() > 0:
            rng = np.random.default_rng(1)
            k = rng.choice(len(t), 20000, p=area / area.sum())
            u, v = rng.random((2, 20000))
            flip = u + v > 1
            u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
            tri = t[k]
            pts = np.concatenate([pts, tri[:, 0] + u[:, None] * (tri[:, 1] - tri[:, 0]) + v[:, None] * (tri[:, 2] - tri[:, 0])])
        pts = pts[np.unique(np.round(pts, 2), axis=0, return_index=True)[1]]
        if len(pts) > 60000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
        hit = inside(pts)
        if name == "pi5":                     # PCB back face resting on the standoff posts is contact, not a clash
            hit &= np.abs(C.from_print(pts)[:, 1] - C.PCB_YR) > 0.05
        if name == "cameras":                 # camera board face resting on its bosses
            hit &= np.abs(np.abs(C.from_print(pts)[:, 0]) - C.CAM_FACE) > 0.05
        if name == "power_switch":            # bezel seated on the wall's outer face
            hit &= np.abs(C.from_print(pts)[:, 0] - C.X_OUT) > 0.05
        if name == "base_plate":              # flange face against the plate
            hit &= np.abs(pts[:, 1]) > 0.05
        rng = f"{pts[hit, 1].min():.1f}..{pts[hit, 1].max():.1f}" if hit.any() else "-"
        flag = "" if not hit.any() or name in EXPECTED else "  <-- CLASH"
        print(f"{name:14s} {len(pts):8d} {hit.sum():9d}  {rng}{flag}")

    part_check()

    try:
        from PIL import Image
    except ImportError:
        return
    out = ROOT / 'slicer_preview' / 'out' / f'{C.__name__.split(".")[-1]}_sections'
    out.mkdir(parents=True, exist_ok=True)
    W, H = int(C.SIZE_X / 0.1), int(C.SIZE_Z / 0.1)
    xl = ((np.arange(W) + 0.5) * 0.1)[None, :]
    zl = ((np.arange(H) + 0.5) * 0.1)[:, None]
    for dv in (0.5, 3.0, 8.0, 17.5, 45.0, 62.0, 75.5, 100.0, 103.0):
        m = np.broadcast_to(C.core(xl, dv, zl), (H, W))
        Image.fromarray(np.where(m, 40, 235).astype(np.uint8)[::-1]).save(out / f'd{dv:05.1f}.png')
    print(f"sections -> {out}")


if __name__ == '__main__':
    main()
