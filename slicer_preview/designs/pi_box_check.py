"""Clash check + section images for the Pi box core.

    python -m designs.pi_box_check      (from slicer_preview/)

Samples every overlay's vertices (and triangle centroids) against the core's
solid function and reports how many land inside resin, per overlay, with the
depth range of the hits. Writes section PNGs (rig depth d) to out/pi_box_sections/.
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stl_slicer import load_stl                      # noqa: E402
from designs import pi_box_core as C                 # noqa: E402
from designs.pi_box_rig import ROOT                  # noqa: E402

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


def main():
    print(f"{'overlay':14s} {'samples':>8s} {'in resin':>9s}  depth range of hits")
    for ov in C.OVERLAYS:
        name = Path(ov['file']).stem
        t = load_stl(ROOT / ov['file']).astype(np.float64)
        pts = np.concatenate([t.reshape(-1, 3), t.mean(1)])
        pts = pts[np.unique(np.round(pts, 2), axis=0, return_index=True)[1]]
        if len(pts) > 60000:
            pts = pts[np.random.default_rng(0).choice(len(pts), 60000, replace=False)]
        hit = inside(pts)
        if name == "pi5":                     # PCB back face resting on the standoff posts is contact, not a clash
            hit &= np.abs(pts[:, 2] - C.PCB_YR) > 0.05
        rng = f"{pts[hit, 1].min():.1f}..{pts[hit, 1].max():.1f}" if hit.any() else "-"
        flag = "" if not hit.any() or name in EXPECTED else "  <-- CLASH"
        print(f"{name:14s} {len(pts):8d} {hit.sum():9d}  {rng}{flag}")

    try:
        from PIL import Image
    except ImportError:
        return
    out = ROOT / 'slicer_preview' / 'out' / 'pi_box_sections'
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
