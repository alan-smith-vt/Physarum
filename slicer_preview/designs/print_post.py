"""Print-only post-processing for implicit (solid_fn) pieces, on CPU or GPU.

bevel_piece(piece, ef, fill_holes=True, extra=None) returns a copy of the piece whose slices get
  - an elephant's-foot bevel: for layers below ef mm, keep only pixels >= (ef - y) inside the section at height ef
    (45 deg), measured from the outline (holes filled first, so windows/recesses on the plate face aren't bevelled);
  - extra(x, y, z, xp) -> bool, OR-ed in after the bevel (e.g. needle supports).
Both `make_slice` (CPU: viewer + to_goo) and `slice_xp(layer, xp)` (to_goo --gpu) are provided. `mask_np` tells
to_goo --gpu to point the design modules' `np` at cupy, since solid_fn is numpy-style mask code.
"""
import numpy as _np    # not `np`: to_goo --gpu repoints every designs.* module's `np` at cupy


def _nd(xp):
    if xp is _np:
        import scipy.ndimage as nd
    else:
        import cupyx.scipy.ndimage as nd
    return nd


def bevel_piece(piece, ef, fill_holes=True, extra=None):
    import geometry as g
    fn = piece['solid_fn']
    W, H = piece['W'], piece['H']
    k = int(ef / g.LY_MM)                                # first layer whose centre is >= ef
    cache = {}

    def grid(xp):
        if xp not in cache:
            cache[xp] = (xp.asarray(((_np.arange(W) + 0.5) * g.PX_MM)[None, :]),
                         xp.asarray(((_np.arange(H) + 0.5) * g.PZ_MM)[:, None]))
        return cache[xp]

    def solid(layer, xp):
        x, z = grid(xp)
        return xp.broadcast_to(fn(x, (layer + 0.5) * g.LY_MM, z), (H, W))

    def slice_xp(layer, xp):
        y = (layer + 0.5) * g.LY_MM
        m = solid(layer, xp)
        if y < ef:
            key = ('dist', xp)
            if key not in cache:
                nd = _nd(xp)
                ref = solid(k, xp)
                if fill_holes:
                    ref = nd.binary_fill_holes(ref)
                cache[key] = nd.distance_transform_edt(ref, sampling=(g.PZ_MM, g.PX_MM))
            m = m & (cache[key] >= ef - y)
        if extra is not None:
            x, z = grid(xp)
            m = m | xp.broadcast_to(extra(x, y, z, xp), (H, W))
        return m.astype(xp.uint8) * 255

    out = dict(piece, slice_xp=slice_xp, make_slice=lambda layer, *a, **kw: slice_xp(layer, _np), mask_np=True)
    return out


def text_inside(tp, pts):
    """Points inside a matplotlib TextPath, even-odd per closed loop (2026-11-03): Path.contains_points on the whole
    compound path fills letter counters (the inside of A, a, e) for some fonts; XOR over the loops keeps them open."""
    from matplotlib.path import Path as _MPath
    inside = _np.zeros(len(pts), bool)
    for poly in tp.to_polygons(closed_only=True):
        inside ^= _MPath(poly).contains_points(pts)
    return inside
