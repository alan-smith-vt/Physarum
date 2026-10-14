"""Signed-distance (SDF) modelling for printable parts, with bounding-box culling and optional anti-aliasing.

A model is a tree of nodes. Every node returns a signed distance in mm (negative inside, positive
outside) at points (X, Y, d): X (H, 1) and Y (1, W) arrays and a scalar d (one layer). Coordinates are
the design's own frame (for the Pi box: rig X, Yr, depth d). Every node knows its bounding box, so a
slice is evaluated tile by tile and each tile only touches the nodes that can reach it.

Distances are exact for primitives and conservative (Lipschitz bounds) after booleans, which is all the
slicer needs: the sign gives the solid, and |distance| near the surface gives anti-aliased edge
coverage and safe culling. Arrays go through `xp` (numpy or cupy), so the same tree runs on the GPU.

  Box, Cyl (circle/ellipse prism along 'x' | 'y' | 'd'), Half (half-space n·p <= c), Fn (custom)
  union / inter / sub (sharp), smooth_union / smooth_sub (fillets of radius k), offset, shell
"""
import numpy as np

INF = 1e9
TIE = 1e-7                                 # mm: closed cuts win ties on their faces (matches mask-style &= ~cut)
AXES = {'x': 0, 'y': 1, 'd': 2}


def _join(bs):
    """Union of boxes (x0, x1, y0, y1, d0, d1); None (unbounded) wins."""
    if any(b is None for b in bs):
        return None
    return tuple(f(b[i] for b in bs) for i, f in enumerate((min, max) * 3))


def _meet(bs):
    """Intersection of boxes; unbounded children don't constrain."""
    bs = [b for b in bs if b is not None]
    if not bs:
        return None
    return tuple(f(b[i] for b in bs) for i, f in enumerate((max, min) * 3))


class Node:
    bounds = None                          # (x0, x1, y0, y1, d0, d1) or None = unbounded

    def hits(self, q, d, m):
        """Can this node's surface come within m of the query tile q = (x0, x1, y0, y1) at depth d?"""
        b = self.bounds
        return b is None or (b[0] - m <= q[1] and q[0] <= b[1] + m and b[2] - m <= q[3] and q[2] <= b[3] + m
                             and b[4] - m <= d <= b[5] + m)

    def __call__(self, X, Y, d, xp=np, q=None, m=0.0):
        """Distance; with a tile q, culled children answer INF ('far outside')."""
        if q is not None and not self.hits(q, d, m):
            return INF
        return self.dist(X, Y, d, xp, q, m)

    # operators read like set algebra
    def __or__(self, o): return Union(self, o)
    def __and__(self, o): return Inter(self, o)
    def __sub__(self, o): return Sub(self, o)


class Box(Node):
    def __init__(self, x0, x1, y0, y1, d0=-INF, d1=INF):
        self.c = ((x0 + x1) / 2, (y0 + y1) / 2, (d0 + d1) / 2)
        self.h = ((x1 - x0) / 2, (y1 - y0) / 2, (d1 - d0) / 2)
        self.bounds = (x0, x1, y0, y1, d0, d1)

    def dist(self, X, Y, d, xp, q, m):
        qx = xp.abs(X - self.c[0]) - self.h[0]
        qy = xp.abs(Y - self.c[1]) - self.h[1]
        qd = abs(d - self.c[2]) - self.h[2]
        out = xp.sqrt(xp.maximum(qx, 0) ** 2 + xp.maximum(qy, 0) ** 2 + max(qd, 0.0) ** 2)
        return out + xp.minimum(xp.maximum(xp.maximum(qx, qy), qd), 0)


class Cyl(Node):
    """Prism of a circle (or ellipse rx, ry) along an axis, between h0 and h1 on that axis.
    c: the centre in the other two coordinates, in (X, Y, d) order with the axis left out."""
    def __init__(self, axis, c, r, h0=-INF, h1=INF, r2=None):
        self.ax = AXES[axis]
        self.c, self.r, self.r2 = c, r, (r if r2 is None else r2)
        self.h0, self.h1 = h0, h1
        lo, hi = [0.0] * 3, [0.0] * 3
        others = [i for i in range(3) if i != self.ax]
        for i, cc, rr in zip(others, c, (self.r, self.r2)):
            lo[i], hi[i] = cc - rr, cc + rr
        lo[self.ax], hi[self.ax] = h0, h1
        self.bounds = (lo[0], hi[0], lo[1], hi[1], lo[2], hi[2])

    def dist(self, X, Y, d, xp, q, m):
        p = (X, Y, d)
        a, b = [p[i] for i in range(3) if i != self.ax]
        u, v = (a - self.c[0]) / self.r, (b - self.c[1]) / self.r2
        rad = (xp.sqrt(u * u + v * v) - 1.0) * min(self.r, self.r2)      # exact for circles, bound for ellipses
        t = p[self.ax]
        along = xp.maximum(self.h0 - t, t - self.h1) if not np.isscalar(t) else max(self.h0 - t, t - self.h1)
        if np.isscalar(along):
            return xp.maximum(rad, along) if along <= 0 else xp.sqrt(xp.maximum(rad, 0) ** 2 + along ** 2)
        return xp.where(along <= 0, xp.maximum(rad, along), xp.sqrt(xp.maximum(rad, 0) ** 2 + xp.maximum(along, 0) ** 2))


class Half(Node):
    """Half-space nx·X + ny·Y + nd·d <= c (normal normalised internally). Unbounded: use inside an Inter."""
    def __init__(self, nx, ny, nd, c):
        n = (nx * nx + ny * ny + nd * nd) ** 0.5
        self.n, self.c = (nx / n, ny / n, nd / n), c / n

    def dist(self, X, Y, d, xp, q, m):
        return self.n[0] * X + self.n[1] * Y + (self.n[2] * d - self.c)


class Fn(Node):
    """Custom distance f(X, Y, d, xp) inside a given bounding box (must be a valid distance bound)."""
    def __init__(self, f, bounds):
        self.f, self.bounds = f, bounds

    def dist(self, X, Y, d, xp, q, m):
        return self.f(X, Y, d, xp)


def _live(children, X, Y, d, xp, q, m):
    return [c(X, Y, d, xp, q, m) for c in children if q is None or c.hits(q, d, m)]


class Union(Node):
    def __init__(self, *c):
        self.c = [x for x in c if x is not None]
        self.bounds = _join([x.bounds for x in self.c])

    def dist(self, X, Y, d, xp, q, m):
        vals = _live(self.c, X, Y, d, xp, q, m)
        if not vals:
            return INF
        out = vals[0]
        for v in vals[1:]:
            out = xp.minimum(out, v)
        return out


class Inter(Node):
    def __init__(self, *c):
        self.c = list(c)
        self.bounds = _meet([x.bounds for x in self.c])

    def dist(self, X, Y, d, xp, q, m):
        if q is not None and any(not c.hits(q, d, m) for c in self.c):
            return INF
        out = None
        for c in self.c:
            v = c(X, Y, d, xp, q, m)
            out = v if out is None else xp.maximum(out, v)
        return out


class Sub(Node):
    """a minus the union of the cutters (only the cutters that reach the tile are evaluated)."""
    def __init__(self, a, *cut):
        self.a, self.cut = a, [x for x in cut if x is not None]
        self.bounds = a.bounds

    def dist(self, X, Y, d, xp, q, m):
        out = self.a(X, Y, d, xp, q, m)
        if np.isscalar(out) and out >= INF:
            return INF
        for c in self.cut:
            if q is None or c.hits(q, d, m):
                out = xp.maximum(out, TIE - c(X, Y, d, xp, q, m))   # a point exactly on a cut's face is cut
        return out


def _smin(a, b, k, xp):
    h = xp.clip(0.5 + 0.5 * (b - a) / k, 0.0, 1.0)
    return b + (a - b) * h - k * h * (1.0 - h)


class SmoothUnion(Union):
    """Union with fillets of radius ~k where the children meet."""
    def __init__(self, k, *c):
        super().__init__(*c)
        self.k = k
        b = self.bounds
        self.bounds = None if b is None else (b[0] - k, b[1] + k, b[2] - k, b[3] + k, b[4] - k, b[5] + k)

    def dist(self, X, Y, d, xp, q, m):
        vals = _live(self.c, X, Y, d, xp, q, m + self.k)
        if not vals:
            return INF
        out = vals[0]
        for v in vals[1:]:
            out = _smin(out, v, self.k, xp)
        return out


class SmoothSub(Sub):
    """a minus cutters, with the cut edges rounded by ~k."""
    def __init__(self, k, a, *cut):
        super().__init__(a, *cut)
        self.k = k

    def dist(self, X, Y, d, xp, q, m):
        out = self.a(X, Y, d, xp, q, m + self.k)
        for c in self.cut:
            if q is None or c.hits(q, d, m + self.k):
                out = -_smin(-out, c(X, Y, d, xp, q, m + self.k), self.k, xp)
        return out


class Offset(Node):
    """Grow (r > 0) or shrink (r < 0) a node by r mm."""
    def __init__(self, a, r):
        self.a, self.r = a, r
        b = a.bounds
        self.bounds = None if b is None else tuple(v + (r if i % 2 else -r) for i, v in enumerate(b))

    def dist(self, X, Y, d, xp, q, m):
        return self.a(X, Y, d, xp, q, m + abs(self.r)) - self.r


class Shell(Node):
    """Hollow shell of thickness t centred on a node's surface."""
    def __init__(self, a, t):
        self.a, self.t = a, t
        self.bounds = Offset(a, t / 2).bounds

    def dist(self, X, Y, d, xp, q, m):
        return xp.abs(self.a(X, Y, d, xp, q, m + self.t)) - self.t / 2


def union(*c): return Union(*c)
def inter(*c): return Inter(*c)
def sub(a, *c): return Sub(a, *c)
def smooth_union(k, *c): return SmoothUnion(k, *c)
def smooth_sub(k, a, *c): return SmoothSub(k, a, *c)
def offset(a, r): return Offset(a, r)
def shell(a, t): return Shell(a, t)


def solid(root, X, Y, d, xp=np, tile=128):
    """Solid mask (distance <= 0). Column X (H, 1) x row Y (1, W) grids are evaluated tile by tile with
    culling (like the slicer); anything else in one go."""
    if X.ndim == 2 and Y.ndim == 2 and X.shape[1] == 1 and Y.shape[0] == 1:
        H, W = X.shape[0], Y.shape[1]
        out = xp.zeros((H, W), bool)
        Xn, Yn = (X, Y) if xp is np else (xp.asnumpy(X), xp.asnumpy(Y))
        for r0 in range(0, H, tile):
            for c0 in range(0, W, tile):
                xs, ys = Xn[r0:r0 + tile], Yn[:, c0:c0 + tile]
                q = (float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
                if not root.hits(q, d, 0.0):
                    continue
                v = root(X[r0:r0 + tile], Y[:, c0:c0 + tile], d, xp, q, 0.0)
                if np.isscalar(v):
                    if v <= 0:
                        out[r0:r0 + tile, c0:c0 + tile] = True
                    continue
                out[r0:r0 + tile, c0:c0 + tile] = xp.broadcast_to(v <= 0, out[r0:r0 + tile, c0:c0 + tile].shape)
        return out
    return root(X, Y, d, xp) <= 0


# ── slicing ──
def sdf_piece(size_x, size_y, size_z, root, to_model, x0=0.0, y0=0.0, z0=0.0, aa=False, tile=128, gpu_tile=2048):
    """Piece dict (like geometry.implicit_piece) for an SDF tree.

    to_model(x, y, z) maps local print coords (x (1, W), scalar y, z (H, 1); mm from the piece corner)
    to the model frame: returns (X (H, 1), Y (1, W), d scalar) for the tree. aa: grayscale edge
    coverage instead of a hard 0/255 threshold. tile: tile edge in pixels for culling (CPU)."""
    import printer
    px, pz, ly = printer.PX_MM, printer.PZ_MM, printer.LY_MM
    W, H = int(np.ceil(size_x / px)), int(np.ceil(size_z / pz))
    N = int(np.ceil(size_y / ly))
    pix = (px * pz) ** 0.5                               # coverage ramp width ~ one pixel
    m = pix                                              # culling margin: only what can touch an edge pixel

    def slice_xp(layer, xp=np, tile_px=None):
        # CPU: small tiles (cache-sized, culling pays). GPU: one tile; per-tile Python/sync overhead dominates.
        tile_px = tile_px or (tile if xp is np else gpu_tile)
        xn = ((np.arange(W) + 0.5) * px)[None, :]
        zn = ((np.arange(H) + 0.5) * pz)[:, None]
        x, z = (xn, zn) if xp is np else (xp.asarray(xn), xp.asarray(zn))
        y = (layer + 0.5) * ly
        out = xp.zeros((H, W), xp.uint8)
        for r0 in range(0, H, tile_px):
            for c0 in range(0, W, tile_px):
                Xn, Yn, _ = to_model(xn[:, c0:c0 + tile_px], y, zn[r0:r0 + tile_px])   # tile bounds on the host:
                q = (float(Xn.min()), float(Xn.max()), float(Yn.min()), float(Yn.max()))  # no GPU sync per tile
                X, Y, d = to_model(x[:, c0:c0 + tile_px], y, z[r0:r0 + tile_px])
                if not root.hits(q, d, m):
                    continue
                dist = root(X, Y, d, xp, q, m)
                if np.isscalar(dist):
                    if dist <= 0:
                        out[r0:r0 + tile_px, c0:c0 + tile_px] = 255
                    continue
                if aa:
                    cov = xp.clip(0.5 - dist / pix, 0.0, 1.0)
                    val = xp.rint(cov * 255).astype(xp.uint8)
                else:
                    val = (dist <= 0).astype(xp.uint8) * 255
                out[r0:r0 + tile_px, c0:c0 + tile_px] = xp.broadcast_to(val, out[r0:r0 + tile_px, c0:c0 + tile_px].shape)
        return out

    return dict(W=W, H=H, N_SLICES=N, OFFSET_X_MM=x0, OFFSET_Y_MM=y0, OFFSET_Z_MM=z0,
                make_slice=lambda layer: slice_xp(layer), slice_xp=slice_xp, aa=aa)
