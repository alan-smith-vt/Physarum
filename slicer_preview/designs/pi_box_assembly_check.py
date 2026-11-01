"""Assembled-state interference check for the Pi box (2026-11-01, after core v3: the floor's GbE cradle hit a Pi
mounting boss and the power board's plugs hit the Pi, and nothing checked parts against each other).

    python -m designs.pi_box_assembly_check [core_module] [--step 0.5]     (from slicer_preview/)
    default core: pi_box_v4_core_flip

Every printed part (core, floor, door, panels: the solids from designs/pi_box_check._parts) and every viewer
overlay (bought parts and envelopes: Pi, plugs, power board + measured plug heights, GbE adapter, cartridges...)
is voxelised on one rig-frame grid (step mm). Overlay meshes use a signed winding count along d, so overlays that
are unions of overlapping boxes still fill correctly. Then every pair is intersected.

Output: one line per overlapping pair, worst first: volume (mm³), voxel count and the rig-frame box of the overlap.
Pairs in EXPECTED (a part and its own envelope, a screw in its insert...) are listed apart and don't fail the run.
Exit code 1 if any unexpected pair overlaps by more than MIN_MM3.
"""
import sys
import itertools
import importlib
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
args = [a for a in sys.argv[1:] if not a.startswith('--')]
STEP = float(sys.argv[sys.argv.index('--step') + 1]) if '--step' in sys.argv else 0.5
if '--step' in sys.argv:
    args = [a for a in args if a != sys.argv[sys.argv.index('--step') + 1]]
sys.argv = [sys.argv[0], args[0] if args else 'pi_box_v4_core_flip']

from stl_slicer import load_stl                          # noqa: E402
import designs.pi_box_check as K                         # noqa: E402  (builds the core module as C)
from designs.pi_box_bat_rig import ROOT, from_print       # noqa: E402

C = K.C
MIN_MM3 = 1.0
IGNORE = {'v4_cam_ribbon_routes'}       # user 2026-11-01: 200 mm ribbons, Pi-camera distances unchanged

# a part and its own envelope / mating hardware: reported, but not a failure. Matched on name prefixes.
GROUPS = [
    {'pi5', 'cooler', 'pi_plugs', 'usbc_plug_envelope', 'eth_usba_plug', 'pi_gpio_harness'},
    {'power_board', 'power_board_parts'},
    {'power_switch', 'power_switch_nut', 'floor'},
    {'mid360s', 'base_plate', 'standoffs', 'box_spacers', 'coupler', 'monopod', 'm12_plug', 'core'},
    {'v4_cartridges', 'v4_cam_', 'v4_flange_screws', 'core'},
    {'ue300', 'eth_cord', 'floor'},
    {'battery', 'core'},
]


def _group_key(name):
    return [i for i, g in enumerate(GROUPS) if any(name == n or (n.endswith('_') and name.startswith(n)) for n in g)]


# a cable route touches what it plugs into at its ends
EXPECTED_PAIRS = {frozenset(p) for p in [('cable_routes', 'power_switch'), ('cable_routes', 'power_board_parts')]}


def expected(a, b):
    if frozenset((a, b)) in EXPECTED_PAIRS:
        return True
    return bool(set(_group_key(a)) & set(_group_key(b)))


# grid (rig frame): X, Yr, d
X0, X1, Y0, Y1, D0, D1 = -C.X_OUT - 5, C.X_OUT + 5, C.Y_MIN - 5, C.Y_MAX + 5, -5.0, C.DEPTH + 20.0
XS = np.arange(X0 + STEP / 2, X1, STEP)
YS = np.arange(Y0 + STEP / 2, Y1, STEP)
DS = np.arange(D0 + STEP / 2, D1, STEP)
SHAPE = (len(XS), len(YS), len(DS))


def vox_mesh(tris_print):
    """Closed (or union-of-closed) mesh in print coords -> bool grid, signed crossings along d."""
    t = from_print(tris_print.reshape(-1, 3)).reshape(-1, 3, 3)       # rig (X, Yr, Zr)
    t[..., 2] *= -1                                                    # d = -Zr
    wind = np.zeros(SHAPE, np.int16)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    n_d = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])   # projected area*2
    ok = np.abs(n_d) > 1e-12
    for (pa, pb, pc, nd) in zip(a[ok], b[ok], c[ok], n_d[ok]):
        xlo, xhi = min(pa[0], pb[0], pc[0]), max(pa[0], pb[0], pc[0])
        ylo, yhi = min(pa[1], pb[1], pc[1]), max(pa[1], pb[1], pc[1])
        i0, i1 = np.searchsorted(XS, xlo), np.searchsorted(XS, xhi, 'right')
        j0, j1 = np.searchsorted(YS, ylo), np.searchsorted(YS, yhi, 'right')
        if i0 >= i1 or j0 >= j1:
            continue
        px, py = np.meshgrid(XS[i0:i1], YS[j0:j1], indexing='ij')
        # barycentric in the X-Yr projection (half-open edges so shared edges count once)
        w0 = (pb[0] - px) * (pc[1] - py) - (pb[1] - py) * (pc[0] - px)
        w1 = (pc[0] - px) * (pa[1] - py) - (pc[1] - py) * (pa[0] - px)
        w2 = (pa[0] - px) * (pb[1] - py) - (pa[1] - py) * (pb[0] - px)
        s = np.sign(nd)
        inside = (w0 * s >= 0) & (w1 * s >= 0) & (w2 * s >= 0) & ((w0 * s > 0) | (w1 * s > 0) | (w2 * s > 0))
        if not inside.any():
            continue
        l0, l1, l2 = w0 / nd, w1 / nd, w2 / nd
        dz = l0 * pa[2] + l1 * pb[2] + l2 * pc[2]                     # d of the hit
        ii, jj = np.nonzero(inside)
        k = np.searchsorted(DS, dz[inside])                            # cells at d >= hit get +-1
        # orientation: outward normal pointing +d (down) -> leaving the solid going +d
        sgn = -1 if nd > 0 else 1
        for i, j, kk in zip(ii + i0, jj + j0, k):
            if kk < SHAPE[2]:
                wind[i, j, kk] += sgn
    vol = np.cumsum(wind, axis=2)
    total = vol[:, :, -1:]                                             # 0 per column for a closed mesh
    if (vol < 0).sum() > (vol > 0).sum():                              # mesh wound the other way
        vol, total = -vol, -total
    # open (non-watertight) STEP meshes leak: filled from the top, a missing face fills to the grid bottom. Fill
    # from both ends and keep the agreement (vol - total = count from the bottom).
    return (vol > 0) & (vol - total > 0)


def vox_fn(fn, xr, yr, dr):
    g = np.zeros(SHAPE, bool)
    i0, i1 = np.searchsorted(XS, xr[0]), np.searchsorted(XS, xr[1])
    j0, j1 = np.searchsorted(YS, yr[0]), np.searchsorted(YS, yr[1])
    X, Y = XS[i0:i1][:, None], YS[j0:j1][None, :]
    for k in range(np.searchsorted(DS, dr[0]), np.searchsorted(DS, dr[1])):
        g[i0:i1, j0:j1, k] = np.broadcast_to(fn(X, Y, float(DS[k])), (i1 - i0, j1 - j0))
    return g


def core_fn(X, Y, d):
    return np.asarray(C.core(Y - C.X0, d, -X - C.Z0), bool)


def main():
    parts = {}
    print(f'grid {SHAPE} at {STEP} mm', flush=True)
    parts['core'] = vox_fn(core_fn, (-C.X_OUT - 1, C.X_OUT + 1), (C.Y_MIN - 1, C.Y_MAX + 1), (0.0, C.DEPTH + 1))
    print('core', parts['core'].sum(), flush=True)
    for name, fn, xr, yr, dr in K._parts():
        key = name.split(' (')[0].replace(' ', '_')
        parts[key] = vox_fn(fn, xr, yr, dr)
        print(key, parts[key].sum(), flush=True)
    for ov in C.OVERLAYS:
        name = Path(ov['file']).stem
        if name in IGNORE:
            continue
        parts[name] = vox_mesh(load_stl(ROOT / ov['file']).astype(np.float64))
        print(name, parts[name].sum(), flush=True)

    v3 = STEP ** 3
    rows = []
    names = list(parts)
    for a, b in itertools.combinations(names, 2):
        o = parts[a] & parts[b]
        n = int(o.sum())
        if not n:
            continue
        i, j, k = np.nonzero(o)
        where = (f'X {XS[i].min():.1f}..{XS[i].max():.1f}  Yr {YS[j].min():.1f}..{YS[j].max():.1f}  '
                 f'd {DS[k].min():.1f}..{DS[k].max():.1f}')
        rows.append((n * v3, a, b, n, where, expected(a, b)))
    rows.sort(reverse=True)
    bad = [r for r in rows if not r[5] and r[0] > MIN_MM3]
    print(f'\nUNEXPECTED overlaps (> {MIN_MM3} mm³): {len(bad)}')
    for vol, a, b, n, where, _ in bad:
        print(f'  {vol:9.1f} mm³  {a} x {b}   {where}')
    print('\nexpected / tiny overlaps:')
    for vol, a, b, n, where, e in rows:
        if e or vol <= MIN_MM3:
            print(f'  {vol:9.1f} mm³  {a} x {b}   {where}{"" if e else "  (tiny)"}')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
