"""Concept sketch (for approval, not a model): omni flash cone above the GoPro MAX2 for tunnel surveys, 2026-10-22.

Stack (from the earlier chat): the LED plate faces up in the MAX2's zenith blind spot, on a post above the camera.
Above it hangs a truncated mirror cone, tip down; a frosted cylinder + cap would enclose it (not modelled here).
  - rays near the axis go up through the hollow tip, into the cone's white interior, and out through a diffuser
    disc on top -> the crown;
  - mid-angle rays hit the mirrored underside and are thrown outward and down -> the walls and the track bed;
  - grazing rays pass under the rim directly -> around the horizon.
A Monte Carlo trace (Lambertian disc emitter, 3D rays, on the GPU with CuPy when it loads) gives the lamp's
intensity against elevation. Each candidate cone is scored by how evenly it lights a tunnel: the lining and track bed
within +-HALF_L of the lamp, in two bores (TUNNELS). Mirror light sent back onto the LED plate or the GoPro counts
as lost. The search uses N_SEARCH rays per candidate; the winner is re-traced with N_FINAL.
The LED plate, mirror finish, GoPro top and tunnel sizes are placeholders. A Blender render of the final cone in
a tunnel scene comes later.

    python light_cone_sketch.py -> light_cone_sketch.png
"""
from itertools import product
from pathlib import Path
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon

try:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import cupy as xp
        xp.zeros(1).sum()
except Exception:
    xp = np

LED_R = 14.0           # emitter disc radius: 4 x XHP70.3 HI (7 mm dies) at 2 x 2, ~20 mm pitch [MEASURE the board]
BOARD_R = 30.0         # LED board on the copper plate + printed water block (54 mm square, as a disc); light sent
                       # back onto it is lost
GOPRO_R = 30.0         # GoPro top as a disc this radius, LIFT mm below the LED [MEASURE]
REFL = 0.90            # specular mirror (silver mylar / aluminium tape lining a printed shell)
CHAMBER = 0.70         # fraction of the hole flux that leaves the diffuser (white interior + opal acrylic)
N_SEARCH, N_FINAL = 100_000, 2_000_000
BINS = np.linspace(-1, 1, 61)          # equal solid-angle bins in sin(elevation)
BC = 0.5 * (BINS[1:] + BINS[:-1])

# Tunnels: a circular bore of diameter D (m) with a flat track bed BED m above the invert; the lamp CAM m above the
# bed, on the tunnel's centre line. Scored on the lining + bed within +-HALF_L m along the tunnel (the stretch one
# photo station covers), skipping the bed within 1 m of the operator's feet.
TUNNELS = ((6.0, 1.0, 2.0), (9.0, 1.5, 2.0))     # ~20 ft and ~30 ft bores [guesses, no site survey]
HALF_L = 4.0
# LED height above the GoPro top, fixed (2026-10-24): low profile, the water block (16 mm) + copper plate (3 mm) + LED
# board sit straight on the GoPro. The dish is in both lenses' view; the GoPro's exposure gets locked by hand.
LIFT_FIXED = 22


def to_np(a):
    return a.get() if xp is not np else a


def emit(n, seed=1):
    g = xp.random.default_rng(seed)
    r = LED_R * xp.sqrt(g.random(n)); ph = 2 * np.pi * g.random(n)
    o = xp.stack([r * xp.cos(ph), r * xp.sin(ph), xp.zeros(n)], 1)
    st = xp.sqrt(g.random(n)); ps = 2 * np.pi * g.random(n)          # Lambertian
    d = xp.stack([st * xp.cos(ps), st * xp.sin(ps), xp.sqrt(1 - st ** 2)], 1)
    return o, d, g


def trace(O, D, g, alpha_deg, z_tip, r_hole, R_rim, lift):
    """Outgoing direction z-components, weights and path (0 direct, 1 mirror, 2 diffuser) for every LED ray."""
    ta = np.tan(np.radians(alpha_deg)); z_top = z_tip + (R_rim - r_hole) / ta
    ox, oy, dx, dy, dz = O[:, 0], O[:, 1], D[:, 0], D[:, 1], D[:, 2]
    # cone: sqrt(x² + y²) = k + z ta for z in [z_tip, z_top]; quadratic in t (rays start at z = 0)
    k = r_hole - z_tip * ta
    a = dx ** 2 + dy ** 2 - (dz * ta) ** 2
    b = 2 * (ox * dx + oy * dy - k * dz * ta)
    c = ox ** 2 + oy ** 2 - k ** 2
    disc = b ** 2 - 4 * a * c
    t = xp.full(len(O), xp.inf)
    sq = xp.sqrt(xp.maximum(disc, 0))
    for s in (-1, 1):
        ti = (-b + s * sq) / (2 * a)
        zi = ti * dz
        ok = (disc >= 0) & (ti > 1e-6) & (zi >= z_tip) & (zi <= z_top) & (k + zi * ta > 0)
        t = xp.where(ok & (ti < t), ti, t)
    hit = xp.isfinite(t)
    th = z_tip / dz
    hole = ~hit & (xp.hypot(ox + th * dx, oy + th * dy) < r_hole)
    p = O + xp.where(hit, t, 0)[:, None] * D
    rad = xp.maximum(xp.hypot(p[:, 0], p[:, 1]), 1e-9)
    al = np.arctan(ta)
    nrm = xp.stack([np.cos(al) * p[:, 0] / rad, np.cos(al) * p[:, 1] / rad, xp.full(len(p), -np.sin(al))], 1)
    dr = D - 2 * (D * nrm).sum(1)[:, None] * nrm
    down = dr[:, 2] < 0
    dzs = xp.where(down, dr[:, 2], -1.0)
    tg, tb = (-lift - p[:, 2]) / dzs, -p[:, 2] / dzs
    lost = down & ((xp.hypot(p[:, 0] + tg * dr[:, 0], p[:, 1] + tg * dr[:, 1]) < GOPRO_R)
                   | (xp.hypot(p[:, 0] + tb * dr[:, 0], p[:, 1] + tb * dr[:, 1]) < BOARD_R))
    zdif = xp.sqrt(g.random(len(O)))                                    # diffuser: Lambertian, upward
    zout = xp.where(hit, dr[:, 2], xp.where(hole, zdif, dz))
    w = xp.where(hit, xp.where(lost, 0.0, REFL), xp.where(hole, CHAMBER, 1.0))
    path = xp.where(hit, 1, xp.where(hole, 2, 0))
    return zout, w, path, z_top


def intensity(z, w, mask=None):
    """Intensity per unit solid angle against sin(elevation); 1 = all the emitted flux spread isotropically."""
    if mask is not None:
        w = xp.where(mask, w, 0.0)
    h, _ = xp.histogram(z, xp.asarray(BINS), weights=w)
    return to_np(h) / (len(z) / len(BC))


def tunnel_points(D, bed, cam, n_phi=180, n_x=33):
    R = D / 2; yb = -R + bed; yc = yb + cam
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False); yy, zz = R * np.sin(phi), R * np.cos(phi)
    keep = yy > yb
    xs = np.linspace(-HALF_L, HALF_L, n_x)
    X, P = np.meshgrid(xs, np.arange(keep.sum()))
    lin = np.stack([X.ravel(), zz[keep][P.ravel()], yy[keep][P.ravel()] - yc], 1)
    nl = -np.stack([np.zeros(lin.shape[0]), zz[keep][P.ravel()], yy[keep][P.ravel()]], 1) / R
    hw = np.sqrt(R ** 2 - yb ** 2)
    X, Z = np.meshgrid(xs, np.linspace(-hw, hw, 41))
    bed_p = np.stack([X.ravel(), Z.ravel(), np.full(X.size, yb - yc)], 1)
    feet = np.hypot(bed_p[:, 0], bed_p[:, 1]) > 1.0
    return (np.concatenate([lin, bed_p[feet]]),
            np.concatenate([nl, np.tile([0, 0, 1.0], (feet.sum(), 1))]))


def illuminance(I, pts, nrm):
    d = np.linalg.norm(pts, axis=1); u = pts / d[:, None]
    return np.interp(u[:, 2], BC, I) * np.clip(-(u * nrm).sum(1), 0, 1) / d ** 2


TPTS = [tunnel_points(*t) for t in TUNNELS]


def tunnel_cv(I):
    """Per-tunnel std/mean of the illuminance, and their average."""
    cvs = [np.std(E) / np.mean(E) for E in (illuminance(I, *tp) for tp in TPTS)]
    return float(np.mean(cvs)), cvs


# ── search ──
t0 = time.time()
O, D, G = emit(N_SEARCH)
GRID = list(product([45, 55, 65, 72, 78, 84], [3, 5, 8, 12, 16], [4, 7, 10, 14, 18], [25, 32, 40, 50, 60],
                    [LIFT_FIXED]))
best = None
for cand in GRID:
    z, w, _, _ = trace(O, D, G, *cand)
    cv, _ = tunnel_cv(intensity(z, w))
    if best is None or cv < best[0]:
        best = (cv, cand)
print(f'search ({"GPU" if xp is not np else "CPU"}, {N_SEARCH:,} rays x {len(GRID)} cones): {time.time() - t0:.1f} s')

AL, ZT, RH, RR, LIFT = best[1]
O, D, G = emit(N_FINAL, seed=2)
Z, W, PATH, ZTOP = trace(O, D, G, AL, ZT, RH, RR, LIFT)
I = intensity(Z, W)
I_bare = intensity(D[:, 2], xp.ones(len(D)))
cv, cvs = tunnel_cv(I)
cv0, cvs0 = tunnel_cv(I_bare)
print(f'best: LED {LIFT} mm above the GoPro, alpha {AL} deg, tip {ZT} mm above LED, hole Ø{2 * RH}, rim Ø{2 * RR}, '
      f'top {ZTOP:.1f} mm above LED')
print('  tunnel non-uniformity (std/mean): ' + ', '.join(
    f'{t[0]:.0f} m bore {a:.2f} (bare LED {b:.2f})' for t, a, b in zip(TUNNELS, cvs, cvs0)))
for name, k in (('direct', 0), ('mirror', 1), ('diffuser', 2)):
    print(f'  {name:8s} {float(W[PATH == k].sum()) / N_FINAL:.0%} of LED flux')

# ── figure ──
fig = plt.figure(figsize=(20, 9))
ax = fig.add_axes([0.02, 0.08, 0.32, 0.84]); ax.set_aspect('equal')
ax.set_title(f'Section (mm): LED {LIFT} mm up a post, cone {AL}° off the axis,\nhole Ø{2 * RH}, rim Ø{2 * RR}, '
             f'top {ZTOP:.0f} mm above the LED', fontsize=11)
ax.add_patch(Rectangle((-GOPRO_R, -LIFT - 22), 2 * GOPRO_R, 22, fc='#555', ec='k'))
ax.text(0, -LIFT - 12, 'GoPro MAX2 top [MEASURE]', ha='center', color='w', fontsize=9)
ax.plot([0, 0], [-LIFT, -3], color='#7f8c8d', lw=8)
ax.add_patch(Rectangle((-BOARD_R, -3), 2 * BOARD_R, 3, fc='#b87333', ec='k'))
ax.add_patch(Rectangle((-LED_R, 0), 2 * LED_R, 1.5, fc='#f1c40f', ec='k'))
ax.text(BOARD_R + 2, -3, '4 x XHP70.3 on\na copper plate', fontsize=9)
shell = 1.2
for s in (-1, 1):
    ax.add_patch(Polygon([(s * RH, ZT), (s * RR, ZTOP), (s * RR, ZTOP + shell), (s * RH, ZT + shell)],
                         fc='#bdc3c7', ec='k'))
    ax.plot([s * RH, s * RR], [ZT, ZTOP], color='#2980b9', lw=3)
ax.add_patch(Rectangle((-RR, ZTOP + shell), 2 * RR, 2.0, fc='#ecf0f1', ec='k', hatch='..'))
ax.text(0, ZTOP + 6, 'opal diffuser disc', ha='center', fontsize=9)
ax.text(-RR - 2, ZTOP - 2, 'mirror (blue)\nunderneath,\nwhite inside', fontsize=9, ha='right', va='top')
# fan of meridional rays from the LED centre, traced with the same code (dotted = lost on the plate / GoPro)
n_fan = 41
th = np.radians(np.linspace(-88, 88, n_fan))
Of = xp.zeros((n_fan, 3)); Df = xp.asarray(np.stack([np.sin(th), np.zeros(n_fan), np.cos(th)], 1))
zf, wf, pf = (to_np(a) for a in trace(Of, Df, G, AL, ZT, RH, RR, LIFT)[:3])
ta = np.tan(np.radians(AL)); k = RH - ZT * ta
for i, a in enumerate(th):
    sx = np.sign(a) or 1.0
    if pf[i] == 1:
        zi = k / (abs(np.tan(a)) - ta); px = sx * zi * abs(np.tan(a))
        ox = sx * np.sqrt(max(1 - zf[i] ** 2, 0))
        ax.plot([0, px], [0, zi], color='#e67e22', lw=0.7)
        ax.plot([px, px + 60 * ox], [zi, zi + 60 * zf[i]], color='#e67e22', lw=0.7, ls='-' if wf[i] else ':')
    elif pf[i] == 2:
        ax.plot([0, ZT * np.tan(a)], [0, ZT], color='#27ae60', lw=0.7)
    else:
        ax.plot([0, 80 * np.sin(a)], [0, 80 * np.cos(a)], color='#c0392b', lw=0.7)
ax.set_xlim(-85, 85); ax.set_ylim(-LIFT - 25, max(45, ZTOP + 15)); ax.grid(alpha=0.3)

ax2 = fig.add_axes([0.38, 0.12, 0.28, 0.76])
el = np.degrees(np.arcsin(BC))
for kk, col, lab in ((0, '#c0392b', 'direct under the rim'), (1, '#e67e22', 'mirror'), (2, '#27ae60', 'diffuser')):
    ax2.plot(el, intensity(Z, W, PATH == kk), color=col, lw=1, label=lab)
ax2.plot(el, I, 'k', lw=2.5, label='total')
ax2.plot(el, I_bare, 'k:', lw=1.5, label='bare LED')
ax2.set_xlabel('elevation (deg)'); ax2.set_ylabel('intensity per solid angle (1 = isotropic)')
ax2.set_title('Lamp intensity against elevation', fontsize=11)
ax2.set_xlim(-90, 90); ax2.grid(alpha=0.3); ax2.legend(fontsize=9)

# tunnel cross-sections through the lamp (x = 0): lining + bed coloured by illuminance / the scored mean
for j, (Dm, bed, cam) in enumerate(TUNNELS):
    for i, (Ii, lab, cvi) in enumerate(((I_bare, 'bare LED', cvs0[j]), (I, 'cone', cvs[j]))):
        a3 = fig.add_axes([0.68 + 0.15 * i, 0.52 - 0.46 * j, 0.145, 0.40]); a3.set_aspect('equal')
        R = Dm / 2; yb = -R + bed; yc = yb + cam
        ph = np.linspace(0, 2 * np.pi, 360); yy, zz = R * np.sin(ph), R * np.cos(ph); m = yy > yb
        pts = np.stack([np.zeros(m.sum()), zz[m], yy[m] - yc], 1); nrm = -np.stack([0 * yy[m], zz[m], yy[m]], 1) / R
        hw = np.sqrt(R ** 2 - yb ** 2); bz = np.linspace(-hw, hw, 80); bz = bz[abs(bz) > 1.0]
        bp = np.stack([0 * bz, bz, np.full(bz.size, yb - yc)], 1)
        Eall = np.concatenate([illuminance(Ii, pts, nrm), illuminance(Ii, bp, np.tile([0, 0, 1.0], (bz.size, 1)))])
        ref = np.mean(illuminance(Ii, *TPTS[j]))
        sc = a3.scatter(np.concatenate([zz[m], bz]), np.concatenate([yy[m], np.full(bz.size, yb)]),
                        c=Eall / ref, cmap='inferno', vmin=0, vmax=2, s=10)
        a3.plot([0], [yc], 'c*', ms=12)
        a3.set_title(f'{Dm:.0f} m bore, {lab}: CV {cvi:.2f}', fontsize=10); a3.axis('off')
fig.colorbar(sc, cax=fig.add_axes([0.985, 0.2, 0.006, 0.6]), label='illuminance / mean')
out = Path(__file__).with_suffix('.png'); fig.savefig(out, dpi=100); print(out)

# ── hand-off to Blender (light_cone_render.py): IES profiles of the lamp and the bare LED, plus the geometry ──
FLUX_LM = 10000.0      # XHP70-class, continuous (~70 W from a DeWalt 20V pack, ~1 h), 2026-10-24


def write_ies(path, I_rel):
    """LM-63-2002, type C, one horizontal angle (axisymmetric). Vertical 0 deg = straight down."""
    g = np.arange(0, 181, 1.0)
    cd = np.interp(np.sin(np.radians(g - 90)), BC, I_rel) * FLUX_LM / (4 * np.pi)
    lines = ['IESNA:LM-63-2002', f'[TEST] light_cone_sketch.py', 'TILT=NONE',
             f'1 -1 1 {len(g)} 1 1 2 0 0 0', '1 1 0',
             ' '.join(f'{v:g}' for v in g), '0', ' '.join(f'{v:.2f}' for v in cd)]
    Path(path).write_text('\n'.join(lines) + '\n')


import json
odir = Path(__file__).with_name('light_cone_renders'); odir.mkdir(exist_ok=True)
write_ies(odir / 'cone.ies', I); write_ies(odir / 'bare.ies', I_bare)
(odir / 'cone.json').write_text(json.dumps(dict(alpha=AL, z_tip=ZT, r_hole=RH, r_rim=RR, z_top=ZTOP, lift=LIFT,
                                                led_r=LED_R, board_r=BOARD_R, gopro_r=GOPRO_R, flux_lm=FLUX_LM,
                                                tunnels=TUNNELS), indent=1))
print(odir)
