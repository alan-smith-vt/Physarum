"""Can the cameras actually see out? Final check for the v4 cartridge (user, 2026-10-29: run it after every change).

    python -m designs.pi_box_v4_view_check [step_px]        (from slicer_preview/; default 8)

Works on the SAME meshes the viewer shows (out/overlays_battery_v4/v4_cartridges.stl + v4_cam_lenses.stl, written
when pi_box_v4_core is imported), not on the model functions, so a placement or frame error shows up here:
  1. the lens front is found from the lens mesh (the ring of vertices furthest along the optical axis);
  2. every Kalibr pixel ray (equidistant f 294, 640 x 400, principal point 192 px from the top) is fired from the
     lens-front centre and 8 points on a Ø9 ring round it (the whole front glass), and tested against every cartridge triangle (Möller-Trumbore)
     and against the core (its SDF, the same solid the viewer slices);
  3. the lens front must not be inside the cartridge either (a mesh-in-mesh test along the axis).
Prints % of rays blocked per camera and FAILS (exit 1) if anything is blocked.
"""
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault('PI_BOX_V4_PI', 'flip')        # the flipped Pi is the baseline (user, 2026-10-29: non-flip parked)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from stl_slicer import load_stl                       # noqa: E402
from sdf import solid                                 # noqa: E402
from designs import pi_box_bat_rig as R               # noqa: E402
from designs import pi_box_v4_core as C               # noqa: E402

F, W, H, CY = 294.0, 640, 400, 192.0
GLASS_R, GLASS_BACK = 4.5, 1.0                       # front element radius tested, and its depth behind the lens tip


def rays(axis, step):
    """Unit ray directions (rig X, Y, Z up) for the pixel grid, camera axis `axis` (Z up), sensor landscape."""
    r = np.cross(axis, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, axis)
    uu, vv = np.meshgrid(np.arange(0, W + 1, step) - W / 2, CY - np.arange(0, H + 1, step))
    rho = np.hypot(uu, vv); th = rho / F
    cu = np.where(rho > 0, uu / np.maximum(rho, 1e-9), 0); cv = np.where(rho > 0, vv / np.maximum(rho, 1e-9), 0)
    d = np.cos(th)[..., None] * axis + np.sin(th)[..., None] * (cu[..., None] * r + cv[..., None] * u)
    return d.reshape(-1, 3), r, u


def hits_mesh(o, D, tri, tmax=200.0, eps=1e-9):
    """Bool per ray: does o + t D (0.05 < t < tmax) cross any triangle? Chunked Möller-Trumbore."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = np.zeros(len(D), bool)
    for i in range(0, len(D), 64):
        d = D[i:i + 64, None, :]
        p = np.cross(d, e2[None])
        det = np.einsum('rtk,tk->rt', p, e1)
        ok = np.abs(det) > eps
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = o - v0
        uu = np.einsum('rtk,tk->rt', p, s) * inv
        q = np.cross(s, e1)
        vv = np.einsum('rtk,rtk->rt', np.broadcast_to(d, p.shape), np.broadcast_to(q[None], p.shape)) * inv
        t = np.einsum('tk,tk->t', q, e2)[None] * inv
        out[i:i + 64] = (ok & (uu >= 0) & (vv >= 0) & (uu + vv <= 1) & (t > 0.05) & (t < tmax)).any(1)
    return out


def blocked_by_core(o, D, ts=np.arange(0.5, 150.0, 0.4)):
    P = (o[None, None] + D[:, None] * ts[None, :, None]).reshape(-1, 3)
    out = np.zeros(len(P), bool)
    dq = np.round(-P[:, 2], 1)                        # rig depth d = -Z
    for dv in np.unique(dq):
        if not 0.0 <= dv <= C.DEPTH:
            continue
        sel = np.where(dq == dv)[0]
        out[sel] = np.broadcast_to(solid(C.ROOT, P[sel, 0], P[sel, 1], float(dv)), sel.shape)
    return out.reshape(len(D), len(ts)).any(1)


def main(step=8):
    od = Path(R.OVERLAY_DIR)
    lens = R.from_print(load_stl(od / 'v4_cam_lenses.stl').astype(float))
    cart = R.from_print(load_stl(od / 'v4_cartridges.stl').astype(float))
    ok = True
    for s in (1, -1):
        axd = C.axis_rig(s)                            # (X, Y, d down)
        axis = np.array([axd[0], axd[1], -axd[2]]); axis /= np.linalg.norm(axis)
        L = lens[np.sign(lens[..., 0].mean(1)) == s].reshape(-1, 3)
        T = cart[np.sign(cart[..., 0].mean(1)) == s]
        proj = L @ axis
        front = L[proj >= proj.max() - 0.05]
        o = front.mean(0)                              # lens-front centre, from the mesh
        D, r, u = rays(axis, step)
        # the WHOLE front glass, not just its centre (user 2026-10-29: a lip hid half the lens while the centre
        # passed): centre + 8 points on a Ø9 ring, GLASS_BACK behind the tip
        o = o - GLASS_BACK * axis
        pts = [o] + [o + GLASS_R * (np.cos(q) * r + np.sin(q) * u) for q in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
        blk_c = np.zeros(len(D), bool)
        blk_k = np.zeros(len(D), bool)
        for p in pts:
            blk_c |= hits_mesh(p, D, T)
            blk_k |= blocked_by_core(p, D)
        inside = hits_mesh(o, np.array([axis, -axis]), T).all()   # crossing walls both ways: enclosed
        print(f"camera {'+X' if s > 0 else '-X'}: lens front {o.round(1)}  rays {len(D)} x {len(pts)}  "
              f"blocked by cartridge {100 * blk_c.mean():.2f} %, by core {100 * blk_k.mean():.2f} %"
              f"{'  LENS FRONT ENCLOSED' if inside else ''}")
        ok &= not (blk_c.any() or blk_k.any() or inside)
    print('PASS: both cameras see out' if ok else 'FAIL: a camera view is blocked')
    return ok


if __name__ == '__main__':
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 8) else 1)
