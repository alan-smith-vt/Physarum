"""What each tracking camera sees, drawn straight from the Kalibr model (no Blender): every pixel's ray is marched
through the real solids (core SDF, both cartridges, the Al base plate) and coloured by what it hits first.

    python -m designs.pi_box_v4_camera_image            (from slicer_preview/) -> out/render_v4flip/png/camimage-*.png

Equidistant fisheye f 294 px, 640 x 400, principal point 192 px from the top, pitch 24 / yaw 30, sensor landscape;
rays from the lens-front centre. The frame covers 1.3x the sensor so the rig's own edges just outside it show;
the real sensor area is outlined. Grey = open view, red = core, orange = cartridge, blue = base plate.
(Blender's fisheye camera does not reproduce this mapping, see pi_box_v4_camera_view.py; use this image.)
"""
import os
import sys
from pathlib import Path
import numpy as np

os.environ.setdefault('PI_BOX_V4_PI', 'flip')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sdf import solid                                 # noqa: E402
from designs import pi_box_v4_core as C               # noqa: E402
from designs import pi_box_v4_cam_cartridge as K      # noqa: E402

F, W, H, CY = 294.0, 640, 400, 192.0
ZOOM, STEP_PX, T_MAX, DT = 1.3, 5, 160.0, 0.4


def occupancy(P):
    """P (N, 3) rig (X, Y, Z up) -> int per point: 0 free, 1 core, 2 cartridge, 3 plate."""
    out = np.zeros(len(P), np.int8)
    dq = np.round(-P[:, 2], 1)
    for dv in np.unique(dq):
        sel = np.where(dq == dv)[0]
        X, Y = P[sel, 0], P[sel, 1]
        m = np.zeros(len(sel), np.int8)
        if -6.2 <= dv <= 0.0:
            m[(np.abs(X) <= 52.2) & (np.abs(Y) <= 62.0)] = 3
        for s in (-1, 1):
            n = (s * X - C.RC[0]) * C.N2[0] + (Y - C.RC[1]) * C.N2[1]
            t = (s * X - C.RC[0]) * C.T2[0] + (Y - C.RC[1]) * C.T2[1]
            h = C.FL_D[1] - dv
            if 0 <= h <= K.H_TOP:
                m[np.broadcast_to(solid(K.ROOT, n, t, float(h)), m.shape) & (m == 0)] = 2
        if 0.0 <= dv <= C.DEPTH:
            m[np.broadcast_to(solid(C.ROOT, X, Y, float(dv)), m.shape) & (m == 0)] = 1
        out[sel] = m
    return out


def image(s):
    axd = C.axis_rig(s)
    f = np.array([axd[0], axd[1], -axd[2]]); f /= np.linalg.norm(f)
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    u = np.cross(r, f)
    q = C.LENS_Q
    o = np.array([s * (C.RC[0] + q[0] * C.N2[0]), C.RC[1] + q[0] * C.N2[1], -q[1]])
    us = np.arange(-W / 2 * ZOOM, W / 2 * ZOOM + 1, STEP_PX)
    vs = np.arange(CY + (ZOOM - 1) * H / 2, CY - H - (ZOOM - 1) * H / 2 - 1, -STEP_PX)
    uu, vv = np.meshgrid(us, vs)
    rho = np.hypot(uu, vv); th = rho / F
    cu, cv = uu / np.maximum(rho, 1e-9), vv / np.maximum(rho, 1e-9)
    D = (np.cos(th)[..., None] * f + np.sin(th)[..., None] * (cu[..., None] * r + cv[..., None] * u)).reshape(-1, 3)
    ts = np.arange(0.3, T_MAX, DT)
    P = (o[None, None] + D[:, None] * ts[None, :, None]).reshape(-1, 3)
    occ = occupancy(P).reshape(len(D), len(ts))
    first = np.where(occ.any(1), occ[np.arange(len(D)), np.argmax(occ > 0, 1)], 0)
    return first.reshape(uu.shape), us, vs


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    out = Path(__file__).resolve().parents[1] / 'out' / 'render_v4flip' / 'png'
    out.mkdir(parents=True, exist_ok=True)
    cmap = ListedColormap(['#b8bcc2', '#c0392b', '#e67e22', '#1f3a93'])
    for s, tag in ((1, 'px'), (-1, 'mx')):
        img, us, vs = image(s)
        inside = (np.abs(us)[None, :] <= W / 2) & (vs[:, None] <= CY) & (vs[:, None] >= CY - H)
        blocked = (img > 0) & inside
        fig, ax = plt.subplots(figsize=(10, 6.8))
        ax.imshow(img, cmap=cmap, vmin=0, vmax=3, extent=(us[0], us[-1], vs[-1], vs[0]), interpolation='nearest')
        ax.add_patch(plt.Rectangle((-W / 2, CY - H), W, H, fill=False, ec='k', lw=2))
        ax.set_title(f'{"+X" if s > 0 else "-X"} camera, Kalibr fisheye from the lens: black box = the sensor. '
                     f'Rig inside the sensor: {100 * blocked.sum() / inside.sum():.2f} % of pixels\n'
                     'grey open, red core, orange cartridge, blue base plate (frame = 1.3x the sensor)', fontsize=9)
        ax.set_xlabel('u (px from the principal point, right)'); ax.set_ylabel('v (px, up)')
        fig.tight_layout(); fig.savefig(out / f'camimage-{tag}.png', dpi=90); plt.close(fig)
        print(f'camera {tag}: rig inside the sensor {100 * blocked.sum() / inside.sum():.2f} %  -> {out / f"camimage-{tag}.png"}')


if __name__ == '__main__':
    main()
