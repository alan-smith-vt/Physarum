"""v4 wide front panel (user approved home:2136): the full front incl. both R30 rounds below STEP (d 72), the centre
strip above it, L-hook side edges, tongues at the step and the rim. Geometry lives in pi_box_v4_core (front_panel_rig).
Viewer: python generate.py pi_box_v4_front_panel. Prints standing: floor edge down, tongues up (no overhangs)."""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
from designs import pi_box_v4_core as _C       # noqa: E402

SUPPORTS = False
EXPOSURE_S = 2.8                                     # smoky black + clear Tenacious (exposure_test_smoke, 2026-10-31)
BOTTOM_EXPOSURE_S = 20.0
FACE_DOWN = False   # user 2026-11-02: outer face on the plate, no supports (rounds curve up to vertical, hooks up)

if FACE_DOWN:
    import numpy as np
    from geometry import implicit_piece

    def _face_down(x, y, z):
        """x -> rig X, layer height y -> rig Yr - Y_MIN, z -> rig d - FRONT_PANEL_D0 (front_panel_rig wants d scalar)."""
        X, Y = x - _C.X_OUT, y + _C.Y_MIN
        rows = [_C.front_panel_rig(X, np.asarray(Y, float), float(dz + _C.FRONT_PANEL_D0))
                for dz in z[:, 0]]
        return np.concatenate([np.broadcast_to(r, (1, X.shape[1])) for r in rows], 0)

    PIECES = [implicit_piece(2 * _C.X_OUT, _C.RC[1] + _C.NECK_L + _C.HOOK_W + 1.0 - _C.Y_MIN,
                             _C.DEPTH - _C.FRONT_PANEL_D0, _face_down, x0=-_C.X_OUT, z0=_C.Y_MIN)]
else:
    PIECES = _C.FRONT_PANEL_PIECES

EF_CH = 0.5          # elephant's foot: 45° x 0.5 bevel round the plate face (user 2026-11-02), print only


def _ef_bevel(piece):
    import geometry as g
    from scipy.ndimage import distance_transform_edt
    base = piece['make_slice']
    k = int(EF_CH / g.LY_MM)                         # first layer whose centre is >= EF_CH
    cache = {}

    def make_slice(layer, *a, **kw):
        out = base(layer, *a, **kw)
        y = (layer + 0.5) * g.LY_MM
        if y >= EF_CH:
            return out
        if 'dist' not in cache:
            ref = base(k, *a, **kw) > 0
            cache['dist'] = distance_transform_edt(ref, sampling=(g.PZ_MM, g.PX_MM))
        return np.where(cache['dist'] >= EF_CH - y, out, 0).astype(out.dtype)
    out = dict(piece, make_slice=make_slice)
    out.pop('solid_fn', None)                          # to_goo --gpu would skip the bevel: CPU export only
    return out


import numpy as np    # noqa: E402
PIECES = [_ef_bevel(p) for p in PIECES]
