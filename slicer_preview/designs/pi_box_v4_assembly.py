"""v4 ASSEMBLED view (2026-11-01): the core (flipped-Pi baseline) as voxels plus every other printed part as solid
overlays: fixed floor, door A, wide front panel, back panel, both camera cartridges, on top of the usual bought-part
overlays. For checking fit by eye; designs/pi_box_assembly_check.py does it by numbers.

    python generate.py pi_box_v4_assembly          viewer: /pi_box_v4_assembly   (press O to toggle overlays)
"""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
os.environ.setdefault('PI_BOX_VOX', '0.5')
import numpy as np                                            # noqa: E402
from designs import pi_box_v4_core_flip as _C                 # noqa: E402

globals().update({k: v for k, v in vars(_C).items() if not k.startswith('__')})

from designs import pi_box_v4_export as _X                    # noqa: E402
from designs import pi_box_bat_rig as _R                      # noqa: E402

_COLORS = {'floor': '#48c9b0', 'door': '#76d7c4', 'panel_front': '#5dade2', 'panel_back': '#5dade2',
           'cart_far': '#af7ac5', 'cart_near': '#af7ac5'}


def _printed_overlays():
    parts = dict(_X.floor_parts())
    parts['cart_far'], parts['cart_near'] = _X.cartridge(1), _X.cartridge(-1)
    out = []
    for name, (v, f) in parts.items():
        tris = np.asarray(v)[np.asarray(f)]                   # rig (X, Yr, Zr)
        path = _R.OVERLAY_DIR / f'asm_{name}.stl'
        _R._write_stl(path, _R.to_print(tris))
        out.append(dict(file=path.relative_to(_R.ROOT).as_posix(), color=_COLORS[name], opacity=0.55,
                        label=f'PRINTED: {name.replace("_", " ")}'))
    return out


OVERLAYS = _C.group_overlays(list(_C.OVERLAYS) + _printed_overlays())
