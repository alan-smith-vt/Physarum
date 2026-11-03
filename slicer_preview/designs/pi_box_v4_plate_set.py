"""One plate: core v4 (flipped Pi) + front panel standing + back panel standing (user 2026-11-02). All smoky 2.8 s.
Panels are turned 90 deg (rotation, not an axis swap, which would mirror them) so their ~100 mm width runs along the
plate's short side, in the free strips either side of the core (core 124 x 100 mm; plate 211.7 x 118.4).
Export: python to_goo.py pi_box_v4_plate_set --gpu
"""
import os
os.environ['PI_BOX_V4_PI'] = 'flip'
from geometry import implicit_piece                       # noqa: E402
from designs import pi_box_v4_core_flip as CORE           # noqa: E402
from designs import pi_box_v4_front_panel as FP           # noqa: E402
from designs import pi_box_v4_back_panel_standing as BPS  # noqa: E402
from designs.print_post import bevel_piece                # noqa: E402

SUPPORTS = False
EXPOSURE_S = CORE.EXPOSURE_S
BOTTOM_EXPOSURE_S = CORE.BOTTOM_EXPOSURE_S
GAP = 3.0                                                  # panel <-> core clearance on the plate


def _turn(raw, x0, z0, extra=None):
    """Rotate an implicit piece 90 deg about the vertical: new x = old z, new z = old_W - old x."""
    import geometry as g
    old_wx = raw['W'] * g.PX_MM                            # old x extent (mm) -> new z extent
    old_wz = raw['H'] * g.PZ_MM                            # old z extent (mm) -> new x extent
    rot = implicit_piece(old_wz, raw['N_SLICES'] * g.LY_MM, old_wx,
                         lambda x, h, z: raw['solid_fn'](old_wx - z, h, x), x0=x0, z0=z0)
    ex = None
    if extra is not None:
        def ex(x, h, z, xp):
            return extra(old_wx - z, h, x, xp)
    return rot, ex, old_wz, old_wx


import geometry as _g                                      # noqa: E402
core = CORE.PIECES[0]
cx0, cz0 = core['OFFSET_X_MM'], core['OFFSET_Z_MM']
cx1, cz1 = cx0 + CORE.SIZE_X, cz0 + CORE.SIZE_Z
czc = (cz0 + cz1) / 2

_front_raw = CORE.FRONT_PANEL_PIECES[0]                    # standing front panel (floor edge down), un-bevelled
_fw = _front_raw['H'] * _g.PZ_MM                           # its depth (old z) -> new x extent
_fl = _front_raw['W'] * _g.PX_MM                           # its width (old x) -> new z extent
front, _, _, _ = _turn(_front_raw, cx0 - GAP - _fw, czc - _fl / 2)

_back_raw = BPS._raw
_bw = _back_raw['H'] * _g.PZ_MM
_bl = _back_raw['W'] * _g.PX_MM
back, back_spikes, _, _ = _turn(_back_raw, cx1 + GAP, czc - _bl / 2, extra=BPS._spikes)

PIECES = [core, bevel_piece(front, FP.EF_CH), bevel_piece(back, BPS.EF_CH, extra=back_spikes)]
LAYOUT = dict(core=(cx0, cx1, cz0, cz1), front=(cx0 - GAP - _fw, cx0 - GAP), back=(cx1 + GAP, cx1 + GAP + _bw),
              span=(cx0 - GAP - _fw, cx1 + GAP + _bw))
