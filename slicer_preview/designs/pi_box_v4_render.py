"""Blender renders of the v4 PROPOSAL (meshes from designs/pi_box_v4_export.py in out/render_v4).

    blender -b -t 4 -P designs/pi_box_v4_render.py -- <ABSOLUTE out dir> [views] --samples 96 --res 1600 1200

Reuses designs/pi_box_render.py (materials, lights, camera helpers); the v3 resins: core smoky black, cartridges
clear turquoise. Views: front34, front, side_px, plan (from above, plate + lidar hidden, +Y up), eyes (close-up of
the +X cartridge), bay (the Pi side with the cartridges and the lowered Pi; GoPro hidden).
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.argv = [a for a in sys.argv]                      # pi_box_render parses the args after '--' at import
from designs import pi_box_render as PR               # noqa: E402

import os                                             # noqa: E402
PR.MESHES = HERE.parent / 'out' / ('render_v4flip' if os.environ.get('PI_BOX_V4_PI') == 'flip' else 'render_v4')
ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT_DIR = Path(ARGV[0]) if ARGV else PR.MESHES / 'png'
VIEWS = [a for a in ARGV[1:] if not a.startswith('--') and not a.replace('.', '').isdigit()] or \
    ['front34', 'front', 'side_px', 'plan', 'eyes', 'bay']
PR.PART_MAT.update(core='smoke', cart_near='turquoise', cart_far='turquoise', v4_cam_boards='pcb',
                   v4_cam_lenses='gloss', power_board='pcb', ue300='white', bank_socket='black',
                   pi_plugs='plug', usbc_plug_envelope='plug')
PR.MATS['plug'] = dict(color=(0.95, 0.45, 0.1), rough=0.5, alpha=0.55)   # [MEASURE] envelopes, translucent
BOX = ['core', 'cart_near', 'cart_far', 'power_switch']
INSIDE = ['pi5', 'cooler', 'battery', 'v4_cam_boards', 'v4_cam_lenses', 'power_board', 'ue300', 'box_spacers',
          'pi_plugs', 'usbc_plug_envelope']
TOP = ['base_plate', 'standoffs', 'cross', 'mid360s', 'coupler', 'm12_plug']
GOPRO = ['gopro_adapter', 'gopro_cage', 'gopro_body', 'gopro_lenses']
POLE = ['monopod']


def main():
    import bpy
    bpy.ops.wm.read_factory_settings(use_empty=True)
    PR.setup_scene()
    objs = PR.load_parts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    every = BOX + INSIDE + TOP + POLE
    for view in VIEWS:
        for c in [o for o in bpy.data.objects if o.type == 'CAMERA']:
            bpy.data.objects.remove(c)
        if view == 'front34':                       # from the front (-Y), +X side, above
            PR.show(objs, every + GOPRO)
            PR.camera('cam', (0, -0.01, -0.02), (0.85, -1.2, 0.55), 0.95)
        elif view == 'front':                       # straight at the face
            PR.show(objs, every)
            PR.camera('cam', (0, 0, -0.05), (0, -1, 0.08), 0.7)
        elif view == 'side_px':
            PR.show(objs, every)
            PR.camera('cam', (0, 0, -0.05), (1, 0, 0.08), 0.75)
        elif view == 'plan':                        # from above, plate + lidar hidden: +Y (back) up in the image
            PR.show(objs, BOX + INSIDE)
            PR.camera('cam', (0, 0, -0.05), (0, -0.0001, 1), 0.55)
        elif view == 'eyes':                        # the +X cartridge close up, along its land normal
            PR.show(objs, BOX + INSIDE + TOP)
            PR.camera('cam', (0.045, -0.046, -0.036), (0.87, -0.5, 0.18), 0.2)
        elif view == 'bay':                         # into the open front bay: cartridges, the lowered Pi
            PR.show(objs, every)
            PR.camera('cam', (0, -0.03, -0.09), (0.35, -1.0, 0.25), 0.42)
        else:
            print('unknown view', view)
            continue
        PR.render(OUT_DIR / f'pi-box-v4-{view}.png')


if __name__ == '__main__':
    main()
