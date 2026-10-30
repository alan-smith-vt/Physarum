"""What each tracking camera actually sees: Blender fisheye renders from the lens of the v4 rig (flipped Pi baseline).

    PI_BOX_V4_PI=flip python -m designs.pi_box_v4_export          (meshes -> out/render_v4flip)
    PI_BOX_V4_LENS="29.6 35.18" blender -b -P designs/pi_box_v4_camera_view.py -- <ABSOLUTE out dir>

Camera: equidistant fisheye, 640 x 400, f 294 px (fisheye_fov = 640 / 294 rad across the width), at the lens-front
centre from pi_box_v4_core, pitch 24 / yaw 30, sensor landscape (as Kalibr). The rig's own parts are tinted
(core red, cartridge orange, plate/cage blue) so any intrusion into the image stands out; the backdrop is grey.
"""
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import os                                             # noqa: E402
os.environ.setdefault('PI_BOX_V4_PI', 'flip')
if '--crop' not in sys.argv:
    from designs import pi_box_render as PR           # noqa: E402

MESHES = HERE.parent / 'out' / 'render_v4flip'
ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT_DIR = Path(ARGV[0]) if ARGV else MESHES / 'png'
F_PX, W, H, CY = 294.0, 640, 400, 192.0
SCALE, SQ = 2, 1600                                   # render at 2x the sensor's pixels, square


def crop(src, dst):          # run with plain python:  python designs/pi_box_v4_camera_view.py --crop <png dir>
    """Cut the real sensor window (W x H, principal point CY px from its top) out of the square fisheye."""
    from PIL import Image
    im = Image.open(src)
    c = SQ / 2
    x0, y0 = c - W / 2 * SCALE, c - CY * SCALE
    im.crop((int(x0), int(y0), int(x0 + W * SCALE), int(y0 + H * SCALE))).save(dst)
TINT = {'core': 'red', 'cart_near': 'orange', 'cart_far': 'orange', 'base_plate': 'navy', 'cross': 'navy',
        'standoffs': 'navy', 'mid360s': 'black', 'panel_front': 'white', 'panel_back': 'white'}
SHOW = ['core', 'cart_near', 'cart_far', 'base_plate', 'cross', 'standoffs', 'mid360s', 'panel_front', 'panel_back',
        'floor', 'door', 'gopro_adapter', 'gopro_cage', 'gopro_body', 'gopro_lenses', 'coupler', 'monopod', 'box_spacers']


def lens_pose(s):
    """Lens-front centre (m) and axis (rig X, Y, Z up) without importing numpy-heavy modules in Blender."""
    yaw, pitch = math.radians(30.0), math.radians(24.0)
    r_front, x_out, y_min = 30.0, 50.0, -62.0
    rc = (x_out - r_front, y_min + r_front)
    n2 = (math.cos(yaw), -math.sin(yaw))
    # pi_box_v4_core.LENS_Q (land coords n, d): Blender's Python has no scipy, so it comes in through the environment
    # (PI_BOX_V4_LENS="n d", printed by: python -c "import designs.pi_box_v4_core as C; print(*C.LENS_Q)")
    n, d = (float(v) for v in os.environ['PI_BOX_V4_LENS'].split())
    p = (s * (rc[0] + n * n2[0]), rc[1] + n * n2[1], -d)
    a = (s * math.cos(pitch) * n2[0], math.cos(pitch) * n2[1], math.sin(pitch))
    return p, a


def main():
    import bpy
    from mathutils import Vector
    PR.MESHES = MESHES
    bpy.ops.wm.read_factory_settings(use_empty=True)
    ground = PR.setup_scene()
    bpy.data.objects.remove(ground)                  # the floor stand-in is not part of the scene the camera sees
    sc = bpy.context.scene
    sc.render.resolution_x = sc.render.resolution_y = SQ          # square, wide enough for the sensor corners
    objs = PR.load_parts()
    for n, ob in objs.items():
        ob.hide_render = n not in SHOW
        if n in TINT:
            ob.data.materials.clear()
            ob.data.materials.append(PR.material(TINT[n]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s, tag in ((1, 'px'), (-1, 'mx')):
        p, a = lens_pose(s)
        bpy.ops.object.camera_add()
        cam = bpy.context.object
        cam.data.type = 'PANO'
        cam.data.panorama_type = 'FISHEYE_EQUIDISTANT'
        cam.data.fisheye_fov = SQ / (F_PX * SCALE)     # same f as the sensor (x SCALE px); circle reaches 78°
        cam.data.sensor_fit = 'HORIZONTAL'
        cam.data.sensor_height = cam.data.sensor_width
        cam.data.clip_start = 0.0005
        fwd = Vector(a).normalized()
        cam.location = Vector(p) * PR.MM + fwd * 0.0002
        cam.rotation_mode = 'QUATERNION'
        cam.rotation_quaternion = fwd.to_track_quat('-Z', 'Z')   # landscape, image up toward world up
        sc.camera = cam
        PR.render(OUT_DIR / f'pi-box-v4-camview-{tag}-full.png')    # crop to the sensor afterwards (no PIL in Blender)
        bpy.data.objects.remove(cam)


if __name__ == '__main__':
    if '--crop' in sys.argv:
        d = Path(sys.argv[sys.argv.index('--crop') + 1])
        for tag in ('px', 'mx'):
            crop(d / f'pi-box-v4-camview-{tag}-full.png', d / f'pi-box-v4-camview-{tag}.png')
    else:
        main()
