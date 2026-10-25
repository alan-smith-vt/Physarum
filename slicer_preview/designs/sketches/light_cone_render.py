"""Blender render of the omni flash cone in a subway tunnel, 2026-10-22 (concept, for approval).

light_cone_sketch.py writes light_cone_renders/{cone,bare}.ies (the traced intensity of the lamp and of the bare
LEDs, absolute candela for FLUX_LM) and cone.json (the winning geometry). This script builds the tunnel, puts a point
light carrying each IES profile at the LED, models the lamp itself (GoPro, post, plate, mirror dish, diffuser) for
looks only (it casts no shadows; the trace already accounts for them), and renders with Cycles on the GPU:
  - an equirectangular frame from the MAX2's position (what the 360 camera records);
  - a perspective frame down the tunnel from behind the operator.
Both variants share one fixed exposure, so the bare-LED and cone frames compare directly.

    python light_cone_render.py [--samples 256] [--bore 6]   (launches Blender in the background, then montages)
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'light_cone_renders'
BLENDER = r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
IES_GAIN = 1000 / 69.89    # Cycles IES Fac -> candela: a flat 1000 cd profile measured 69.9 lx at 1 m (Blender 5.2)


def arg(name, default):
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    return type(default)(a[a.index(name) + 1]) if name in a else default


SAMPLES = arg('--samples', 256)
BORE = arg('--bore', 6.0)


def build_and_render():
    import bpy
    import bmesh

    G = json.loads((OUT / 'cone.json').read_text())
    D, BED, CAM = next(t for t in G['tunnels'] if abs(t[0] - BORE) < 1e-6)
    R = D / 2; yb = -R + BED                         # bed height relative to the bore centre
    LEN = 80.0
    lamp_z = yb + CAM                                # lamp (LED face) height; world z = 0 at the bore centre

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def mat(name, color, rough=0.9, metal=0.0, emit=None, transl=0.0):
        m = bpy.data.materials.new(name); m.use_nodes = True
        p = m.node_tree.nodes['Principled BSDF']
        p.inputs['Base Color'].default_value = (*color, 1)
        p.inputs['Roughness'].default_value = rough
        p.inputs['Metallic'].default_value = metal
        if transl:
            p.inputs['Transmission Weight'].default_value = transl
        if emit:
            p.inputs['Emission Color'].default_value = (*emit[0], 1)
            p.inputs['Emission Strength'].default_value = emit[1]
        return m

    def concrete():
        m = mat('concrete', (0.32, 0.31, 0.29))
        nt = m.node_tree; p = nt.nodes['Principled BSDF']
        noise = nt.nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 3.0
        noise.inputs['Detail'].default_value = 8.0
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].color = (0.20, 0.19, 0.18, 1); ramp.color_ramp.elements[1].color = (0.42, 0.41, 0.38, 1)
        # ring-segment joints every 1.5 m along the tunnel, as dark bands (a distance scale in the frames)
        coord = nt.nodes.new('ShaderNodeTexCoord'); sep = nt.nodes.new('ShaderNodeSeparateXYZ')
        wave = nt.nodes.new('ShaderNodeMath'); wave.operation = 'PINGPONG'; wave.inputs[1].default_value = 0.75
        band = nt.nodes.new('ShaderNodeMath'); band.operation = 'LESS_THAN'; band.inputs[1].default_value = 0.02
        mix = nt.nodes.new('ShaderNodeMix'); mix.data_type = 'RGBA'; mix.inputs['B'].default_value = (0.06, 0.06, 0.06, 1)
        L = nt.links
        L.new(coord.outputs['Object'], noise.inputs['Vector']); L.new(noise.outputs['Fac'], ramp.inputs['Fac'])
        L.new(coord.outputs['Object'], sep.inputs['Vector']); L.new(sep.outputs['X'], wave.inputs[0])
        L.new(wave.outputs[0], band.inputs[0]); L.new(band.outputs[0], mix.inputs['Factor'])
        L.new(ramp.outputs['Color'], mix.inputs['A']); L.new(mix.outputs['Result'], p.inputs['Base Color'])
        return m

    def add_obj(name, bm, material, shadow=True, camera=True):
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
        ob.data.materials.append(material)
        ob.visible_shadow = shadow; ob.visible_camera = camera
        return ob

    # ── tunnel: lining arc above the bed, extruded along X, faces inward; bed plane; two rails ──
    bm = bmesh.new(); n = 192; ph0 = math.asin(yb / R)
    phis = [ph0 + (math.pi - 2 * ph0) * i / n for i in range(n + 1)]       # from one bed edge round the crown
    prof = [(R * math.cos(p), R * math.sin(p)) for p in phis]
    nx = 160
    vs = [[bm.verts.new((-LEN / 2 + LEN * j / nx, y, z)) for (y, z) in prof] for j in range(nx + 1)]
    for j in range(nx):
        for i in range(n):
            bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:                                                        # inward
        f.normal_flip()
    add_obj('lining', bm, concrete())
    hw = R * math.cos(ph0)
    bm = bmesh.new(); bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=1)
    bmesh.ops.scale(bm, vec=(LEN / 2, hw, 1), verts=bm.verts); bmesh.ops.translate(bm, vec=(0, 0, yb), verts=bm.verts)
    add_obj('bed', bm, mat('ballast', (0.18, 0.17, 0.16), rough=1.0))
    steel = mat('rail', (0.5, 0.5, 0.5), rough=0.35, metal=1.0)
    for y in (-0.72, 0.72):
        bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1)
        bmesh.ops.scale(bm, vec=(LEN, 0.07, 0.15), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(0, y, yb + 0.075), verts=bm.verts)
        add_obj(f'rail{y}', bm, steel)

    # ── lamp + camera (mm -> m); looks only: no shadows, the IES already has them ──
    s = 0.001; L0 = (0, 0, lamp_z)
    black = mat('gopro', (0.02, 0.02, 0.02), rough=0.5)
    bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1)
    bmesh.ops.scale(bm, vec=(2 * G['gopro_r'] * s, 0.03, 0.065), verts=bm.verts)
    bmesh.ops.translate(bm, vec=(0, 0, lamp_z - G['lift'] * s - 0.0325), verts=bm.verts)
    add_obj('max2', bm, black, shadow=False, camera=False)                   # the camera sits inside it
    bm = bmesh.new(); bmesh.ops.create_cone(bm, segments=24, radius1=0.004, radius2=0.004, depth=G['lift'] * s)
    bmesh.ops.translate(bm, vec=(0, 0, lamp_z - G['lift'] * s / 2), verts=bm.verts)
    add_obj('post', bm, black, shadow=False)
    bm = bmesh.new(); bmesh.ops.create_cone(bm, segments=48, radius1=G['led_r'] * s, radius2=G['led_r'] * s, depth=0.0015)
    bmesh.ops.translate(bm, vec=(0, 0, lamp_z), verts=bm.verts)
    add_obj('leds', bm, mat('led', (1, 1, 0.9), emit=((1, 0.97, 0.9), 50.0)), shadow=False)
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, segments=96, cap_ends=False, radius1=G['r_hole'] * s, radius2=G['r_rim'] * s,
                          depth=(G['z_top'] - G['z_tip']) * s)
    bmesh.ops.translate(bm, vec=(0, 0, lamp_z + (G['z_tip'] + G['z_top']) / 2 * s), verts=bm.verts)
    dish = add_obj('dish', bm, mat('mirror', (0.95, 0.95, 0.95), rough=0.03, metal=1.0), shadow=False)
    dish.modifiers.new('shell', 'SOLIDIFY').thickness = 0.0012
    bm = bmesh.new(); bmesh.ops.create_cone(bm, segments=96, radius1=G['r_rim'] * s, radius2=G['r_rim'] * s, depth=0.002)
    bmesh.ops.translate(bm, vec=(0, 0, lamp_z + (G['z_top'] + 2.2) * s), verts=bm.verts)
    add_obj('diffuser', bm, mat('opal', (1, 1, 1), emit=((1, 1, 1), 3.0)), shadow=False)

    light = bpy.data.lights.new('flash', 'POINT'); light.shadow_soft_size = G['led_r'] * s
    lob = bpy.data.objects.new('flash', light); lob.location = (0, 0, lamp_z + 0.002); sc.collection.objects.link(lob)
    light.use_nodes = True
    nt = light.node_tree; ies = nt.nodes.new('ShaderNodeTexIES'); ies.mode = 'EXTERNAL'
    nt.links.new(ies.outputs['Fac'], nt.nodes['Emission'].inputs['Strength'])
    # A point light of P watts radiates P / (4 pi) W/sr, and 1 W ~ 683 lm here, so P = 4 pi / 683 is 1 cd; the IES
    # node's output isn't candela, so IES_GAIN (measured) makes 1 cd in the file = 1 cd in the scene.
    light.energy = 4 * math.pi / 683 * IES_GAIN

    # ── render settings ──
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for dev_type in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = dev_type; prefs.get_devices()
            if any(d.type == dev_type for d in prefs.devices):
                break
        except TypeError:
            pass
    for d in prefs.devices:
        d.use = d.type != 'CPU'
    sc.cycles.device = 'GPU'; sc.cycles.samples = SAMPLES; sc.cycles.use_denoising = True
    sc.cycles.max_bounces = 6
    sc.view_settings.view_transform = 'Standard'
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.world = bpy.data.worlds.new('dark'); sc.world.use_nodes = True
    sc.world.node_tree.nodes['Background'].inputs['Strength'].default_value = 0.0

    eq = bpy.data.cameras.new('max2'); eq.type = 'PANO'; eq.panorama_type = 'EQUIRECTANGULAR'
    eq.clip_start = 0.01
    eqo = bpy.data.objects.new('max2_cam', eq); eqo.location = (0, 0, lamp_z - G['lift'] * s - 0.03)
    eqo.rotation_euler = (math.pi / 2, 0, -math.pi / 2)      # looking along +X, horizon level
    pe = bpy.data.cameras.new('chase'); pe.lens = 18; pe.clip_start = 0.01
    peo = bpy.data.objects.new('chase_cam', pe); peo.location = (-3.0, 0.6, lamp_z + 0.2)
    peo.rotation_euler = (math.radians(88), 0, math.radians(-95))
    for o in (eqo, peo):
        sc.collection.objects.link(o)

    for variant in ('bare', 'cone'):
        ies.filepath = str(OUT / f'{variant}.ies')
        dish.hide_render = variant == 'bare'
        bpy.data.objects['diffuser'].hide_render = variant == 'bare'
        for cam, (w, h) in ((eqo, (2048, 1024)), (peo, (1600, 900))):
            sc.camera = cam; sc.render.resolution_x, sc.render.resolution_y = w, h
            sc.render.filepath = str(OUT / f'{variant}_{cam.name}_{BORE:.0f}m.exr')
            bpy.ops.render.render(write_still=True)
    tonemap(bpy)


def tonemap(bpy):
    """One exposure for every frame: the dish's MAX2 frame median goes to mid-grey. Writes the PNGs."""
    import numpy as np
    sc = bpy.context.scene
    imgs = {}
    for f in OUT.glob(f'*_{BORE:.0f}m.exr'):
        im = bpy.data.images.load(str(f)); a = np.array(im.pixels[:], np.float32).reshape(im.size[1], im.size[0], 4)
        imgs[f] = (im, a)
    ref = next(a for f, (_, a) in imgs.items() if f.name.startswith('cone_max2'))
    lum = ref[..., :3] @ np.array([0.2126, 0.7152, 0.0722])
    k = 0.18 / np.median(lum[lum > 0])
    print(f'exposure: x{k:.3g} on linear')
    for f, (im, a) in imgs.items():
        x = a[..., :3] * k
        x = x / (1 + x / 4.0)                                          # soft shoulder, mid-tones ~linear
        x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)
        out = bpy.data.images.new(f.stem, im.size[0], im.size[1])
        out.pixels[:] = np.concatenate([np.clip(x, 0, 1), np.ones_like(x[..., :1])], -1).ravel()
        out.filepath_raw = str(f.with_suffix('.png')); out.file_format = 'PNG'; out.save()


def montage():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fig, axs = plt.subplots(2, 2, figsize=(20, 11.5), gridspec_kw=dict(width_ratios=[2048 / 1024, 1600 / 900]))
    for i, v in enumerate(('bare', 'cone')):
        for j, cam in enumerate(('max2_cam', 'chase_cam')):
            axs[i, j].imshow(mpimg.imread(OUT / f'{v}_{cam}_{BORE:.0f}m.png')); axs[i, j].axis('off')
            axs[i, j].set_title(f'{"bare LEDs" if v == "bare" else "mirror dish"}: '
                                f'{"MAX2 equirect" if j == 0 else "behind the operator"}', fontsize=13)
    fig.suptitle(f'{BORE:.0f} m bore, same exposure; joint bands every 1.5 m', fontsize=14)
    fig.tight_layout(); out = OUT / f'montage_{BORE:.0f}m.png'; fig.savefig(out, dpi=80); print(out)


if __name__ == '__main__':
    if 'bpy' in sys.modules or Path(sys.executable).name.lower().startswith('blender'):
        build_and_render()
    else:
        subprocess.run([BLENDER, '-b', '--factory-startup', '-P', __file__, '--', *sys.argv[1:]], check=True)
        montage()
