"""Blender render of the whole survey stack, carried by an operator in a subway tunnel, 2026-10-24 (concept).

Monopole -> Pi box (v2 rig meshes from out/render/, run designs/pi_box_export.py first) -> Mid-360S -> GoPro ->
omni light cone (geometry from light_cone_renders/cone.json, light from cone.ies; run light_cone_sketch.py first).
Head (2026-10-24, continuous ~10k lm, low profile): LED board on a copper plate on a printed water block mounted
straight on the GoPro's top, as wide as the GoPro (39 mm). Custom loop, pump below the head: 6 mm tubes run down the
GoPro's +Y face (the lenses' stitch line), past the lidar on its +Y side, to a cooling module in a ~92 mm square on the
box's +Y (battery-pod) face: 60 mm-class radiator + slim fan, LED driver, printed reservoir + mini pump. Sizes are placeholders [no parts chosen yet].
A mannequin (1.75 m, hard hat, hi-vis vest) holds the stick two-handed, tilted TILT deg forward (+X, the walking
direction). The lamp tilts with the stick; the cone was only scored vertical. The rig is yawed YAW deg about the stick;
at 0 the GoPro lenses face forward-down and back-up, in the tilt plane [ASSUMED, check the Arca mount orientation].

Lamp brightness is physical, not cosmetic:
  - the IES point light lights the scene (it already holds the mirror, the diffuser and the GoPro's shadow);
  - the LED face and the opal diffuser are emitters calibrated in luminance (LED: FLUX over the disc; diffuser: the
    IES straight-up candela over the rim disc), seen only by camera and glossy rays, so the mirror shows the LED
    image without lighting the scene twice (reflective caustics off for the same reason).
Frames:
  wide        operator + stack in the tunnel, lamp on
  side_geo    operator + stack in profile, lamp off, studio fill (the geometry)
  close_geo   GoPro -> head with the tube run, lamp off, studio fill
  head_geo    the water-cooled head, lamp off
  rad_geo     radiator, fan, reservoir + pump on the box's +Y face, lamp off
  lidar_band  what the Mid-360 sees around it (equirect from its optical centre [ASSUMED 45 mm up], stick axis up):
              rig parts grey, the cooling loop red, its -7..+52 deg band between the blue lines
  lens_front, lens_back   the MAX2's two fisheyes (200 deg equidistant), lamp on, with and without a veiling-glare
              model (GLARE of the light into a ~r^-2.8 PSF tail [GUESS: not measured on a MAX2])
Each frame is auto-exposed on its own median (as the GoPro would), so brightness doesn't compare between frames.

    python light_cone_stack_render.py [--samples 256] [--bore 6] [--tilt 40] [--yaw 0] [--stick 1.2]
        -> light_cone_renders/stack_*.png, stack_lenses_*.png, stack.blend

--lights box3 (2026-10-24, concept for a go / no-go): no top lamp. Three flush lights in the Pi box instead, one
beside each tracking camera on the +-X faces (the cameras face the operator's left and right) and one on the -Y Pi
wall, which faces forward (the battery wall faces the operator). Each is a flush opal window (Lambertian, an area
light carrying FLUX lm plus a calibrated emitter seen by camera and glossy rays). The rig is yawed 90 deg so -Y leads.
Cooling isn't modelled. Extra frames: box_geo_l/r (the box, lamps off), track_l/r (the tracking cameras, 160 deg
fisheye, lamps on). Output prefix box3_.
    python light_cone_stack_render.py --lights box3 [--side_lm 4000] [--fwd_lm 3000]
"""
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'light_cone_renders'
BLENDER = r'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
GOPRO_TOP = 169.8          # GoPro top in the rig frame (mm, gopro_body.stl bbox); the LED sits `lift` above it
LENS_C = (24.4, 0.0, 143.3)  # lens front centre (+-X) in the rig frame (gopro_lenses.stl bbox)
MONOPOD_BOT = 260.0        # monopod.stl bottom, mm below the rig origin
GLARE = 0.03
LIDAR_C = 45.0             # Mid-360 optical centre above the rig origin, mm [ASSUMED: mid-window]


def arg(name, default):
    a = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    return type(default)(a[a.index(name) + 1]) if name in a else default


SAMPLES = arg('--samples', 256)
BORE = arg('--bore', 6.0)
TILT = arg('--tilt', 40.0)
LIGHTS = arg('--lights', 'cone')
YAW = arg('--yaw', 90.0 if LIGHTS == 'box3' else 0.0)
SIDE_LM = arg('--side_lm', 4000.0)
FWD_LM = arg('--fwd_lm', 3000.0)
PFX = 'box3' if LIGHTS == 'box3' else 'stack'
WIN_R = 15.0               # box3 window radius, mm [placeholder]
TRACK_C = 54.5             # tracking-camera lens front (+-X) in the rig frame, at y -40.1, z -30 (lens_*.stl bbox)
STICK = arg('--stick', 1.2)          # lower hand to monopod bottom, m


def ies_up_candela(path):
    t = path.read_text().split('TILT=NONE', 1)[1].split()
    mult, nv, nh = float(t[2]), int(t[3]), int(t[4])
    va = [float(x) for x in t[13:13 + nv]]
    cd = [float(x) * mult for x in t[13 + nv + nh:13 + 2 * nv + nh]]
    return cd[va.index(180.0)]


def ies_profile(path):
    t = path.read_text().split('TILT=NONE', 1)[1].split()
    mult, nv, nh = float(t[2]), int(t[3]), int(t[4])
    return [float(x) for x in t[13:13 + nv]], [float(x) * mult for x in t[13 + nv + nh:13 + 2 * nv + nh]]


def glare(a):
    """Veiling glare: GLARE of each pixel's light spread by a PSF with a ~r^-2.8 tail (FFT convolution)."""
    import numpy as np
    h, w = a.shape[:2]
    fy = np.minimum(np.arange(h), h - np.arange(h))[:, None]; fx = np.minimum(np.arange(w), w - np.arange(w))[None]
    k = (1 + (fy ** 2 + fx ** 2) / 1.5 ** 2) ** -1.4; k /= k.sum()
    K = np.fft.rfft2(k)
    conv = np.stack([np.fft.irfft2(np.fft.rfft2(a[..., c]) * K, s=(h, w)) for c in range(3)], -1)
    return (1 - GLARE) * a + GLARE * np.maximum(conv, 0)


def build_and_render():
    import bpy
    import bmesh
    import numpy as np
    from mathutils import Vector, Matrix
    sys.path[:0] = [str(HERE.parent), str(HERE)]
    import pi_box_render as pbr
    import light_cone_render as lcr

    G = json.loads((OUT / 'cone.json').read_text())
    D, BED, _ = next(t for t in G['tunnels'] if abs(t[0] - BORE) < 1e-6)
    R = D / 2; yb = -R + BED
    LEN = 80.0
    s = 0.001
    rig_s = STICK + MONOPOD_BOT * s                      # rig origin along the stick from the lower hand
    lamp_s = rig_s + (GOPRO_TOP + G['lift']) * s         # LED face along the stick

    bpy.ops.wm.read_factory_settings(use_empty=True)
    sc = bpy.context.scene

    def mat(name, color, rough=0.9, metal=0.0, emit=None):
        m = bpy.data.materials.new(name); m.use_nodes = True
        p = m.node_tree.nodes['Principled BSDF']
        p.inputs['Base Color'].default_value = (*color, 1)
        p.inputs['Roughness'].default_value = rough
        p.inputs['Metallic'].default_value = metal
        if emit:
            p.inputs['Emission Color'].default_value = (*emit[0], 1)
            p.inputs['Emission Strength'].default_value = emit[1]
            m.cycles.emission_sampling = 'NONE'                  # hit by camera/glossy rays only, never light-sampled
        return m

    def concrete():
        m = mat('concrete', (0.32, 0.31, 0.29))
        nt = m.node_tree; p = nt.nodes['Principled BSDF']
        noise = nt.nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = 3.0
        noise.inputs['Detail'].default_value = 8.0
        ramp = nt.nodes.new('ShaderNodeValToRGB')
        ramp.color_ramp.elements[0].color = (0.20, 0.19, 0.18, 1); ramp.color_ramp.elements[1].color = (0.42, 0.41, 0.38, 1)
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

    def add_obj(name, bm, material, shadow=True, parent=None):
        me = bpy.data.meshes.new(name); bm.to_mesh(me); bm.free()
        ob = bpy.data.objects.new(name, me); sc.collection.objects.link(ob)
        ob.data.materials.append(material); ob.visible_shadow = shadow; ob.parent = parent
        return ob

    def seg(name, p0, p1, r, material, parent=None, shadow=True, segs=24, caps=True):
        """Cylinder of radius r from p0 to p1 (in the parent's frame)."""
        p0, p1 = Vector(p0), Vector(p1); d = p1 - p0
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, segments=segs, cap_ends=caps, radius1=r, radius2=r, depth=d.length)
        bmesh.ops.rotate(bm, matrix=Vector((0, 0, 1)).rotation_difference(d.normalized()).to_matrix(), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(p0 + p1) / 2, verts=bm.verts)
        return add_obj(name, bm, material, shadow, parent)

    def ball(name, c, r, material, scale=(1, 1, 1), parent=None):
        bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=16, radius=r)
        bmesh.ops.scale(bm, vec=scale, verts=bm.verts); bmesh.ops.translate(bm, vec=c, verts=bm.verts)
        for f in bm.faces:
            f.smooth = True
        return add_obj(name, bm, material, True, parent)

    def limb(name, p0, p1, r, material):
        seg(name, p0, p1, r, material); ball(name + '_j', p1, r, material)

    # ── tunnel (as light_cone_render.py) ──
    bm = bmesh.new(); n = 192; ph0 = math.asin(yb / R)
    prof = [(R * math.cos(p), R * math.sin(p)) for p in (ph0 + (math.pi - 2 * ph0) * i / n for i in range(n + 1))]
    nx = 160
    vs = [[bm.verts.new((-LEN / 2 + LEN * j / nx, y, z)) for (y, z) in prof] for j in range(nx + 1)]
    for j in range(nx):
        for i in range(n):
            bm.faces.new((vs[j][i], vs[j][i + 1], vs[j + 1][i + 1], vs[j + 1][i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    for f in bm.faces:
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

    # ── stick frame: lower hand at the origin, stick along +Z, tilted forward about Y, rig yawed about the stick ──
    grip = Vector((0.22, -0.05, yb + 0.98))
    tilt = bpy.data.objects.new('stick_tilt', None); sc.collection.objects.link(tilt)
    tilt.location = grip; tilt.rotation_euler = (0, math.radians(TILT), 0)
    spin = bpy.data.objects.new('rig_yaw', None); sc.collection.objects.link(spin)
    spin.parent = tilt; spin.rotation_euler = (0, 0, math.radians(YAW))
    bpy.context.view_layer.update()
    W = spin.matrix_world.copy()                          # stick frame -> world

    # ── rig: Pi box, lidar, GoPro, monopod, from the exported meshes (mm, rig frame) ──
    parts = pbr.load_parts()
    for name in ('core_cut_near', 'core_cut_far'):       # section views only; the full core is shown
        bpy.data.objects.remove(parts.pop(name))
    for name, ob in parts.items():
        ob.parent = spin; ob.location = (0, 0, rig_s)
        if name in pbr.GOPRO:
            ob.visible_shadow = False                     # the trace already has the GoPro's shadow
    seg('pole', (0, 0, -0.15), (0, 0, STICK + 0.01), 0.015, pbr.material('alu_dark'), spin, segs=32)
    seg('grip_foam', (0, 0, -0.12), (0, 0, 0.5), 0.019, pbr.material('black'), spin, segs=32)

    # ── lamp (calibrated emitters, see the docstring) ──
    up_cd = ies_up_candela(OUT / 'cone.ies')
    led_S = G['flux_lm'] / (683 * math.pi * (G['led_r'] * s) ** 2)          # exitance, W/m^2 (1 W = 683 lm)
    dif_S = math.pi * up_cd / (math.pi * (G['r_rim'] * s) ** 2) / 683
    print(f'LED exitance {led_S:.3g} W/m2, diffuser {dif_S:.3g} W/m2 (straight up {up_cd:.0f} cd)')
    black = mat('lamp_black', (0.02, 0.02, 0.02), rough=0.5)
    zl = GOPRO_TOP + G['lift']                            # LED face, rig frame mm

    def R(x, y, z):                                       # rig frame mm -> stick frame m
        return Vector((x * s, y * s, rig_s + z * s))

    def box(name, x0, x1, y0, y1, z0, z1, material, shadow=True):
        bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1)
        bmesh.ops.scale(bm, vec=((x1 - x0) * s, (y1 - y0) * s, (z1 - z0) * s), verts=bm.verts)
        bmesh.ops.translate(bm, vec=R((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2), verts=bm.verts)
        return add_obj(name, bm, material, shadow, spin)

    def tube(name, pts, r, material):
        obs = [seg(f'{name}{k}', R(*a), R(*b), r * s, material, spin, segs=16)
               for k, (a, b) in enumerate(zip(pts, pts[1:]))]
        return obs + [ball(f'{name}_b{k}', R(*p), r * s, material, parent=spin) for k, p in enumerate(pts[1:-1])]

    copper = mat('copper', (0.95, 0.55, 0.38), 0.3, 1.0)
    brass = mat('brass', (0.85, 0.7, 0.35), 0.3, 1.0)
    printed = pbr.material('clear_blue')
    hot, cold = mat('hose_hot', (0.45, 0.05, 0.03), 0.5), mat('hose_cold', (0.04, 0.12, 0.45), 0.5)
    if LIGHTS == 'cone':
        head = [
            seg('led_board', R(0, 0, zl - 2.4), R(0, 0, zl - 0.8), 0.022, mat('alu_plate', (0.7, 0.7, 0.72), 0.3, 1.0),
                spin, segs=48),
            box('cold_plate', -19, 19, -22, 22, zl - 5.4, zl - 2.4, copper),          # 3 mm copper
            box('water_block', -19.5, 19.5, -27, 27, zl - 21.4, zl - 5.4, printed), # printed, sits on the GoPro top
        ]
        # barbs on the block's +Y face: outlet high (air leaves the block), inlet low (feeds the centre jet)
        Z_OUT, Z_IN = zl - 9, zl - 17
        for nme, x, z in (('barb_out', 14, Z_OUT), ('barb_in', 8, Z_IN)):
            head.append(seg(nme, R(x, 27, z), R(x, 34, z), 2.5 * s, brass, spin, segs=16))
        for i in range(3):                                    # dish struts from the block's top edge
            a = 2 * math.pi * i / 3 + math.pi / 6
            head.append(seg(f'strut{i}', R(18 * math.cos(a), 18 * math.sin(a), zl - 5.4),
                            R(G['r_rim'] * math.cos(a), G['r_rim'] * math.sin(a), zl + G['z_top']), 1.5 * s, black,
                            spin, segs=8))
        for ob in head:
            ob.visible_shadow = False                         # the trace has the block's shadow
        # one artery: cold hose, hot hose and the LED power lead side by side down the GoPro's +Y face (stitch line), under
        # the GoPro onto the cross plate's +X+Y arm, then down the outside of the (40, 40) standoff stacked along the
        # diagonal (blue inner, red, power outer) so the standoff hides them from the lidar; two clips hold the lot
        lead = mat('lead', (0.03, 0.03, 0.03), 0.6)
        loop = tube('hose_in', [(8, 34, Z_IN), (8, 38, Z_IN), (8, 38, 88), (20.2, 11.8, 80.1), (40.2, 31.8, 80.1),
                                (47.5, 47.5, 80.1), (47.5, 47.5, 4), (49, 81, 4), (49, 81, -102), (45, 81, -102)], 3,
                    cold)
        loop += tube('hose_out', [(14, 34, Z_OUT), (14, 38, Z_OUT), (14, 38, 88), (16, 16, 80.1), (36, 36, 80.1),
                                  (52, 52, 80.1), (52, 52, 11), (-10, 66, 11), (-10, 80, 11), (-10, 80, -16)], 3, hot)
        loop += tube('led_lead', [(2, 27, zl - 3.5), (2, 38, zl - 3.5), (2, 38, 88), (11.8, 20.2, 80.1),
                                  (31.8, 40.2, 80.1), (56.5, 56.5, 80.1), (56.5, 56.5, 17), (32, 72, 17), (32, 81, 17),
                                  (32, 81, -22)], 2.5, lead)
        for nme, a, b, z, r in (('clip_arm', (21.8, 30.2), (30.2, 21.8), 80.1, 5),       # strap across the arm
                                ('clip_standoff', (38, 38), (61, 61), 40, 5.5)):        # strap round standoff + artery
            loop.append(seg(nme, R(*a, z), R(*b, z), r * s, printed, spin, segs=16))
        # cooling module in a ~92 mm square on the pod panel, 5 mm off it: 60 mm-class radiator + 10 mm fan on the -X
        # side, a 28 mm column on the +X side with the LED driver (DeWalt lead in), printed reservoir and mini pump below
        loop += [box('radiator', -46, 14, 66, 86, -114, -22, pbr.material('alu_dark')),
                 box('rad_fan', -46, 14, 86, 96, -98, -38, pbr.material('black')),
                 seg('rad_fan_hub', R(-16, 86, -68), R(-16, 97, -68), 12 * s, pbr.material('cam'), spin, segs=32),
                 seg('rad_port_top', R(-10, 80, -22), R(-10, 80, -16), 2.5 * s, brass, spin, segs=16),
                 seg('rad_port_bot', R(14, 76, -85), R(18, 76, -85), 2.5 * s, brass, spin, segs=16),   # into the res
                 box('led_driver', 18, 46, 66, 96, -40, -22, pbr.material('black')),
                 box('reservoir', 18, 46, 66, 96, -90, -40, printed),
                 seg('pump', R(32, 81, -114), R(32, 81, -90), 13 * s, pbr.material('black'), spin, segs=32)]
        led_m = mat('led', (1, 1, 0.9), emit=((1, 0.97, 0.9), led_S))
        leds = seg('leds', (0, 0, lamp_s - 0.0008), (0, 0, lamp_s), G['led_r'] * s, led_m, spin, shadow=False, segs=48)
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, segments=96, cap_ends=False, radius1=G['r_hole'] * s, radius2=G['r_rim'] * s,
                              depth=(G['z_top'] - G['z_tip']) * s)
        bmesh.ops.translate(bm, vec=(0, 0, lamp_s + (G['z_tip'] + G['z_top']) / 2 * s), verts=bm.verts)
        dish = add_obj('dish', bm, mat('mirror', (0.95, 0.95, 0.95), rough=0.03, metal=1.0), False, spin)
        dish.modifiers.new('shell', 'SOLIDIFY').thickness = 0.0012
        opal_m = mat('opal', (0.9, 0.9, 0.9), rough=0.6, emit=((1, 1, 1), dif_S))
        diff = seg('diffuser', (0, 0, lamp_s + (G['z_top'] + 1.2) * s), (0, 0, lamp_s + (G['z_top'] + 3.2) * s),
                   G['r_rim'] * s, opal_m, spin, shadow=False, segs=96)
        for ob in (leds, diff):
            ob.visible_diffuse = False; ob.visible_transmission = False
        light = bpy.data.lights.new('flash', 'POINT'); light.shadow_soft_size = G['led_r'] * s
        lob = bpy.data.objects.new('flash', light); lob.parent = spin; lob.location = (0, 0, lamp_s + 0.002)
        sc.collection.objects.link(lob)
        light.use_nodes = True
        nt = light.node_tree; ies = nt.nodes.new('ShaderNodeTexIES'); ies.mode = 'EXTERNAL'
        ies.filepath = str(OUT / 'cone.ies')
        nt.links.new(ies.outputs['Fac'], nt.nodes['Emission'].inputs['Strength'])
        light.energy = 4 * math.pi / 683 * lcr.IES_GAIN       # 1 cd in the file = 1 cd in the scene

        def lamp_on(on):
            lob.hide_render = not on
            led_m.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = led_S if on else 0
            opal_m.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = dif_S if on else 0

    else:
        # ── three flush lights: +-X beside the tracking cameras, -Y (forward) on the Pi wall ──
        win = []
        lamps = []
        for nme, c, n, lm in (('lamp_l', (52.2, -5, -30), (1, 0, 0), SIDE_LM),
                              ('lamp_r', (-52.2, -5, -30), (-1, 0, 0), SIDE_LM),
                              ('lamp_fwd', (0, -60.5, -45), (0, -1, 0), FWD_LM)):
            c, n = Vector(c), Vector(n)
            S = lm / (683 * math.pi * (WIN_R * s) ** 2)                 # window exitance, W/m^2
            m = mat(nme + '_win', (0.9, 0.9, 0.88), rough=0.6, emit=((1, 0.97, 0.92), S))
            w = seg(nme + '_win', R(*c), R(*(c + n * 0.6)), WIN_R * s, m, spin, shadow=False, segs=48)
            w.visible_diffuse = False; w.visible_transmission = False
            seg(nme + '_bezel', R(*(c - n * 0.2)), R(*(c + n * 0.4)), (WIN_R + 2) * s, black, spin, segs=48)
            L = bpy.data.lights.new(nme, 'AREA'); L.shape = 'DISK'; L.size = 2 * WIN_R * s
            L.energy = lm / 683                                          # measured: area W = lm / 683
            o = bpy.data.objects.new(nme, L); o.parent = spin; sc.collection.objects.link(o)
            o.location = R(*(c + n * 1.0)); o.rotation_euler = n.to_track_quat('-Z', 'Y').to_euler()
            win.append((m, S)); lamps.append((o, c, n, lm))
        loop = [w for w in bpy.data.objects if w.name.endswith(('_win', '_bezel'))]

        def lamp_on(on):
            for o, *_ in lamps:
                o.hide_render = not on
            for m, S in win:
                m.node_tree.nodes['Principled BSDF'].inputs['Emission Strength'].default_value = S if on else 0

    # ── operator: feet at the origin on the bed, facing +X ──
    fz = yb
    skin, cloth = mat('skin', (0.55, 0.38, 0.3), 0.6), mat('cloth', (0.05, 0.06, 0.09), 0.85)
    vest, hat = mat('hivis', (0.9, 0.42, 0.02), 0.6), mat('hardhat', (0.85, 0.85, 0.82), 0.35)
    for sy in (-1, 1):
        hip, knee, ank = (0, 0.1 * sy, fz + 0.92), (0.02, 0.105 * sy, fz + 0.5), (0, 0.11 * sy, fz + 0.09)
        limb(f'thigh{sy}', hip, knee, 0.07, cloth); limb(f'shin{sy}', knee, ank, 0.055, cloth)
        bm = bmesh.new(); bmesh.ops.create_cube(bm, size=1)
        bmesh.ops.scale(bm, vec=(0.27, 0.1, 0.08), verts=bm.verts)
        bmesh.ops.translate(bm, vec=(0.06, 0.11 * sy, fz + 0.04), verts=bm.verts); add_obj(f'boot{sy}', bm, cloth)
    ball('pelvis', (0, 0, fz + 0.95), 0.16, cloth, (0.8, 1.05, 0.7))
    seg('torso', (0, 0, fz + 0.98), (0, 0, fz + 1.36), 0.15, vest, segs=32).scale = (0.75, 1.2, 1)
    ball('chest', (0, 0, fz + 1.36), 0.15, vest, (0.75, 1.2, 0.6))
    seg('neck', (0, 0, fz + 1.4), (0.01, 0, fz + 1.53), 0.05, skin)
    ball('head', (0.02, 0, fz + 1.63), 0.105, skin, (1.05, 0.9, 1.1))
    ball('hat', (0.02, 0, fz + 1.68), 0.13, hat, (1.1, 0.95, 0.65))
    hands = {-1: W @ Vector((0, 0, 0)), 1: W @ Vector((0, 0, 0.4))}     # right hand low, left hand high
    for sy, hand in hands.items():
        sh = Vector((0, 0.2 * sy, fz + 1.43)); a_, b_ = 0.31, 0.29
        d = hand - sh; L = min(d.length, a_ + b_ - 1e-4)
        hand = sh + d.normalized() * L
        x = (a_ ** 2 - b_ ** 2 + L ** 2) / (2 * L); hgt = math.sqrt(max(a_ ** 2 - x ** 2, 0))
        hint = Vector((0, 0.5 * sy, -1)); hint -= hint.project(d); hint.normalize()
        elbow = sh + d.normalized() * x + hint * hgt
        ball(f'shoulder{sy}', sh, 0.06, vest)
        limb(f'upperarm{sy}', sh, elbow, 0.05, cloth); limb(f'forearm{sy}', elbow, hand, 0.042, cloth)
        ball(f'hand{sy}', hand, 0.048, skin, (1, 1, 1.3))

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
    sc.cycles.caustics_reflective = False; sc.cycles.caustics_refractive = False
    sc.view_settings.view_transform = 'Standard'
    sc.render.image_settings.file_format = 'OPEN_EXR'
    sc.world = bpy.data.worlds.new('dark'); sc.world.use_nodes = True
    bg = sc.world.node_tree.nodes['Background'].inputs['Strength']; bg.default_value = 0.0

    def aim(o, loc, target, up=Vector((0, 0, 1))):
        o.location = loc
        f = (Vector(target) - Vector(loc)).normalized(); r = f.cross(up).normalized(); u = r.cross(f)
        o.rotation_euler = Matrix((r, u, -f)).transposed().to_euler()

    def cam(name, loc, target, lens=None, fisheye=False, up=Vector((0, 0, 1)), fov=200):
        c = bpy.data.cameras.new(name); c.clip_start = 0.003
        if fisheye:
            c.type = 'PANO'; c.panorama_type = 'FISHEYE_EQUIDISTANT'; c.fisheye_fov = math.radians(fov)
        else:
            c.lens = lens
        o = bpy.data.objects.new(name, c); sc.collection.objects.link(o); aim(o, loc, target, up)
        return o

    fills = []

    def fill(name, loc, target, size, power):
        l = bpy.data.lights.new(name, 'AREA'); l.size = size; l.energy = power
        o = bpy.data.objects.new(name, l); sc.collection.objects.link(o); aim(o, loc, target); fills.append(o)
        return o

    top_c = W @ R(0, 25, 150)
    ax = (W.to_3x3() @ Vector((0, 0, 1))).normalized()
    stick_up = ax
    eye = Vector((0.1, 0, fz + 1.65))
    if LIGHTS == 'cone':
        lamp_w = W @ Vector((0, 0, lamp_s)); v = eye - lamp_w
        va, cdv = ies_profile(OUT / 'cone.ies'); ang = math.degrees(v.angle(-ax))
        cd_eye = float(np.interp(ang, va, cdv))
        print(f'operator eye: {ang:.1f} deg off the lamp nadir, {v.length:.2f} m: {cd_eye:.0f} cd, '
              f'{cd_eye / v.length ** 2:.0f} lx direct at the eye')
        hd, rd = W @ R(0, 0, zl - 8), W @ R(0, 45, 60)
    else:
        for o, c, n, lm in lamps:                                   # Lambertian window toward the eye
            v = eye - W @ R(*c); cos = (W.to_3x3() @ n).normalized().dot(v.normalized())
            print(f'operator eye vs {o.name}: {math.degrees(math.acos(cos)):.0f} deg off its axis, {v.length:.2f} m: '
                  f'{max(cos, 0) * lm / math.pi / v.length ** 2:.0f} lx direct at the eye')
        hd = rd = W @ R(0, -20, -60)
    frames = [
        ('wide', cam('wide', (3.6, 2.0, yb + 1.7), (0.3, -0.1, yb + 1.25), lens=17), (1920, 1080), True),
        ('side_geo', cam('side_geo', (0.55, 2.55, yb + 1.15), (0.55, 0, yb + 1.15), lens=17), (1400, 1400), False),
        ('close_geo', cam('close_geo', top_c + Vector((0.5, 0.62, 0.1)), top_c, lens=40, up=stick_up),
         (1200, 1600), False),
    ]
    if LIGHTS == 'cone':
        frames += [
            ('head_geo', cam('head_geo', hd + Vector((0.17, 0.2, 0.05)), hd, lens=50, up=stick_up), (1400, 1200),
             False),
            ('rad_geo', cam('rad_geo', rd + (W.to_3x3() @ Vector((0.45, 0.95, 0.2))), rd, lens=40, up=stick_up),
             (1100, 1500), False)]
    else:
        for side, sx in (('l', 1), ('r', -1)):
            frames.append((f'box_geo_{side}', cam(f'box_geo_{side}', hd + (W.to_3x3() @ Vector((0.3 * sx, -0.32, 0.08))),
                                                  hd, lens=50, up=stick_up), (1200, 1400), False))
            p = W @ R(sx * TRACK_C, -40.1, -30); d = W.to_3x3() @ Vector((sx, 0, 0))
            frames.append((f'track_{side}', cam(f'track_{side}', p, p + d, fisheye=True, up=stick_up, fov=160),
                           (1200, 1200), True))
    for side, sx in (('front', 1), ('back', -1)):
        p = W @ (Vector(LENS_C) * s * Vector((sx, 1, 1)) + Vector((sx * 0.0005, 0, rig_s)))
        d = W.to_3x3() @ Vector((sx, 0, 0))
        frames.append((f'lens_{side}', cam(f'lens_{side}', p, p + d, fisheye=True, up=stick_up), (1600, 1600), True))
    key = fill('key', top_c + Vector((0.5, -0.6, 0.6)), top_c, 0.5, 60)
    rim = fill('rim', top_c + Vector((-0.6, 0.3, 0.4)), top_c, 0.4, 30)
    big = fill('big', (1.5, 2.2, yb + 2.3), (0.5, 0, yb + 1.1), 1.5, 800)
    fill('rad_key', rd + (W.to_3x3() @ Vector((-0.2, 0.6, 0.4))), rd, 0.4, 40)
    if LIGHTS == 'box3':
        fill('box_key', rd + (W.to_3x3() @ Vector((0.1, -0.6, 0.5))), rd, 0.4, 40)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / f'{PFX}.blend'))

    def expose(a, path):
        lum = a @ np.array([0.2126, 0.7152, 0.0722])
        x = a * (0.18 / np.median(lum[lum > 0]))
        x = x / (1 + x / 4.0)
        x = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(np.maximum(x, 0), 1 / 2.4) - 0.055)
        h, w = a.shape[:2]
        out = bpy.data.images.new(path.stem, w, h)
        out.pixels[:] = np.concatenate([np.clip(x, 0, 1), np.ones_like(x[..., :1])], -1).ravel()
        out.filepath_raw = str(path); out.file_format = 'PNG'; out.save()

    for name, o, (w, h), lit in frames:
        lamp_on(lit)
        for f in fills:
            f.hide_render = lit
        bg.default_value = 0.0 if lit else 0.02
        lens = name.startswith('lens')
        bpy.data.objects['gopro_lenses'].hide_render = lens                 # the camera sits at the lens
        for nme in ('lens_near', 'lens_far'):
            bpy.data.objects[nme].hide_render = name.startswith('track')
        sc.camera = o; sc.render.resolution_x, sc.render.resolution_y = w, h
        sc.render.filepath = str(OUT / f'{PFX}_{name}.exr')
        bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(sc.render.filepath)
        a = np.array(im.pixels[:], np.float32).reshape(h, w, 4)[..., :3]
        expose(a, OUT / f'{PFX}_{name}.png')
        if lens or name.startswith('track'):
            expose(glare(a), OUT / f'{PFX}_{name}_glare.png')
    lidar_band(sc, spin, W, R, loop)


def lidar_band(sc, spin, W, R, loop):
    """Silhouettes around the Mid-360 (alpha), rig alone vs rig + cooling loop; prints the band's blocked fraction."""
    import bpy
    import numpy as np
    from mathutils import Matrix, Vector
    for o in bpy.data.objects:
        if o.type == 'MESH':
            o.hide_render = o.parent is not spin or o.name == 'mid360s'
    sc.render.film_transparent = True; sc.cycles.samples = 16; sc.cycles.use_denoising = False
    c = bpy.data.cameras.new('lidar'); c.type = 'PANO'; c.panorama_type = 'EQUIRECTANGULAR'; c.clip_start = 0.005
    o = bpy.data.objects.new('lidar_band', c); sc.collection.objects.link(o)
    # camera looks along rig +Y with the stick as up, so image rows are lidar elevation
    look = Matrix(((1, 0, 0), (0, 0, 1), (0, -1, 0))).transposed()      # cam X = +X, cam Y = +Z, cam -Z = +Y
    o.matrix_world = W @ Matrix.Translation(R(0, 0, LIDAR_C)) @ look.to_4x4()
    sc.camera = o; w, h = 2048, 1024; sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.image_settings.color_mode = 'RGBA'
    al = {}
    for tag, show in (('rig', False), ('loop', True)):
        for ob in loop:
            ob.hide_render = not show
        sc.render.filepath = str(OUT / f'{PFX}_lidar_{tag}.exr'); bpy.ops.render.render(write_still=True)
        im = bpy.data.images.load(sc.render.filepath)
        al[tag] = np.array(im.pixels[:], np.float32).reshape(h, w, 4)[..., 3] > 0.5
    el = (np.arange(h) + 0.5) / h * 180 - 90                 # pixel rows run bottom-up
    band = (el >= -7) & (el <= 52)
    wgt = np.cos(np.radians(el))[band][:, None]
    f = lambda m: float((m[band] * wgt).sum() / (wgt.sum() * w))
    print(f'lidar band blocked: rig {f(al["rig"]):.1%}, rig + added parts {f(al["loop"]):.1%} '
          f'(loop adds {f(al["loop"] & ~al["rig"]):.1%})')
    rgb = np.ones((h, w, 3), np.float32)
    rgb[al['rig']] = 0.55; rgb[al['loop'] & ~al['rig']] = (0.85, 0.1, 0.1)
    for e in (-7, 52):
        r = int((e + 90) / 180 * h); rgb[r - 1:r + 2] = (0.1, 0.4, 0.9)
    out = bpy.data.images.new('lidar_band', w, h)
    out.pixels[:] = np.concatenate([rgb, np.ones((h, w, 1), np.float32)], -1).ravel()
    out.filepath_raw = str(OUT / f'{PFX}_lidar_band.png'); out.file_format = 'PNG'; out.save()


def montage():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    fig, axs = plt.subplots(2, 2, figsize=(16, 16.5))
    for i, side in enumerate(('front', 'back')):
        for j, g in enumerate(('', '_glare')):
            axs[i, j].imshow(mpimg.imread(OUT / f'{PFX}_lens_{side}{g}.png')); axs[i, j].axis('off')
            axs[i, j].set_title(f'MAX2 {side} lens, {"with veiling glare" if g else "no glare"}', fontsize=14)
    fig.suptitle(f'Stick {TILT:.0f} deg forward, rig yaw {YAW:.0f} deg; 200 deg equidistant fisheye', fontsize=15)
    fig.tight_layout(); out = OUT / f'{PFX}_lenses_tilt{TILT:.0f}.png'; fig.savefig(out, dpi=70); print(out)


if __name__ == '__main__':
    if 'bpy' in sys.modules or Path(sys.executable).name.lower().startswith('blender'):
        build_and_render()
    else:
        subprocess.run([BLENDER, '-b', '--factory-startup', '-P', __file__, '--', *sys.argv[1:]], check=True)
        montage()
