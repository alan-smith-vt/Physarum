"""Blender renders of the chosen Pi box rig (run AFTER designs/pi_box_export.py).

    blender -b -P designs/pi_box_render.py -- <out_dir> [view ...] [--samples N] [--res W H]

Views: box, assembled, assembled_gopro, exploded, internals. Cycles on the GPU (OptiX,
falling back to CUDA, then CPU). Meshes come from out/render/ in the rig frame (mm, Z up).
"""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

HERE = Path(__file__).resolve().parent
MESHES = HERE.parent / 'out' / 'render'
ARGV = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else []
OUT_DIR = Path(ARGV[0]) if ARGV else MESHES / 'png'
VIEWS = [a for a in ARGV[1:] if not a.startswith('--') and not a.isdigit()] or \
    ['box', 'assembled', 'assembled_gopro', 'exploded', 'internals']
SAMPLES = int(ARGV[ARGV.index('--samples') + 1]) if '--samples' in ARGV else 96
RES = (int(ARGV[ARGV.index('--res') + 1]), int(ARGV[ARGV.index('--res') + 2])) if '--res' in ARGV else (1600, 1200)
MM = 0.001

# ── materials ──
MATS = {
    'resin':     dict(color=(0.055, 0.058, 0.065), rough=0.42, spec=0.45),
    'clear':     dict(color=(0.85, 0.9, 0.95), rough=0.06, transmission=1.0, ior=1.5),
    'alu':       dict(color=(0.86, 0.87, 0.89), rough=0.28, metal=1.0),
    'alu_dark':  dict(color=(0.55, 0.56, 0.58), rough=0.35, metal=1.0),
    'black':     dict(color=(0.02, 0.02, 0.022), rough=0.5),
    'gloss':     dict(color=(0.01, 0.01, 0.012), rough=0.08, spec=0.8),
    'pcb':       dict(color=(0.04, 0.32, 0.12), rough=0.45),
    'lidar':     dict(color=(0.06, 0.065, 0.07), rough=0.35),     # Mid-360S housing is dark
    'battery':   dict(color=(0.035, 0.035, 0.04), rough=0.35),
    'orange':    dict(color=(0.85, 0.36, 0.05), rough=0.45),
    'ffc':       dict(color=(0.72, 0.42, 0.12), rough=0.35),
    'red':       dict(color=(0.6, 0.04, 0.03), rough=0.4),
    'white':     dict(color=(0.82, 0.82, 0.8), rough=0.4),
    'navy':      dict(color=(0.03, 0.05, 0.1), rough=0.45),
    'cam':       dict(color=(0.03, 0.03, 0.035), rough=0.3),
    'steel':     dict(color=(0.8, 0.8, 0.82), rough=0.2, metal=1.0),
}
PART_MAT = {
    'core': 'resin', 'floor': 'resin', 'panel_pi': 'resin', 'panel_pod': 'clear',
    'base_plate': 'alu', 'cross': 'alu', 'standoffs': 'alu', 'box_spacers': 'alu_dark', 'coupler': 'alu_dark',
    'mid360s': 'lidar', 'pi5': 'pcb', 'cooler': 'alu_dark', 'cameras': 'cam', 'battery': 'battery',
    'buck_wago': 'orange', 'eth_adapter': 'white', 'power_switch': 'steel', 'pd_trigger': 'black',
    'monopod': 'black', 'cable_ribbons': 'ffc', 'cable_lidar': 'black', 'cable_lidar_power': 'red',
    'cable_battery': 'red', 'cable_buck_in': 'red', 'cable_usbc': 'black',
    'gopro_adapter': 'navy', 'gopro_cage': 'black', 'gopro_body': 'black', 'gopro_lenses': 'gloss',
}
BOX = ['core', 'floor', 'panel_pi', 'panel_pod', 'power_switch']
INSIDE = ['pi5', 'cooler', 'cameras', 'battery', 'buck_wago', 'eth_adapter', 'pd_trigger', 'box_spacers',
          'cable_ribbons', 'cable_lidar_power', 'cable_battery', 'cable_buck_in', 'cable_usbc']
TOP = ['base_plate', 'standoffs', 'cross', 'mid360s', 'coupler']
GOPRO = ['gopro_adapter', 'gopro_cage', 'gopro_body', 'gopro_lenses']
POLE = ['monopod']


def material(name):
    if name in bpy.data.materials:
        return bpy.data.materials[name]
    spec = MATS[name]
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*spec['color'], 1.0)
    b.inputs['Roughness'].default_value = spec.get('rough', 0.5)
    b.inputs['Metallic'].default_value = spec.get('metal', 0.0)
    if 'spec' in spec:
        b.inputs['Specular IOR Level'].default_value = spec['spec']
    if spec.get('transmission'):
        b.inputs['Transmission Weight'].default_value = spec['transmission']
        b.inputs['IOR'].default_value = spec.get('ior', 1.45)
    return m


def load_parts():
    objs = {}
    for f in sorted(MESHES.iterdir()):
        if f.suffix not in ('.ply', '.stl'):
            continue
        name = f.stem
        before = set(bpy.data.objects)
        if f.suffix == '.ply':
            bpy.ops.wm.ply_import(filepath=str(f))
        else:
            bpy.ops.wm.stl_import(filepath=str(f))
        ob = (set(bpy.data.objects) - before).pop()
        ob.name = name
        ob.scale = (MM, MM, MM)
        ob.data.materials.clear()
        ob.data.materials.append(material(PART_MAT.get(name, 'resin')))
        if f.suffix == '.ply':                          # smooth printed/aluminium parts, keep edges crisp
            for p in ob.data.polygons:
                p.use_smooth = True
            bpy.context.view_layer.objects.active = ob
            mod = ob.modifiers.new('weighted', 'WEIGHTED_NORMAL')
            mod.keep_sharp = True
        objs[name] = ob
    return objs


def setup_scene():
    sc = bpy.context.scene
    sc.render.engine = 'CYCLES'
    prefs = bpy.context.preferences.addons['cycles'].preferences
    for dev_type in ('OPTIX', 'CUDA'):
        try:
            prefs.compute_device_type = dev_type
            prefs.get_devices()
            gpus = [d for d in prefs.devices if d.type == dev_type]
            if gpus:
                for d in prefs.devices:
                    d.use = d.type == dev_type
                sc.cycles.device = 'GPU'
                print(f'render device: {dev_type} {[d.name for d in gpus]}')
                break
        except TypeError:
            continue
    sc.cycles.samples = SAMPLES
    sc.cycles.use_denoising = True
    sc.render.resolution_x, sc.render.resolution_y = RES
    sc.render.film_transparent = False
    sc.view_settings.view_transform = 'AgX'
    sc.view_settings.look = 'AgX - Base Contrast'

    # World: Blender's bundled studio HDRI lights the scene and gives metal something to reflect;
    # the camera sees a plain mid-grey backdrop instead (Light Path: Is Camera Ray).
    world = bpy.data.worlds.new('studio')
    sc.world = world
    world.use_nodes = True
    nt = world.node_tree
    nt.nodes.clear()
    hdri = Path(bpy.utils.resource_path('LOCAL')) / 'datafiles' / 'studiolights' / 'world' / 'studio.exr'
    env = nt.nodes.new('ShaderNodeTexEnvironment')
    env.image = bpy.data.images.load(str(hdri))
    light_bg = nt.nodes.new('ShaderNodeBackground')
    light_bg.inputs['Strength'].default_value = 0.6
    cam_bg = nt.nodes.new('ShaderNodeBackground')
    cam_bg.inputs['Color'].default_value = (0.42, 0.44, 0.47, 1.0)
    cam_bg.inputs['Strength'].default_value = 1.0
    path = nt.nodes.new('ShaderNodeLightPath')
    mix = nt.nodes.new('ShaderNodeMixShader')
    out = nt.nodes.new('ShaderNodeOutputWorld')
    nt.links.new(env.outputs['Color'], light_bg.inputs['Color'])
    nt.links.new(path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    nt.links.new(light_bg.outputs['Background'], mix.inputs[1])
    nt.links.new(cam_bg.outputs['Background'], mix.inputs[2])
    nt.links.new(mix.outputs['Shader'], out.inputs['Surface'])

    # ground: a mid-grey matte floor that catches soft shadows without bouncing much light
    bpy.ops.mesh.primitive_plane_add(size=6.0, location=(0, 0, -0.32))
    ground = bpy.context.object
    gm = bpy.data.materials.new('ground')
    gm.use_nodes = True
    gm.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.32, 0.33, 0.35, 1)
    gm.node_tree.nodes['Principled BSDF'].inputs['Roughness'].default_value = 0.95
    ground.data.materials.append(gm)

    def area(name, loc, size, power, color=(1, 1, 1)):
        bpy.ops.object.light_add(type='AREA', location=loc)
        lt = bpy.context.object
        lt.name = name
        lt.data.size = size
        lt.data.energy = power
        lt.data.color = color
        aim(lt, Vector((0, 0, -0.04)))
        return lt

    area('key', (0.55, -0.65, 0.75), 0.6, 18)
    area('rim', (-0.2, 0.75, 0.6), 0.5, 14, (1.0, 0.97, 0.92))
    return ground


def aim(ob, target):
    d = target - ob.location
    ob.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()


def camera(name, target, direction, dist, lens=50):
    bpy.ops.object.camera_add()
    cam = bpy.context.object
    cam.name = name
    cam.data.lens = lens
    cam.data.clip_start = 0.01
    t = Vector(target)
    cam.location = t + Vector(direction).normalized() * dist
    aim(cam, t)
    bpy.context.scene.camera = cam
    return cam


def show(objs, names, offsets=None):
    offsets = offsets or {}
    for n, ob in objs.items():
        ob.hide_render = n not in names
        ob.location = Vector(offsets.get(n, (0, 0, 0))) * MM


def render(path):
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print('wrote', path, flush=True)


def main():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    setup_scene()
    objs = load_parts()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    iso = (1.0, -1.15, 0.8)                                  # from +X / -Y (Pi side), above
    pod = (0.95, 1.1, 0.65)                                  # from +X / +Y (battery side)
    every = BOX + INSIDE + TOP + POLE + ['cable_lidar']

    for view in VIEWS:
        for c in [o for o in bpy.data.objects if o.type == 'CAMERA']:
            bpy.data.objects.remove(c)
        if view == 'box':
            show(objs, BOX + INSIDE)
            camera('cam', (0, 0.01, -0.095), iso, 0.62)
        elif view == 'assembled':
            show(objs, every)
            camera('cam', (0, 0.005, -0.04), iso, 0.78)
        elif view == 'assembled_gopro':
            show(objs, every + GOPRO)
            camera('cam', (0, 0.005, -0.01), pod, 0.86)
        elif view == 'exploded':
            off = {n: (0, 0, 70) for n in TOP}
            off.update({n: (0, 0, 115) for n in GOPRO})
            off.update(floor=(0, 0, -75), panel_pi=(0, -60, -35), panel_pod=(0, 60, -35),
                       monopod=(0, 0, -150), coupler=(0, 0, 70))
            show(objs, BOX + INSIDE + TOP + GOPRO + POLE, off)
            camera('cam', (0, 0.0, 0.0), iso, 1.3)
        elif view == 'internals':
            show(objs, ['core', 'floor', 'power_switch'] + INSIDE)
            camera('cam', (0, 0.0, -0.095), (0.75, -1.25, 0.35), 0.6)
        else:
            print('unknown view', view)
            continue
        render(OUT_DIR / f'pi-box-{view}.png')
    (OUT_DIR / 'views.json').write_text(json.dumps(VIEWS))


main()
