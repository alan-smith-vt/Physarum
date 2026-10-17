"""Serve pre-generated slice data to the Three.js viewer.

    /                 the last design generated (_slices.bin / _meta.json)
    /<design>         any design kept by generate.py under out/preview/<design>/
    /<alias>          short names for the Pi box parts (ALIASES below)
    /designs          index of everything available
"""
import json, os
from flask import Flask, Response, send_from_directory, send_file, redirect

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
META_FILE = os.path.join(SCRIPT_DIR, '_meta.json')
PREVIEW_DIR = os.path.join(SCRIPT_DIR, 'out', 'preview')

ALIASES = {
    'battery': 'pi_box_bat_core',          # battery in the box
    'batteryFloor': 'pi_box_bat_lids',
    'batteryPanel': 'pi_box_bat_panel',
    'v3': 'pi_box_bat_core_v3',             # v3 concept: cameras as cartridges, Pi +6 X
    'v3cam': 'pi_box_cam_cartridge',
    'collar': 'stick_collar',               # split cone collar for the monopod stick
    'v3floor': 'pi_box_bat_floor_v3',
}

app = Flask(__name__)

# Mtime-based cache per file — auto-picks up regenerated designs without a restart
_cache = {}


def _read_cached(filepath, binary=True):
    mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0
    hit = _cache.get(filepath)
    if not hit or hit[0] != mtime:
        with open(filepath, 'rb' if binary else 'r') as f:
            hit = (mtime, f.read())
        _cache[filepath] = hit
    return hit[1]


def _design_dir(name):
    name = ALIASES.get(name, name)
    path = os.path.join(PREVIEW_DIR, name)
    return path if os.path.isfile(os.path.join(path, 'meta.json')) else None


@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "viewer.html")


@app.route("/meta")
def meta():
    return Response(_read_cached(META_FILE, binary=False), mimetype="application/json")


@app.route("/slices")
def slices():
    return Response(_read_cached(SLICES_FILE), mimetype="application/octet-stream")


@app.route("/designs")
def designs():
    have = sorted(os.listdir(PREVIEW_DIR)) if os.path.isdir(PREVIEW_DIR) else []
    back = {v: k for k, v in ALIASES.items()}
    rows = ''.join(f'<li><a href="/{back.get(d, d)}">{back.get(d, d)}</a> <span style="color:#777">{d}</span></li>'
                   for d in have)
    return (f'<body style="background:#0c0e12;color:#ccc;font:14px monospace;padding:20px">'
            f'<h3>designs</h3><ul>{rows}</ul><p><a href="/">/</a> = last generated</p></body>')


@app.route("/stl/<path:filename>")
def stl(filename):
    return send_file(os.path.join(PROJECT_ROOT, filename),
                     mimetype="application/octet-stream")


@app.route("/<name>")
def design_page(name):
    if name.endswith('.js'):
        return send_from_directory(SCRIPT_DIR, name)
    if _design_dir(name):
        return send_from_directory(SCRIPT_DIR, "viewer.html")
    return f"No design '{name}' — see /designs (run generate.py {name} first)", 404


@app.route("/<name>/<item>")
def design_data(name, item):
    if item.endswith('.js'):                       # relative script paths from /<name>/
        return send_from_directory(SCRIPT_DIR, item)
    path = _design_dir(name)
    if not path:
        return "Not found", 404
    if item == 'meta':
        return Response(_read_cached(os.path.join(path, 'meta.json'), binary=False), mimetype="application/json")
    if item == 'slices':
        return Response(_read_cached(os.path.join(path, 'slices.bin')), mimetype="application/octet-stream")
    return "Not found", 404


if __name__ == "__main__":
    app.run(port=5555, debug=True)
