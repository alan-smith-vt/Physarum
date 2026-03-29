"""Serve pre-generated slice data to the Three.js viewer."""
import json, os
from flask import Flask, Response, send_from_directory, send_file

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SLICES_FILE = os.path.join(SCRIPT_DIR, '_slices.bin')
META_FILE = os.path.join(SCRIPT_DIR, '_meta.json')

app = Flask(__name__)

# Mtime-based cache — auto-picks up new files without restart
_cache = {'slices_mtime': 0, 'slices_data': b'',
          'meta_mtime': 0, 'meta_data': ''}


def _read_cached(filepath, key, binary=True):
    mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0
    if mtime != _cache[f'{key}_mtime']:
        with open(filepath, 'rb' if binary else 'r') as f:
            _cache[f'{key}_data'] = f.read()
        _cache[f'{key}_mtime'] = mtime
    return _cache[f'{key}_data']


@app.route("/")
def index():
    return send_from_directory(SCRIPT_DIR, "viewer.html")


@app.route("/<path:filename>")
def static_js(filename):
    if filename.endswith('.js'):
        return send_from_directory(SCRIPT_DIR, filename)
    return "Not found", 404


@app.route("/meta")
def meta():
    data = _read_cached(META_FILE, 'meta', binary=False)
    return Response(data, mimetype="application/json")


@app.route("/slices")
def slices():
    data = _read_cached(SLICES_FILE, 'slices', binary=True)
    return Response(data, mimetype="application/octet-stream")


@app.route("/stl/<path:filename>")
def stl(filename):
    return send_file(os.path.join(PROJECT_ROOT, filename),
                     mimetype="application/octet-stream")


if __name__ == "__main__":
    app.run(port=5555, debug=True)
