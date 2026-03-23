"""
Analysis server for Physarum filament extraction.
Run from the Physarum folder: python -m analysis.server
"""

import os
import json
import struct
import numpy as np
from flask import Flask, Response, jsonify, send_file, request

from PCO import PCOReader
from .analysis import AnalysisPipeline


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def json_response(data):
    return Response(json.dumps(data, cls=NumpyEncoder), mimetype='application/json')


app = Flask(__name__)

# --- Config ---
PCO_FILE = 'chromosome.pco'
reader = PCOReader(PCO_FILE)
reader.load_metadata()
header = reader.header
index = reader.index

pipeline = AnalysisPipeline()
pipeline.load_pco(PCO_FILE)

PCO_DIR = os.path.join(os.path.dirname(__file__), '..', 'PCO')
ANALYSIS_DIR = os.path.dirname(__file__)


# --- Static files ---

@app.route('/')
def serve_analysis():
    return send_file(os.path.join(ANALYSIS_DIR, 'analysis.html'))

@app.route('/pco_viewer.js')
def serve_viewer_js():
    return send_file(os.path.join(PCO_DIR, 'pco_viewer.js'))

@app.route('/analysis.js')
def serve_analysis_js():
    return send_file(os.path.join(ANALYSIS_DIR, 'analysis.js'))

@app.route('/config.json')
def serve_config():
    return send_file(os.path.join(ANALYSIS_DIR, 'config.json'))


# --- PCO data endpoints ---

@app.route('/metadata')
def metadata():
    nodes = {nid: count for nid, (offset, count) in index.items()}
    return jsonify({
        'max_depth': header['max_depth'],
        'root_min': list(header['root_min']),
        'root_size': header['root_size'],
        'total_points': header['total_points'],
        'schema': header['schema'],
        'bytes_per_point': header['bytes_per_point'],
        'num_nodes': header['num_nodes'],
        'nodes': nodes,
    })

@app.route('/level/<int:level>')
def get_level(level):
    """Get all nodes at a level. Supports ?offset=N&limit=M for chunking."""
    all_node_ids = sorted([nid for nid in index.keys() if len(nid) == level + 1])

    offset = request.args.get('offset', 0, type=int)
    limit = request.args.get('limit', len(all_node_ids), type=int)
    node_ids = all_node_ids[offset:offset + limit]

    prefix = struct.pack('I', len(node_ids))
    chunks = []
    for nid in node_ids:
        data = reader.read_node(nid)
        _, count = index[nid]
        prefix += nid.encode('ascii').ljust(24, b'\0')
        prefix += struct.pack('I', count)
        chunks.append(data)
    body = prefix + b''.join(chunks)

    # Include total count header so client knows when done
    resp = Response(body, mimetype='application/octet-stream')
    resp.headers['X-Total-Nodes'] = str(len(all_node_ids))
    return resp


# --- Pipeline API ---

@app.route('/api/bounds')
def api_bounds():
    return jsonify({'min': pipeline.bounds_min, 'max': pipeline.bounds_max})

@app.route('/api/cross_section', methods=['POST'])
def api_cross_section():
    params = request.json
    result = pipeline.cross_section(
        axis=params.get('axis', 2),
        value=params['value'],
        thickness=params.get('thickness', 0.5),
    )
    return json_response(result)

@app.route('/api/nearest', methods=['POST'])
def api_nearest():
    params = request.json
    result = pipeline.find_nearest(params['point'])
    return json_response(result)

@app.route('/api/node_bounds', methods=['POST'])
def api_node_bounds():
    """Get node bounding boxes at a Z value. Body: {z}"""
    params = request.json
    result = pipeline.node_bounds_at_z(params['z'])
    return json_response(result)

@app.route('/api/all_node_bounds')
def api_all_node_bounds():
    """Get all node bounding boxes for the grid overlay."""
    result = pipeline.all_node_bounds()
    return json_response(result)

@app.route('/api/trace', methods=['POST'])
def api_trace():
    params = request.json
    kwargs = {k: v for k, v in params.items() if k != 'seed'}
    result = pipeline.trace_centerline(params['seed'], **kwargs)
    return json_response(result)

@app.route('/api/trace_stations', methods=['POST'])
def api_trace_stations():
    """Start a station trace in background thread."""
    import threading

    params = request.json
    kwargs = {k: v for k, v in params.items() if k != 'seed'}
    progress_interval = kwargs.pop('progress_interval', 1)

    # Shared state for progress
    app.trace_progress = {'phase': 'starting', 'step': 0, 'done': False}
    app.trace_result = None

    def on_progress(snapshot):
        app.trace_progress = snapshot

    def run_trace():
        try:
            result = pipeline.trace_with_stations(
                params['seed'],
                progress_callback=on_progress,
                progress_interval=progress_interval,
                **kwargs)
            app.trace_result = result
        except Exception as e:
            app.trace_progress = {'phase': 'error', 'error': str(e), 'done': True}

    t = threading.Thread(target=run_trace, daemon=True)
    t.start()
    return jsonify({'status': 'started'})

@app.route('/api/trace_stations_progress')
def api_trace_stations_progress():
    """Poll for current trace progress."""
    progress = getattr(app, 'trace_progress', {'phase': 'idle', 'done': True})
    return json_response(progress)

@app.route('/api/trace_multi', methods=['POST'])
def api_trace_multi():
    """Start a multi-filament trace in background thread."""
    import threading

    params = request.json
    seeds = params['seeds']
    kwargs = {k: v for k, v in params.items() if k not in ('seeds',)}
    progress_interval = kwargs.pop('progress_interval', 1)

    app.trace_progress = {'phase': 'starting', 'step': 0, 'done': False}
    app.trace_result = None

    def on_progress(snapshot):
        app.trace_progress = snapshot

    def run_trace():
        try:
            result = pipeline.trace_multi_filament(
                seeds,
                progress_callback=on_progress,
                progress_interval=progress_interval,
                **kwargs)
            app.trace_result = result
        except Exception as e:
            import traceback
            traceback.print_exc()
            app.trace_progress = {'phase': 'error', 'error': str(e), 'done': True}

    t = threading.Thread(target=run_trace, daemon=True)
    t.start()
    return jsonify({'status': 'started'})

@app.route('/api/reset', methods=['POST'])
def api_reset():
    pipeline.reset()
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    print(f"Analysis server — {PCO_FILE}")
    print(f"  {header['total_points']:,} points, {header['num_nodes']:,} nodes, depth {header['max_depth']}")
    app.run(host='0.0.0.0', port=5001, debug=True)