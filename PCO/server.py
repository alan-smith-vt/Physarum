"""
Flask server for streaming PCO point cloud data to a web viewer.

Endpoints:
    GET /                - serves the viewer HTML
    GET /metadata        - returns PCO header + node list as JSON
    GET /level/<int>     - returns all points at a given octree level as
                           JSON prefix table + raw binary body
"""

import sys
import json
import struct
from flask import Flask, Response, jsonify, send_file

sys.path.insert(0, '.')  # adjust to wherever your PCO package lives
from .pco_reader import PCOReader
from .pco_format import PCOFormat

app = Flask(__name__)

# --- Load PCO file at startup ---
PCO_FILE = 'chromosome.pco'  # change as needed
reader = PCOReader(PCO_FILE)
reader.load_metadata()
header = reader.header
index = reader.index


@app.route('/')
def serve_viewer():
    return send_file('viewer.html')


@app.route('/pco_viewer.js')
def serve_viewer_js():
    return send_file('pco_viewer.js')


@app.route('/metadata')
def metadata():
    """Return header info + full node list with point counts."""
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
    """
    Return all points at a given octree level.
    
    Response format:
        First 4 bytes: uint32 number of nodes
        Then for each node:
            24 bytes: node_id (ascii, null-padded)
            4 bytes:  uint32 point count
        Then: raw concatenated binary point data
    """
    node_ids = [nid for nid in index.keys() if len(nid) == level + 1]
    
    # Build prefix table and collect binary data
    prefix = struct.pack('I', len(node_ids))
    chunks = []
    
    for nid in sorted(node_ids):
        data = reader.read_node(nid)
        _, count = index[nid]
        prefix += nid.encode('ascii').ljust(24, b'\0')
        prefix += struct.pack('I', count)
        chunks.append(data)
    
    body = prefix + b''.join(chunks)
    return Response(body, mimetype='application/octet-stream')


if __name__ == '__main__':
    print(f"Serving {PCO_FILE}")
    print(f"  {header['total_points']:,} points, {header['num_nodes']:,} nodes, depth {header['max_depth']}")
    app.run(host='0.0.0.0', port=5000, debug=True)
