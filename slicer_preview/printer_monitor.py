"""Web monitor for the Saturn 4 Ultra 16K (SDCP, via sdcp.py). Read-only: no print/pause/stop controls.

    python printer_monitor.py            -> http://localhost:5556   (also reachable from the LAN, e.g. a phone)

A background thread polls the printer's status every POLL_S seconds (one short WebSocket call each), so any number
of browser tabs cost the printer nothing extra. The page refreshes itself. Camera: a "snapshot" button grabs one
frame with ffmpeg (sdcp.py's rules: checks the viewer count first, ffmpeg exits by itself, never killed).
"""
import sys
import time
import threading
import subprocess
from pathlib import Path
from flask import Flask, jsonify, send_file, Response

sys.path.insert(0, str(Path(__file__).parent))
import sdcp  # noqa: E402

POLL_S = 5.0
PORT = 5556
SNAP = Path(__file__).parent / 'out' / 'monitor_snapshot.jpg'

app = Flask(__name__)
state = {'ok': False, 'error': 'connecting...', 'at': 0, 'status': None, 'printer': None}
_lock = threading.Lock()
_printer = None


def _poll():
    global _printer
    while True:
        try:
            if _printer is None:
                _printer = sdcp.Printer.find() if hasattr(sdcp.Printer, 'find') else None
            s = _printer.status()
            with _lock:
                state.update(ok=True, error='', at=time.time(), status=s, printer=_printer.ip)
        except SystemExit:
            with _lock:
                state.update(ok=False, error='no printer answered on the LAN', at=time.time())
            _printer = None
        except Exception as e:                      # network blips: keep the last status, show the error
            with _lock:
                state.update(ok=False, error=repr(e), at=time.time())
            _printer = None
        time.sleep(POLL_S)


@app.route('/api/status')
def api_status():
    with _lock:
        st = dict(state)
    s = st.get('status') or {}
    pi = s.get('PrintInfo', {}) if isinstance(s, dict) else {}
    cur, tot = pi.get('CurrentTicks', 0) / 1000, pi.get('TotalTicks', 0) / 1000
    return jsonify(ok=st['ok'], error=st['error'], age=time.time() - st['at'] if st['at'] else None,
                   ip=st['printer'], state=sdcp.PRINT_STATUS.get(pi.get('Status'), pi.get('Status')),
                   file=pi.get('Filename', ''), layer=pi.get('CurrentLayer'), layers=pi.get('TotalLayer'),
                   elapsed=cur, total=tot, remaining=max(tot - cur, 0) if tot else None,
                   err=pi.get('ErrorNumber'), tank=s.get('TempOfTank'), uv=s.get('TempOfUVLED'))


@app.route('/api/snapshot', methods=['POST'])
def api_snapshot():
    if _printer is None:
        return jsonify(ok=False, error='printer not connected'), 503
    try:
        url = _printer.video_url()                   # refuses if the camera's viewer slots are full
    except SystemExit as e:
        return jsonify(ok=False, error=str(e)), 503
    SNAP.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'fatal', '-use_wallclock_as_timestamps', '1',
                        '-buffer_size', '4000000', '-timeout', '10000000', '-i', url, '-frames:v', '1',
                        '-update', '1', '-y', str(SNAP)], timeout=60)
    return jsonify(ok=r.returncode == 0 and SNAP.exists(), t=time.time())


@app.route('/snapshot.jpg')
def snapshot():
    return send_file(SNAP, mimetype='image/jpeg', max_age=0) if SNAP.exists() else ('', 404)


PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Saturn monitor</title><style>
body{font:15px system-ui,sans-serif;background:#111;color:#ddd;margin:0;padding:18px;max-width:760px}
h1{font-size:18px;margin:0 0 12px} .k{color:#888;width:110px;display:inline-block}
.bar{background:#333;border-radius:4px;height:22px;margin:10px 0;overflow:hidden}
.fill{background:#2e86c1;height:100%;transition:width .5s} .row{margin:4px 0}
.st{font-size:22px;font-weight:600} .bad{color:#e74c3c} .done{color:#2ecc71}
button{background:#333;color:#ddd;border:1px solid #555;border-radius:4px;padding:6px 12px;cursor:pointer}
img{max-width:100%;margin-top:10px;border-radius:4px}
</style></head><body><h1>Saturn 4 Ultra 16K</h1>
<div class="st" id="st">...</div><div class="bar"><div class="fill" id="fill" style="width:0"></div></div>
<div class="row"><span class="k">file</span><span id="file"></span></div>
<div class="row"><span class="k">layer</span><span id="layer"></span></div>
<div class="row"><span class="k">elapsed</span><span id="el"></span></div>
<div class="row"><span class="k">remaining</span><span id="rem"></span></div>
<div class="row"><span class="k">finishes</span><span id="eta"></span></div>
<div class="row"><span class="k">temps</span><span id="temp"></span></div>
<div class="row"><span class="k">link</span><span id="link"></span></div>
<p><button onclick="snap()">camera snapshot</button> <span id="sn"></span></p><img id="img">
<script>
const f = s => s == null ? '-' : (s >= 3600 ? Math.floor(s/3600) + ' h ' : '') + Math.floor(s%3600/60) + ' min';
async function tick() {
  try {
    const d = await (await fetch('/api/status')).json();
    const st = document.getElementById('st');
    st.textContent = d.state ?? '?'; st.className = 'st ' + (d.err ? 'bad' : d.state === 'complete' ? 'done' : '');
    const p = d.layers ? 100 * d.layer / d.layers : 0;
    document.getElementById('fill').style.width = p.toFixed(1) + '%';
    document.getElementById('file').textContent = d.file || '-';
    document.getElementById('layer').textContent = d.layers ? `${d.layer} / ${d.layers}  (${p.toFixed(1)} %)` : '-';
    document.getElementById('el').textContent = f(d.elapsed);
    document.getElementById('rem').textContent = f(d.remaining);
    document.getElementById('eta').textContent = d.remaining && d.state !== 'complete'
      ? new Date(Date.now() + d.remaining * 1000).toLocaleTimeString([], {hour: '2-digit', minute: '2-digit'}) : '-';
    document.getElementById('temp').textContent = `tank ${d.tank ?? '-'} °C, UV LED ${d.uv != null ? d.uv.toFixed(1) : '-'} °C`
      + (d.err ? `   error ${d.err}` : '');
    document.getElementById('link').innerHTML = d.ok ? `${d.ip}, updated ${Math.round(d.age)} s ago`
      : `<span class="bad">${d.error}</span>`;
    document.title = d.layers ? `${p.toFixed(0)}% ${d.state} - Saturn` : 'Saturn monitor';
  } catch (e) { document.getElementById('link').innerHTML = '<span class="bad">monitor server unreachable</span>'; }
}
async function snap() {
  const s = document.getElementById('sn'); s.textContent = 'grabbing a frame...';
  const d = await (await fetch('/api/snapshot', {method: 'POST'})).json();
  s.textContent = d.ok ? '' : (d.error || 'snapshot failed');
  if (d.ok) document.getElementById('img').src = '/snapshot.jpg?t=' + d.t;
}
tick(); setInterval(tick, 5000);
</script></body></html>"""


@app.route('/')
def index():
    return Response(PAGE, mimetype='text/html')


if __name__ == '__main__':
    threading.Thread(target=_poll, daemon=True).start()
    app.run(host='0.0.0.0', port=PORT, debug=False, threaded=True)
