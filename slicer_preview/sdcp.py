"""Talk to the Saturn 4 Ultra 16K over the network (ELEGOO SDCP V3) — no ChituBox.

  discovery   UDP broadcast "M99999" → port 3000, printer replies with JSON
  control     WebSocket ws://<ip>:3030/websocket, JSON requests by Cmd number
  upload      HTTP POST http://<ip>:3030/uploadFile/upload, 1 MB multipart chunks

Usage:
  python sdcp.py discover
  python sdcp.py status
  python sdcp.py files
  python sdcp.py history [N]
  python sdcp.py upload out/part.goo          (stores on printer, doesn't print; per-chunk timing)
  python sdcp.py upload f.goo chunk=4194304 check=0   (upload speed tests)
  python sdcp.py print part.goo --yes         (starts a print!)
  python sdcp.py pause | resume | stop
  python sdcp.py watch                        (live status log, Ctrl+C to quit)
  python sdcp.py camera                       (live view in ffplay; quit with q / close the window)
  python sdcp.py snapshot [out.jpg]           (one camera frame)
  python sdcp.py timelapse [N] [out.mp4]      (timelapse of the Nth-latest print, default 1 = latest)

Verified 2026-10-11: discover, status, files, history, watch, upload, print
(exposure_test.goo ran to completion). NOT yet verified: pause/resume/stop.

Camera (found 2026-10-14):
  - Cmd 386 {"Enable": 1} switches the stream on and returns rtsp://<ip>:554/video (H.264, 1280×720, 30 fps).
  - The SDP says "H264/9000": the RTP clock is misreported (H.264 is always 90000), so players that trust
    the timestamps show ~3 fps. Use -use_wallclock_as_timestamps 1 (time frames by arrival).
  - UDP only: the server answers a TCP-interleaved SETUP with a UDP transport, which ffmpeg rejects
    ("Nonmatching transport"). Expect some packet loss over Wi-Fi; -buffer_size 4000000 helps.
  - At most 2 viewers (attributes MaximumVideoStreamAllowed). Sessions that are killed without an RTSP
    TEARDOWN stay counted (NumberOfVideoStreamConnected) and did NOT time out after minutes; once over the
    limit, Cmd 386 returns Ack 1 for both on and off. Only a printer power-cycle cleared it. So: let
    ffmpeg/ffplay exit on their own (frame limit, q, window close, -timeout), never kill them, and check
    the count before connecting (camera/snapshot below refuse when it's full).
  - Timelapses: history detail has TimeLapseVideoUrl (a path); download http://<ip>:3030<path>.
"""
import sys, json, time, uuid, socket, hashlib, asyncio, datetime
from pathlib import Path

PORT_UDP = 3000
PORT_WS = 3030
CHUNK = 1024 * 1024

CMD_STATUS, CMD_ATTRIBUTES = 0, 1
CMD_PRINT, CMD_PAUSE, CMD_STOP, CMD_RESUME = 128, 129, 130, 131
CMD_FILES, CMD_HISTORY, CMD_HISTORY_DETAIL = 258, 320, 321
CMD_VIDEO = 386


def discover(timeout=3.0):
    """Broadcast and return the list of printers that reply."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(timeout)
    s.sendto(b'M99999', ('255.255.255.255', PORT_UDP))
    found = []
    try:
        while True:
            data, _ = s.recvfrom(8192)
            found.append(json.loads(data)['Data'])
    except socket.timeout:
        pass
    return found


class Printer:
    def __init__(self, ip, mainboard_id):
        self.ip, self.mid = ip, mainboard_id

    @classmethod
    def find(cls):
        found = discover()
        if not found:
            sys.exit("No SDCP printer answered on the LAN.")
        return cls(found[0]['MainboardIP'], found[0]['MainboardID'])

    def _req(self, cmd, data=None):
        return json.dumps({"Id": "", "Topic": f"sdcp/request/{self.mid}", "Data": {
            "Cmd": cmd, "Data": data or {}, "RequestID": uuid.uuid4().hex,
            "MainboardID": self.mid, "TimeStamp": int(time.time()), "From": 0}})

    async def _call(self, cmd, data=None, want_topic='response'):
        import websockets
        async with websockets.connect(f'ws://{self.ip}:{PORT_WS}/websocket', open_timeout=5) as ws:
            await ws.send(self._req(cmd, data))
            while True:
                m = json.loads(await asyncio.wait_for(ws.recv(), 10))
                topic = m.get('Topic', '')
                if want_topic == 'response' and topic.startswith('sdcp/response') and m['Data']['Cmd'] == cmd:
                    return m['Data']['Data']
                if want_topic != 'response' and topic.startswith(f'sdcp/{want_topic}'):
                    return m

    def call(self, cmd, data=None, want_topic='response'):
        return asyncio.run(self._call(cmd, data, want_topic))

    def status(self):
        return self.call(CMD_STATUS, want_topic='status')['Status']

    def attributes(self):
        return self.call(CMD_ATTRIBUTES, want_topic='attributes')['Attributes']

    def files(self, path='/local'):
        return [f['name'] for f in self.call(CMD_FILES, {"Url": path})['FileList']]

    def history(self):
        ids = self.call(CMD_HISTORY).get('HistoryData', [])
        rows = []
        for i in range(0, len(ids), 10):
            rows += self.call(CMD_HISTORY_DETAIL, {"Id": ids[i:i + 10]}).get('HistoryDetailList', [])
        return sorted(rows, key=lambda t: t.get('BeginTime', 0))

    def upload(self, path, chunk=CHUNK, check=1):
        """Chunked upload to the printer's local storage. Does not start a print.
        Prints per-chunk time and rate: uploads run ~0.28 MB/s against ~3.3 MB/s for downloads over the same
        link (2026-10-14), so the time goes to the printer's per-chunk handling. chunk / check are for testing that."""
        import requests
        path = Path(path)
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        upload_id = uuid.uuid4().hex
        t_start = time.time()
        for offset in range(0, len(data), chunk):
            t0 = time.time()
            piece = data[offset:offset + chunk]
            r = requests.post(f'http://{self.ip}:{PORT_WS}/uploadFile/upload', timeout=60,
                              data={'S-File-MD5': md5, 'Check': str(check), 'Offset': str(offset),
                                    'Uuid': upload_id, 'TotalSize': str(len(data))},
                              files={'File': (path.name, piece, 'application/octet-stream')})
            r.raise_for_status()
            body = r.json()
            if not body.get('success', body.get('code') in (0, '000000')):
                raise RuntimeError(f"Upload rejected at offset {offset}: {body}")
            dt, el = time.time() - t0, time.time() - t_start
            done = min(offset + chunk, len(data))
            print(f"  {done:,}/{len(data):,} bytes  chunk {dt:5.2f} s ({len(piece) / dt / 1e6:4.2f} MB/s)  "
                  f"total {el:6.1f} s ({done / el / 1e6:4.2f} MB/s)", flush=True)
        return path.name

    def video_url(self):
        """Switch the camera stream on and return its RTSP URL, refusing if all viewer slots are taken
        (stale sessions would lock the camera until a power-cycle; see the module notes)."""
        a = self.attributes()
        n, cap = a.get('NumberOfVideoStreamConnected', 0), a.get('MaximumVideoStreamAllowed', 2)
        if n >= cap:
            sys.exit(f"Camera full: {n} viewer(s) connected, limit {cap}. Close other viewers; "
                     f"stale sessions only clear with a printer power-cycle.")
        r = self.call(CMD_VIDEO, {"Enable": 1})
        if r.get('Ack') != 0 or not r.get('VideoUrl'):
            sys.exit(f"Camera refused: {r}")
        return r['VideoUrl']

    def timelapse(self, nth=1, out=None):
        tasks = [t for t in self.history() if t.get('TimeLapseVideoUrl')]
        if len(tasks) < nth:
            sys.exit(f"Only {len(tasks)} print(s) with a timelapse.")
        t = tasks[-nth]
        name = Path(t.get('TaskName', 'print')).stem
        out = Path(out or Path(__file__).parent / 'out' / 'timelapse' /
                   f"{name}_{datetime.datetime.fromtimestamp(t['BeginTime']):%Y-%m-%d_%H%M}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        import requests
        with requests.get(f"http://{self.ip}:{PORT_WS}{t['TimeLapseVideoUrl']}", stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out, 'wb') as f:
                for block in r.iter_content(1 << 20):
                    f.write(block)
        return out

    def start_print(self, filename, start_layer=0):
        if not filename.startswith('/'):
            filename = f'/local/{filename}'
        return self.call(CMD_PRINT, {"Filename": filename, "StartLayer": start_layer})


# PrintInfo.Status per the SDCP V3 spec (not all confirmed on this printer)
PRINT_STATUS = {0: 'idle', 1: 'homing', 2: 'dropping', 3: 'exposing', 4: 'lifting',
                5: 'pausing', 6: 'paused', 7: 'stopping', 8: 'stopped', 9: 'complete',
                10: 'file checking', 12: 'recovery', 13: 'printing', 16: 'preparing'}


async def _watch(p, interval=3.0):
    """Print every status/notice/error message, collapsing unchanged status."""
    import websockets
    last = None
    while True:
        try:
            async with websockets.connect(f'ws://{p.ip}:{PORT_WS}/websocket', open_timeout=5) as ws:
                print(f"[{time.strftime('%H:%M:%S')}] connected to {p.ip}", flush=True)
                async def poll():
                    while True:
                        await ws.send(p._req(CMD_STATUS))
                        await asyncio.sleep(interval)
                poller = asyncio.create_task(poll())
                try:
                    async for raw in ws:
                        m = json.loads(raw)
                        topic = m.get('Topic', '').split('/')[1] if '/' in m.get('Topic', '') else '?'
                        stamp = time.strftime('%H:%M:%S')
                        if topic == 'status':
                            s = m['Status']
                            pi = s.get('PrintInfo', {})
                            line = (f"{PRINT_STATUS.get(pi.get('Status'), pi.get('Status'))}  "
                                    f"layer {pi.get('CurrentLayer')}/{pi.get('TotalLayer')}  "
                                    f"{pi.get('CurrentTicks', 0) / 1000:.0f}/{pi.get('TotalTicks', 0) / 1000:.0f} s  "
                                    f"err={pi.get('ErrorNumber')}  tank {s.get('TempOfTank')}°C  "
                                    f"UV {s.get('TempOfUVLED', 0):.1f}°C  {pi.get('Filename', '')}")
                            if line != last:
                                print(f"[{stamp}] {line}", flush=True)
                                last = line
                        elif topic == 'response' and m['Data'].get('Cmd') == CMD_STATUS:
                            continue
                        else:
                            print(f"[{stamp}] {topic}: {json.dumps(m.get('Data', m))}", flush=True)
                finally:
                    poller.cancel()
        except (OSError, asyncio.TimeoutError, Exception) as e:
            print(f"[{time.strftime('%H:%M:%S')}] connection lost ({e!r}), retrying in 5 s", flush=True)
            await asyncio.sleep(5)


def _fmt_time(v):
    return datetime.datetime.fromtimestamp(v).strftime('%Y-%m-%d %H:%M') if v else '-'


def main(argv):
    if not argv:
        sys.exit(__doc__)
    cmd, args = argv[0], argv[1:]

    if cmd == 'discover':
        for p in discover():
            print(json.dumps(p, indent=1))
        return

    p = Printer.find()
    if cmd == 'status':
        print(json.dumps(p.status(), indent=1))
    elif cmd == 'files':
        print('\n'.join(p.files()))
    elif cmd == 'history':
        n = int(args[0]) if args else 10
        for t in p.history()[-n:]:
            print(f"{_fmt_time(t.get('BeginTime'))} → {_fmt_time(t.get('EndTime'))}  "
                  f"status={t.get('TaskStatus')}  {t.get('TaskName')}")
    elif cmd == 'upload':
        opt = dict(a.split('=', 1) for a in args[1:] if '=' in a)   # testing: chunk=4194304 check=0
        print(f"Uploaded {p.upload(args[0], int(opt.get('chunk', CHUNK)), int(opt.get('check', 1)))} to {p.ip}")
    elif cmd == 'print':
        if '--yes' not in args:
            sys.exit("Starting a print moves the plate into the vat. Re-run with --yes.")
        print(p.start_print(args[0]))
    elif cmd == 'watch':
        asyncio.run(_watch(p))
    elif cmd in ('camera', 'snapshot'):
        import subprocess
        url = p.video_url()
        flags = ['-hide_banner', '-loglevel', 'error', '-use_wallclock_as_timestamps', '1',
                 '-buffer_size', '4000000', '-timeout', '10000000']     # exits by itself if the stream stalls
        if cmd == 'camera':
            print(f"{url}  (quit with q or by closing the window, not Ctrl+C, so the session is released)")
            subprocess.run(['ffplay', *flags, '-window_title', 'Saturn camera', url])
        else:
            out = Path(args[0]) if args else Path(__file__).parent / 'out' / 'timelapse' /                 f"snapshot_{time.strftime('%Y-%m-%d_%H%M%S')}.jpg"
            out.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['ffmpeg', *flags, '-i', url, '-frames:v', '1', '-update', '1', '-y', str(out)])
            print(out)
    elif cmd == 'timelapse':
        nth = int(args[0]) if args else 1
        print(p.timelapse(nth, args[1] if len(args) > 1 else None))
    elif cmd in ('pause', 'resume', 'stop'):
        print(p.call({'pause': CMD_PAUSE, 'resume': CMD_RESUME, 'stop': CMD_STOP}[cmd]))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
