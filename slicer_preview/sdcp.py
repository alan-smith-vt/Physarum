"""Talk to the Saturn 4 Ultra 16K over the network (ELEGOO SDCP V3) — no ChituBox.

  discovery   UDP broadcast "M99999" → port 3000, printer replies with JSON
  control     WebSocket ws://<ip>:3030/websocket, JSON requests by Cmd number
  upload      HTTP POST http://<ip>:3030/uploadFile/upload, 1 MB multipart chunks

Usage:
  python sdcp.py discover
  python sdcp.py status
  python sdcp.py files
  python sdcp.py history [N]
  python sdcp.py upload out/part.goo          (stores on printer, doesn't print)
  python sdcp.py print part.goo --yes         (starts a print!)
  python sdcp.py pause | resume | stop
  python sdcp.py watch                        (live status log, Ctrl+C to quit)

Verified 2026-10-11: discover, status, files, history, watch, upload, print
(exposure_test.goo ran to completion). NOT yet verified: pause/resume/stop.
"""
import sys, json, time, uuid, socket, hashlib, asyncio, datetime
from pathlib import Path

PORT_UDP = 3000
PORT_WS = 3030
CHUNK = 1024 * 1024

CMD_STATUS, CMD_ATTRIBUTES = 0, 1
CMD_PRINT, CMD_PAUSE, CMD_STOP, CMD_RESUME = 128, 129, 130, 131
CMD_FILES, CMD_HISTORY, CMD_HISTORY_DETAIL = 258, 320, 321


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

    def upload(self, path):
        """Chunked upload to the printer's local storage. Does not start a print."""
        import requests
        path = Path(path)
        data = path.read_bytes()
        md5 = hashlib.md5(data).hexdigest()
        upload_id = uuid.uuid4().hex
        for offset in range(0, len(data), CHUNK):
            chunk = data[offset:offset + CHUNK]
            r = requests.post(f'http://{self.ip}:{PORT_WS}/uploadFile/upload', timeout=60,
                              data={'S-File-MD5': md5, 'Check': '1', 'Offset': str(offset),
                                    'Uuid': upload_id, 'TotalSize': str(len(data))},
                              files={'File': (path.name, chunk, 'application/octet-stream')})
            r.raise_for_status()
            body = r.json()
            if not body.get('success', body.get('code') in (0, '000000')):
                raise RuntimeError(f"Upload rejected at offset {offset}: {body}")
            print(f"  {min(offset + CHUNK, len(data)):,}/{len(data):,} bytes", flush=True)
        return path.name

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
        print(f"Uploaded {p.upload(args[0])} to {p.ip}")
    elif cmd == 'print':
        if '--yes' not in args:
            sys.exit("Starting a print moves the plate into the vat. Re-run with --yes.")
        print(p.start_print(args[0]))
    elif cmd == 'watch':
        asyncio.run(_watch(p))
    elif cmd in ('pause', 'resume', 'stop'):
        print(p.call({'pause': CMD_PAUSE, 'resume': CMD_RESUME, 'stop': CMD_STOP}[cmd]))
    else:
        sys.exit(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:])
