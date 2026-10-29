"""Send a .goo to the Saturn 4 Ultra 16K over the network (SDCP V3) and start it.

THIS IS THE ONLY WAY PRINTS GET TO THE PRINTER: its USB port is dead and we don't use the Elegoo software.

Usage:  python send_to_printer.py out/<design>.goo     upload + start
        python send_to_printer.py out/<design>.goo --no-start
        python send_to_printer.py --status              idle/printing, layer, file
        python send_to_printer.py --list                files on the printer

The printer is found by UDP broadcast ("M99999" → port 3000); last seen at 192.168.0.4 (home LAN, 192.168.0.x).
Upload: HTTP POST in 1 MB chunks to :3030/uploadFile/upload. Control: websocket :3030/websocket.
Start (Cmd 128) needs the printer-side path, /local/<name>.goo — a bare name is refused with Ack 1.
"""
import asyncio, json, sys, time, uuid, hashlib, os, socket
import requests, websockets

FALLBACK_IP = "192.168.0.4"


def discover(timeout=2.0):
    """(ip, mainboard_id) of the first printer that answers the SDCP broadcast."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    s.settimeout(timeout)
    for bc in ("255.255.255.255", "192.168.0.255"):
        s.sendto(b"M99999", (bc, 3000))
    try:
        d = json.loads(s.recvfrom(4096)[0])["Data"]
        return d["MainboardIP"], d["MainboardID"]
    except (socket.timeout, KeyError, ValueError):
        sys.exit(f"No SDCP printer answered (is it on? last seen at {FALLBACK_IP})")


async def _cmd(ip, mb, c, data=None, wait=4.0):
    """Send one SDCP command; return every message received within `wait` s."""
    out = []
    async with websockets.connect(f"ws://{ip}:3030/websocket") as ws:
        await ws.send(json.dumps({"Id": "", "Topic": f"sdcp/request/{mb}", "Data": {
            "Cmd": c, "Data": data or {}, "RequestID": uuid.uuid4().hex,
            "MainboardID": mb, "TimeStamp": int(time.time()), "From": 0}}))
        end = time.time() + wait
        while time.time() < end:
            try:
                out.append(json.loads(await asyncio.wait_for(ws.recv(), end - time.time())))
            except asyncio.TimeoutError:
                break
    return out


def cmd(ip, mb, c, data=None, wait=4.0):
    return asyncio.run(_cmd(ip, mb, c, data, wait))


def ack(msgs):
    return next((m["Data"]["Data"].get("Ack") for m in msgs if "Data" in m and "Data" in m["Data"]), None)


def status(ip, mb):
    for m in cmd(ip, mb, 0):
        if "Status" in m:
            st = m["Status"]; p = st["PrintInfo"]
            state = "idle" if st["CurrentStatus"] == [0] else f"busy {st['CurrentStatus']}"
            print(f"{state}  layer {p['CurrentLayer']}/{p['TotalLayer']}  file '{p['Filename']}'  "
                  f"print status {p['Status']}  error {p['ErrorNumber']}")
            return st


def upload(ip, path):
    b = open(path, "rb").read()
    md5, uid, chunk = hashlib.md5(b).hexdigest(), uuid.uuid4().hex, 1 << 20
    for off in range(0, len(b), chunk):
        r = requests.post(f"http://{ip}:3030/uploadFile/upload",
                          data={"S-File-MD5": md5, "Check": 1, "Offset": off, "Uuid": uid, "TotalSize": len(b)},
                          files={"File": (os.path.basename(path), b[off:off + chunk])}, timeout=60)
        if r.status_code != 200 or not r.json().get("success"):
            sys.exit(f"Upload failed at offset {off}: {r.status_code} {r.text[:200]}")
    print(f"Uploaded {os.path.basename(path)} ({len(b):,} bytes)")


def main(args):
    ip, mb = discover()
    print(f"Printer at {ip}")
    if "--status" in args:
        status(ip, mb); return
    if "--list" in args:
        for m in cmd(ip, mb, 258, {"Url": "/local"}):
            for f in m.get("Data", {}).get("Data", {}).get("FileList", []):
                print(f["name"])
        return
    files = [a for a in args if not a.startswith("--")]
    if not files:
        sys.exit(__doc__)
    path = files[0]
    st = status(ip, mb)
    if st and st["CurrentStatus"] != [0]:
        sys.exit("Printer is busy; not uploading")
    upload(ip, path)
    if "--no-start" in args:
        return
    a = ack(cmd(ip, mb, 128, {"Filename": f"/local/{os.path.basename(path)}", "StartLayer": 0}, 6))
    if a != 0:
        sys.exit(f"Start refused (Ack {a}); check the plate/vat on the printer screen")
    time.sleep(2)
    status(ip, mb)


if __name__ == "__main__":
    main(sys.argv[1:])
