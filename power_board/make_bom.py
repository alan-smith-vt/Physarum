"""JLCPCB BOM + CPL from the KiCad position file (run export_fab.ps1 first).

LCSC codes: ICs, connectors, power parts and the E96 resistors were checked on their LCSC/JLC part pages
(2026-10-16). Codes marked (*) are common JLC Basic parts from memory: the JLC BOM tool confirms or
rejects them at upload. J3's mating plug (KF2EDGK-3.5-2P, C440847) is ordered separately.
"""
import csv
from collections import defaultdict
from pathlib import Path

FAB = Path(__file__).with_name("fab")

# value prefix (as in the schematic) -> (LCSC code, part description)
LCSC = {
    "CH224A": ("C42459160", "WCH CH224A USB-PD sink, ESSOP-10"),
    "TPS56637RPA": ("C841386", "TI TPS56637RPAR 6 A buck"),
    "TPS26631PWP": ("C2866319", "TI TPS26631PWPR eFuse"),
    "INA226": ("C49851", "TI INA226AIDGSR"),
    "3.3u MDA7050": ("C2847540", "KOHERelec MDA7050-3R3M 3.3 uH 9.5 A"),
    "SMBJ16A": ("C10211", "SMBJ16A TVS"),
    "3A hold PTC": ("C22446884", "LUTE 2920L300/30GR PTC 3 A hold"),
    "BAT54WS": ("C2991936", "BAT54WS Schottky SOD-323"),
    "USB-C receptacle": ("C3020560", "GCT USB4105-GF-A"),
    "To Pi GPIO": ("C158038", "JST B10B-PH-K-S"),
    "Latching 16 mm": ("C144395", "JST B4B-XH-A"),
    "Lidar 15 V": ("C441321", "Kefa KF2EDGV-3.5-2P (plug C440847 separate)"),
    "10m 1%": ("C323772", "FORT CUM2512 10 mOhm 1% 2 W"),
    "green": ("C72043*", "LED green 0603"),
    "blue": ("C72041*", "LED blue 0603"),
    "22u": ("C52306", "Samsung CL32A226KAJNNNE 22 uF 25 V X5R 1210"),
    "10u": ("C337485", "Taiyo UMK325AB7106KM-P 10 uF 50 V X7R 1210"),
    "1u 25V": ("C28323*", "1 uF 25 V 0805"),
    "100n": ("C14663*", "100 nF 50 V X7R 0603"),
    "100p": ("C14858*", "100 pF 0603"),
    "442k": ("C23175", "442k 1% 0603"),
    "30.1k": ("C23000", "30.1k 1% 0603"),
    "9.09k": ("C23125", "9.09k 1% 0603"),
    "100R": ("C17901", "100 Ohm 1% 1206"),
    "75k": ("C23242*", "75k 1% 0603"),
    "56k": ("C23206*", "56k 1% 0603"),
    "27k": ("C22967*", "27k 1% 0603"),
    "20k": ("C4184*", "20k 1% 0603"),
    "100k": ("C25803*", "100k 1% 0603"),
    "10k": ("C25804*", "10k 1% 0603"),
    "6.8k": ("C23212*", "6.8k 1% 0603"),
    "1k": ("C21190*", "1k 1% 0603"),
    "0R": ("C21189*", "0 Ohm 0603"),
}


def lookup(val):
    for key in sorted(LCSC, key=len, reverse=True):     # longest prefix first: "100k" before "10k"
        if val.startswith(key):
            return LCSC[key]
    raise SystemExit(f"no LCSC code for value {val!r}")


rows = list(csv.DictReader((FAB / "power_board_pos_raw.csv").open(encoding="utf-8")))
groups = defaultdict(list)
for r in rows:
    if r["Ref"].startswith("H"):
        continue
    code, desc = lookup(r["Val"])
    groups[(code, desc, r["Package"])].append(r["Ref"])

with (FAB / "power_board_bom_jlc.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Comment", "Designator", "Footprint", "LCSC Part #"])
    for (code, desc, pkg), refs in sorted(groups.items(), key=lambda g: g[1][0]):
        w.writerow([desc, ",".join(sorted(refs, key=lambda s: (s[0], int(s[1:])))), pkg, code.rstrip("*")])

with (FAB / "power_board_cpl_jlc.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["Designator", "Mid X", "Mid Y", "Layer", "Rotation"])
    for r in rows:
        if r["Ref"].startswith("H"):
            continue
        w.writerow([r["Ref"], f'{float(r["PosX"]):.3f}mm', f'{float(r["PosY"]):.3f}mm', "Top", f'{float(r["Rot"]):.0f}'])

unverified = sorted({c for (c, _, _) in groups if c.endswith("*")})
print(f"BOM lines {len(groups)}, parts {sum(len(v) for v in groups.values())}; codes to confirm in JLC's tool: {unverified}")
