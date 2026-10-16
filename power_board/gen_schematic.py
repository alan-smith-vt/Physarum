"""Generate the lidar rig power board schematic (KiCad) from a netlist table.

Every part is a project symbol (a box with named pins, or a 2-pin passive).
Pins connect through net labels placed on the pin ends, so the drawing is a
labelled parts sheet rather than a wired diagram; ERC and netlist are exact.

    python gen_schematic.py        -> power_board.kicad_sch, power_board.kicad_sym, project files
Design notes and value derivations: Notes vault, Reference/Lidar Rig Power Board.md
"""
import uuid
from pathlib import Path

HERE = Path(__file__).parent
NAME = "power_board"
LIB = "pb"
ROOT_UUID = "6a1f6c1e-6c7a-4b8e-9d55-0b1d2f3e4a01"

FP_R = "Resistor_SMD:R_0603_1608Metric"
FP_C = "Capacitor_SMD:C_0603_1608Metric"
FP_C1210 = "Capacitor_SMD:C_1210_3225Metric"
FP_C0805 = "Capacitor_SMD:C_0805_2012Metric"
FP_LED = "LED_SMD:LED_0603_1608Metric"


_UID_NS = uuid.UUID(ROOT_UUID)
_uid_count = 0


def uid():
    """Deterministic UUIDs so regenerating keeps the board's links to symbols (and quiet git diffs)."""
    global _uid_count
    _uid_count += 1
    return str(uuid.uuid5(_UID_NS, str(_uid_count)))


# ---------------------------------------------------------------- parts
# Two-pin passives: (ref, kind, value, footprint, net pin1, net pin2, note); for D/LED pin 1 is the cathode
PASSIVES = [
    # Input: USB-C, fuse, TVS, shunt
    ("F1", "F", "3A hold PTC", "Fuse:Fuse_1812_4532Metric", "VBUS_RAW", "VIN15", "bank cuts out ~3 A anyway"),
    ("D1", "D", "SMBJ16A", "Diode_SMD:D_SMB", "VIN15", "GND", "TVS"),
    ("R1", "R", "10m 1% 1W", "Resistor_SMD:R_2512_6332Metric", "VIN15", "V15", "INA226 shunt"),
    ("C1", "C", "1u 25V", FP_C0805, "VBUS_RAW", "GND", "U1 VHV"),
    ("R2", "R", "56k 1%", FP_R, "PD_CFG1", "GND", "CH224Q CFG1: 15 V"),
    # Buck
    ("C2", "C", "10u 35V", FP_C1210, "V15", "GND", "buck Cin"),
    ("C3", "C", "10u 35V", FP_C1210, "V15", "GND", "buck Cin"),
    ("C4", "C", "100n 50V", FP_C, "V15", "GND", "buck Cin HF"),
    ("C5", "C", "100n 16V", FP_C, "BUCK_BOOT", "BUCK_SW", "bootstrap"),
    ("L1", "L", "3.3u Isat>=10A", "Inductor_SMD:L_Vishay_IHLP-3232", "BUCK_SW", "V5", "IHLP-3232DZ-3R3 class"),
    ("R3", "R", "75k 1%", FP_R, "V5", "BUCK_FB", "FB top: 5.10 V"),
    ("R4", "R", "10k 1%", FP_R, "BUCK_FB", "GND", "FB bottom"),
    ("R5", "R", "20k", FP_R, "V5", "BUCK_FF", "feed-forward (optional)"),
    ("C6", "C", "100p", FP_C, "BUCK_FF", "BUCK_FB", "feed-forward (optional)"),
    ("C7", "C", "22u 16V", FP_C1210, "V5", "GND", "buck Cout"),
    ("C8", "C", "22u 16V", FP_C1210, "V5", "GND", "buck Cout"),
    ("C9", "C", "22u 16V", FP_C1210, "V5", "GND", "buck Cout"),
    ("R6", "R", "100k", FP_R, "V5", "BUCK_PG", "PG pull-up"),
    # Pi USB-C out: Rp 10k = 3 A advertisement at 5 V
    ("R7", "R", "10k", FP_R, "V5", "OUT_CC1", "Rp"),
    ("R8", "R", "10k", FP_R, "V5", "OUT_CC2", "Rp"),
    # Lidar eFuse
    ("C10", "C", "1u 25V", FP_C0805, "V15", "GND", "eFuse Cin"),
    ("R9", "R", "442k 1%", FP_R, "V15", "EF_UVLO", "UVLO 10.0 V / OVP 20.0 V"),
    ("R10", "R", "30.1k 1%", FP_R, "EF_UVLO", "EF_OVP", ""),
    ("R11", "R", "30.1k 1%", FP_R, "EF_OVP", "GND", ""),
    ("R12", "R", "9.09k 1%", FP_R, "EF_ILIM", "GND", "ILIM 2.0 A"),
    ("C11", "C", "100n", FP_C, "EF_DVDT", "GND", "soft-start"),
    ("R13", "R", "100k", FP_R, "LIDAR_EN", "GND", "lidar default OFF"),
    ("C12", "C", "1u 25V", FP_C0805, "VLIDAR", "GND", "eFuse Cout"),
    # INA226
    ("C13", "C", "100n", FP_C, "3V3_PI", "GND", "INA226 VS"),
    # Soft power with the latching switch: buck EN = switch OR Pi keep-alive (diode-OR).
    # Switch on: 15 V -> 100k/27k -> 3.19 V (safe for a Pi GPIO). Switch off: Pi sees SW_SENSE low,
    # shuts down, and drops KEEP_ALIVE at halt (gpio-poweroff), which turns the buck off.
    ("R17", "R", "100R 1206", "Resistor_SMD:R_1206_3216Metric", "V15", "SW_FEED", "limits a short in the switch lead; button LED (~10 mA) drops ~1 V"),
    ("R18", "R", "100k 1%", FP_R, "SW_ON", "SW_TAP", "switch divider top"),
    ("R19", "R", "27k 1%", FP_R, "SW_TAP", "GND", "switch divider bottom: 3.19 V"),
    ("D4", "D", "BAT54WS", "Diode_SMD:D_SOD-323", "BUCK_EN", "SW_TAP", "switch -> EN"),
    ("D5", "D", "BAT54WS", "Diode_SMD:D_SOD-323", "BUCK_EN", "KEEP_ALIVE", "Pi keep-alive -> EN"),
    ("R20", "R", "100k", FP_R, "BUCK_EN", "GND", "EN off when neither drives"),
    ("R21", "R", "10k", FP_R, "SW_TAP", "SW_SENSE", "series into Pi GPIO"),
    ("R22", "R", "100k", FP_R, "KEEP_ALIVE", "GND", "keep-alive low while Pi boots"),
    # Status LEDs
    ("R14", "R", "1k", FP_R, "V5", "LED5_A", ""),
    ("D2", "LED", "green", FP_LED, "GND", "LED5_A", "5 V on (K=1, A=2)"),
    ("R15", "R", "6.8k", FP_R, "VLIDAR", "LEDL_A", "~2 mA at 15 V"),
    ("D3", "LED", "blue", FP_LED, "GND", "LEDL_A", "lidar on (K=1, A=2)"),
    ("R16", "R", "0R", FP_R, "SW_ON", "BTN_LED_A", "Gebildet 12-24 V LED has its own resistor: lit while the switch is on"),
]

def usb_pins(vbus, cc1, cc2, dp, dm, side):
    """All 17 pads of a 16P USB-C receptacle (GCT USB4105), so the board matches the symbol pad for pad."""
    return ([(n, "VBUS", vbus, side) for n in ("A4", "A9", "B4", "B9")]
            + [(n, "GND", "GND", side) for n in ("A1", "A12", "B1", "B12", "SH")]
            + [("A5", "CC1", cc1, side), ("B5", "CC2", cc2, side)]
            + [(n, "D+", dp, side) for n in ("A6", "B6")]
            + [(n, "D-", dm, side) for n in ("A7", "B7")]
            + [("A8", "SBU1", None, side), ("B8", "SBU2", None, side)])


# ICs and connectors: (ref, symbol name, value, footprint, [(pin no, pin name, net or None for NC, side)])
# side: "L" or "R"
ICS = [
    ("J1", "USB_C_SINK", "USB-C receptacle (input from bank)", "Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal",
     usb_pins("VBUS_RAW", "IN_CC1", "IN_CC2", "IN_DP", "IN_DM", "R")),
    # CH224A: same function as the CH224Q in ESSOP-10 (hand-reworkable). CFG2/CFG3 have internal
    # 10k pull-ups to the chip's own rail, so they stay floating: tying them to the Pi's I2C would
    # back-power the Pi's 3V3 while the Pi is off. The INA226 reads the bus voltage instead.
    ("U1", "CH224A", "CH224A", "Package_SO:SSOP-10-1EP_3.9x4.9mm_P1mm_EP2.1x3.3mm", [
        ("1", "VHV", "VBUS_RAW", "L"), ("8", "VBUS", "VBUS_RAW", "L"),
        ("7", "CC1", "IN_CC1", "L"), ("6", "CC2", "IN_CC2", "L"),
        ("4", "DP", "IN_DP", "L"), ("5", "DM", "IN_DM", "L"),
        ("9", "CFG1", "PD_CFG1", "R"), ("2", "CFG2/SCL", None, "R"),
        ("3", "CFG3/SDA", None, "R"), ("10", "PG", "PD_PG_N", "R"),
        ("11", "GND", "GND", "R"),
    ]),
    ("U5", "INA226", "INA226 (addr 0x40)", "Package_SO:MSOP-10_3x3mm_P0.5mm", [
        ("10", "IN+", "VIN15", "L"), ("9", "IN-", "V15", "L"), ("8", "VBUS", "V15", "L"),
        ("6", "VS", "3V3_PI", "L"), ("7", "GND", "GND", "L"),
        ("5", "SCL", "SCL", "R"), ("4", "SDA", "SDA", "R"),
        ("1", "A1", "GND", "R"), ("2", "A0", "GND", "R"), ("3", "ALERT", None, "R"),
    ]),
    ("U3", "TPS56637", "TPS56637RPA", "power_board:RPA0010A", [
        ("8", "VIN", "V15", "L"), ("1", "EN", "BUCK_EN", "L"), ("10", "MODE", "GND", "L"),
        ("3", "AGND", "GND", "L"), ("9", "PGND", "GND", "L"),
        ("7", "BOOT", "BUCK_BOOT", "R"), ("6", "SW", "BUCK_SW", "R"), ("2", "FB", "BUCK_FB", "R"),
        ("4", "PG", "BUCK_PG", "R"), ("5", "NC", None, "R"),
    ]),
    ("J2", "USB_C_SOURCE", "USB-C receptacle (5 V to Pi)", "Connector_USB:USB_C_Receptacle_GCT_USB4105-xx-A_16P_TopMnt_Horizontal",
     usb_pins("V5", "OUT_CC1", "OUT_CC2", None, None, "L")),
    ("U4", "TPS26631", "TPS26631PWP", "Package_SO:HTSSOP-20-1EP_4.4x6.5mm_P0.65mm_EP3.4x6.5mm_Mask2.96x2.96mm_ThermalVias", [
        ("1", "IN", "V15", "L"), ("2", "IN", "V15", "L"), ("3", "IN", "V15", "L"),
        ("6", "IN_SYS", "V15", "L"), ("7", "UVLO", "EF_UVLO", "L"), ("8", "OVP", "EF_OVP", "L"),
        ("13", "~{SHDN}", "LIDAR_EN", "L"), ("9", "GND", "GND", "L"), ("21", "EP", "GND", "L"),
        ("4", "B_GATE", None, "L"), ("5", "DRV", None, "L"),
        ("18", "OUT", "VLIDAR", "R"), ("19", "OUT", "VLIDAR", "R"), ("20", "OUT", "VLIDAR", "R"),
        ("10", "dVdT", "EF_DVDT", "R"), ("11", "ILIM", "EF_ILIM", "R"), ("12", "MODE", None, "R"),
        ("15", "~{FLT}", "LIDAR_FLT_N", "R"), ("14", "IMON", None, "R"),
        ("16", "PGTH", "GND", "R"), ("17", "PGOOD", None, "R"),
    ]),
    ("J3", "LIDAR_OUT", "Lidar 15 V (pluggable 3.5 mm)", "Connector_Phoenix_MC:PhoenixContact_MCV_1,5_2-G-3.5_1x02_P3.50mm_Vertical", [
        ("1", "+15V", "VLIDAR", "L"), ("2", "GND", "GND", "L"),
    ]),
    ("J4", "BUTTON", "Latching 16 mm LED button (JST-XH 4)", "Connector_JST:JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical", [
        ("1", "SW_COM", "SW_FEED", "L"), ("2", "SW_NO", "SW_ON", "L"),
        ("3", "LED+", "BTN_LED_A", "L"), ("4", "LED-", "GND", "L"),
    ]),
    ("J5", "PI_HEADER", "To Pi GPIO (JST-PH 10)", "Connector_JST:JST_PH_B10B-PH-K_1x10_P2.00mm_Vertical", [
        ("1", "3V3", "3V3_PI", "L"), ("2", "GND", "GND", "L"),
        ("3", "SDA", "SDA", "L"), ("4", "SCL", "SCL", "L"),
        ("5", "LIDAR_EN", "LIDAR_EN", "L"), ("6", "SW_SENSE", "SW_SENSE", "L"),
        ("7", "KEEP_ALIVE", "KEEP_ALIVE", "L"), ("8", "~{LIDAR_FLT}", "LIDAR_FLT_N", "L"),
        ("9", "~{PD_PG}", "PD_PG_N", "L"), ("10", "BUCK_PG", "BUCK_PG", "L"),
    ]),
]

G = 2.54


def fnum(v):
    return f"{v:.3f}".rstrip("0").rstrip(".")


def effects(hide=False, justify=None):
    j = f" (justify {justify})" if justify else ""
    h = " hide" if hide else ""
    return f"(effects (font (size 1.27 1.27)){j}{h})"


def prop(name, val, x, y, hide=False, rot=0):
    return f'(property "{name}" "{val}" (at {fnum(x)} {fnum(y)} {rot}) {effects(hide)})'


# ---------------------------------------------------------------- symbols
def passive_symbol(kind):
    """Vertical 2-pin symbol, pin 1 top (y=+3.81), pin 2 bottom."""
    body = {
        "R": "(rectangle (start -1.016 -2.54) (end 1.016 2.54) (stroke (width 0.254) (type default)) (fill (type none)))",
        "C": "(polyline (pts (xy -2.032 -0.762) (xy 2.032 -0.762)) (stroke (width 0.508) (type default)) (fill (type none))) "
             "(polyline (pts (xy -2.032 0.762) (xy 2.032 0.762)) (stroke (width 0.508) (type default)) (fill (type none)))",
        "L": "(arc (start 0 -2.54) (mid 0.635 -1.905) (end 0 -1.27) (stroke (width 0) (type default)) (fill (type none))) "
             "(arc (start 0 -1.27) (mid 0.635 -0.635) (end 0 0) (stroke (width 0) (type default)) (fill (type none))) "
             "(arc (start 0 0) (mid 0.635 0.635) (end 0 1.27) (stroke (width 0) (type default)) (fill (type none))) "
             "(arc (start 0 1.27) (mid 0.635 1.905) (end 0 2.54) (stroke (width 0) (type default)) (fill (type none)))",
        "F": "(rectangle (start -0.762 -2.54) (end 0.762 2.54) (stroke (width 0.254) (type default)) (fill (type none))) "
             "(polyline (pts (xy 0 -2.54) (xy 0 2.54)) (stroke (width 0) (type default)) (fill (type none)))",
        # diode / LED: bar at the top, so pin 1 (top) is the cathode
        "D": "(polyline (pts (xy -1.27 1.27) (xy 1.27 1.27)) (stroke (width 0.254) (type default)) (fill (type none))) "
             "(polyline (pts (xy -1.27 -1.27) (xy 1.27 -1.27) (xy 0 1.27) (xy -1.27 -1.27)) (stroke (width 0.254) (type default)) (fill (type none)))",
    }
    body["LED"] = body["D"] + (" (polyline (pts (xy 1.524 0) (xy 2.794 1.27)) (stroke (width 0) (type default)) (fill (type none)))"
                               " (polyline (pts (xy 1.524 -1.016) (xy 2.794 0.254)) (stroke (width 0) (type default)) (fill (type none)))")
    return (f'(symbol "{LIB}:{kind}" (pin_numbers hide) (pin_names (offset 0) hide) (in_bom yes) (on_board yes) '
            f'{prop("Reference", kind[0], 2.54, 0)} {prop("Value", kind, 2.54, -2.54)} '
            f'{prop("Footprint", "", 0, 0, True)} {prop("Datasheet", "", 0, 0, True)} '
            f'(symbol "{kind}_0_1" {body[kind]}) '
            f'(symbol "{kind}_1_1" '
            f'(pin passive line (at 0 3.81 270) (length 1.27) (name "~" {effects()}) (number "1" {effects()})) '
            f'(pin passive line (at 0 -3.81 90) (length 1.27) (name "~" {effects()}) (number "2" {effects()}))))')


def box_geometry(pins):
    left = [p for p in pins if p[3] == "L"]
    right = [p for p in pins if p[3] == "R"]
    rows = max(len(left), len(right))
    w = 15.24 if max((len(p[1]) for p in pins), default=0) < 8 else 20.32
    h = (rows + 1) * G
    pos = {}
    for side, lst in (("L", left), ("R", right)):
        for i, p in enumerate(lst):
            y = h / 2 - G * (i + 1)
            x = -w / 2 - G if side == "L" else w / 2 + G
            pos[(p[0], p[1], side, i)] = (x, y)
    return w, h, left, right


def box_symbol(name, pins):
    w, h, left, right = box_geometry(pins)
    out = [f'(symbol "{LIB}:{name}" (pin_names (offset 1.016)) (in_bom yes) (on_board yes) '
           f'{prop("Reference", "U", 0, h / 2 + 1.27)} {prop("Value", name, 0, -h / 2 - 1.27)} '
           f'{prop("Footprint", "", 0, 0, True)} {prop("Datasheet", "", 0, 0, True)} '
           f'(symbol "{name}_0_1" (rectangle (start {fnum(-w/2)} {fnum(h/2)}) (end {fnum(w/2)} {fnum(-h/2)}) '
           f'(stroke (width 0.254) (type default)) (fill (type background)))) (symbol "{name}_1_1"']
    for side, lst in (("L", left), ("R", right)):
        for i, (num, pname, net, _) in enumerate(lst):
            y = h / 2 - G * (i + 1)
            x = -w / 2 - G if side == "L" else w / 2 + G
            ang = 0 if side == "L" else 180
            out.append(f'(pin passive line (at {fnum(x)} {fnum(y)} {ang}) (length 2.54) '
                       f'(name "{pname}" {effects()}) (number "{num}" {effects()}))')
    out.append("))")
    return " ".join(out)


# ---------------------------------------------------------------- placement
def label(net, x, y, rot=0):
    just = "left bottom" if rot == 0 else "right bottom"
    return f'(label "{net}" (at {fnum(x)} {fnum(y)} {rot}) {effects(justify=just)} (uuid "{uid()}"))'


def instance(lib_id, ref, value, fp, x, y, note=""):
    return (f'(symbol (lib_id "{lib_id}") (at {fnum(x)} {fnum(y)} 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) '
            f'(uuid "{uid()}") '
            f'{prop("Reference", ref, x + 3.81, y - 1.27)} {prop("Value", value, x + 3.81, y + 1.27)} '
            f'{prop("Footprint", fp, x, y, True)} {prop("Datasheet", "", x, y, True)} '
            f'{prop("Description", note, x, y, True)} '
            f'(instances (project "{NAME}" (path "/{ROOT_UUID}" (reference "{ref}") (unit 1)))))')


def build():
    lib_syms, items = [], []
    kinds = sorted({p[1] for p in PASSIVES})
    for k in kinds:
        lib_syms.append(passive_symbol(k))
    for ref, sym, value, fp, pins in ICS:
        lib_syms.append(box_symbol(sym, pins))

    # ICs in a row of columns, passives in a grid below
    x0, y0 = 30.48, 40.64
    col_x = x0
    row_y = y0
    max_h = 0
    for ref, sym, value, fp, pins in ICS:
        w, h, left, right = box_geometry(pins)
        span = w + 2 * 20.32
        if col_x + span > 400:
            col_x = x0
            row_y += max_h + 20.32
            max_h = 0
        cx = col_x + span / 2
        cy = row_y + h / 2
        cx = round(cx / G) * G
        cy = round(cy / G) * G
        items.append(instance(f"{LIB}:{sym}", ref, value, fp, cx, cy))
        for side, lst in (("L", left), ("R", right)):
            for i, (num, pname, net, _) in enumerate(lst):
                py = h / 2 - G * (i + 1)
                px = -w / 2 - G if side == "L" else w / 2 + G
                ax, ay = cx + px, cy - py
                if net is None:
                    items.append(f'(no_connect (at {fnum(ax)} {fnum(ay)}) (uuid "{uid()}"))')
                else:
                    items.append(label(net, ax, ay, 180 if side == "L" else 0))
        col_x += span
        max_h = max(max_h, h)

    py0 = row_y + max_h + 25.4
    per_row = 12
    for i, (ref, kind, value, fp, n1, n2, note) in enumerate(PASSIVES):
        cx = x0 + (i % per_row) * 30.48
        cy = py0 + (i // per_row) * 25.4
        items.append(instance(f"{LIB}:{kind}", ref, value, fp, cx, cy, note))
        items.append(label(n1, cx, cy - 3.81))
        items.append(label(n2, cx, cy + 3.81))

    sch = (f'(kicad_sch (version 20231120) (generator "eeschema") (generator_version "8.0") '
           f'(uuid "{ROOT_UUID}") (paper "A2") '
           f'(title_block (title "Lidar rig power board") (rev "0.1") '
           f'(comment 1 "Generated by gen_schematic.py - edit the table, not the sheet") '
           f'(comment 2 "Values: Notes Reference/Lidar Rig Power Board.md")) '
           f'(lib_symbols {" ".join(lib_syms)}) '
           + " ".join(items)
           + f' (sheet_instances (path "/" (page "1"))))')
    (HERE / f"{NAME}.kicad_sch").write_text(sch, encoding="utf-8")

    lib = "(kicad_symbol_lib (version 20231120) (generator \"gen_schematic\") " + " ".join(
        s.replace(f'(symbol "{LIB}:', '(symbol "', 1) for s in lib_syms) + ")"
    (HERE / f"{LIB}.kicad_sym").write_text(lib, encoding="utf-8")
    (HERE / "sym-lib-table").write_text(
        f'(sym_lib_table (version 7) (lib (name "{LIB}")(type "KiCad")(uri "${{KIPRJMOD}}/{LIB}.kicad_sym")(options "")(descr "power board")))\n',
        encoding="utf-8")
    pro = HERE / f"{NAME}.kicad_pro"
    if not pro.exists():
        pro.write_text('{"meta": {"filename": "%s.kicad_pro", "version": 1}}\n' % NAME, encoding="utf-8")


if __name__ == "__main__":
    build()
