"""CHOSEN VERSION (2026-10-13). Battery-in-box variant of designs/archive/pole/pi_box_rig.py (copy): the 15 V bank stands in a +Y pod
beyond the buck bay instead of at the bottom of the pole, so no power lead runs down the stick.

Differences from the archived pole version are marked "BATTERY". Everything else is the same design.

Shared geometry for the lidar rig's Pi box (core, floor, panels) + viewer overlays.

Box frame (mm): origin at the underside centre of the aluminium base plate,
X = across the camera walls, Yr = Pi (-) to buck bay (+), Zr = up. The box hangs
below the plate, so we mostly use depth d = -Zr. The box is turned 90° on the
square 80 × 80 standoff pattern so its buck bay faces the lidar's M12 connector:
the lidar's connector axis is box +Yr (LIDAR_YAW).

Print frame (Physarum: X/Z plate plane, Y up): the core prints flange-down on
the build plate, so  print (x, y, z) = (X, d, Yr).  That's a 180° turn about X
of the rig, not a mirror.

Concept and reasoning: D:\\Notes\\Reference\\lidar-photo-rig-cage-handoff.md
(Pi box section, 2026-10-13). Sizes marked [MEASURE] are guesses until the
parts are in hand.
"""
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]          # Physarum/
MODELS = ROOT / 'models'
OVERLAY_DIR = ROOT / 'slicer_preview' / 'out' / 'overlays_battery'

RESIN_HOLE_COMP = 0.35        # holes/slots print ~0.35 undersize (cage v1 fit strip)
HORIZ_HOLE_EXTRA = 0.2        # horizontal holes also sag ~0.15-0.2 in print height (cage v2 fit strip)

# ── Box envelope ──
X_OUT, X_IN = 50.0, 46.0      # camera wall outer / inner face (|X|)
# DESIGN CONSTRAINT (2026-10-13): the box is symmetric about the aluminium plate (centred on the
# standoff square) in Yr, so Y_MAX = -Y_MIN. Both ends overhang the plate edge (±45) by 17 mm.
Y_MIN = -62.0                 # box ends (outside the panel lips)
Y_MAX = -Y_MIN
DEPTH, FLANGE = 180.0, 10.0   # core bottom edge / flange thickness  BATTERY: the 157 mm bank stands on the floor
RIM_D = 14.0                  # panel-stop rims at both ends (deeper than the bore bosses: room for the lidar cable notch)
CHAMFER = 1.0                 # flange plate-face edge: elephant foot + scraper entry

# ── Well (coupler + monopod platform pass through) ──
WELL_R, TUBE_T = 26.0, 2.0

# ── Box mount: M4×18 + 6 mm Al spacer (Ø8) per standoff, stepped bore ──
POST = 40.0
SPACER_D, SPACER_SEAT = 8.0, 5.85     # print 5.85 deep, sand until the spacer sits flush
SPACER_FIT = 0.25                     # extra on top of RESIN_HOLE_COMP: Ø8.35 as cut was just too tight (core print 2026-10-14)
HEAD_D = 10.2                         # M4 truss head ~9.5 + clearance [MEASURE]

# ── Lidar M3 button heads stand proud under the plate: (±24, ±18) in the lidar's frame
#    (24 along its connector axis), so (±18, ±24) in the box frame ──
LIDAR_YAW = 90.0                      # lidar frame -> box frame, degrees about Zr (connector -> +Yr)
LIDAR_M3 = (18.0, 24.0)
M3_HEAD_D, M3_HEAD_RECESS = 7.0, 2.5

# ── Panels slide up in grooves in the camera-wall ends ──
# Side panels: 1/8" (3.175) sheet: printed blanks for the prototype, cut polycarbonate later.
# They sit in true slots in the camera-wall ends: a 1.5 mm lip outside the panel stops it bowing
# out between the flange (top) and the floor (bottom); two small bumps in each slot take up rattle.
PANEL_T = 3.175                   # panel thickness
PANEL_LIP = 1.5                   # wall material outside the panel at each box end
GROOVE_W = PANEL_T + 0.4              # cast/extruded sheet ±0.2; + RESIN_HOLE_COMP applied where cut
GROOVE_D = 2.0                        # into the wall from its inner face

# Cameras in the -Y corners (2026-10-14): board's -Yr edge on the panel-slot solid band, so the lens axis sits
# over the corner M4 (a Ø HEAD_D bore through the camera block reaches it; screw first, then the camera).
CAM_YR = Y_MIN + PANEL_LIP + GROOVE_W + RESIN_HOLE_COMP + 4.0 + 12.5     # ≈ -40.1
CAM_RIB_Y = CAM_YR + 12.5 + 1.5                                          # rib flanking the board's +Yr edge

# ── Webs, wall feet, dowels, floor bosses ──
WEB_T = 3.0
WEB_SLOTS = [(30.0, 41.0, 24.0, 34.0), (28.5, 45.5, 76.0, 98.0)]   # |X| range, depth range: cable pass-throughs
# (low slot: the lidar's RJ45 + boot ~13.5 x 16 and its Ø~16 ferrite, estimated from a product photo [MEASURE])
FOOT_X0, FOOT_Y, FOOT_H = 42.0, 9.0, 10.0          # |X| 42..50, |Yr| < 9, bottom 10 mm
# Three dowels: the +X Pi-side one (and that half of its foot) gave way to the UE300, which stands
# flush against the web there.
PINS = [(-46.0, -5.5), (-46.0, 5.5), (46.0, 5.5)]
PIN_SLIP, PIN_PRESS, PIN_DEPTH = 3.4, 3.3, 4.0      # fit-strip values (no extra comp)
BOSSES = [(sx * 44.0, y) for sx in (-1, 1) for y in (Y_MIN + 10.5, Y_MAX - 10.5)]   # clear of the slots
BOSS_R, INSERT_D, INSERT_DEPTH = 4.5, 4.0, 6.0      # glued M3 brass insert, Ø4 × 3 (in hand); M3×6 button from below

# ── BATTERY: Anker 20K 87 W bank, 157 × 74 × 25, standing on the floor. Compact layout (2026-10-13):
#    the bank sits right past the well tube (its face just clear of the floor collar), between two stop
#    ribs (|X| 30..46) and the +Y panel. The buck, WAGOs and power switch live under the Pi instead. ──
BAT_H_MARGIN = 1.0                            # unmeasured bank height: +1 so the floor always closes (foam takes up rattle)
BAT = dict(x=37.5, y0=30.8, y1=55.8, d0=DEPTH - 157.0 - BAT_H_MARGIN, d1=DEPTH)
BAT_RIB = (30.0, X_IN, 28.5, 30.5)            # |X| range, Yr range; starts at BAT_RAIL_D0 (cord passes above)
BAT_RAIL_D0 = 40.0
# Cradle rails, two per camera wall (near each Yr face of the bank, leaving the middle of its top free for
# the built-in cord), from the flange to the floor. Face 0.5 off the bank (+0.2: outer faces print
# oversize); above the bank's top a 45° shoulder reaches in over its top corners, so with the floor on
# the bank is trapped top and bottom (slid up in from below before the floor goes on). Bank height [MEASURE].
# Two rails per wall, each backed solid through the TPMS wall as a rib. The first merges with the stop rib
# into one stepped flap (stop part across the bank's face, rail part beside it); both stay clear of the
# M4 bores' screw path (head clearance Yr 34.7..45.3) and the built-in cord (Yr ~43).
BAT_RAIL_X, BAT_RAIL_W = BAT['x'] + 0.5 + 0.2, 3.5
BAT_RAIL_YS = (BAT['y0'] + 2.0, BAT['y1'] - 3.5)
BAT_SHOULDER_X = 33.5                         # shoulder's innermost |X| (bank corners at 37)
# Power: the bank stands PORTS DOWN, its front (display + button) facing the +Y panel. Its two
# ports sit over a floor window, for charging the bank or other devices from under the rig (not
# while the rig runs: with two outputs in use the built-in cord drops to 5/9 V). The built-in cord
# (~15 cm) leaves the bank's top end, runs over the stop rib into the +X bay beside the tube, to a PD
# trigger (female-input board, or the male trigger + a PD coupler) set to 15 V; its thin 15 V pair runs
# through the +X upper web slot to the power switch and WAGOs under the Pi. Positions [MEASURE].
PORT_WINDOW = (-22.0, 22.0, 36.8, 49.8)       # floor: X range, Yr range under the bank's end face
CORD_ROOT = (33.0, 43.3, -(DEPTH - 157.0))    # built-in cord leaves the bank's top end here
PD_TRIGGER = (30.0, 42.0, 8.0, 20.0, 26.0, 48.0)    # USB-C female coupler on the built-in cord: X, Yr, depth (+X bay by the tube)
# The PD trigger is itself a long corded plug: it plugs into the coupler and its 15 V lead (plenty of
# slack) runs through the upper web slot to the switch and WAGOs under the Pi.
BUTTON = dict(x=-22.0, d=DEPTH - 22.0, hole=10.0)   # bank's display button, through the clear +Y panel
# BATTERY: office link. A TP-Link UE300 (USB 3 gigabit; body 71 × 25.4 × 15) stands on the floor in
# the corner between the Pi pocket and the +X camera wall, RJ45 end down through a floor cutout, in a
# printed cradle. It is beside the Pi's USB stack, not under it: the fold-out cord ends in a full
# USB-A plug (~20-25 mm with its housing) that needs to go straight up into the blue port before the
# cord can turn, then loops down and back up to the adapter (~5-6 cm of cord). [MEASURE] cord, plug.
# The lidar stays on the Pi's own Ethernet jack.
# Now flush against the web (Pi side), between the floor collar and the +X wall; a ledge on the web
# (ETH_SEAT) caps its top end so it can't slide off the floor with the rig upside down.
ETH_ADAPTER = dict(x=38.15, y=-14.6, wx=15.2, wy=25.4, h=71.0)  # centre X, Yr; footprint along X / Yr
ETH_SEAT = dict(clear=1.4, reach=5.0)         # gap over the adapter's top (0.4 + 1 unmeasured margin), ledge reach off the web (Yr) [MEASURE] height
ADAPTER_JACK = (13.5, 16.0)                   # RJ45 opening in its end face, along X / Yr [MEASURE]
USB3_PORT = (12.0, -41.0, -102.5)             # blue port mouth: Pi top 14 + 88.5 (+X stack assumed) [MEASURE]
CRADLE_H, CRADLE_WALL, CRADLE_CLEAR = 12.0, 2.0, 0.3
# LIDAR CABLE ENTRY (2026-10-13): the M12 plug sticks straight out along +Yr (X 0, ~23 above the plate
# underside) to Yr ~83, so the cable turns 180° outside the box and lies down into an open slot in the
# flange, cut from the +Y end inward past the plate edge (Yr 45) to CABLE_SLOT_Y0. The bare cable is laid
# in from the end (no plug threading); it drops in over the bank's top, and the +Y panel slides up and
# closes the slot's outer part. Then: slack coil in the -X bay beside the tube, the -X low web slot
# (RJ45 + ferrite pass there) to the Pi side, up into the Ethernet jack. Cable Ø [MEASURE].
# Revised: the flange plate stays solid; the cable goes in sideways through a U-notch cut up into the
# +Y rim from below (between the 4 mm plate and the panel's top edge, which closes it). Cable Ø [MEASURE].
CABLE_D = 7.0
CABLE_NOTCH_W = 9.0                           # notch width (X, centred on 0); height FLANGE_PLATE..RIM_D
# Push-in cable clips: a C of two claws round a vertical cable, on a short stem off a surface; opening
# 0.85 Ø (claws flex ~0.5 mm: elastic), 45° lead-in below (print). (axis, base, along, d0, cable Ø):
# axis 'x' sits on a wall/rib face at signed X = base, at Yr = along; axis 'y' on a web/rib face at
# signed Yr = base, at X = along. Clips stand off toward -sign(base) unless an optional 6th entry
# gives the direction (+1 / -1), e.g. on the battery-side face of a web.
CLIP_T, CLIP_H, CLIP_STEM = 1.2, 14.0, 1.0
CLIP_S, CLIP_RIB_DS = 4.5, (55.0, 80.0, 104.0)   # small clip Ø; camera rib clip depths (below the camera blocks, above the UE300)
CLIPS = ([('y', 28.5, -38.0, 76.0, CABLE_D),   # -X bay, on the battery stop rib: lidar cable below its coil
          ('y', -1.5, -37.0, 110.0, CABLE_D),  # -X web, Pi side: lidar cable after the low slot
          ('y', 28.5, 38.0, 103.0, CABLE_D),   # +X bay stop rib: one big, centred on the rib's run
          ('y', 1.5, 37.0, 50.75, CLIP_S, 1),  # +X web, battery side (opposite the stop rib): two small,
          ('y', 1.5, 37.0, 107.0, CLIP_S, 1)]  #   one centred between the two slots, one past the low slot
         + [('x', sx * 43.5, CAM_RIB_Y, cd, CLIP_S) for sx in (-1, 1) for cd in CLIP_RIB_DS if sx < 0 or cd < 100])   # camera ribs (USB-C on -X)


def clip_centre(axis, base, along, D, out=None):
    """(X, Yr) of the cable centre held by a clip."""
    c = (D + RESIN_HOLE_COMP) / 2 + CLIP_T + CLIP_STEM
    u = base + (out if out else -np.sign(base)) * c
    return (u, along) if axis == 'x' else (along, u)

# ── Power switch: 16 mm panel-mount latching push button (LED ring ideal) in the 15 V line between
#    the PD trigger and the WAGOs, so it cuts lidar + Pi together. On the -X camera wall under the Pi,
#    opposite the UE300 (+X); its ~30 mm body points in below the Pi's port end. Plain round hole;
#    the nut still seats. [MEASURE] body, thread.
# Part: Gebildet 16 mm stainless latching push button, 12-24 V / 5 A, power-symbol LED. Solder the
# leads to the switch tabs (heat-shrink them) instead of its clip-on socket plug, which adds ~20 mm.
POWER_SWITCH = dict(side=-1, y=CAM_YR, d=155.0, hole=16.0, body_r=8.0, body_len=30.0)
# Buck (36 × 10 × 45) stands against the -Y panel under the Pi; two WAGO 221-413 beside the switch. [MEASURE]
BUCK = (-10.0, 26.0, -56.0, -46.0, 128.0, 173.0)   # X, Yr, depth ranges
WAGOS = [(-14.0, -9.0, -45.0, -36.0, 166.0, 178.0), (-7.5, -2.5, -45.0, -36.0, 166.0, 178.0)]

# ── Wall vents (instead of panel vents): vertical slots through both camera walls. Low on the Pi
#    side to feed the cooler now that its panel is solid; beside the buck bay on the battery side,
#    below the power switch. The high exhaust slots (EXHAUST) stay. (Yr0, Yr1, d0, d1) ──
WALL_VENTS = ([(-52.0 + 4 * i, -49.6 + 4 * i, 120.0, 160.0) for i in range(4)] +
              [(12.0 + 4 * i, 14.4 + 4 * i, 60.0, 100.0) for i in range(8)])

# ── Weight reduction (2026-10-13): the flange is a 4 mm plate except Ø20 × 10 bosses at the four
#    stepped bores and 10 mm rims along the panel ends (the panels' top stop). The camera walls are
#    an open sheet gyroid (resin drains; doubles as the vents) except solid zones: top and bottom
#    bands, the slot ends, the camera and switch pads, round the web; plus internal vertical ribs. ──
FLANGE_PLATE = 4.0
CORNER_BOSS_R = 10.0
GYROID_PERIOD, GYROID_T = 12.0, 0.35          # mm; sheet half-thickness in gyroid units (~25 % solid; chosen over gyroid 7 / diamond 9)
WALL_TOP_SOLID, WALL_BOTTOM_SOLID = FLANGE + 2.0, 14.0
CAM_PAD = (15.0, 16.0)                        # half-width along Yr, reach past CAM_D: solid round each camera
SWITCH_PAD_R = 14.0
WEB_PAD = 6.0                                 # |Yr| solid round the web / wall feet
# Internal vertical ribs (3 wide, 2.5 deep), (Yr, top depth): one flanking the camera pad's outer
# edge (the web is the rib on its other side). On the battery side the cradle rails are the ribs.
WALL_RIBS = ((CAM_RIB_Y, FLANGE),)   # flanks the camera board's +Yr edge; the cradle rails are the battery-side ribs


# Wall fill, selectable for design review: env PI_BOX_FILL = "kind:a:b" (default gyroid:12:0.35).
#   gyroid|schwarzp|diamond : TPMS sheet, a = period mm, b = sheet half-thickness (TPMS units)
#   honeycomb               : round holes on a hex grid, a = pitch, b = hole Ø
#   slits                   : vertical slits, a = pitch, b = slit width (bridged every 30 mm)
#   lozenge                 : 45° diamond mesh (self-supporting), a = pitch, b = bar width
import os
WALL_FILL = os.environ.get('PI_BOX_FILL', f'gyroid:{GYROID_PERIOD:g}:{GYROID_T:g}')


FILL_RAMP = 6.0               # mm above the bottom band: TPMS sheets thicken until they merge (no flat start)


def wall_fill(X, Y, d):
    kind, a, b = WALL_FILL.split(':')
    a, b = float(a), float(b)
    k = 2 * np.pi / a
    t = min(max((d - (DEPTH - WALL_BOTTOM_SOLID - FILL_RAMP)) / FILL_RAMP, 0.0), 1.0)   # d is a scalar (also on GPU)
    if kind in ('gyroid', 'schwarzp', 'diamond'):
        b = b + t * ((3.1 if kind == 'schwarzp' else 1.6) - b)   # past max |TPMS|: fully solid at the band
    if kind == 'gyroid':
        g = np.sin(k * X) * np.cos(k * Y) + np.sin(k * Y) * np.cos(k * d) + np.sin(k * d) * np.cos(k * X)
        return np.abs(g) <= b
    if kind == 'schwarzp':
        return np.abs(np.cos(k * X) + np.cos(k * Y) + np.cos(k * d)) <= b
    if kind == 'diamond':
        sx, sy, sz, cx, cy, cz = np.sin(k * X), np.sin(k * Y), np.sin(k * d), np.cos(k * X), np.cos(k * Y), np.cos(k * d)
        return np.abs(sx * sy * sz + sx * cy * cz + cx * sy * cz + cx * cy * sz) <= b
    if kind == 'honeycomb':
        row = np.floor(d / (a * 0.866))
        yy = Y - (row % 2) * a / 2
        cy_ = np.round(yy / a) * a + (row % 2) * a / 2
        cd_ = (row + 0.5) * a * 0.866
        return (Y - cy_) ** 2 + (d - cd_) ** 2 > (b / 2) ** 2
    if kind == 'slits':
        return (np.mod(Y, a) > b) | (np.mod(d, 30.0) < 3.0)
    if kind == 'lozenge':
        u, v = np.mod(Y + d, a), np.mod(Y - d, a)
        return (u < b) | (v < b)
    raise ValueError(kind)


def gyroid(X, Y, d):                                   # kept name: the core calls this for the wall fill
    return wall_fill(X, Y, d)


def wall_solid_zone(Y, d, X=None):
    """Where the camera walls stay solid (everything else is gyroid). X: only the switch wall gets its pad."""
    gw = GROOVE_W + RESIN_HOLE_COMP
    z = (d <= WALL_TOP_SOLID) | (d >= DEPTH - WALL_BOTTOM_SOLID)
    z = z | (Y <= Y_MIN + PANEL_LIP + gw + 4.0) | (Y >= Y_MAX - PANEL_LIP - gw - 4.0)
    z = z | ((np.abs(Y - CAM_YR) <= CAM_PAD[0]) & (d <= CAM_D + CAM_PAD[1]))
    pad = (Y - POWER_SWITCH['y']) ** 2 + (d - POWER_SWITCH['d']) ** 2 <= SWITCH_PAD_R ** 2
    if X is not None:
        pad = pad & (np.sign(X) == POWER_SWITCH['side'])
    z = z | pad
    z = z | (np.abs(Y) <= WEB_PAD)
    for r, top in WALL_RIBS:
        z = z | ((np.abs(Y - r) <= 1.5) & (d >= top))
    for ry in BAT_RAIL_YS:                             # the cradle rails run through the wall as ribs
        z = z | (np.abs(Y - ry) <= BAT_RAIL_W / 2)
    z = z | ((Y >= BAT_RIB[2]) & (Y <= BAT_RAIL_YS[0]))   # ...the first one merged with the stop rib
    return z

# ── Flange vents (no suction cups when printed flange-down) ──
VENTS = [(0, -45), (-15, -45), (15, -45), (0, 38), (-15, 38), (15, 38),
         (37, 20), (37, -20), (-37, 20), (-37, -20)]
VENT_D = 4.0

# ── Pi 5 (stands on its short edge, ports down, HDMI edge toward -X) ──
PCB_YR = -31.0                # back (underside) of the PCB
PI_TOP_D = 14.0               # SD/PCIe end of the board; the card stops ~2 mm short of the flange
PI_HOLES = [(sx * 24.5, PI_TOP_D + bx) for sx in (-1, 1) for bx in (3.5, 61.5)]
PI_POST_R = 3.2
# Glued M2.5 brass insert (kit: L 2.5) at the PCB end, clearance hole beyond for the M2.5×6 socket cap's tip
PI_INSERT_D, PI_INSERT_L, PI_SCREW_CLEAR, PI_HOLE_DEPTH = 3.5, 3.0, 2.7, 8.0   # [MEASURE] insert OD

# ── OV9281 cameras (25 × 23.86 board, lens 18 mm from the board face) ──
# Lens axes just under the flange: the stiffest spot, ~55 mm from the lidar instead of ~90.
CAM_D = 30.0                              # lens axis depth (both cameras; CAM_YR above); 30 (was 23, 2026-10-14): board clears the upper Pi standoff on its way in
CAM_WINDOW = 19.0                         # ROUND lens window Ø (lens holder must pass: round, or square ≤ ~13) [MEASURE]
CAM_BOSS_R, CAM_BOSS_H = 3.0, 2.0
# Inset (2026-10-13): the 160° lens can't sit behind the wall (each 1 mm of recess needs ~5.7 mm more
# window radius at 80° off-axis), so the board moves CAM_INSET deeper and the lens front (18 mm from the
# board face, spec sheet) stands ~4 mm proud instead of ~12. The board mounts on a solid block that grows
# from the flange plate (no overhang) to the board's far edge, window bored through; a pocket at the
# connector edge clears the FFC connector and lets the ribbon out.
CAM_INSET, CAM_LENS_L = 8.0, 18.0
CAM_FACE = X_IN - CAM_BOSS_H - CAM_INSET      # |X| of the board's front face (the block's inner face)
CAM_POCKET = (6.0, 4.0)                       # connector pocket: width in from the board edge, depth into the block
# Glued M2 brass insert (kit: L 2) in each boss, clearance beyond for the M2×5 socket cap's tip
CAM_INSERT_D, CAM_INSERT_L, CAM_SCREW_CLEAR, CAM_HOLE_DEPTH = 3.2, 2.5, 2.2, 5.0   # [MEASURE] insert OD
# Connector edge direction per side. Board 25 × 23.86; the lens sits 8.25 from the edge opposite
# the connector and the 21 × 12.5 holes are symmetric about it. The -X (near) camera turns its
# connector down so its ribbon runs down to the Pi, clear of the Pi's USB-C plug ~25 mm down.
CAM_CONN_DIR = {-1: 'down', 1: 'down'}      # both down (2026-10-13): the ribbons drop straight out of the block's pocket


def cam_holes(s):
    a, b = (6.25, 10.5) if CAM_CONN_DIR[s] == '-y' else (10.5, 6.25)
    return [(CAM_YR + sy * a, CAM_D + sd * b) for sy in (-1, 1) for sd in (-1, 1)]


def cam_board(s):
    """(Yr0, Yr1, d0, d1) of the board outline."""
    if CAM_CONN_DIR[s] == '-y':
        return CAM_YR - 15.6, CAM_YR + 8.25, CAM_D - 12.5, CAM_D + 12.5
    return CAM_YR - 12.5, CAM_YR + 12.5, CAM_D - 8.25, CAM_D + 15.6
GUARD_R0, GUARD_R1, GUARD_X = (CAM_WINDOW + RESIN_HOLE_COMP) / 2, 11.7, X_OUT + 2.0   # low guard lip, ~2 mm behind the lens front

# ── Exhaust slots in the camera walls beside the Pi ──
EXHAUST = [(-38 - 4 * i, -36 - 4 * i) for i in range(4)]   # Yr ranges
EXHAUST_D = (20.0, 45.0)


def hole(cx, cy, d, a, b):
    """Disk test for a hole drawn at the wanted size, compensated for resin shrink."""
    r = (d + RESIN_HOLE_COMP) / 2
    return (a - cx) ** 2 + (b - cy) ** 2 <= r * r


def hole_h(cu, cd, dia, u, d):
    """Horizontal-axis hole (u across, d = print height): +comp, and taller by HORIZ_HOLE_EXTRA."""
    ru, rd = (dia + RESIN_HOLE_COMP) / 2, (dia + RESIN_HOLE_COMP + HORIZ_HOLE_EXTRA) / 2
    return ((u - cu) / ru) ** 2 + ((d - cd) / rd) ** 2 <= 1.0


def to_print(pts):
    """Rig (X, Yr, Zr) -> print (x, y, z).  BATTERY: turned 90° on the plate, x = Yr, z = -X."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([pts[..., 1], -pts[..., 2], -pts[..., 0]], axis=-1)


def from_print(pts):
    """Print (x, y, z) -> rig (X, Yr, Zr)."""
    pts = np.asarray(pts, dtype=np.float64)
    return np.stack([-pts[..., 2], pts[..., 0], -pts[..., 1]], axis=-1)


# ═════════════════════════════ overlays ═════════════════════════════
def _write_stl(path, tris):
    tris = np.asarray(tris, dtype=np.float32).reshape(-1, 3, 3)
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=[('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
    rec['n'], rec['v'] = n, tris
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(b'\0' * 80 + struct.pack('<I', len(tris)))
        f.write(rec.tobytes())


def _box(x0, x1, y0, y1, z0, z1):
    """Axis-aligned box in rig coords -> triangles."""
    v = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)])
    q = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    return np.array([[v[a], v[b], v[c]] for a, b, c, d in q] + [[v[a], v[c], v[d]] for a, b, c, d in q])


def _cyl(c, axis, r, h0, h1, n=32):
    """Cylinder along rig axis 'x'|'y'|'z' through point c, from h0 to h1 along that axis."""
    t = np.linspace(0, 2 * np.pi, n + 1)
    ring = np.stack([np.cos(t), np.sin(t)], 1) * r
    ax = 'xyz'.index(axis)
    oth = [i for i in range(3) if i != ax]

    def pt(u, w, h):
        p = np.array(c, dtype=float)
        p[oth[0]] += u
        p[oth[1]] += w
        p[ax] = h
        return p

    tris = []
    for i in range(n):
        a0, a1 = ring[i], ring[i + 1]
        p00, p01 = pt(*a0, h0), pt(*a1, h0)
        p10, p11 = pt(*a0, h1), pt(*a1, h1)
        tris += [[p00, p01, p11], [p00, p11, p10],
                 [pt(0, 0, h0), p01, p00], [pt(0, 0, h1), p10, p11]]
    return np.array(tris)


def _frames(pts, hint):
    """Per-segment direction and a perpendicular 'width' vector (from hint)."""
    pts = np.asarray(pts, float)
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        t = (b - a) / np.linalg.norm(b - a)
        w = np.cross(t, hint)
        if np.linalg.norm(w) < 1e-6:
            w = np.cross(t, [1.0, 0, 0] if abs(t[0]) < 0.9 else [0, 1.0, 0])
        w /= np.linalg.norm(w)
        out.append((a, b, t, w, np.cross(w, t)))
    return out


def _tube(pts, r, n=10):
    """Round cable along a polyline (rig coords); joints get a short sleeve overlap."""
    tris = []
    ang = np.linspace(0, 2 * np.pi, n + 1)
    for a, b, t, u, v in _frames(pts, [0, 0, 1.0]):
        a, b = a - t * r * 0.5, b + t * r * 0.5
        ring = [np.cos(q) * u * r + np.sin(q) * v * r for q in ang]
        for i in range(n):
            tris += [[a + ring[i], a + ring[i + 1], b + ring[i + 1]], [a + ring[i], b + ring[i + 1], b + ring[i]]]
    return np.array(tris)


def _strip(pts, width, thick, hint):
    """Flat ribbon along a polyline; its width lies across the path, perpendicular to hint."""
    tris = []
    for a, b, t, w, n in _frames(pts, hint):
        tris.append(_quad_box(a, b, w * width / 2, n * thick / 2))
    return np.concatenate(tris)


def _quad_box(a, b, hw, hn):
    c = [a - hw - hn, a + hw - hn, a + hw + hn, a - hw + hn, b - hw - hn, b + hw - hn, b + hw + hn, b - hw + hn]
    f = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return np.array([[c[i], c[j], c[k]] for i, j, k, l in f] + [[c[i], c[k], c[l]] for i, j, k, l in f])


# ── Cable routes (rig X, Yr, Zr; Zr = -depth). Schematic: they show the path and the
#    space it needs, not the slack. Ribbons leave the Pi connectors along the board
#    (upward assumed), 22-pin FFC ~13 mm wide. ──
PI_CONN = [(-19.0, -35.5, -(PI_TOP_D + 49)), (-19.0, -35.5, -(PI_TOP_D + 57))]     # near, far
CAM_CONN = [(-(CAM_FACE + 2.0), CAM_YR, -(CAM_D + 15.0)),            # -X near: connector edge down (in its pocket)
            (CAM_FACE + 2.0, CAM_YR, -(CAM_D + 15.0))]                 # +X far: connector edge down too
RIBBON_NEAR = [PI_CONN[0], (-26, -35.5, -60), (-32, CAM_YR, -(CAM_D + 23)), (CAM_CONN[0][0], CAM_YR, -(CAM_D + 23)),
               CAM_CONN[0]]                                                                    # up into its pocket
RIBBON_FAR = [PI_CONN[1], (-26, -35.5, -68), (-26, -35.5, -42), (-29.8, -35.5, -40), (-29.8, -29.6, -40),
              (29.8, -29.6, -40), (29.8, -29.6, -(CAM_D + 23)), (33, CAM_YR, -(CAM_D + 23)), (CAM_FACE + 2.0, CAM_YR, -(CAM_D + 23)), CAM_CONN[1]]   # up into its pocket
# Pi 5 port end (bottom, ~99-104 mm down): Ethernet is the jack beside the micro-HDMI edge (-X here,
# board y ~2-18 -> X -26..-10); the two USB-A stacks are toward +X.
ETH = (-18.0, -40.0, -(PI_TOP_D + 88.5))
# Lidar cable (see LIDAR CABLE ENTRY above): plug, 180° loop outside the +Y end, in through the flange
# slot, over the bank, a 2-turn slack coil in the -X bay, the -X low web slot, up into the Pi's jack.
PANEL_NOTCH_X, PANEL_NOTCH_D, PANEL_NOTCH_W = 22.0, 20.0, 8.0   # notch centre X, bottom depth, width (unused)
M12_ZR, M12_PLUG = 23.6, (38.3, 83.0, 8.0)    # connector axis height (rig Zr); plug Yr range, radius [MEASURE]
SLACK = dict(c=(-35.0, 15.0, -60.0), r=8.5, turns=2.0, x0=-38.0, x1=-32.0)   # coil in the Yr-d plane, clear of the Yr 15 rib


def _slack_pts(n=40):
    _, cy, cz = SLACK['c']
    t = np.linspace(0, 2 * np.pi * SLACK['turns'], int(n * SLACK['turns']) + 1)
    x = SLACK['x0'] + (SLACK['x1'] - SLACK['x0']) * t / t[-1]
    return [(xx, cy + SLACK['r'] * np.sin(a), cz + SLACK['r'] * np.cos(a)) for a, xx in zip(t, x)]


_NOTCH_D = (FLANGE_PLATE + RIM_D) / 2                                   # cable centre in the rim notch
_LOOP_R = (M12_ZR + _NOTCH_D) / 2                                         # plug axis down to the notch
_LOOP = [(0, M12_PLUG[1] + _LOOP_R * np.sin(a), M12_ZR - _LOOP_R + _LOOP_R * np.cos(a)) for a in np.linspace(0, np.pi, 13)]
LIDAR_CABLE = (_LOOP +                                                   # 180° turn outside the box end
               [(0, 62, -_NOTCH_D), (0, 51, -_NOTCH_D), (0, 47, -15), (-20, 40, -15), (-29, 30, -15), (-35, 18, -30)]   # rim notch, over the bank
               + _slack_pts()
               + [(-38, 22.8, -72), (-38, 22.8, -92), (-37, 0, -93),       # stop-rib clip, -X low web slot
                  (-37, -7.2, -100), (-37, -7.2, -126),                    # Pi-side web clip
                  (-30, -40, -128), (-18, -42, -135), ETH])                # up into the jack
SW_C = (POWER_SWITCH['side'] * (X_IN - POWER_SWITCH['body_len']), POWER_SWITCH['y'], -POWER_SWITCH['d'])   # switch body's inner end
WAGO_C = (-8.0, -40.0, -172.0)
LIDAR_POWER = [(-30, -40, -112), (-20, -48, -140), (-12, -44, -160), WAGO_C]   # Livox split's power pair -> WAGO
BUILTIN_CORD = [CORD_ROOT, (30, 43, -16), (29, 30, -16), (33, 22, -22), (36, 18, -PD_TRIGGER[4])]   # between the cradle rails, into the +X bay
BATTERY_LEAD = [(36, 10, -PD_TRIGGER[5] + 2), (37, 5.95, -50), (37, 5.95, -68), (37, 0, -90), (28, -25, -95),
                (24, -50, -110), (-10, -52, -150), SW_C, WAGO_C]  # trigger -> battery-side web clip -> low web slot -> under the Pi -> switch -> WAGO
BUCK_IN = [WAGO_C, (8.0, -46.5, -170.0)]
USBC_LEAD = [(8.0, -51.0, -BUCK[4]), (-30, -50, -128), (-39.05, CAM_RIB_Y, -126), (-39.05, CAM_RIB_Y, -50),
             (-33, CAM_RIB_Y, -47), (-33, -34, -25)]              # up the -X camera rib's clips, inboard of the camera block, to the plug


def _decimate(tris, grid=0.25):
    """Vertex clustering: snap to a grid and drop collapsed triangles (overlay only)."""
    q = np.round(tris / grid) * grid
    a, b, c = q[:, 0], q[:, 1], q[:, 2]
    keep = ~(np.all(a == b, 1) | np.all(b == c, 1) | np.all(a == c, 1))
    return q[keep]


def _load(name):
    from stl_slicer import load_stl
    return load_stl(MODELS / name).astype(np.float64)


def build_overlays():
    """Write transformed overlay STLs (print frame) and return the viewer list.

    Reference models (gitignored, see models/README.md): rpi5.stl from the official
    Raspberry Pi 5 STEP (MIT), mid360s.stl from Livox's Mid-360S STEP.
    """
    out = []

    def emit(name, rig_tris, color, opacity=0.35, label=None):
        path = OVERLAY_DIR / f'{name}.stl'
        _write_stl(path, to_print(rig_tris))
        out.append(dict(file=str(path.relative_to(ROOT)).replace('\\', '/'), color=color, opacity=opacity,
                        label=label or name.replace('_', ' ')))

    # Pi 5: model board x 0..85 (SD end at 0), y 0..56 (HDMI edge at 0), z up (components)
    if (MODELS / 'rpi5.stl').exists():
        t = _decimate(_load('rpi5.stl'))
        bx, by, bz = t[..., 0], t[..., 1], t[..., 2]
        emit('pi5', np.stack([by - 28.0, PCB_YR - bz, -(PI_TOP_D + bx)], -1), '#2ecc71', 0.55, 'Pi 5 (official STEP)')
    # Active Cooler envelope [MEASURE]: over the SoC area, ~14 mm above the board
    emit('cooler', _box(-24, 18, PCB_YR - 1.6 - 14, PCB_YR - 1.6, -(PI_TOP_D + 62), -(PI_TOP_D + 12)), '#95a5a6', 0.35, 'Active Cooler envelope [MEASURE]')

    # Mid-360S: model Y up (base at -25.92), connector +X; sits on the 6.35 plate
    if (MODELS / 'mid360s.stl').exists():
        t = _decimate(_load('mid360s.stl'), 0.15)
        emit('mid360s', np.stack([t[..., 2], t[..., 0], t[..., 1] + 25.92 + 6.35], -1), '#bdc3c7', 0.5, 'Mid-360S (Livox STEP)')

    # Aluminium plate, standoffs + spacers, coupler, monopod platform + pole
    emit('base_plate', _box(-45, 45, -45, 45, 0, 6.35), '#7f8c8d', 0.3)
    posts = np.concatenate([_cyl((sx * POST, sy * POST, 0), 'z', 3, 6.35, 72.35)
                            for sx in (-1, 1) for sy in (-1, 1)])
    spacers = np.concatenate([_cyl((sx * POST, sy * POST, 0), 'z', SPACER_D / 2, -SPACER_D + 2, 0)
                              for sx in (-1, 1) for sy in (-1, 1)])
    emit('standoffs', posts, '#7f8c8d', 0.3)
    emit('box_spacers', spacers, '#e67e22', 0.7)
    emit('coupler', _cyl((0, 0, 0), 'z', 6.4, -20.3, 0, 6), '#7f8c8d', 0.6)
    # Amazon Basics 67" monopod, handle end up (measured roughly 2026-10-13): Ø35 end cap at the stud,
    # Ø40 ring, then the foam grip (~135 mm long, narrower than the Ø40 ring). Strap lug cut off for v1.
    emit('monopod', np.concatenate([_cyl((0, 0, 0), 'z', 17.5, -28.3, -20.3),
                                    _cyl((0, 0, 0), 'z', 20.0, -38.3, -28.3),
                                    _cyl((0, 0, 0), 'z', 19.0, -20.3 - 135, -38.3),
                                    _cyl((0, 0, 0), 'z', 15, -260, -20.3 - 135)]), '#34495e', 0.3,
         'monopod: cap Ø35, ring Ø40, grip < Ø40 to 135 mm')

    # OV9281 boards + lenses (board face on the camera block), back-side parts 3.57 tall (spec sheet)
    cams = []
    for s in (-1, 1):
        xf = s * CAM_FACE                                 # board front face
        y0, y1, d0, d1 = cam_board(s)
        cams.append(_box(min(xf, xf - s), max(xf, xf - s), y0, y1, -d1, -d0))
        xb = xf - s * 4.57                                # back of the tallest back-side parts
        cams.append(_box(min(xf - s, xb), max(xf - s, xb), y0 + 2, y1 - 2, -d1 + 2, -d0 - 2))
        cams.append(_cyl((0, CAM_YR, -CAM_D), 'x', 7, xf, xf + s * CAM_LENS_L) if s > 0 else
                    _cyl((0, CAM_YR, -CAM_D), 'x', 7, xf + s * CAM_LENS_L, xf))
    emit('cameras', np.concatenate(cams), '#8e44ad', 0.6, 'OV9281 boards + lenses')

    # BATTERY: the bank
    a = ETH_ADAPTER
    emit('eth_adapter', np.concatenate([
        _box(a['x'] - a['wx'] / 2, a['x'] + a['wx'] / 2, a['y'] - a['wy'] / 2, a['y'] + a['wy'] / 2, -DEPTH, -DEPTH + a['h']),
        _box(USB3_PORT[0] - 6, USB3_PORT[0] + 6, USB3_PORT[1] - 2.5, USB3_PORT[1] + 2.5,
             USB3_PORT[2] - 22, USB3_PORT[2]),                          # USB-A plug + housing
        _tube([(a['x'], a['y'], -DEPTH + a['h']), (a['x'], a['y'], -DEPTH + a['h'] + 4), (25, -37, -DEPTH + a['h'] + 4),
               (25, -38, USB3_PORT[2] - 28), (USB3_PORT[0], USB3_PORT[1], USB3_PORT[2] - 28),
               (USB3_PORT[0], USB3_PORT[1], USB3_PORT[2] - 22)], 1.8)]),
        '#ecf0f1', 0.8, 'TP-Link UE300 → office link (RJ45 in the floor)')
    x0, x1, y0, y1, d0, d1 = PD_TRIGGER
    emit('pd_trigger', np.concatenate([_box(x0, x1, y0, y1, -d1, -d0), _tube(BUILTIN_CORD, 2.0)]),
         '#2c3e50', 0.9, 'built-in cord (bank top) → USB-C coupler [MEASURE]')
    emit('battery', _box(-BAT['x'] + 0.5, BAT['x'] - 0.5, BAT['y0'], BAT['y1'], -BAT['d1'], -BAT['d0']),
         '#34495e', 0.5, 'Anker 20K bank, ports DOWN, display toward +Y')

    # 15→5 V buck and two WAGO 221-413 under the Pi [MEASURE]
    x0, x1, y0, y1, d0, d1 = BUCK
    emit('buck_wago', np.concatenate([_box(x0, x1, y0, y1, -d1, -d0)] +
                                     [_box(a, b, c, e, -g, -f) for a, b, c, e, f, g in WAGOS]),
         '#f39c12', 0.5, 'buck + 2 WAGO under the Pi [MEASURE]')

    sw, sx = POWER_SWITCH, POWER_SWITCH['side']
    bezel = sorted((sx * (X_OUT + 0.05), sx * (X_OUT + 3)))       # sits on the wall's outer face
    body = sorted((sx * (X_OUT - sw['body_len']), sx * X_OUT))
    emit('power_switch', np.concatenate([_cyl((0, sw['y'], -sw['d']), 'x', 9.5, *bezel),
                                         _cyl((0, sw['y'], -sw['d']), 'x', sw['body_r'], *body)]),
         '#e74c3c', 0.8, 'power switch, 16 mm latching (15 V line)')

    # Cables (schematic routes, see the route constants above)
    emit('cable_ribbons', np.concatenate([_strip(RIBBON_NEAR[:-1], 13, 0.3, [0, 1.0, 0]),
                                          _strip(RIBBON_NEAR[-2:], 13, 0.3, [1.0, 0, 0]),
                                          _strip(RIBBON_FAR[:-1], 13, 0.3, [0, 1.0, 0]),
                                          _strip(RIBBON_FAR[-2:], 13, 0.3, [1.0, 0, 0])]), '#e67e22', 1.0,
         'camera ribbons (15→22-pin, 150 mm)')
    emit('m12_plug', _cyl((0, 0, M12_ZR), 'y', M12_PLUG[2], M12_PLUG[0], M12_PLUG[1]), '#1f4e79', 0.9,
         'M12 plug (straight, ~45 long) [MEASURE]')
    emit('cable_lidar', _tube(LIDAR_CABLE, 3.5), '#1f4e79', 1.0, 'lidar cable: M12 → flange slot → slack → RJ45 (Pi Ethernet)')
    emit('cable_lidar_power', _tube(LIDAR_POWER, 2.0), '#8e1b1b', 1.0, 'lidar power pair (split) → WAGO')
    emit('cable_battery', _tube(BATTERY_LEAD, 2.0), '#e74c3c', 1.0, 'PD trigger cord (15 V) → power switch → WAGO')
    emit('cable_buck_in', _tube(BUCK_IN, 1.5), '#f1948a', 1.0, 'WAGO → buck input')
    emit('cable_usbc', _tube(USBC_LEAD, 2.0), '#111111', 1.0, 'buck USB-C → Pi power (right-angle plug)')
    return out
