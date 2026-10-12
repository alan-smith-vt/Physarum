# Lidar photo-rig cage handoff (2026-10-11)

This update follows "Lidar photo-rig handoff.md" (Oct 7). It records what was built, tested, decided and ordered on 2026-10-11. Where the two disagree, this file wins.

## Status

- **Resin mock:** assembled and fit-checked against the real Mid-360S. The base plate came out perfect: the M3 pattern and the locating nubs (in place of pins) line up, the lidar sits flat, the cross clears the dome, and the M12 connector sits between the posts.
- **Aluminum plates:** ready to order from SendCutSend with the DXFs written by `designs/cage_dxf.py` (`python designs/cage_dxf.py` → `out/`).
  - **Base plate:** 6061-T6, 1/4", 90 × 90 mm, R5 corners. No services.
    - SendCutSend won't countersink parts under 1" × 4", so countersink the centre Ø6.6 hole by hand with an 82° bit (5/8" or 3/4"). Stop when a 1/4-20 flat head sits flush.
  - **Cross top:** 6061-T6, 3/16", 92 × 92 mm X-shape. Have SendCutSend **tap the centre hole 1/4-20** (drawn Ø5.1).
- **Hardware:** on order (Amazon), listed below.

## Changes from the Oct 7 handoff

| Item | Oct 7 plan | Now | Why |
|---|---|---|---|
| Base screws (M3 lidar, M4 standoff) | Countersunk flat-heads from below | Button/truss heads from below, plain clearance holes | SendCutSend can only countersink one face, and the parts are too small for it anyway. An M4 countersink would leave ~0.8 mm of wall at the R5 corners. Flat-heads would also fight the dowels. |
| Dowel holes | Press fit | Ø3.2, pins glued with red threadlocker | Laser holes aren't accurate enough for a press fit, and Ø3.0 is below the minimum hole size for 1/4" plate |
| Anti-twist pin on the cross | 3 mm pin | Dropped | The GoPro cage has no pin feature. Replaced by a CAMVATE locking 1/4-20 adapter, which also sets the heading. |
| Standoffs | 65 mm, 7 mm hex | 60 mm round Ø6 M4 F-F (8 mm thread depth, solid core) + 6 mm Al spacer on top = 66 mm | Nothing longer than 60 mm was available. About half the torsional stiffness of 7 mm hex, so a hard knock twists it ~0.3–0.5°. That's still inside the 1–2° budget. Torque **1.5–2 N·m** (1 mm thread wall). |
| Logger | Phone first, Pi fallback | **Raspberry Pi 5 8 GB**; the phone is only a pocket display over the Pi's WiFi hotspot | Simpler and more robust |
| Lidar power | User's Anker A1647 at 12 V | **Anker A1383** (20 Ah, 65 W, 15V⎓3A), trigger cable set to **15 V** | The A1647 tops out at 12 V / ~18 W, too tight for lidar + Pi |
| Pole | Insta360 3 m selfie stick | Amazon Basics 67" monopod | The selfie stick sags 3–5" with only the GoPro. Use the rig at ~1.5–2 m extension. |

## Order list

- **Plates:** SendCutSend, see above.
- **Screws:**
  - M3×10 button heads (lidar)
  - M4×12 truss heads (base → standoff), M4×18 truss heads (cross + spacer → standoff)
  - 1/4-20 × ½" flat head (base → coupler)
- **Mounting:** 1/4-20 coupling nut 20.3 mm, CAMVATE 1/4-20 locking double-ended adapter.
- **Posts:** M4 Al spacers 4.2 × 8 × 6 mm, 60 mm round M4 standoffs.
- **Pins:** 3×6 mm dowel pins.
- **Glue:** blue/red threadlocker (GH1200 combo).
- **Tool:** 82° countersink set.
- **Electronics:**
  - Pi 5 8 GB, PNY 512 GB microSD, official Pi 5 Active Cooler
  - Anker A1383, PlusRoc 12/24 V→5 V 5 A USB-C buck (2-pack)
  - WAGO 221 kit (use the 221-413 three-ports)
- **Pole:** Amazon Basics 67" monopod.

Screw lengths assume the 8 mm standoff thread depth:
- M4×12 gives 5.65 mm of thread in the standoff.
- M4×18 through the 4.8 mm cross plus 6 mm spacer gives ~7.2 mm.
- M3×10 gives ~3.6 mm into the lidar's 5 mm holes.

Use blue threadlocker on the M3/M4 screws, and red on the centre flat head and the CAMVATE.

## Power wiring

```
A1383 USB-C port → PD trigger (15 V) → WAGO 221-413 (+ and −) ─┬─ lidar red/black
                                                                └─ PlusRoc buck → USB-C → Pi 5
```

- Check polarity with a meter before connecting either device.
- The fuse is optional: the bank cuts out at ~3 A.
- If the buck is a true 5 A, set `usb_max_current_enable=1` in the Pi's config.txt.
- Plug into the bank's USB-C **port**, not its built-in cable. With both outputs in use, the built-in cable drops to 5/9 V.

## Open tasks

1. **Pi/battery enclosure.** Undecided between:
   - (a) two side boxes under the base plate, each hung from 2 corner standoff screws with metal sleeves so the clamp stack stays metal; or
   - (b) one box at the bottom of the pole, with the lidar cable spiralled loosely around the pole (~1 turn per 25–30 cm) to handle telescoping.

   The stock cable is ~62" in total, so (b) works up to ~1.3–1.4 m of extension. The user wants to trial (b) with a bag first. Design notes:
   - Battery: 157 × 74 × 25 mm, 440 g.
   - Pi: stands on its edge.
   - Pi cooling: the official cooler blows out sideways. Give the box an intake grille facing the fan (~5 mm gap) and an exhaust high on the opposite side. Check `vcgencmd measure_temp` and `get_throttled` under load in the closed box; the target is <70 °C.
   - The monopod has a 1.5–2" platform, which needs a ~40–45 mm collar around the coupler (or ~50 mm clearance if using side boxes).
   - Cable bend radius ≥ 60 mm (Livox spec).
2. **Optional custom lidar cable:** M12 12-pin A-coded **female** cordset with flying leads (Solnique 1 m, 24 AWG).
   - Map each pin with a meter and confirm against the **official** Mid-360S manual. The pinout seen so far comes from an unofficial copy: power pins 1 & 9, GND 2/3/5, Ethernet 4–7, signals 8/10–12.
   - Crimp the Ethernet wires to RJ45 1/2/3/6 (TX+, TX−, RX+, RX−).
3. **Load test** once everything arrives: lidar + Pi with live GLIM for 15–20 minutes. Watch for bank dropouts and Pi throttling.
4. **Upside-down / harness mode:** possible with no plate changes. Screw the pole into the cross's tapped hole. The side boxes would block the view in that mode, so carry the battery and Pi on the harness.

## Resin printing lessons (Saturn 4 Ultra 16K, Fast gray / Tenacious 75/25 blend)

- **Exposure:** 2.4–2.6 s with 20 s bottom layers.
  - Plates still need a metal scraper to remove.
  - The cross printed cleanly at 2.6 s.
- **Printable area:** the LCD image is 211.7 × 118.4 mm, not the marketing 218.9 × 128.9. `to_goo` now refuses anything larger.
- **Hole sizes:**
  - Vertical holes print ~0.35 mm undersize (diameter).
  - Horizontal holes:
    - A 3.75 mm round hole was too tight.
    - A 3.85 w × 4.25 h ellipse was too loose.
    - Next time try +0.35 plus ~0.15–0.2 extra vertically, and grip where the screw tips actually land.
- **Don't print a big flat plate with delicate parts in the same job.**
  - In v2, the 90 × 90 base plate (~8000 mm² per peel) delaminated a cross arm mid-print.
  - It also left a step on every part when it ended at 6.35 mm.
- **Firmware speed-up:** the printer switches to a faster ~10 s layer cycle after layer 50, whatever the file says.
- **Suspect spot on the plate or film:**
  - The lone v2 standoff at plate X 85…103, Z −56.5…+8.5 mm failed after its first layer. That's centred coordinates, X along the long axis.
  - The cause is unconfirmed; it may have been peel-related.
- **Network control:** `slicer_preview/sdcp.py` drives the printer over WiFi (the USB port is dead).
  - Commands: discover, status, files, upload, `print <file> --yes`, watch.
  - A start can return `Ack 1` right after an upload. Retry after a few seconds.

## Benchmark: NavVis systems vs this rig (researched 2026-10-11)

| System | Lidar | Price | Published accuracy |
|---|---|---|---|
| **This rig** | 1× Livox Mid-360S | **~$2.45k** total (GoPro MAX2 + accessories ~$1k, lidar ~$900, Amazon hardware + Pi ~$500, plates ~$50). A second unit is ~$2.2k thanks to spares. No subscription. | First test: 1.5 cm loop-closure gap after mapping 2 apartment floors including stairs (a drift/consistency number, not accuracy) |
| **NavVis CLX** (handheld, Sep 2026) | 2 sensors, 200k points/s each, 905 nm, 40 m @ 10%. NavVis doesn't name them, but the specs and the photos (tilted dark domes) match **2× Mid-360**. Both tilted outward to cover floor and ceiling. | **$11,700** device, plus **$4,050/yr** IVION (required). "Starting at" ~$17,950 as a package (official) | 8 mm in a 100 m² test area |
| **NavVis MLX** | 1× 32-layer 905 nm, 640k points/s. Brand not published; probably Hesai XT32-class. | ~$30–40k (£30,900 reported at launch). Not official. | — |
| **NavVis VLX 3** (wearable) | 2× Hesai XT32M2X (confirmed), ~1.28M points/s | ~$100k (marketplace listings). The user was quoted $100–150k with software. Not official. | 5 mm in a 500 m² test area |

**Takeaways:**
- The CLX shows a pair of Mid-360-class sensors is commercially viable. The price buys tracking, calibration, the IVION viewer and support.
- CLX cameras: three 12 MP cameras (official spec). From the photos, one points forward and two sit near and slightly behind the side lidar domes, so they're almost certainly fisheyes stitched into a panorama. Our GoPro MAX2 (2 fisheyes, ~29 MP stitched) does the same job; the cage replaces their factory camera-to-lidar calibration. Our GoPro sits ~15–20 cm above the lidar, against their near-coincident cameras, so expect slightly more photo-to-scan misalignment on close objects.
- A dual or tilted Mid-360 is a possible v2 for this rig, once we see where the single upright unit leaves gaps.
- Mid-360 noise is ~2 cm per point, but averaging many points removes most of it. Drift, calibration and range bias don't average out. So 8 mm in one room is plausible without anything special. Holding accuracy over a large site is the hard part.
- **To compare with NavVis's numbers:** measure 5–10 spans with a laser distance meter (room lengths, wall-to-wall, and **floor-to-floor at the stairs**) and compare them with the point cloud. Report the mean and the worst error.

Sources:
- [NavVis CLX order page](https://www.navvis.com/navvis-clx-order)
- [NavVis CLX specs](https://knowledge.navvis.com/docs/navvis-clx-specifications)
- [NavVis pricing](https://www.navvis.com/en/pricing)
- [Hesai/NavVis VLX 3 announcement](https://www.geoweeknews.com/articles/navvis-partners-with-hesai-for-its-new-vlx3-system/)
- [MLX launch price](https://geospatialworld.net/?p=277558)
- [NavVis device comparison matrix](https://knowledge.navvis.com/docs/device-comparison-matrix)
- [VLX 3 marketplace price](https://www.accio.com/plp/navvis-vlx-3-price)
