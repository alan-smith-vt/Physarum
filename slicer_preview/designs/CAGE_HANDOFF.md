# Lidar photo-rig cage handoff (2026-10-11)

This update follows "Lidar photo-rig handoff.md" (Oct 7). It records what was built, tested, decided and ordered on 2026-10-11. Where the two disagree, this file wins.

## Status

- **Resin mock:** assembled and fit-checked against the real Mid-360S. The base plate came out perfect: the M3 pattern and the locating nubs (in place of pins) line up, the lidar sits flat, the cross clears the dome, and the M12 connector sits between the posts.
- **Aluminum plates:** ready to order from SendCutSend with the DXFs written by `designs/cage_dxf.py` (`python designs/cage_dxf.py` → `out/`).
  - **Base plate:** 6061-T6, 1/4", 90 × 90 mm, R5 corners. No services.
    - SendCutSend won't countersink parts under 1" × 4", so countersink the centre Ø6.6 hole by hand with an 82° bit (5/8" or 3/4"). Stop when a 1/4-20 flat head sits flush.
  - **Cross top:** 6061-T6, 3/16", 92 × 92 mm X-shape. Have SendCutSend **tap the centre hole 1/4-20** (drawn Ø5.1).
- **Hardware:** on order (Amazon), listed below.

## SendCutSend DXF rejection and fix (2026-10-12)

- **What happened:** SendCutSend flagged `cage_cross_top.dxf`. It had "a lot of nodes instead of smooth arcs": `cage_dxf.py` built the outline from shapely buffers, so every fillet and pad was a polyline with 64 points per quarter circle.
- **Script fix:** `cage_dxf.py` now draws the outlines analytically as separate LINE, ARC and CIRCLE entities.
  - Base plate: 4 lines, 4 arcs, 11 circles.
  - Cross: 8 lines, 24 arcs, 5 circles.
  - Each run checks the outline's area and shape against the shapely version.
  - Commits `0bd0ba1` and `0b09e3b` on Physarum main.
  - Future cut files get drawn directly by the script this way.
- **Cross-check in Onshape:** both plates were also drawn as Onshape sketches (lines, arcs and circles joined end to end) and extruded to 1/4" and 3/16".
  - The part volumes matched the script to 0.1 mm³, and the exported DXFs have the same entity counts as the script's.
  - The document is "Lidar cage plates (SendCutSend)": https://cad.onshape.com/documents/45245c7c39d14768d2d10a8e. It's **public**, because the account is a free one.
  - The cross sits 120 mm off origin, which doesn't affect the DXF.
  - This was a one-time check, not the source of truth.
- **Order state:**
  - **Cross:** re-uploaded with the Onshape DXF and accepted as 92 × 92 mm, 6061-T6, .187". It still needs the 1/4-20 tapping service checked on the order.
  - **Base plate:** left as originally uploaded. Its DXF is faceted only at the four R5 corners, and SendCutSend didn't flag it.

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

## GoPro mount: CAMVATE for testing, slider clamp for production

**The aluminum cross DXF has no pin hole** (it was dropped), and the centre hole is tapped 1/4-20. Drilling the GoPro cage for a pin won't work, because the cage has to turn as it screws onto the stud.

- **Testing:** use the CAMVATE locking adapter.
  - Red Loctite on the cross thread.
  - After screwing the cage on, lock its heading with the upper wheel.
  - The heading can shift when you remount, so either leave the GoPro on between sessions or re-run the targetless GoPro↔lidar calibration on each session's data.
- **Production:** a low-profile Arca-Swiss slider clamp, so the GoPro snaps on at the same heading every time.
  1. Thread a **1/4-20 set screw** into the tapped cross with **red Loctite**, flush with or ≤4 mm below the underside (the dome clearance is ~5.9 mm). Let it cure ~24 h.
     - 3/8" is the default length; 1/2" if the clamp's thread is deep.
     - It must not bottom out in the clamp before the clamp seats on the cross.
  2. Screw a **mini clamp with a female 1/4-20 base and a screw-knob lock** (not a lever; aim for a ~12–15 mm low-profile body) down tight onto the cross, with red Loctite. Paint a witness mark across clamp and cross.
     - Its angle is wherever it stops tightening. That's fine, because calibration measures it.
     - Put a thin washer under it if you need to move where the stitch seams land.
  3. Fix an **Arca plate with an anti-twist lip/pad** permanently to the GoPro cage (Loctite).
     - The cage and plate then stay together as one unit and click in at a repeatable heading.
     - The GoPro↔lidar calibration holds between sessions.
- This approach was chosen over the CAMVATE + clamp stack (~30–35 mm tall against ~15–20 mm), because a taller stack means more leverage and flex in a knock and a bigger camera-to-lidar offset.
- Search: "Arca Swiss mini quick release clamp knob low profile" and "Arca Swiss quick release plate 38mm anti-twist".

## Calibration (no surveyed room needed)

| Pair | Method |
|---|---|
| Lidar ↔ its IMU | From Livox (manual/SDK) |
| Tracking camera lens models | Kalibr/OpenCV with a fisheye (Kannala-Brandt) model, by waving an AprilGrid ~10 min per camera |
| Tracking cameras ↔ IMU, including time offset | Kalibr camera-IMU: AprilGrid on a wall, move the rig through all axes for ~1 min |
| GoPro (and tracking cameras) ↔ lidar | Targetless: `direct_visual_lidar_calibration` (from GLIM's author; supports 360° cameras), run on any textured scene |

- Check the result by colouring the cloud with the GoPro photos (edges should line up) and with AprilTag cross-checks.
- Surveyed points are only needed to report absolute accuracy or to georeference.

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

## First validation job: subway station alongside a NavVis MLX scan

Another contractor will scan the same station with a NavVis MLX. Plan for a direct, fair comparison:

- **Targets:**
  - Use our **20 matte laminated AprilTags** (already in the GoPro kit), each paired with a new **letter-size (8.5×11) 2×2 black/white checkerboard**, laminated matte and printed solid matte black. Number each pair (T01…T20) with the ID large in a corner.
  - The checkerboard centre is the precise point both lidars resolve from intensity. The AprilTag gives our cameras the ID and pose.
  - Mount flat and rigid, with magnets or removable tape (check the station's rules). Targets must not move between the two scans.
- **Placement** (a long, repetitive platform is the worst case for drift):
  - Both ends, then every 15–20 m along the platform, alternating walls.
  - Every level (platform, mezzanine, concourse) and both ends of each stair or escalator.
  - About 8–15 locations; the rest are spares.
- **Mid-360 limit:** each square is 10.8 cm, so walk within **~3–5 m** of every target and pause 1–2 s. Beyond ~8 m the centre isn't well defined. The MLX resolves letter size easily.
- **Hold-out:**
  - Use about half the targets as check points that our processing never sees.
  - Process twice: with **no targets**, then with **only the control half**.
  - Ask whether the MLX contractor uses control points, so the comparison is like-for-like.
- **Independent ground truth:**
  - 5–10 laser-distance-meter spans between targets, including the full platform length and floor-to-floor at a stair.
  - Surveyed station control coordinates, if the transit authority has them.
- **Coordination:**
  - Put the targets up before both scans (same day if possible).
  - Get their point cloud as E57/LAS, plus their accuracy report.
  - Photograph each target in place and keep a target sheet.
- **Comparison:**
  - Target-to-target distances (ours vs MLX vs laser).
  - A cloud-to-cloud deviation map after best-fit alignment (CloudCompare M3C2), to show where drift builds up.
  - End-to-end platform length and floor-to-floor heights.
- **Future use:** on jobs that do hire a surveyor, have them survey a few of our tags/targets. That pins our dense data and photos to survey control and gives a free accuracy check each time.

## Logger software plan (decided 2026-10-11)

- **The Pi always records raw data**: lidar points, the lidar's IMU, GoPro shutter times, and any tracking cameras added later. The final result is computed offline on the PC from these recordings at full quality (GLIM with loop closure, AprilTags, the impact filter, and camera fusion if cameras get added).
- **The live preview is only a coverage check.** It can drift, lag or crash without any data being lost.
  - Run lightweight lidar odometry on the Pi: KISS-ICP, FAST-LIO2 with 10–20 cm downsampling, or GLIM's lightweight mode.
  - Stream a top-down 2D map to the phone at ~1 Hz over the Pi's WiFi hotspot: the floor plan from the scans, the walked path, and dots where GoPro photos were taken. Shading by scan count is optional.
  - This takes away the open risk of whether full GLIM can run live on a Pi 5.
- **Time sync:** the Pi is the master clock.
  - Lidar and its IMU: PTP over Ethernet (`ptp4l`; the Pi 5 timestamps packets in hardware). The lidar's PPS/GPS signal wires are only needed for GPS time.
  - Tracking cameras: frames are timestamped on the Pi's clock. For best results, hardware-trigger them from a GPIO pin.
- **Tracking cameras ORDERED 2026-10-12:** a udevqal 2-pack of OV9281 1 MP mono global-shutter cameras with **160°** lenses (CSI; 15→22-pin cables included, 150 mm long).
  - They must mount **rigidly on the cage/base plate**, never on the pole. Use a fisheye (Kannala-Brandt) calibration model, and mask out the parts of the rig in view.
  - The short CSI cables push toward a **split layout: Pi box under the base plate at the top**, with short lidar/camera runs and the lidar cable coiled in the box, and the **battery at the bottom of the pole** as counterweight, sending only a thin 15 V two-wire lead up the pole to the WAGOs.
- **Tracking camera details**, originally meant to be bought only if corridors or stairwells drift: 1–2 OV9281 global-shutter fisheye cameras (~$30–60 each) on the Pi 5's two camera connectors (22-pin; 15-to-22 adapters are often needed).
  - They're recorded only, not used live. Save them as JPEG at about 640×400 grayscale, 20 fps, roughly 1–3 MB/s. The Pi 5 has no hardware video encoder.
  - Cost: +1–3 W, so roughly 3.5 h of runtime instead of 4 on the A1383.
  - Facing opposite ways matches the MLX's layout.

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
- **The 75/25 blend is very ductile.** We bent the ~1–1.5 mm-thick slab that split off the v2 cross arm (fully cured, sheet in the layer plane):
  - It went **180° without breaking**, with a ~7 mm gap at the bend (inner radius ≈ 3.5 mm), which is **~15% strain at the outer surface**.
  - It sprang back to ~90°, so it permanently deformed.
  - When the strip was bent again later, it **snapped**. One big yield uses up the material's toughness (internal crazing/damage).
  - Good for snap-fits, clips, cable clamps and impact-tolerant brackets.
  - **v2 cross-arm break test** (an intact arm of the v2 cross, the one printed at 2.4 s with the split arm) (the 4.8 mm-thick arm bent as a ~42 mm cantilever, base held):
    - It sprang back fully from ~25 mm at the pad, which is **~10% surface strain, elastic**.
    - It broke suddenly at ~50–60° arm rotation: a clean crack on one face and a ~5 mm chipped band on the other (most likely a compression curl, i.e. the chip is the compression side).
    - It broke through the bulk, **not between layers**. So even the 2.4 s layer bond holds up in bending; the v2 split was caused by peel forces during that print, not by a weak cure.
    - Thick sections loaded fast break with little warning; thin, slowly bent ones yield first.
  - **Design rule:** keep flexing parts in the elastic range (≲5–8% surface strain, leaving margin below the ~10% that recovered). No living hinges, and nothing that relies on bending past yield and still working afterwards.
  - The flip side is that it's soft. Expect creep under sustained load, which is why resin must stay out of any clamped joint that holds the calibration.
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
- MLX cameras (official datasheet): 4 × 12 MP fisheye panoramic cameras (2.14 mm, rolling shutter; 360° held overhead, 270° on the harness) plus **2 × 2 MP global-shutter fisheye visual-odometry cameras** (1.34 mm) that help the lidar tracking. A possible upgrade if GLIM drifts in featureless corridors: a ~$30–60 OV9281 global-shutter camera on the Pi. Our GoPro is rolling-shutter and shoots stills only, so it can't fill that role.
- A dual or tilted Mid-360 is a possible v2 for this rig, once we see where the single upright unit leaves gaps.
- Mid-360 noise is ~2 cm per point, but averaging many points removes most of it. Drift, calibration and range bias don't average out. So 8 mm in one room is plausible without anything special. Holding accuracy over a large site is the hard part.
- **To compare with NavVis's numbers:** measure 5–10 spans with a laser distance meter (room lengths, wall-to-wall, and **floor-to-floor at the stairs**) and compare them with the point cloud. Report the mean and the worst error.

Sources:
- [NavVis CLX order page](https://www.navvis.com/navvis-clx-order)
- [NavVis CLX specs](https://knowledge.navvis.com/docs/navvis-clx-specifications)
- [NavVis pricing](https://www.navvis.com/en/pricing)
- [Hesai/NavVis VLX 3 announcement](https://www.geoweeknews.com/articles/navvis-partners-with-hesai-for-its-new-vlx3-system/)
- [MLX launch price](https://geospatialworld.net/?p=277558)
- [NavVis MLX specs](https://knowledge.navvis.com/docs/navvis-mlx-specifications.md)
- [Geo Week News: MLX launch](https://www.geoweeknews.com/articles/navvis-handheld-laser-scanner-mlx-mobile-mapping-slam-lidar/)
- [NavVis device comparison matrix](https://knowledge.navvis.com/docs/device-comparison-matrix)
- [VLX 3 marketplace price](https://www.accio.com/plp/navvis-vlx-3-price)
