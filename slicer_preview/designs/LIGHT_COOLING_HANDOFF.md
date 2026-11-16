# Omni light: LED, driver and water cooling. Handoff, 2026-11-12

Planning session only (2026-11-11/12). Nothing is designed in code, printed or tested yet. Earlier decisions on the
light (continuous, DeWalt belt pack, low-profile head on the GoPro, optional add-on) are in the docstring of
`sketches/light_cone_stack_render.py` and are not repeated here.

Numbers marked **(est.)** are back-of-envelope estimates from this session, not measurements or datasheet values.

## Next step
First run planned for 2026-11-17 (see "First bench run" below). Then bench test the LED and driver: bench supply -> LDH-65-1750W -> CXB3590, LED on the 69 mm heatsink with a fan.
The test has to answer three things (see "Bench test" below): how dimming behaves above 17.7 V input, whether DIM-
may share a ground with the input, and where the output sags at low input voltage.

## Parts chosen
Purchase status was not recorded: check with the user what has actually been ordered.

| Part | Choice | Source | Notes |
|---|---|---|---|
| LED | Cree CXB3590-0000-000N0HCD50E (36 V, 5000K, CRI 80, bin CD = 12,000 lm min at 2.4 A / 85 °C) | Mouser 941-CXB35900NHCD50E, $39.76 | 34.85 mm square, 30 mm emitting disc, pads on the front, flat ceramic back |
| Driver | Mean Well LDH-65-1750W (boost, constant current, lead-wire style) | Mouser 709-LDH-65-1750W | 8 in stock on 2026-11-11, next batch 2027-01-29: a spare was suggested |
| Bench supply | Jesverty SPS-3010, 30 V / 10 A | Amazon, $47.99 | Stands in for the battery |
| Pump | HiLetgo 2-pack, 12 V, 240 L/h, 3 m head, 4.2 W, brushless (DC30A-1230 class) | Amazon, $13.69 | 8 mm barbs; not self-priming; rated water temperature unchecked |
| Heatsink | 69 × 69 × 36 mm aluminium, 27 fins | Amazon, $7.99 | Bench mount for the LED now, cold end of the loop later |
| Blowers | 5015, 12 V, dual ball bearing, 2-pack, 5 CFM each | Amazon, $16.99 | Watch for 24 V lookalikes |
| LED plate | PATIKIL 6061 aluminium, 4 × 3 × 1/8 inch (3 mm), 3 pieces | Amazon, $8.79 | Chosen over 1.5 mm for flatness and tappable thread |
| Thermal paste | Arctic MX-4, 4 g | Amazon, ~$5 | Not electrically conductive |
| Tubing | Silicone, 6 mm bore / 9 mm outside | not picked | Stretches over the 8 mm pump barbs |

Datasheets are in `C:\Users\Alan\Downloads\`: `XLamp_CXB3590.pdf` and `LDH-65-spec.pdf`.

## LED and driver pairing (from the datasheets)
- LED voltage at 1.75 A, typical part: ~35.5 V at 25 °C case, ~35.1 V at 55 °C. LED power ~62 W.
- Light: ~9,700 lm minimum (est.: 12,000 lm bin × ~81% at 1.75 A and ~40 °C case, read off the relative-flux curve).
- Heat into the plate: ~40 W (est.).
- Battery draw: ~3.7 A at 18 V, ~4.4 A at 15 V (driver efficiency 92.5-96%).
- Driver: input 9.5-32 V, output 12.5-37 V at 1.75 A, 64.75 W, 75 × 53 × 22.7 mm, 159 g. Full load to 50 °C
  ambient at 12-18 V input, 60 °C at 18-32 V.

**Risks found in the datasheets**
1. **Dimming is only specified when output >= 2 × input** (driver note 2). With the LED at ~35.4 V the input must be
   below ~17.7 V. A fresh DeWalt 20V pack is 18-20.5 V, so dimming is out of spec on a fresh pack. Full brightness is
   unaffected. Fallbacks: run at full brightness only, or the 72 V CXB3590 with the LDH-65-700 (~48 W).
2. **37 V output ceiling.** A typical LED clears it by ~1.5 V. Cree allows up to +3 V on forward voltage, so a
   worst-case part (~38.5 V) reaches the driver's over-voltage trip (38-60 V). The trip latches until power is
   cycled; nothing is damaged, but that LED would not run on this driver.
3. **An output short destroys the driver** (datasheet). Running with no LED trips the over-voltage latch. Connect the
   LED before powering, never hot-plug it, and keep no unpluggable connector between driver and LED.

**Dimming input**
- 0-10 V DC or 10 V PWM at 1-3 kHz across DIM+ (blue) and DIM- (white). Open = 100%. Minimum ~8%.
- No resistance mode: a pot works only as a voltage divider from a supply.
- At 0 V the output falls to the input voltage, which is below this LED's turn-on, so the LED goes dark.
- Datasheet: do not connect DIM- to Vo-. It does not say whether DIM- may share a ground with Vin-.

## Electrical layout (rig)
Along the positive wire: DeWalt 20V adapter -> fuse (~10 A, at the adapter) -> pole cord (~18 AWG) -> main switch
-> low-voltage cutoff (~15 V, 10 A, with hysteresis) -> split:
1. driver input -> driver output -> LED;
2. 12 V buck converter (fixed output, ~2-3 A) -> pump, blowers, and the dimming pot through a series resistor
   (about 2.2k above a 10k pot, so the top of the range is ~10 V).

- Over-temperature shutoff: a thermal switch on the plate beside the LED that closes at 60-70 °C and shorts DIM+ to
  DIM-. The LED goes dark and the pump keeps running.
- If DIM- must not share the input ground, feed the pot from a small isolated DC-DC module instead.
- Not chosen yet: buck converter, cutoff module, fuse holder, switch, connectors, thermal switch, pot.
- A custom carrier PCB (JLC) for the small parts is worth doing only after the bench test, and mainly if the light
  goes on several rigs. The Mean Well driver stays a bought module.

## Bench test
- Power: bench supply -> driver input (red/black) -> driver output -> LED. Set the current limit to ~5 A.
- Control: 9 V battery across a 10k pot; wiper to DIM+, battery negative to DIM-. Start with the pot at the low end.
- LED mounted with paste in the middle of the 69 mm heatsink base, held by a clamp on its white border (never on the
  yellow surface). The heatsink can be drilled and tapped M3. A fan is required: without one the heatsink overheats
  within minutes at full power. Expected case temperature with a fan: 55-65 °C (est.).
- Wires soldered to the two front pads (tin pads and wire first, ~20 AWG silicone wire, solder before mounting).
- Measure the input voltage at the driver's wires with a multimeter, not from the supply's display.
- To find out: dimming behaviour from 15 V up to 21 V input; the voltage across DIM+ and DIM- as the pot turns
  (loading); whether DIM- can be tied to Vin-; the input voltage where the output sags.
- Safety: ~10,000 lm from a 30 mm disc. Point it at a wall or into a box.

## First bench run: full brightness, no dimming (plan, 2026-11-16)
In hand: bench supply, driver, LED, 69 mm heatsink, paste. Missing: 9 V battery, 10k pot, a standalone fan (the
5015 blowers can't run: the supply's single output feeds the driver). So this run is full brightness only; the
dimming questions wait for the pot and battery.

**Prep at home (night of 2026-11-16)**
- Solder ~22 AWG stranded leads (enough to keep the driver off to the side) to the LED's two front pads. Tin pads and
  wire first. Stranded, not solid: solid levers on the joints and can lift a pad or crack the ceramic.
- Mark the LED+ lead (red tape or a knot).
- No alligator clips on the LED pads: a slip can short the driver output (destroys it) or hot-plug the LED onto up
  to 60 V; the teeth also sit next to the yellow emitter. The bench supply can't drive the LED directly either
  (30 V max, LED needs ~35.5 V).

**Assembly at the office**
- LED on the heatsink with MX-4, clamped on the white border only. Tape the leads to the heatsink so pulls land on
  the tape, not the pads.
- LED leads to the driver output with WAGO 221 levers (22 AWG is in range; strip ~11 mm, tug test). Red driver
  lead to LED+, black to LED-. Keep the WAGOs off the heatsink. **Make or break a WAGO only with the supply output
  off, then wait a few seconds** (the output cap can hold up to 60 V after an over-voltage trip).
- DIM+ (blue) and DIM- (white) left open = 100 %. Insulate their ends separately.
- Supply to the driver input (red/black) with alligator clips: fine there.

**Setting the supply's current limit** (the A display shows actual draw, so it reads 0 with no load)
1. Nothing connected; voltage ~1-2 V, current knob at minimum.
2. Clip the supply's red and black leads together, output on: CC lights and the A display shows the limit.
3. Turn the current knob to 5.0 A. Output off, unclip, set 15.0 V.
Only short the supply's own leads, never the driver output.

**Eye safety:** ~10,000 lm from a 30 mm disc; never look straight at it (sunglasses don't make staring safe).
Point it at a wall or ceiling ~1 m+ away, or face down onto a light surface ~15 cm+ below with the fins up (good
for convection; don't rest it on the fins). Cardboard shield between you and the emitter, kept ~15-20 cm+ from the
LED face (~20 W of light heats whatever is close). Bounce light is safe. To check it's lit, look at the bounce or
through a phone camera.

**Run**
1. Output off; connect everything; 15.0 V, 5 A limit.
2. Output on. Expect CV mode, ~4.4 A. CC lit means the driver wants more than 5 A: output off and check.
3. Record with a multimeter: input voltage at the driver's wires (not the supply display), input current
   (supply), LED voltage at the pads (~35.5 V expected).
4. **No fan, so bursts of ~30-60 s.** Stop at ~70 °C heatsink base (IR thermometer), or when a finger can't stay
   on the fin edge for 2-3 s. Any desk/USB fan across the fins allows longer runs. Never leave it unattended.
5. Optional: step the input down from 15 V to find where the output sags (raise the limit to ~7 A below 13 V, or
   the supply's own limit will look like a driver sag).

## Water loop (design intent, nothing modelled)
Loop order: pump -> LED block -> heatsink block -> reservoir -> pump. Expected flow 1.2-1.5 L/min (est., from a
straight-line pump curve with 6 mm bore tubing).

**LED block (head)**
- Stack, top to bottom: clamp frame on the LED border -> LED (soldered leads) -> 3 mm aluminium plate with paste ->
  O-ring -> printed block. Screws pass through frame and plate, outside the O-ring, into brass inserts in the block.
- Single centre jet aimed at the plate behind the LED: 3 mm bore, 5 mm or longer, chamfered entry, 6-9 mm from the
  plate. ~3 m/s, turbulent, under 0.1 bar across the nozzle (est.). The chamber is flooded.
- Outlet high on the chamber so air is carried out. No bleed port on the block to start with; add a bleed screw
  opposite the outlet only if the bench shows trapped air (the rig works tilted 30-45°).
- Printed barbs: ~8 mm outside, 5 mm bore, filleted root. Clamp frame should be low; metal tabs are safer than resin
  next to the emitter.
- Plate above water: ~6 K (est.). Water warms under 1 K per pass.

**Cold end (on the Pi box +Y face)**
- No separate radiator. The 69 mm heatsink's flat base is the plate; a printed block seals to it with an O-ring.
- Not a jet here: a 5 mm centre inlet feeding a 1-1.5 mm flooded gap over the whole base, with printed bumps to hold
  the gap and a perimeter groove to the outlet.
- The same print carries the reservoir (fill plug with O-ring at the top, small air pocket, return entering near the
  top) and the pump mount (pump at the bottom, below the reservoir, always flooded).
- Air: two 5015 blowers side by side at one end, a short tapered duct to the 69 × 36 mm fin opening, and a cover over
  the fin tips. Leave the blower intakes and the far-end exhaust clear.
- Expected water temperature: ~30 K above ambient at full brightness (est.; ~5 CFM delivered through the fins).
- Fallback if too hot: a 70 mm axial fan on the fin tips (~20 K above ambient, est.; adds ~15 mm depth), or dim.

**Rig envelope (from the render script; box extents approximate)**
- Pi box +Y face ~92 mm square. Total ~305 mm from box bottom to LED.
- Nothing may rise above the box top on that face: the lidar sees from 7° below to 52° above horizontal.

## Considered and dropped
- Chip-Lok COB holders (Ideal 50-2303CR, 65 mm; 50-2300AN, 50 mm): wider than the 39 mm head, and they still need
  something to screw into. Soldering was chosen.
- CPU tower cooler for the bench: replaced by the 69 mm heatsink, which can be drilled.
- 120 mm radiator (about 120 × 155 × 27 mm): overhangs the 92 mm face; 80/92 mm radiators would fit but need an air
  gap behind them. The heatsink and printed block were chosen for a flat fit.
- Ready-made aluminium water blocks (40 × 40 × 12 mm under the LED, 40 × 120 × 12 mm at the cold end): the user
  prefers printing the blocks.
- 70 mm axial fan on the fin tips: more airflow, but the blower layout was chosen for the footprint.
- Alphacool DC-LT 2600 pump: 0.6 m head, too weak for the jet.
- 1.5 mm plate: bows more easily and holds only ~3 turns of M3 thread.

## Open items
- Run the bench test and settle the dimming question.
- Confirm the pump's rated water temperature (~55 °C water is expected in a 25 °C tunnel).
- Choose the remaining electrical parts (buck converter, cutoff, fuse, switch, connectors, thermal switch, pot).
- Design the two printed blocks, the duct and the clamp frame; update the render script's cooling module
  (it still shows a 60 mm radiator, a copper plate and a 28 mm driver column, all superseded).
- Read the Cree datasheet's mechanical pages (17 onward, not read this session) for pad positions and the clamp
  keep-out before designing the frame.
- Still unresolved from before: shadows from the light in the tracking cameras' view.
