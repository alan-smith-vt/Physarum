# Manufacturing outputs for JLCPCB: Gerbers + drill (zipped), position (CPL) file, renders.
# BOM comes from make_bom.py (LCSC codes live there). Output: power_board/fab/
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$k = "$env:LOCALAPPDATA\Programs\KiCad\10.0\bin\kicad-cli.exe"
$out = Join-Path $PSScriptRoot 'fab'
if (Test-Path $out) { Remove-Item $out -Recurse -Force }
New-Item -ItemType Directory -Force "$out\gerbers" | Out-Null

& $k pcb export gerbers --layers 'F.Cu,In1.Cu,In2.Cu,B.Cu,F.Paste,F.Silkscreen,B.Silkscreen,F.Mask,B.Mask,Edge.Cuts' `
    --subtract-soldermask --use-drill-file-origin -o "$out\gerbers" power_board.kicad_pcb | Out-Null
& $k pcb export drill --format excellon --excellon-separate-th --generate-map --map-format gerberx2 `
    -o "$out\gerbers" power_board.kicad_pcb | Out-Null
Compress-Archive -Path "$out\gerbers\*" -DestinationPath "$out\power_board_gerbers.zip"
& $k pcb export pos --side front --format csv --units mm -o "$out\power_board_pos_raw.csv" power_board.kicad_pcb | Out-Null
& $k pcb render -o "$out\render_top.png" --side top -w 1300 -h 1650 --quality high power_board.kicad_pcb | Out-Null
& $k sch export pdf -o "$out\power_board_schematic.pdf" power_board.kicad_sch | Out-Null
Get-ChildItem $out | ForEach-Object { "$($_.Name)  $($_.Length)" }
