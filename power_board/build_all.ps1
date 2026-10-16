# Rebuild the power board from the generator scripts: schematic -> netlist -> placement -> hand routes
# -> Freerouting for the signals -> import -> DRC. Run from anywhere: pwsh power_board/build_all.ps1
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$kb = "$env:LOCALAPPDATA\Programs\KiCad\10.0\bin"
$java = (Get-ChildItem "$env:LOCALAPPDATA\Programs\Microsoft\jdk-17*\bin\java.exe" | Select-Object -First 1).FullName
$freerouting = 'D:\Tools\freerouting\freerouting-1.9.0.jar'

python gen_schematic.py
& "$kb\kicad-cli.exe" sch erc -o erc.rpt power_board.kicad_sch | Out-Null
& "$kb\kicad-cli.exe" sch export netlist --format kicadsexpr -o out.net power_board.kicad_sch | Out-Null
& "$kb\python.exe" gen_board.py
& "$kb\python.exe" route_buck.py
& "$kb\python.exe" route_power.py
& "$kb\kicad-cli.exe" pcb drc --refill-zones --save-board -o drc.rpt power_board.kicad_pcb | Out-Null
& "$kb\python.exe" autoroute.py export
if (Test-Path power_board.ses) { Remove-Item power_board.ses }
& $java -jar $freerouting -de power_board.dsn -do power_board.ses -mp 40 -inc NOROUTE -mt 1 2>&1 | Select-String 'completed|Saving|ERROR'
& "$kb\python.exe" autoroute.py import
& "$kb\kicad-cli.exe" pcb drc --schematic-parity --refill-zones --save-board -o drc.rpt power_board.kicad_pcb | Out-Null
"ERC violations: " + (Get-Content erc.rpt | Select-String '^\[').Count
Get-Content drc.rpt | Select-String '^\[' | ForEach-Object { ($_ -split ':')[0] } | Group-Object | Sort-Object Count -Descending | ForEach-Object { "$($_.Count) $($_.Name)" }
