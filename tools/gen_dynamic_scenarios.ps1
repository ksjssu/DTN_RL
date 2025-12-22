$ErrorActionPreference = "Stop"

# Config
$BaseFile = Join-Path $PSScriptRoot "..\dynamic_traffic_100k_full_settings.txt"
$OutRoot = Join-Path $PSScriptRoot "..\scenarios\dynamic"

if (-not (Test-Path $BaseFile)) {
  Write-Error "Base settings not found: $BaseFile"
}

# Ensure output folders
$null = New-Item -ItemType Directory -Force -Path $OutRoot | Out-Null
$null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot 'prophet') | Out-Null
foreach ($d in 'heuristic_d010','heuristic_d020','heuristic_d030') {
  $null = New-Item -ItemType Directory -Force -Path (Join-Path $OutRoot $d) | Out-Null
}

$base = Get-Content -Path $BaseFile -Raw -Encoding UTF8

function New-ScenarioText {
  param(
    [double]$HeuristicDelta,
    [string]$BufferLabel, # e.g. 05M, 10M ... 50M
    [string]$ScenarioName,
    [string]$ReportDir
  )

  $txt = $base

  # Normalize line endings
  $txt = $txt -replace "\r\n","`n"

  # Strip existing headers we will control and any report blocks/RL lines
  $patternsToRemove = @(
    '^\s*Scenario\.name\s*=.*$',
    '^\s*Scenario\.endTime\s*=.*$',
    '^\s*MovementModel\.rngSeed\s*=.*$',
    '^\s*Group\.router\s*=.*$',
    '^\s*Group\.bufferSize\s*=.*$',
    '^\s*Report\.nrofReports\s*=.*$',
    '^\s*Report\.report\d+\s*=.*$',
    '^\s*Report\.reportDir\s*=.*$',
    '^\s*RLBridgeReport\..*$',
    '^\s*RLStateReport\..*$',
    '^\s*NodeContactBufferWindowReport\..*$',
    '^\s*DeliveredMessagesReport\..*$',
    '^\s*MessageStatsReport\..*$',
    '^\s*ContactTimesReport\..*$',
    '^\s*StepBinMetricsReport\..*$'
  )
  foreach ($pat in $patternsToRemove) {
    $txt = (($txt -split "`n") | Where-Object { $_ -notmatch $pat }) -join "`n"
  }

  # Prepend header with Scenario.name first, then RNG seeds, then end time
  $seedList = (1..10) -join '; '
  $header = @(
    ("Scenario.name = " + $ScenarioName),
    ("MovementModel.rngSeed = [" + $seedList + "]"),
    'Scenario.endTime = 100000'
  ) -join "`n"
  $txt = $header + "`n" + $txt

  # Router and buffer (append once)
  $txt = $txt + "`nGroup.router = ProphetRouter" + ("`nGroup.bufferSize = " + $BufferLabel)

  # HeuristicDelta (0.0 for pure Prophet)
  if ($txt -match '^\s*ProphetRouter\.heuristicDelta\s*=') {
    $txt = $txt -replace '(?m)^\s*ProphetRouter\.heuristicDelta\s*=.*$',("ProphetRouter.heuristicDelta = " + $HeuristicDelta.ToString('0.0'))
  } else {
    $txt = $txt + ("`nProphetRouter.heuristicDelta = " + $HeuristicDelta.ToString('0.0'))
  }

  # Reports: DeliveryRate + BufferLoadHeuristic (no overlays)
  $reportBlock = @(
    "Report.reportDir = $ReportDir",
    "Report.nrofReports = 2",
    "Report.report1 = DeliveryRatePerIntervalReport",
    "Report.report2 = BufferLoadHeuristicReport",
    "DeliveryRatePerIntervalReport.binSize = 3600",
    "DeliveryRatePerIntervalReport.cumulativeMode = true",
    "BufferLoadHeuristicReport.sampleInterval = 20",
    "BufferLoadHeuristicReport.lowThreshold = 0.33",
    "BufferLoadHeuristicReport.highThreshold = 0.66",
    "BufferLoadHeuristicReport.bufOccMaxAge = 600",
    "BufferLoadHeuristicReport.logActions = false",
    "BufferLoadHeuristicReport.output = NUL"
  ) -join "`n"

  $txt = $txt + "`n`n" + $reportBlock + "`n"
  return $txt
}

function New-DrlScenarioText {
  param(
    [string]$BufferLabel,
    [string]$ScenarioName,
    [string]$ReportDir,
    [int]$Port
  )

  $txt = $base
  $txt = $txt -replace "\r\n","`n"
  $patternsToRemove = @(
    '^\s*Scenario\.name\s*=.*$',
    '^\s*Scenario\.endTime\s*=.*$',
    '^\s*MovementModel\.rngSeed\s*=.*$',
    '^\s*Group\.router\s*=.*$',
    '^\s*Group\.bufferSize\s*=.*$',
    '^\s*Report\.nrofReports\s*=.*$',
    '^\s*Report\.report\d+\s*=.*$',
    '^\s*Report\.reportDir\s*=.*$',
    '^\s*RLBridgeReport\..*$',
    '^\s*RLStateReport\..*$',
    '^\s*NodeContactBufferWindowReport\..*$',
    '^\s*DeliveredMessagesReport\..*$',
    '^\s*MessageStatsReport\..*$',
    '^\s*ContactTimesReport\..*$',
    '^\s*StepBinMetricsReport\..*$'
  )
  foreach ($pat in $patternsToRemove) { $txt = (($txt -split "`n") | Where-Object { $_ -notmatch $pat }) -join "`n" }

  $seedList = (1..10) -join '; '
  $header = @(
    ("Scenario.name = " + $ScenarioName),
    ("MovementModel.rngSeed = [" + $seedList + "]"),
    'Scenario.endTime = 100000'
  ) -join "`n"
  $txt = $header + "`n" + $txt

  # Router/buffer and heuristic off
  $txt = $txt + "`nGroup.router = ProphetRouter" + ("`nGroup.bufferSize = " + $BufferLabel) + "`nProphetRouter.heuristicDelta = 0.0"

  $reportBlock = @(
    "Report.reportDir = $ReportDir",
    "Report.nrofReports = 3",
    "Report.report1 = DeliveryRatePerIntervalReport",
    "Report.report2 = RLBridgeReport",
    "Report.report3 = RLStateReport",
    "DeliveryRatePerIntervalReport.binSize = 3600",
    "DeliveryRatePerIntervalReport.cumulativeMode = true",
    "RLBridgeReport.sampleInterval = 20",
    "RLBridgeReport.windowSize = 600",
    "RLBridgeReport.contactsNormMode = cmax",
    "RLBridgeReport.contactsCmax = 10",
    "RLBridgeReport.maxMessagesPerNode = 20",
    "RLBridgeReport.deltaLimit = 0.6",
    "RLBridgeReport.bufOccMaxAge = 600",
    "RLBridgeReport.logActions = false",
    ("RLBridgeReport.url = http://127.0.0.1:" + $Port + "/infer_and_update"),
    "RLStateReport.sampleInterval = 20",
    "RLStateReport.windowSize = 600",
    "RLStateReport.contactsNormMode = cmax",
    "RLStateReport.contactsCmax = 10",
    "RLStateReport.maxMessagesPerNode = 20",
    "RLStateReport.wRelay = 5.0",
    "RLStateReport.wDrop = 8.0",
    "RLStateReport.wAbort = 0.5"
  ) -join "`n"

  return $txt + "`n`n" + $reportBlock + "`n"
}

function Write-Scenario {
  param(
    [double]$Delta,
    [int]$BufferMib,
    [string]$OutPath,
    [string]$ScenarioPrefix,
    [string]$ReportDir
  )
  $bufLabel = "{0}M" -f $BufferMib
  $name = "{0}_buf{1}_100k_rng%%MovementModel.rngSeed%%" -f $ScenarioPrefix, ("{0:D2}" -f $BufferMib)
  $text = New-ScenarioText -HeuristicDelta $Delta -BufferLabel $bufLabel -ScenarioName $name -ReportDir $ReportDir
  $outFile = Join-Path $OutPath ("{0}_buf{1}_100k.txt" -f $ScenarioPrefix, ("{0:D2}" -f $BufferMib))
  $outDir = Split-Path $outFile -Parent
  $null = New-Item -ItemType Directory -Force -Path $outDir | Out-Null
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($outFile, $text, $utf8NoBom)
  Write-Host ("Wrote: " + $outFile)
}

# Generate Prophet (delta=0.0)
foreach ($buf in 5..50 | Where-Object { ($_ % 5) -eq 0 }) {
  $outPath = Join-Path $OutRoot 'prophet'
  $reportDir = "reports/prophet_buf{0:D2}/" -f $buf
  Write-Scenario -Delta 0.0 -BufferMib $buf -OutPath $outPath -ScenarioPrefix 'prophet' -ReportDir $reportDir
}

# Generate Heuristic deltas
$heuristics = @(
  @{ key='heuristic_d010'; delta=0.1 },
  @{ key='heuristic_d020'; delta=0.2 },
  @{ key='heuristic_d030'; delta=0.3 }
)
foreach ($h in $heuristics) {
  foreach ($buf in 5..50 | Where-Object { ($_ % 5) -eq 0 }) {
    $outPath = Join-Path $OutRoot $h.key
    $reportDir = "reports/{0}_buf{1:D2}/" -f $h.key, $buf
    Write-Scenario -Delta $h.delta -BufferMib $buf -OutPath $outPath -ScenarioPrefix $h.key -ReportDir $reportDir
  }
}

# Generate DRL train/eval scenarios
function BufToPort([int]$buf) {
  switch ($buf) {
    5 { return 5005 }
    10 { return 5010 }
    15 { return 5015 }
    20 { return 5020 }
    25 { return 5025 }
    30 { return 5030 }
    35 { return 5035 }
    40 { return 6040 }
    45 { return 5045 }
    50 { return 5050 }
    default { return (5000 + $buf) }
  }
}

$outTrain = Join-Path $OutRoot 'drl_train'
$outEval = Join-Path $OutRoot 'drl_eval'
$null = New-Item -ItemType Directory -Force -Path $outTrain | Out-Null
$null = New-Item -ItemType Directory -Force -Path $outEval | Out-Null

foreach ($buf in 5..50 | Where-Object { ($_ % 5) -eq 0 }) {
  $bufLabel = "{0}M" -f $buf
  $port = BufToPort $buf
  # Train
  $nameT = "drl_train_buf{0}_100k_rng%%MovementModel.rngSeed%%" -f ("{0:D2}" -f $buf)
  $textT = New-DrlScenarioText -BufferLabel $bufLabel -ScenarioName $nameT -ReportDir ("reports_drl_train/buf{0:D2}/" -f $buf) -Port $port
  $outFileT = Join-Path $outTrain ("drl_train_buf{0}_100k.txt" -f ("{0:D2}" -f $buf))
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($outFileT, $textT, $utf8NoBom)
  Write-Host ("Wrote: " + $outFileT)
  # Eval
  $nameE = "drl_eval_buf{0}_100k_rng%%MovementModel.rngSeed%%" -f ("{0:D2}" -f $buf)
  $textE = New-DrlScenarioText -BufferLabel $bufLabel -ScenarioName $nameE -ReportDir ("reports_drl_eval/buf{0:D2}/" -f $buf) -Port $port
  $outFileE = Join-Path $outEval ("drl_eval_buf{0}_100k.txt" -f ("{0:D2}" -f $buf))
  $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
  [IO.File]::WriteAllText($outFileE, $textE, $utf8NoBom)
  Write-Host ("Wrote: " + $outFileE)
}

Write-Host "Done. Use: one.bat -b 100 <scenario-file> (no overlays)" -ForegroundColor Green
