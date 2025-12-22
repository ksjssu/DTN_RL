param(
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 5000,
    [string]$Scenario = "scenario_prophet_buf05_100k.txt",
    [string]$Overlay = "rl_training_overlay.txt",
    [string]$Runtime = "runtime_100k_rng1_100.txt",
    [string]$Baseline = "default_settings.txt",
    [int]$BatchRuns = 100
)

$ErrorActionPreference = "Stop"

Write-Host "[RL] Starting DRL server for 5M buffer training" -ForegroundColor Cyan
$env:TRAIN_UNTIL_SECONDS = "100000"
$env:SAVE_AT_SECONDS = "0"
$env:MODEL_DIR = "models/buf05"
$env:MODEL_PATH = ""
$env:EVAL_ONLY = "false"

Start-Process -FilePath "python" -ArgumentList "toolkit/drl_server.py" -WindowStyle Minimized
Start-Sleep -Seconds 5

Write-Host "[RL] Launching simulation batch: 5M buffer, seeds 1-100" -ForegroundColor Cyan
$scenarioPath = Resolve-Path $Scenario
$overlayPath = Resolve-Path $Overlay
$runtimePath = Resolve-Path $Runtime
$baselinePath = Resolve-Path $Baseline

& .\one.bat -b $BatchRuns $baselinePath $runtimePath $overlayPath $scenarioPath

Write-Host "[RL] Training run completed. Stop DRL server manually when finished." -ForegroundColor Green
