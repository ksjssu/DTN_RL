param(
    [string]$ServerHost = "127.0.0.1",
    [int]$ServerPort = 5000,
    [string]$Scenario = "scenario_prophet_buf05_100k.txt",
    [string]$Overlay = "rl_eval_overlay.txt",
    [string]$Runtime = "runtime_100k_rng1_100.txt",
    [string]$Baseline = "default_settings.txt",
    [int]$BatchRuns = 100,
    [string]$ModelPath = "models/buf05/final_ppo.pt"
)
$ErrorActionPreference = "Stop"
Write-Host "[RL] Starting DRL server in evaluation mode" -ForegroundColor Cyan
$env:EVAL_ONLY = "true"
$env:MODEL_PATH = $ModelPath
Start-Process -FilePath "python" -ArgumentList "toolkit/drl_server.py" -WindowStyle Minimized
Start-Sleep -Seconds 5
Write-Host "[RL] Launching evaluation batch" -ForegroundColor Cyan
$scenarioPath = Resolve-Path $Scenario
$overlayPath = Resolve-Path $Overlay
$runtimePath = Resolve-Path $Runtime
$baselinePath = Resolve-Path $Baseline
& .\one.bat -b $BatchRuns $baselinePath $runtimePath $overlayPath $scenarioPath
Write-Host "[RL] Evaluation run completed." -ForegroundColor Green
