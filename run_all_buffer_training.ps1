# DRL 버퍼 사이즈별 학습 스크립트
# 사용법: .\run_all_buffer_training.ps1 -BufferSizes @("05", "10", "50") -StartSeed 1 -EndSeed 30

param(
    [string[]]$BufferSizes = @("05", "10", "15", "20", "25", "30", "35", "40", "45", "50"),
    [int]$StartSeed = 1,
    [int]$EndSeed = 30,
    [string]$Mode = "training"  # "training" or "evaluation"
)

# 로그 디렉토리 생성
if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs"
}

foreach ($bufSize in $BufferSizes) {
    $bufSizeMB = "${bufSize}M"
    Write-Host "=== Starting DRL $Mode for ${bufSizeMB}B buffer ===" -ForegroundColor Green

    # 포트 설정 (충돌 방지)
    $port = switch ($bufSize) {
        "05" { "5005" }
        "10" { "5010" }
        "15" { "5015" }
        "20" { "5020" }
        "25" { "5025" }
        "30" { "5030" }
        "35" { "5035" }
        "40" { "5040" }
        "45" { "5045" }
        "50" { "5050" }
        default { "5000" }
    }

    # 모델 디렉토리 생성
    $modelDir = "models/buf${bufSize}"
    if (!(Test-Path $modelDir)) {
        New-Item -ItemType Directory -Path $modelDir -Force
        Write-Host "Created model directory: $modelDir" -ForegroundColor Yellow
    }

    # 환경 변수 설정
    $env:DRL_POLICY = "ppo"
    $env:DRL_PORT = $port
    $env:MODEL_DIR = $modelDir

    if ($Mode -eq "training") {
        $env:TRAIN_UNTIL_SECONDS = "100000"
        Remove-Item Env:EVAL_ONLY -ErrorAction SilentlyContinue
        $configFile = "scenario_prophet_buf${bufSize}_train_100k.txt"
        $overlayFile = "rl_training_overlay_buf${bufSize}.txt"
        $runtimeFile = "runtime_100k_rng1_30.txt"
        $seedRange = "${StartSeed}:${EndSeed}"
    } else {
        $env:EVAL_ONLY = "true"
        $env:MODEL_PATH = "${modelDir}/ppo_model.pt"
        Remove-Item Env:TRAIN_UNTIL_SECONDS -ErrorAction SilentlyContinue
        $configFile = "scenario_prophet_buf${bufSize}_eval_100k.txt"
        $overlayFile = "rl_eval_overlay_buf${bufSize}.txt"
        $runtimeFile = "runtime_100k_rng1_100.txt"
        $seedRange = "1:100"
    }

    Write-Host "Port: $port, Model Dir: $modelDir" -ForegroundColor Cyan
    Write-Host "Config: $configFile" -ForegroundColor Cyan
    Write-Host "Overlay: $overlayFile" -ForegroundColor Cyan

    # DRL 서버 백그라운드 시작
    $logFile = "logs/drl_server_buf${bufSize}_${Mode}.log"
    Write-Host "Starting DRL server (port $port)..." -ForegroundColor Yellow

    $serverJob = Start-Job -ScriptBlock {
        param($port, $bufSize, $modelDir, $mode, $logFile)

        $env:DRL_POLICY = "ppo"
        $env:DRL_PORT = $port
        $env:MODEL_DIR = $modelDir

        if ($mode -eq "training") {
            $env:TRAIN_UNTIL_SECONDS = "100000"
        } else {
            $env:EVAL_ONLY = "true"
            $env:MODEL_PATH = "${modelDir}/ppo_model.pt"
        }

        # 로그 파일로 출력 리다이렉트
        python toolkit/drl_server.py 2>&1 | Tee-Object -FilePath $logFile

    } -ArgumentList $port, $bufSize, $modelDir, $Mode, $logFile

    # 서버 시작 대기 및 확인
    Write-Host "Waiting for DRL server to start..." -ForegroundColor Yellow
    Start-Sleep -Seconds 15

    # 서버 상태 확인
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:$port/health" -TimeoutSec 5 -ErrorAction Stop
        Write-Host "✓ DRL server is running on port $port" -ForegroundColor Green
    } catch {
        Write-Host "⚠ Warning: Could not verify DRL server status" -ForegroundColor Yellow
    }

    # 시뮬레이션 실행
    Write-Host "Starting simulation with seed range: $seedRange" -ForegroundColor Yellow
    $simStart = Get-Date

    try {
        & ./one.bat -b $seedRange default_settings.txt $runtimeFile $overlayFile $configFile
        $simEnd = Get-Date
        $duration = $simEnd - $simStart
        Write-Host "✓ Simulation completed in $($duration.TotalMinutes.ToString('F1')) minutes" -ForegroundColor Green
    } catch {
        Write-Host "✗ Simulation failed: $($_.Exception.Message)" -ForegroundColor Red
    }

    # 서버 종료
    Write-Host "Stopping DRL server..." -ForegroundColor Yellow
    Stop-Job $serverJob -ErrorAction SilentlyContinue
    Remove-Job $serverJob -ErrorAction SilentlyContinue

    # 서버 프로세스 강제 종료 (포트 해제)
    Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -eq "python" -and (netstat -ano | Select-String ":$port ")
    } | Stop-Process -Force -ErrorAction SilentlyContinue

    Write-Host "=== Completed $Mode for ${bufSizeMB}B buffer ===" -ForegroundColor Green
    Start-Sleep -Seconds 3
}

Write-Host "`n=== All buffer sizes completed! ===" -ForegroundColor Magenta
Write-Host "Check logs/ directory for detailed output" -ForegroundColor Cyan