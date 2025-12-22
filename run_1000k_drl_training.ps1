# 100만초 DRL 연속 학습 스크립트 (10M 버퍼)

param(
    [string]$BufferSize = "10",
    [string]$ModelDir = "models/buf10_1000k",
    [string]$Scenario = "scenarios/dynamic/drl_train/drl_train_buf10_1000k.txt"
)

Write-Host "=== 100만초 DRL 연속 학습 시작 ===" -ForegroundColor Green
Write-Host "버퍼 크기: ${BufferSize}M"
Write-Host "모델 디렉토리: $ModelDir"
Write-Host "시나리오: $Scenario"

# 환경변수 설정 - 100만초 연속 학습
$env:EPISODE_SECONDS = "1000000"        # 100만초 에피소드
$env:TOTAL_EPISODES = "1"               # 1개 에피소드만
$env:MODEL_DIR = $ModelDir              # 모델 저장 디렉토리
$env:SAVE_AT_SECONDS = "100000"         # 10만초마다 중간 저장
$env:TRAIN_UNTIL_SECONDS = "1000000"    # 100만초까지 학습

# 리워드 추적 활성화
$env:STEP_REWARD_TRACKING = "true"      # Step별 리워드 로깅
$env:PPO_DEBUG_ACTIONS = "false"        # 액션 로깅은 비활성화 (성능)

Write-Host "`n환경 변수 설정:"
Write-Host "EPISODE_SECONDS: $env:EPISODE_SECONDS"
Write-Host "TOTAL_EPISODES: $env:TOTAL_EPISODES"
Write-Host "MODEL_DIR: $env:MODEL_DIR"
Write-Host "SAVE_AT_SECONDS: $env:SAVE_AT_SECONDS"
Write-Host "TRAIN_UNTIL_SECONDS: $env:TRAIN_UNTIL_SECONDS"
Write-Host "STEP_REWARD_TRACKING: $env:STEP_REWARD_TRACKING"

# 모델 디렉토리 생성
New-Item -ItemType Directory -Path $ModelDir -Force | Out-Null

Write-Host "`n=== DRL 서버 시작 ===" -ForegroundColor Yellow
Write-Host "실시간 리워드 모니터링을 보려면 별도 터미널에서:"
Write-Host "python toolkit/plot_step_rewards_1000k.py --buffer $BufferSize" -ForegroundColor Cyan
Write-Host "`n학습 진행 상황은 reports/reward_logs/ 에서 확인 가능"
Write-Host "100만초 학습 완료 후 Ctrl+C로 서버 중지하세요.`n"

# DRL 서버 시작 (백그라운드로 시작)
Start-Job -ScriptBlock {
    param($ModelDir)
    $env:MODEL_DIR = $ModelDir
    python toolkit/drl_server.py
} -ArgumentList $ModelDir -Name "DRLServer"

Write-Host "DRL 서버가 백그라운드에서 시작되었습니다." -ForegroundColor Green

# 잠시 기다린 후 시뮬레이션 시작
Start-Sleep -Seconds 3

Write-Host "`n=== ONE 시뮬레이터 시작 ===" -ForegroundColor Yellow
Write-Host "시나리오: $Scenario"

# 시뮬레이션 실행
& "./one.sh" $Scenario

Write-Host "`n=== 시뮬레이션 완료 ===" -ForegroundColor Green

# DRL 서버 종료
Write-Host "DRL 서버 종료 중..." -ForegroundColor Yellow
Get-Job -Name "DRLServer" | Stop-Job
Get-Job -Name "DRLServer" | Remove-Job

Write-Host "`n=== 학습 완료 ===" -ForegroundColor Green
Write-Host "모델이 저장된 위치: $ModelDir"
Write-Host "리워드 로그: reports/reward_logs/"
Write-Host "리워드 그래프: reports/reward_plots/"