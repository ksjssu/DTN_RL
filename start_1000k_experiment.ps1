# 100만초 DRL 실험 시작 스크립트

param(
    [string]$BufferSize = "10"
)

Write-Host "=== 100만초 DRL 실험 시작 ===" -ForegroundColor Green
Write-Host "버퍼 크기: ${BufferSize}M"

# 실시간 리워드 모니터링 시작 (별도 창)
Write-Host "`n실시간 리워드 모니터링 시작..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList "-NoExit", "-Command", "python toolkit/plot_step_rewards_1000k.py --buffer $BufferSize --interval 10"

# 2초 대기
Start-Sleep -Seconds 2

# 메인 학습 시작
Write-Host "`n메인 학습 프로세스 시작..." -ForegroundColor Yellow
& ".\run_1000k_drl_training.ps1" -BufferSize $BufferSize

Write-Host "`n=== 실험 완료 ===" -ForegroundColor Green