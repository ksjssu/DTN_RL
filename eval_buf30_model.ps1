# 30M 버퍼 모델 평가 스크립트

param(
    [string]$ModelPath = "models/buf30/buf30/drl_train_buf30_100k_rng1_run9_t100000_ppo.pt",
    [int]$Episodes = 5,
    [string]$OutputSuffix = "eval"
)

Write-Host "=== 30M 버퍼 모델 평가 시작 ===" -ForegroundColor Green
Write-Host "모델: $ModelPath"
Write-Host "에피소드: $Episodes"

# 평가 모드 환경변수 설정
$env:EVAL_ONLY = "true"              # 학습 비활성화
$env:MODEL_PATH = $ModelPath          # 로드할 모델 경로
$env:TOTAL_EPISODES = $Episodes       # 평가 에피소드 수
$env:EPISODE_SECONDS = "100000"       # 10만초 에피소드

# 30M 버퍼 전용 설정
$env:MODEL_DIR = "models/buf30"       # 저장 디렉토리

Write-Host "Environment variables set:"
Write-Host "EVAL_ONLY: $env:EVAL_ONLY"
Write-Host "MODEL_PATH: $env:MODEL_PATH"
Write-Host "TOTAL_EPISODES: $env:TOTAL_EPISODES"
Write-Host "EPISODE_SECONDS: $env:EPISODE_SECONDS"

Write-Host "`n=== DRL 서버 시작 ===" -ForegroundColor Yellow
Write-Host "평가가 완료되면 Ctrl+C로 서버를 중지하세요.`n"

# DRL 서버 시작
python toolkit/drl_server.py