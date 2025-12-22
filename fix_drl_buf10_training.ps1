# Fix DRL buf10 training with proper environment variables

# Set environment variables for 10M buffer training
$env:TRAIN_UNTIL_SECONDS = "360000"  # 100시간 학습 후 자동 저장
$env:SAVE_AT_SECONDS = "108000"      # 30시간마다 중간 저장
$env:MODEL_DIR = "models/buf10"      # buf10 전용 모델 디렉토리
$env:EPISODE_SECONDS = "3600"        # 1시간 에피소드
$env:TOTAL_EPISODES = "100"          # 100 에피소드

Write-Host "Environment variables set for buf10 training:"
Write-Host "TRAIN_UNTIL_SECONDS: $env:TRAIN_UNTIL_SECONDS"
Write-Host "SAVE_AT_SECONDS: $env:SAVE_AT_SECONDS"
Write-Host "MODEL_DIR: $env:MODEL_DIR"
Write-Host "EPISODE_SECONDS: $env:EPISODE_SECONDS"
Write-Host "TOTAL_EPISODES: $env:TOTAL_EPISODES"

# Create model directory
New-Item -ItemType Directory -Path "models/buf10" -Force

# Start DRL server with proper settings
Write-Host "Starting DRL server..."
python toolkit/drl_server.py