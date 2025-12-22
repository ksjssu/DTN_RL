# DRL Training Quickstart

이 문서는 버퍼 크기별 DRL 학습을 반복해서 돌릴 때 따라야 할 최소 절차를 정리한 가이드입니다. 한 번만 읽고 나면 매번 파편적으로 검색할 필요 없이 그대로 따라 하면 됩니다.

## 1. 1회만 필요한 준비
- 프로젝트 루트에서 PowerShell 기준:
  ```powershell
  .\.venv\Scripts\Activate.ps1
  python -m pip install --upgrade pip
  python -m pip install -r toolkit\requirements.txt
  ```
- 의존성 설치는 최초 1회만 하면 되고, 추후에는 `.venv`만 다시 활성화하면 됩니다.

## 2. 학습 기본 절차
1. **가상환경 활성화**
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
2. **DRL 서버 실행**
   ```powershell
  $env:DRL_PORT = "5060"                # 버퍼별로 포트 구분
  $env:MODEL_DIR = "models\buf60"       # 체크포인트 저장 경로
  # 필요 시 보상 파라미터 조정 (예) $env:RELAY_MAX="1.8"
  python toolkit\drl_server.py
   ```
   - 기본 보상 파라미터: `RELAY_MIN=-0.2`, `RELAY_MAX=1.5`, `RELAY_SLOPE=0.12`, `RELAY_MID=40`, `CONDITIONAL_RELAY_GAIN=1.0`, `PRESSURE_SLOPE=0.2`, `PRESSURE_MID=35`, `PRESSURE_CLIP=0.15`, `ABORT_GAIN=0.5`, `ABORT_SLOPE=0.22`, `ABORT_MID=18`.
   - 동일 포트에서 다른 실험을 재사용할 때는 기존 서버를 종료(CTRL+C) 후 다시 실행하세요.
3. **시뮬레이터 학습 실행 (`drl_train`)**
   - 새 터미널(또는 탭)에서:
     ```bash
     ./one.sh scenarios/dynamic/drl_train/drl_train_buf60_100k.txt
     ```
   - 여러 시드를 돌리려면 `./one.sh -b 1:50 <시나리오>` 또는 `run_drl_train_buf60_100.bat` 같은 배치 스크립트를 사용.
4. **결과 확인**
   - 체크포인트: `models/buf60/buf60_epN_ppo.pt`
   - 학습 로그: `reports_drl_train/buf60/drl_episode_rewards.txt`, `drl_diagnostic_log.txt`

## 3. 평가 절차 (`drl_eval`)
1. **DRL 서버를 평가 모드로 실행**
   ```powershell
   $env:DRL_PORT = "5060"
   $env:EVAL_ONLY = "1"
   $env:MODEL_PATH = "models\buf60\buf60_ep50_ppo.pt"   # 원하는 체크포인트
   python toolkit\drl_server.py
   ```
2. **시뮬레이터 평가 실행**
   ```bash
   ./one.sh scenarios/dynamic/drl_eval/drl_eval_buf60_100k.txt
   ```
3. **결과 확인**
   - 평가 로그: `reports_drl_eval/buf60/...`
   - 전달 성공률 평균: `python toolkit/average_delivery_rate.py --dir reports_drl_eval/buf60`

## 4. 반복 시 주의사항
- 의존성 재설치는 필요 없음 → `.venv\Scripts\Activate.ps1`만 다시 실행.
- 버퍼를 바꿀 때는 `DRL_PORT`, `MODEL_DIR`, 시나리오 파일만 변경하면 동일 절차로 진행 가능.
- 보상 파라미터를 수정했다면 `docs/DRL_AGENT_SPEC.md`에 정리된 가중치 표를 참고해 예상 동작을 확인할 것.
- 전달 성공률은 `python toolkit/average_delivery_rate.py --dir reports_drl_eval/bufXX`로 빠르게 집계 가능.

## 5. 자주 쓰는 명령 요약
```powershell
# 가상환경 활성화
.\.venv\Scripts\Activate.ps1

# buf70 학습 서버
$env:DRL_PORT="5070"; $env:MODEL_DIR="models\buf70"; python toolkit\drl_server.py

# buf70 학습 시나리오
./one.sh scenarios/dynamic/drl_train/drl_train_buf70_100k.txt

# buf70 평가 (예: ep50)
$env:DRL_PORT="5070"; $env:EVAL_ONLY="1"; $env:MODEL_PATH="models\buf70\buf70_ep50_ppo.pt"; python toolkit\drl_server.py
./one.sh scenarios/dynamic/drl_eval/drl_eval_buf70_100k.txt
```
