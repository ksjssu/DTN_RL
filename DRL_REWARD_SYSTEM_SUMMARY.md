# DTN DRL 보상 시스템 개발 기록

## 개요
DTN (Delay/Disruption Tolerant Networking) 환경에서 Deep Reinforcement Learning을 이용한 라우팅 최적화 프로젝트의 보상 시스템 개발 과정을 기록합니다.

## 현재 보상 시스템 구조 (최신 상태 - 압력 기반 복원)

### 1. 5M 버퍼
- **기본 보상**: `delivered × 10.0`
- **압력 기반 릴레이 보상/페널티** (0.1 이상 차이):
  - `pressure_diff = 내 점유율 - 네트워크 평균 점유율`
  - `pressure_diff ≥ 0.1`: 릴레이 보상 = `min(pressure_diff × 2.0 × relayed, relayed × 5.0)`
  - `pressure_diff ≤ -0.1`: 릴레이 페널티 = `max(pressure_diff × 1.0 × relayed, -relayed × 2.0)`

### 2. 10M 버퍼 ⭐ 압력 기반 복원 (최신)
- **기본 보상**: `delivered × 10.0`
- **압력 기반 릴레이 보상/페널티** (5M과 동일):
  - `pressure_diff = 내 점유율 - 네트워크 평균 점유율`
  - `pressure_diff ≥ 0.1`: 릴레이 보상 = `min(pressure_diff × 2.0 × relayed, relayed × 5.0)`
  - `pressure_diff ≤ -0.1`: 릴레이 페널티 = `max(pressure_diff × 1.0 × relayed, -relayed × 2.0)`
- **전략**: "네트워크 상황에 따른 동적 협력 - 내가 상대적으로 여유있으면 도와주고, 부족하면 자제"

### 3. 15M 버퍼
- **기본 보상**: `delivered × 10.0`
- **압력 기반 릴레이 보상/페널티** (0.1 이상 차이):
  - 5M, 10M과 동일한 압력 기반 시스템

### 4. 20M 버퍼 ⭐ 압력 기반 복원 (최신)
- **기본 보상**: `delivered × 10.0`
- **압력 기반 릴레이 보상/페널티** (5M, 10M, 15M과 동일):
  - `pressure_diff = 내 점유율 - 네트워크 평균 점유율`
  - `pressure_diff ≥ 0.1`: 릴레이 보상 = `min(pressure_diff × 2.0 × relayed, relayed × 5.0)`
  - `pressure_diff ≤ -0.1`: 릴레이 페널티 = `max(pressure_diff × 1.0 × relayed, -relayed × 2.0)`

### 5. 25M 버퍼
- **기본 보상**: `delivered × 10.0`
- **스마트 릴레이** (임계값 기준):
  - `버퍼 점유율 ≤ 0.6`: 릴레이 보상 `relayed × 5.0`
  - `버퍼 점유율 ≥ 0.8`: 릴레이 페널티 `relayed × -3.0`
  - `0.6 ~ 0.8`: 추가 보상/페널티 없음

### 6. 30M+ 버퍼
- **기본 보상**: `delivered × 10.0 + relayed × 10.0`
- **스마트 릴레이**: 버퍼 점유율 ≤ 0.7일 때만 `relayed × 5.0`

## 주요 개발 변경 사항

### Phase 1: 복잡한 보상 시스템 단순화
- **이전**: 여러 보상 파라미터 (delivered, relayed, drops, aborted, delay penalty)
- **이후**: 전달 성공 보상 + 버퍼별 특화 보상
- **사용자 피드백**: "보상 파라미터가 너무 많습니다. 전달성공이랑 다른 하나만 남게 해주세요"

### Phase 2: 압력 기반 보상 도입 (5M, 10M, 15M)
- **개념**: 네트워크 전체 상황에 따른 동적 릴레이 전략
- **압력 차이**: `내 버퍼 점유율 - 네트워크 평균 점유율`
- **이론**: 상대적으로 여유있는 노드가 더 많은 릴레이 부담

### Phase 3: 10M 버퍼 성능 개선
- **문제**: 복잡한 압력 기반 시스템으로 인한 성능 저하
- **해결**: 단순한 OR 조건 기반 스마트 릴레이로 변경
- **결과**: 더 직관적이고 협력적인 전략

### Phase 4: 임계값 최적화 및 압력 기반 복원
- **0.1 기준 도입**: 노이즈에 덜 민감한 안정적 학습
- **0.7 → 0.5 → 0.35 조정**: 더 엄격한 조건으로 품질 높은 릴레이
- **AND → OR → 압력 기반 복원**: 10M, 20M 성능 문제로 인한 압력 기반 시스템 재도입

### Phase 5: Multi-seed 학습 필요성 인식 (현재)
- **문제 발견**: 10M 버퍼에서 훈련 중 성능은 좋지만 평가 시 성능 저하
- **원인 분석**: 소용량 버퍼의 높은 시드 민감도, 30M+ 대용량은 단일 시드로도 안정
- **해결 방향**: Multi-seed 학습으로 robust한 정책 학습

## 100만초 장기 학습 시스템

### 파일 구조
```
scenarios/dynamic/drl_train/drl_train_buf10_1000k.txt  # 100만초 시나리오
toolkit/plot_step_rewards_1000k.py                     # 실시간 리워드 모니터링
run_1000k_drl_training.ps1                            # 100만초 학습 스크립트
start_1000k_experiment.ps1                            # 실험 시작 스크립트
```

### 실행 방법
```powershell
# 자동 실행 (추천)
.\start_1000k_experiment.ps1 -BufferSize 10

# 수동 실행
python toolkit/drl_server.py  # DRL 서버 먼저 시작
./one.sh scenarios/dynamic/drl_train/drl_train_buf10_1000k.txt  # 시뮬레이션
```

### Step-by-step 리워드 추적
- **로그 파일**: `reports/reward_logs/step_rewards_buf10M.csv`
- **실시간 모니터링**: `python toolkit/plot_step_rewards_1000k.py --buffer 10`
- **환경변수**: `STEP_REWARD_TRACKING=true`

## 핵심 코드 변경 사항

### DRL 서버 (`toolkit/drl_server.py`)
1. **버퍼별 보상 함수**: `get_buffer_specific_reward()`
2. **압력 기반 보상 로직**: 5M, 15M에 적용
3. **10M 스마트 릴레이**: OR 조건 기반 협력 전략
4. **Step 리워드 로깅**: `log_step_rewards()` 함수 추가
5. **액션 로깅**: `log_actions_for_visualization()` 함수 추가

### 주요 환경 변수
```bash
EPISODE_SECONDS=1000000     # 100만초 에피소드
TOTAL_EPISODES=1            # 1개 에피소드
SAVE_AT_SECONDS=100000      # 10만초마다 저장
STEP_REWARD_TRACKING=true   # Step별 리워드 로깅
PPO_DEBUG_ACTIONS=true      # 액션 로깅 (선택사항)
```

## 성능 분석 도구

### 실시간 모니터링
- **리워드 그래프**: 시간별 총 리워드, 호스트당 평균 리워드
- **이동평균**: 트렌드 분석
- **갱신 간격**: 기본 30초, 조정 가능

### 로그 분석
- **CSV 형식**: sim_time, total_reward, avg_reward_per_host, num_hosts
- **Excel 호환**: 추가 분석 가능
- **자동 저장**: 실험 종료 시 그래프 PNG 저장

## 향후 개발 방향

### 고려 사항
1. **다른 버퍼 크기 최적화**: 15M, 20M, 25M 보상 시스템 개선
2. **하이퍼파라미터 튜닝**: 임계값, 보상 가중치 최적화
3. **성능 벤치마킹**: 휴리스틱 기법 대비 성능 비교
4. **SCI 논문 준비**: 5M-50M 모든 버퍼에서 휴리스틱 기법 초과 성능

### 실험 우선순위
1. **10M 버퍼**: 새로운 OR 조건 기반 성능 검증
2. **100만초 학습**: 장기 패턴 학습 효과 분석
3. **버퍼별 특성화**: 각 버퍼 크기별 최적 전략 도출

## 주요 성능 문제 및 해결 방안

### 10M/20M 버퍼 성능 이슈
**문제**: 훈련 중에는 휴리스틱 기법보다 좋은 성능을 보이지만, 평가 시 성능이 급격히 저하

**원인 분석**:
1. **과적합**: 특정 훈련 환경에만 최적화
2. **시드 민감도**: 소용량 버퍼는 초기 조건에 매우 민감
3. **복잡한 보상 신호**: 압력 기반 보상의 불안정성
4. **탐험/활용 불균형**: 평가 시 결정론적 정책 사용

**해결 방안**:
1. **Multi-seed 학습**: 3-5개 시드로 robust한 정책 학습
2. **압력 기반 복원**: 동적 네트워크 상황 인식 능력 향상
3. **Transfer Learning**: 30M+ 모델에서 학습된 지식 전이
4. **Ensemble 방법**: 여러 시드 모델의 앙상블 활용

### 버퍼별 특성 요약
- **5M-20M**: 높은 시드 민감도, 압력 기반 보상 필요, Multi-seed 학습 권장
- **25M**: 중간 단계, 임계값 기반으로 충분
- **30M+**: 안정적, 단일 시드로도 robust, 단순한 보상 구조

---
*마지막 업데이트: 2024년 (10M, 20M 버퍼 압력 기반 복원 및 Multi-seed 학습 계획)*