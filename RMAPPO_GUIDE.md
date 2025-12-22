# Recurrent MAPPO (R-MAPPO) for DTN Routing

## 개요

DTN 환경에서 LSTM 기반 Recurrent MAPPO를 구현한 버전입니다. 기존 feedforward MAPPO 대비 **partial observability와 temporal dependencies**를 더 잘 처리할 수 있습니다.

## 주요 특징

### 1. **LSTM Networks**
```python
Actor (Decentralized):
  Local Obs (5D) → Embedding → LSTM → Policy Head → Action

Critic (Centralized):
  Global Obs (15D) → Embedding → LSTM → Value Head → State Value
```

### 2. **Hidden State Management**
- 각 Agent별로 LSTM hidden state 유지
- Episode 시작 시 자동 reset
- Checkpoint에 hidden state 저장/복원

### 3. **Truncated BPTT**
- 긴 에피소드를 짧은 sequence로 분할 (기본 16 steps)
- Gradient vanishing/exploding 방지
- 메모리 효율적

### 4. **CTDE (Centralized Training, Decentralized Execution)**
- Actor: Local observation만 사용 (실행 시)
- Critic: Global observation 사용 (학습 시)

## 설치 및 실행

### 1. 기본 실행 (기존과 동일)

```bash
# DRL 서버 시작 (R-MAPPO)
python toolkit/drl_server_rmappo.py

# 시뮬레이션 실행
./one.sh -b 10 scenarios/dynamic/drl_train/drl_train_buf10_100k.txt
```

### 2. 환경변수 설정

```bash
# LSTM 설정
export LSTM_HIDDEN_SIZE=128        # LSTM hidden size (기본: 128)
export LSTM_NUM_LAYERS=1           # LSTM layers (기본: 1)
export TRUNCATED_BPTT_LEN=16       # Sequence length (기본: 16)
export RESET_HIDDEN_ON_EPISODE=true # Episode 시작 시 reset

# PPO 하이퍼파라미터
export PPO_LR=3e-4                 # Learning rate
export PPO_CLIP=0.2                # PPO clip epsilon
export PPO_EPOCHS=10               # Training epochs
export PPO_BATCH_SIZE=64           # Batch size
export PPO_GAMMA=0.99              # Discount factor
export PPO_GAE_LAMBDA=0.95         # GAE lambda

# 학습 설정
export EPISODE_SECONDS=100000      # Episode length
export TOTAL_EPISODES=50           # Total episodes
export MODEL_DIR=models_rmappo     # Model directory

# 평가 모드
export EVAL_ONLY=true              # Evaluation only (no training)
export MODEL_PATH=models_rmappo/rmappo_buf10_ep50.pt  # Load model
```

### 3. 실험 예시

#### 작은 실험 (디버깅용)
```bash
# 짧은 에피소드로 빠르게 테스트
export EPISODE_SECONDS=10000
export TOTAL_EPISODES=5
export LSTM_HIDDEN_SIZE=64
export TRUNCATED_BPTT_LEN=8

python toolkit/drl_server_rmappo.py
```

#### 전체 학습
```bash
# buf10M, 50 episodes
export EPISODE_SECONDS=100000
export TOTAL_EPISODES=50
export LSTM_HIDDEN_SIZE=128
export LSTM_NUM_LAYERS=1
export STEP_REWARD_TRACKING=true

python toolkit/drl_server_rmappo.py
```

#### 평가 (학습된 모델)
```bash
export EVAL_ONLY=true
export MODEL_PATH=models_rmappo/rmappo_buf10_ep50.pt

python toolkit/drl_server_rmappo.py
```

## Feedforward MAPPO vs R-MAPPO 비교

| 특성 | Feedforward MAPPO | R-MAPPO |
|------|-------------------|---------|
| **네트워크** | MLP (128-128-1) | LSTM (128 hidden) |
| **상태** | Markovian | Non-Markovian |
| **메모리** | 없음 (각 step 독립) | LSTM hidden state |
| **학습** | Transition 기반 | Sequence 기반 |
| **POMDP 대응** | ⚠️ 제한적 | ✅ 우수 |
| **Temporal Pattern** | ❌ 어려움 | ✅ 학습 가능 |
| **계산 비용** | 낮음 | 중간 (LSTM) |
| **Sample Efficiency** | 중간 | 높음 (expected) |
| **수렴 속도** | 빠름 | 중간 |
| **최종 성능** | 좋음 | **더 좋음 (expected)** |

## 기대 효과

### 1. **작은 버퍼 (5M-20M) 성능 개선**

**현재 문제:**
- 훈련 중 좋은 성능 → 평가 시 성능 저하
- Seed 민감도 높음
- 불안정한 학습

**R-MAPPO 해결책:**
- LSTM이 접촉 패턴 기억
- Belief state 유지
- 더 robust한 정책

### 2. **Long-term Dependencies 포착**

```python
# Feedforward: 현재만 봄
t=0: pred=0.6 → 행동 A

# R-MAPPO: 과거 이력 고려
t=0: [과거 접촉: 5번, 성공: 4번] + pred=0.6 → 행동 A (신뢰도 높음)
```

### 3. **Sample Efficiency 향상**

- LSTM이 시간적 구조 학습
- 더 적은 데이터로 학습 가능
- 실패 경험도 시간 context로 활용

## 실험 설계

### Phase 1: 기본 검증 (1-2주)

```bash
# buf30M (안정적인 버퍼)에서 먼저 테스트
# Feedforward vs R-MAPPO 비교

# 1. Feedforward MAPPO
python toolkit/drl_server.py &
./one.sh -b 10 scenarios/dynamic/drl_train/drl_train_buf30_100k.txt

# 2. R-MAPPO
python toolkit/drl_server_rmappo.py &
./one.sh -b 10 scenarios/dynamic/drl_train/drl_train_buf30_100k.txt

# 비교:
# - 학습 곡선
# - 최종 delivery rate
# - 학습 속도
```

### Phase 2: 작은 버퍼 테스트 (2-3주)

```bash
# buf10M, buf20M에서 성능 개선 확인

for buf in 10 20; do
    export EPISODE_SECONDS=100000
    export TOTAL_EPISODES=50
    python toolkit/drl_server_rmappo.py &
    ./one.sh -b 50 scenarios/dynamic/drl_train/drl_train_buf${buf}_100k.txt
done

# 기대 결과:
# - 평가 시 성능 저하 완화
# - 더 안정적인 학습
```

### Phase 3: Ablation Study (3-4주)

```bash
# LSTM 설정 최적화

# 1. Hidden size
for hidden in 64 128 256; do
    export LSTM_HIDDEN_SIZE=$hidden
    python toolkit/drl_server_rmappo.py
done

# 2. LSTM layers
for layers in 1 2 3; do
    export LSTM_NUM_LAYERS=$layers
    python toolkit/drl_server_rmappo.py
done

# 3. Truncation length
for trunc in 8 16 32 64; do
    export TRUNCATED_BPTT_LEN=$trunc
    python toolkit/drl_server_rmappo.py
done
```

### Phase 4: 전체 평가 (4-6주)

```bash
# 모든 버퍼 크기에서 R-MAPPO vs Feedforward

# PowerShell script
.\run_rmappo_experiments.ps1 -BufferSizes @("05", "10", "15", "20", "25", "30", "35", "40", "45", "50")
```

## 성능 지표

### 1. Primary Metrics
- **Delivery Rate**: 메시지 전달률
- **Training Stability**: 학습 안정성 (variance)
- **Evaluation Gap**: 훈련 vs 평가 성능 차이

### 2. R-MAPPO Specific Metrics
- **Hidden State Variance**: Hidden state 변화량
- **Sequence Length Impact**: Truncation 길이 영향
- **Memory Utilization**: LSTM 메모리 사용량

### 3. Comparison Metrics
```python
# Feedforward vs R-MAPPO

Metric 1: Sample Efficiency
  - Episodes to reach 50% delivery rate
  - Expected: R-MAPPO < Feedforward

Metric 2: Final Performance
  - Best delivery rate achieved
  - Expected: R-MAPPO > Feedforward

Metric 3: Robustness
  - Performance variance across seeds
  - Expected: R-MAPPO < Feedforward (lower variance)

Metric 4: Generalization
  - Eval performance / Train performance
  - Expected: R-MAPPO closer to 1.0
```

## 디버깅 및 모니터링

### 1. 학습 진행 확인

```bash
# Step-level rewards
tail -f reports/reward_logs/step_rewards_buf10M_rmappo.csv

# Episode rewards
tail -f reports_drl_train/buf10/drl_episode_rewards.txt

# Training diagnostics
tail -f reports/drl_policy_diagnostics.txt
```

### 2. Hidden State 모니터링

Hidden state가 제대로 업데이트되는지 확인:

```python
# drl_server_rmappo.py에 추가
def log_hidden_states(self):
    actor_states = len(self.actor_hidden_manager.states)
    critic_states = len(self.critic_hidden_manager.states)
    print(f"[HiddenStates] Actor: {actor_states}, Critic: {critic_states}")
```

### 3. Sequence Length 확인

```python
# Truncated BPTT가 제대로 동작하는지
print(f"[Sequences] Total: {len(self.sequences)}, Current: {len(self.current_sequence['local_obs'])}")
```

## 문제 해결

### 1. LSTM 학습이 불안정함

**증상:**
- Loss가 NaN/Inf
- Gradient exploding

**해결:**
```bash
# Gradient clipping 강화
export PPO_GRAD_CLIP=0.5

# Learning rate 낮춤
export PPO_LR=1e-4

# Sequence length 줄임
export TRUNCATED_BPTT_LEN=8
```

### 2. 메모리 부족

**증상:**
- OOM error
- 느린 학습

**해결:**
```bash
# Batch size 줄임
export PPO_BATCH_SIZE=32
export PPO_MINIBATCH=32

# Truncation length 줄임
export TRUNCATED_BPTT_LEN=8

# LSTM hidden size 줄임
export LSTM_HIDDEN_SIZE=64
```

### 3. Hidden state가 reset 안 됨

**증상:**
- Episode boundary에서 성능 급변
- 이상한 행동 패턴

**해결:**
```bash
# Episode 시작 시 강제 reset
export RESET_HIDDEN_ON_EPISODE=true

# 또는 코드에서 수동 reset
Handler.AGENT.reset_hidden_states(episode_start=True)
```

## 예상 결과

### 학습 곡선 비교

```
Delivery Rate (%)

50%│                        ╱─────R-MAPPO
   │                    ╱───╯
40%│                ╱───╯
   │            ╱───╯
30%│        ╱───╯                Feedforward
   │    ╱───╯          ╱─────────
20%│╱───╯          ╱───╯
   │           ╱───╯
10%│       ╱───╯
   │   ╱───╯
 0%└──────────────────────────────→ Episodes
   0   10   20   30   40   50

Expected:
- R-MAPPO: 더 빠른 학습, 더 높은 최종 성능
- Feedforward: 초기 빠르나 plateau
```

### 버퍼별 성능 개선 (예상)

| Buffer | Feedforward | R-MAPPO | Improvement |
|--------|-------------|---------|-------------|
| 5M     | 25% ± 10%   | **35% ± 5%**  | +10% ⭐⭐⭐ |
| 10M    | 35% ± 8%    | **42% ± 4%**  | +7% ⭐⭐⭐ |
| 20M    | 45% ± 6%    | **50% ± 3%**  | +5% ⭐⭐ |
| 30M    | 52% ± 3%    | **54% ± 2%**  | +2% ⭐ |
| 50M    | 58% ± 2%    | **59% ± 2%**  | +1% - |

작은 버퍼일수록 R-MAPPO 효과 클 것으로 예상!

## 논문 작성 아이디어

### Title
"Recurrent Multi-Agent PPO for DTN Routing: Handling Partial Observability with LSTM"

### Contributions
1. **Novel Application**: DTN에 R-MAPPO 최초 적용
2. **Performance Improvement**: 작은 버퍼에서 10% 성능 향상
3. **Robustness**: Seed sensitivity 완화
4. **Analysis**: LSTM hidden state의 interpretability

### Experiments
- 5 buffer sizes (5M-50M)
- 3 seeds per configuration
- Comparison with: Prophet, Heuristic, Feedforward MAPPO
- Ablation study: LSTM size, layers, truncation

### Expected Results
- R-MAPPO > Feedforward MAPPO > Baselines
- Variance reduction in small buffers
- Sample efficiency improvement

## 다음 단계

1. ✅ **R-MAPPO 구현 완료**
2. 🔄 **Basic 검증 실험** (buf30M)
3. ⏭️ **Small buffer 실험** (buf10M, buf20M)
4. ⏭️ **Hyperparameter tuning**
5. ⏭️ **Full evaluation** (all buffers)
6. ⏭️ **논문 작성**

## 참고 자료

### 구현 참고
- [Stable-Baselines3-Contrib RecurrentPPO](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)
- [minimalRL ppo-lstm.py](https://github.com/seungeunrho/minimalRL/blob/master/ppo-lstm.py)
- [MarcoMeter/recurrent-ppo-truncated-bptt](https://github.com/MarcoMeter/recurrent-ppo-truncated-bptt)

### 논문 참고
- MAPPO: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games"
- DRQN: "Deep Recurrent Q-Learning for Partially Observable MDPs"
- PPO: "Proximal Policy Optimization Algorithms"

---

**Good Luck with R-MAPPO! 🚀**
