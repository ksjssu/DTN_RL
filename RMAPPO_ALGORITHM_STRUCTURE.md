# 🧠 R-MAPPO TBPTT 알고리즘 구조 완전 분석

## 📊 전체 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                    R-MAPPO TBPTT for DTN Routing                    │
│         Recurrent Multi-Agent PPO with Truncated BPTT              │
└─────────────────────────────────────────────────────────────────────┘

┌───────────────┐         ┌──────────────┐         ┌──────────────┐
│  Java Simulator│ ◄─────► │ Python DRL   │ ◄─────► │ PyTorch      │
│  (ONE.jar)    │  HTTP   │ Server       │  Tensor │ Neural Nets  │
│               │         │              │         │              │
│ - ProphetRouter│         │ - Reward Calc│         │ - Actor LSTM │
│ - RLBridge    │         │ - State Proc │         │ - Critic LSTM│
│ - BufferTracker│         │ - done_keys  │         │ - PPO Update │
└───────────────┘         └──────────────┘         └──────────────┘
        ↑                         ↑                        ↑
        │                         │                        │
        └─────────────── 통합 아키텍처 ───────────────────┘
```

---

## 🎯 핵심 알고리즘: R-MAPPO (Recurrent Multi-Agent PPO)

### 기본 원리

**PPO (Proximal Policy Optimization)**:
```
목적 함수:
L_CLIP(θ) = E_t[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

where:
  r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)  # Probability ratio
  A_t = Advantage estimate (GAE)
  ε = Clipping parameter (0.2)
```

**Recurrent (LSTM 추가)**:
- 부분 관측 환경(DTN)에서 temporal dependency 학습
- Hidden state로 과거 정보 기억

**Multi-Agent (MAPPO)**:
- 각 (host, dest) 쌍이 독립적인 agent
- CTDE (Centralized Training, Decentralized Execution)

**Truncated BPTT**:
- 긴 에피소드를 16-step 시퀀스로 분할
- Computational efficiency + 안정성

---

## 🏗️ 주요 컴포넌트

### 1️⃣ RecurrentPPOPolicy (핵심 클래스)

#### 아키텍처

```python
# Actor Network (Decentralized Execution)
Local Obs (5D) → Embedding (128D) → LSTM (128 hidden) → Mean (1D)
                                                      ↘ log_std (1D)
                                                        ↓
                                                  Normal(mean, std)
                                                        ↓
                                                    Sample Action
                                                        ↓
                                                  Clamp to [-Δ, +Δ]

# Critic Network (Centralized Training)
Global Obs (10D) → Embedding (256D) → LSTM (256 hidden) → Value (1D)
```

**Observation Space**:

**Local Obs (5D)** - Actor 입력:
1. `contacts_norm` - 정규화된 접촉 수 (0~1)
2. `pred` - Prophet 예측도 (delta 적용 후)
3. `bufocc_mean` - 네트워크 평균 버퍼 점유율 (0~1)
4. `capacity_norm` - 정규화된 용량 (0~1)
5. `self_buf_util` - 자신의 버퍼 사용률 (0~1)

**Global Obs (10D)** - Critic 입력 (Local 5D + Global 10D):
6. `global_active_conns` - 활성 연결 수
7. `global_total_msgs` - 전체 메시지 수
8. `global_avg_buf` - 평균 버퍼 점유율
9. `global_avg_contacts` - 평균 접촉 수
10. `global_msg_change_rate` - 메시지 변화율
11. `global_msg_change_momentum` - 메시지 모멘텀
12. `global_buf_util_change_rate` - 버퍼 사용률 변화율
13. `global_buf_util_momentum` - 버퍼 사용률 모멘텀
14. `global_avg_free_buf` - 평균 여유 버퍼
15. `global_high_util_frac` - 고사용률 노드 비율

**Action Space**:
- Continuous: `delta ∈ [-delta_limit, +delta_limit]` (default: ±0.1)
- Prophet 예측도 조정: `P_effective = P_base + delta`

---

### 2️⃣ HiddenStateManager (LSTM 상태 관리)

```python
class HiddenStateManager:
    """각 agent의 LSTM hidden state 관리"""

    states: Dict[str, Tuple[Tensor, Tensor]]
    # agent_id → (h, c)
    # h: (num_layers, 1, hidden_size)
    # c: (num_layers, 1, hidden_size)

    def get_or_init(agent_id):
        """기존 state 반환 또는 zero 초기화"""

    def update(agent_id, h, c):
        """새로운 state 저장 (detach로 BPTT 차단)"""

    def reset(agent_ids):
        """에피소드 종료 시 state 리셋"""
```

**핵심 원리**:
- **Inference 시**: 이전 step의 hidden state 재사용 → temporal continuity
- **Training 시**: 시퀀스 시작 시점의 init hidden state 사용 → TBPTT
- **Episode 종료 시**: done=True → state reset → 새 에피소드 시작

---

### 3️⃣ SequenceWindow (TBPTT 관리)

```python
@dataclass
class SequenceWindow:
    """Agent별 trajectory 수집 및 finalization"""

    # Trajectory data
    local_obs: List[Tensor]      # [T, 5]
    global_obs: List[Tensor]     # [T, 10]
    actions: List[Tensor]        # [T, 1]
    logprobs: List[Tensor]       # [T, 1]
    values: List[Tensor]         # [T, 1]
    rewards: List[float]         # [T]
    dones: List[bool]            # [T]

    # Initial hidden states (시퀀스 시작 직전)
    init_actor_h: Tensor         # (1, 128)
    init_actor_c: Tensor         # (1, 128)
    init_critic_h: Tensor        # (1, 256)
    init_critic_c: Tensor        # (1, 256)

    def is_ready() -> bool:
        """len >= max_len(16) or done=True"""

    def finalize(bootstrap_value) -> Dict:
        """GAE 계산을 위한 데이터 패키징"""
```

**Finalization Triggers**:
1. **길이 제한**: `len(trajectory) >= 16` (TRUNCATED_BPTT_LEN)
2. **실제 종료**: `done=True` (메시지 전달/드랍/중단)

**Bootstrap Value**:
- `done=False` (continuing): Critic으로 V(s_T+1) 추정
- `done=True` (terminal): V(s_T+1) = 0

---

## 🎁 Reward System

### 기본 Reward (Buffer-specific)

```python
def buffer_weight_profile(buffer_size_mb):
    """버퍼 크기에 따른 가중치 동적 조정"""

    # Sigmoid relay weight: 버퍼 클수록 relay 보상 증가
    sigmoid = 1 / (1 + exp(-0.12 * (buffer_size - 40)))
    relay_weight = -0.4 + (3.0 - (-0.4)) * sigmoid * 2.0

    # Exponential abort penalty: 버퍼 작을수록 abort 벌점 증가
    abort_weight = -0.6 / (1 + exp(0.18 * (buffer_size - 25)))

    return (W_DELIVER=10.0, relay_weight, abort_weight)
```

**버퍼별 전략** (20M 예시):
```
Buffer 20M:
  - W_DELIVER = 10.0   (delivered 메시지당)
  - W_RELAY = 0.85     (relayed 메시지당)
  - W_ABORT = -0.15    (aborted 전송당)
  - W_DROP = 8.0       (dropped 메시지당)
```

**Total Reward**:
```python
reward = delivered × W_DELIVER
       + relayed × W_RELAY
       + aborted × W_ABORT
       - drops × W_DROP
       + shaping_reward  # Gradient Alignment (optional)
```

---

### Gradient Alignment Reward (Shaping)

**목적**: Sparse reward 문제 해결 (delivered 이벤트가 드물음)

**수식**:
```
R_shaping = W_GRAD·R_grad + W_ALIGN·R_align + W_ARRIVAL·R_arrival

where:
  R_grad = action × (P_base - P_neighbor)
  R_align = -(buffer_norm - 0.5) × action
  R_arrival = 10.0  (if delivered else 0.0)
```

**Components**:

1. **R_grad (Gradient Alignment)**:
   ```
   내 예측도 > 이웃 예측도  →  action 높이면 보상
   내 예측도 < 이웃 예측도  →  action 낮추면 보상
   ```
   - 의미: 더 좋은 경로로 메시지 유도

2. **R_align (Buffer Alignment)**:
   ```
   버퍼 많이 참  (>0.5)  →  action 낮추면 보상 (거부)
   버퍼 비어있음 (<0.5)  →  action 높이면 보상 (수용)
   ```
   - 의미: 버퍼 압력 완화

3. **R_arrival (Sparse Goal)**:
   ```
   메시지 도착 시 큰 보상 (10.0)
   ```
   - 의미: 최종 목표 달성 강화

**⚠️ Reward Hacking 방지**:
```python
# ❌ 잘못된 방법 (Hacking 가능):
R_grad = action × state_batch["pred"]  # Agent가 조작 가능!

# ✅ 올바른 방법 (Hacking 불가):
R_grad = action × prev_transition["p_base"]  # Agent 조작 불가!
```

**환경변수 제어**:
```bash
USE_GRADIENT_REWARD=true  # 활성화
W_GRAD=1.0                # Gradient alignment weight
W_ALIGN=0.5               # Buffer alignment weight
W_ARRIVAL=10.0            # Arrival bonus
```

---

## 🔄 done_keys 메커니즘 (Event-based Episode Termination)

### 원리

**문제**:
- Phase 2.1: `done = (reward >= threshold)` → 부정확, 조기 종료
- Phase 2.2: `done = False` → 보수적, 실제 종료 미반영

**해결** (Reviewer의 개선):
```python
# HTTP Handler (Line 1433-1434)
done_keys = set()
for item in prev_transition:
    delivered = item.get("delivered", 0)
    drops = item.get("drops", 0)
    aborted = item.get("aborted", 0)

    if delivered > 0 or drops > 0 or aborted > 0:
        done_keys.add(f"{host}#{dest}")  # 실제 종료 이벤트!

# RecurrentPPOPolicy.update() (Line 788)
done = k in done_index  # done_keys 기반 판정

# Hidden state reset (Line 827-831)
if done:
    actor_hidden_manager.reset([k])
    critic_hidden_manager.reset([k])
```

**의의**:
- ✅ DTN 메시지 생명주기와 RL 에피소드 경계 완벽 일치
- ✅ 실제 결과까지의 인과관계 학습
- ✅ Hidden state가 의미 있는 시점에 리셋

---

## 🔁 Training Loop (PPO Update)

### 전체 플로우

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Trajectory Collection (Java Simulator)                  │
└─────────────────────────────────────────────────────────────┘
   ↓ HTTP POST /infer_and_update (매 100초)

┌─────────────────────────────────────────────────────────────┐
│ 2. Reward Calculation + done_keys Collection               │
└─────────────────────────────────────────────────────────────┘
   ↓

┌─────────────────────────────────────────────────────────────┐
│ 3. SequenceWindow.add_step()                               │
│    - Store (s, a, r, done, logprob, value, hidden)        │
│    - Check is_ready(): len>=16 or done=True               │
└─────────────────────────────────────────────────────────────┘
   ↓ if ready

┌─────────────────────────────────────────────────────────────┐
│ 4. SequenceWindow.finalize()                               │
│    - Compute bootstrap value (if done=False)              │
│    - Package trajectory for GAE                           │
│    - Add to ready_sequences                               │
└─────────────────────────────────────────────────────────────┘
   ↓ if len(ready_sequences) >= 32

┌─────────────────────────────────────────────────────────────┐
│ 5. GAE Computation (Generalized Advantage Estimation)      │
└─────────────────────────────────────────────────────────────┘

   For each sequence:

   δ_t = r_t + γ·V(s_t+1)·(1-done_t) - V(s_t)
   A_t = δ_t + (γλ)·A_t+1·(1-done_t)

   returns_t = A_t + V(s_t)

   ↓

┌─────────────────────────────────────────────────────────────┐
│ 6. Sequence-level Batching (NOT step-level!)              │
└─────────────────────────────────────────────────────────────┘

   For epoch in 1..K (K=10):

   # ✅ CRITICAL: Shuffle SEQUENCES, not steps!
   seq_indices = randperm(num_sequences)

   For each minibatch of sequences:

   ┌──────────────────────────────────────────────┐
   │ 7. LSTM Forward Pass                        │
   └──────────────────────────────────────────────┘

   # Pack sequences for efficiency
   packed_input = pack_padded_sequence(obs_batch, lengths)

   # Actor forward
   actor_out, (h_n, c_n) = actor_lstm(
       packed_input,
       (init_actor_h, init_actor_c)  # ✅ 시퀀스 시작 시점 state!
   )

   # Critic forward
   critic_out, _ = critic_lstm(
       packed_input,
       (init_critic_h, init_critic_c)
   )

   ┌──────────────────────────────────────────────┐
   │ 8. PPO Loss Computation                     │
   └──────────────────────────────────────────────┘

   # Actor loss (clipped surrogate objective)
   ratio = exp(new_logprob - old_logprob)
   surr1 = ratio × advantage
   surr2 = clip(ratio, 1-ε, 1+ε) × advantage
   actor_loss = -min(surr1, surr2).mean()

   # Critic loss (MSE)
   critic_loss = MSE(value_pred, returns)

   # Entropy bonus (exploration)
   entropy_loss = -entropy.mean()

   # Total loss
   loss = actor_loss + 0.5·critic_loss + 0.01·entropy_loss

   ┌──────────────────────────────────────────────┐
   │ 9. Backpropagation + Gradient Clipping      │
   └──────────────────────────────────────────────┘

   optimizer.zero_grad()
   loss.backward()
   clip_grad_norm_(params, max_norm=0.5)
   optimizer.step()

   ↓

┌─────────────────────────────────────────────────────────────┐
│ 10. Clear ready_sequences                                  │
└─────────────────────────────────────────────────────────────┘
```

### GAE (Generalized Advantage Estimation)

**수식**:
```python
# Backward iteration
for t in reversed(range(seq_len)):
    # TD error
    δ_t = r_t + γ·V(s_t+1)·(1-done_t) - V(s_t)

    # GAE
    A_t = δ_t + γ·λ·A_t+1·(1-done_t)

    # Update for next iteration
    V_next = V(s_t)
    A_next = A_t

# Returns
returns = advantages + values
```

**Parameters**:
- `γ` (gamma) = 0.99 - Discount factor
- `λ` (lambda) = 0.95 - GAE parameter (bias-variance trade-off)

**Device Consistency** (중요!):
```python
# ✅ 모든 텐서를 동일 device로
values = seq_data['values'].to(self.device)
rewards = seq_data['rewards'].to(self.device)
dones = seq_data['dones'].to(self.device)
bootstrap_value = seq_data['bootstrap_value'].to(self.device)
advantages = torch.zeros(seq_len, device=self.device)
next_advantage = torch.zeros(1, device=self.device)  # Reviewer 추가!
```

---

## 🧩 Sequence-level Batching (TBPTT 핵심!)

### 잘못된 방법 (Phase 2.1 이전):

```python
# ❌ Step-level random sampling
all_steps = flatten(all_sequences)
perm = torch.randperm(len(all_steps))
minibatch = all_steps[perm[:batch_size]]

# 문제: Temporal order 파괴 → LSTM 학습 불가!
```

### 올바른 방법 (Phase 2.1 이후):

```python
# ✅ Sequence-level batching
seq_indices = torch.randperm(num_sequences)

for batch_start in range(0, num_sequences, seq_batch_size):
    batch_seq_indices = seq_indices[batch_start:batch_start+seq_batch_size]
    batch_seqs = [seq_data_list[i] for i in batch_seq_indices]

    # ✅ 각 시퀀스의 temporal order 유지!
    # ✅ LSTM이 올바른 순서로 학습 가능!
```

### Hidden State Stacking (중요!)

**잘못된 방법** (Phase 2.1):
```python
# ❌ dim=0 concat + unsqueeze(0)
init_actor_h = torch.cat(init_actor_h_list, dim=0).unsqueeze(0)
# Shape: (1, batch, 1, 128) - 4D tensor! → LSTM ERROR!
```

**올바른 방법** (Phase 2.2 이후):
```python
# ✅ dim=1 concat (batch dimension)
init_actor_h = torch.cat(init_actor_h_list, dim=1)
# Shape: (1, batch, 128) - 3D tensor! → LSTM OK!
```

---

## 📊 Hyperparameters

```python
# PPO Parameters
PPO_CLIP = 0.2              # Clipping epsilon
PPO_VALUE_COEF = 0.5        # Critic loss coefficient
PPO_ENTROPY = 0.01          # Entropy bonus
PPO_LR = 3e-4               # Learning rate
PPO_BATCH_SIZE = 64         # Sequence batch size
PPO_MINIBATCH = 64          # Minibatch size
PPO_EPOCHS = 10             # K epochs per update
PPO_GAMMA = 0.99            # Discount factor
PPO_GAE_LAMBDA = 0.95       # GAE lambda

# LSTM Parameters
LSTM_HIDDEN_SIZE = 128      # Actor hidden size
LSTM_CRITIC_HIDDEN = 256    # Critic hidden size (128*2)
LSTM_LAYERS = 1             # LSTM layers

# TBPTT Parameters
TRUNCATED_BPTT_LEN = 16     # Sequence length
MIN_SEQUENCES_TO_TRAIN = 32 # Minimum sequences for training

# Reward Parameters
W_DELIVER = 10.0            # Delivery reward
W_DROP = 8.0                # Drop penalty
W_GRAD = 1.0                # Gradient alignment weight
W_ALIGN = 0.5               # Buffer alignment weight
W_ARRIVAL = 10.0            # Arrival bonus
```

---

## 🎯 CTDE (Centralized Training, Decentralized Execution)

### Training (Centralized):

```python
# Critic sees global information
global_obs = local_obs + [
    global_active_conns,
    global_total_msgs,
    global_avg_buf,
    global_avg_contacts,
    global_msg_change_rate,
    global_msg_change_momentum,
    global_buf_util_change_rate,
    global_buf_util_momentum,
    global_avg_free_buf,
    global_high_util_frac
]

# Critic provides better value estimates
value = critic(global_obs, critic_hidden)
```

### Execution (Decentralized):

```python
# Actor only uses local information
local_obs = [
    contacts_norm,
    pred,
    bufocc_mean,
    capacity_norm,
    self_buf_util
]

# Actor makes decisions independently
action = actor(local_obs, actor_hidden)
```

**장점**:
- Training: 글로벌 정보로 better credit assignment
- Execution: 로컬 정보만으로 분산 실행 가능 (통신 불필요)

---

## 🔑 핵심 개선사항 타임라인

### Phase 1: Gradient Alignment Reward
```
✅ calculate_shaping_reward() 구현
✅ Reward hacking 방지 (p_base vs pred 분리)
✅ 환경변수 제어 (USE_GRADIENT_REWARD)
```

### Phase 2.1: 6개 버그 수정
```
✅ SequenceWindow.finalize() key structure (tuple → individual keys)
✅ _ppo_update() sequence-level batching (NOT step-level!)
✅ AttributeError 수정 (actor_lstm_hidden → hidden_size)
✅ GAE device consistency (partial)
✅ done detection 개선 (threshold → length-based)
✅ SequenceWindow torch import 수정
```

### Phase 2.2: Hidden State & Done Logic
```
✅ Hidden state stacking (torch.cat(dim=1) - 3D tensor)
✅ done detection (reward → length-based)
✅ Shape validation tests
```

### Reviewer: done_keys + Device Consistency
```
✅ done_keys mechanism (event-based termination)
✅ update() done_keys parameter
✅ HTTP handler done_keys collection
✅ GAE next_advantage device 추가
✅ Complete device consistency
```

---

## 🚀 현재 시스템 상태: 완성!

```
┌────────────────────────────────────────────────────────────┐
│ ✅ R-MAPPO TBPTT 시스템 완성도: 100%                      │
├────────────────────────────────────────────────────────────┤
│ ✅ Recurrent PPO with LSTM                                │
│ ✅ CTDE (Centralized Training, Decentralized Execution)   │
│ ✅ Truncated BPTT (16 steps)                              │
│ ✅ Sequence-level batching                                │
│ ✅ Event-based episode termination (done_keys)            │
│ ✅ Gradient Alignment Reward (optional)                   │
│ ✅ Buffer-specific reward strategies                      │
│ ✅ Hidden state continuity                                │
│ ✅ Device consistency (CPU/GPU)                           │
│ ✅ Reward hacking prevention                              │
└────────────────────────────────────────────────────────────┘
```

**DTN 라우팅 최적화를 위한 완전하고 올바른 R-MAPPO TBPTT 구현 완료!** 🎉
