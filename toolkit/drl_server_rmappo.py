#!/usr/bin/env python3
"""
Recurrent MAPPO (R-MAPPO) with GRU cores for DTN routing.

Based on:
- Stable-Baselines3-Contrib RecurrentPPO
- minimalRL ppo-lstm.py
- MarcoMeter/recurrent-ppo-truncated-bptt

Key features:
- GRU-based actor and critic for handling partial observability
- Hidden state management for continuous episodes
- Truncated BPTT for long episodes
- CTDE (Centralized Training, Decentralized Execution)
- Gradient Alignment Reward for sparse reward problem

Endpoint: POST http://127.0.0.1:5000/infer_and_update

===========================================
📡 Observation & Transition Protocol 명세
===========================================

Java Simulator와 Python DRL Server 간 통신 프로토콜:

1. state_batch (현재 상태 - Java → Python):
   POST body의 "state_batch" 필드에 포함됨.
   각 item은 현재 라우팅 결정을 위한 상태 정보:

   {
     "host": str,              # 노드 ID (예: "p0")
     "dest": str,              # 목적지 ID (예: "p5")
     "contacts_norm": float,   # 정규화된 접촉 수 (0.0~1.0)
     "pred": float,            # ⚠️ P_action (delta 적용 후 예측도) - Policy 입력용
     "bufocc_mean": float,     # 평균 버퍼 점유율 (0.0~1.0)
     "capacity_norm": float,   # 정규화된 용량 (0.0~1.0)
     "self_buf_util": float,   # 자신의 버퍼 사용률 (0.0~1.0)
     "global_active_conns": float,        # 글로벌 활성 연결 수
     "global_total_msgs": float,          # 글로벌 전체 메시지 수
     "global_avg_buf": float,             # 글로벌 평균 버퍼 점유율
     "global_avg_contacts": float,        # 글로벌 평균 접촉 수
     "global_msg_change_rate": float,     # 메시지 변화율
     "global_msg_change_momentum": float, # 메시지 변화 모멘텀
     "global_buf_util_change_rate": float,# 버퍼 사용률 변화율
     "global_buf_util_momentum": float,   # 버퍼 사용률 모멘텀
     "global_avg_free_buf": float,        # 글로벌 평균 여유 버퍼
     "global_high_util_frac": float,      # 고사용률 노드 비율
   }

2. prev_transition (이전 결과 - Java → Python):
   POST body의 "prev_transition" 필드에 포함됨.
   이전 step에서 수행한 action의 결과 (Reward 계산용):

   {
     "host": str,              # 노드 ID
     "dest": str,              # 목적지 ID
     "delivered": float,       # 전달 완료된 메시지 수
     "relayed": float,         # 중계한 메시지 수
     "drops": float,           # 드랍된 메시지 수
     "aborted": float,         # 중단된 전송 수
     "buffer_size_mb": float,  # 버퍼 크기 (MB)

     # ===== Gradient Alignment용 추가 필드 (Java RLBridgeReport에서 추가 필요) =====
     "p_base": float,          # ⭐ 순수 PROPHET 예측도 (delta 적용 전) - Reward 계산용
     "neighbor_p_base": float, # 이웃 노드의 순수 PROPHET 예측도 (TODO: 향후 구현, 현재 0.0)
     "my_buffer_norm": float,  # 내 노드의 버퍼 사용률 (0.0~1.0)
     "action": float,          # 이전 step에서 적용한 delta 값 (-1.0~1.0)
   }

3. actions (응답 - Python → Java):
   {
     "policy_id": str,
     "actions": [
       {
         "host": str,
         "per_message": [
           {"dest": str, "delta": float},  # delta: -delta_limit ~ +delta_limit
           ...
         ]
       },
       ...
     ],
     "actions_kv": {"host#dest": delta, ...}  # Flat map for quick lookup
   }

⚠️ 리워드 해킹 방지 (중요!):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  - p_base (prev_transition):
    → delta 적용 전 순수 PROPHET 예측도
    → Reward 계산에만 사용
    → Agent가 조작할 수 없는 값

  - pred (state_batch):
    → delta 적용 후 예측도 (p_base + action)
    → Policy 입력으로만 사용
    → Agent가 관찰하는 값

  ❌ 만약 reward 계산에 pred를 사용하면:
     Agent가 높은 action을 출력 → pred 증가 → reward 증가
     → "Reward Hacking" 발생! (실제 성능과 무관하게 보상 증가)

  ✅ 올바른 방식:
     Reward = f(p_base, action, delivered, ...)
     p_base는 Agent가 변경할 수 없는 외부 값이므로 안전!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📌 Java 측 구현 TODO:
  - RLBridgeReport.java 수정하여 prev_transition에 다음 필드 추가:
    1. "p_base": 순수 PROPHET 예측도 (ProphetRouter에서 가져오기)
    2. "action": 이전 step에서 적용한 delta 값
    3. "my_buffer_norm": 버퍼 사용률
  - (선택) "neighbor_p_base": 접촉한 이웃의 예측도 계산 로직 추가
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import heapq
import math
import os
import random
import pathlib
import time
import numpy as np
import traceback

# Recommended defaults (can be overridden via environment)
ENV_DEFAULTS = {
    # === Active reward knobs (non-zero) ===
    "PBASE_REWARD_WEIGHT": "1.0",
    "PBASE_DIFF_EPS": "0",
    "BUFFER_DIFF_REWARD_WEIGHT": "4.0",
    "SUCCESS_RATE_WEIGHT": "400.0",
    "SUCCESS_RATE_EPS": "0",
    "SUCCESS_RATE_BONUS_WEIGHT": "400.0",
    # === Disabled reward knobs (all zero) ===
    "DELIVERY_REWARD_WEIGHT": "0.0",
    "OVERHEAD_PENALTY_WEIGHT": "0.0",
    "HIGH_LOAD_PENALTY_WEIGHT": "0.0",
    "HIGH_LOAD_UTIL_BASE": "0.35",
    "HIGH_LOAD_UTIL_THRESHOLD": "0.65",
    "HIGH_LOAD_FLAG_THRESHOLD": "0.45",
    "RELAY_BONUS_WEIGHT": "0.0",
    "RELAY_LOW_UTIL_THRESHOLD": "0.65",
    "RELAY_PRED_THRESHOLD": "0.6",
    "ACTION_GAIN_LOW": "0.0",
    "ACTION_GAIN_HIGH": "0.0",
    "ACTION_GAIN_THRESHOLD": "0.8",
    "MID_RANGE_REWARD": "0.0",
    "MID_UTIL_THRESHOLD": "0.0",
    "MID_UTIL_PENALTY_WEIGHT": "0.0",
    "MID_UTIL_PRESSURE_MIN": "0.0",
    "RELAY_HIGH_LOAD_FACTOR_MIN": "0.15",
    "HEURISTIC_REWARD_WEIGHT": "0.0",
    "HEURISTIC_PBASE_MIN": "0.6",
    "HEURISTIC_BUF_MAX": "0.55",
    "TREND_UTIL_THRESHOLD": "0.75",
    "TREND_DELTA_THRESHOLD": "0.01",
    "TREND_MIN_STEPS": "5",
    "TREND_PENALTY_WEIGHT": "0.0",
    # === Training / buffer / misc settings ===
    "TRUNCATED_BPTT_LEN": "256",
    "MAX_SEQS_PER_UPDATE": "768",
    "MIN_SEQUENCES_TO_TRAIN": "64",
    "TARGET_REWARD_TIME": "0",
    "TARGET_REWARD_WEIGHT": "0",
    "TOPK_BUFFER_SIZE": "256",
    "TOPK_MIX_RATIO": "0.30",
    "TOPK_MIN_SCORE": "0.0",
}
for key, value in ENV_DEFAULTS.items():
    os.environ.setdefault(key, value)


def env_float(name, fallback=None):
    if fallback is None:
        fallback = ENV_DEFAULTS.get(name, "0.0")
    return float(os.environ.get(name, fallback))


def env_int(name, fallback=None):
    if fallback is None:
        fallback = ENV_DEFAULTS.get(name, "0")
    return int(float(os.environ.get(name, fallback)))


def env_str(name, fallback=None):
    if fallback is None:
        fallback = ENV_DEFAULTS.get(name, "")
    return os.environ.get(name, fallback)

# Allow optional CPU-only mode via env; default uses available CUDA
FORCE_CPU = os.environ.get("DRL_FORCE_CPU", "false").lower() in ("1", "true", "yes")
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

HOST = os.environ.get("DRL_HOST", "127.0.0.1")
PORT = int(float(os.environ.get("DRL_PORT", "5010")))

# R-MAPPO specific configurations
LSTM_HIDDEN_SIZE = int(os.environ.get("LSTM_HIDDEN_SIZE", "128"))
LSTM_NUM_LAYERS = int(os.environ.get("LSTM_NUM_LAYERS", "1"))
TRUNCATED_BPTT_LEN = int(os.environ.get("TRUNCATED_BPTT_LEN", "256"))  # Sequence length for training
RESET_HIDDEN_ON_EPISODE = os.environ.get("RESET_HIDDEN_ON_EPISODE", "true").lower() == "true"

# Action logging configuration
PPO_DEBUG_ACTIONS = os.environ.get("PPO_DEBUG_ACTIONS", "false").lower() == "true"

# Step-by-step reward tracking configuration
STEP_REWARD_TRACKING = os.environ.get("STEP_REWARD_TRACKING", "true").lower() == "true"

# Reward configuration (delivery + Prophet alignment + overhead penalty)
DELIVERY_REWARD_WEIGHT = env_float("DELIVERY_REWARD_WEIGHT")
DELIVERY_REWARD_UTIL_MAX = env_float("DELIVERY_REWARD_UTIL_MAX")
PBASE_REWARD_WEIGHT = env_float("PBASE_REWARD_WEIGHT")
PBASE_DIFF_EPS = env_float("PBASE_DIFF_EPS")
OVERHEAD_PENALTY_WEIGHT = env_float("OVERHEAD_PENALTY_WEIGHT")
HIGH_LOAD_PENALTY_WEIGHT = env_float("HIGH_LOAD_PENALTY_WEIGHT")
HIGH_LOAD_UTIL_BASE = env_float("HIGH_LOAD_UTIL_BASE")
HIGH_LOAD_UTIL_THRESHOLD = env_float("HIGH_LOAD_UTIL_THRESHOLD")
HIGH_LOAD_FLAG_THRESHOLD = env_float("HIGH_LOAD_FLAG_THRESHOLD")
TARGET_REWARD_TIME = env_int("TARGET_REWARD_TIME")
TARGET_REWARD_WEIGHT = env_float("TARGET_REWARD_WEIGHT")
RELAY_BONUS_WEIGHT = env_float("RELAY_BONUS_WEIGHT")
RELAY_LOW_UTIL_THRESHOLD = env_float("RELAY_LOW_UTIL_THRESHOLD")
RELAY_PRED_THRESHOLD = env_float("RELAY_PRED_THRESHOLD")
ACTION_GAIN_LOW = env_float("ACTION_GAIN_LOW")
ACTION_GAIN_HIGH = env_float("ACTION_GAIN_HIGH")
ACTION_GAIN_THRESHOLD = env_float("ACTION_GAIN_THRESHOLD")
MID_RANGE_REWARD = env_float("MID_RANGE_REWARD")
MID_UTIL_THRESHOLD = env_float("MID_UTIL_THRESHOLD")
MID_UTIL_PENALTY_WEIGHT = env_float("MID_UTIL_PENALTY_WEIGHT")
MID_UTIL_PRESSURE_MIN = env_float("MID_UTIL_PRESSURE_MIN")
RELAY_HIGH_LOAD_FACTOR_MIN = env_float("RELAY_HIGH_LOAD_FACTOR_MIN")
HEURISTIC_REWARD_WEIGHT = env_float("HEURISTIC_REWARD_WEIGHT")
HEURISTIC_PBASE_MIN = env_float("HEURISTIC_PBASE_MIN")
HEURISTIC_BUF_MAX = env_float("HEURISTIC_BUF_MAX")
BUFFER_DIFF_REWARD_WEIGHT = env_float("BUFFER_DIFF_REWARD_WEIGHT")
TREND_UTIL_THRESHOLD = env_float("TREND_UTIL_THRESHOLD")
TREND_DELTA_THRESHOLD = env_float("TREND_DELTA_THRESHOLD")
TREND_MIN_STEPS = env_int("TREND_MIN_STEPS")
TREND_PENALTY_WEIGHT = env_float("TREND_PENALTY_WEIGHT")
SUCCESS_RATE_WEIGHT = env_float("SUCCESS_RATE_WEIGHT")
SUCCESS_RATE_EPS = env_float("SUCCESS_RATE_EPS")
SUCCESS_RATE_BONUS_WEIGHT = env_float("SUCCESS_RATE_BONUS_WEIGHT")
TOPK_BUFFER_SIZE = env_int("TOPK_BUFFER_SIZE")
TOPK_MIX_RATIO = env_float("TOPK_MIX_RATIO")
TOPK_MIN_SCORE = env_float("TOPK_MIN_SCORE")

def log_step_rewards(sim_time, rewards_per_host, buffer_size_mb=None, episode_num=1):
    """Log step-by-step rewards for tracking and graphing."""
    if not STEP_REWARD_TRACKING:
        return

    try:
        # Create reward logs directory
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Determine log file based on buffer size
        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"step_rewards_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "step_rewards_rmappo.csv")

        # Write header if file doesn't exist
        write_header = not os.path.exists(log_file)

        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("episode,sim_time,total_reward,avg_reward_per_host,num_hosts\n")

            total_reward = sum(rewards_per_host.values()) if rewards_per_host else 0.0
            num_hosts = len(rewards_per_host) if rewards_per_host else 0
            avg_reward = total_reward / num_hosts if num_hosts > 0 else 0.0

            # Log actual episode number
            f.write(f"{episode_num},{sim_time},{total_reward},{avg_reward},{num_hosts}\n")
    except Exception:
        pass  # Silent fail for logging

# Training configuration
EPISODE_SECONDS = int(float(os.environ.get("EPISODE_SECONDS", "100000")))
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "50"))
SIM_END_SECONDS = int(float(os.environ.get("SIM_END_SECONDS", "5000000")))

# Checkpoint control
MODEL_DIR = os.environ.get("MODEL_DIR", "models_rmappo")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() in ("1", "true", "yes")



def extract_buffer_size_mb(sim_id):
    """Extract buffer size from sim_id."""
    try:
        sim_id_lower = sim_id.lower()
        if "buf" in sim_id_lower:
            import re
            match = re.search(r'buf(\d+)', sim_id_lower)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None


class HiddenStateManager:
    """
    Manages recurrent hidden states for each agent (GRU-friendly).

    Key features:
    - Agent-specific hidden state storage
    - Automatic initialization
    - Episode reset capability
    """
    def __init__(self, num_layers=1, hidden_size=128, device='cpu', has_cell_state=False):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device
        self.has_cell_state = has_cell_state
        self.states = {}  # agent_id -> hidden tensor (or tuple for LSTM compatibility)

    def _init_state(self):
        h = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
        if self.has_cell_state:
            c = torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
            return (h, c)
        return h

    def get_or_init(self, agent_id):
        """Get existing hidden state or initialize new one."""
        if agent_id not in self.states:
            self.states[agent_id] = self._init_state()
        return self.states[agent_id]

    def update(self, agent_id, h, c=None):
        """Update hidden state for agent."""
        if self.has_cell_state:
            if c is None:
                raise ValueError("Cell state required for LSTM hidden manager")
            self.states[agent_id] = (h.detach(), c.detach())
        else:
            self.states[agent_id] = h.detach()

    def reset(self, agent_ids=None):
        """Reset hidden states for specified agents or all."""
        if agent_ids is None:
            self.states.clear()
        else:
            for aid in agent_ids:
                self.states.pop(aid, None)

    def get_all_states(self):
        """Get all hidden states (for checkpointing)."""
        if self.has_cell_state:
            return {k: (h.cpu(), c.cpu()) for k, (h, c) in self.states.items()}
        return {k: h.cpu() for k, h in self.states.items()}

    def set_all_states(self, states_dict):
        """Restore all hidden states (from checkpoint)."""
        if self.has_cell_state:
            self.states = {k: (h.to(self.device), c.to(self.device))
                           for k, (h, c) in states_dict.items()}
        else:
            self.states = {k: h.to(self.device) for k, h in states_dict.items()}


@dataclass
class SequenceWindow:
    """
    Agent별 시퀀스 윈도우 for TBPTT

    타이밍 정의:
    - init_*_hidden: 시퀀스 첫 step **이전** hidden state
    - 각 step의 transition: (s_t, a_t, r_t+1, s_t+1)
    - bootstrap_value: 시퀀스 종료 시점의 다음 state value
    """
    key: str  # "host#dest"
    max_len: int = TRUNCATED_BPTT_LEN

    # Trajectory 데이터
    local_obs: List = field(default_factory=list)  # List[torch.Tensor]
    global_obs: List = field(default_factory=list)  # List[torch.Tensor]
    actions: List = field(default_factory=list)  # List[torch.Tensor]
    logprobs: List = field(default_factory=list)  # List[torch.Tensor]
    values: List = field(default_factory=list)  # List[torch.Tensor]
    rewards: List = field(default_factory=list)  # List[float]
    dones: List = field(default_factory=list)  # List[bool]

    # 초기 Hidden States (시퀀스 시작 **직전**)
    init_actor_h: Optional = None  # torch.Tensor
    init_critic_h: Optional = None  # torch.Tensor

    # 메모리 관리
    last_update_time: float = field(default_factory=lambda: time.time())

    def add_step(self, local_obs, global_obs, action, logprob, value,
                 reward, done, actor_h_prev=None, critic_h_prev=None):
        """
        Transition 추가

        Args:
            actor_h_prev: (h, c) tuple - 이 step **이전** actor hidden
            critic_h_prev: (h, c) tuple - 이 step **이전** critic hidden
        """
        # 첫 step이면 init hidden 저장
        if len(self.local_obs) == 0:
            if actor_h_prev is not None:
                self.init_actor_h = actor_h_prev.detach()
            if critic_h_prev is not None:
                self.init_critic_h = critic_h_prev.detach()

        self.local_obs.append(local_obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.last_update_time = time.time()

    def is_ready(self):
        """
        Finalize 준비 여부
        - 길이 >= max_len OR
        - done=True (메시지 도착/드랍)
        """
        return (len(self.local_obs) >= self.max_len) or \
               (len(self.dones) > 0 and self.dones[-1])

    def finalize(self, bootstrap_value=None):
        """
        텐서 변환 및 반환

        Args:
            bootstrap_value: done=False 시퀀스의 다음 step value

        Returns:
            dict with keys: 'local_obs', 'init_actor_h', 'init_critic_h', etc.
        """
        if len(self.local_obs) == 0:
            return None

        # Bootstrap 결정
        if self.dones[-1]:  # 에피소드 종료
            bootstrap = 0.0
        elif bootstrap_value is not None:
            bootstrap = bootstrap_value
        else:
            bootstrap = 0.0

        # torch는 이미 클래스 레벨에서 import됨 (TORCH_OK 체크됨)
        # 따라서 여기서 try-except 불필요
        return {
            'key': self.key,
            'length': len(self.local_obs),
            'local_obs': torch.stack(self.local_obs),
            'global_obs': torch.stack(self.global_obs),
            'actions': torch.stack(self.actions).unsqueeze(-1),
            'logprobs': torch.stack(self.logprobs).squeeze(-1),
            'values': torch.stack(self.values),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            # Store done mask as float to avoid dtype issues during GAE
            'dones': torch.tensor(self.dones, dtype=torch.float32),
            'init_actor_h': self.init_actor_h,
            'init_critic_h': self.init_critic_h,
            'bootstrap_value': torch.tensor(bootstrap, dtype=torch.float32)
        }


class RecurrentPPOPolicy:
    """
    Recurrent MAPPO with GRU cores for handling partial observability in DTN.

    Architecture:
    - Actor (Decentralized): Local obs (7D) -> GRU -> Action
    - Critic (Centralized): Global obs (19D) -> GRU -> Value

    Key features:
    - GRU networks for temporal dependencies
    - Hidden state management per agent
    - Truncated BPTT for long episodes
    - CTDE (Centralized Training, Decentralized Execution)
    """

    def __init__(self, obs_dim=7, global_obs_dim=19, hidden_size=128, num_layers=1):
        # Hyperparameters
        self.clip_eps = float(os.environ.get("PPO_CLIP", "0.2"))
        self.value_coef = float(os.environ.get("PPO_VALUE_COEF", "0.5"))
        self.entropy_coef = float(os.environ.get("PPO_ENTROPY", "0.015"))
        self.lr = float(os.environ.get("PPO_LR", "3e-4"))
        self.batch_size = int(os.environ.get("PPO_BATCH_SIZE", "1024"))
        self.minibatch = int(os.environ.get("PPO_MINIBATCH", "256"))
        self.epochs = int(os.environ.get("PPO_EPOCHS", "3"))  # Reduced from 10 for faster updates
        self.gamma = float(os.environ.get("PPO_GAMMA", "0.99"))
        self.gae_lambda = float(os.environ.get("PPO_GAE_LAMBDA", "0.95"))
        self.max_sequences_per_update = int(os.environ.get("MAX_SEQS_PER_UPDATE", "768"))  # Reduced from 1024

        # Device configuration: use GPU if available
        device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        print(f"[RecurrentPPOPolicy] Using device: {self.device}")

        self.obs_dim = obs_dim
        self.global_obs_dim = global_obs_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # ===== Actor Network (Decentralized) =====
        # Input: local observations (7D)
        # GRU processes temporal sequence
        # Output: action distribution parameters

        self.actor_embedding = nn.Linear(obs_dim, hidden_size).to(self.device)
        self.actor_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.actor_head = nn.Linear(hidden_size, 1).to(self.device)  # Mean of action distribution
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_std_min = -3.0

        # ===== Critic Network (Centralized) =====
        # Input: global observations (19D = 7 local + 12 global)
        # GRU processes temporal sequence
        # Output: state value

        self.critic_embedding = nn.Linear(global_obs_dim, hidden_size * 2).to(self.device)
        self.critic_gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.critic_head = nn.Linear(hidden_size * 2, 1).to(self.device)

        # Optimizer for all parameters
        self.opt = optim.Adam(
            list(self.actor_embedding.parameters()) +
            list(self.actor_gru.parameters()) +
            list(self.actor_head.parameters()) +
            [self.log_std] +
            list(self.critic_embedding.parameters()) +
            list(self.critic_gru.parameters()) +
            list(self.critic_head.parameters()),
            lr=self.lr
        )

        # Hidden state managers
        self.actor_hidden_manager = HiddenStateManager(num_layers, hidden_size, self.device)
        self.critic_hidden_manager = HiddenStateManager(num_layers, hidden_size * 2, self.device)

        # Sequence management (NEW: SequenceWindow based)
        self.sequence_windows = {}  # key -> SequenceWindow
        self.ready_sequences = []   # List[dict] - finalized sequences ready for training
        self.topk_buffer_size = TOPK_BUFFER_SIZE
        self.topk_mix_ratio = TOPK_MIX_RATIO
        self.topk_min_score = TOPK_MIN_SCORE
        self.topk_sequences = []  # min-heap storing (score, counter, seq_copy)
        self._topk_counter = 0

        # Memory management settings
        self.max_window_age = float(os.environ.get("MAX_WINDOW_AGE", "300.0"))  # 5 minutes
        self.max_active_windows = int(os.environ.get("MAX_ACTIVE_WINDOWS", "1000"))
        self.min_sequences_to_train = int(os.environ.get("MIN_SEQUENCES_TO_TRAIN", "64"))  # Increased to stabilize updates

        # Last actions cache (for reward assignment)
        # Elements include hidden states captured before the action
        self.last_actions = {}  # key -> (local_obs, global_obs, u_pre, logprob, value, actor_h_prev, critic_h_prev)

        self.policy_id = "rmappo_v1"
        print(f"[RecurrentPPOPolicy] Initialized R-MAPPO:")
        print(f"  - Actor GRU: {obs_dim}D -> {hidden_size} (L={num_layers})")
        print(f"  - Critic GRU: {global_obs_dim}D -> {hidden_size*2} (L={num_layers})")
        print(f"  - Truncated BPTT: {TRUNCATED_BPTT_LEN} steps")

    def _to_tensor(self, arr):
        """Convert array to tensor."""
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _construct_global_obs(self, local_obs, global_features):
        """Construct global observation by concatenating local obs with global features."""
        return torch.cat([local_obs, global_features], dim=1)

    def _tanh_gaussian_sample(self, mean):
        """Sample action from tanh-squashed Gaussian distribution."""
        log_std_clamped = torch.clamp(torch.nan_to_num(self.log_std.detach()),
                                      min=self.log_std_min, max=2.0)
        std = log_std_clamped.exp()
        std = torch.clamp(std, min=1e-6, max=10.0).expand_as(mean)

        # Sanitize mean and std
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-2))

        try:
            normal = torch.distributions.Normal(mean, std)
            u = normal.rsample()
        except Exception as e:
            print(f"[ERROR] Normal distribution failed: {e}")
            u = torch.zeros_like(mean)

        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        a = torch.tanh(u)
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))

        # Log-prob with tanh correction
        logp = normal.log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), min=1e-6))
        logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))

        return u, a, logp.squeeze(-1)

    def _tanh_gaussian_logprob(self, mean, u):
        """Calculate log probability of action."""
        log_std_clamped = torch.clamp(torch.nan_to_num(self.log_std.detach()),
                                      min=self.log_std_min, max=2.0)
        std = log_std_clamped.exp()
        std = torch.clamp(std, min=1e-6, max=10.0).expand_as(mean)

        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-2))
        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))

        try:
            normal = torch.distributions.Normal(mean, std)
            a = torch.tanh(u)
            a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
            logp = normal.log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), min=1e-6))
            logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
        except Exception as e:
            print(f"[ERROR] logprob calculation failed: {e}")
            logp = torch.zeros_like(u)

        return logp.squeeze(-1)

    def act_batch(self, obs_batch, delta_limit, keys, global_features=None):
        """
        Generate actions for a batch of observations using R-MAPPO with BATCHED GRU processing.

        OPTIMIZED: Uses batched GRU forward passes instead of sequential per-agent processing.
        This provides ~28x speedup for large batches (e.g., 1137 agents).

        This method:
        1. Processes local observations through actor GRU in BATCH
        2. Generates actions from learned policy
        3. Evaluates state values using critic GRU with global observations in BATCH
        4. Manages hidden states for each agent

        Args:
            obs_batch: List of local observations (5D each)
            delta_limit: Action scale limit
            keys: List of agent keys for tracking
            global_features: Global features (10D) for critic

        Returns:
            List of scaled actions
        """
        if len(obs_batch) == 0:
            return []

        # Convert observations to tensors
        local_obs = torch.nan_to_num(self._to_tensor(obs_batch))  # (batch, 7)
        batch_size = local_obs.size(0)

        # Validate batch size matches keys length
        if keys is not None and len(keys) != batch_size:
            raise ValueError(
                f"Batch size mismatch: obs_batch has {batch_size} items but keys has {len(keys)} items. "
                f"This usually indicates a bug in the calling code."
            )

        # Handle global features
        if global_features is None or len(global_features) == 0:
            global_feat_tensor = torch.zeros(batch_size, self.global_obs_dim - self.obs_dim,
                                            device=self.device)
        else:
            global_feat_tensor = torch.nan_to_num(self._to_tensor(global_features))

        # Prepare for GRU (batch_size, seq_len=1, features)
        local_obs_seq = local_obs.unsqueeze(1)  # (batch, 1, 7)

        # ===== Actor Forward (BATCHED GRU) =====
        # OPTIMIZATION: Collect all hidden states first, then process in batch
        actor_h_list = []
        actor_hiddens_prev = []  # Store for reward assignment

        for key in keys:
            h_prev = self.actor_hidden_manager.get_or_init(key)
            actor_hiddens_prev.append(h_prev)
            actor_h_list.append(h_prev)

        # Stack hidden states: (num_layers, batch_size, hidden_size)
        h_init_actor = torch.cat(actor_h_list, dim=1)  # (num_layers, batch, hidden)

        # BATCHED embedding and GRU forward (ONE call for all agents!)
        embedded_actor = torch.tanh(self.actor_embedding(local_obs_seq))  # (batch, 1, hidden)
        gru_out_actor, h_new_actor = self.actor_gru(
            embedded_actor, h_init_actor
        )  # (batch, 1, hidden)

        # Update hidden states for all agents
        for i, key in enumerate(keys):
            # Extract hidden state for this agent
            h_new_i = h_new_actor[:, i:i+1, :]  # (num_layers, 1, hidden)
            self.actor_hidden_manager.update(key, h_new_i)

        # Policy head (batched)
        mean = self.actor_head(gru_out_actor.squeeze(1))  # (batch, 1)

        # Sample actions (batched)
        u_pre, actions, logprobs = self._tanh_gaussian_sample(mean)
        actions = actions.squeeze(-1)  # (batch,)
        u_pre = u_pre.squeeze(-1)  # (batch,)

        # Adaptive action gain: push more extreme values when buffer is critical
        self_buf_util = local_obs[:, 4]  # column index for self utilization
        gain_low = torch.full_like(self_buf_util, ACTION_GAIN_LOW)
        gain_high = torch.full_like(self_buf_util, ACTION_GAIN_HIGH)
        gain_vec = torch.where(self_buf_util >= ACTION_GAIN_THRESHOLD, gain_high, gain_low)
        actions = torch.tanh(actions * gain_vec)

        # Scale actions
        scale = delta_limit if (delta_limit is not None and delta_limit > 0) else 1.0
        action_scaled = (actions * scale).detach()

        # ===== Critic Forward (BATCHED GRU) =====
        global_obs = self._construct_global_obs(local_obs, global_feat_tensor)  # (batch, 15)
        global_obs_seq = global_obs.unsqueeze(1)  # (batch, 1, 19)

        # Collect critic hidden states
        critic_h_list = []
        critic_hiddens_prev = []

        for key in keys:
            h_prev = self.critic_hidden_manager.get_or_init(key)
            critic_hiddens_prev.append(h_prev)
            critic_h_list.append(h_prev)

        # Stack hidden states
        h_init_critic = torch.cat(critic_h_list, dim=1)  # (num_layers, batch, hidden*2)

        # BATCHED embedding and GRU forward
        embedded_critic = torch.tanh(self.critic_embedding(global_obs_seq))  # (batch, 1, hidden*2)
        gru_out_critic, h_new_critic = self.critic_gru(
            embedded_critic, h_init_critic
        )  # (batch, 1, hidden*2)

        # Update hidden states for all agents
        for i, key in enumerate(keys):
            h_new_i = h_new_critic[:, i:i+1, :]
            self.critic_hidden_manager.update(key, h_new_i)

        # Value head (batched)
        values = self.critic_head(gru_out_critic.squeeze(1)).squeeze(-1)  # (batch,)

        # Store last actions for reward assignment
        act_list = action_scaled.cpu().tolist()
        for i, k in enumerate(keys):
            self.last_actions[k] = (
                local_obs[i].detach(),
                global_obs[i].detach(),
                u_pre[i].detach(),
                logprobs[i].detach(),
                values[i].detach(),
                actor_hiddens_prev[i],   # Hidden BEFORE step
                critic_hiddens_prev[i],   # Hidden BEFORE step
            )

        return act_list

    def update(self, rewards_per_host, rewards_per_key, mapping_host_to_keys, delta_limit,
               epochs=None, done_keys=None):
        """
        Update policy using SequenceWindow-based TBPTT.

        NEW Implementation:
        1. Adds transitions to agent-specific SequenceWindows
        2. Finalizes ready sequences (length limit or done=True)
        3. Computes bootstrap values for non-terminal sequences
        4. Manages memory (cleanup old windows)
        5. Trains when enough sequences are ready
        """
        any_added = False

        # Build transitions from last step
        acted_by_host = defaultdict(list)
        invalid_keys = []
        for k, data in self.last_actions.items():
            if len(data) != 7:
                invalid_keys.append(k)
                continue
            host = k.split('#', 1)[0] if '#' in k else ''
            acted_by_host[host].append(k)

        if invalid_keys:
            print(f"[WARN] Skipping {len(invalid_keys)} action entries with outdated format")
            for k in invalid_keys:
                self.last_actions.pop(k, None)

        done_index = done_keys or set()

        for host, acted_keys in acted_by_host.items():
            if not acted_keys:
                continue

            r_host = float(rewards_per_host.get(host, 0.0))
            r_each = r_host / len(acted_keys) if len(acted_keys) > 0 else 0.0

            for k in acted_keys:
                # Unpack last_actions (7 elements)
                local_obs, global_obs, action, logprob, value, actor_h_prev, critic_h_prev = self.last_actions[k]
                r_key = float(rewards_per_key.get(k, r_each))

                # Done 판정: prev_transition에서 전달/드랍 이벤트 발생 시 즉시 종료
                done = k in done_index

                # Get or create sequence window for this agent
                if k not in self.sequence_windows:
                    self.sequence_windows[k] = SequenceWindow(key=k, max_len=TRUNCATED_BPTT_LEN)

                window = self.sequence_windows[k]
                window.add_step(
                    local_obs, global_obs, action, logprob, value, r_key, done,
                    actor_h_prev, critic_h_prev
                )
                any_added = True

                # Check if sequence is ready to finalize
                if window.is_ready():
                    # Compute bootstrap value for non-terminal sequences
                    if not done:
                        # Re-evaluate current state for bootstrap
                        with torch.no_grad():
                            h_prev = self.critic_hidden_manager.get_or_init(k)
                            g_obs = window.global_obs[-1].unsqueeze(0).unsqueeze(0)
                            embedded = torch.tanh(self.critic_embedding(g_obs))
                            gru_out, _ = self.critic_gru(embedded, h_prev)
                            bootstrap_val = self.critic_head(gru_out.squeeze()).item()
                    else:
                        bootstrap_val = 0.0

                    # Finalize and add to ready sequences
                    seq_data = window.finalize(bootstrap_val)
                    if seq_data is not None:
                        self._enqueue_sequence(seq_data)
                        # Debug: Log sequence finalization
                        print(f"[TBPTT] Finalized sequence: key={k}, len={len(window.local_obs)}, "
                              f"done={done}, bootstrap={bootstrap_val:.3f}")

                    # Remove window
                    del self.sequence_windows[k]

                    # Reset hidden states if done
                    if done:
                        self.actor_hidden_manager.reset([k])
                        self.critic_hidden_manager.reset([k])
                        # Debug: Log hidden state reset
                        print(f"[TBPTT] Reset hidden states for key={k} (done=True)")

        # Clear last actions
        self.last_actions.clear()

        if not any_added:
            return {"updated": False}

        # Memory management: cleanup old windows
        self._cleanup_old_windows()

        # Train if enough sequences ready
        if len(self.ready_sequences) >= self.min_sequences_to_train:
            return self._ppo_update(epochs or self.epochs)

        return {"updated": False, "ready_sequences": len(self.ready_sequences)}

    def _cleanup_old_windows(self):
        """
        오래되거나 과도한 sequence window 정리

        두 가지 정리 전략:
        1. TTL 초과: last_update_time이 max_window_age 초과
        2. 개수 제한: sequence_windows 개수가 max_active_windows 초과
        """
        current_time = time.time()
        to_remove = []

        # 1. TTL 초과 window 제거
        for k, window in self.sequence_windows.items():
            if current_time - window.last_update_time > self.max_window_age:
                to_remove.append(k)

        for k in to_remove:
            # 강제 finalize (bootstrap=0.0)
            seq_data = self.sequence_windows[k].finalize(bootstrap_value=0.0)
            if seq_data is not None:
                self._enqueue_sequence(seq_data)

            # Window 제거
            del self.sequence_windows[k]

            # Hidden state reset
            self.actor_hidden_manager.reset([k])
            self.critic_hidden_manager.reset([k])

        # 2. 최대 개수 제한
        if len(self.sequence_windows) > self.max_active_windows:
            # 오래된 순서로 정렬
            sorted_keys = sorted(
                self.sequence_windows.keys(),
                key=lambda k: self.sequence_windows[k].last_update_time
            )

            # 초과분 제거
            n_to_remove = len(self.sequence_windows) - self.max_active_windows

            for k in sorted_keys[:n_to_remove]:
                seq_data = self.sequence_windows[k].finalize(bootstrap_value=0.0)
                if seq_data is not None:
                    self._enqueue_sequence(seq_data)

                del self.sequence_windows[k]
                self.actor_hidden_manager.reset([k])
                self.critic_hidden_manager.reset([k])

    def _enqueue_sequence(self, seq_data):
        """Append finalized sequence to training queue + optional top-k buffer."""
        if seq_data is None:
            return
        self.ready_sequences.append(seq_data)
        self._maybe_store_topk(seq_data)

    def _maybe_store_topk(self, seq_data):
        """Keep high-quality sequences in a min-heap for future mixing."""
        if (
            self.topk_buffer_size <= 0
            or seq_data is None
            or not TORCH_OK  # sequences rely on torch tensors
        ):
            return
        rewards = seq_data.get('rewards')
        if rewards is None or not torch.is_tensor(rewards):
            return
        score = float(rewards.sum().item())
        if score <= self.topk_min_score:
            return
        seq_copy = self._clone_sequence(seq_data)
        entry = (score, self._topk_counter, seq_copy)
        self._topk_counter += 1
        if len(self.topk_sequences) < self.topk_buffer_size:
            heapq.heappush(self.topk_sequences, entry)
        else:
            if score > self.topk_sequences[0][0]:
                heapq.heapreplace(self.topk_sequences, entry)

    def _sample_topk_sequences(self, count):
        """Sample up to `count` cloned sequences from the priority buffer."""
        if count <= 0 or not self.topk_sequences:
            return []
        actual = min(count, len(self.topk_sequences))
        picked = random.sample(self.topk_sequences, actual)
        return [self._clone_sequence(entry[2]) for entry in picked]

    def _clone_sequence(self, seq_data):
        """Deep-clone sequence tensors so they can be reused safely."""
        if seq_data is None:
            return None

        def clone_tensor(tensor):
            if tensor is None:
                return None
            return tensor.clone().detach()

        return {
            'key': seq_data.get('key'),
            'length': seq_data.get('length', len(seq_data.get('local_obs', []))),
            'local_obs': clone_tensor(seq_data['local_obs']),
            'global_obs': clone_tensor(seq_data['global_obs']),
            'actions': clone_tensor(seq_data['actions']),
            'logprobs': clone_tensor(seq_data['logprobs']),
            'values': clone_tensor(seq_data['values']),
            'rewards': clone_tensor(seq_data['rewards']),
            'dones': clone_tensor(seq_data['dones']),
            'init_actor_h': clone_tensor(seq_data.get('init_actor_h')),
            'init_critic_h': clone_tensor(seq_data.get('init_critic_h')),
            'bootstrap_value': clone_tensor(seq_data['bootstrap_value'])
        }

    def _compute_gae_for_sequence(self, seq_data):
        """
        Compute Generalized Advantage Estimation for a single sequence.

        Args:
            seq_data: dict with keys:
                - 'values': torch.Tensor [seq_len]
                - 'rewards': torch.Tensor [seq_len]
                - 'dones': torch.Tensor [seq_len]
                - 'bootstrap_value': float (V(s_T+1) for non-terminal sequences)

        Returns:
            advantages: torch.Tensor [seq_len]
            returns: torch.Tensor [seq_len]
        """
        # as_tensor guards against stale numpy arrays or python lists sneaking in
        values = torch.as_tensor(seq_data['values'], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(seq_data['rewards'], dtype=torch.float32, device=self.device)
        # CRITICAL FIX: convert boolean done flags to float so 1.0 - done works safely
        dones = torch.as_tensor(seq_data['dones'], dtype=torch.float32, device=self.device)
        bootstrap_value = torch.tensor(seq_data['bootstrap_value'], dtype=torch.float32, device=self.device)

        seq_len = len(rewards)
        advantages = torch.zeros(seq_len, dtype=torch.float32, device=self.device)

        # GAE calculation (backward iteration)
        # A_t = δ_t + (γλ) * A_t+1
        # where δ_t = r_t + γ * V(s_t+1) * (1 - done_t) - V(s_t)

        next_value = bootstrap_value
        next_advantage = torch.zeros(1, device=self.device)

        for t in reversed(range(seq_len)):
            # TD error
            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]

            # Advantage with GAE
            advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage * (1.0 - dones[t])

            # Update for next iteration (t-1)
            next_value = values[t]
            next_advantage = advantages[t]

        # Returns = Advantages + Values
        returns = advantages + values

        return advantages, returns

    def _ppo_update(self, epochs):
        """
        Perform PPO update using SequenceWindow-based TBPTT.

        KEY FIX: Sequence-level batching (NOT step-level random sampling!)
        - Shuffle sequences, not individual steps
        - Process entire sequences through GRU
        - Maintains temporal continuity for proper TBPTT
        """
        if len(self.ready_sequences) == 0:
            return {"updated": False}

        # Debug: Log PPO training start
        total_available = len(self.ready_sequences)
        n_train = min(total_available, max(1, self.max_sequences_per_update))
        print(f"\n[PPO] Starting training with {n_train} sequences (queued={total_available})")

        # 1. Compute GAE for each sequence
        seq_data_list = []
        total_steps = 0

        invalid_seqs = 0
        base_sequences = list(self.ready_sequences[:n_train])

        # Mix in high-performing sequences from the priority buffer
        priority_sequences = []
        if self.topk_buffer_size > 0 and self.topk_mix_ratio > 0 and self.topk_sequences:
            target_extra = int(len(base_sequences) * self.topk_mix_ratio)
            if target_extra <= 0 and len(base_sequences) > 0:
                target_extra = 1
            priority_sequences = self._sample_topk_sequences(target_extra)
            if priority_sequences:
                print(f"[PPO] Injecting {len(priority_sequences)} top-k sequences (buffer={len(self.topk_sequences)})")

        seq_candidates = base_sequences + priority_sequences

        for seq in seq_candidates:
            local = seq['local_obs']
            global_obs = seq['global_obs']
            if local.shape[-1] != self.obs_dim or global_obs.shape[-1] != self.global_obs_dim:
                invalid_seqs += 1
                continue

            advantages, returns = self._compute_gae_for_sequence(seq)
            seq['advantages'] = advantages
            seq['returns'] = returns
            seq['local_obs'] = local.to(self.device)
            seq['global_obs'] = global_obs.to(self.device)
            seq['actions'] = seq['actions'].to(self.device)
            seq['logprobs'] = seq['logprobs'].to(self.device)
            seq_data_list.append(seq)
            total_steps += len(seq['local_obs'])

        if invalid_seqs > 0:
            print(f"[PPO] Skipped {invalid_seqs} sequence(s) with outdated dims "
                  f"(expected local={self.obs_dim}, global={self.global_obs_dim})")

        if not seq_data_list:
            del self.ready_sequences[:n_train]
            return {"updated": False, "reason": "no_valid_sequences"}

        # Normalize advantages GLOBALLY across all sequences
        all_advantages = torch.cat([seq['advantages'] for seq in seq_data_list], dim=0)
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std()

        for seq in seq_data_list:
            seq['advantages'] = (seq['advantages'] - adv_mean) / (adv_std + 1e-8)

        # 2. Sequence-level batching (올바른 TBPTT!)
        num_seqs = len(seq_data_list)
        seq_batch_size = max(1, min(self.minibatch // 32, num_seqs))  # 시퀀스 단위 배치 크기

        for epoch in range(epochs):
            # Shuffle sequences (NOT steps!)
            seq_indices = torch.randperm(num_seqs)

            for batch_start in range(0, num_seqs, seq_batch_size):
                batch_end = min(batch_start + seq_batch_size, num_seqs)
                batch_seq_indices = seq_indices[batch_start:batch_end]

                # Gather sequences for this batch
                batch_seqs = [seq_data_list[i] for i in batch_seq_indices]
                seq_lengths = [len(seq['local_obs']) for seq in batch_seqs]
                max_len = max(seq_lengths)
                batch_size = len(batch_seqs)

                # Prepare padded tensors
                padded_local = torch.zeros(batch_size, max_len, self.obs_dim, device=self.device)
                padded_global = torch.zeros(batch_size, max_len, self.global_obs_dim, device=self.device)
                padded_actions = torch.zeros(batch_size, max_len, 1, device=self.device)
                padded_old_logp = torch.zeros(batch_size, max_len, device=self.device)
                padded_adv = torch.zeros(batch_size, max_len, device=self.device)
                padded_ret = torch.zeros(batch_size, max_len, device=self.device)

                # Stack init hidden states
                init_actor_h_list = []
                init_critic_h_list = []

                for i, seq in enumerate(batch_seqs):
                    seq_len = seq_lengths[i]

                    # Fill padded tensors
                    padded_local[i, :seq_len] = seq['local_obs']
                    padded_global[i, :seq_len] = seq['global_obs']
                    padded_actions[i, :seq_len] = seq['actions']
                    padded_old_logp[i, :seq_len] = seq['logprobs']
                    padded_adv[i, :seq_len] = seq['advantages']
                    padded_ret[i, :seq_len] = seq['returns']

                    # Get init hidden states
                    init_actor_h_list.append(
                        seq['init_actor_h'].to(self.device) if seq['init_actor_h'] is not None
                        else torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
                    )
                    init_critic_h_list.append(
                        seq['init_critic_h'].to(self.device) if seq['init_critic_h'] is not None
                        else torch.zeros(self.num_layers, 1, self.hidden_size * 2, device=self.device)
                    )

                # Stack hidden states: (num_layers, batch_size, hidden)
                # Each tensor is (num_layers, 1, hidden), concat on dim=1 (batch dimension)
                init_actor_h = torch.cat(init_actor_h_list, dim=1)  # (num_layers, batch, hidden)
                init_critic_h = torch.cat(init_critic_h_list, dim=1)  # (num_layers, batch, hidden*2)

                # GRU forward pass with pack_padded_sequence
                # Actor
                actor_embedded = torch.tanh(self.actor_embedding(padded_local))
                packed_actor = torch.nn.utils.rnn.pack_padded_sequence(
                    actor_embedded, seq_lengths, batch_first=True, enforce_sorted=False
                )
                packed_actor_out, _ = self.actor_gru(packed_actor, init_actor_h)
                actor_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_actor_out, batch_first=True)

                # Critic
                critic_embedded = torch.tanh(self.critic_embedding(padded_global))
                packed_critic = torch.nn.utils.rnn.pack_padded_sequence(
                    critic_embedded, seq_lengths, batch_first=True, enforce_sorted=False
                )
                packed_critic_out, _ = self.critic_gru(packed_critic, init_critic_h)
                critic_out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_critic_out, batch_first=True)

                # Get predictions
                mean = self.actor_head(actor_out)  # (batch, max_len, 1)
                value_pred = self.critic_head(critic_out).squeeze(-1)  # (batch, max_len)

                # Compute log probs
                new_logp = self._tanh_gaussian_logprob(mean, padded_actions)

                # Create mask for valid positions
                mask = torch.zeros(batch_size, max_len, device=self.device)
                for i, length in enumerate(seq_lengths):
                    mask[i, :length] = 1.0

                # Apply mask and flatten
                new_logp_valid = (new_logp * mask).view(-1)[mask.view(-1).bool()]
                old_logp_valid = (padded_old_logp * mask).view(-1)[mask.view(-1).bool()]
                adv_valid = (padded_adv * mask).view(-1)[mask.view(-1).bool()]
                ret_valid = (padded_ret * mask).view(-1)[mask.view(-1).bool()]
                value_pred_valid = (value_pred * mask).view(-1)[mask.view(-1).bool()]

                # PPO loss
                ratio = (new_logp_valid - old_logp_valid).exp()
                surr1 = ratio * adv_valid
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_valid
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = (value_pred_valid - ret_valid).pow(2).mean()

                # Entropy
                log_std_clamped = torch.clamp(self.log_std, min=self.log_std_min)
                entropy = (0.5 + 0.5 * math.log(2 * math.pi)) + log_std_clamped
                entropy = entropy.sum()

                # Total loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                # Optimize
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor_embedding.parameters()) +
                    list(self.actor_gru.parameters()) +
                    list(self.actor_head.parameters()) +
                    [self.log_std] +
                    list(self.critic_embedding.parameters()) +
                    list(self.critic_gru.parameters()) +
                    list(self.critic_head.parameters()),
                    0.5
                )
                self.opt.step()

        # Drop only the sequences we just trained on, keep the rest queued
        del self.ready_sequences[:n_train]

        return {
            "updated": True,
            "trained_on": total_steps,
            "num_sequences": num_seqs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item()
        }

    def reset_hidden_states(self, episode_start=False):
        """Reset hidden states at episode boundaries."""
        if RESET_HIDDEN_ON_EPISODE and episode_start:
            self.actor_hidden_manager.reset()
            self.critic_hidden_manager.reset()
            print("[RecurrentPPO] Reset all hidden states (episode start)")

    def save(self, path):
        """Save model checkpoint."""
        ckpt = {
            "actor_embedding": self.actor_embedding.state_dict(),
            "actor_gru": self.actor_gru.state_dict(),
            "actor_head": self.actor_head.state_dict(),
            "critic_embedding": self.critic_embedding.state_dict(),
            "critic_gru": self.critic_gru.state_dict(),
            "critic_head": self.critic_head.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "opt": self.opt.state_dict(),
            "actor_hiddens": self.actor_hidden_manager.get_all_states(),
            "critic_hiddens": self.critic_hidden_manager.get_all_states(),
            "meta": {
                "policy_id": self.policy_id,
                "obs_dim": self.obs_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "recurrent_core": "gru",
            },
        }
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(ckpt, str(p))
            return {"ok": True, "path": str(p)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def load(self, path):
        """Load model checkpoint."""
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.actor_embedding.load_state_dict(ckpt.get("actor_embedding", {}))
            actor_gru_state = ckpt.get("actor_gru")
            if actor_gru_state is None and ckpt.get("actor_lstm") is not None:
                raise RuntimeError("Checkpoint contains LSTM weights; retrain for GRU architecture.")
            if actor_gru_state:
                self.actor_gru.load_state_dict(actor_gru_state)
            self.actor_head.load_state_dict(ckpt.get("actor_head", {}))
            self.critic_embedding.load_state_dict(ckpt.get("critic_embedding", {}))
            critic_gru_state = ckpt.get("critic_gru")
            if critic_gru_state is None and ckpt.get("critic_lstm") is not None:
                raise RuntimeError("Checkpoint contains LSTM weights; retrain for GRU architecture.")
            if critic_gru_state:
                self.critic_gru.load_state_dict(critic_gru_state)
            self.critic_head.load_state_dict(ckpt.get("critic_head", {}))

            ls = ckpt.get("log_std")
            if ls is not None:
                with torch.no_grad():
                    self.log_std.copy_(ls.to(self.device))

            opt = ckpt.get("opt")
            if opt:
                try:
                    self.opt.load_state_dict(opt)
                except Exception:
                    pass

            # Restore hidden states
            actor_hiddens = ckpt.get("actor_hiddens")
            if actor_hiddens:
                self.actor_hidden_manager.set_all_states(actor_hiddens)

            critic_hiddens = ckpt.get("critic_hiddens")
            if critic_hiddens:
                self.critic_hidden_manager.set_all_states(critic_hiddens)

            meta = ckpt.get("meta", {})
            pid = meta.get("policy_id")
            if pid:
                self.policy_id = pid

            return {"ok": True, "meta": meta}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class HeuristicPolicy:
    """Fallback heuristic policy when PyTorch is not available."""
    def __init__(self):
        pass

    def act_batch(self, obs_batch, delta_limit, keys=None, global_features=None):
        out = []
        for obs in obs_batch:
            c = float(obs[0] or 0.0)  # contacts_norm
            p = float(obs[1] or 0.0)  # pred
            self_util = float(obs[4] if len(obs) > 4 else 0.0)  # self_buf_util

            base = c * max(0.0, 1.0 - self_util) * max(0.0, 1.0 - p)
            if delta_limit is not None and delta_limit > 0:
                delta = delta_limit * base
                delta = max(-delta_limit, min(delta_limit, delta))
            else:
                delta = base
            out.append(delta)
        return out

    def update(self, *args, **kwargs):
        return {}

    def reset_hidden_states(self, episode_start=False):
        pass

    def save(self, path):
        return {"ok": False, "error": "Heuristic policy cannot be saved"}

    def load(self, path):
        return {"ok": False, "error": "Heuristic policy cannot be loaded"}


# HTTP Handler (reuses most of the logic from original drl_server.py)
class Handler(BaseHTTPRequestHandler):
    # Initialize policy
    if TORCH_OK:
        AGENT = RecurrentPPOPolicy(obs_dim=7, global_obs_dim=19,
                                   hidden_size=LSTM_HIDDEN_SIZE,
                                   num_layers=LSTM_NUM_LAYERS)
        BASE_POLICY_ID = AGENT.policy_id
    else:
        AGENT = HeuristicPolicy()
        BASE_POLICY_ID = "heuristic_v1"

    POLICY_ID = BASE_POLICY_ID

    if MODEL_PATH:
        try:
            if TORCH_OK:
                res = AGENT.load(MODEL_PATH)
                if res.get("ok", False):
                    print(f"[BOOT] Loaded R-MAPPO model from {MODEL_PATH}")
                else:
                    print(f"[BOOT] Failed to load model: {res.get('error')}")
            else:
                print(f"[BOOT] MODEL_PATH set but torch unavailable")
        except Exception as e:
            print(f"[BOOT] Exception loading model: {e}")
        else:
            if TORCH_OK:
                try:
                    Handler.AGENT.optimizer = optim.Adam(
                        Handler.AGENT.parameters(),
                        lr=Handler.AGENT.lr,
                        eps=Handler.AGENT.adam_eps
                    )
                    print("[BOOT] Optimizer reset after loading model")
                except Exception as opt_exc:
                    print(f"[BOOT] Failed to reset optimizer: {opt_exc}")

    # Episode tracking
    EP_CUR = None
    EP_REWARD_ACC = 0.0
    STATE_CACHE = {}
    BUF_TREND = deque(maxlen=64)
    EP_COUNT = 0
    EP_DELIVERED_SUM = 0.0
    EP_DROPPED_SUM = 0.0
    EP_PRESSURE_SUM = 0.0
    EP_RELAYED_SUM = 0.0
    TARGET_REWARD_APPLIED = False
    LAST_ACTIVE_HOSTS = set()
    TRAINING_ENABLED = not EVAL_ONLY
    MODEL_SAVED_AT = None
    LAST_SUCCESS_RATE = None
    SIGNAL_LOG_PATH = os.path.join("reports", "reward_logs", "reward_signal_stats.csv")
    SIGNAL_LOG_HEADER = "time,signal,count,min,max,mean,stddev\n"
    SIGNAL_LOG_READY = False

    @classmethod
    def _ensure_signal_log(cls):
        if cls.SIGNAL_LOG_READY:
            return
        directory = os.path.dirname(cls.SIGNAL_LOG_PATH)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if not os.path.exists(cls.SIGNAL_LOG_PATH):
            with open(cls.SIGNAL_LOG_PATH, "w", encoding="utf-8") as fh:
                fh.write(cls.SIGNAL_LOG_HEADER)
        cls.SIGNAL_LOG_READY = True

    @staticmethod
    def _compute_stats(values):
        count = len(values)
        if count == 0:
            return None
        min_v = min(values)
        max_v = max(values)
        mean = sum(values) / count
        if count > 1:
            var = sum((v - mean) ** 2 for v in values) / count
            std = math.sqrt(max(var, 0.0))
        else:
            std = 0.0
        return count, min_v, max_v, mean, std

    @classmethod
    def log_signal_stats(cls, sim_time, signal_values):
        cls._ensure_signal_log()
        rows = []
        for name, values in signal_values.items():
            stats = cls._compute_stats(values)
            if not stats:
                continue
            count, min_v, max_v, mean, std = stats
            rows.append(f"{sim_time},{name},{count},{min_v:.6f},{max_v:.6f},{mean:.6f},{std:.6f}\n")
        if rows:
            with open(cls.SIGNAL_LOG_PATH, "a", encoding="utf-8") as fh:
                fh.writelines(rows)

    def do_POST(self):
        # Episode end handler
        if self.path == "/episode_end":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8")) if length > 0 else {}
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(("Bad Request: %s" % e).encode("utf-8"))
                return

            sim_id = str(req.get("sim_id", ""))
            avg = float(req.get("average_delivery_rate", "nan"))
            delivered = int(req.get("delivered", 0))
            created = int(req.get("created", 0))

            Handler.EP_COUNT += 1

            # Reset hidden states at episode boundaries
            if TORCH_OK:
                Handler.AGENT.reset_hidden_states(episode_start=True)

            # Save model
            if TORCH_OK and Handler.EP_COUNT > 0:
                try:
                    os.makedirs(MODEL_DIR, exist_ok=True)
                    buffer_size = extract_buffer_size_mb(sim_id)

                    if buffer_size:
                        model_name = f"rmappo_buf{int(buffer_size)}_ep{Handler.EP_COUNT}.pt"
                    else:
                        model_name = f"rmappo_{sim_id or 'sim'}_ep{Handler.EP_COUNT}.pt"

                    path = MODEL_PATH or os.path.join(MODEL_DIR, model_name)
                    res = Handler.AGENT.save(path)
                    print(f"[EPISODE_END] Saved R-MAPPO model ep{Handler.EP_COUNT}: {res.get('path')}")

                    if Handler.EP_COUNT >= TOTAL_EPISODES:
                        Handler.TRAINING_ENABLED = False
                        print(f"[TRAINING_COMPLETE] Finished {TOTAL_EPISODES} episodes")
                except Exception as e:
                    print(f"[ERROR] Failed to save model: {e}")

            # Reset episode stats
            Handler.EP_DELIVERED_SUM = 0.0
            Handler.EP_DROPPED_SUM = 0.0
            Handler.EP_PRESSURE_SUM = 0.0
            Handler.EP_RELAYED_SUM = 0.0
            Handler.TARGET_REWARD_APPLIED = False
            Handler.LAST_ACTIVE_HOSTS = set()
            Handler.LAST_SUCCESS_RATE = None

            body = json.dumps({"ok": True, "episode": Handler.EP_COUNT}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # Main inference endpoint - reuse logic from original server
        # (The reward calculation and inference logic remains mostly the same)
        # For brevity, I'll include a simplified version here
        # In production, copy the full logic from original drl_server.py

        if self.path != "/infer_and_update":
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self.wfile.write(("Bad Request: %s" % e).encode("utf-8"))
            return

        # Extract request data
        delta_limit = float(req.get("delta_limit", 1.00))
        prev_tr = req.get("prev_transition", []) or []
        state_batch = req.get("state_batch", []) or []
        sim_time = int(float(req.get("time", 0)))
        sim_id = str(req.get("sim_id", ""))

        # Helper function
        def _to_float(val, default=0.0):
            try:
                f = float(val)
            except Exception:
                return default
            if math.isnan(f) or math.isinf(f):
                return default
            return f

        # Build rewards with delivery + Prophet alignment + overhead penalty
        rewards_per_host = {}
        rewards_per_key = {}
        done_keys = set()
        stress_keys = set()
        pbase_signals = []
        buffer_diff_signals = []
        success_rate_deltas = []
        def _scale_dense(weight: float, signal: float) -> float:
            """Scale dense reward signals directly by their magnitude."""
            if weight == 0.0 or signal == 0.0:
                return 0.0
            return weight * signal

        def _scale_success(weight: float, delta: float) -> float:
            """Clip success-rate delta so the weight remains a true multiplier."""
            if weight == 0.0 or delta == 0.0:
                return 0.0
            clipped = max(-1.0, min(1.0, delta))
            return weight * clipped

        for item in prev_tr:
            try:
                host = str(item.get("host", ""))
                if not host:
                    continue
                dest = str(item.get("dest", ""))
                key = f"{host}#{dest}" if dest else None

                delivered = _to_float(item.get("delivered", 0.0), 0.0)
                relayed = _to_float(item.get("relayed", 0.0), 0.0)
                drops = _to_float(item.get("drops", 0.0), 0.0)
                aborted = _to_float(item.get("aborted", 0.0), 0.0)

                reward = 0.0

                # 1) Delivery reward (only when local buffer has headroom)
                if (
                    delivered > 0
                    and DELIVERY_REWARD_WEIGHT != 0.0
                    and my_buf_norm <= DELIVERY_REWARD_UTIL_MAX
                ):
                    reward += _scale_dense(DELIVERY_REWARD_WEIGHT, min(delivered, 1.0))

                # 2) Prophet base comparison reward
                my_p_base = _to_float(item.get("p_base", 0.0), 0.0)
                my_buf_norm = _to_float(item.get("my_buffer_norm", 0.0), 0.0)
                neighbor_p_base = _to_float(item.get("neighbor_p_base", 0.0), 0.0)
                if (
                    PBASE_REWARD_WEIGHT != 0.0
                    and math.isfinite(my_p_base)
                    and math.isfinite(neighbor_p_base)
                ):
                    diff = neighbor_p_base - my_p_base
                    if abs(diff) >= PBASE_DIFF_EPS:
                        magnitude = abs(diff)
                        direction = 1.0 if diff > 0 else -1.0
                        action_aligned = (relayed > 0 and direction > 0) or (relayed == 0 and direction < 0)
                        signal = magnitude if action_aligned else -magnitude
                        pbase_signals.append(diff)
                        reward += _scale_dense(PBASE_REWARD_WEIGHT, signal)

                # 3) Heuristic-aligned reward: high predictability & ample buffer
                if (
                    HEURISTIC_REWARD_WEIGHT != 0.0
                    and relayed > 0
                    and my_p_base >= HEURISTIC_PBASE_MIN
                    and my_buf_norm <= HEURISTIC_BUF_MAX
                ):
                    # Scale by how much the node exceeds thresholds to mimic heuristic intent
                    p_scale = (my_p_base - HEURISTIC_PBASE_MIN) / max(1e-6, 1.0 - HEURISTIC_PBASE_MIN)
                    buf_scale = max(0.0, (HEURISTIC_BUF_MAX - my_buf_norm) / max(1e-6, HEURISTIC_BUF_MAX))
                    heuristic_signal = relayed * (0.5 * p_scale + 0.5 * buf_scale)
                    reward += _scale_dense(HEURISTIC_REWARD_WEIGHT, heuristic_signal)

                # 4) Buffer-vs-network average differential reward
                cached = Handler.STATE_CACHE.get(key) if key else None
                if (
                    BUFFER_DIFF_REWARD_WEIGHT != 0.0
                    and relayed > 0
                    and cached
                    and math.isfinite(my_buf_norm)
                ):
                    avg_buf = cached.get("buf_mean")
                    if avg_buf is not None and math.isfinite(avg_buf):
                        diff = my_buf_norm - avg_buf
                        buffer_diff_signals.append(diff)
                        reward += _scale_dense(BUFFER_DIFF_REWARD_WEIGHT, diff)

                if reward != 0.0:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + reward
                    if key:
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + reward

                if key and (delivered > 0 or drops > 0 or aborted > 0):
                    done_keys.add(key)
                if key and (drops > 0 or aborted > 0):
                    stress_keys.add(key)

                Handler.EP_DELIVERED_SUM += delivered
                Handler.EP_DROPPED_SUM += drops
                Handler.EP_RELAYED_SUM += relayed
            except Exception:
                continue

        # Prepare observations
        obs_batch = []
        global_features_batch = []
        keys = []
        filtered_items = []  # Track items that pass validation
        key_self_util = {}
        key_pred = {}
        key_pressure_diff = {}
        map_host_to_keys = defaultdict(list)

        for item in state_batch:
            try:
                host = str(item.get("host", ""))
                dest = str(item.get("dest", ""))

                c = _to_float(item.get("contacts_norm", 0.0), 0.0)
                p = _to_float(item.get("pred", 0.0), 0.0)
                buf_mean = _to_float(item.get("bufocc_mean"), 0.0)
                cap_norm = _to_float(item.get("capacity_norm"), 0.0)
                self_util = _to_float(item.get("self_buf_util"), 0.0)
                pressure_diff = _to_float(item.get("pressure_diff", 0.0), 0.0)
                rate_delta = _to_float(item.get("rate_delta", 0.0), 0.0)

                if any(math.isnan(x) or math.isinf(x) for x in (c, p, buf_mean, cap_norm, self_util, pressure_diff, rate_delta)):
                    continue

                obs_batch.append([c, p, buf_mean, cap_norm, self_util, pressure_diff, rate_delta])

                # Global features
                g_features = [
                    _to_float(item.get("global_active_conns", 0.0), 0.0),
                    _to_float(item.get("global_total_msgs", 0.0), 0.0),
                    _to_float(item.get("global_avg_buf", 0.0), 0.0),
                    _to_float(item.get("global_avg_contacts", 0.0), 0.0),
                    _to_float(item.get("global_msg_change_rate", 0.0), 0.0),
                    _to_float(item.get("global_msg_change_momentum", 0.0), 0.0),
                    _to_float(item.get("global_buf_util_change_rate", 0.0), 0.0),
                    _to_float(item.get("global_buf_util_momentum", 0.0), 0.0),
                    _to_float(item.get("global_avg_free_buf", 0.0), 0.0),
                    _to_float(item.get("global_high_util_frac", 0.0), 0.0),
                    _to_float(item.get("global_high_util_flag", 0.0), 0.0),
                    _to_float(item.get("global_avg_buf_delta", 0.0), 0.0),
                ]

                if any(math.isnan(x) or math.isinf(x) for x in g_features):
                    continue

                global_features_batch.append(g_features)

                k = f"{host}#{dest}"
                keys.append(k)
                filtered_items.append(item)  # Store item that passed validation
                map_host_to_keys[host].append(k)
                key_self_util[k] = self_util
                key_pred[k] = p
                key_pressure_diff[k] = pressure_diff
                Handler.STATE_CACHE[k] = {
                    "buf_mean": buf_mean,
                    "self_util": self_util,
                    "timestamp": sim_time,
                }
            except Exception:
                continue

        # Log step rewards
        if STEP_REWARD_TRACKING and rewards_per_host:
            buffer_size = extract_buffer_size_mb(sim_id)
            log_step_rewards(sim_time, rewards_per_host, buffer_size, Handler.EP_COUNT)

        if map_host_to_keys:
            Handler.LAST_ACTIVE_HOSTS = set(map_host_to_keys.keys())

        # Derive network-wide high load indicator from global features
        high_load_factor = 0.0
        high_load_active = False
        relay_load_active = False
        trend_alert = False
        success_rate = None
        avg_buf_value = None
        if global_features_batch:
            try:
                high_util = float(global_features_batch[0][9])
                high_load_factor = max(0.0, (high_util - HIGH_LOAD_UTIL_BASE) / max(1e-6, 1.0 - HIGH_LOAD_UTIL_BASE))
                high_load_factor = min(1.0, high_load_factor)
                flag_val = float(global_features_batch[0][10])
                high_load_active = (flag_val >= HIGH_LOAD_FLAG_THRESHOLD) or (high_util >= HIGH_LOAD_FLAG_THRESHOLD)
                relay_load_active = high_load_active or (high_load_factor >= RELAY_HIGH_LOAD_FACTOR_MIN)
                avg_buf_value = float(global_features_batch[0][2])
            except Exception:
                high_load_factor = 0.0
                high_load_active = False
                relay_load_active = False
                avg_buf_value = None

        # Trend detection: anticipate upcoming high load spikes
        if avg_buf_value is not None and math.isfinite(avg_buf_value):
            Handler.BUF_TREND.append((sim_time, avg_buf_value))
            if TREND_MIN_STEPS > 1 and len(Handler.BUF_TREND) >= TREND_MIN_STEPS:
                recent = list(Handler.BUF_TREND)[-TREND_MIN_STEPS:]
                deltas = [
                    recent[i + 1][1] - recent[i][1]
                    for i in range(len(recent) - 1)
                ]
                if deltas:
                    avg_slope = sum(deltas) / len(deltas)
                    if avg_slope >= TREND_DELTA_THRESHOLD:
                        trend_alert = True

        # Current cumulative success rate
        denom_sr = Handler.EP_DELIVERED_SUM + Handler.EP_DROPPED_SUM
        if denom_sr > 0:
            success_rate = Handler.EP_DELIVERED_SUM / denom_sr

        if (
            TARGET_REWARD_TIME > 0 and TARGET_REWARD_WEIGHT > 0
            and not Handler.TARGET_REWARD_APPLIED
            and sim_time >= TARGET_REWARD_TIME
            and success_rate is not None
        ):
            bonus = TARGET_REWARD_WEIGHT * success_rate
            target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
            if target_hosts and bonus != 0.0:
                per_host_bonus = bonus
                for host in target_hosts:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host_bonus
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host_bonus
                Handler.TARGET_REWARD_APPLIED = True
                print(f"[REWARD] t={sim_time:.1f}s episode bonus success_rate={success_rate:.3f} "
                      f"bonus={bonus:.3f} targets={len(target_hosts)}")

        # Debug: Log done_keys collection
        if done_keys:
            print(f"[DEBUG] t={sim_time:.1f}s: done_keys collected: {len(done_keys)} keys")
            if len(done_keys) <= 10:
                print(f"[DEBUG]   Keys: {sorted(list(done_keys))[:10]}")

        # Apply additional penalty during sustained high-load phases
        if HIGH_LOAD_PENALTY_WEIGHT > 0 and high_load_factor > 0 and map_host_to_keys:
            penalty_per_key = -HIGH_LOAD_PENALTY_WEIGHT * high_load_factor
            penalized_keys = 0
            for host, host_keys in map_host_to_keys.items():
                if not host_keys:
                    continue
                filtered_keys = [
                    k for k in host_keys
                    if key_self_util.get(k, 0.0) >= HIGH_LOAD_UTIL_THRESHOLD or k in stress_keys
                ]
                if not filtered_keys:
                    continue
                total_penalty = penalty_per_key * len(filtered_keys)
                rewards_per_host[host] = rewards_per_host.get(host, 0.0) + total_penalty
                for k in filtered_keys:
                    rewards_per_key[k] = rewards_per_key.get(k, 0.0) + penalty_per_key
                penalized_keys += len(filtered_keys)
            if penalized_keys > 0:
                print(f"[REWARD] t={sim_time:.1f}s high-load penalty factor={high_load_factor:.3f} "
                      f"per_key={penalty_per_key:.3f} penalized_keys={penalized_keys}")

        # Penalize hosts sitting in the mid-80% range with positive pressure even if high-load flag is off
        if MID_UTIL_PENALTY_WEIGHT > 0 and map_host_to_keys:
            mid_penalized = 0
            for host, host_keys in map_host_to_keys.items():
                if not host_keys:
                    continue
                for k in host_keys:
                    util = key_self_util.get(k, 0.0)
                    pressure = key_pressure_diff.get(k, 0.0)
                    if util >= MID_UTIL_THRESHOLD and pressure >= MID_UTIL_PRESSURE_MIN:
                        scale = (util - MID_UTIL_THRESHOLD) / max(1e-6, 1.0 - MID_UTIL_THRESHOLD)
                        penalty = -MID_UTIL_PENALTY_WEIGHT * scale
                        rewards_per_key[k] = rewards_per_key.get(k, 0.0) + penalty
                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + penalty
                        mid_penalized += 1
            if mid_penalized > 0:
                print(f"[REWARD] t={sim_time:.1f}s mid-util penalty applied={mid_penalized} "
                      f"threshold={MID_UTIL_THRESHOLD:.2f}")

        # Trend-based penalty: buffer rising rapidly before high-load flag triggers
        if TREND_PENALTY_WEIGHT > 0 and trend_alert and map_host_to_keys:
            trend_penalized = 0
            for host, host_keys in map_host_to_keys.items():
                if not host_keys:
                    continue
                for k in host_keys:
                    util = key_self_util.get(k, 0.0)
                    if util >= TREND_UTIL_THRESHOLD:
                        scale = (util - TREND_UTIL_THRESHOLD) / max(1e-6, 1.0 - TREND_UTIL_THRESHOLD)
                        penalty = -TREND_PENALTY_WEIGHT * scale
                        rewards_per_key[k] = rewards_per_key.get(k, 0.0) + penalty
                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + penalty
                        trend_penalized += 1
            if trend_penalized > 0:
                print(f"[REWARD] t={sim_time:.1f}s trend penalty applied={trend_penalized} "
                      f"avg_slope>= {TREND_DELTA_THRESHOLD:.3f}")

        # Penalize drops in cumulative success rate
        if success_rate is not None and Handler.LAST_SUCCESS_RATE is not None:
            delta_sr = success_rate - Handler.LAST_SUCCESS_RATE
            success_rate_deltas.append(delta_sr)
            target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
            if delta_sr < -SUCCESS_RATE_EPS and SUCCESS_RATE_WEIGHT > 0 and target_hosts:
                penalty = _scale_success(SUCCESS_RATE_WEIGHT, delta_sr)  # negative value
                for host in target_hosts:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + penalty
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + penalty
                print(f"[REWARD] t={sim_time:.1f}s success-rate penalty delta={delta_sr:.4f}")
            elif delta_sr > SUCCESS_RATE_EPS and SUCCESS_RATE_BONUS_WEIGHT > 0 and target_hosts:
                bonus = _scale_success(SUCCESS_RATE_BONUS_WEIGHT, delta_sr)
                for host in target_hosts:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + bonus
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + bonus
                print(f"[REWARD] t={sim_time:.1f}s success-rate bonus delta={delta_sr:.4f}")

        if success_rate is not None:
            Handler.LAST_SUCCESS_RATE = success_rate

        Handler.log_signal_stats(
            sim_time,
            {
                "pbase_signal": pbase_signals,
                "buffer_diff": buffer_diff_signals,
                "success_rate_delta": success_rate_deltas,
            },
        )

        # Bonus for low-util relays during detected high-load phases (Prophet-guided)
        if RELAY_BONUS_WEIGHT > 0 and relay_load_active and map_host_to_keys:
            relay_bonus_count = 0
            for host, host_keys in map_host_to_keys.items():
                if not host_keys:
                    continue
                for k in host_keys:
                    if (
                        key_self_util.get(k, 1.0) <= RELAY_LOW_UTIL_THRESHOLD
                        and key_pred.get(k, 0.0) >= RELAY_PRED_THRESHOLD
                    ):
                        bonus = RELAY_BONUS_WEIGHT * max(0.0, key_pred[k])
                        if bonus == 0.0:
                            continue
                        rewards_per_key[k] = rewards_per_key.get(k, 0.0) + bonus
                        host_reward = rewards_per_host.get(host, 0.0) + bonus
                        rewards_per_host[host] = host_reward
                        relay_bonus_count += 1
            if relay_bonus_count > 0:
                print(f"[REWARD] t={sim_time:.1f}s relay bonus applied={relay_bonus_count} "
                      f"bonus_per_key~{RELAY_BONUS_WEIGHT:.2f}")

        # Update policy
        update_result = {}
        if TORCH_OK and Handler.TRAINING_ENABLED:
            try:
                update_result = Handler.AGENT.update(
                    rewards_per_host,
                    rewards_per_key,
                    map_host_to_keys,
                    delta_limit,
                    done_keys=done_keys
                )
            except Exception as exc:
                print(f"[ERROR] PPO update failed at t={sim_time:.1f}s: {exc}")
                traceback.print_exc()
                # Best effort recovery to avoid wedging future updates
                try:
                    Handler.AGENT.last_actions.clear()
                except Exception:
                    pass
                update_result = {"updated": False}

            # Debug: Log update result
            if update_result.get("updated"):
                print(f"[DEBUG] t={sim_time:.1f}s: PPO update completed!")
                print(f"[DEBUG]   Losses: actor={update_result.get('actor_loss', 0):.4f}, "
                      f"critic={update_result.get('critic_loss', 0):.4f}")
            elif update_result.get("ready_sequences"):
                ready = update_result.get("ready_sequences", 0)
                print(f"[DEBUG] t={sim_time:.1f}s: {ready} sequences ready (need 32 for training)")

        # Infer actions
        if TORCH_OK:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit, keys, global_features_batch)
        else:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit)

        # Group actions by host
        per_host = defaultdict(list)
        flat_map = {}
        rewards_payload = {}
        for i, item in enumerate(filtered_items):
            try:
                host = str(item.get("host", ""))
                dest = str(item.get("dest", ""))
                delta = float(actions_raw[i])
                if delta_limit is not None and delta_limit > 0:
                    delta_c = round(max(-delta_limit, min(delta_limit, delta)), 4)
                else:
                    delta_c = round(delta, 4)
                per_host[host].append({"dest": dest, "delta": delta_c})
                flat_map[f"{host}#{dest}"] = delta_c
            except Exception:
                continue
        for key, value in rewards_per_key.items():
            try:
                rewards_payload[key] = float(value)
            except Exception:
                continue

        actions = [{"host": h, "per_message": v} for h, v in per_host.items()]
        resp = {
            "policy_id": Handler.POLICY_ID,
            "actions": actions,
            "actions_kv": flat_map,
            "rewards_kv": rewards_payload,
        }

        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    print(f"Starting R-MAPPO server at http://{HOST}:{PORT}/infer_and_update")
    print(f"  - PyTorch: {'enabled' if TORCH_OK else 'disabled'}")
    print(f"  - GRU hidden size: {LSTM_HIDDEN_SIZE}")
    print(f"  - GRU layers: {LSTM_NUM_LAYERS}")
    print(f"  - Truncated BPTT: {TRUNCATED_BPTT_LEN} steps")
    print(f"  - Mode: {'EVAL_ONLY' if EVAL_ONLY else 'TRAINING'}")

    srv = HTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
