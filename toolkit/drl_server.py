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
   "p_base": float,                # ⭐ 순수 PROPHET 예측도 (delta 적용 전) - Reward 계산용
   "neighbor_p_base": float,       # 실제 릴레이 이웃의 순수 PROPHET 예측도 (있으면 사용)
   "max_neighbor_p_base": float,   # 네트워크 내 최대 이웃 p_base (진단/옵션 보상용)
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
  - (선택) "neighbor_p_base": 접촉한 실제 이웃의 예측도 (릴레이 이벤트 시 기록)
  - (선택) "max_neighbor_p_base": 네트워크 내 최대 이웃 예측도 (진단/옵션 보상)
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import glob
import inspect
import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import os
import random
import pathlib
import time
import numpy as np
import traceback

# ============================================================================
# 보상 & 학습 파라미터 설정
# ============================================================================
# 기본값은 아래 os.environ.get()의 두 번째 인자에 정의되어 있습니다.
# 환경 변수로 override 가능: export PBASE_REWARD_WEIGHT=2.0
#
# [활성화된 보상 - ACTIVE]
#   - Prophet Base Comparison (neighbor-based): 1.0
#   - Buffer Differential: 1.0
#   - Success Rate: 0.1 (penalty) / 1.0 (bonus)
#
# [비활성화 - DISABLED]
#   - Delivery, Overhead, High Load, Relay, Mid-Util, Trend, Heuristic
# ============================================================================

# Allow optional CPU-only mode via env; default uses available CUDA
FORCE_CPU = os.environ.get("DRL_FORCE_CPU", "false").lower() in ("1", "true", "yes")
if FORCE_CPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

HOST = os.environ.get("DRL_HOST", "127.0.0.1")
PORT = int(float(os.environ.get("DRL_PORT", "5010")))

# R-MAPPO specific configurations
LSTM_HIDDEN_SIZE = int(os.environ.get("LSTM_HIDDEN_SIZE", "128"))
LSTM_NUM_LAYERS = int(os.environ.get("LSTM_NUM_LAYERS", "1"))
TRUNCATED_BPTT_LEN = int(os.environ.get("TRUNCATED_BPTT_LEN", "80"))  # Sequence length (~40 min @100s sampling)
RESET_HIDDEN_ON_EPISODE = os.environ.get("RESET_HIDDEN_ON_EPISODE", "true").lower() == "true"

# Action logging configuration
PPO_DEBUG_ACTIONS = os.environ.get("PPO_DEBUG_ACTIONS", "false").lower() == "true"

# Step-by-step reward tracking configuration
STEP_REWARD_TRACKING = os.environ.get("STEP_REWARD_TRACKING", "true").lower() == "true"
REPORT_OUTPUT_DIR = os.environ.get("REPORT_OUTPUT_DIR", os.environ.get("REPORT_DIR", "reports"))
SCRIPT_BASENAME = os.path.basename(__file__) if "__file__" in globals() else "drl_server_rmappo.py"

# ============================================================================
# Reward Configuration Variables (수정은 위 주석의 기본값 참고)
# ============================================================================
PBASE_REWARD_WEIGHT = float(os.environ.get("PBASE_REWARD_WEIGHT", "0"))     # neighbor-based p_base
MAX_PBASE_REWARD_WEIGHT = float(os.environ.get("MAX_PBASE_REWARD_WEIGHT", "0.0"))  # max-based p_base (disabled by default)
PBASE_DIFF_EPS = float(os.environ.get("PBASE_DIFF_EPS", "0.02"))
# Mid-buffer-zone P_base shaping (applies only when avg_buf is between thresholds)
PBASE_MID_RELAY_WEIGHT = float(os.environ.get("PBASE_MID_RELAY_WEIGHT", "0"))
PBASE_MID_ABORT_WEIGHT = float(os.environ.get("PBASE_MID_ABORT_WEIGHT", "0"))
BUFFER_DIFF_REWARD_WEIGHT = float(os.environ.get("BUFFER_DIFF_REWARD_WEIGHT", "0"))
BUFFER_DIFF_ABORT_WEIGHT = float(os.environ.get("BUFFER_DIFF_ABORT_WEIGHT", "0"))
BUFFER_DIFF_EMA_BETA = float(os.environ.get("BUFFER_DIFF_EMA_BETA", "0.9"))
if BUFFER_DIFF_EMA_BETA < 0.0:
    BUFFER_DIFF_EMA_BETA = 0.0
elif BUFFER_DIFF_EMA_BETA > 0.999:
    BUFFER_DIFF_EMA_BETA = 0.999
BUFFER_DIFF_SELF_EMA_BETA = float(os.environ.get("BUFFER_DIFF_SELF_EMA_BETA", "0.7"))
if BUFFER_DIFF_SELF_EMA_BETA < 0.0:
    BUFFER_DIFF_SELF_EMA_BETA = 0.0
elif BUFFER_DIFF_SELF_EMA_BETA > 0.999:
    BUFFER_DIFF_SELF_EMA_BETA = 0.999
BUFFER_DIFF_LOW_THRESHOLD = float(os.environ.get("BUFFER_DIFF_LOW_THRESHOLD", "0.4"))
BUFFER_DIFF_HIGH_THRESHOLD = float(os.environ.get("BUFFER_DIFF_HIGH_THRESHOLD", "0.85"))
if BUFFER_DIFF_LOW_THRESHOLD < 0.0:
    BUFFER_DIFF_LOW_THRESHOLD = 0.0
if BUFFER_DIFF_LOW_THRESHOLD > 1.0:
    BUFFER_DIFF_LOW_THRESHOLD = 1.0
if BUFFER_DIFF_HIGH_THRESHOLD < 0.0:
    BUFFER_DIFF_HIGH_THRESHOLD = 0.0
if BUFFER_DIFF_HIGH_THRESHOLD > 1.0:
    BUFFER_DIFF_HIGH_THRESHOLD = 1.0
if BUFFER_DIFF_LOW_THRESHOLD > BUFFER_DIFF_HIGH_THRESHOLD:
    BUFFER_DIFF_LOW_THRESHOLD, BUFFER_DIFF_HIGH_THRESHOLD = BUFFER_DIFF_HIGH_THRESHOLD, BUFFER_DIFF_LOW_THRESHOLD
SKIP_LOW_PRED_PENALTY_WEIGHT = float(os.environ.get("SKIP_LOW_PRED_PENALTY_WEIGHT", "0"))
SKIP_LOW_PRED_BUF_THRESHOLD = float(os.environ.get("SKIP_LOW_PRED_BUF_THRESHOLD", str(BUFFER_DIFF_LOW_THRESHOLD)))
if SKIP_LOW_PRED_BUF_THRESHOLD < 0.0:
    SKIP_LOW_PRED_BUF_THRESHOLD = 0.0
if SKIP_LOW_PRED_BUF_THRESHOLD > 1.0:
    SKIP_LOW_PRED_BUF_THRESHOLD = 1.0
SKIP_LOW_PRED_BUF_MODE = os.environ.get("SKIP_LOW_PRED_BUF_MODE", "global").strip().lower()
SKIP_LOW_PRED_GAUSS_SIGMA = float(os.environ.get("SKIP_LOW_PRED_GAUSS_SIGMA", "0.05"))
if SKIP_LOW_PRED_GAUSS_SIGMA <= 1e-6:
    SKIP_LOW_PRED_GAUSS_SIGMA = 1e-6
PEER_HIGH_PRED_PENALTY_WEIGHT = float(os.environ.get("PEER_HIGH_PRED_PENALTY_WEIGHT", "0.0"))
PEER_HIGH_PRED_BUF_THRESHOLD = float(os.environ.get("PEER_HIGH_PRED_BUF_THRESHOLD", str(BUFFER_DIFF_HIGH_THRESHOLD)))
if PEER_HIGH_PRED_BUF_THRESHOLD < 0.0:
    PEER_HIGH_PRED_BUF_THRESHOLD = 0.0
if PEER_HIGH_PRED_BUF_THRESHOLD > 1.0:
    PEER_HIGH_PRED_BUF_THRESHOLD = 1.0
PEER_HIGH_PRED_BUF_MODE = os.environ.get("PEER_HIGH_PRED_BUF_MODE", "global").strip().lower()
PEER_HIGH_PRED_GAUSS_SIGMA = float(os.environ.get("PEER_HIGH_PRED_GAUSS_SIGMA", "0.05"))
if PEER_HIGH_PRED_GAUSS_SIGMA <= 1e-6:
    PEER_HIGH_PRED_GAUSS_SIGMA = 1e-6
BUF_SLOPE_PENALTY_GAIN = float(os.environ.get("BUF_SLOPE_PENALTY_GAIN", "100.0"))
if BUF_SLOPE_PENALTY_GAIN < 0.0:
    BUF_SLOPE_PENALTY_GAIN = 0.0
DELIVERY_DELAY_PENALTY_WEIGHT = float(os.environ.get("DELIVERY_DELAY_PENALTY_WEIGHT", "0"))
DELIVERY_DELAY_DECAY = float(os.environ.get("DELIVERY_DELAY_DECAY", "3600.0"))
if DELIVERY_DELAY_DECAY <= 1e-6:
    DELIVERY_DELAY_DECAY = 1e-6
OVERHEAD_PENALTY_WEIGHT = float(os.environ.get("OVERHEAD_PENALTY_WEIGHT", "1.0"))
OVERHEAD_DECAY = float(os.environ.get("OVERHEAD_DECAY", "5.0"))
if OVERHEAD_DECAY <= 1e-6:
    OVERHEAD_DECAY = 1e-6
DROP_TTL_PENALTY_WEIGHT = float(os.environ.get("DROP_TTL_PENALTY_WEIGHT", "0"))
DROP_TTL_GAUSS_SIGMA = float(os.environ.get("DROP_TTL_GAUSS_SIGMA", "0.08"))
if DROP_TTL_GAUSS_SIGMA <= 1e-6:
    DROP_TTL_GAUSS_SIGMA = 1e-6

# Reward mixer (non-fix reward network) configuration
NON_FIX_REWARD_NETWORK = os.environ.get("NON_FIX_REWARD_NETWORK", "false").lower() in ("1", "true", "yes")
REWARD_MIXER_CHANNELS = ("skip_low_pred_pen", "peer_high_pred_pen", "sr_sparse_bonus")
REWARD_MIXER_LR = float(os.environ.get("REWARD_MIXER_LR", "5e-4"))
REWARD_MIXER_HIDDEN = int(os.environ.get("REWARD_MIXER_HIDDEN", "32"))
REWARD_MIXER_TARGET_TEMP = float(os.environ.get("REWARD_MIXER_TARGET_TEMP", "1.5"))
REWARD_MIXER_EMA_BETA = float(os.environ.get("REWARD_MIXER_EMA_BETA", "0.9"))
if REWARD_MIXER_EMA_BETA < 0.0:
    REWARD_MIXER_EMA_BETA = 0.0
elif REWARD_MIXER_EMA_BETA > 0.999:
    REWARD_MIXER_EMA_BETA = 0.999
REWARD_MIXER_GRAD_CLIP = float(os.environ.get("REWARD_MIXER_GRAD_CLIP", "1.0"))
REWARD_MIXER_ZERO_BOOST = float(os.environ.get("REWARD_MIXER_ZERO_BOOST", "0.3"))
REWARD_MIXER_CONTEXT_DIM = 6

# ============================================================================
# RATE-BASED ZONE DETECTION PARAMETERS (Data-driven thresholds)
# Based on analysis of Episodes 1-27 buffer change rates
# ============================================================================
BUF_RATE_EMA_ALPHA = float(os.environ.get("BUF_RATE_EMA_ALPHA", "0.1"))
# Threshold values based on 2σ (standard deviation) analysis:
#   - Actual data range: -0.000083 to +0.000028
#   - Mean: 0.000000, Std: 0.000001
#   - 2σ threshold captures statistically significant changes (95.4% coverage)
#   - Episode 1: 3.4% high zone, 2.2% low zone (actively penalizing)
#   - Episode 19: 1.4% high zone, 1.2% low zone (stable, less penalty)
RATE_HIGH_THRESHOLD = float(os.environ.get("RATE_HIGH_THRESHOLD", "0.000000"))  # +2σ
RATE_LOW_THRESHOLD = float(os.environ.get("RATE_LOW_THRESHOLD", "-0.000000"))   # -2σ , 0.000002
# ============================================================================

SUCCESS_RATE_WEIGHT = float(os.environ.get("SUCCESS_RATE_WEIGHT", "0"))   # 원래 0.1으로 함함
SUCCESS_RATE_EPS = float(os.environ.get("SUCCESS_RATE_EPS", "0.00"))
SUCCESS_RATE_BONUS_WEIGHT = float(os.environ.get("SUCCESS_RATE_BONUS_WEIGHT", "0"))    # 원래 1.0으로 함함ㅁㅁ
SUCCESS_RATE_SMOOTH_BETA = float(os.environ.get("SUCCESS_RATE_SMOOTH_BETA", "0.99"))  # baseline: 장기 평균
STEP_SUCCESS_RATE_BETA = float(os.environ.get("STEP_SUCCESS_RATE_BETA", "0.5"))      # 현재 SR: 빠르게 반응
if SUCCESS_RATE_SMOOTH_BETA < 0.0:
    SUCCESS_RATE_SMOOTH_BETA = 0.0
elif SUCCESS_RATE_SMOOTH_BETA > 0.999:
    SUCCESS_RATE_SMOOTH_BETA = 0.999
SUCCESS_RATE_EXP_GAIN = float(os.environ.get("SUCCESS_RATE_EXP_GAIN", "4.0"))
if SUCCESS_RATE_EXP_GAIN < 0.0:
    SUCCESS_RATE_EXP_GAIN = 0.0
SUCCESS_RATE_EXP_CLAMP = float(os.environ.get("SUCCESS_RATE_EXP_CLAMP", "10.0"))
if SUCCESS_RATE_EXP_CLAMP < 0.0:
    SUCCESS_RATE_EXP_CLAMP = 0.0
DELIVERY_REWARD_WEIGHT = float(os.environ.get("DELIVERY_REWARD_WEIGHT", "0"))
DELIVERY_REWARD_UTIL_MAX = float(os.environ.get("DELIVERY_REWARD_UTIL_MAX", "0.65"))
HIGH_LOAD_PENALTY_WEIGHT = float(os.environ.get("HIGH_LOAD_PENALTY_WEIGHT", "0.0"))
HIGH_LOAD_UTIL_BASE = float(os.environ.get("HIGH_LOAD_UTIL_BASE", "0.35"))
HIGH_LOAD_UTIL_THRESHOLD = float(os.environ.get("HIGH_LOAD_UTIL_THRESHOLD", "0.65"))
HIGH_LOAD_FLAG_THRESHOLD = float(os.environ.get("HIGH_LOAD_FLAG_THRESHOLD", "0.45"))
TARGET_REWARD_TIME = int(float(os.environ.get("TARGET_REWARD_TIME", "0")))
TARGET_REWARD_WEIGHT = float(os.environ.get("TARGET_REWARD_WEIGHT", "0"))
SR_BONUS_WEIGHT = float(os.environ.get("SR_BONUS_WEIGHT", "0"))
RELAY_BONUS_WEIGHT = float(os.environ.get("RELAY_BONUS_WEIGHT", "0"))
RELAY_OCC_SIGMA = float(os.environ.get("RELAY_OCC_SIGMA", "0.145"))
if RELAY_OCC_SIGMA <= 1e-6:
    RELAY_OCC_SIGMA = 1e-6
RELAY_LOW_OCC_GAIN = float(os.environ.get("RELAY_LOW_OCC_GAIN", "1.0"))
RELAY_HIGH_OCC_GAIN = float(os.environ.get("RELAY_HIGH_OCC_GAIN", "1.0"))
if RELAY_LOW_OCC_GAIN < 0.0:
    RELAY_LOW_OCC_GAIN = 0.0
if RELAY_HIGH_OCC_GAIN < 0.0:
    RELAY_HIGH_OCC_GAIN = 0.0
RELAY_TTL_DECAY_K = float(os.environ.get("RELAY_TTL_DECAY_K", "0.0"))
if RELAY_TTL_DECAY_K < 0.0:
    RELAY_TTL_DECAY_K = 0.0
ACTION_GAIN_LOW = float(os.environ.get("ACTION_GAIN_LOW", "1.0"))
ACTION_GAIN_HIGH = float(os.environ.get("ACTION_GAIN_HIGH", "4.0"))
ACTION_GAIN_THRESHOLD = float(os.environ.get("ACTION_GAIN_THRESHOLD", "0.8"))
ACTION_SECOND_TANH = os.environ.get("ACTION_SECOND_TANH", "false").lower() in ("1", "true", "yes")
GAUSSIAN_ENTROPY_OFFSET = 0.5 + 0.5 * math.log(2 * math.pi)
# Log-std initialization presets keep exploration within practical bounds.
# - LOG_STD_INIT (direct override) takes precedence.
# - LOG_STD_INIT_PRESET=zero (default) starts near σ≈1.0 (log_std≈0.0).
# - LOG_STD_INIT_PRESET=negative starts near σ≈0.36 (log_std≈-1.0).
_log_std_init_raw = os.environ.get("LOG_STD_INIT", "").strip()
if _log_std_init_raw:
    try:
        LOG_STD_INIT = float(_log_std_init_raw)
    except Exception:
        LOG_STD_INIT = 0.0
else:
    preset = os.environ.get("LOG_STD_INIT_PRESET", "zero").strip().lower()
    if preset in ("negative", "neg", "low", "-1", "-1.0"):
        LOG_STD_INIT = -1.0
    else:
        LOG_STD_INIT = 0.0
_log_std_max_raw = os.environ.get("LOG_STD_MAX", "").strip()
if _log_std_max_raw:
    try:
        LOG_STD_MAX = float(_log_std_max_raw)
    except Exception:
        LOG_STD_MAX = 0.5
else:
    LOG_STD_MAX = None
MID_RANGE_REWARD = float(os.environ.get("MID_RANGE_REWARD", "0.5"))
MID_UTIL_PENALTY_WEIGHT = float(os.environ.get("MID_UTIL_PENALTY_WEIGHT", "0.0"))
MID_UTIL_THRESHOLD = float(os.environ.get("MID_UTIL_THRESHOLD", "0.80"))
MID_UTIL_PRESSURE_MIN = float(os.environ.get("MID_UTIL_PRESSURE_MIN", "0.01"))
HEURISTIC_REWARD_WEIGHT = float(os.environ.get("HEURISTIC_REWARD_WEIGHT", "0.0"))
HEURISTIC_PBASE_MIN = float(os.environ.get("HEURISTIC_PBASE_MIN", "0.6"))
HEURISTIC_BUF_MAX = float(os.environ.get("HEURISTIC_BUF_MAX", "0.55"))
TREND_PENALTY_WEIGHT = float(os.environ.get("TREND_PENALTY_WEIGHT", "0.0"))
TREND_UTIL_THRESHOLD = float(os.environ.get("TREND_UTIL_THRESHOLD", "0.75"))
TREND_DELTA_THRESHOLD = float(os.environ.get("TREND_DELTA_THRESHOLD", "0.01"))
TREND_MIN_STEPS = int(float(os.environ.get("TREND_MIN_STEPS", "5")))

# ===== Buffer × PBase Intersect Reward (Hard AND gating) =====
# Disabled by default; enable via env to experiment
BUFFER_PBASE_AND_WEIGHT = float(os.environ.get("BUFFER_PBASE_AND_WEIGHT", "0.00"))
BUFFER_PBASE_AND_NEG_WEIGHT = float(os.environ.get("BUFFER_PBASE_AND_NEG_WEIGHT", "0"))
BUFFER_PBASE_EPS_BUF = float(os.environ.get("BUFFER_PBASE_EPS_BUF", "0.00"))
# If not set, falls back to PBASE_DIFF_EPS
BUFFER_PBASE_EPS_PBASE = float(os.environ.get("BUFFER_PBASE_EPS_PBASE", str(PBASE_DIFF_EPS)))

# ===== Success-rate rewards (sparse + stepwise can be combined) =====
# Sparse: immediate per-delivery bonus tied directly to each successful arrival.
# Stepwise: small penalties/bonuses based on the change in cumulative success rate.
# By default, both are allowed concurrently to provide dense shaping plus a high-value
# terminal objective.
SPARSE_SUCCESS_REWARD = os.environ.get("SPARSE_SUCCESS_REWARD", "true").lower() in ("1", "true", "yes")
# Increase default sparse final-success-rate reward weight (3x)
SR_SPARSE_WEIGHT = float(os.environ.get("SR_SPARSE_WEIGHT", "0.1"))
# If true, divide the sparse reward by the number of active hosts to keep total bounded
SR_SPARSE_NORMALIZE = os.environ.get("SR_SPARSE_NORMALIZE", "false").lower() in ("1", "true", "yes")
SR_SPARSE_COOLDOWN = float(os.environ.get("SR_SPARSE_COOLDOWN", "5.0"))
if SR_SPARSE_COOLDOWN < 0.0:
    SR_SPARSE_COOLDOWN = 0.0
# Occupancy Gaussian parameters for sparse reward
SR_SPARSE_OCC_CENTER = float(os.environ.get("SR_SPARSE_OCC_CENTER", "0.6"))
SR_SPARSE_OCC_SIGMA = float(os.environ.get("SR_SPARSE_OCC_SIGMA", "0.145"))
if SR_SPARSE_OCC_SIGMA <= 1e-6:
    SR_SPARSE_OCC_SIGMA = 1e-6
STATE_BONUS_WEIGHT = float(os.environ.get("STATE_BONUS_WEIGHT", "0"))
STATE_BONUS_SIGMA = float(os.environ.get("STATE_BONUS_SIGMA", "0.145"))
if STATE_BONUS_SIGMA <= 1e-6:
    STATE_BONUS_SIGMA = 1e-6
STATE_DELTA_BONUS_WEIGHT = float(os.environ.get("STATE_DELTA_BONUS_WEIGHT", "0.0"))
STATE_DELTA_BONUS_SIGMA = float(os.environ.get("STATE_DELTA_BONUS_SIGMA", "0.145"))
if STATE_DELTA_BONUS_SIGMA <= 1e-6:
    STATE_DELTA_BONUS_SIGMA = 1e-6
# Optional EMA smoothing for sparse reward contributions (0=disabled)
SR_SPARSE_EMA_BETA = float(os.environ.get("SR_SPARSE_EMA_BETA", "0"))
if SR_SPARSE_EMA_BETA < 0.0:
    SR_SPARSE_EMA_BETA = 0.0
elif SR_SPARSE_EMA_BETA > 0.999:
    SR_SPARSE_EMA_BETA = 0.999
# Combine sparse success reward with stepwise success-rate deltas (enabled by default).
# When true, stepwise deltas remain active even if SPARSE_SUCCESS_REWARD is enabled.
SR_COMBINE_WITH_STEP = os.environ.get("SR_COMBINE_WITH_STEP", "true").lower() in ("1", "true", "yes")
# Optional scale for stepwise weights only when combined with sparse.
SR_COMBINED_STEP_SCALE = float(os.environ.get("SR_COMBINED_STEP_SCALE", "1.0"))

# ===== Success-Rate EMA Normalization (Exploration Moving Average friendly) =====
# Enable EMA-based normalization so late-episode small success-rate changes still
# produce meaningful reward signals while clamping to avoid blow-ups.
SR_NORM_ENABLE = os.environ.get("SR_NORM_ENABLE", "true").lower() in ("1", "true", "yes")
SR_DELTA_EMA_BETA = float(os.environ.get("SR_DELTA_EMA_BETA", "0.9"))
SR_EMA_WARMUP_STEPS = int(float(os.environ.get("SR_EMA_WARMUP_STEPS", "10")))
# Use 2*EPS as default denominator floor to prevent excessive gain when deltas are tiny
SR_NORM_MIN_DENOM = float(os.environ.get("SR_NORM_MIN_DENOM", str(max(1e-6, 2 * SUCCESS_RATE_EPS))))
SR_NORM_MAX_GAIN = float(os.environ.get("SR_NORM_MAX_GAIN", "2.0"))

# Optional: adapt EPS based on EMA magnitude
SR_ADAPTIVE_EPS = os.environ.get("SR_ADAPTIVE_EPS", "true").lower() in ("1", "true", "yes")
SR_EPS_K = float(os.environ.get("SR_EPS_K", "0.5"))
SR_EPS_FLOOR = float(os.environ.get("SR_EPS_FLOOR", "0.001"))


def _parse_float_list_env(raw_value, default):
    """Utility to parse comma-separated env lists."""
    if not raw_value:
        return list(default)
    try:
        values = [float(x.strip()) for x in raw_value.split(",") if x.strip()]
        return values or list(default)
    except Exception:
        return list(default)


# ===== Meta-Agent (LR/Entropy Controller) defaults =====
META_AGENT_ENABLED = os.environ.get("META_AGENT_ENABLED", "true").lower() in ("1", "true", "yes")
META_AGENT_LR_CHOICES = _parse_float_list_env(
    os.environ.get("META_AGENT_LR_CHOICES", ""),
    [5e-4, 3e-4, 1.5e-4, 8e-5, 4e-5],
)
META_AGENT_ENTROPY_CHOICES = _parse_float_list_env(
    os.environ.get("META_AGENT_ENT_CHOICES", ""),
    [0.05, 0.035, 0.025, 0.015],
)
META_AGENT_REWARD_KEYS = (
    "success_penalty",
    "success_bonus",
    "skip_low_pred_pen",
    "peer_high_pred_pen",
    "sr_sparse_bonus",
)
META_AGENT_STATE_KEYS = (
    "avg_delivery_rate",
    "delivery_ema",
    "delta_delivery",
    "total_reward",
    "reward_ema",
    "delta_reward",
    "success_penalty",
    "success_bonus",
    "skip_low_pred_pen",
    "peer_high_pred_pen",
    "sr_sparse_bonus",
    "lr",
    "entropy_coef",
    "last_action_id",
)
META_AGENT_EMA_BETA = float(os.environ.get("META_AGENT_EMA_BETA", "0.6"))
META_AGENT_LR_MIN = float(os.environ.get("META_AGENT_LR_MIN", str(min(META_AGENT_LR_CHOICES))))
META_AGENT_LR_MAX = float(os.environ.get("META_AGENT_LR_MAX", str(max(META_AGENT_LR_CHOICES))))
META_AGENT_ENTROPY_MIN = float(os.environ.get("META_AGENT_ENTROPY_MIN", str(min(META_AGENT_ENTROPY_CHOICES))))
META_AGENT_ENTROPY_MAX = float(os.environ.get("META_AGENT_ENTROPY_MAX", str(max(META_AGENT_ENTROPY_CHOICES))))
META_AGENT_BUFFER_CAP = int(os.environ.get("META_AGENT_BUFFER_CAP", "1024"))
META_AGENT_BATCH_SIZE = int(os.environ.get("META_AGENT_BATCH_SIZE", "64"))
META_AGENT_GAMMA = float(os.environ.get("META_AGENT_GAMMA", "0.95"))
META_AGENT_TAU = float(os.environ.get("META_AGENT_TAU", "0.01"))
META_AGENT_ACTOR_LR = float(os.environ.get("META_AGENT_ACTOR_LR", "3e-4"))
META_AGENT_CRITIC_LR = float(os.environ.get("META_AGENT_CRITIC_LR", "5e-4"))
META_AGENT_ACTION_NOISE = float(os.environ.get("META_AGENT_ACTION_NOISE", "0.15"))
META_AGENT_LOG_DIR = os.path.join("reports", "meta_agent_logs")
META_AGENT_DELIVERY_WEIGHT = float(os.environ.get("META_AGENT_DELIVERY_WEIGHT", "1.0"))
META_AGENT_TOTAL_REWARD_WEIGHT = float(os.environ.get("META_AGENT_TOTAL_REWARD_WEIGHT", "1.0"))
META_AGENT_TOTAL_REWARD_NORM = float(os.environ.get("META_AGENT_TOTAL_REWARD_NORM", "100.0"))
META_CONTROLLER = None

def _final_report_value(sim_id, report_name, value_index):
    """Read the last numeric value from a report file."""
    if not sim_id or not report_name or REPORT_OUTPUT_DIR is None:
        return None
    sim_path = pathlib.Path(sim_id)
    base_name = sim_path.name if sim_path.name else sim_id
    filename = f"{base_name}_{report_name}.txt"

    def _candidate_paths():
        bases = []
        if REPORT_OUTPUT_DIR:
            bases.append(REPORT_OUTPUT_DIR)
        extra = os.environ.get("REPORT_SEARCH_DIRS", "")
        for raw in extra.split(os.pathsep):
            candidate = raw.strip()
            if candidate:
                bases.append(candidate)
        bases.append(".")
        seen = set()
        for base in bases:
            base = os.path.abspath(base)
            if base in seen:
                continue
            seen.add(base)
            direct = os.path.join(base, filename)
            if os.path.isfile(direct):
                yield direct
                continue
            glob_pattern = os.path.join(base, "**", filename)
            for match in glob.iglob(glob_pattern, recursive=True):
                if os.path.isfile(match):
                    yield match

    report_path = None
    for candidate in _candidate_paths():
        report_path = candidate
        break
    if report_path is None or not os.path.isfile(report_path):
        return None
    last_value = None
    try:
        with open(report_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) <= value_index:
                    continue
                try:
                    val = float(parts[value_index])
                except Exception:
                    continue
                if math.isnan(val) or math.isinf(val):
                    continue
                last_value = val
    except Exception:
        return None
    return last_value


def get_final_overhead_from_reports(sim_id):
    """Fetch final overhead ratio from OverheadPerIntervalReport if available."""
    return _final_report_value(sim_id, "OverheadPerIntervalReport", 4)


def get_final_latency_from_reports(sim_id):
    """Fetch final average latency from LatencyPerIntervalReport if available."""
    return _final_report_value(sim_id, "LatencyPerIntervalReport", 3)


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


def log_reward_parameters(episode_num, sim_time, buffer_size_mb=None):
    """Record raw reward parameter weights once per episode."""
    try:
        if Handler.LAST_PARAM_LOG_EP == episode_num:
            return
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)
        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"reward_parameters_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "reward_parameters_rmappo.csv")
        write_header = not os.path.exists(log_file)
        cols = [
            "episode","sim_time",
            "SUCCESS_RATE_WEIGHT","SUCCESS_RATE_BONUS_WEIGHT",
            "SKIP_LOW_PRED_PENALTY_WEIGHT","PEER_HIGH_PRED_PENALTY_WEIGHT","DROP_TTL_PENALTY_WEIGHT"
        ]
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(cols) + "\n")
            row = {
                "episode": episode_num,
                "sim_time": sim_time,
                "SUCCESS_RATE_WEIGHT": SUCCESS_RATE_WEIGHT,
                "SUCCESS_RATE_BONUS_WEIGHT": SUCCESS_RATE_BONUS_WEIGHT,
                "SKIP_LOW_PRED_PENALTY_WEIGHT": SKIP_LOW_PRED_PENALTY_WEIGHT,
                "PEER_HIGH_PRED_PENALTY_WEIGHT": PEER_HIGH_PRED_PENALTY_WEIGHT,
                "DROP_TTL_PENALTY_WEIGHT": DROP_TTL_PENALTY_WEIGHT,
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")
        Handler.LAST_PARAM_LOG_EP = episode_num
    except Exception:
        pass


def accumulate_episode_totals(episode_num, components):
    """Accumulate per-episode sums of the active reward components."""
    try:
        totals = Handler.CORE_EP_TOTALS[episode_num]
        totals["success_penalty"] += components.get("success_penalty", 0.0)
        totals["success_bonus"] += components.get("success_bonus", 0.0)
        totals["skip_low_pred_pen"] += components.get("skip_low_pred_pen", 0.0)
        totals["peer_high_pred_pen"] += components.get("peer_high_pred_pen", 0.0)
        totals["drop_ttl_penalty"] += components.get("drop_ttl_penalty", 0.0)
        totals["sr_sparse_bonus"] += components.get("sr_sparse_bonus", 0.0)
        totals["sr_sparse_count"] += components.get("sr_sparse_count", 0.0)
        totals["relay_bonus"] += components.get("relay_bonus", 0.0)
        totals["state_bonus"] += components.get("state_bonus", 0.0)
        totals["state_delta_bonus"] += components.get("state_delta_bonus", 0.0)
        totals["delivery_delay_pen"] += components.get("delivery_delay_pen", 0.0)
        totals["overhead_penalty"] += components.get("overhead_penalty", 0.0)
        totals["total_reward"] += components.get("total_hosts_reward", 0.0)
    except Exception:
        pass


def log_episode_core_totals(episode_num, buffer_size_mb=None):
    """Write aggregated per-episode totals for the core reward components."""
    try:
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)
        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"episode_reward_core_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "episode_reward_core_rmappo.csv")
        cols = [
            "episode","success_penalty","success_bonus",
            "skip_low_pred_pen","peer_high_pred_pen","drop_ttl_penalty",
            "sr_sparse_bonus","sr_sparse_count","relay_bonus","state_bonus","state_delta_bonus",
            "delivery_delay_pen","overhead_penalty","total_reward"
        ]
        write_header = not os.path.exists(log_file)
        totals = Handler.CORE_EP_TOTALS.pop(episode_num, {
            "success_penalty": 0.0,
            "success_bonus": 0.0,
            "skip_low_pred_pen": 0.0,
            "peer_high_pred_pen": 0.0,
            "drop_ttl_penalty": 0.0,
            "sr_sparse_bonus": 0.0,
            "sr_sparse_count": 0.0,
            "relay_bonus": 0.0,
            "state_bonus": 0.0,
            "state_delta_bonus": 0.0,
            "delivery_delay_pen": 0.0,
            "overhead_penalty": 0.0,
            "total_reward": 0.0,
        })
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(cols) + "\n")
            row = {
                "episode": episode_num,
                "success_penalty": totals.get("success_penalty", 0.0),
                "success_bonus": totals.get("success_bonus", 0.0),
                "skip_low_pred_pen": totals.get("skip_low_pred_pen", 0.0),
                "peer_high_pred_pen": totals.get("peer_high_pred_pen", 0.0),
                "drop_ttl_penalty": totals.get("drop_ttl_penalty", 0.0),
                "sr_sparse_bonus": totals.get("sr_sparse_bonus", 0.0),
                "sr_sparse_count": totals.get("sr_sparse_count", 0.0),
                "relay_bonus": totals.get("relay_bonus", 0.0),
                "state_bonus": totals.get("state_bonus", 0.0),
                "state_delta_bonus": totals.get("state_delta_bonus", 0.0),
                "delivery_delay_pen": totals.get("delivery_delay_pen", 0.0),
                "overhead_penalty": totals.get("overhead_penalty", 0.0),
                "total_reward": totals.get("total_reward", 0.0),
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    except Exception:
        pass


def log_episode_reward(episode_num, sim_id, total_reward, avg_delivery_rate=None,
                       delivered=None, created=None, avg_overhead=None,
                       avg_delay=None, buffer_size_mb=None):
    """Log per-episode total reward (and optional delivery summary) to CSV.

    Columns: episode, sim_id, total_reward, avg_delivery_rate, delivered, created,
             avg_overhead, avg_delay
    Writes to reports/reward_logs/episode_rewards[_buf{MB}M]_rmappo.csv
    """
    try:
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"episode_rewards_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "episode_rewards_rmappo.csv")

        write_header = not os.path.exists(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("episode,sim_id,total_reward,avg_delivery_rate,delivered,created,avg_overhead,avg_delay\n")
            # Normalize None to empty string for optional fields
            avg_str = "" if avg_delivery_rate is None or (isinstance(avg_delivery_rate, float) and math.isnan(avg_delivery_rate)) else str(avg_delivery_rate)
            deliv_str = "" if delivered is None else str(delivered)
            created_str = "" if created is None else str(created)
            overhead_str = "" if avg_overhead is None or (isinstance(avg_overhead, float) and math.isnan(avg_overhead)) else str(avg_overhead)
            delay_str = "" if avg_delay is None or (isinstance(avg_delay, float) and math.isnan(avg_delay)) else str(avg_delay)
            f.write(f"{episode_num},{sim_id},{total_reward},{avg_str},{deliv_str},{created_str},{overhead_str},{delay_str}\n")
    except Exception:
        pass

def log_episode_losses(episode_num, sim_id, actor_loss_avg=None, critic_loss_avg=None,
                       updates=0, entropy_avg=None, buffer_size_mb=None):
    """Log per-episode average actor/critic losses to CSV.

    Columns: episode, sim_id, actor_loss_avg, critic_loss_avg, updates, entropy_avg
    Writes to reports/train_logs/episode_losses[_buf{MB}M]_rmappo.csv
    """
    try:
        log_dir = os.path.join("reports", "train_logs")
        os.makedirs(log_dir, exist_ok=True)

        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"episode_losses_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "episode_losses_rmappo.csv")

        write_header = not os.path.exists(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("episode,sim_id,actor_loss_avg,critic_loss_avg,updates,entropy_avg\n")

            def _fmt(x):
                try:
                    return "" if x is None else str(round(float(x), 6))
                except Exception:
                    return ""

            f.write(
                f"{episode_num},{sim_id},{_fmt(actor_loss_avg)},{_fmt(critic_loss_avg)},{int(updates)},{_fmt(entropy_avg)}\n"
            )
    except Exception:
        pass

def log_update_metrics(sim_time, metrics: dict, buffer_size_mb=None, episode_num=1):
    """Append per-update PPO diagnostics to CSV for analysis.

    Columns: episode,sim_time,ratio_mean,ratio_std,kl,std_mean,adv_var
    """
    try:
        log_dir = os.path.join("reports", "train_logs")
        os.makedirs(log_dir, exist_ok=True)

        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"update_metrics_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "update_metrics_rmappo.csv")

        write_header = not os.path.exists(log_file)
        cols = ["episode","sim_time","ratio_mean","ratio_std","kl","std_mean","adv_var"]
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(cols) + "\n")
            row = {
                "episode": episode_num,
                "sim_time": sim_time,
                "ratio_mean": metrics.get("ratio_mean", 0.0),
                "ratio_std": metrics.get("ratio_std", 0.0),
                "kl": metrics.get("kl", 0.0),
                "std_mean": metrics.get("std_mean", 0.0),
                "adv_var": metrics.get("adv_var", 0.0),
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    except Exception:
        pass

def log_step_reward_components(sim_time, components, buffer_size_mb=None, episode_num=1):
    """Log per-step reward component totals for analysis/plotting.

    components: dict with keys like 'pbase','pbase_max','pbase_mid_relay','pbase_mid_abort',
                'buffer_diff','skip_low_pred_pen','peer_high_pred_pen','success_delta',
                'delivery','heuristic','high_load_penalty','mid_util_penalty','trend_penalty',
                'relay_bonus','sparse_delivery','target_bonus','total_hosts_reward'
                (any missing treated as 0).
    """
    if not STEP_REWARD_TRACKING:
        return

    try:
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"step_reward_components_buf{int(buffer_size_mb)}M_rmappo.csv")
            log_file_detailed = os.path.join(log_dir, f"step_reward_components_detailed_buf{int(buffer_size_mb)}M_rmappo.csv")
            log_file_core = os.path.join(log_dir, f"step_reward_core_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "step_reward_components_rmappo.csv")
            log_file_detailed = os.path.join(log_dir, "step_reward_components_detailed_rmappo.csv")
            log_file_core = os.path.join(log_dir, "step_reward_core_rmappo.csv")

        # Episode numbering inside handler is zero-based until episode_end.
        # Shift by +1 so logs use human-friendly 1-based episodes.
        ep_id = (episode_num or 0) + 1

        # Ensure stable column order
        cols = [
            "episode","sim_time",
            "pbase","pbase_max","pbase_mid_relay","pbase_mid_abort","buffer_diff",
            "skip_low_pred_pen","peer_high_pred_pen","drop_ttl_penalty","success_penalty","success_bonus","success_delta",
            "delivery","heuristic","high_load_penalty","state_bonus","state_delta_bonus",
            "mid_util_penalty","trend_penalty","relay_bonus",
            "sparse_delivery","target_bonus","total_hosts_reward"
        ]

        write_header = not os.path.exists(log_file)
        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(cols) + "\n")

            row = {
                "episode": ep_id,
                "sim_time": sim_time,
                "pbase": components.get("pbase", 0.0),
                "pbase_max": components.get("pbase_max", 0.0),
                "pbase_mid_relay": components.get("pbase_mid_relay", 0.0),
                "pbase_mid_abort": components.get("pbase_mid_abort", 0.0),
                "buffer_diff": components.get("buffer_diff", 0.0),
                "skip_low_pred_pen": components.get("skip_low_pred_pen", 0.0),
                "peer_high_pred_pen": components.get("peer_high_pred_pen", 0.0),
                "drop_ttl_penalty": components.get("drop_ttl_penalty", 0.0),
                "success_penalty": components.get("success_penalty", 0.0),
                "success_bonus": components.get("success_bonus", 0.0),
                "success_delta": components.get("success_delta", 0.0),
                "delivery": components.get("delivery", 0.0),
                "heuristic": components.get("heuristic", 0.0),
                "high_load_penalty": components.get("high_load_penalty", 0.0),
                "state_bonus": components.get("state_bonus", 0.0),
                "state_delta_bonus": components.get("state_delta_bonus", 0.0),
                "mid_util_penalty": components.get("mid_util_penalty", 0.0),
                "trend_penalty": components.get("trend_penalty", 0.0),
                "relay_bonus": components.get("relay_bonus", 0.0),
                "sparse_delivery": components.get("sparse_delivery", 0.0),
                "target_bonus": components.get("target_bonus", 0.0),
                "total_hosts_reward": components.get("total_hosts_reward", 0.0),
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")

        # Log a core file with only the four active reward parameters for quick inspection
        core_cols = [
            "episode","sim_time",
            "success_penalty","success_bonus","skip_low_pred_pen","peer_high_pred_pen","drop_ttl_penalty",
            "sr_sparse_bonus","sr_sparse_count","relay_bonus","state_bonus","state_delta_bonus","total_reward"
        ]
        write_core_header = not os.path.exists(log_file_core)
        with open(log_file_core, "a", encoding="utf-8") as f_core:
            if write_core_header:
                f_core.write(",".join(core_cols) + "\n")
            core_row = {
                "episode": ep_id,
                "sim_time": sim_time,
                "success_penalty": components.get("success_penalty", 0.0),
                "success_bonus": components.get("success_bonus", 0.0),
                "skip_low_pred_pen": components.get("skip_low_pred_pen", 0.0),
                "peer_high_pred_pen": components.get("peer_high_pred_pen", 0.0),
                "drop_ttl_penalty": components.get("drop_ttl_penalty", 0.0),
                "sr_sparse_bonus": components.get("sr_sparse_bonus", 0.0),
                "sr_sparse_count": components.get("sr_sparse_count", 0.0),
                "relay_bonus": components.get("relay_bonus", 0.0),
                "state_bonus": components.get("state_bonus", 0.0),
                "state_delta_bonus": components.get("state_delta_bonus", 0.0),
                "total_reward": components.get("total_hosts_reward", 0.0),
            }
            f_core.write(",".join(str(core_row[c]) for c in core_cols) + "\n")

        # Log per-episode reward parameter weights (unscaled) for auditing
        log_reward_parameters(ep_id, sim_time, buffer_size_mb)

        # Aggregate per-episode totals for core reward parameters
        accumulate_episode_totals(ep_id, components)

        # Also write a detailed version with separate success bonus/penalty columns to distinguish contributions
        cols_det = [
            "episode","sim_time",
            "pbase","pbase_max","pbase_mid_relay","pbase_mid_abort","buffer_diff","skip_low_pred_pen","peer_high_pred_pen","drop_ttl_penalty",
            "success_bonus","success_penalty","success_delta",
            "delivery","heuristic","high_load_penalty","state_bonus","state_delta_bonus",
            "mid_util_penalty","trend_penalty","relay_bonus",
            "sparse_delivery","target_bonus","total_hosts_reward"
        ]
        write_header_det = not os.path.exists(log_file_detailed)
        with open(log_file_detailed, "a", encoding="utf-8") as f2:
            if write_header_det:
                f2.write(",".join(cols_det) + "\n")
            row2 = {
                "episode": ep_id,
                "sim_time": sim_time,
                "pbase": components.get("pbase", 0.0),
                "pbase_max": components.get("pbase_max", 0.0),
                "pbase_mid_relay": components.get("pbase_mid_relay", 0.0),
                "pbase_mid_abort": components.get("pbase_mid_abort", 0.0),
                "buffer_diff": components.get("buffer_diff", 0.0),
                "skip_low_pred_pen": components.get("skip_low_pred_pen", 0.0),
                "peer_high_pred_pen": components.get("peer_high_pred_pen", 0.0),
                "drop_ttl_penalty": components.get("drop_ttl_penalty", 0.0),
                "success_bonus": components.get("success_bonus", 0.0),
                "success_penalty": components.get("success_penalty", 0.0),
                "success_delta": components.get("success_delta", 0.0),
                "delivery": components.get("delivery", 0.0),
                "heuristic": components.get("heuristic", 0.0),
                "high_load_penalty": components.get("high_load_penalty", 0.0),
                "state_bonus": components.get("state_bonus", 0.0),
                "state_delta_bonus": components.get("state_delta_bonus", 0.0),
                "mid_util_penalty": components.get("mid_util_penalty", 0.0),
                "trend_penalty": components.get("trend_penalty", 0.0),
                "relay_bonus": components.get("relay_bonus", 0.0),
                "sparse_delivery": components.get("sparse_delivery", 0.0),
                "target_bonus": components.get("target_bonus", 0.0),
                "total_hosts_reward": components.get("total_hosts_reward", 0.0),
            }
            f2.write(",".join(str(row2[c]) for c in cols_det) + "\n")
    except Exception:
        pass


class EpisodeReplayBuffer:
    """Simple replay buffer for episode-level transitions."""

    def __init__(self, capacity):
        self.capacity = max(1, int(capacity))
        self.buffer = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state):
        self.buffer.append(
            (
                np.array(state, dtype=np.float32),
                np.array(action, dtype=np.float32),
                float(reward),
                np.array(next_state, dtype=np.float32),
            )
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = zip(*batch)
        return (
            np.stack(states),
            np.stack(actions),
            np.array(rewards, dtype=np.float32).reshape(-1, 1),
            np.stack(next_states),
        )


class MetaAgentController:
    """Off-policy controller that learns LR/entropy settings from episode summaries."""

    def __init__(self, log_dir):
        self.state_keys = tuple(META_AGENT_STATE_KEYS)
        self.reward_keys = tuple(META_AGENT_REWARD_KEYS)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.state_dim = len(self.state_keys)
        self.action_dim = 2  # lr, entropy
        self.lr_min = META_AGENT_LR_MIN
        self.lr_max = META_AGENT_LR_MAX
        self.ent_min = META_AGENT_ENTROPY_MIN
        self.ent_max = META_AGENT_ENTROPY_MAX
        self.lr_choices = sorted(set(META_AGENT_LR_CHOICES)) or [self.lr_min]
        self.entropy_choices = sorted(set(META_AGENT_ENTROPY_CHOICES)) or [self.ent_min]
        self.noise_std = META_AGENT_ACTION_NOISE
        self.gamma = META_AGENT_GAMMA
        self.tau = META_AGENT_TAU
        self.ema_beta = META_AGENT_EMA_BETA
        self.buffer = EpisodeReplayBuffer(META_AGENT_BUFFER_CAP)
        self.batch_size = META_AGENT_BATCH_SIZE
        self.actor = self._build_actor().to(self.device)
        self.actor_target = self._build_actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = self._build_critic().to(self.device)
        self.critic_target = self._build_critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        beta = float(os.environ.get("META_ADAM_BETA", os.environ.get("META_AGENT_ADAM_BETA", "0.9")))
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=META_AGENT_ACTOR_LR, betas=(beta, beta))
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=META_AGENT_CRITIC_LR, betas=(beta, beta))
        self.prev_state = None
        self.prev_action = None
        self.prev_delivery = None
        self.prev_total_reward = None
        self.delivery_ema = None
        self.reward_ema = None
        self.last_action_id = 0
        self.log_dir = log_dir
        self.log_file = os.path.join(self.log_dir, "episode_meta.csv")
        self.log_header = ["episode", "reward", "lr", "entropy_coef", "applied"] + list(self.state_keys)
        self.delivery_weight = META_AGENT_DELIVERY_WEIGHT
        self.total_reward_weight = META_AGENT_TOTAL_REWARD_WEIGHT
        self.total_reward_norm = max(1e-6, META_AGENT_TOTAL_REWARD_NORM)

    def _build_actor(self):
        return nn.Sequential(
            nn.Linear(self.state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim),
            nn.Tanh(),
        )

    def _build_critic(self):
        return nn.Sequential(
            nn.Linear(self.state_dim + self.action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def sync_from_values(self, current_lr, current_entropy):
        lr_norm = self._normalize(current_lr, self.lr_min, self.lr_max)
        ent_norm = self._normalize(current_entropy, self.ent_min, self.ent_max)
        lr_idx = self._closest_choice_index(current_lr, self.lr_choices)
        ent_idx = self._closest_choice_index(current_entropy, self.entropy_choices)
        self.last_action_id = self._compose_action_id(lr_idx, ent_idx)
        self.prev_action = np.array([lr_norm, ent_norm], dtype=np.float32)

    def _normalize(self, value, min_v, max_v):
        span = max(1e-8, max_v - min_v)
        norm = (float(value) - min_v) / span
        return max(-1.0, min(1.0, (norm * 2.0) - 1.0))

    def _denormalize(self, norm_value, min_v, max_v):
        span = max_v - min_v
        return min_v + ((float(norm_value) + 1.0) * 0.5 * span)

    def _compose_action_id(self, lr_idx, ent_idx):
        lr_bucket_count = max(1, len(self.lr_choices))
        ent_bucket_count = max(1, len(self.entropy_choices))
        return int(lr_idx * ent_bucket_count + ent_idx)

    def _closest_choice_index(self, value, choices):
        if not choices:
            return 0
        best_idx = 0
        best_dist = float("inf")
        for idx, candidate in enumerate(choices):
            dist = abs(float(value) - candidate)
            if dist < best_dist:
                best_idx = idx
                best_dist = dist
        return best_idx

    def _select_choice_from_norm(self, norm_value, choices):
        if not choices:
            return 0.0, 0
        if len(choices) == 1:
            return choices[0], 0
        frac = (float(norm_value) + 1.0) * 0.5
        frac = max(0.0, min(1.0, frac))
        idx = int(round(frac * (len(choices) - 1)))
        idx = max(0, min(len(choices) - 1, idx))
        return choices[idx], idx

    def _sanitize(self, value, default=0.0):
        try:
            val = float(value)
        except Exception:
            return default
        if math.isnan(val) or math.isinf(val):
            return default
        return val

    def _update_ema(self, ema_value, new_value):
        if ema_value is None:
            return new_value
        beta = max(0.0, min(0.999, self.ema_beta))
        return (beta * ema_value) + ((1.0 - beta) * new_value)

    def _log_episode(self, episode, reward_value, lr_value, entropy_value, state_vec, applied):
        try:
            os.makedirs(self.log_dir, exist_ok=True)
            write_header = not os.path.exists(self.log_file)
            with open(self.log_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(",".join(self.log_header) + "\n")
                row = {
                    "episode": int(episode),
                    "reward": reward_value,
                    "lr": lr_value,
                    "entropy_coef": entropy_value,
                    "applied": int(applied),
                }
                for key, value in zip(self.state_keys, state_vec):
                    row[key] = value
                f.write(",".join(str(row.get(col, "")) for col in self.log_header) + "\n")
        except Exception:
            pass

    def _compose_state(self, reward_components, total_reward, current_lr, entropy_coef, delivery_rate):
        components = {}
        src = reward_components or {}
        for key in self.reward_keys:
            components[key] = self._sanitize(src.get(key, 0.0))

        delivery = self._sanitize(delivery_rate)
        total_reward = self._sanitize(total_reward)
        delta_delivery = 0.0 if self.prev_delivery is None else (delivery - self.prev_delivery)
        delta_reward = 0.0 if self.prev_total_reward is None else (total_reward - self.prev_total_reward)
        self.delivery_ema = self._update_ema(self.delivery_ema, delivery)
        self.reward_ema = self._update_ema(self.reward_ema, total_reward)

        state_map = {
            "avg_delivery_rate": delivery,
            "delivery_ema": self.delivery_ema if self.delivery_ema is not None else delivery,
            "delta_delivery": delta_delivery,
            "total_reward": total_reward,
            "reward_ema": self.reward_ema if self.reward_ema is not None else total_reward,
            "delta_reward": delta_reward,
            "success_penalty": components.get("success_penalty", 0.0),
            "success_bonus": components.get("success_bonus", 0.0),
            "skip_low_pred_pen": components.get("skip_low_pred_pen", 0.0),
            "peer_high_pred_pen": components.get("peer_high_pred_pen", 0.0),
            "sr_sparse_bonus": components.get("sr_sparse_bonus", 0.0),
            "lr": self._sanitize(current_lr, self.lr_min),
            "entropy_coef": self._sanitize(entropy_coef, self.ent_min),
            "last_action_id": float(self.last_action_id),
        }
        state_vec = np.array([state_map[key] for key in self.state_keys], dtype=np.float32)
        self.prev_delivery = delivery
        self.prev_total_reward = total_reward
        return state_vec, state_map

    def _compute_meta_reward(self, delivery_rate, total_reward):
        delivery_clean = self._sanitize(delivery_rate, 0.0)
        total_clean = self._sanitize(total_reward, 0.0)
        total_norm = total_clean / self.total_reward_norm
        meta_reward = (delivery_clean * self.delivery_weight) + (total_norm * self.total_reward_weight)
        return meta_reward

    def _select_action(self, state_vec):
        state_tensor = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        action = self.actor(state_tensor).detach().cpu().numpy()[0]
        if self.noise_std > 0.0:
            noise = np.random.normal(0.0, self.noise_std, size=self.action_dim)
            action = np.clip(action + noise, -1.0, 1.0)
        return action.astype(np.float32)

    def _train(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.float32, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            next_actions = self.actor_target(next_states_t)
            q_target = self.critic_target(torch.cat([next_states_t, next_actions], dim=-1))
            q_target = rewards_t + (self.gamma * q_target)

        q_val = self.critic(torch.cat([states_t, actions_t], dim=-1))
        critic_loss = F.mse_loss(q_val, q_target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_actions = self.actor(states_t)
        actor_loss = -self.critic(torch.cat([states_t, actor_actions], dim=-1)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target_net, source_net):
        for target_param, src_param in zip(target_net.parameters(), source_net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + src_param.data * self.tau
            )

    def step(
        self,
        episode,
        delivery_rate,
        total_reward,
        reward_components,
        entropy_coef,
        current_lr,
    ):
        state_vec, state_map = self._compose_state(
            reward_components,
            total_reward,
            current_lr,
            entropy_coef,
            delivery_rate,
        )
        reward_value = self._compute_meta_reward(delivery_rate, total_reward)
        if self.prev_state is not None and self.prev_action is not None:
            self.buffer.push(self.prev_state, self.prev_action, reward_value, state_vec)
            self._train()

        action_vec = self._select_action(state_vec)
        lr_value, lr_idx = self._select_choice_from_norm(action_vec[0], self.lr_choices)
        entropy_value, ent_idx = self._select_choice_from_norm(action_vec[1], self.entropy_choices)
        lr_value = max(self.lr_min, min(self.lr_max, lr_value))
        entropy_value = max(self.ent_min, min(self.ent_max, entropy_value))
        self.last_action_id = self._compose_action_id(lr_idx, ent_idx)

        self.prev_state = state_vec
        self.prev_action = action_vec

        self._log_episode(episode, reward_value, lr_value, entropy_value, state_vec, True)

        return {
            "applied": True,
            "reason": "offpolicy_meta",
            "lr": lr_value,
            "entropy_coef": entropy_value,
        }


def log_sparse_delivery_events(sim_time, events, buffer_size_mb=None, episode_num=1):
    """Log every sparse reward grant for debugging/analysis."""
    if not STEP_REWARD_TRACKING or not events:
        return

    try:
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"sparse_delivery_events_buf{int(buffer_size_mb)}M_rmappo.csv")
        else:
            log_file = os.path.join(log_dir, "sparse_delivery_events_rmappo.csv")

        cols = [
            "episode","sim_time","host","dest","delivered","reward",
            "active_hosts","norm_factor","ttl_ratio","avg_hops","avg_buf"
        ]
        write_header = not os.path.exists(log_file)

        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write(",".join(cols) + "\n")
            for ev in events:
                row = {
                    "episode": episode_num,
                    "sim_time": sim_time,
                    "host": ev.get("host", ""),
                    "dest": ev.get("dest", ""),
                    "delivered": ev.get("delivered", 0.0),
                    "reward": ev.get("reward", 0.0),
                    "active_hosts": ev.get("active_hosts", 0),
                    "norm_factor": ev.get("norm_factor", 1.0),
                    "ttl_ratio": ev.get("ttl_ratio", 0.0),
                    "avg_hops": ev.get("avg_hops", 0.0),
                    "avg_buf": ev.get("avg_buf", 0.0),
                }
                f.write(",".join(str(row[c]) for c in cols) + "\n")
    except Exception:
        pass


def main_print(message):
    """Print startup info with contextual line numbers for quick log correlation."""
    try:
        caller = inspect.currentframe().f_back
        line = caller.f_lineno if caller else -1
        prefix = f"[{SCRIPT_BASENAME}:L{line}] "
    except Exception:
        prefix = ""
    print(f"{prefix}{message}")

# Training configuration
EPISODE_SECONDS = int(float(os.environ.get("EPISODE_SECONDS", "57600")))
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "1000"))
SIM_END_SECONDS = int(float(os.environ.get("SIM_END_SECONDS", "57600000")))

# Checkpoint control
MODEL_DIR = os.environ.get("MODEL_DIR", "models_rmappo")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() in ("1", "true", "yes")
NO_TRAINING_CAP = os.environ.get("NO_TRAINING_CAP", "false").lower() in ("1", "true", "yes")

# Stability/exploration schedules (env-overridable)
ENTROPY_COEF_START = float(os.environ.get("ENTROPY_COEF_START", "0.001"))
ENTROPY_COEF_END = float(os.environ.get("ENTROPY_COEF_END", "0.001"))
ENTROPY_DECAY_START_EP = int(os.environ.get("ENTROPY_DECAY_START_EP", "50"))
ENTROPY_DECAY_END_EP = int(os.environ.get("ENTROPY_DECAY_END_EP", str(TOTAL_EPISODES)))
EPISODIC_UPDATES = os.environ.get("EPISODIC_UPDATES", "true").lower() in ("1", "true", "yes")
EPISODIC_MAX_SEQUENCES = int(os.environ.get("EPISODIC_MAX_SEQUENCES", "0"))
STEP_MAX_SEQUENCES = int(os.environ.get("STEP_MAX_SEQUENCES", "36"))
LR_SCHEDULE = [
    (1, 200, 3e-4),
    (201, 400, 3e-4),
    (401, 600, 3e-4),
    (601, 800, 3e-4),
    (801, 1000, 3e-4),
]
LR_SCHED_START = float(os.environ.get("PPO_LR_START", str(LR_SCHEDULE[0][2])))
LR_SCHED_END = float(os.environ.get("PPO_LR_END", str(LR_SCHEDULE[-1][2])))
LR_DECAY_START_EP = int(os.environ.get("LR_DECAY_START_EP", str(LR_SCHEDULE[0][0])))
LR_DECAY_END_EP = int(os.environ.get("LR_DECAY_END_EP", str(LR_SCHEDULE[-1][1])))
NO_TRAINING_CAP = os.environ.get("NO_TRAINING_CAP", "false").lower() in ("1", "true", "yes")

# PPO stability knobs (optional, env-overridable)
PPO_KL_TARGET = float(os.environ.get("PPO_KL_TARGET", "0.015"))  # Standard PPO target
PPO_KL_STOP_MULT = float(os.environ.get("PPO_KL_STOP_MULT", "1.5"))  # Stop if KL > 1.5 * target
PPO_UPDATE_MIN_ADV_STD = float(os.environ.get("PPO_UPDATE_MIN_ADV_STD", "0.0"))  # gate updates if advantage std too small
PPO_VALUE_CLIP = float(os.environ.get("PPO_VALUE_CLIP", "10.0"))  # Enable with reasonable range
LOG_STD_LR_MULT = float(os.environ.get("LOG_STD_LR_MULT", "2.0"))  # lr multiplier for log_std param group
PPO_GRAD_CLIP = float(os.environ.get("PPO_GRAD_CLIP", "1.2"))  # grad clip max-norm
PPO_WEIGHT_DECAY = float(os.environ.get("PPO_WEIGHT_DECAY", "1e-3"))  # L2 regularization for optimizer



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


def lr_for_episode(episode_num):
    """Piecewise LR schedule driven by episode number."""
    if episode_num is None or episode_num <= 0:
        return LR_SCHED_START
    for start_ep, end_ep, lr_value in LR_SCHEDULE:
        if start_ep <= episode_num <= end_ep:
            return lr_value
    return LR_SCHEDULE[-1][2]


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
    low_zone_relay_flags: List = field(default_factory=list)  # Track diagnostic mask

    # 초기 Hidden States (시퀀스 시작 **직전**)
    init_actor_h: Optional = None  # torch.Tensor
    init_critic_h: Optional = None  # torch.Tensor

    # 메모리 관리
    last_update_time: float = field(default_factory=lambda: time.time())

    def add_step(self, local_obs, global_obs, action, logprob, value,
                 reward, done, actor_h_prev=None, critic_h_prev=None,
                 low_zone_relay=False):
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
        self.low_zone_relay_flags.append(bool(low_zone_relay))
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
            'low_zone_relay_flags': torch.tensor(self.low_zone_relay_flags, dtype=torch.bool),
            'init_actor_h': self.init_actor_h,
            'init_critic_h': self.init_critic_h,
            'bootstrap_value': torch.tensor(bootstrap, dtype=torch.float32)
        }


@dataclass
class RewardChannelEvent:
    """Tracks per-channel reward contributions for later reweighting."""
    host: Optional[str]
    key: Optional[str]
    value: float


class RewardMixerNetwork(nn.Module):
    """
    Lightweight network that keeps the sparse reward channels balanced by
    learning soft weights conditioned on context and running statistics.
    """

    def __init__(self, channels, hidden_dim=32, context_dim=REWARD_MIXER_CONTEXT_DIM, device=None):
        super().__init__()
        self.channels = list(channels)
        self.num_channels = len(self.channels)
        self.context_dim = context_dim
        self.device = torch.device(device or "cpu")
        input_dim = self.num_channels * 3 + self.context_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_channels),
        ).to(self.device)
        beta = float(os.environ.get("REWARD_MIXER_ADAM_BETA", "0.9"))
        self.optimizer = optim.Adam(self.parameters(), lr=REWARD_MIXER_LR, betas=(beta, beta))
        self.register_buffer("running_abs", torch.zeros(self.num_channels, device=self.device))

    def _build_target(self, abs_totals):
        recent_pref = torch.relu(abs_totals.max() - abs_totals)
        hist_pref = torch.relu(self.running_abs.max() - self.running_abs)
        pref = recent_pref + hist_pref
        pref = pref + (abs_totals <= 1e-6).float() * REWARD_MIXER_ZERO_BOOST
        if torch.all(pref <= 1e-9):
            pref = torch.ones_like(pref)
        temp = max(1e-3, REWARD_MIXER_TARGET_TEMP)
        return torch.softmax(pref / temp, dim=-1)

    def forward_step(self, totals, abs_totals, context_vec):
        totals = totals.to(self.device)
        abs_totals = abs_totals.to(self.device)
        context_vec = context_vec.to(self.device)
        total_mag = abs_totals.sum()
        if total_mag <= 1e-9:
            weights = torch.full((self.num_channels,), 1.0 / self.num_channels, device=self.device)
            target = weights.clone()
            return totals.detach(), weights.detach(), target.detach(), 0.0

        features = torch.cat([totals, abs_totals, self.running_abs, context_vec], dim=0)
        logits = self.net(features)
        weights = torch.softmax(logits, dim=-1)
        target = self._build_target(abs_totals)
        loss = F.mse_loss(weights, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), REWARD_MIXER_GRAD_CLIP)
        self.optimizer.step()
        with torch.no_grad():
            self.running_abs.mul_(REWARD_MIXER_EMA_BETA).add_(abs_totals * (1.0 - REWARD_MIXER_EMA_BETA))
            scaled = weights * total_mag
            adjusted = torch.sign(totals) * scaled
        return adjusted.detach(), weights.detach(), target.detach(), float(loss.detach())


class RewardMixerController:
    """Wrapper owning the reward mixer network plus diagnostics."""

    def __init__(self, channels):
        self.channels = list(channels)
        self.enabled = TORCH_OK and NON_FIX_REWARD_NETWORK and bool(self.channels)
        self.net = RewardMixerNetwork(
            self.channels,
            hidden_dim=REWARD_MIXER_HIDDEN,
            context_dim=REWARD_MIXER_CONTEXT_DIM,
            device="cpu",
        ) if self.enabled else None
        self.last_info = None

    def mix(self, totals, abs_totals, context_vec):
        if not self.enabled or self.net is None:
            return totals, {}
        totals_t = torch.tensor(totals, dtype=torch.float32, device=self.net.device)
        abs_t = torch.tensor(abs_totals, dtype=torch.float32, device=self.net.device)
        context_t = torch.tensor(context_vec, dtype=torch.float32, device=self.net.device)
        adjusted, weights, target, loss_val = self.net.forward_step(totals_t, abs_t, context_t)
        info = {
            "weights": weights.cpu().tolist(),
            "target": target.cpu().tolist(),
            "loss": loss_val,
        }
        self.last_info = info
        return adjusted.cpu().tolist(), info

    def state_dict(self):
        if not self.enabled or self.net is None:
            return None
        return {
            "channels": list(self.channels),
            "net": self.net.state_dict(),
            "optimizer": self.net.optimizer.state_dict(),
            "last_info": self.last_info,
        }

    def load_state_dict(self, state):
        if not self.enabled or self.net is None or not state:
            return False
        saved_channels = state.get("channels")
        if saved_channels and list(saved_channels) != list(self.channels):
            print(f"[REWARD_MIXER] Channel mismatch (saved={saved_channels}, current={self.channels}); loading anyway")
        net_state = state.get("net")
        if net_state:
            self.net.load_state_dict(net_state)
        opt_state = state.get("optimizer")
        if opt_state:
            try:
                self.net.optimizer.load_state_dict(opt_state)
            except Exception as exc:
                print(f"[REWARD_MIXER] Failed to load optimizer state: {exc}")
        self.last_info = state.get("last_info")
        return True

    def save(self, path):
        if not self.enabled or self.net is None:
            return {"ok": False, "error": "Reward mixer disabled"}
        state = self.state_dict()
        if state is None:
            return {"ok": False, "error": "Reward mixer has no state"}
        try:
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(state, path)
            return {"ok": True, "path": path}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def load(self, path):
        if not self.enabled or self.net is None:
            return {"ok": False, "error": "Reward mixer disabled"}
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": f"Missing mixer checkpoint: {path}"}
        try:
            state = torch.load(path, map_location=self.net.device)
            self.load_state_dict(state)
            return {"ok": True, "path": path}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


def build_reward_mixer_context(success_rate, step_success_rate, sr_delta, buf_rate_ema, dynamic_zone):
    """Generate a fixed-length context vector for the mixer."""
    ctx = [
        0.0 if success_rate is None or not math.isfinite(success_rate) else float(success_rate),
        0.0 if step_success_rate is None or not math.isfinite(step_success_rate) else float(step_success_rate),
        0.0 if sr_delta is None or not math.isfinite(sr_delta) else float(sr_delta),
        1.0 if dynamic_zone == "low" else 0.0,
        1.0 if dynamic_zone == "high" else 0.0,
        0.0 if buf_rate_ema is None or not math.isfinite(buf_rate_ema) else float(buf_rate_ema),
    ]
    if len(ctx) != REWARD_MIXER_CONTEXT_DIM:
        raise ValueError("Reward mixer context dimension mismatch")
    return ctx


def apply_reward_mixer_adjustments(controller, records, step_comp, rewards_per_host, rewards_per_key, context_vec):
    """Rebalance tracked reward channels and push adjustments into host/key reward maps."""
    if controller is None or not controller.enabled or not records:
        return
    totals = []
    abs_totals = []
    has_signal = False
    for ch in controller.channels:
        events = records.get(ch) or []
        total_val = sum(ev.value for ev in events)
        totals.append(total_val)
        abs_val = sum(abs(ev.value) for ev in events)
        abs_totals.append(abs_val)
        if abs_val > 1e-9:
            has_signal = True
    if not has_signal:
        return


    adjusted, info = controller.mix(totals, abs_totals, context_vec)
    for idx, ch in enumerate(controller.channels):
        events = records.get(ch) or []
        if not events:
            continue
        raw_total = totals[idx]
        adj_total = adjusted[idx] if adjusted else raw_total
        delta_total = adj_total - raw_total
        if abs(delta_total) < 1e-9:
            continue
        abs_total = abs_totals[idx]
        if abs_total <= 0.0:
            share = delta_total / len(events)
            for ev in events:
                if ev.host:
                    rewards_per_host[ev.host] = rewards_per_host.get(ev.host, 0.0) + share
                if ev.key:
                    rewards_per_key[ev.key] = rewards_per_key.get(ev.key, 0.0) + share
        else:
            for ev in events:
                weight = abs(ev.value) / abs_total if abs_total > 0 else 1.0 / len(events)
                delta = delta_total * weight
                if ev.host:
                    rewards_per_host[ev.host] = rewards_per_host.get(ev.host, 0.0) + delta
                if ev.key:
                    rewards_per_key[ev.key] = rewards_per_key.get(ev.key, 0.0) + delta
        step_comp[ch] = step_comp.get(ch, 0.0) + delta_total
    if info:
        info = dict(info)
        info["raw_totals"] = totals
        info["adjusted_totals"] = adjusted
        info["context"] = context_vec
        print(
            "[REWARD_MIXER] "
            f"ctx={context_vec} raw={totals} adj={adjusted} "
            f"weights={info.get('weights')} loss={info.get('loss'):.6f}"
        )


def shape_success_rate_delta(delta_value):
    """Exponential shaping so larger success-rate gaps grow super-linearly."""
    if SUCCESS_RATE_EXP_GAIN <= 0.0 or delta_value == 0.0:
        return delta_value
    mag = abs(delta_value)
    if not math.isfinite(mag):
        return delta_value
    mag = max(0.0, min(1.0, mag))
    shaped = math.expm1(mag * SUCCESS_RATE_EXP_GAIN)
    if SUCCESS_RATE_EXP_CLAMP > 0.0:
        shaped = min(shaped, SUCCESS_RATE_EXP_CLAMP)
    return math.copysign(shaped, delta_value)


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

    def __init__(self, obs_dim=7, global_obs_dim=20, hidden_size=128, num_layers=1):
        # Hyperparameters
        self.clip_eps = float(os.environ.get("PPO_CLIP", "0.20"))
        # Stability tuning (hardcoded defaults; still overridable via env)
        self.value_coef = float(os.environ.get("PPO_VALUE_COEF", "0.3"))
        self.entropy_coef = float(os.environ.get("PPO_ENTROPY", str(ENTROPY_COEF_START)))
        self.lr = float(os.environ.get("PPO_LR", str(LR_SCHED_START)))
        # Full-batch PPO over all finalized sequences (no mini-batching)
        self.epochs = int(os.environ.get("PPO_EPOCHS", "3"))
        self.gamma = float(os.environ.get("PPO_GAMMA", "0.99"))
        self.gae_lambda = float(os.environ.get("PPO_GAE_LAMBDA", "0.96"))
        # Always train on exactly 32 sequences per update (hardcoded)
        self.max_sequences_per_update = 32
        # Episode-level updates are optional (EPISODIC_UPDATES controls update timing)

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
        # Gaussian distribution: predict mean (log_std is shared parameter)
        self.actor_head = nn.Linear(hidden_size, 1).to(self.device)
        self.log_std = nn.Parameter(torch.tensor([LOG_STD_INIT], device=self.device))
        self.log_std_min = -3.0

        # ===== Critic Network (Centralized) =====
        # Input: global observations (20D = 7 local + 13 global)
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

        # Optimizer for all parameters (log_std has its own LR multiplier)
        base_params = (
            list(self.actor_embedding.parameters()) +
            list(self.actor_gru.parameters()) +
            list(self.actor_head.parameters()) +
            list(self.critic_embedding.parameters()) +
            list(self.critic_gru.parameters()) +
            list(self.critic_head.parameters())
        )
        adam_beta1 = float(os.environ.get("PPO_ADAM_BETA1", "0.9"))
        adam_beta2 = float(os.environ.get("PPO_ADAM_BETA2", str(adam_beta1)))
        self.opt = optim.Adam(
            [
                {"params": base_params, "weight_decay": PPO_WEIGHT_DECAY},
                {"params": [self.log_std], "lr": self.lr * LOG_STD_LR_MULT, "weight_decay": PPO_WEIGHT_DECAY},
            ],
            lr=self.lr,
            betas=(adam_beta1, adam_beta2),
        )
        self.optimizer = self.opt

        # Hidden state managers
        self.actor_hidden_manager = HiddenStateManager(num_layers, hidden_size, self.device)
        self.critic_hidden_manager = HiddenStateManager(num_layers, hidden_size * 2, self.device)

        # Sequence management (NEW: SequenceWindow based)
        self.sequence_windows = {}  # key -> SequenceWindow
        self.ready_sequences = []   # List[dict] - finalized sequences ready for training
        self.max_ready_sequences = int(os.environ.get("MAX_READY_SEQUENCES", "8192"))

        # Memory management settings
        self.max_window_age = float(os.environ.get("MAX_WINDOW_AGE", "300.0"))  # 5 minutes
        self.max_active_windows = int(os.environ.get("MAX_ACTIVE_WINDOWS", "1000"))
        # Require exactly 32 sequences before training (hardcoded batch size)
        self.min_sequences_to_train = 12
        self.episodic_updates = EPISODIC_UPDATES

        # Last actions cache (for reward assignment)
        # Elements include hidden states captured before the action
        self.last_actions = {}  # key -> (local_obs, global_obs, u_pre, logprob, value, actor_h_prev, critic_h_prev)

        self.policy_id = "rmappo_v1"
        print(f"[RecurrentPPOPolicy] Initialized R-MAPPO:")
        print(f"  - Actor GRU: {obs_dim}D -> {hidden_size} (L={num_layers})")
        print(f"  - Critic GRU: {global_obs_dim}D -> {hidden_size*2} (L={num_layers})")
        print(f"  - Truncated BPTT: {TRUNCATED_BPTT_LEN} steps")

    def update_lr(self, new_lr):
        """Update optimizer learning rate (keeps log_std group scaled)."""
        self.lr = float(new_lr)
        if hasattr(self, "opt") and self.opt is not None:
            if len(self.opt.param_groups) >= 1:
                self.opt.param_groups[0]["lr"] = self.lr
            if len(self.opt.param_groups) >= 2:
                self.opt.param_groups[1]["lr"] = self.lr * LOG_STD_LR_MULT
        if hasattr(self, "optimizer") and self.optimizer is not None and self.optimizer is not self.opt:
            if len(self.optimizer.param_groups) >= 1:
                self.optimizer.param_groups[0]["lr"] = self.lr
            if len(self.optimizer.param_groups) >= 2:
                self.optimizer.param_groups[1]["lr"] = self.lr * LOG_STD_LR_MULT

    def _to_tensor(self, arr):
        """Convert array to tensor."""
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _construct_global_obs(self, local_obs, global_features):
        """Construct global observation by concatenating local obs with global features."""
        return torch.cat([local_obs, global_features], dim=1)

    def _tanh_gaussian_sample(self, mean):
        """Sample action from tanh-squashed Gaussian distribution."""
        log_std_raw = torch.nan_to_num(self.log_std).to(dtype=torch.float64)
        log_std_clamped = torch.clamp(log_std_raw, min=float(self.log_std_min))
        if LOG_STD_MAX is not None and LOG_STD_MAX > 0:
            log_std_clamped = torch.clamp(log_std_clamped, max=float(LOG_STD_MAX))
        std = log_std_clamped.exp()
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        mean64 = mean.to(dtype=torch.float64)
        std = std.to(device=mean64.device).expand_as(mean64)
        std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-2))

        try:
            normal = torch.distributions.Normal(mean64, std)
            u = normal.rsample()
        except Exception as e:
            print(f"[ERROR] Normal distribution failed: {e}")
            u = torch.zeros_like(mean64)

        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        a = torch.tanh(u)
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))

        logp = normal.log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), min=1e-12))
        logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))

        return (
            u.to(mean.dtype),
            a.to(mean.dtype),
            logp.squeeze(-1).to(mean.dtype),
        )

    def _tanh_gaussian_logprob(self, mean, u):
        """Calculate log probability for tanh-squashed Gaussian."""
        log_std_raw = torch.nan_to_num(self.log_std).to(dtype=torch.float64)
        log_std_clamped = torch.clamp(log_std_raw, min=float(self.log_std_min))
        if LOG_STD_MAX is not None and LOG_STD_MAX > 0:
            log_std_clamped = torch.clamp(log_std_clamped, max=float(LOG_STD_MAX))
        std = log_std_clamped.exp()

        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        mean64 = mean.to(dtype=torch.float64)
        std = std.to(device=mean64.device).expand_as(mean64)
        std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-2))

        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        u64 = u.to(dtype=torch.float64)

        try:
            normal = torch.distributions.Normal(mean64, std)
            a = torch.tanh(u64)
            a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))
            logp = normal.log_prob(u64) - torch.log(torch.clamp(1 - a.pow(2), min=1e-12))
            logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
        except Exception as e:
            print(f"[ERROR] logprob calculation failed: {e}")
            logp = torch.zeros_like(u64)

        return logp.squeeze(-1).to(mean.dtype)

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

        # Policy head (batched) - Gaussian mean
        mean = self.actor_head(gru_out_actor.squeeze(1))  # (batch, 1)

        # Sample actions from Gaussian (batched)
        u_pre, actions, logprobs = self._tanh_gaussian_sample(mean)
        actions = actions.squeeze(-1)
        u_pre = u_pre.squeeze(-1)

        # Scale actions to [-delta_limit, delta_limit]
        scale = delta_limit if (delta_limit is not None and delta_limit > 0) else 1.0
        actions_scaled = actions * scale
        action_scaled = actions_scaled.detach()

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
               epochs=None, done_keys=None, low_zone_relays=None,
               train_now=True, allow_cleanup=True, allow_trim=True):
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
        low_zone_relays = set(low_zone_relays or [])

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
        selected_keys = None
        if self.episodic_updates and STEP_MAX_SEQUENCES > 0:
            all_keys = list(self.last_actions.keys())
            if len(all_keys) > STEP_MAX_SEQUENCES:
                random.shuffle(all_keys)
                selected_keys = set(all_keys[:STEP_MAX_SEQUENCES])

        for host, acted_keys in acted_by_host.items():
            if not acted_keys:
                continue

            r_host = float(rewards_per_host.get(host, 0.0))
            r_each = r_host / len(acted_keys) if len(acted_keys) > 0 else 0.0

            for k in acted_keys:
                if selected_keys is not None and k not in selected_keys:
                    continue
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
                    actor_h_prev, critic_h_prev,
                    low_zone_relay=(k in low_zone_relays)
                )
                any_added = True

                # Check if sequence is ready to finalize
                if window.is_ready():
                    # Compute bootstrap value for non-terminal sequences
                    # Use last in-window value prediction for consistency with TBPTT window
                    if not done:
                        try:
                            bootstrap_val = float(window.values[-1].detach().cpu().item())
                        except Exception:
                            bootstrap_val = 0.0
                    else:
                        bootstrap_val = 0.0

                    # Finalize and add to ready sequences
                    seq_data = window.finalize(bootstrap_val)
                    if seq_data is not None:
                        self.ready_sequences.append(seq_data)
                        if allow_trim:
                            self._trim_ready_sequences()
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
        if allow_cleanup:
            self._cleanup_old_windows()

        # Train if enough sequences ready (streaming updates always enabled)
        if train_now and len(self.ready_sequences) >= self.min_sequences_to_train:
            return self._ppo_update(epochs or self.epochs)

        return {"updated": False, "ready_sequences": len(self.ready_sequences)}

    def finalize_all_sequences(self, force_terminal=False, allow_trim=True):
        """Finalize all open sequence windows at episode end."""
        keys = list(self.sequence_windows.keys())
        for k in keys:
            window = self.sequence_windows.get(k)
            if window is None:
                continue
            if force_terminal and window.dones:
                window.dones[-1] = True
            try:
                last_v = float(window.values[-1].detach().cpu().item()) if window.values else 0.0
            except Exception:
                last_v = 0.0
            bootstrap_val = 0.0 if force_terminal else last_v
            seq_data = window.finalize(bootstrap_val)
            if seq_data is not None:
                self.ready_sequences.append(seq_data)
                if allow_trim:
                    self._trim_ready_sequences()
            del self.sequence_windows[k]
        if keys:
            self.actor_hidden_manager.reset(keys)
            self.critic_hidden_manager.reset(keys)

    def train_ready_sequences(self, force=False, epochs=None, consume_all=False):
        """Train on queued sequences (optionally ignore min_sequences_to_train)."""
        if not self.ready_sequences:
            return {"updated": False}
        if not force and len(self.ready_sequences) < self.min_sequences_to_train:
            return {"updated": False, "ready_sequences": len(self.ready_sequences)}
        if force:
            orig_max = self.max_sequences_per_update
            cap = max(0, int(EPISODIC_MAX_SEQUENCES or 0))
            if cap > 0:
                self.max_sequences_per_update = max(orig_max, min(len(self.ready_sequences), cap))
            else:
                self.max_sequences_per_update = orig_max
            try:
                if consume_all:
                    last_result = {"updated": False}
                    while self.ready_sequences:
                        last_result = self._ppo_update(epochs or self.epochs)
                    return last_result
                return self._ppo_update(epochs or self.epochs)
            finally:
                self.max_sequences_per_update = orig_max
        return self._ppo_update(epochs or self.epochs)

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
            # 강제 finalize: use last stored value as bootstrap if available for consistency
            try:
                last_v = float(self.sequence_windows[k].values[-1].detach().cpu().item()) if self.sequence_windows[k].values else 0.0
            except Exception:
                last_v = 0.0
            seq_data = self.sequence_windows[k].finalize(bootstrap_value=last_v)
            if seq_data is not None:
                self.ready_sequences.append(seq_data)
                self._trim_ready_sequences()

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
                try:
                    last_v = float(self.sequence_windows[k].values[-1].detach().cpu().item()) if self.sequence_windows[k].values else 0.0
                except Exception:
                    last_v = 0.0
                seq_data = self.sequence_windows[k].finalize(bootstrap_value=last_v)
                if seq_data is not None:
                    self.ready_sequences.append(seq_data)
                    self._trim_ready_sequences()

                del self.sequence_windows[k]
                self.actor_hidden_manager.reset([k])
                self.critic_hidden_manager.reset([k])

    def _trim_ready_sequences(self):
        """
        Prevent runaway growth of finalized sequence queue which can exhaust GPU memory.
        Drops the oldest sequences when the queue exceeds max_ready_sequences.
        """
        if self.max_ready_sequences <= 0:
            return
        overflow = len(self.ready_sequences) - self.max_ready_sequences
        if overflow > 0:
            del self.ready_sequences[:overflow]
            print(
                f"[PPO] Dropped {overflow} stale sequences to cap queue at "
                f"{self.max_ready_sequences} (queued={len(self.ready_sequences)})"
            )

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
            low_zone_mask: torch.Tensor [seq_len] bool mask for diagnostics
        """
        # as_tensor guards against stale numpy arrays or python lists sneaking in
        values = torch.as_tensor(seq_data['values'], dtype=torch.float32, device=self.device)
        rewards = torch.as_tensor(seq_data['rewards'], dtype=torch.float32, device=self.device)
        # CRITICAL FIX: convert boolean done flags to float so 1.0 - done works safely
        dones = torch.as_tensor(seq_data['dones'], dtype=torch.float32, device=self.device)
        low_zone_flags = torch.as_tensor(
            seq_data.get('low_zone_relay_flags', torch.zeros_like(rewards)),
            dtype=torch.bool,
            device=self.device
        )
        bootstrap_value = torch.as_tensor(seq_data['bootstrap_value'], dtype=torch.float32, device=self.device)

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

        return advantages, returns, low_zone_flags

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
        batch_start = time.time()
        print(f"\n[PPO] Starting training with {n_train} sequences (queued={total_available})")

        # 1. Compute GAE for each sequence
        seq_data_list = []
        total_steps = 0

        invalid_seqs = 0
        for seq in self.ready_sequences[:n_train]:
            local = seq['local_obs']
            global_obs = seq['global_obs']
            if local.shape[-1] != self.obs_dim or global_obs.shape[-1] != self.global_obs_dim:
                invalid_seqs += 1
                continue

            advantages, returns, low_zone_mask = self._compute_gae_for_sequence(seq)
            seq['advantages'] = advantages
            seq['returns'] = returns
            seq['low_zone_mask'] = low_zone_mask
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

        # Normalize advantages GLOBALLY across all sequences (pre-pass; used for gating signal)
        all_advantages = torch.cat([seq['advantages'] for seq in seq_data_list], dim=0)
        adv_count = all_advantages.numel()
        if adv_count == 0:
            del self.ready_sequences[:n_train]
            return {"updated": False, "reason": "adv_count_zero"}
        adv_mean = all_advantages.mean()
        adv_std = all_advantages.std(unbiased=False)
        if not torch.isfinite(adv_std):
            raise RuntimeError("Advantage std became non-finite before normalization")
        low_zone_mask_all = torch.cat([seq['low_zone_mask'].to(self.device) for seq in seq_data_list], dim=0)
        low_zone_adv_stats = {}
        if low_zone_mask_all.any():
            low_adv = all_advantages[low_zone_mask_all]
            if low_adv.numel() > 0:
                low_zone_adv_stats["low_zone_adv_mean"] = float(low_adv.mean().item())
                low_zone_adv_stats["low_zone_adv_median"] = float(torch.median(low_adv).item())
                try:
                    denom = low_zone_adv_stats["low_zone_adv_median"]
                    ratio = low_zone_adv_stats["low_zone_adv_mean"] / denom if denom not in (0.0, None) else float("inf")
                except Exception:
                    ratio = float("inf")
                low_zone_adv_stats["low_zone_adv_ratio"] = ratio
                print(
                    "[DIAG] low-zone relay advantages: "
                    f"count={low_adv.numel()} mean={low_zone_adv_stats['low_zone_adv_mean']:.5f} "
                    f"median={low_zone_adv_stats['low_zone_adv_median']:.5f} "
                    f"mean/median={ratio:.5f}"
                )

        for seq in seq_data_list:
            seq['advantages'] = (seq['advantages'] - adv_mean) / (adv_std + 1e-8)
            seq['low_zone_mask'] = seq['low_zone_mask'].to(self.device)

        # Optional gating: skip update if advantage std too small (little/no signal)
        if PPO_UPDATE_MIN_ADV_STD > 0.0:
            try:
                if float(adv_std.item()) < PPO_UPDATE_MIN_ADV_STD:
                    del self.ready_sequences[:n_train]
                    return {"updated": False, "reason": "low_adv_std", "adv_std": float(adv_std.item())}
            except Exception:
                pass

        # 2. Full-batch over all ready sequences (no mini-batching)
        num_seqs = len(seq_data_list)
        for epoch in range(epochs):
            # Gather all sequences
            batch_seqs = seq_data_list
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

            # Get predictions - Gaussian mean
            mean = self.actor_head(actor_out)  # (batch, max_len, 1)
            value_pred = self.critic_head(critic_out).squeeze(-1)  # (batch, max_len)

            # Compute log probs for Gaussian using latent samples
            new_logp = self._tanh_gaussian_logprob(mean, padded_actions)

            # Create mask for valid positions
            mask = torch.zeros(batch_size, max_len, device=self.device)
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1.0

            # Recompute advantages/returns using current value predictions for consistency
            for i, seq in enumerate(batch_seqs):
                L = seq_lengths[i]
                if L <= 0:
                    continue
                vals = value_pred[i, :L]
                rews = seq['rewards'].to(self.device)
                dns = seq['dones'].to(self.device)
                # Backward GAE
                gae = torch.zeros(1, device=self.device)
                adv_seq = torch.zeros(L, device=self.device)
                for t in range(L - 1, -1, -1):
                    next_nonterm = 1.0 - dns[t]
                    next_val = vals[t] if t == L - 1 else vals[t + 1]
                    delta = rews[t] + self.gamma * next_val * next_nonterm - vals[t]
                    gae = delta + self.gamma * self.gae_lambda * next_nonterm * gae
                    adv_seq[t] = gae
                ret_seq = adv_seq + vals
                padded_adv[i, :L] = adv_seq
                padded_ret[i, :L] = ret_seq

            # Global advantage normalization over valid positions
            flat_mask = mask.view(-1).bool()
            adv_valid_all = (padded_adv * mask).view(-1)[flat_mask]
            if adv_valid_all.numel() == 0:
                del self.ready_sequences[:n_train]
                return {"updated": False, "reason": "masked_adv_count_zero"}
            adv_mean = adv_valid_all.mean()
            adv_std = adv_valid_all.std(unbiased=False)
            if not torch.isfinite(adv_std):
                raise RuntimeError("Masked advantage std became non-finite during PPO update")
            padded_adv = (padded_adv - adv_mean) / (adv_std + 1e-8)

            # Apply mask and flatten
            new_logp_valid = (new_logp * mask).view(-1)[flat_mask]
            old_logp_valid = (padded_old_logp * mask).view(-1)[flat_mask]
            adv_valid = (padded_adv * mask).view(-1)[flat_mask]
            ret_valid = (padded_ret * mask).view(-1)[flat_mask]
            value_pred_valid = (value_pred * mask).view(-1)[flat_mask]

            # PPO loss
            ratio = (new_logp_valid - old_logp_valid).exp()
            surr1 = ratio * adv_valid
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_valid
            policy_loss = -torch.min(surr1, surr2).mean()

            # Optional value function clipping
            if PPO_VALUE_CLIP > 0.0:
                # Build padded old values from sequences
                padded_old_values = torch.zeros(batch_size, max_len, device=self.device)
                for i, seq in enumerate(batch_seqs):
                    L = seq_lengths[i]
                    if L > 0:
                        v = seq['values'].to(self.device)
                        padded_old_values[i, :L] = v
                old_value_valid = (padded_old_values * mask).view(-1)[mask.view(-1).bool()]
                v_clipped = old_value_valid + torch.clamp(value_pred_valid - old_value_valid, -PPO_VALUE_CLIP, PPO_VALUE_CLIP)
                value_loss_unclipped = (value_pred_valid - ret_valid).pow(2)
                value_loss_clipped = (v_clipped - ret_valid).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = (value_pred_valid - ret_valid).pow(2).mean()

            # Entropy from log_std parameter (tanh-Gaussian)
            log_std_clamped = torch.clamp(self.log_std, min=self.log_std_min)
            if LOG_STD_MAX is not None and LOG_STD_MAX > 0:
                log_std_clamped = torch.clamp(log_std_clamped, max=float(LOG_STD_MAX))
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
                PPO_GRAD_CLIP
            )
            self.opt.step()

            # Early stop on large KL (optional)
            if PPO_KL_TARGET > 0.0 and PPO_KL_STOP_MULT > 0.0:
                try:
                    approx_kl = (old_logp_valid - new_logp_valid).mean().item()
                except Exception:
                    approx_kl = 0.0
                if approx_kl > PPO_KL_TARGET * PPO_KL_STOP_MULT:
                    # print a light debug note and break epochs to avoid destructive update
                    print(f"[PPO] Early stop: approx_KL={approx_kl:.4f} target={PPO_KL_TARGET:.4f} mult={PPO_KL_STOP_MULT:.2f}")
                    break

        # Drop only the sequences we just trained on, keep the rest queued
        del self.ready_sequences[:n_train]
        batch_elapsed = time.time() - batch_start
        print(f"[PPO] Batch done: sequences={n_train} elapsed={batch_elapsed:.2f}s queued_remaining={len(self.ready_sequences)}")

        # Diagnostics
        try:
            ratio_mean = float(ratio.mean().item()) if ratio.numel() > 0 else 0.0
            ratio_std = float(ratio.std(unbiased=False).item()) if ratio.numel() > 1 else 0.0
            kl = float((old_logp_valid - new_logp_valid).mean().item()) if old_logp_valid.numel() > 0 else 0.0
        except Exception:
            ratio_mean = 0.0
            ratio_std = 0.0
            kl = 0.0
        try:
            std_mean = float(
                torch.clamp(torch.nan_to_num(self.log_std).to(dtype=torch.float64), min=float(self.log_std_min))
                .exp()
                .mean()
                .item()
            )
        except Exception:
            std_mean = 0.0
        try:
            adv_var = float(adv_valid.var(unbiased=False).item()) if adv_valid.numel() > 1 else 0.0
        except Exception:
            adv_var = 0.0

        result = {
            "updated": True,
            "trained_on": total_steps,
            "num_sequences": num_seqs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "ratio_mean": ratio_mean,
            "ratio_std": ratio_std,
            "kl": kl,
            "std_mean": std_mean,
            "adv_var": adv_var,
        }
        result.update(low_zone_adv_stats)
        return result

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
    def _safe_write(self, payload):
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        try:
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass

    # Initialize policy
    if TORCH_OK:
        AGENT = RecurrentPPOPolicy(obs_dim=7, global_obs_dim=21,
                                   hidden_size=LSTM_HIDDEN_SIZE,
                                   num_layers=LSTM_NUM_LAYERS)
        BASE_POLICY_ID = AGENT.policy_id
    else:
        AGENT = HeuristicPolicy()
        BASE_POLICY_ID = "heuristic_v1"

    POLICY_ID = BASE_POLICY_ID
    _BOOTSTRAP_MIXER_LOAD = None
    _BOOTSTRAP_MIXER_SAVE = None

    # Optional: load initial model if provided
    if MODEL_PATH:
        try:
            if TORCH_OK:
                res = AGENT.load(MODEL_PATH)
                if res.get("ok", False):
                    print(f"[BOOT] Loaded R-MAPPO model from {MODEL_PATH}")
                    _BOOTSTRAP_MIXER_LOAD = MODEL_PATH
                else:
                    print(f"[BOOT] Failed to load model: {res.get('error')}")
            else:
                print(f"[BOOT] MODEL_PATH set but torch unavailable")
        except Exception as e:
            print(f"[BOOT] Exception loading model: {e}")
        else:
            if TORCH_OK:
                try:
                    beta = float(os.environ.get("PPO_ADAM_BETA", "0.9"))
                    Handler.AGENT.optimizer = optim.Adam(
                        Handler.AGENT.parameters(),
                        lr=Handler.AGENT.lr,
                        eps=Handler.AGENT.adam_eps,
                        betas=(beta, beta)
                    )
                    print("[BOOT] Optimizer reset after loading model")
                except Exception as opt_exc:
                    print(f"[BOOT] Failed to reset optimizer: {opt_exc}")

    # Establish a baseline checkpoint to allow per-combo resets
    BASELINE_PATH = os.path.join(MODEL_DIR, "rmappo_baseline_init.pt") if TORCH_OK else None
    if TORCH_OK:
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            if not os.path.isfile(BASELINE_PATH):
                res = AGENT.save(BASELINE_PATH)
                if res.get("ok", False):
                    print(f"[BOOT] Saved baseline checkpoint at {BASELINE_PATH}")
                    _BOOTSTRAP_MIXER_SAVE = BASELINE_PATH
                else:
                    print(f"[BOOT] Failed to save baseline: {res.get('error')}")
            else:
                print(f"[BOOT] Baseline exists at {BASELINE_PATH}")
        except Exception as e:
            print(f"[BOOT] Failed to establish baseline: {e}")

    # Episode tracking
    EP_CUR = None
    EP_REWARD_ACC = 0.0
    STATE_CACHE = {}
    BUF_TREND = deque(maxlen=64)
    EP_COUNT = 0
    LAST_PARAM_LOG_EP = None
    LAST_OCC_AVG = None
    CORE_EP_TOTALS = defaultdict(lambda: {
        "success_penalty":0.0,
        "success_bonus":0.0,
        "skip_low_pred_pen":0.0,
        "peer_high_pred_pen":0.0,
        "drop_ttl_penalty":0.0,
        "sr_sparse_bonus":0.0,
        "sr_sparse_count":0.0,
        "relay_bonus":0.0,
        "state_bonus":0.0,
        "state_delta_bonus":0.0,
        "sr_bonus":0.0,
        "delivery_delay_pen":0.0,
        "overhead_penalty":0.0,
        "total_reward":0.0
    })
    EP_DELIVERED_SUM = 0.0
    EP_DELIVERED_TTL_SUM = 0.0
    EP_DROPPED_SUM = 0.0
    EP_CREATED_SUM = 0.0
    EP_PRESSURE_SUM = 0.0
    EP_RELAYED_SUM = 0.0
    TARGET_REWARD_APPLIED = False
    LAST_ACTIVE_HOSTS = set()
    LAST_SPARSE_EVENT_TIME = {}
    SPARSE_BONUS_EMA = None
    TRAINING_ENABLED = not EVAL_ONLY
    MODEL_SAVED_AT = None
    LAST_SUCCESS_RATE = None
    SR_SMOOTHED = None
    STEP_SR_SMOOTHED = None
    # Success-rate EMA normalization state
    SR_DELTA_EMA = None
    SR_EMA_COUNT = 0

    # Episode-level loss accumulators (for plotting)
    EP_LOSS_POLICY_SUM = 0.0
    EP_LOSS_VALUE_SUM = 0.0
    EP_LOSS_ENTROPY_SUM = 0.0
    EP_LOSS_UPDATE_COUNT = 0
    EP_HIGH_PENALTY_SUM = 0.0
    EP_LOW_PENALTY_SUM = 0.0
    LOW_ZONE_RELAY_KEYS = set()
    REWARD_MIXER = RewardMixerController(REWARD_MIXER_CHANNELS)
    if REWARD_MIXER and REWARD_MIXER.enabled:
        print(f"[BOOT] Reward mixer enabled for channels={REWARD_MIXER_CHANNELS}")
    else:
        print("[BOOT] Reward mixer disabled (set NON_FIX_REWARD_NETWORK=1 and ensure torch is available)")

    @classmethod
    def _reward_mixer_ckpt_path(cls, base_path):
        if not base_path:
            return None
        return f"{base_path}.mixer.pt"

    @classmethod
    def save_reward_mixer_checkpoint(cls, base_path):
        if not base_path or not cls.REWARD_MIXER or not cls.REWARD_MIXER.enabled:
            return
        mix_path = cls._reward_mixer_ckpt_path(base_path)
        if not mix_path:
            return
        try:
            res = cls.REWARD_MIXER.save(mix_path)
            if not res.get("ok", False):
                print(f"[REWARD_MIXER] Failed to save checkpoint {mix_path}: {res.get('error')}")
        except Exception as exc:
            print(f"[REWARD_MIXER] Exception during mixer save: {exc}")

    @classmethod
    def load_reward_mixer_checkpoint(cls, base_path):
        if not base_path or not cls.REWARD_MIXER or not cls.REWARD_MIXER.enabled:
            return
        mix_path = cls._reward_mixer_ckpt_path(base_path)
        if not mix_path or not os.path.isfile(mix_path):
            return
        res = cls.REWARD_MIXER.load(mix_path)
        if res.get("ok", False):
            print(f"[REWARD_MIXER] Loaded checkpoint from {mix_path}")
        else:
            print(f"[REWARD_MIXER] Failed to load mixer checkpoint {mix_path}: {res.get('error')}")

    # ============================================================================
    # RATE-BASED ZONE DETECTION
    # Track global buffer change rate for dynamic zone determination
    # ============================================================================
    PREV_GLOBAL_BUF = None
    BUF_RATE_EMA = 0.0

    # Store pending reward weights to apply at next episode boundary
    NEXT_REWARD_WEIGHTS = None

    def do_POST(self):
        global SUCCESS_RATE_BONUS_WEIGHT
        # Episode end handler
        if self.path == "/episode_end":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8")) if length > 0 else {}
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self._safe_write(("Bad Request: %s" % e).encode("utf-8"))
                return

            sim_id = str(req.get("sim_id", ""))
            buffer_size_mb = extract_buffer_size_mb(sim_id)
            avg = float(req.get("average_delivery_rate", "nan"))
            delivered = int(req.get("delivered", 0))
            created = int(req.get("created", 0))

            Handler.EP_COUNT += 1
            print(
                f"[DIAG] Episode {Handler.EP_COUNT} skip/peer reward totals: "
                f"high={Handler.EP_HIGH_PENALTY_SUM:.4f} low={Handler.EP_LOW_PENALTY_SUM:.4f}"
            )
            Handler.EP_HIGH_PENALTY_SUM = 0.0
            Handler.EP_LOW_PENALTY_SUM = 0.0

            # Episodic training: finalize all sequences and update once per episode
            episode_update_result = {}
            if TORCH_OK and Handler.TRAINING_ENABLED and EPISODIC_UPDATES:
                try:
                    update_start = time.time()
                    Handler.AGENT.finalize_all_sequences(force_terminal=True, allow_trim=False)
                    queued_sequences = len(Handler.AGENT.ready_sequences)
                    episode_update_result = Handler.AGENT.train_ready_sequences(force=True, consume_all=True)
                    update_elapsed = time.time() - update_start
                    print(f"[EPISODE_END] PPO update time={update_elapsed:.2f}s sequences={queued_sequences}")
                except Exception as exc:
                    print(f"[ERROR] Episodic PPO update failed at episode {Handler.EP_COUNT}: {exc}")
                    traceback.print_exc()
                    episode_update_result = {"updated": False}
                if episode_update_result.get("updated"):
                    try:
                        Handler.EP_LOSS_POLICY_SUM += float(episode_update_result.get('policy_loss', 0.0) or 0.0)
                        Handler.EP_LOSS_VALUE_SUM += float(episode_update_result.get('value_loss', 0.0) or 0.0)
                        Handler.EP_LOSS_ENTROPY_SUM += float(episode_update_result.get('entropy', 0.0) or 0.0)
                        Handler.EP_LOSS_UPDATE_COUNT += 1
                    except Exception:
                        pass
                    try:
                        sim_time_end = int(Handler.EP_COUNT * EPISODE_SECONDS)
                        log_update_metrics(sim_time_end, episode_update_result, buffer_size_mb, Handler.EP_COUNT)
                    except Exception:
                        pass

            # Reset rate-based zone detection state for new episode
            Handler.PREV_GLOBAL_BUF = None
            Handler.BUF_RATE_EMA = 0.0

            # Reset hidden states at episode boundaries
            if TORCH_OK:
                Handler.AGENT.reset_hidden_states(episode_start=True)
                meta_applied = False
                if META_AGENT_ENABLED and 'META_CONTROLLER' in globals():
                    controller = globals().get("META_CONTROLLER")
                    if controller is not None:
                        try:
                            delivery_for_meta = avg
                            if isinstance(delivery_for_meta, float) and math.isnan(delivery_for_meta):
                                delivery_for_meta = None
                            reward_snapshot = {}
                            try:
                                totals_ref = Handler.CORE_EP_TOTALS.get(Handler.EP_COUNT, {})
                                for key in META_AGENT_REWARD_KEYS:
                                    reward_snapshot[key] = float(totals_ref.get(key, 0.0))
                            except Exception:
                                reward_snapshot = {}
                            meta_result = controller.step(
                                episode=Handler.EP_COUNT,
                                delivery_rate=delivery_for_meta,
                                total_reward=(Handler.EP_REWARD_ACC or 0.0),
                                reward_components=reward_snapshot,
                                entropy_coef=getattr(Handler.AGENT, "entropy_coef", ENTROPY_COEF_START),
                                current_lr=getattr(Handler.AGENT, "lr", LR_SCHED_START),
                            )
                            if meta_result and meta_result.get("applied"):
                                new_entropy = float(meta_result.get("entropy_coef", Handler.AGENT.entropy_coef))
                                new_lr = float(meta_result.get("lr", Handler.AGENT.lr))
                                Handler.AGENT.entropy_coef = new_entropy
                                Handler.AGENT.update_lr(new_lr)
                                meta_applied = True
                                reason = meta_result.get("reason", "meta")
                                print(
                                    f"[META_AGENT] ep{Handler.EP_COUNT} applied lr={Handler.AGENT.lr:.6f} "
                                    f"entropy={Handler.AGENT.entropy_coef:.6f} ({reason})"
                                )
                        except Exception as meta_exc:
                            print(f"[META_AGENT] Failed to apply control: {meta_exc}")
                if not meta_applied:
                    # Exploration schedule: hold then decay entropy coefficient linearly
                    try:
                        if Handler.EP_COUNT <= ENTROPY_DECAY_START_EP:
                            new_entropy = ENTROPY_COEF_START
                        else:
                            span = max(1.0, float(ENTROPY_DECAY_END_EP - ENTROPY_DECAY_START_EP))
                            frac = min(1.0, max(0.0, (Handler.EP_COUNT - ENTROPY_DECAY_START_EP) / span))
                            new_entropy = ENTROPY_COEF_START + frac * (ENTROPY_COEF_END - ENTROPY_COEF_START)
                        Handler.AGENT.entropy_coef = float(new_entropy)
                        print(f"[EPISODE_END] Entropy coef updated to {Handler.AGENT.entropy_coef:.6f} (ep={Handler.EP_COUNT})")
                    except Exception:
                        pass
                    # LR schedule: piecewise by episode range
                    try:
                        new_lr = lr_for_episode(Handler.EP_COUNT)
                        Handler.AGENT.update_lr(new_lr)
                        print(f"[EPISODE_END] LR updated to {Handler.AGENT.lr:.6f} (ep={Handler.EP_COUNT})")
                    except Exception:
                        pass

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
                    try:
                        Handler.save_reward_mixer_checkpoint(path)
                    except Exception as mix_exc:
                        print(f"[REWARD_MIXER] Failed to save mixer at episode end: {mix_exc}")

                    if Handler.EP_COUNT >= TOTAL_EPISODES and not NO_TRAINING_CAP:
                        Handler.TRAINING_ENABLED = False
                        print(f"[TRAINING_COMPLETE] Finished {TOTAL_EPISODES} episodes (set NO_TRAINING_CAP=1 to keep training)")
                except Exception as e:
                    print(f"[ERROR] Failed to save model: {e}")

            # Emit episode total reward CSV row
            try:
                buffer_size = extract_buffer_size_mb(sim_id)
                avg_overhead = get_final_overhead_from_reports(sim_id)
                avg_delay_metric = get_final_latency_from_reports(sim_id)
                if avg_overhead is None and Handler.EP_DELIVERED_SUM > 0:
                    avg_overhead = (
                        (Handler.EP_RELAYED_SUM - Handler.EP_DELIVERED_SUM) / Handler.EP_DELIVERED_SUM
                    )
                log_episode_reward(
                    episode_num=Handler.EP_COUNT,
                    sim_id=sim_id,
                    total_reward=round(float(Handler.EP_REWARD_ACC or 0.0), 6),
                    avg_delivery_rate=(None if (avg is None or (isinstance(avg, float) and math.isnan(avg))) else avg),
                    delivered=delivered,
                    created=created,
                    avg_overhead=avg_overhead,
                    avg_delay=avg_delay_metric,
                    buffer_size_mb=buffer_size,
                )
                log_episode_core_totals(Handler.EP_COUNT, buffer_size)
                # Compute and log episode loss averages
                try:
                    upd = max(1, int(Handler.EP_LOSS_UPDATE_COUNT or 0))
                    actor_loss_avg = (Handler.EP_LOSS_POLICY_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                    critic_loss_avg = (Handler.EP_LOSS_VALUE_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                    entropy_avg = (Handler.EP_LOSS_ENTROPY_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                except Exception:
                    actor_loss_avg = None
                    critic_loss_avg = None
                    entropy_avg = None
                log_episode_losses(
                    episode_num=Handler.EP_COUNT,
                    sim_id=sim_id,
                    actor_loss_avg=actor_loss_avg,
                    critic_loss_avg=critic_loss_avg,
                    updates=Handler.EP_LOSS_UPDATE_COUNT or 0,
                    entropy_avg=entropy_avg,
                    buffer_size_mb=buffer_size,
                )
                # Snapshot for response payload before resets
                _actor_loss_avg_snapshot = actor_loss_avg
                _critic_loss_avg_snapshot = critic_loss_avg
                _entropy_avg_snapshot = entropy_avg
                _loss_updates_snapshot = int(Handler.EP_LOSS_UPDATE_COUNT or 0)
            except Exception as e:
                print(f"[WARN] Failed to log episode reward: {e}")

            # If there is a pending reward weight update scheduled for next episode, apply it now
            try:
                if Handler.NEXT_REWARD_WEIGHTS:
                    cfg = Handler.NEXT_REWARD_WEIGHTS or {}
                    # Apply known weights if present
                    globals_ref = globals()
                    for key, val in cfg.items():
                        if key in globals_ref and isinstance(val, (int, float)):
                            globals_ref[key] = float(val)
                    print(f"[EPISODE_END] Applied pending reward weights for episode {Handler.EP_COUNT}: {cfg}")
                    Handler.NEXT_REWARD_WEIGHTS = None
            except Exception as e:
                print(f"[WARN] Failed to apply next-episode reward weights: {e}")

            # Snapshot total reward before reset for response payload
            try:
                final_total_reward = round(float(Handler.EP_REWARD_ACC or 0.0), 6)
            except Exception:
                final_total_reward = 0.0

            # Reset episode stats
            Handler.EP_DELIVERED_SUM = 0.0
            Handler.EP_DELIVERED_TTL_SUM = 0.0
            Handler.EP_DROPPED_SUM = 0.0
            Handler.EP_CREATED_SUM = 0.0
            Handler.EP_PRESSURE_SUM = 0.0
            Handler.EP_RELAYED_SUM = 0.0
            Handler.TARGET_REWARD_APPLIED = False
            Handler.LAST_ACTIVE_HOSTS = set()
            Handler.LAST_SPARSE_EVENT_TIME = {}
            Handler.SPARSE_BONUS_EMA = None
            Handler.LAST_SUCCESS_RATE = None
            Handler.SR_SMOOTHED = None
            Handler.STEP_SR_SMOOTHED = None
            Handler.SR_DELTA_EMA = None
            Handler.SR_EMA_COUNT = 0
            Handler.EP_REWARD_ACC = 0.0
            Handler.EP_LOSS_POLICY_SUM = 0.0
            Handler.EP_LOSS_VALUE_SUM = 0.0
            Handler.EP_LOSS_ENTROPY_SUM = 0.0
            Handler.EP_LOSS_UPDATE_COUNT = 0

            # Include total episode reward and optionally avg delivery rate in the response for convenience
            resp_payload = {
                "ok": True,
                "episode": Handler.EP_COUNT,
                "total_reward": final_total_reward,
            }
            try:
                # Include average losses for plotting convenience
                if '_actor_loss_avg_snapshot' in locals() and _actor_loss_avg_snapshot is not None:
                    resp_payload['avg_actor_loss'] = float(_actor_loss_avg_snapshot)
                if '_critic_loss_avg_snapshot' in locals() and _critic_loss_avg_snapshot is not None:
                    resp_payload['avg_critic_loss'] = float(_critic_loss_avg_snapshot)
                if '_entropy_avg_snapshot' in locals() and _entropy_avg_snapshot is not None:
                    resp_payload['avg_entropy'] = float(_entropy_avg_snapshot)
                if '_loss_updates_snapshot' in locals():
                    resp_payload['loss_updates'] = int(_loss_updates_snapshot)
            except Exception:
                pass
            if avg is not None and (not isinstance(avg, float) or not math.isnan(avg)):
                try:
                    resp_payload["avg_delivery_rate"] = float(avg)
                except Exception:
                    pass
            body = json.dumps(resp_payload).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self._safe_write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # Set reward weights handler (runtime control)
        if self.path == "/set_reward_weights":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8")) if length > 0 else {}
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self._safe_write(("Bad Request: %s" % e).encode("utf-8"))
                return

            # apply_mode: now | next_episode
            apply_mode = str(req.get("apply", "now")).lower()
            weights = req.get("weights", req)  # allow flat JSON as well

            # Accepted keys (subset of reward vars)
            allowed = {
                "PBASE_REWARD_WEIGHT",
                "MAX_PBASE_REWARD_WEIGHT",
                "PBASE_DIFF_EPS",
                "BUFFER_DIFF_REWARD_WEIGHT",
                "SUCCESS_RATE_WEIGHT",
                "SUCCESS_RATE_BONUS_WEIGHT",
                "SUCCESS_RATE_EPS",
                "DELIVERY_REWARD_WEIGHT",
                "DELIVERY_REWARD_UTIL_MAX",
                "HIGH_LOAD_PENALTY_WEIGHT",
                "MID_UTIL_PENALTY_WEIGHT",
                "MID_UTIL_THRESHOLD",
                "RELAY_BONUS_WEIGHT",
            }

            if apply_mode == "next" or apply_mode == "next_episode":
                # Stash for application at next /episode_end
                cfg = {}
                for k, v in (weights or {}).items():
                    if k in allowed:
                        try:
                            cfg[k] = float(v)
                        except Exception:
                            pass
                Handler.NEXT_REWARD_WEIGHTS = cfg
                print(f"[CONTROL] Queued next-episode reward weights: {cfg}")
                body = json.dumps({"ok": True, "applied": False, "queued": True, "weights": cfg}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self._safe_write(body)
                return

            # Apply immediately (unsafe if done mid-episode, but useful for experimentation)
            applied = {}
            try:
                globals_ref = globals()
                for k, v in (weights or {}).items():
                    if k in allowed:
                        try:
                            globals_ref[k] = float(v)
                            applied[k] = float(v)
                        except Exception:
                            continue
                print(f"[CONTROL] Applied reward weights NOW: {applied}")
            except Exception as e:
                print(f"[ERROR] Failed to apply reward weights: {e}")

            body = json.dumps({"ok": True, "applied": True, "weights": applied}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
            return

        # Save current model checkpoint
        if self.path == "/save_model":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8")) if length > 0 else {}
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self._safe_write(("Bad Request: %s" % e).encode("utf-8"))
                return
            if not TORCH_OK:
                self.send_response(400)
                self.end_headers()
                self._safe_write(b"PyTorch disabled")
                return
            out_path = req.get("path") or os.path.join(MODEL_DIR, f"rmappo_manual_{int(time.time())}.pt")
            res = Handler.AGENT.save(out_path)
            if res.get("ok", False):
                Handler.save_reward_mixer_checkpoint(out_path)
            body = json.dumps({"ok": res.get("ok", False), "path": out_path, "error": res.get("error")}).encode("utf-8")
            self.send_response(200 if res.get("ok", False) else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
            return

        # Load model checkpoint from path
        if self.path == "/load_model":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8")) if length > 0 else {}
            except Exception as e:
                self.send_response(400)
                self.end_headers()
                self._safe_write(("Bad Request: %s" % e).encode("utf-8"))
                return
            if not TORCH_OK:
                self.send_response(400)
                self.end_headers()
                self._safe_write(b"PyTorch disabled")
                return
            in_path = req.get("path")
            if not in_path:
                self.send_response(400)
                self.end_headers()
                self._safe_write(b"Missing 'path'")
                return
            res = Handler.AGENT.load(in_path)
            # Reset optimizer after load
            try:
                beta = float(os.environ.get("PPO_ADAM_BETA", "0.9"))
                Handler.AGENT.optimizer = optim.Adam(
                    Handler.AGENT.parameters(),
                    lr=Handler.AGENT.lr,
                    eps=Handler.AGENT.adam_eps,
                    betas=(beta, beta)
                )
            except Exception:
                pass
            if res.get("ok", False):
                Handler.load_reward_mixer_checkpoint(in_path)
            ok = res.get("ok", False)
            body = json.dumps({"ok": ok, "meta": res.get("meta"), "error": res.get("error")}).encode("utf-8")
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
            return

        # Restore baseline checkpoint (per-combo reset)
        if self.path == "/restore_baseline":
            if not TORCH_OK or not Handler.BASELINE_PATH or not os.path.isfile(Handler.BASELINE_PATH):
                self.send_response(500)
                self.end_headers()
                self._safe_write(b"Baseline not available")
                return
            res = Handler.AGENT.load(Handler.BASELINE_PATH)
            try:
                beta = float(os.environ.get("PPO_ADAM_BETA", "0.9"))
                Handler.AGENT.optimizer = optim.Adam(
                    Handler.AGENT.parameters(),
                    lr=Handler.AGENT.lr,
                    eps=Handler.AGENT.adam_eps,
                    betas=(beta, beta)
                )
                Handler.AGENT.reset_hidden_states(episode_start=True)
            except Exception:
                pass
            if res.get("ok", False):
                Handler.load_reward_mixer_checkpoint(Handler.BASELINE_PATH)
            ok = res.get("ok", False)
            body = json.dumps({"ok": ok, "baseline": Handler.BASELINE_PATH, "error": res.get("error")}).encode("utf-8")
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
            return
            return

        # Main inference endpoint - reuse logic from original server
        # (The reward calculation and inference logic remains mostly the same)
        # For brevity, I'll include a simplified version here
        # In production, copy the full logic from original drl_server.py

        if self.path != "/infer_and_update":
            self.send_response(404)
            self.end_headers()
            self._safe_write(b"Not Found")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
        except Exception as e:
            self.send_response(400)
            self.end_headers()
            self._safe_write(("Bad Request: %s" % e).encode("utf-8"))
            return

        # Extract request data
        delta_limit = float(req.get("delta_limit", 10.00))
        prev_tr = req.get("prev_transition", []) or []
        state_batch = req.get("state_batch", []) or []
        sim_time = int(float(req.get("time", 0)))
        sim_id = str(req.get("sim_id", ""))
        buffer_size_mb = extract_buffer_size_mb(sim_id)

        # Helper function
        def _to_float(val, default=0.0):
            try:
                f = float(val)
            except Exception:
                return default
            if math.isnan(f) or math.isinf(f):
                return default
            return f

        def _get_buf_mean(cached_entry, mode):
            if not cached_entry:
                return None
            if mode == "local":
                return cached_entry.get("buf_mean_local", cached_entry.get("buf_mean"))
            return cached_entry.get("buf_mean_global", cached_entry.get("buf_mean"))

        # Build rewards with delivery + Prophet alignment + overhead penalty
        rewards_per_host = {}
        rewards_per_key = {}
        done_keys = set()
        stress_keys = set()
        # Accumulators for per-step component logging (host-level contributions)
        step_comp = {
            'pbase': 0.0,          # neighbor-based
            'pbase_max': 0.0,      # max-based
            'pbase_mid_relay': 0.0,
            'pbase_mid_abort': 0.0,
            'buffer_diff': 0.0,
            'skip_low_pred_pen': 0.0,
            'peer_high_pred_pen': 0.0,
            'success_delta': 0.0,
            'success_bonus': 0.0,
            'success_penalty': 0.0,
            'delivery': 0.0,
            'heuristic': 0.0,
            'high_load_penalty': 0.0,
            'mid_util_penalty': 0.0,
            'trend_penalty': 0.0,
            'relay_bonus': 0.0,
            'state_bonus': 0.0,
            'state_delta_bonus': 0.0,
            'sr_bonus': 0.0,
            'sparse_delivery': 0.0,
            'sr_sparse_bonus': 0.0,
            'sr_sparse_count': 0.0,
            'target_bonus': 0.0,
            'pbase_buf_and': 0.0,      # hard AND positive
            'pbase_buf_and_pen': 0.0,  # hard AND negative
            'drop_ttl_penalty': 0.0,
            'delivery_delay_pen': 0.0,
            'overhead_penalty': 0.0,
        }
        reward_mixer_records = None
        if getattr(Handler, "REWARD_MIXER", None) and Handler.REWARD_MIXER.enabled:
            reward_mixer_records = {ch: [] for ch in Handler.REWARD_MIXER.channels}
        sparse_delivery_events = []

        # ============================================================================
        # MESSAGE_COUNT_NORMALIZATION
        # Divide penalties by total message count for fair comparison
        # ============================================================================
        total_messages = float(len(state_batch)) if len(state_batch) > 0 else 1.0

        # ============================================================================
        # RATE-BASED ZONE DETECTION
        # Calculate global buffer change rate for dynamic zone determination
        # Uses derivative (d_buf/dt) instead of absolute buffer value
        # Parameters: BUF_RATE_EMA_ALPHA, RATE_HIGH_THRESHOLD, RATE_LOW_THRESHOLD
        # ============================================================================

        # Step 1: Calculate current global buffer average
        current_global_buf = 0.0
        buf_count = 0
        for item in state_batch:
            buf_val = _to_float(item.get("global_avg_buf", 0.0), 0.0)
            if math.isfinite(buf_val):
                current_global_buf += buf_val
                buf_count += 1
        if buf_count > 0:
            current_global_buf /= buf_count

        # Step 2: Calculate buffer change rate
        global_buf_for_sparse = current_global_buf if math.isfinite(current_global_buf) else 0.0
        if Handler.PREV_GLOBAL_BUF is not None:
            buf_rate = current_global_buf - Handler.PREV_GLOBAL_BUF
            # EMA smoothing to reduce noise
            Handler.BUF_RATE_EMA = (BUF_RATE_EMA_ALPHA * buf_rate +
                                   (1 - BUF_RATE_EMA_ALPHA) * Handler.BUF_RATE_EMA)
        else:
            Handler.BUF_RATE_EMA = 0.0

        # Step 3: Determine dynamic zone based on global buffer utilization
        if math.isfinite(current_global_buf):
            if current_global_buf >= BUFFER_DIFF_HIGH_THRESHOLD:
                dynamic_zone = 'high'
            elif current_global_buf <= BUFFER_DIFF_LOW_THRESHOLD:
                dynamic_zone = 'low'
            else:
                dynamic_zone = 'mid'
        else:
            # Fallback to rate-based zone if average is unavailable
            if Handler.BUF_RATE_EMA > RATE_HIGH_THRESHOLD:
                dynamic_zone = 'high'
            elif Handler.BUF_RATE_EMA < RATE_LOW_THRESHOLD:
                dynamic_zone = 'low'
            else:
                dynamic_zone = 'mid'

        # Debug log
        if buf_count > 0:
            main_print(f"[RATE_ZONE] buf={current_global_buf:.3f}, rate={Handler.BUF_RATE_EMA:.6f}, zone={dynamic_zone}")

        # Step 4: Update state for next step
        Handler.PREV_GLOBAL_BUF = current_global_buf
        buf_slope = getattr(Handler, "BUF_RATE_EMA", 0.0)
        if not math.isfinite(buf_slope):
            buf_slope = 0.0
        slope_penalty_scale = abs(buf_slope) * BUF_SLOPE_PENALTY_GAIN  # amplify derivative for reward magnitude

        # Step-level success rate (delivered/created) smoothing
        total_delivered = 0.0
        for item in prev_tr:
            try:
                total_delivered += _to_float(item.get("delivered", 0.0), 0.0)
            except Exception:
                continue
        created_since_last = _to_float(req.get("created_since_last", 0.0), 0.0)
        step_sr = None
        if created_since_last > 0.0:
            step_sr = max(0.0, min(1.0, total_delivered / created_since_last))
        if step_sr is not None and math.isfinite(step_sr):
            if Handler.STEP_SR_SMOOTHED is None or not math.isfinite(Handler.STEP_SR_SMOOTHED):
                Handler.STEP_SR_SMOOTHED = step_sr
            else:
                Handler.STEP_SR_SMOOTHED = (
                    STEP_SUCCESS_RATE_BETA * Handler.STEP_SR_SMOOTHED
                    + (1.0 - STEP_SUCCESS_RATE_BETA) * step_sr
                )
        created_increment = created_since_last if (created_since_last > 0.0 and math.isfinite(created_since_last)) else 0.0
        step_ttl_delivered = 0.0
        relay_sr_scale = None
        ttled_sr_scale = None
        denom_created = Handler.EP_CREATED_SUM + created_increment
        if denom_created > 0.0:
            cumulative_delivered = Handler.EP_DELIVERED_SUM + total_delivered
            cumulative_ttled = Handler.EP_DELIVERED_TTL_SUM + step_ttl_delivered
            try:
                relay_sr_scale = max(0.0, min(1.0, cumulative_delivered / denom_created))
                ttled_sr_scale = max(0.0, min(1.0, cumulative_ttled / denom_created))
            except Exception:
                relay_sr_scale = None
                ttled_sr_scale = None

        state_weight = 0.0
        if STATE_BONUS_WEIGHT > 0 and math.isfinite(current_global_buf):
            occ_norm = max(0.0, min(1.0, current_global_buf))
            try:
                state_weight = math.exp(-((occ_norm - SR_SPARSE_OCC_CENTER) ** 2) / (2.0 * (STATE_BONUS_SIGMA ** 2)))
            except Exception:
                state_weight = 0.0
        relay_occ_near = 0.0
        relay_occ_far = 0.0
        relay_occ_base_scale = 0.0
        relay_occ_delta = 0.0
        relay_occ_step_total = 0.0
        if (
            RELAY_BONUS_WEIGHT != 0.0
            and math.isfinite(current_global_buf)
            and relay_sr_scale is not None
            and relay_sr_scale > 0.0
        ):
            try:
                relay_occ_near = math.exp(
                    -((current_global_buf - SR_SPARSE_OCC_CENTER) ** 2)
                    / (2.0 * (RELAY_OCC_SIGMA ** 2))
                )
            except Exception:
                relay_occ_near = 0.0
            relay_occ_near = max(0.0, min(1.0, relay_occ_near))
            relay_occ_far = max(0.0, min(1.0, 1.0 - relay_occ_near))
            if relay_occ_near > 0.0 or relay_occ_far > 0.0:
                relay_occ_delta = current_global_buf - SR_SPARSE_OCC_CENTER
                relay_occ_base_scale = (
                    RELAY_BONUS_WEIGHT * relay_sr_scale
                ) / max(1.0, total_messages)
            relay_occ_step_total = 0.0
        step_relay_total = 0.0
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
                skip_low_pred = _to_float(item.get("skip_low_pred", 0.0), 0.0)
                peer_high_pred = _to_float(item.get("peer_high_pred", 0.0), 0.0)
                avg_delay = _to_float(item.get("avg_delay", 0.0), 0.0)

                reward = 0.0
                my_buf_norm = _to_float(item.get("my_buffer_norm", 0.0), 0.0)

                # 1) Delivery reward (only when local buffer has headroom)
                if relayed > 0:
                    step_relay_total += relayed
                ttl_ratio = _to_float(item.get("avg_ttl_ratio", 1.0), 1.0)
                if not math.isfinite(ttl_ratio):
                    ttl_ratio = 1.0
                ttl_ratio = max(0.0, min(1.0, ttl_ratio))
                ttl_weight = 1.0
                if RELAY_TTL_DECAY_K > 0.0:
                    ttl_weight = math.exp(-RELAY_TTL_DECAY_K * max(0.0, 1.0 - ttl_ratio))

                if (
                    delivered > 0
                    and DELIVERY_REWARD_WEIGHT != 0.0
                    and my_buf_norm <= DELIVERY_REWARD_UTIL_MAX
                ):
                    comp = DELIVERY_REWARD_WEIGHT * delivered * ttl_weight
                    reward += comp
                    step_comp['delivery'] += comp
                if (
                    DELIVERY_DELAY_PENALTY_WEIGHT != 0.0
                    and delivered > 0
                    and math.isfinite(avg_delay)
                    and avg_delay > 0.0
                ):
                    delay_factor = 1.0 - math.exp(-max(0.0, avg_delay) / DELIVERY_DELAY_DECAY)
                    if delay_factor > 0.0:
                        event_scale = delivered / max(1.0, total_messages)
                        comp_delay = -abs(DELIVERY_DELAY_PENALTY_WEIGHT) * delay_factor * event_scale * 100.0
                        reward += comp_delay
                        step_comp['delivery_delay_pen'] += comp_delay
                if (
                    OVERHEAD_PENALTY_WEIGHT != 0.0
                    and relayed > 0
                ):
                    extra_relays = max(0.0, relayed - delivered)
                    if extra_relays > 0.0:
                        overhead_factor = 1.0 - math.exp(-extra_relays / OVERHEAD_DECAY)
                        if overhead_factor > 0.0:
                            event_scale = extra_relays / max(1.0, total_messages)
                            comp_over = -abs(OVERHEAD_PENALTY_WEIGHT) * overhead_factor * event_scale * 10.0
                            reward += comp_over
                            step_comp['overhead_penalty'] += comp_over
                if state_weight > 0.0:
                    comp_state = 0.0
                    occ_high = (
                        current_global_buf is not None
                        and math.isfinite(current_global_buf)
                        and current_global_buf >= SR_SPARSE_OCC_CENTER
                    )
                    if relayed > 0 and not occ_high:
                        comp_state += STATE_BONUS_WEIGHT * state_weight * relayed
                    if skip_low_pred > 0 and occ_high:
                        comp_state += STATE_BONUS_WEIGHT * state_weight * skip_low_pred
                    if comp_state != 0.0:
                        comp_state /= max(1.0, total_messages)
                        reward += comp_state
                        step_comp['state_bonus'] += comp_state
                if relay_occ_base_scale > 0.0 and (relay_occ_near > 0.0 or relay_occ_far > 0.0):
                    reward_scale = relay_occ_base_scale * relay_occ_near
                    penalty_scale = relay_occ_base_scale * relay_occ_far
                    if relay_occ_delta <= 0.0:
                        if relayed > 0 and RELAY_LOW_OCC_GAIN > 0.0 and reward_scale > 0.0:
                            comp = reward_scale * RELAY_LOW_OCC_GAIN * relayed
                            reward += comp
                            step_comp['relay_bonus'] += comp
                            relay_occ_step_total += comp
                        if skip_low_pred > 0 and RELAY_LOW_OCC_GAIN > 0.0 and penalty_scale > 0.0:
                            comp = -penalty_scale * RELAY_LOW_OCC_GAIN * skip_low_pred
                            reward += comp
                            step_comp['relay_bonus'] += comp
                            relay_occ_step_total += comp
                    else:
                        if relayed > 0 and RELAY_HIGH_OCC_GAIN > 0.0 and penalty_scale > 0.0:
                            comp = -penalty_scale * RELAY_HIGH_OCC_GAIN * relayed
                            reward += comp
                            step_comp['relay_bonus'] += comp
                            relay_occ_step_total += comp
                        if skip_low_pred > 0 and RELAY_HIGH_OCC_GAIN > 0.0 and reward_scale > 0.0:
                            comp = reward_scale * RELAY_HIGH_OCC_GAIN * skip_low_pred
                            reward += comp
                            step_comp['relay_bonus'] += comp
                            relay_occ_step_total += comp
                if (
                    delivered > 0
                    and SPARSE_SUCCESS_REWARD
                    and SR_SPARSE_WEIGHT != 0.0
                ):
                    ttl_ratio = _to_float(item.get("avg_ttl_ratio", 1.0), 1.0)
                    hop_avg = _to_float(item.get("avg_hops", 0.0), 0.0)
                    if relayed > 0:
                        step_relay_total += relayed
                    sparse_delivery_events.append({
                        "host": host,
                        "dest": dest,
                        "key": key,
                        "delivered": delivered,
                        "ttl_ratio": ttl_ratio,
                        "avg_hops": hop_avg,
                        "avg_buf": global_buf_for_sparse,
                    })

                # 2) Prophet base comparison reward (neighbor-based and optional max-based)
                my_p_base = _to_float(item.get("p_base", 0.0), 0.0)
                neighbor_p_base = _to_float(item.get("neighbor_p_base", 0.0), 0.0)
                max_neighbor_p_base = _to_float(item.get("max_neighbor_p_base", 0.0), 0.0)
                # 2A) Neighbor-based (actual peer) comparison
                if (
                    PBASE_REWARD_WEIGHT != 0.0
                    and math.isfinite(my_p_base)
                    and math.isfinite(neighbor_p_base)
                    and relayed > 0
                ):
                    diff = neighbor_p_base - my_p_base
                    if abs(diff) >= PBASE_DIFF_EPS:
                        comp = PBASE_REWARD_WEIGHT * diff  # magnitude-proportional reward/penalty
                        reward += comp
                        step_comp['pbase'] += comp

                # 2B) Max-based (diagnostic) comparison
                if (
                    MAX_PBASE_REWARD_WEIGHT != 0.0
                    and math.isfinite(my_p_base)
                    and math.isfinite(max_neighbor_p_base)
                    and relayed > 0
                ):
                    diffm = max_neighbor_p_base - my_p_base
                    if abs(diffm) >= PBASE_DIFF_EPS:
                        compm = MAX_PBASE_REWARD_WEIGHT * diffm  # magnitude-proportional
                        reward += compm
                        step_comp['pbase_max'] += compm

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
                    comp = HEURISTIC_REWARD_WEIGHT * relayed * (0.5 * p_scale + 0.5 * buf_scale)
                    reward += comp
                    step_comp['heuristic'] += comp

                # 4) Buffer-vs-network average differential reward (heuristic-style thresholds)
                cached = Handler.STATE_CACHE.get(key) if key else None
                avg_buf_cached = None
                if cached:
                    avg_buf_cached = cached.get("buf_mean_global", cached.get("buf_mean"))
                    if avg_buf_cached is not None and not math.isfinite(avg_buf_cached):
                        avg_buf_cached = None
                if (
                    relayed > 0
                    and avg_buf_cached is not None
                    and avg_buf_cached <= BUFFER_DIFF_LOW_THRESHOLD
                ):
                    Handler.LOW_ZONE_RELAY_KEYS.add(key)
                if (
                    BUFFER_DIFF_REWARD_WEIGHT != 0.0
                    and cached
                    and math.isfinite(my_buf_norm)
                ):
                    avg_buf = current_global_buf
                    if avg_buf is not None and math.isfinite(avg_buf):
                        state_sign = 0.0
                        if avg_buf <= BUFFER_DIFF_LOW_THRESHOLD:
                            state_sign = 0  # small reward when buffers are scarce system-wide
                        elif avg_buf >= BUFFER_DIFF_HIGH_THRESHOLD:
                            state_sign = -1  # penalty when network-wide buffers are saturated
                        else:
                            state_sign = 0  # strong reward for keeping buffers inside mid range
                        base_weight = abs(BUFFER_DIFF_REWARD_WEIGHT)
                        if relayed > 0 and state_sign != 0.0:
                            event_scale = relayed / max(1.0, total_messages)
                            comp = state_sign * base_weight * event_scale
                            reward += comp
                            step_comp['buffer_diff'] += comp
                        if aborted > 0 and BUFFER_DIFF_ABORT_WEIGHT != 0.0:
                            # Abort is always treated as penalty to discourage wasting contacts (configurable)
                            abort_weight = abs(BUFFER_DIFF_ABORT_WEIGHT)
                            event_scale_fail = aborted / max(1.0, total_messages)
                            comp_fail = -abort_weight * event_scale_fail
                            reward += comp_fail
                            step_comp['buffer_diff'] += comp_fail

                # 4a) Skip bonus when utilization slope rises (encourage deferring to stronger peers)
                if (
                    SKIP_LOW_PRED_PENALTY_WEIGHT != 0.0
                    and skip_low_pred > 0
                    and buf_slope > 0.0
                    and slope_penalty_scale > 0.0
                ):
                    event_scale = skip_low_pred / max(1.0, total_messages)
                    comp = abs(SKIP_LOW_PRED_PENALTY_WEIGHT) * slope_penalty_scale * event_scale
                    reward += comp
                    step_comp['skip_low_pred_pen'] += comp
                    if reward_mixer_records is not None:
                        reward_mixer_records['skip_low_pred_pen'].append(
                            RewardChannelEvent(host=host, key=key, value=comp)
                        )

                # 4a-2) Relay bonus when utilization slope drops (encourage transmitting during recovery)
                if (
                    PEER_HIGH_PRED_PENALTY_WEIGHT != 0.0
                    and peer_high_pred > 0
                    and buf_slope < 0.0
                    and slope_penalty_scale > 0.0
                ):
                    event_scale = peer_high_pred / max(1.0, total_messages)
                    comp = abs(PEER_HIGH_PRED_PENALTY_WEIGHT) * slope_penalty_scale * event_scale
                    reward += comp
                    step_comp['peer_high_pred_pen'] += comp
                    if reward_mixer_records is not None:
                        reward_mixer_records['peer_high_pred_pen'].append(
                            RewardChannelEvent(host=host, key=key, value=comp)
                        )
                if (
                    DROP_TTL_PENALTY_WEIGHT != 0.0
                    and drops > 0
                ):
                    drop_ttl_ratio = ttl_ratio
                    if not math.isfinite(drop_ttl_ratio) or drop_ttl_ratio < 0.0:
                        drop_ttl_ratio = 0.0
                    drop_ttl_ratio = max(0.0, min(1.0, drop_ttl_ratio))
                    ttl_age = max(0.0, 1.0 - drop_ttl_ratio)
                    ttl_scale = 0.0
                    if ttl_age > 0.0:
                        try:
                            ttl_scale = 1.0 - math.exp(
                                -(ttl_age ** 2) / (2.0 * (DROP_TTL_GAUSS_SIGMA ** 2))
                            )
                        except Exception:
                            ttl_scale = 0.0
                        ttl_scale = max(0.0, min(1.0, ttl_scale))
                    if ttl_scale > 0.0:
                        comp = -abs(DROP_TTL_PENALTY_WEIGHT) * ttl_scale * drops / max(1.0, total_messages)
                        reward += comp
                        step_comp['drop_ttl_penalty'] += comp

                # 4b) Buffer × PBase Intersect (Hard AND gating)
                if (
                    (BUFFER_PBASE_AND_WEIGHT != 0.0 or BUFFER_PBASE_AND_NEG_WEIGHT != 0.0)
                    and relayed > 0
                    and math.isfinite(my_p_base) and math.isfinite(neighbor_p_base)
                    and cached is not None and math.isfinite(my_buf_norm)
                ):
                    try:
                        avg_buf2 = cached.get("buf_mean_local", cached.get("buf_mean")) if cached else None
                        if avg_buf2 is not None and math.isfinite(avg_buf2):
                            d_buf = my_buf_norm - avg_buf2
                            d_p = neighbor_p_base - my_p_base
                            # Positive reward: both above thresholds
                            if (d_buf >= BUFFER_PBASE_EPS_BUF) and (d_p >= BUFFER_PBASE_EPS_PBASE) and BUFFER_PBASE_AND_WEIGHT != 0.0:
                                comp_and = BUFFER_PBASE_AND_WEIGHT * relayed
                                reward += comp_and
                                step_comp['pbase_buf_and'] += comp_and
                            # Negative penalty: want to send (buffer high) but neighbor worse
                            if (d_buf >= BUFFER_PBASE_EPS_BUF) and (d_p <= -BUFFER_PBASE_EPS_PBASE) and BUFFER_PBASE_AND_NEG_WEIGHT != 0.0:
                                pen_and = BUFFER_PBASE_AND_NEG_WEIGHT * relayed
                                reward -= pen_and
                                step_comp['pbase_buf_and_pen'] -= pen_and
                    except Exception:
                        pass

                if reward != 0.0:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + reward
                    if key:
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + reward

                if key and (delivered > 0 or drops > 0 or aborted > 0):
                    done_keys.add(key)
                if key and (drops > 0 or aborted > 0):
                    stress_keys.add(key)

                Handler.EP_DELIVERED_SUM += delivered
                if delivered > 0:
                    step_ttl_delivered += delivered * ttl_weight
                Handler.EP_DROPPED_SUM += drops
                Handler.EP_RELAYED_SUM += relayed
            except Exception:
                continue

        Handler.EP_CREATED_SUM += created_increment
        Handler.EP_DELIVERED_TTL_SUM += step_ttl_delivered

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
                global_avg_buf = _to_float(item.get("global_avg_buf", 0.0), 0.0)

                if any(math.isnan(x) or math.isinf(x) for x in (c, p, buf_mean, cap_norm, self_util, pressure_diff, rate_delta)):
                    continue

                obs_batch.append([c, p, buf_mean, cap_norm, self_util, pressure_diff, rate_delta])

                # Global features
                g_features = [
                    _to_float(item.get("global_active_conns", 0.0), 0.0),
                    _to_float(item.get("global_total_msgs", 0.0), 0.0),
                    global_avg_buf,
                    _to_float(item.get("global_avg_contacts", 0.0), 0.0),
                    _to_float(item.get("global_msg_change_rate", 0.0), 0.0),
                    _to_float(item.get("global_msg_change_momentum", 0.0), 0.0),
                    _to_float(item.get("global_buf_util_change_rate", 0.0), 0.0),
                    _to_float(item.get("global_buf_util_momentum", 0.0), 0.0),
                    _to_float(item.get("global_avg_free_buf", 0.0), 0.0),
                    _to_float(item.get("global_high_util_frac", 0.0), 0.0),
                    _to_float(item.get("global_high_util_flag", 0.0), 0.0),
                    _to_float(item.get("global_avg_buf_delta", 0.0), 0.0),
                    _to_float(item.get("global_episode_time_norm", 0.0), 0.0),
                    _to_float(item.get("global_created_rate", 0.0), 0.0),
                ]

                if any(math.isnan(x) or math.isinf(x) for x in g_features):
                    continue

                global_features_batch.append(g_features)

                k = f"{host}#{dest}"
                keys.append(k)
                filtered_items.append(item)  # Store item that passed validation
                map_host_to_keys[host].append(k)
                prev_entry = Handler.STATE_CACHE.get(k)
                prev_global = None
                prev_self = None
                if prev_entry:
                    prev_global = prev_entry.get("buf_mean_global", prev_entry.get("buf_mean"))
                    prev_self = prev_entry.get("self_buf_ema", prev_entry.get("self_util"))
                if prev_global is not None and math.isfinite(prev_global):
                    smoothed_global = BUFFER_DIFF_EMA_BETA * prev_global + (1.0 - BUFFER_DIFF_EMA_BETA) * global_avg_buf
                else:
                    smoothed_global = global_avg_buf
                if prev_self is not None and math.isfinite(prev_self):
                    smoothed_self = BUFFER_DIFF_SELF_EMA_BETA * prev_self + (1.0 - BUFFER_DIFF_SELF_EMA_BETA) * self_util
                else:
                    smoothed_self = self_util
                key_self_util[k] = self_util
                key_pred[k] = p
                key_pressure_diff[k] = pressure_diff
                Handler.STATE_CACHE[k] = {
                    "buf_mean": smoothed_global,
                    "buf_mean_local": buf_mean,
                    "buf_mean_global": smoothed_global,
                    "self_buf_ema": smoothed_self,
                    "self_util": self_util,
                    "timestamp": sim_time,
                }
            except Exception:
                continue

        # Episode accumulation moved to after all reward contributions

        prev_occ_for_delta = Handler.LAST_OCC_AVG
        Handler.LAST_OCC_AVG = current_global_buf if math.isfinite(current_global_buf) else Handler.LAST_OCC_AVG
        if map_host_to_keys:
            Handler.LAST_ACTIVE_HOSTS = set(map_host_to_keys.keys())

        # State occupancy bonus now handled per-delivery (see delivery block)

        # Delta-based state reward encouraging movement toward desired occupancy
        if (
            STATE_DELTA_BONUS_WEIGHT > 0
            and prev_occ_for_delta is not None
            and math.isfinite(current_global_buf)
            and map_host_to_keys
        ):
            prev_norm = max(0.0, min(1.0, prev_occ_for_delta))
            curr_norm = max(0.0, min(1.0, current_global_buf))
            try:
                sigma_sq = STATE_DELTA_BONUS_SIGMA ** 2
                falloff = math.exp(-((curr_norm - SR_SPARSE_OCC_CENTER) ** 2) / (2.0 * sigma_sq))
            except Exception:
                falloff = 0.0
            abs_prev = abs(prev_norm - SR_SPARSE_OCC_CENTER)
            abs_curr = abs(curr_norm - SR_SPARSE_OCC_CENTER)
            delta_term = abs_prev - abs_curr
            state_delta_total = STATE_DELTA_BONUS_WEIGHT * delta_term * falloff
            if state_delta_total != 0.0:
                per_host = state_delta_total / len(map_host_to_keys)
                for host in map_host_to_keys.keys():
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host
                step_comp['state_delta_bonus'] += state_delta_total

        if (
            SPARSE_SUCCESS_REWARD
            and SR_SPARSE_WEIGHT != 0.0
            and sparse_delivery_events
        ):
            host_count = len(map_host_to_keys) if map_host_to_keys else 0
            if host_count == 0 and Handler.LAST_ACTIVE_HOSTS:
                host_count = len(Handler.LAST_ACTIVE_HOSTS)
            norm = max(1, host_count) if (SR_SPARSE_NORMALIZE and host_count > 0) else 1.0
            sparse_logs = []
            for ev in sparse_delivery_events:
                host = ev.get("host")
                key = ev.get("key")
                delivered_amt = ev.get("delivered", 0.0) or 0.0
                if delivered_amt <= 0:
                    continue
                ttl_ratio = _to_float(ev.get("ttl_ratio", 1.0), 1.0)
                if not math.isfinite(ttl_ratio) or ttl_ratio < 0.0:
                    ttl_ratio = 0.0
                ttl_ratio = max(0.0, min(1.0, ttl_ratio))
                avg_hops = _to_float(ev.get("avg_hops", 0.0), 0.0)
                if not math.isfinite(avg_hops) or avg_hops < 0.0:
                    avg_hops = 0.0
                avg_buf = _to_float(ev.get("avg_buf", 0.0), 0.0)
                if not math.isfinite(avg_buf) or avg_buf < 0.0:
                    avg_buf = 0.0
                avg_buf = max(0.0, min(1.0, avg_buf))
                base_term = SR_SPARSE_WEIGHT * delivered_amt * ttl_ratio
                bonus = base_term
                if bonus <= 0.0:
                    continue
                if SR_SPARSE_NORMALIZE:
                    bonus /= norm
                cooldown_scale = 1.0
                if SR_SPARSE_COOLDOWN > 0.0 and host:
                    last_time = Handler.LAST_SPARSE_EVENT_TIME.get(host)
                    if last_time is not None:
                        elapsed = max(0.0, float(sim_time) - float(last_time))
                        if elapsed < SR_SPARSE_COOLDOWN:
                            cooldown_scale = elapsed / SR_SPARSE_COOLDOWN
                    Handler.LAST_SPARSE_EVENT_TIME[host] = float(sim_time)
                    bonus *= cooldown_scale

                effective_bonus = bonus
                if SR_SPARSE_EMA_BETA > 0.0:
                    prev_bonus = Handler.SPARSE_BONUS_EMA
                    if prev_bonus is None or not math.isfinite(prev_bonus):
                        smoothed_bonus = bonus
                    else:
                        smoothed_bonus = (
                            SR_SPARSE_EMA_BETA * prev_bonus + (1.0 - SR_SPARSE_EMA_BETA) * bonus
                        )
                    Handler.SPARSE_BONUS_EMA = smoothed_bonus
                    effective_bonus = smoothed_bonus
                else:
                    Handler.SPARSE_BONUS_EMA = bonus

                if host:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + effective_bonus
                if key:
                    rewards_per_key[key] = rewards_per_key.get(key, 0.0) + effective_bonus
                step_comp['sparse_delivery'] += effective_bonus
                step_comp['sr_sparse_bonus'] += effective_bonus
                step_comp['sr_sparse_count'] += 1
                if reward_mixer_records is not None:
                    reward_mixer_records['sr_sparse_bonus'].append(
                        RewardChannelEvent(host=host, key=key, value=effective_bonus)
                    )
                sparse_logs.append({
                    "host": host or "",
                    "dest": ev.get("dest", "") or "",
                    "delivered": delivered_amt,
                    "reward": effective_bonus,
                    "active_hosts": host_count,
                    "norm_factor": norm,
                    "cooldown_scale": cooldown_scale,
                    "ttl_ratio": ttl_ratio,
                    "avg_hops": avg_hops,
                    "avg_buf": avg_buf,
                })
            if sparse_logs:
                log_sparse_delivery_events(sim_time, sparse_logs, buffer_size_mb, Handler.EP_COUNT)

        # Derive network-wide high load indicator from global features
        high_load_factor = 0.0
        high_load_active = False
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
                avg_buf_value = float(global_features_batch[0][2])
            except Exception:
                high_load_factor = 0.0
                high_load_active = False
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

        if reward_mixer_records and Handler.REWARD_MIXER:
            try:
                ctx_vec = build_reward_mixer_context(
                    Handler.STEP_SR_SMOOTHED,
                    Handler.STEP_SR_SMOOTHED,
                    Handler.SR_DELTA_EMA,
                    Handler.BUF_RATE_EMA,
                    dynamic_zone,
                )
                apply_reward_mixer_adjustments(
                    Handler.REWARD_MIXER,
                    reward_mixer_records,
                    step_comp,
                    rewards_per_host,
                    rewards_per_key,
                    ctx_vec,
                )
            except Exception as mix_exc:
                print(f"[REWARD_MIXER] Failed to rebalance rewards: {mix_exc}")

        # Optional legacy TARGET_REWARD_* path (time-based trigger)
        if Handler.STEP_SR_SMOOTHED is not None and not Handler.TARGET_REWARD_APPLIED:
            if (
                TARGET_REWARD_TIME > 0
                and TARGET_REWARD_WEIGHT > 0
                and sim_time >= TARGET_REWARD_TIME
            ):
                bonus = TARGET_REWARD_WEIGHT * Handler.STEP_SR_SMOOTHED
                target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
                if target_hosts and bonus != 0.0:
                    per_host_bonus = bonus
                    for host in target_hosts:
                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host_bonus
                        for key in map_host_to_keys.get(host, []):
                            rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host_bonus
                    # host-level total contribution for logging
                    step_comp['target_bonus'] += per_host_bonus * len(target_hosts)
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
                step_comp['high_load_penalty'] += total_penalty
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
                        step_comp['mid_util_penalty'] += penalty
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
                        step_comp['trend_penalty'] += penalty
            if trend_penalized > 0:
                print(f"[REWARD] t={sim_time:.1f}s trend penalty applied={trend_penalized} "
                      f"avg_slope>= {TREND_DELTA_THRESHOLD:.3f}")

        sr_reward_metric = None
        if Handler.STEP_SR_SMOOTHED is not None and math.isfinite(Handler.STEP_SR_SMOOTHED):
            sr_reward_metric = Handler.STEP_SR_SMOOTHED

        baseline_prev = Handler.LAST_SUCCESS_RATE if (
            Handler.LAST_SUCCESS_RATE is not None and math.isfinite(Handler.LAST_SUCCESS_RATE)
        ) else None

        # DEBUG: SR reward conditions
        print(f"[DEBUG_SR] t={sim_time:.0f} sr_metric={sr_reward_metric} baseline={baseline_prev} "
              f"STEP_SR={Handler.STEP_SR_SMOOTHED} LAST_SR={Handler.LAST_SUCCESS_RATE} "
              f"SR_COMBINE={SR_COMBINE_WITH_STEP} SPARSE={SPARSE_SUCCESS_REWARD}")

        # Penalize/bonus based on divergence between baseline success rate and current step rate.
        if (
            sr_reward_metric is not None
            and baseline_prev is not None
            and (SR_COMBINE_WITH_STEP or not SPARSE_SUCCESS_REWARD)
        ):
            delta_sr = sr_reward_metric - baseline_prev
            # Update EMA of |delta_sr| for normalization
            try:
                abs_delta = abs(delta_sr)
                if Handler.SR_DELTA_EMA is None or not math.isfinite(Handler.SR_DELTA_EMA):
                    Handler.SR_DELTA_EMA = abs_delta
                    Handler.SR_EMA_COUNT = 1
                else:
                    Handler.SR_DELTA_EMA = (
                        SR_DELTA_EMA_BETA * Handler.SR_DELTA_EMA
                        + (1.0 - SR_DELTA_EMA_BETA) * abs_delta
                    )
                    Handler.SR_EMA_COUNT += 1
            except Exception:
                pass

            # Adaptive EPS threshold (optional)
            eps_thr = SUCCESS_RATE_EPS
            if SR_ADAPTIVE_EPS and Handler.SR_DELTA_EMA is not None and math.isfinite(Handler.SR_DELTA_EMA):
                try:
                    eps_thr = max(SR_EPS_FLOOR, SR_EPS_K * Handler.SR_DELTA_EMA)
                except Exception:
                    eps_thr = SUCCESS_RATE_EPS

            # Deadband: ignore tiny jitters around zero
            rate_delta = delta_sr
            if abs(rate_delta) <= eps_thr:
                shaped_delta = 0.0
            else:
                shaped_delta = shape_success_rate_delta(rate_delta)

            # Normalization gain based on EMA (with warm-up and clamping)
            mult = 1.0
            use_norm = SR_NORM_ENABLE and Handler.SR_EMA_COUNT >= SR_EMA_WARMUP_STEPS
            if use_norm:
                try:
                    denom = max(
                        SR_NORM_MIN_DENOM,
                        2.0 * SUCCESS_RATE_EPS,
                        Handler.SR_DELTA_EMA if Handler.SR_DELTA_EMA is not None else 0.0,
                    )
                    if denom <= 1e-12 or not math.isfinite(denom):
                        denom = SR_NORM_MIN_DENOM
                    mult = 1.0 / denom
                    if not math.isfinite(mult):
                        mult = 1.0
                    mult = max(1.0, min(SR_NORM_MAX_GAIN, mult))
                except Exception:
                    mult = 1.0

            target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)

            # If combining sparse with stepwise, optionally scale the stepwise weights
            eff_penalty_w = SUCCESS_RATE_WEIGHT
            eff_bonus_w = SUCCESS_RATE_BONUS_WEIGHT
            if SPARSE_SUCCESS_REWARD and SR_COMBINE_WITH_STEP:
                eff_penalty_w = SUCCESS_RATE_WEIGHT * SR_COMBINED_STEP_SCALE
                eff_bonus_w = SUCCESS_RATE_BONUS_WEIGHT * SR_COMBINED_STEP_SCALE

            neg_delta = min(shaped_delta, 0.0)
            pos_delta = max(shaped_delta, 0.0)

            if neg_delta < 0.0 and eff_penalty_w > 0 and target_hosts:
                penalty = (eff_penalty_w * mult) * neg_delta  # neg_delta<=0 => penalty<=0
                for host in target_hosts:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + penalty
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + penalty
                # host-level total contribution (penalty < 0)
                step_comp['success_delta'] += penalty * len(target_hosts)
                step_comp['success_penalty'] += penalty * len(target_hosts)
                print(
                    f"[REWARD] t={sim_time:.1f}s success-rate penalty raw_delta={rate_delta:.4f} exp_delta={neg_delta:.4f} "
                    f"ema={Handler.SR_DELTA_EMA if Handler.SR_DELTA_EMA is not None else float('nan'):.4f} "
                    f"eps={eps_thr:.4f} mult={mult:.2f} eff_w={(eff_penalty_w*mult):.3f}"
                )
            if pos_delta > 0.0 and eff_bonus_w > 0 and target_hosts:
                bonus = (eff_bonus_w * mult) * pos_delta
                for host in target_hosts:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + bonus
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + bonus
                # host-level total contribution
                step_comp['success_delta'] += bonus * len(target_hosts)
                step_comp['success_bonus'] += bonus * len(target_hosts)
                print(
                    f"[REWARD] t={sim_time:.1f}s success-rate bonus  raw_delta={rate_delta:.4f} exp_delta={pos_delta:.4f} "
                    f"ema={Handler.SR_DELTA_EMA if Handler.SR_DELTA_EMA is not None else float('nan'):.4f} "
                    f"eps={eps_thr:.4f} mult={mult:.2f} eff_w={(eff_bonus_w*mult):.3f}"
                )

        if sr_reward_metric is not None:
            if baseline_prev is None:
                Handler.LAST_SUCCESS_RATE = sr_reward_metric
            else:
                Handler.LAST_SUCCESS_RATE = (
                    SUCCESS_RATE_SMOOTH_BETA * baseline_prev
                    + (1.0 - SUCCESS_RATE_SMOOTH_BETA) * sr_reward_metric
                )
            Handler.SR_SMOOTHED = Handler.LAST_SUCCESS_RATE
        else:
            Handler.SR_SMOOTHED = None

        if relay_occ_step_total != 0.0 and math.isfinite(current_global_buf):
            print(
                f"[REWARD] t={sim_time:.1f}s relay occ={current_global_buf:.3f} "
                f"total={relay_occ_step_total:.5f}"
            )

        sr_bonus_total = 0.0
        if (
            SR_BONUS_WEIGHT != 0.0
            and ttled_sr_scale is not None
            and math.isfinite(ttled_sr_scale)
        ):
            sr_bonus_total = SR_BONUS_WEIGHT * ttled_sr_scale
            target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
            if sr_bonus_total != 0.0 and target_hosts:
                per_host = sr_bonus_total / len(target_hosts)
                for host in target_hosts:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host
                    for key in map_host_to_keys.get(host, []):
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host
                step_comp['sr_bonus'] += sr_bonus_total

        # After all contributions: accumulate episode total and log components
        try:
            total_step_reward = sum(rewards_per_host.values()) if rewards_per_host else 0.0
            Handler.EP_REWARD_ACC = (Handler.EP_REWARD_ACC or 0.0) + float(total_step_reward)
            if STEP_REWARD_TRACKING:
                step_comp['total_hosts_reward'] = total_step_reward
                log_step_reward_components(sim_time, step_comp, buffer_size_mb, Handler.EP_COUNT)
                # Optional simple step log
                log_step_rewards(sim_time, rewards_per_host or {}, buffer_size_mb, Handler.EP_COUNT)
        except Exception:
            pass

        Handler.EP_HIGH_PENALTY_SUM += step_comp.get('peer_high_pred_pen', 0.0)
        Handler.EP_LOW_PENALTY_SUM += step_comp.get('skip_low_pred_pen', 0.0)

        # Update policy
        update_result = {}
        if TORCH_OK and Handler.TRAINING_ENABLED:
            try:
                update_result = Handler.AGENT.update(
                    rewards_per_host,
                    rewards_per_key,
                    map_host_to_keys,
                    delta_limit,
                    done_keys=done_keys,
                    low_zone_relays=list(Handler.LOW_ZONE_RELAY_KEYS),
                    train_now=not EPISODIC_UPDATES,
                    allow_cleanup=not EPISODIC_UPDATES,
                    allow_trim=not EPISODIC_UPDATES
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
            finally:
                Handler.LOW_ZONE_RELAY_KEYS.clear()

            # Debug: Log update result
            if update_result.get("updated"):
                print(f"[DEBUG] t={sim_time:.1f}s: PPO update completed!")
                print(
                    f"[DEBUG]   Losses: policy={update_result.get('policy_loss', 0):.4f}, "
                    f"value={update_result.get('value_loss', 0):.4f}, "
                    f"entropy={update_result.get('entropy', 0):.4f}"
                )
                print(
                    f"[DEBUG]   Stats: ratio_mean={update_result.get('ratio_mean', 0):.4f}, "
                    f"ratio_std={update_result.get('ratio_std', 0):.4f}, kl={update_result.get('kl', 0):.4f}, "
                    f"std_mean={update_result.get('std_mean', 0):.4f}, adv_var={update_result.get('adv_var', 0):.4e}"
                )
                try:
                    # Accumulate per-episode losses for plotting
                    Handler.EP_LOSS_POLICY_SUM += float(update_result.get('policy_loss', 0.0) or 0.0)
                    Handler.EP_LOSS_VALUE_SUM += float(update_result.get('value_loss', 0.0) or 0.0)
                    Handler.EP_LOSS_ENTROPY_SUM += float(update_result.get('entropy', 0.0) or 0.0)
                    Handler.EP_LOSS_UPDATE_COUNT += 1
                except Exception:
                    pass
                # Log per-update metrics to CSV
                try:
                    buffer_size = extract_buffer_size_mb(sim_id)
                    log_update_metrics(sim_time, update_result, buffer_size, Handler.EP_COUNT)
                except Exception:
                    pass
            elif (not EPISODIC_UPDATES) and update_result.get("ready_sequences"):
                ready = update_result.get("ready_sequences", 0)
                need = getattr(Handler.AGENT, "min_sequences_to_train", 0)
                print(f"[DEBUG] t={sim_time:.1f}s: {ready} sequences ready (need {need} for training)")
        else:
            Handler.LOW_ZONE_RELAY_KEYS.clear()

        # Infer actions
        if TORCH_OK:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit, keys, global_features_batch)
        else:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit)

        # Group actions by host
        per_host = defaultdict(list)
        flat_map = {}
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

        actions = [{"host": h, "per_message": v} for h, v in per_host.items()]
        resp = {"policy_id": Handler.POLICY_ID, "actions": actions, "actions_kv": flat_map}


        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._safe_write(body)


bootstrap_mixer = getattr(Handler, "_BOOTSTRAP_MIXER_LOAD", None)
if bootstrap_mixer:
    try:
        Handler.load_reward_mixer_checkpoint(bootstrap_mixer)
    except Exception as exc:
        print(f"[BOOT] Mixer load failed for {bootstrap_mixer}: {exc}")

bootstrap_save = getattr(Handler, "_BOOTSTRAP_MIXER_SAVE", None)
if bootstrap_save:
    try:
        Handler.save_reward_mixer_checkpoint(bootstrap_save)
    except Exception as exc:
        print(f"[BOOT] Mixer save failed for {bootstrap_save}: {exc}")


if META_AGENT_ENABLED and TORCH_OK:
    try:
        META_CONTROLLER = MetaAgentController(log_dir=META_AGENT_LOG_DIR)
        META_CONTROLLER.sync_from_values(
            current_lr=getattr(Handler.AGENT, "lr", META_AGENT_LR_CHOICES[0]),
            current_entropy=getattr(Handler.AGENT, "entropy_coef", META_AGENT_ENTROPY_CHOICES[0]),
        )
        main_print(
            "[META_AGENT] Enabled off-policy LR/entropy controller "
            f"(lr_range=[{META_AGENT_LR_MIN}, {META_AGENT_LR_MAX}], "
            f"entropy_range=[{META_AGENT_ENTROPY_MIN}, {META_AGENT_ENTROPY_MAX}])"
        )
    except Exception as meta_init_exc:
        META_CONTROLLER = None
        print(f"[META_AGENT] Failed to initialize controller: {meta_init_exc}")


def main():
    main_print(f"Starting R-MAPPO server at http://{HOST}:{PORT}/infer_and_update")
    main_print(f"  - PyTorch: {'enabled' if TORCH_OK else 'disabled'}")
    main_print(f"  - GRU hidden size: {LSTM_HIDDEN_SIZE}")
    main_print(f"  - GRU layers: {LSTM_NUM_LAYERS}")
    main_print(f"  - Truncated BPTT: {TRUNCATED_BPTT_LEN} steps")
    main_print(f"  - Mode: {'EVAL_ONLY' if EVAL_ONLY else 'TRAINING'}")
    main_print(f"  - Updates: {'EPISODIC' if EPISODIC_UPDATES else 'STREAMING'}")
    if SPARSE_SUCCESS_REWARD and SR_COMBINE_WITH_STEP:
        main_print("  - Success-Rate Reward: SPARSE+STEP (combined)")
        main_print(f"    * SR_SPARSE_WEIGHT={SR_SPARSE_WEIGHT} (per-delivery), normalize={SR_SPARSE_NORMALIZE}")
        main_print(f"    * Stepwise scale (combined) SR_COMBINED_STEP_SCALE={SR_COMBINED_STEP_SCALE}")
    elif SPARSE_SUCCESS_REWARD:
        main_print("  - Success-Rate Reward: SPARSE_ONLY (per-step delta disabled)")
        main_print(f"    * SR_SPARSE_WEIGHT={SR_SPARSE_WEIGHT} (per-delivery), normalize={SR_SPARSE_NORMALIZE}")
    else:
        main_print("  - Success-Rate Reward: DENSE (EMA-normalized deltas)")
    if NO_TRAINING_CAP:
        main_print("  - Training Cap: DISABLED (NO_TRAINING_CAP=1)")
    else:
        main_print(f"  - Training Cap: TOTAL_EPISODES={TOTAL_EPISODES}")
    if TORCH_OK:
        try:
            main_print(f"  - PPO LR={Handler.AGENT.lr}; value_coef={Handler.AGENT.value_coef}; gamma={Handler.AGENT.gamma}; gae_lambda={Handler.AGENT.gae_lambda}")
        except Exception:
            pass
    if TORCH_OK:
        try:
            upper_desc = "no upper bound" if LOG_STD_MAX is None or LOG_STD_MAX <= 0 else f"<= {LOG_STD_MAX}"
            main_print(f"  - Exploration clamp: log_std >= {Handler.AGENT.log_std_min} ({upper_desc})")
            main_print(
                f"  - Entropy coef schedule: {ENTROPY_COEF_START} (<=ep{ENTROPY_DECAY_START_EP})"
                f" -> {ENTROPY_COEF_END} by ep{ENTROPY_DECAY_END_EP}"
            )
            lr_schedule_desc = ", ".join(
                f"ep{start}-{end}={lr}" for start, end, lr in LR_SCHEDULE
            )
            main_print(f"  - LR schedule: {lr_schedule_desc}")
        except Exception:
            pass

    # Rate-based zone detection settings
    main_print("  - Rate-Based Zone Detection: ENABLED (data-driven 2σ thresholds)")
    main_print(f"    * RATE_HIGH_THRESHOLD={RATE_HIGH_THRESHOLD:.6f} (buffer increasing)")
    main_print(f"    * RATE_LOW_THRESHOLD={RATE_LOW_THRESHOLD:.6f} (buffer decreasing)")
    main_print(f"    * BUF_RATE_EMA_ALPHA={BUF_RATE_EMA_ALPHA:.2f} (smoothing factor)")

    srv = HTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
