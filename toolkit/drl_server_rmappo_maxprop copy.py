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
import copy
import math
import os
import re
import sys
import random
import pathlib
import time
try:
    import numpy as np
    NP_OK = True
except Exception:
    np = None
    NP_OK = False
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

# If numpy is missing, force-disable torch-mode (training code depends on numpy).
if TORCH_OK and not NP_OK:
    TORCH_OK = False

HOST = os.environ.get("DRL_HOST", "127.0.0.1")
PORT = int(float(os.environ.get("DRL_PORT", "5011")))

# ---------------------------------------------------------------------------
# Report/log output directory helpers
# ---------------------------------------------------------------------------
def _default_report_output_dir():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        repo_root = os.path.abspath(os.path.join(here, os.pardir))
        return os.path.join(repo_root, "reports")
    except Exception:
        return "reports"


DEFAULT_REPORT_OUTPUT_DIR = _default_report_output_dir()


def _sanitize_sim_id_for_filename(sim_id: str, max_len: int = 120) -> str:
    s = str(sim_id or "").strip()
    if not s:
        return ""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    if not s:
        return ""
    if len(s) > max_len:
        s = s[:max_len].rstrip("._-")
    return s


def _canonical_sim_id_for_group_logs(sim_id: str) -> str:
    """
    Canonicalize sim_id for log *filenames* so batch sweeps (e.g., rng1..rng10)
    append into a single per-scenario CSV while still keeping the original
    sim_id inside each row.
    """
    s = str(sim_id or "").strip()
    if not s:
        return ""
    # Common run-index suffix pattern used in this repo: "..._rng{N}"
    s = re.sub(r"_rng\d+\b", "", s)
    s = re.sub(r"_+", "_", s).strip("._-")
    return s


# ---------------------------------------------------------------------------
# Minimal, dependency-light server mode
# ---------------------------------------------------------------------------
# This repo's full R-MAPPO server depends on PyTorch and NumPy. For quick
# end-to-end smoke tests and for environments that don't have those packages
# installed, we still want a runnable server for the MaxProp++ v1 protocol.
if __name__ == "__main__" and not TORCH_OK:
    def _to_float(val, default=0.0):
        try:
            f = float(val)
        except Exception:
            return default
        if math.isnan(f) or math.isinf(f):
            return default
        return f

    def _to_int(val, default=0):
        try:
            i = int(float(val))
        except Exception:
            return default
        return i

    def _clamp(x, lo, hi):
        try:
            x = float(x)
        except Exception:
            x = lo
        if not math.isfinite(lo):
            lo = x
        if not math.isfinite(hi):
            hi = x
        if lo > hi:
            lo, hi = hi, lo
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _clamp01(x):
        if not math.isfinite(x):
            return 0.0
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return float(x)

    def _compute_v1_reward(req):
        mp_enable = os.environ.get("MP_V1_REWARD_ENABLE", "true").lower() in ("1", "true", "yes")
        if not mp_enable:
            return None
        mp_form = os.environ.get("MP_V1_REWARD_FORM", "prod").strip().lower()
        # Default scale reduced (1/100 of previous 1000.0) to keep value targets well-conditioned.
        mp_scale = _to_float(os.environ.get("MP_V1_REWARD_SCALE", "10.0"), 10.0)
        mp_eps = _to_float(os.environ.get("MP_V1_REWARD_EPS", "1e-6"), 1e-6)
        # Defaults: prioritize delivery success while keeping mild pressure on overhead/energy.
        # Tune via env vars as needed per scenario.
        mp_a = _to_float(os.environ.get("MP_V1_ALPHA_SUCCESS", "4.0"), 4.0)
        mp_b = _to_float(os.environ.get("MP_V1_BETA_OVERHEAD", "0"), 0.1)
        mp_g = _to_float(os.environ.get("MP_V1_GAMMA_ENERGY", "0"), 0.5)
        mp_d = _to_float(os.environ.get("MP_V1_DELTA_DROP", "0.0"), 0.0)
        e_norm = _to_float(os.environ.get("MP_V1_ENERGY_RATE_NORM", "2.5"), 2.5)
        e_mode = str(os.environ.get("MP_V1_ENERGY_RATE_MODE", "avg_per_host") or "").strip().lower()
        if e_mode not in ("avg", "avg_per_host", "total"):
            e_mode = "avg_per_host"
        if not math.isfinite(e_norm) or e_norm <= 1e-12:
            e_norm = 2.5
        dt = _to_float(req.get("dt", 1.0), 1.0)
        if not math.isfinite(dt) or dt <= 1e-9:
            dt = 1.0
        prev = req.get("prev_transition", []) or []
        created = 0
        transferred = 0
        delivered = 0
        dropped = 0
        energy_used = 0.0
        n_hosts = 0
        for tr in prev:
            if not isinstance(tr, dict):
                continue
            created += _to_int(tr.get("created_cnt", 0), 0)
            transferred += _to_int(tr.get("transferred_cnt", 0), 0)
            delivered += _to_int(tr.get("delivered_cnt", 0), 0)
            dropped += _to_int(tr.get("dropped", 0), 0)
            energy_used += _to_float(tr.get("energy_used", 0.0), 0.0)
            n_hosts += 1

        S = 0.0
        if created > 0:
            S = _clamp01(float(delivered) / float(created))
        extra = max(0.0, float(transferred) - float(delivered))
        O = _clamp01(extra / max(1.0, float(transferred)))
        D = 0.0
        if created > 0:
            D = _clamp01(float(dropped) / float(created))
        e_rate_total = max(0.0, float(energy_used)) / float(dt)
        denom_hosts = float(max(1, int(n_hosts)))
        e_rate_avg = e_rate_total / denom_hosts
        e_rate_used = e_rate_total if e_mode == "total" else e_rate_avg
        E = _clamp01(e_rate_used / max(1e-12, e_norm))

        if mp_form == "log":
            import math as _m
            term_s = _m.log(max(mp_eps, S))
            term_o = _m.log(max(mp_eps, O))
            term_e = _m.log(max(mp_eps, 1.0 - E))
            term_d = _m.log(max(mp_eps, 1.0 - D))
            total = mp_scale * (
                mp_a * term_s
                + mp_b * term_o
                + mp_g * term_e
                + mp_d * term_d
            )
        else:
            total = mp_scale * (
                (max(mp_eps, S) ** mp_a)
                * (max(mp_eps, O) ** mp_b)
                * (max(mp_eps, 1.0 - E) ** mp_g)
                * (max(mp_eps, 1.0 - D) ** mp_d)
            )

        return {
            "total": float(total),
            "S": float(S),
            "O": float(O),
            "E": float(E),
            "D": float(D),
            "form": str(mp_form),
            "scale": float(mp_scale),
            "energy_rate_total": float(e_rate_total),
            "energy_rate_avg": float(e_rate_avg),
            "energy_rate_mode": str(e_mode),
            "energy_rate_norm": float(e_norm),
            "created": int(created),
            "transferred": int(transferred),
            "delivered": int(delivered),
            "dropped": int(dropped),
            "energy_used": float(energy_used),
            "n_hosts": int(n_hosts),
        }

    def _handle(req):
        mode = str(req.get("mode", "")).strip().lower()
        if mode == "episode_end":
            # Minimal-mode episode logging: keep the same CSV filenames as torch mode.
            try:
                sim_id = str(req.get("sim_id", "") or "")
            except Exception:
                sim_id = ""
            try:
                sim_time_end = int(float(req.get("time", 0) or 0))
            except Exception:
                sim_time_end = 0

            def _extract_buf(sim_id_s):
                try:
                    import re
                    m = re.search(r'buf(\d+)', str(sim_id_s or "").lower())
                    return float(m.group(1)) if m else None
                except Exception:
                    return None

            buffer_size_mb = _extract_buf(sim_id)
            base_dir = os.environ.get(
                "REPORT_OUTPUT_DIR",
                os.environ.get("REPORT_DIR", DEFAULT_REPORT_OUTPUT_DIR),
            )
            reward_dir = os.path.join(base_dir, "reward_logs")
            train_dir = os.path.join(base_dir, "train_logs")
            os.makedirs(reward_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)

            # Use accumulated totals from this minimal server process.
            try:
                HandlerMin.EP_COUNT += 1
            except Exception:
                pass

            created = int(getattr(HandlerMin, "CREATED_SUM", 0) or 0)
            delivered = int(getattr(HandlerMin, "DELIVERED_SUM", 0) or 0)
            transferred = int(getattr(HandlerMin, "TRANSFERRED_SUM", 0) or 0)
            dropped = int(getattr(HandlerMin, "DROPPED_SUM", 0) or 0)
            total_reward = float(getattr(HandlerMin, "REWARD_SUM", 0.0) or 0.0)
            combo_reward = float(getattr(HandlerMin, "COMBO_SUM", 0.0) or 0.0)

            avg_delivery_rate = (float(delivered) / float(created)) if created > 0 else ""
            avg_overhead = ((float(transferred) - float(delivered)) / float(delivered)) if delivered > 0 else ""

            reward_files = []
            loss_files = []
            upd_files = []
            if buffer_size_mb:
                reward_files = [os.path.join(reward_dir, f"episode_rewards_buf{int(buffer_size_mb)}M_rmappo.csv")]
                loss_files = [os.path.join(train_dir, f"episode_losses_buf{int(buffer_size_mb)}M_rmappo.csv")]
                upd_files = [os.path.join(train_dir, f"update_metrics_buf{int(buffer_size_mb)}M_rmappo.csv")]
            else:
                safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(sim_id))
                if safe_sim:
                    reward_files.append(os.path.join(reward_dir, f"episode_rewards_{safe_sim}_rmappo.csv"))
                    loss_files.append(os.path.join(train_dir, f"episode_losses_{safe_sim}_rmappo.csv"))
                    upd_files.append(os.path.join(train_dir, f"update_metrics_{safe_sim}_rmappo.csv"))
                reward_files.append(os.path.join(reward_dir, "episode_rewards_rmappo.csv"))
                loss_files.append(os.path.join(train_dir, "episode_losses_rmappo.csv"))
                upd_files.append(os.path.join(train_dir, "update_metrics_rmappo.csv"))

            def _append_csv(path, header, row):
                write_header = not os.path.exists(path)
                with open(path, "a", encoding="utf-8") as f:
                    if write_header:
                        f.write(header + "\n")
                    f.write(row + "\n")

            # Record MP v1 reward parameters (so we can verify env overrides applied).
            try:
                mp_enable = os.environ.get("MP_V1_REWARD_ENABLE", "true").lower() in ("1", "true", "yes")
                mp_form = os.environ.get("MP_V1_REWARD_FORM", "prod").strip().lower()
                mp_scale = _to_float(os.environ.get("MP_V1_REWARD_SCALE", "10.0"), 10.0)
                mp_eps = _to_float(os.environ.get("MP_V1_REWARD_EPS", "1e-6"), 1e-6)
                mp_a = _to_float(os.environ.get("MP_V1_ALPHA_SUCCESS", "1.0"), 1.0)
                mp_b = _to_float(os.environ.get("MP_V1_BETA_OVERHEAD", "0.0"), 0.0)
                mp_g = _to_float(os.environ.get("MP_V1_GAMMA_ENERGY", "0.0"), 0.0)
                mp_d = _to_float(os.environ.get("MP_V1_DELTA_DROP", "0.0"), 0.0)
                e_norm = _to_float(os.environ.get("MP_V1_ENERGY_RATE_NORM", "2.5"), 2.5)
                e_mode = str(os.environ.get("MP_V1_ENERGY_RATE_MODE", "avg_per_host") or "").strip().lower()
                if e_mode not in ("avg", "avg_per_host", "total"):
                    e_mode = "avg_per_host"
                if not math.isfinite(e_norm) or e_norm <= 1e-12:
                    e_norm = 2.5

                safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(sim_id))
                param_files = []
                if buffer_size_mb:
                    param_files.append(os.path.join(reward_dir, f"reward_parameters_mp_v1_buf{int(buffer_size_mb)}M_rmappo.csv"))
                else:
                    if safe_sim:
                        param_files.append(os.path.join(reward_dir, f"reward_parameters_mp_v1_{safe_sim}_rmappo.csv"))
                    param_files.append(os.path.join(reward_dir, "reward_parameters_mp_v1_rmappo.csv"))

                header = (
                    "episode,sim_time,sim_id,MP_V1_REWARD_ENABLE,MP_V1_REWARD_FORM,"
                    "MP_V1_REWARD_SCALE,MP_V1_REWARD_EPS,MP_V1_ALPHA_SUCCESS,MP_V1_BETA_OVERHEAD,"
                    "MP_V1_GAMMA_ENERGY,MP_V1_DELTA_DROP,MP_V1_ENERGY_RATE_NORM,MP_V1_ENERGY_RATE_MODE"
                )
                ep_num = getattr(HandlerMin, "EP_COUNT", 1)
                row = (
                    f"{ep_num},{sim_time_end},{sim_id},{1 if mp_enable else 0},{mp_form},"
                    f"{mp_scale},{mp_eps},{mp_a},{mp_b},{mp_g},{mp_d},{e_norm},{e_mode}"
                )
                for pf in param_files:
                    # Backward-compatible: if an older header exists, write to a new *_v2.csv
                    try:
                        if os.path.exists(pf):
                            with open(pf, "r", encoding="utf-8") as rf:
                                first = (rf.readline() or "").strip()
                            if first and "MP_V1_ENERGY_RATE_MODE" not in first:
                                if pf.lower().endswith(".csv"):
                                    pf = pf[:-4] + "_v2.csv"
                    except Exception:
                        pass
                    _append_csv(pf, header, row)
            except Exception:
                pass

            ep_num = getattr(HandlerMin, "EP_COUNT", 1)
            for ep_reward_file in reward_files:
                _append_csv(
                    ep_reward_file,
                    "episode,sim_id,total_reward,combo_reward,avg_delivery_rate,delivered,created,avg_overhead,avg_delay",
                    f"{ep_num},{sim_id},{round(total_reward,6)},{round(combo_reward,6)},"
                    f"{avg_delivery_rate},{delivered},{created},{avg_overhead},",
                )
            for ep_loss_file in loss_files:
                # episode_losses (no training in minimal mode)
                _append_csv(
                    ep_loss_file,
                    "episode,sim_id,actor_loss_avg,critic_loss_avg,cost_critic_loss_avg,updates,entropy_avg",
                    f"{ep_num},{sim_id},,,,,0,",
                )
            for upd_file in upd_files:
                # update_metrics (no update in minimal mode)
                _append_csv(
                    upd_file,
                    "episode,sim_time,ratio_mean,ratio_std,kl,std_mean,adv_var,cost_value_loss",
                    f"{ep_num},{sim_time_end},0,0,0,0,0,0",
                )

            # Reset accumulators for next episode
            HandlerMin.REWARD_SUM = 0.0
            HandlerMin.COMBO_SUM = 0.0
            HandlerMin.CREATED_SUM = 0
            HandlerMin.TRANSFERRED_SUM = 0
            HandlerMin.DELIVERED_SUM = 0
            HandlerMin.DROPPED_SUM = 0

            return {"policy_id": "heuristic_v1", "status": "episode_end_ack", "actions": []}

        action_spec = req.get("action_spec") or {}
        lam_min = _to_float(action_spec.get("lambda_cost_min", 0.0), 0.0)
        lam_max = _to_float(action_spec.get("lambda_cost_max", 2.0), 2.0)
        tau_min = _to_float(action_spec.get("tau_age_min", 0.0), 0.0)
        tau_max = _to_float(action_spec.get("tau_age_max", 3600.0), 3600.0)
        beta_min = _to_float(action_spec.get("beta_x_min", 0.0), 0.0)
        beta_max = _to_float(action_spec.get("beta_x_max", 1.0), 1.0)
        kx_min = _to_float(action_spec.get("k_x_min", 0.1), 0.1)
        kx_max = _to_float(action_spec.get("k_x_max", 2.0), 2.0)
        mr_min = _to_float(action_spec.get("m_relay_min", -1e9), -1e9)
        mr_max = _to_float(action_spec.get("m_relay_max", 5.0), 5.0)

        tau_default = 600.0
        beta_default = 0.2
        kx_default = 1.0

        action_scope = str(req.get("action_scope", "")).strip().lower()

        items = (req.get("state_batch") or [])
        reward_debug = _compute_v1_reward(req)
        mp_debug_print = os.environ.get("MP_V1_REWARD_DEBUG_PRINT", "false").lower() in ("1", "true", "yes")
        if reward_debug is not None and mp_debug_print:
            try:
                step_id = int(float(req.get("step_id", -1) or -1))
            except Exception:
                step_id = -1
            print(
                "[MP_V1_REWARD] "
                f"step={step_id} total={reward_debug.get('total', 0.0):.6f} "
                f"S={reward_debug.get('S', 0.0):.3f} O={reward_debug.get('O', 0.0):.3f} "
                f"E={reward_debug.get('E', 0.0):.3f} D={reward_debug.get('D', 0.0):.3f} "
                f"e_rate={reward_debug.get('energy_rate', 0.0):.6f} "
                f"e_norm={reward_debug.get('energy_rate_norm', 0.0):.6f}"
            )
        # Accumulate episode totals for logging at episode_end.
        if reward_debug is not None:
            try:
                HandlerMin.REWARD_SUM = float(getattr(HandlerMin, "REWARD_SUM", 0.0) or 0.0) + float(reward_debug.get("total", 0.0) or 0.0)
                HandlerMin.COMBO_SUM = float(getattr(HandlerMin, "COMBO_SUM", 0.0) or 0.0) + float(reward_debug.get("total", 0.0) or 0.0)
                HandlerMin.CREATED_SUM = int(getattr(HandlerMin, "CREATED_SUM", 0) or 0) + int(reward_debug.get("created", 0) or 0)
                HandlerMin.TRANSFERRED_SUM = int(getattr(HandlerMin, "TRANSFERRED_SUM", 0) or 0) + int(reward_debug.get("transferred", 0) or 0)
                HandlerMin.DELIVERED_SUM = int(getattr(HandlerMin, "DELIVERED_SUM", 0) or 0) + int(reward_debug.get("delivered", 0) or 0)
                HandlerMin.DROPPED_SUM = int(getattr(HandlerMin, "DROPPED_SUM", 0) or 0) + int(reward_debug.get("dropped", 0) or 0)
            except Exception:
                pass
        if action_scope in ("global", "shared"):
            e_sum = 0.0
            b_sum = 0.0
            n = 0
            for item in items:
                obs = item.get("obs") or []
                e_norm = _to_float(obs[0], 0.5) if len(obs) > 0 else 0.5
                buf_occ = _to_float(obs[3], 0.0) if len(obs) > 3 else 0.0
                e_sum += e_norm
                b_sum += buf_occ
                n += 1

            e_avg = (e_sum / n) if n > 0 else 0.5
            b_avg = (b_sum / n) if n > 0 else 0.0

            lam = lam_min
            tau = _clamp(tau_default, tau_min, tau_max)
            beta = _clamp(beta_default, beta_min, beta_max)
            kx = _clamp(kx_default, kx_min, kx_max)
            mr = mr_min

            if b_avg >= 0.85 or e_avg <= 0.20:
                lam = _clamp(lam_min + 0.5 * (lam_max - lam_min), lam_min, lam_max)
                kx = _clamp(0.7, kx_min, kx_max)
                mr = _clamp(0.2, mr_min, mr_max)

            actions = []
            for item in items:
                host = str(item.get("host", ""))
                if not host:
                    continue
                actions.append({
                    "host": host,
                    "lambda_cost": float(lam),
                    "tau_age": float(tau),
                    "beta_x": float(beta),
                    "k_x": float(kx),
                    "m_relay": float(mr),
                })

            return {
                "policy_id": "heuristic_v1",
                "action_global": {
                    "lambda_cost": float(lam),
                    "tau_age": float(tau),
                    "beta_x": float(beta),
                    "k_x": float(kx),
                    "m_relay": float(mr),
                },
                "actions": actions,
                "reward_debug": reward_debug,
            }

        actions = []
        for item in items:
            host = str(item.get("host", ""))
            if not host:
                continue
            obs = item.get("obs") or []
            e_norm = _to_float(obs[0], 0.5) if len(obs) > 0 else 0.5
            buf_occ = _to_float(obs[3], 0.0) if len(obs) > 3 else 0.0

            lam = lam_min
            tau = _clamp(tau_default, tau_min, tau_max)
            beta = _clamp(beta_default, beta_min, beta_max)
            kx = _clamp(kx_default, kx_min, kx_max)
            mr = mr_min

            if buf_occ >= 0.85 or e_norm <= 0.20:
                lam = _clamp(lam_min + 0.5 * (lam_max - lam_min), lam_min, lam_max)
                kx = _clamp(0.7, kx_min, kx_max)
                mr = _clamp(0.2, mr_min, mr_max)

            actions.append({
                "host": host,
                "lambda_cost": float(lam),
                "tau_age": float(tau),
                "beta_x": float(beta),
                "k_x": float(kx),
                "m_relay": float(mr),
            })

        return {"policy_id": "heuristic_v1", "actions": actions, "reward_debug": reward_debug}

    class HandlerMin:
        EP_COUNT = 0
        REWARD_SUM = 0.0
        COMBO_SUM = 0.0
        CREATED_SUM = 0
        TRANSFERRED_SUM = 0
        DELIVERED_SUM = 0
        DROPPED_SUM = 0

    class _MinimalHandler(BaseHTTPRequestHandler):
        def _safe_write(self, payload):
            if isinstance(payload, str):
                payload = payload.encode("utf-8")
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_POST(self):
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

            proto = str(req.get("protocol", "")).strip()
            if proto != "rmappo_maxprop_v1":
                self.send_response(400)
                self.end_headers()
                self._safe_write(b"Unsupported protocol")
                return

            resp_obj = _handle(req)
            body = json.dumps(resp_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)

    print(f"[BOOT] torch unavailable; starting minimal MaxProp++ v1 server at http://{HOST}:{PORT}/infer_and_update")
    try:
        _mp_scale_dbg = os.environ.get("MP_V1_REWARD_SCALE", "10.0")
        _mp_form_dbg = os.environ.get("MP_V1_REWARD_FORM", "prod")
        _mp_enorm_dbg = os.environ.get("MP_V1_ENERGY_RATE_NORM", "2.5")
        _mp_emode_dbg = os.environ.get("MP_V1_ENERGY_RATE_MODE", "avg_per_host")
        print(f"[BOOT] MP_V1_REWARD_SCALE={_mp_scale_dbg} MP_V1_REWARD_FORM={_mp_form_dbg} MP_V1_ENERGY_RATE_NORM={_mp_enorm_dbg} MP_V1_ENERGY_RATE_MODE={_mp_emode_dbg}")
    except Exception:
        pass
    srv = HTTPServer((HOST, PORT), _MinimalHandler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    sys.exit(0)

# R-MAPPO specific configurations
LSTM_HIDDEN_SIZE = int(os.environ.get("LSTM_HIDDEN_SIZE", "128"))
LSTM_NUM_LAYERS = int(os.environ.get("LSTM_NUM_LAYERS", "1"))
TRUNCATED_BPTT_LEN = int(os.environ.get("TRUNCATED_BPTT_LEN", "80"))  # Sequence length (~40 min @100s sampling)
RESET_HIDDEN_ON_EPISODE = os.environ.get("RESET_HIDDEN_ON_EPISODE", "true").lower() == "true"

# Optional Layer Normalization between MLP and GRU layers in R-MAPPO
PPO_LAYER_NORM = os.environ.get("PPO_LAYER_NORM", "false").lower() in ("1", "true", "yes")

# Action logging configuration
PPO_DEBUG_ACTIONS = os.environ.get("PPO_DEBUG_ACTIONS", "false").lower() == "true"

# MP v1 action logging (episode-level aggregates). Default ON.
# Set MP_V1_ACTION_LOG_ENABLE=0 to disable.
MP_V1_ACTION_LOG_ENABLE = os.environ.get("MP_V1_ACTION_LOG_ENABLE", "true").lower() in ("1", "true", "yes")
MP_V1_ACTION_LOG_PATH = os.environ.get("MP_V1_ACTION_LOG_PATH", "").strip()

# Step-by-step reward tracking configuration
STEP_REWARD_TRACKING = os.environ.get("STEP_REWARD_TRACKING", "true").lower() == "true"
REPORT_OUTPUT_DIR = os.environ.get(
    "REPORT_OUTPUT_DIR",
    os.environ.get("REPORT_DIR", DEFAULT_REPORT_OUTPUT_DIR),
)
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
OVERHEAD_PENALTY_WEIGHT = float(os.environ.get("OVERHEAD_PENALTY_WEIGHT", "0.0"))
OVERHEAD_DECAY = float(os.environ.get("OVERHEAD_DECAY", "5.0"))
if OVERHEAD_DECAY <= 1e-6:
    OVERHEAD_DECAY = 1e-6
DROP_TTL_PENALTY_WEIGHT = float(os.environ.get("DROP_TTL_PENALTY_WEIGHT", "0"))
DROP_TTL_GAUSS_SIGMA = float(os.environ.get("DROP_TTL_GAUSS_SIGMA", "0.08"))
if DROP_TTL_GAUSS_SIGMA <= 1e-6:
    DROP_TTL_GAUSS_SIGMA = 1e-6

# ===== Combined step-level reward (success × (1-overhead)) =====
# Enable a dense, step-level shaping reward that promotes high delivery success
# with low overhead using a product (or log-sum) form.
COMBO_REWARD_ENABLE = os.environ.get("COMBO_REWARD_ENABLE", "true").lower() in ("1", "true", "yes")
COMBO_FORM = os.environ.get("COMBO_FORM", "prod").strip().lower()  # 'prod' or 'log'
COMBO_ALPHA = float(os.environ.get("COMBO_ALPHA", "1.3"))  # success exponent
COMBO_BETA = float(os.environ.get("COMBO_BETA", "4.0"))   # (1-overhead) exponent
COMBO_EPS = float(os.environ.get("COMBO_EPS", "1e-6"))
COMBO_SCALE = float(os.environ.get("COMBO_SCALE", "500000000.0"))
# When true, suppress existing overhead reward-shaping to avoid double-counting
COMBO_SUPPRESS_OVERHEAD_SHAPING = os.environ.get("COMBO_SUPPRESS_OVERHEAD_SHAPING", "true").lower() in ("1", "true", "yes")
# When true, suppress reward-level constraint penalty shaping (combo acts as main reward)
COMBO_SUPPRESS_CONSTRAINT_SHAPING = os.environ.get("COMBO_SUPPRESS_CONSTRAINT_SHAPING", "true").lower() in ("1", "true", "yes")
# Hindsight credit assignment: attribute delivery reward to the step when message was created
COMBO_HINDSIGHT_CREDIT = os.environ.get("COMBO_HINDSIGHT_CREDIT", "false").lower() in ("1", "true", "yes")
# TTL-weighted delivery: bonus scale for fast delivery (0 = no bonus, 0.5 = 50% bonus for ttl_ratio=1)
COMBO_TTL_BONUS_SCALE = float(os.environ.get("COMBO_TTL_BONUS_SCALE", "0.0"))
# TTL weighting mode for combo hindsight credit delivery events:
# - "none": each delivered msg counts as 1.0 (pure success-rate)
# - "bonus": each delivered msg counts as (1 + COMBO_TTL_BONUS_SCALE * ttl_ratio)
# - "exp": each delivered msg counts as exp(-k * (1 - ttl_ratio))  (late arrivals -> smaller)
COMBO_TTL_WEIGHT_MODE = os.environ.get("COMBO_TTL_WEIGHT_MODE", "bonus").strip().lower()
COMBO_TTL_EXP_K = float(os.environ.get("COMBO_TTL_EXP_K", "6.0"))
if COMBO_TTL_EXP_K < 0.0:
    COMBO_TTL_EXP_K = 0.0
# TTL weighting for *immediate* (non-hindsight) combo success-rate S:
# - "none": S uses plain delivered/created_since_last (legacy)
# - "exp":  S uses (Σ delivered * exp(-k*(1-ttl_ratio))) / created_since_last
COMBO_IMMEDIATE_TTL_WEIGHT_MODE = os.environ.get("COMBO_IMMEDIATE_TTL_WEIGHT_MODE", "exp").strip().lower()
COMBO_IMMEDIATE_TTL_EXP_K = float(os.environ.get("COMBO_IMMEDIATE_TTL_EXP_K", str(COMBO_TTL_EXP_K)))
if COMBO_IMMEDIATE_TTL_EXP_K < 0.0:
    COMBO_IMMEDIATE_TTL_EXP_K = 0.0
# Delay (in steps) before computing true delivery rate for a step
COMBO_HINDSIGHT_DELAY_STEPS = int(os.environ.get("COMBO_HINDSIGHT_DELAY_STEPS", "50"))

# ============================================================================
# MaxProp++ v1 (parameter-control) reward configuration
# ============================================================================
MP_V1_REWARD_ENABLE = os.environ.get("MP_V1_REWARD_ENABLE", "true").lower() in ("1", "true", "yes")
MP_V1_REWARD_FORM = os.environ.get("MP_V1_REWARD_FORM", "prod").strip().lower()  # 'prod' or 'log'
MP_V1_REWARD_SCALE = float(os.environ.get("MP_V1_REWARD_SCALE", "10.0"))
MP_V1_REWARD_EPS = float(os.environ.get("MP_V1_REWARD_EPS", str(COMBO_EPS)))

# MP_v1 controller mode
# - If enabled, bypass MA-SAC (including recurrent policy) and always use the built-in heuristic
#   action mapping for MaxProp++ parameters. Useful as a deterministic baseline for comparisons.
MP_V1_FORCE_HEURISTIC = os.environ.get("MP_V1_FORCE_HEURISTIC", "false").lower() in ("1", "true", "yes")

# MP v1 reward weights (defaults tuned to prioritize delivery success).
# - alpha: emphasize delivery success
# - beta/gamma: mild pressure on overhead/energy so solutions remain realistic
MP_V1_ALPHA_SUCCESS = float(os.environ.get("MP_V1_ALPHA_SUCCESS", "4.0"))
MP_V1_BETA_OVERHEAD = float(os.environ.get("MP_V1_BETA_OVERHEAD", "0.1"))
MP_V1_GAMMA_ENERGY = float(os.environ.get("MP_V1_GAMMA_ENERGY", "0.5"))
MP_V1_DELTA_DROP = float(os.environ.get("MP_V1_DELTA_DROP", "0.0"))

# Normalize energy consumption rate (energy/sec) into [0,1] using this scale.
# You will likely want to tune this per-scenario/energy model.
MP_V1_ENERGY_RATE_NORM = float(os.environ.get("MP_V1_ENERGY_RATE_NORM", "2.5"))
if not math.isfinite(MP_V1_ENERGY_RATE_NORM) or MP_V1_ENERGY_RATE_NORM <= 1e-12:
    MP_V1_ENERGY_RATE_NORM = 2.5
MP_V1_ENERGY_RATE_MODE = str(os.environ.get("MP_V1_ENERGY_RATE_MODE", "avg_per_host") or "").strip().lower()
if MP_V1_ENERGY_RATE_MODE not in ("avg", "avg_per_host", "total"):
    MP_V1_ENERGY_RATE_MODE = "avg_per_host"

# Print per-step MP v1 reward diagnostics (useful for setting MP_V1_ENERGY_RATE_NORM).
MP_V1_REWARD_DEBUG_PRINT = os.environ.get("MP_V1_REWARD_DEBUG_PRINT", "false").lower() in ("1", "true", "yes")

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
DELIVERY_EMA_DENSE_WEIGHT = float(os.environ.get("DELIVERY_EMA_DENSE_WEIGHT", "0.3"))
DELIVERY_EMA_ALPHA = float(os.environ.get("DELIVERY_EMA_ALPHA", "0.05"))
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
# Action distribution configuration (gaussian | beta)
ACTION_DIST = os.environ.get("ACTION_DIST", "beta").strip().lower()
if ACTION_DIST not in ("gaussian", "beta"):
    ACTION_DIST = "gaussian"
BETA_MIN_PARAM = float(os.environ.get("BETA_MIN_PARAM", "1.5"))
if BETA_MIN_PARAM <= 0.0:
    BETA_MIN_PARAM = 0.1
BETA_EPS = float(os.environ.get("BETA_EPS", "1e-6"))
if BETA_EPS <= 0.0:
    BETA_EPS = 1e-6

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
    preset = os.environ.get("LOG_STD_INIT_PRESET", "negative").strip().lower()
    if preset in ("negative", "neg", "low", "-1", "-1.0"):
        LOG_STD_INIT = -1.0
    else:
        LOG_STD_INIT = 0.0
_log_std_max_raw = os.environ.get("LOG_STD_MAX", "2.0").strip()
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
SPARSE_SUCCESS_REWARD = os.environ.get("SPARSE_SUCCESS_REWARD", "false").lower() in ("1", "true", "yes")
# Default off: combo 리워드를 주 보상으로 사용
SR_SPARSE_WEIGHT = float(os.environ.get("SR_SPARSE_WEIGHT", "0"))
# If true, divide the sparse reward by the number of active hosts to keep total bounded
SR_SPARSE_NORMALIZE = os.environ.get("SR_SPARSE_NORMALIZE", "false").lower() in ("1", "true", "yes")
# Cooldown disabled: keep zero regardless of env to avoid throttling sparse rewards
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

# ===== Success constraint (Lagrangian) controls =====
SR_CONSTRAINT_ALPHA = float(os.environ.get("SR_CONSTRAINT_ALPHA", "0.30"))
if SR_CONSTRAINT_ALPHA < 0.0:
    SR_CONSTRAINT_ALPHA = 0.0
elif SR_CONSTRAINT_ALPHA > 1.0:
    SR_CONSTRAINT_ALPHA = 1.0
SR_CONSTRAINT_TARGET_SCALE = float(os.environ.get("SR_CONSTRAINT_TARGET_SCALE", "1"))
if SR_CONSTRAINT_TARGET_SCALE < 0.0:
    SR_CONSTRAINT_TARGET_SCALE = 0.0
elif SR_CONSTRAINT_TARGET_SCALE > 1.0:
    SR_CONSTRAINT_TARGET_SCALE = 1.0
SR_CONSTRAINT_BASE_TARGET = float(os.environ.get("SR_CONSTRAINT_BASE_TARGET", "0.95"))
SR_CONSTRAINT_LAMBDA_INIT = float(os.environ.get("SR_CONSTRAINT_LAMBDA_INIT", "0.2"))
if SR_CONSTRAINT_LAMBDA_INIT < 0.0:
    SR_CONSTRAINT_LAMBDA_INIT = 0.0
SR_CONSTRAINT_LAMBDA_LR = float(os.environ.get("SR_CONSTRAINT_LAMBDA_LR", "0.10"))
if SR_CONSTRAINT_LAMBDA_LR < 0.0:
    SR_CONSTRAINT_LAMBDA_LR = 0.0
SR_CONSTRAINT_LAMBDA_MAX = float(os.environ.get("SR_CONSTRAINT_LAMBDA_MAX", "40.0"))
if SR_CONSTRAINT_LAMBDA_MAX < 0.0:
    SR_CONSTRAINT_LAMBDA_MAX = 0.0
SR_CONSTRAINT_DENOM = float(os.environ.get("SR_CONSTRAINT_DENOM", "500.0"))
if not math.isfinite(SR_CONSTRAINT_DENOM) or SR_CONSTRAINT_DENOM <= 0.0:
    SR_CONSTRAINT_DENOM = float(EPISODE_SECONDS) if EPISODE_SECONDS > 0 else 1.0
SR_CONSTRAINT_MIN_DENOM = float(os.environ.get("SR_CONSTRAINT_MIN_DENOM", "1.0"))
if SR_CONSTRAINT_MIN_DENOM <= 0.0:
    SR_CONSTRAINT_MIN_DENOM = 1.0
SR_CONSTRAINT_MAX_DENOM = float(os.environ.get("SR_CONSTRAINT_MAX_DENOM", "2000.0"))
if SR_CONSTRAINT_MAX_DENOM < SR_CONSTRAINT_MIN_DENOM:
    SR_CONSTRAINT_MAX_DENOM = SR_CONSTRAINT_MIN_DENOM
SR_CONSTRAINT_USE_CREATED = os.environ.get("SR_CONSTRAINT_USE_CREATED", "true").lower() in ("1", "true", "yes")
SR_CONSTRAINT_SQRT_CREATED = os.environ.get("SR_CONSTRAINT_SQRT_CREATED", "true").lower() in ("1", "true", "yes")
SR_CONSTRAINT_PENALTY_GAIN = float(os.environ.get("SR_CONSTRAINT_PENALTY_GAIN", "1000.0"))
if SR_CONSTRAINT_PENALTY_GAIN < 0.0:
    SR_CONSTRAINT_PENALTY_GAIN = 0.0
SR_CONSTRAINT_COUNT_WEIGHT = float(os.environ.get("SR_CONSTRAINT_COUNT_WEIGHT", "0"))
if SR_CONSTRAINT_COUNT_WEIGHT < 0.0:
    SR_CONSTRAINT_COUNT_WEIGHT = 0.0

# ===== PID-Lagrangian controller =====
PID_KP = float(os.environ.get("PID_KP", "0.05"))
PID_KI = float(os.environ.get("PID_KI", "0.01"))
PID_KD = float(os.environ.get("PID_KD", "0.005"))
PID_INTEGRAL_MAX = float(os.environ.get("PID_INTEGRAL_MAX", "10.0"))
LAMBDA_MIN = float(os.environ.get("LAMBDA_MIN", "0.3"))
if LAMBDA_MIN < 0.0:
    LAMBDA_MIN = 0.0
PID_RELAX_FACTOR = float(os.environ.get("PID_RELAX_FACTOR", "0.3"))
if PID_RELAX_FACTOR < 0.0:
    PID_RELAX_FACTOR = 0.0
elif PID_RELAX_FACTOR > 1.0:
    PID_RELAX_FACTOR = 1.0
PID_NEG_INTEGRAL_DECAY = float(os.environ.get("PID_NEG_INTEGRAL_DECAY", "0.95"))
if PID_NEG_INTEGRAL_DECAY < 0.0:
    PID_NEG_INTEGRAL_DECAY = 0.0
elif PID_NEG_INTEGRAL_DECAY > 1.0:
    PID_NEG_INTEGRAL_DECAY = 1.0

# ===== Adaptive baseline target =====
ADAPTIVE_TARGET = os.environ.get("ADAPTIVE_TARGET", "true").lower() in ("1", "true", "yes")
ADAPTIVE_WARMUP_EPISODES = int(os.environ.get("ADAPTIVE_WARMUP_EPISODES", "3"))
ADAPTIVE_SLOW_ALPHA = float(os.environ.get("ADAPTIVE_SLOW_ALPHA", "0.05"))
ADAPTIVE_FAST_ALPHA = float(os.environ.get("ADAPTIVE_FAST_ALPHA", "0.30"))

# ===== Loss-level Lagrangian =====
LOSS_LAGRANGIAN = os.environ.get("LOSS_LAGRANGIAN", "false").lower() in ("1", "true", "yes")

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


def _parse_episode_schedule_env(raw_value):
    """
    Parse schedules expressed as "start:end:value" triples separated by commas.
    Returns a sorted list of (start_ep, end_ep, value) tuples.
    """
    schedule = []
    if not raw_value:
        return schedule
    parts = [p.strip() for p in raw_value.split(",") if p.strip()]
    for part in parts:
        tokens = [t.strip() for t in part.split(":")]
        if len(tokens) != 3:
            continue
        try:
            start_ep = int(float(tokens[0]))
            end_ep = int(float(tokens[1]))
            value = float(tokens[2])
            if end_ep < start_ep:
                start_ep, end_ep = end_ep, start_ep
            schedule.append((start_ep, end_ep, value))
        except Exception:
            continue
    schedule.sort(key=lambda item: (item[0], item[1]))
    return schedule


# ===== Meta-Agent (LR/Entropy Controller) defaults =====
META_AGENT_ENABLED = os.environ.get("META_AGENT_ENABLED", "false").lower() in ("1", "true", "yes")
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


def log_step_rewards(
    sim_time,
    rewards_per_host,
    buffer_size_mb=None,
    episode_num=1,
    total_reward_override=None,
    num_hosts_override=None,
):
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

            if total_reward_override is not None:
                try:
                    total_reward = float(total_reward_override)
                except Exception:
                    total_reward = 0.0
            else:
                total_reward = sum(rewards_per_host.values()) if rewards_per_host else 0.0

            if num_hosts_override is not None:
                try:
                    num_hosts = int(num_hosts_override)
                except Exception:
                    num_hosts = len(rewards_per_host) if rewards_per_host else 0
            else:
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


def log_mp_v1_reward_parameters(episode_num, sim_time, sim_id=None, buffer_size_mb=None, algo_tag="rmappo"):
    """Record MaxProp++ v1 (parameter-control) reward parameters once per episode."""
    try:
        if not sim_id:
            sim_id = ""
        algo_tag = str(algo_tag or "rmappo").strip().lower()
        base_dir = REPORT_OUTPUT_DIR if 'REPORT_OUTPUT_DIR' in globals() else "reports"
        log_dir = os.path.join(base_dir, "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_files = []
        if buffer_size_mb:
            log_files.append(os.path.join(log_dir, f"reward_parameters_mp_v1_buf{int(buffer_size_mb)}M_{algo_tag}.csv"))
        else:
            safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(sim_id))
            if safe_sim:
                log_files.append(os.path.join(log_dir, f"reward_parameters_mp_v1_{safe_sim}_{algo_tag}.csv"))
            log_files.append(os.path.join(log_dir, f"reward_parameters_mp_v1_{algo_tag}.csv"))

        cols = [
            "episode", "sim_time", "sim_id",
            "MP_V1_REWARD_ENABLE", "MP_V1_REWARD_FORM",
            "MP_V1_REWARD_SCALE", "MP_V1_REWARD_EPS",
            "MP_V1_ALPHA_SUCCESS", "MP_V1_BETA_OVERHEAD", "MP_V1_GAMMA_ENERGY", "MP_V1_DELTA_DROP",
            "MP_V1_ENERGY_RATE_NORM", "MP_V1_ENERGY_RATE_MODE",
        ]
        row = {
            "episode": episode_num,
            "sim_time": sim_time,
            "sim_id": sim_id,
            "MP_V1_REWARD_ENABLE": int(1 if MP_V1_REWARD_ENABLE else 0),
            "MP_V1_REWARD_FORM": str(MP_V1_REWARD_FORM),
            "MP_V1_REWARD_SCALE": float(MP_V1_REWARD_SCALE),
            "MP_V1_REWARD_EPS": float(MP_V1_REWARD_EPS),
            "MP_V1_ALPHA_SUCCESS": float(MP_V1_ALPHA_SUCCESS),
            "MP_V1_BETA_OVERHEAD": float(MP_V1_BETA_OVERHEAD),
            "MP_V1_GAMMA_ENERGY": float(MP_V1_GAMMA_ENERGY),
            "MP_V1_DELTA_DROP": float(MP_V1_DELTA_DROP),
            "MP_V1_ENERGY_RATE_NORM": float(MP_V1_ENERGY_RATE_NORM),
            "MP_V1_ENERGY_RATE_MODE": str(MP_V1_ENERGY_RATE_MODE),
        }

        for log_file in log_files:
            # Backward-compatible: if an older header exists, write to a new *_v2.csv
            try:
                if os.path.exists(log_file):
                    with open(log_file, "r", encoding="utf-8") as rf:
                        first = (rf.readline() or "").strip()
                    if first and "MP_V1_ENERGY_RATE_MODE" not in first:
                        if log_file.lower().endswith(".csv"):
                            log_file = log_file[:-4] + "_v2.csv"
            except Exception:
                pass
            write_header = not os.path.exists(log_file)
            with open(log_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(",".join(cols) + "\n")
                f.write(",".join(str(row[c]) for c in cols) + "\n")
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
        totals["constraint_penalty"] += components.get("constraint_penalty", 0.0)
        totals["combo_reward"] += components.get("combo_reward", 0.0)
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
            "delivery_delay_pen","overhead_penalty","constraint_penalty","combo_reward","total_reward"
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
            "constraint_penalty": 0.0,
            "combo_reward": 0.0,
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
                "constraint_penalty": totals.get("constraint_penalty", 0.0),
                "combo_reward": totals.get("combo_reward", 0.0),
                "total_reward": totals.get("total_reward", 0.0),
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")
    except Exception:
        pass


class PIDLambdaController:
    """PID controller for Lagrangian multiplier updates with anti-windup."""

    def __init__(self, kp, ki, kd, lambda_init, lambda_max, integral_max=10.0,
                 lambda_min=0.0, relax_factor=1.0, neg_integral_decay=1.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.lambda_val = lambda_init
        self.lambda_max = lambda_max
        self.integral = 0.0
        self.integral_max = integral_max
        self.lambda_min = lambda_min
        self.relax_factor = relax_factor
        self.neg_integral_decay = neg_integral_decay
        self.prev_error = 0.0

    def step(self, error):
        """Update lambda given error (positive = constraint violated, need more lambda)."""
        # Asymmetric gains: relax when delivery is above baseline (error < 0)
        gain_scale = self.relax_factor if error < 0.0 else 1.0
        # Proportional (scaled)
        p_term = gain_scale * self.kp * error
        # Integral with anti-windup (accumulation also scaled when relaxing)
        self.integral += gain_scale * error
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        # Decay negative integral to prevent long-term lambda suppression
        if self.integral < 0.0 and self.neg_integral_decay < 1.0:
            self.integral *= self.neg_integral_decay
        i_term = self.ki * self.integral
        # Derivative (backward difference, unscaled -- damper regardless of sign)
        d_term = self.kd * (error - self.prev_error)
        self.prev_error = error
        # Update with floor and ceiling
        self.lambda_val = max(self.lambda_min, self.lambda_val + p_term + i_term + d_term)
        if self.lambda_max > 0:
            self.lambda_val = min(self.lambda_val, self.lambda_max)
        return self.lambda_val

    def state_dict(self):
        return {
            "lambda_val": self.lambda_val,
            "integral": self.integral,
            "prev_error": self.prev_error,
        }

    def load_state_dict(self, d):
        self.lambda_val = d.get("lambda_val", self.lambda_val)
        self.integral = d.get("integral", self.integral)
        self.prev_error = d.get("prev_error", self.prev_error)


def update_success_constraint_from_episode(episode_num):
    """Update success constraint tracking using the finalized episode totals.

    When ADAPTIVE_TARGET is enabled, uses dual-EMA approach:
    - Slow EMA tracks the achievable delivery rate (adaptive baseline)
    - Fast EMA tracks current performance
    - PID controller adjusts lambda based on (slow - fast) error
    This is environment-agnostic: no hardcoded delivery rate target needed.

    When ADAPTIVE_TARGET is disabled, falls back to the original
    success-metric-based approach with PID lambda updates.
    """
    try:
        totals = Handler.CORE_EP_TOTALS.get(episode_num)
        if not totals:
            return

        created_total = getattr(Handler, "EP_CREATED_SUM", 0.0) or 0.0
        if not math.isfinite(created_total) or created_total < 0.0:
            created_total = 0.0
        Handler.CONSTRAINT_LAST_CREATED = created_total

        # ---- Adaptive target path (environment-agnostic) ----
        if ADAPTIVE_TARGET:
            delivered_total = getattr(Handler, "EP_DELIVERED_SUM", 0.0) or 0.0
            delivery_rate = delivered_total / max(1.0, created_total)
            Handler.CONSTRAINT_LAST_SUCCESS = delivery_rate

            ep_count = getattr(Handler, "EP_COUNT", 1)

            # Warmup: collect baseline delivery rates, no constraint
            if ep_count <= ADAPTIVE_WARMUP_EPISODES:
                Handler.ADAPTIVE_WARMUP_RATES.append(delivery_rate)
                # Track best even during warmup
                Handler.BEST_DELIVERY_RATE = max(Handler.BEST_DELIVERY_RATE, delivery_rate)
                if len(Handler.ADAPTIVE_WARMUP_RATES) >= ADAPTIVE_WARMUP_EPISODES:
                    baseline = sum(Handler.ADAPTIVE_WARMUP_RATES) / len(Handler.ADAPTIVE_WARMUP_RATES)
                    Handler.DELIVERY_RATE_SLOW_EMA = baseline
                    Handler.DELIVERY_RATE_FAST_EMA = baseline
                Handler.CONSTRAINT_STEP_PENALTY = 0.0
                print(
                    "[CONSTRAINT-ADAPTIVE] ep={ep} warmup delivery_rate={dr:.4f} "
                    "collected={n}/{total}"
                    .format(ep=episode_num, dr=delivery_rate,
                            n=len(Handler.ADAPTIVE_WARMUP_RATES),
                            total=ADAPTIVE_WARMUP_EPISODES)
                )
                return

            # Guard: ensure EMAs are initialized
            if Handler.DELIVERY_RATE_SLOW_EMA is None:
                Handler.DELIVERY_RATE_SLOW_EMA = delivery_rate
                Handler.DELIVERY_RATE_FAST_EMA = delivery_rate

            # Update fast EMA (tracks current performance)
            Handler.DELIVERY_RATE_FAST_EMA = (
                (1.0 - ADAPTIVE_FAST_ALPHA) * Handler.DELIVERY_RATE_FAST_EMA
                + ADAPTIVE_FAST_ALPHA * delivery_rate
            )

            # Best delivery rate: strict max, never decreases
            # Same principle as copy 3's SUCCESS_PROXY_BEST — no hardcoded target
            Handler.BEST_DELIVERY_RATE = max(
                Handler.BEST_DELIVERY_RATE, delivery_rate
            )

            # Slow EMA still updated for logging, but NOT used as target
            Handler.DELIVERY_RATE_SLOW_EMA = (
                (1.0 - ADAPTIVE_SLOW_ALPHA) * Handler.DELIVERY_RATE_SLOW_EMA
                + ADAPTIVE_SLOW_ALPHA * delivery_rate
            )

            # Target: ratcheted best (like copy 3's fixed target, but adaptive)
            # Scale by SR_CONSTRAINT_TARGET_SCALE (default 1.0) for tuning
            adaptive_target = SR_CONSTRAINT_TARGET_SCALE * Handler.BEST_DELIVERY_RATE

            # Error: target - current delivery rate
            # Positive = delivery below best achievable -> increase lambda
            # Negative = delivery at/above best -> relax lambda
            error = adaptive_target - delivery_rate
            deficit = max(0.0, error)
            Handler.SUCCESS_CONSTRAINT_DEFICIT = deficit

            # ---- Overhead-based adaptive lambda floor ----
            relayed_total = getattr(Handler, "EP_RELAYED_SUM", 0.0) or 0.0
            delivered_total_oh = getattr(Handler, "EP_DELIVERED_SUM", 0.0) or 0.0
            if delivered_total_oh > 0:
                overhead_ratio = (relayed_total - delivered_total_oh) / delivered_total_oh
            else:
                overhead_ratio = 0.0

            if Handler.OVERHEAD_EMA is None:
                Handler.OVERHEAD_EMA = overhead_ratio
                Handler.OVERHEAD_PEAK = overhead_ratio
            else:
                Handler.OVERHEAD_EMA = 0.8 * Handler.OVERHEAD_EMA + 0.2 * overhead_ratio

            if Handler.OVERHEAD_PEAK is None or overhead_ratio > Handler.OVERHEAD_PEAK:
                Handler.OVERHEAD_PEAK = overhead_ratio

            if Handler.OVERHEAD_PEAK > 0:
                reduction_ratio = max(0.0, min(1.0, 1.0 - Handler.OVERHEAD_EMA / Handler.OVERHEAD_PEAK))
                effective_min = LAMBDA_MIN * reduction_ratio
            else:
                effective_min = 0.0

            if Handler.PID_CONTROLLER is not None:
                Handler.PID_CONTROLLER.lambda_min = effective_min

            # PID update
            if Handler.PID_CONTROLLER is not None:
                new_lambda = Handler.PID_CONTROLLER.step(error)
            else:
                new_lambda = max(effective_min, Handler.CONSTRAINT_LAMBDA + SR_CONSTRAINT_LAMBDA_LR * error)
                if SR_CONSTRAINT_LAMBDA_MAX > 0.0:
                    new_lambda = min(new_lambda, SR_CONSTRAINT_LAMBDA_MAX)
            Handler.CONSTRAINT_LAMBDA = new_lambda

            # Step penalty for reward-shaping fallback (used when LOSS_LAGRANGIAN=false)
            Handler.CONSTRAINT_STEP_PENALTY = (
                Handler.CONSTRAINT_LAMBDA * deficit * SR_CONSTRAINT_PENALTY_GAIN
            )

            print(
                "[CONSTRAINT-ADAPTIVE] ep={ep} delivery_rate={dr:.4f} "
                "best={best:.4f} target={tgt:.4f} error={err:.4f} deficit={def_:.4f} "
                "lambda={lam:.4f} integral={integ:.4f} "
                "overhead_ema={oh:.2f} lambda_min_eff={lmin:.4f} created={created:.1f}"
                .format(
                    ep=episode_num, dr=delivery_rate,
                    best=Handler.BEST_DELIVERY_RATE,
                    tgt=adaptive_target,
                    err=error, def_=deficit,
                    lam=Handler.CONSTRAINT_LAMBDA,
                    integ=Handler.PID_CONTROLLER.integral if Handler.PID_CONTROLLER else 0.0,
                    oh=Handler.OVERHEAD_EMA if Handler.OVERHEAD_EMA is not None else 0.0,
                    lmin=effective_min,
                    created=created_total,
                )
            )
            return

        # ---- Legacy path (fixed target, PID lambda updates) ----
        sr_total = float(totals.get("sr_sparse_bonus", 0.0) or 0.0)
        sr_count = float(totals.get("sr_sparse_count", 0.0) or 0.0)
        if SR_CONSTRAINT_USE_CREATED and created_total > 0.0:
            base_denom = created_total
            if SR_CONSTRAINT_SQRT_CREATED and base_denom > 1.0:
                base_denom = math.sqrt(base_denom)
            denom = max(SR_CONSTRAINT_MIN_DENOM, min(base_denom, SR_CONSTRAINT_MAX_DENOM))
        else:
            denom = max(SR_CONSTRAINT_DENOM, SR_CONSTRAINT_MIN_DENOM)
        Handler.CONSTRAINT_LAST_DENOM = denom
        weighted_bonus = sr_total + SR_CONSTRAINT_COUNT_WEIGHT * sr_count
        success_metric = weighted_bonus / denom if denom > 0.0 else 0.0
        Handler.CONSTRAINT_LAST_SUCCESS = success_metric
        prev_ema = Handler.SUCCESS_PROXY_EMA
        if prev_ema is None or SR_CONSTRAINT_ALPHA <= 0.0:
            Handler.SUCCESS_PROXY_EMA = success_metric
        else:
            Handler.SUCCESS_PROXY_EMA = (
                (1.0 - SR_CONSTRAINT_ALPHA) * prev_ema + SR_CONSTRAINT_ALPHA * success_metric
            )
        Handler.SUCCESS_PROXY_BEST = max(Handler.SUCCESS_PROXY_BEST, success_metric)
        dynamic_target = SR_CONSTRAINT_TARGET_SCALE * Handler.SUCCESS_PROXY_BEST
        Handler.SUCCESS_PROXY_TARGET = max(SR_CONSTRAINT_BASE_TARGET, dynamic_target)
        deficit = max(0.0, Handler.SUCCESS_PROXY_TARGET - Handler.SUCCESS_PROXY_EMA)
        Handler.SUCCESS_CONSTRAINT_DEFICIT = deficit

        # ---- Overhead-based adaptive lambda floor (legacy path) ----
        relayed_total = getattr(Handler, "EP_RELAYED_SUM", 0.0) or 0.0
        delivered_total_oh = getattr(Handler, "EP_DELIVERED_SUM", 0.0) or 0.0
        if delivered_total_oh > 0:
            overhead_ratio = (relayed_total - delivered_total_oh) / delivered_total_oh
        else:
            overhead_ratio = 0.0

        if Handler.OVERHEAD_EMA is None:
            Handler.OVERHEAD_EMA = overhead_ratio
            Handler.OVERHEAD_PEAK = overhead_ratio
        else:
            Handler.OVERHEAD_EMA = 0.8 * Handler.OVERHEAD_EMA + 0.2 * overhead_ratio

        if Handler.OVERHEAD_PEAK is None or overhead_ratio > Handler.OVERHEAD_PEAK:
            Handler.OVERHEAD_PEAK = overhead_ratio

        if Handler.OVERHEAD_PEAK > 0:
            reduction_ratio = max(0.0, min(1.0, 1.0 - Handler.OVERHEAD_EMA / Handler.OVERHEAD_PEAK))
            effective_min = LAMBDA_MIN * reduction_ratio
        else:
            effective_min = 0.0

        if Handler.PID_CONTROLLER is not None:
            Handler.PID_CONTROLLER.lambda_min = effective_min

        # PID lambda update (replaces simple gradient ascent)
        error = Handler.SUCCESS_PROXY_TARGET - Handler.SUCCESS_PROXY_EMA
        if Handler.PID_CONTROLLER is not None:
            new_lambda = Handler.PID_CONTROLLER.step(error)
        else:
            new_lambda = max(effective_min, Handler.CONSTRAINT_LAMBDA + SR_CONSTRAINT_LAMBDA_LR * error)
            if SR_CONSTRAINT_LAMBDA_MAX > 0.0:
                new_lambda = min(new_lambda, SR_CONSTRAINT_LAMBDA_MAX)
        Handler.CONSTRAINT_LAMBDA = new_lambda

        if SR_CONSTRAINT_PENALTY_GAIN > 0.0 and denom > 0.0:
            Handler.CONSTRAINT_STEP_PENALTY = (
                Handler.CONSTRAINT_LAMBDA * deficit * SR_CONSTRAINT_PENALTY_GAIN
            ) / denom
        else:
            Handler.CONSTRAINT_STEP_PENALTY = 0.0
        print(
            "[CONSTRAINT] ep={ep} metric={metric:.4f} ema={ema:.4f} "
            "target={target:.4f} deficit={deficit:.4f} lambda={lam:.4f} "
            "denom={denom:.1f} created={created:.1f}"
            .format(
                ep=episode_num,
                metric=success_metric,
                ema=Handler.SUCCESS_PROXY_EMA,
                target=Handler.SUCCESS_PROXY_TARGET,
                deficit=deficit,
                lam=Handler.CONSTRAINT_LAMBDA,
                denom=denom,
                created=created_total,
            )
        )
    except Exception as exc:
        import traceback
        print(f"[WARN] Failed to update success constraint: {exc}")
        traceback.print_exc()


def log_episode_reward(episode_num, sim_id, total_reward, avg_delivery_rate=None,
                       delivered=None, created=None, avg_overhead=None,
                       avg_delay=None, buffer_size_mb=None, combo_reward=None,
                       drop_rate=None, avg_energy_used=None, algo_tag="rmappo"):
    """Log per-episode total reward (and optional delivery summary) to CSV.

    Columns: episode, sim_id, total_reward, combo_reward, avg_delivery_rate, delivered,
             created, avg_overhead, avg_delay, drop_rate, avg_energy_used
    Writes to reports/reward_logs/episode_rewards[_buf{MB}M]_{algo_tag}.csv
    """
    try:
        algo_tag = str(algo_tag or "rmappo").strip().lower()
        base_dir = REPORT_OUTPUT_DIR if 'REPORT_OUTPUT_DIR' in globals() else "reports"
        log_dir = os.path.join(base_dir, "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_files = []
        if buffer_size_mb:
            log_files.append(os.path.join(log_dir, f"episode_rewards_buf{int(buffer_size_mb)}M_{algo_tag}.csv"))
        else:
            safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(sim_id))
            if safe_sim:
                log_files.append(os.path.join(log_dir, f"episode_rewards_{safe_sim}_{algo_tag}.csv"))
            log_files.append(os.path.join(log_dir, f"episode_rewards_{algo_tag}.csv"))

        header = (
            "episode,sim_id,total_reward,combo_reward,avg_delivery_rate,delivered,created,"
            "avg_overhead,avg_delay,drop_rate,avg_energy_used\n"
        )

        def _ensure_schema(path, expected_header):
            try:
                if not os.path.exists(path):
                    return
                with open(path, "r", encoding="utf-8") as rf:
                    first = rf.readline()
                    if first == expected_header:
                        return
                    rest = rf.read().splitlines()
                expected_n = len(expected_header.strip().split(","))
                out_lines = [expected_header.rstrip("\n")]
                for line in rest:
                    if not line:
                        continue
                    cols = line.split(",")
                    if len(cols) < expected_n:
                        cols = cols + ([""] * (expected_n - len(cols)))
                    out_lines.append(",".join(cols))
                with open(path, "w", encoding="utf-8") as wf:
                    wf.write("\n".join(out_lines) + "\n")
            except Exception:
                return

        # Normalize None to empty string for optional fields
        avg_str = "" if avg_delivery_rate is None or (isinstance(avg_delivery_rate, float) and math.isnan(avg_delivery_rate)) else str(avg_delivery_rate)
        deliv_str = "" if delivered is None else str(delivered)
        created_str = "" if created is None else str(created)
        overhead_str = "" if avg_overhead is None or (isinstance(avg_overhead, float) and math.isnan(avg_overhead)) else str(avg_overhead)
        delay_str = "" if avg_delay is None or (isinstance(avg_delay, float) and math.isnan(avg_delay)) else str(avg_delay)
        combo_str = "" if combo_reward is None else str(combo_reward)
        drop_str = "" if drop_rate is None or (isinstance(drop_rate, float) and math.isnan(drop_rate)) else str(drop_rate)
        energy_str = "" if avg_energy_used is None or (isinstance(avg_energy_used, float) and math.isnan(avg_energy_used)) else str(avg_energy_used)
        row = f"{episode_num},{sim_id},{total_reward},{combo_str},{avg_str},{deliv_str},{created_str},{overhead_str},{delay_str},{drop_str},{energy_str}\n"

        for log_file in log_files:
            _ensure_schema(log_file, header)
            write_header = not os.path.exists(log_file)
            with open(log_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header)
                f.write(row)
    except Exception:
        pass

def log_episode_losses(episode_num, sim_id, actor_loss_avg=None, critic_loss_avg=None,
                       cost_critic_loss_avg=None, updates=0, entropy_avg=None, buffer_size_mb=None, algo_tag="rmappo"):
    """Log per-episode average actor/critic/cost_critic losses to CSV.

    Columns: episode, sim_id, actor_loss_avg, critic_loss_avg, cost_critic_loss_avg, updates, entropy_avg
    Writes to reports/train_logs/episode_losses[_buf{MB}M]_{algo_tag}.csv
    """
    try:
        algo_tag = str(algo_tag or "rmappo").strip().lower()
        base_dir = REPORT_OUTPUT_DIR if 'REPORT_OUTPUT_DIR' in globals() else "reports"
        log_dir = os.path.join(base_dir, "train_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_files = []
        if buffer_size_mb:
            log_files.append(os.path.join(log_dir, f"episode_losses_buf{int(buffer_size_mb)}M_{algo_tag}.csv"))
        else:
            safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(sim_id))
            if safe_sim:
                log_files.append(os.path.join(log_dir, f"episode_losses_{safe_sim}_{algo_tag}.csv"))
            log_files.append(os.path.join(log_dir, f"episode_losses_{algo_tag}.csv"))

        def _fmt(x):
            try:
                return "" if x is None else str(round(float(x), 6))
            except Exception:
                return ""

        header = "episode,sim_id,actor_loss_avg,critic_loss_avg,cost_critic_loss_avg,updates,entropy_avg\n"
        row = (
            f"{episode_num},{sim_id},{_fmt(actor_loss_avg)},{_fmt(critic_loss_avg)},"
            f"{_fmt(cost_critic_loss_avg)},{int(updates)},{_fmt(entropy_avg)}\n"
        )

        for log_file in log_files:
            write_header = not os.path.exists(log_file)
            with open(log_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header)
                f.write(row)
    except Exception:
        pass

def log_update_metrics(sim_time, metrics: dict, buffer_size_mb=None, episode_num=1, algo_tag="rmappo"):
    """Append per-update diagnostics to CSV for analysis."""
    try:
        algo_tag = str(algo_tag or "rmappo").strip().lower()
        base_dir = REPORT_OUTPUT_DIR if 'REPORT_OUTPUT_DIR' in globals() else "reports"
        log_dir = os.path.join(base_dir, "train_logs")
        os.makedirs(log_dir, exist_ok=True)

        log_files = []
        if buffer_size_mb:
            log_files.append(os.path.join(log_dir, f"update_metrics_buf{int(buffer_size_mb)}M_{algo_tag}.csv"))
        else:
            raw_sim = metrics.get("sim_id", "") if isinstance(metrics, dict) else ""
            safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(raw_sim))
            if not safe_sim:
                safe_sim = _sanitize_sim_id_for_filename(_canonical_sim_id_for_group_logs(os.environ.get("SIM_ID", "")))
            if safe_sim:
                log_files.append(os.path.join(log_dir, f"update_metrics_{safe_sim}_{algo_tag}.csv"))
            log_files.append(os.path.join(log_dir, f"update_metrics_{algo_tag}.csv"))

        if algo_tag == "masac":
            cols = ["episode", "sim_time", "policy_loss", "value_loss", "alpha", "entropy"]
            row = {
                "episode": episode_num,
                "sim_time": sim_time,
                "policy_loss": metrics.get("policy_loss", ""),
                "value_loss": metrics.get("value_loss", ""),
                "alpha": metrics.get("alpha", ""),
                "entropy": metrics.get("entropy", ""),
            }
        else:
            cols = ["episode", "sim_time", "ratio_mean", "ratio_std", "kl", "std_mean", "adv_var", "cost_value_loss"]
            row = {
                "episode": episode_num,
                "sim_time": sim_time,
                "ratio_mean": metrics.get("ratio_mean", 0.0),
                "ratio_std": metrics.get("ratio_std", 0.0),
                "kl": metrics.get("kl", 0.0),
                "std_mean": metrics.get("std_mean", 0.0),
                "adv_var": metrics.get("adv_var", 0.0),
                "cost_value_loss": metrics.get("cost_value_loss", 0.0),
            }

        header = ",".join(cols) + "\n"
        line = ",".join(str(row.get(c, "")) for c in cols) + "\n"

        for log_file in log_files:
            write_header = not os.path.exists(log_file)
            with open(log_file, "a", encoding="utf-8") as f:
                if write_header:
                    f.write(header)
                f.write(line)
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
            "mid_util_penalty","trend_penalty","constraint_penalty","relay_bonus",
            "combo_reward",
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
                "constraint_penalty": components.get("constraint_penalty", 0.0),
                "relay_bonus": components.get("relay_bonus", 0.0),
                "combo_reward": components.get("combo_reward", 0.0),
                "sparse_delivery": components.get("sparse_delivery", 0.0),
                "target_bonus": components.get("target_bonus", 0.0),
                "total_hosts_reward": components.get("total_hosts_reward", 0.0),
            }
            f.write(",".join(str(row[c]) for c in cols) + "\n")

        # Log a core file with only the four active reward parameters for quick inspection
        core_cols = [
            "episode","sim_time",
            "success_penalty","success_bonus","skip_low_pred_pen","peer_high_pred_pen","drop_ttl_penalty",
            "sr_sparse_bonus","sr_sparse_count","relay_bonus","state_bonus","state_delta_bonus",
            "constraint_penalty","combo_reward","total_reward"
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
                "constraint_penalty": components.get("constraint_penalty", 0.0),
                "combo_reward": components.get("combo_reward", 0.0),
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
            "mid_util_penalty","trend_penalty","constraint_penalty","relay_bonus","combo_reward",
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
                "constraint_penalty": components.get("constraint_penalty", 0.0),
                "relay_bonus": components.get("relay_bonus", 0.0),
                "combo_reward": components.get("combo_reward", 0.0),
                "sparse_delivery": components.get("sparse_delivery", 0.0),
                "target_bonus": components.get("target_bonus", 0.0),
                "total_hosts_reward": components.get("total_hosts_reward", 0.0),
            }
            f2.write(",".join(str(row2[c]) for c in cols_det) + "\n")
    except Exception:
        pass


def log_mp_v1_action_episode_stats(episode_num, sim_id, stats, buffer_size_mb=None):
    """
    Episode-level aggregate action statistics for MP v1 (lambda, tau, beta, kx, m_relay).

    This is meant for diagnosis when performance seems insensitive to actions
    (e.g., actions stuck at bounds or m_relay always >= 0).
    """
    if not MP_V1_ACTION_LOG_ENABLE:
        return
    if not isinstance(stats, dict) or int(stats.get("n", 0) or 0) <= 0:
        return

    out_dir = os.path.join("reports", "action_logs")
    try:
        os.makedirs(out_dir, exist_ok=True)
    except Exception:
        pass

    if MP_V1_ACTION_LOG_PATH:
        path = MP_V1_ACTION_LOG_PATH
    else:
        safe_sim = (sim_id or "sim").replace(os.sep, "_").replace(":", "_")
        path = os.path.join(out_dir, f"mp_v1_action_episode_stats_{safe_sim}.csv")

    header = (
        "episode,sim_id,n,"
        "lam_mean,lam_std,lam_min_obs,lam_max_obs,lam_frac_min,lam_frac_max,"
        "tau_mean,tau_std,tau_min_obs,tau_max_obs,tau_frac_min,tau_frac_max,"
        "beta_mean,beta_std,beta_min_obs,beta_max_obs,beta_frac_min,beta_frac_max,"
        "kx_mean,kx_std,kx_min_obs,kx_max_obs,kx_frac_min,kx_frac_max,"
        "mr_mean,mr_std,mr_min_obs,mr_max_obs,mr_frac_min,mr_frac_max,mr_frac_ge0,"
        "buffer_size_mb\n"
    )

    def _g(k, default=""):
        v = stats.get(k, None)
        if v is None:
            return default
        try:
            if isinstance(v, float):
                if not math.isfinite(v):
                    return default
                return f"{v:.6g}"
            if isinstance(v, int):
                return str(v)
            # fall back
            return str(v)
        except Exception:
            return default

    try:
        need_header = (not os.path.isfile(path)) or os.path.getsize(path) == 0
    except Exception:
        need_header = True

    try:
        with open(path, "a", encoding="utf-8") as f:
            if need_header:
                f.write(header)
            f.write(
                f"{int(episode_num)},{sim_id},{int(stats.get('n', 0) or 0)},"
                f"{_g('lam_mean')},{_g('lam_std')},{_g('lam_min_obs')},{_g('lam_max_obs')},{_g('lam_frac_min')},{_g('lam_frac_max')},"
                f"{_g('tau_mean')},{_g('tau_std')},{_g('tau_min_obs')},{_g('tau_max_obs')},{_g('tau_frac_min')},{_g('tau_frac_max')},"
                f"{_g('beta_mean')},{_g('beta_std')},{_g('beta_min_obs')},{_g('beta_max_obs')},{_g('beta_frac_min')},{_g('beta_frac_max')},"
                f"{_g('kx_mean')},{_g('kx_std')},{_g('kx_min_obs')},{_g('kx_max_obs')},{_g('kx_frac_min')},{_g('kx_frac_max')},"
                f"{_g('mr_mean')},{_g('mr_std')},{_g('mr_min_obs')},{_g('mr_max_obs')},{_g('mr_frac_min')},{_g('mr_frac_max')},{_g('mr_frac_ge0')},"
                f"{'' if buffer_size_mb is None else float(buffer_size_mb)}\n"
            )
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
EPISODE_SECONDS = int(float(os.environ.get("EPISODE_SECONDS", "43200")))
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "2000"))
SIM_END_SECONDS = int(float(os.environ.get("SIM_END_SECONDS", "86400000")))

# Checkpoint control
MODEL_DIR = os.environ.get("MODEL_DIR", "models_rmappo")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
MP_V1_MODEL_PATH = os.environ.get("MP_V1_MODEL_PATH", "")
MP_V1_MODEL_DIR = os.environ.get("MP_V1_MODEL_DIR", "models_masac")
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() in ("1", "true", "yes")
NO_TRAINING_CAP = os.environ.get("NO_TRAINING_CAP", "false").lower() in ("1", "true", "yes")

# Stability/exploration schedules (env-overridable)
# Entropy: 0.01 → 0.001 over ep 1-150 (explore early, exploit later)
ENTROPY_COEF_START = float(os.environ.get("ENTROPY_COEF_START", "0.01"))
ENTROPY_COEF_END = float(os.environ.get("ENTROPY_COEF_END", "0.001"))
ENTROPY_DECAY_START_EP = int(os.environ.get("ENTROPY_DECAY_START_EP", "1"))
ENTROPY_DECAY_END_EP = int(os.environ.get("ENTROPY_DECAY_END_EP", "150"))
ENTROPY_SCHEDULE = _parse_episode_schedule_env(os.environ.get("ENTROPY_SCHEDULE", ""))
EPISODIC_UPDATES = os.environ.get("EPISODIC_UPDATES", "true").lower() in ("1", "true", "yes")
EPISODIC_TRAIN_INTERVAL_STEPS = int(float(os.environ.get("EPISODIC_TRAIN_INTERVAL_STEPS", "20") or 0))
if EPISODIC_TRAIN_INTERVAL_STEPS < 0:
    EPISODIC_TRAIN_INTERVAL_STEPS = 0
EPISODIC_MAX_SEQUENCES = int(os.environ.get("EPISODIC_MAX_SEQUENCES", "0"))
STEP_MAX_SEQUENCES = int(os.environ.get("STEP_MAX_SEQUENCES", "36"))
MP_V1_STEP_MAX_SEQUENCES = int(os.environ.get("MP_V1_STEP_MAX_SEQUENCES", "0"))

# Hardcode: only train at episode_end (no mid-episode or streaming PPO updates).
# This avoids fragmented updates and uses "batched" learning per episode.
FORCE_EPISODE_END_ONLY_UPDATES = True
# LR: 3e-4 → 1e-5 over ep 1-200 (linear decay for DTN stability)
LR_SCHED_START = float(os.environ.get("PPO_LR_START", "1e-3"))
LR_SCHED_END = float(os.environ.get("PPO_LR_END", "1e-5"))
LR_DECAY_START_EP = int(os.environ.get("LR_DECAY_START_EP", "1"))
LR_DECAY_END_EP = int(os.environ.get("LR_DECAY_END_EP", "150"))
# Legacy piecewise schedule (kept for reference, not used with linear decay)
LR_SCHEDULE = [
    (1, 200, 3e-4),  # placeholder, linear decay is used instead
]
NO_TRAINING_CAP = os.environ.get("NO_TRAINING_CAP", "false").lower() in ("1", "true", "yes")

# PPO stability knobs (optional, env-overridable)
PPO_KL_TARGET = float(os.environ.get("PPO_KL_TARGET", "0.015"))  # Standard PPO target
PPO_KL_STOP_MULT = float(os.environ.get("PPO_KL_STOP_MULT", "1.5"))  # Stop if KL > 1.5 * target
PPO_UPDATE_MIN_ADV_STD = float(os.environ.get("PPO_UPDATE_MIN_ADV_STD", "0.0"))  # gate updates if advantage std too small
PPO_VALUE_CLIP = float(os.environ.get("PPO_VALUE_CLIP", "10.0"))  # Enable with reasonable range
LOG_STD_LR_MULT = float(os.environ.get("LOG_STD_LR_MULT", "0.25"))  # lr multiplier for log_std param group
PPO_GRAD_CLIP = float(os.environ.get("PPO_GRAD_CLIP", "1.2"))  # grad clip max-norm
PPO_WEIGHT_DECAY = float(os.environ.get("PPO_WEIGHT_DECAY", "1e-3"))  # L2 regularization for optimizer
PPO_COST_VALUE_COEF = float(os.environ.get("PPO_COST_VALUE_COEF", "0.1"))  # cost critic loss weight (reward critic uses PPO_VALUE_COEF)



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
    """
    Linear LR decay from LR_SCHED_START to LR_SCHED_END over episodes.

    - ep 1 to LR_DECAY_START_EP: LR_SCHED_START (constant)
    - ep LR_DECAY_START_EP to LR_DECAY_END_EP: linear decay
    - ep LR_DECAY_END_EP+: LR_SCHED_END (constant)

    Example: PPO_LR_START=5e-4 PPO_LR_END=1e-5 LR_DECAY_START_EP=1 LR_DECAY_END_EP=200
    → ep 1-200: 5e-4 → 1e-5 linear decay, ep 200+: 1e-5
    """
    if episode_num is None or episode_num <= 0:
        return LR_SCHED_START
    if episode_num <= LR_DECAY_START_EP:
        return LR_SCHED_START
    if episode_num >= LR_DECAY_END_EP:
        return LR_SCHED_END
    # Linear interpolation
    progress = (episode_num - LR_DECAY_START_EP) / max(1, LR_DECAY_END_EP - LR_DECAY_START_EP)
    return LR_SCHED_START + progress * (LR_SCHED_END - LR_SCHED_START)


def entropy_coef_for_episode(episode_num):
    """
    Return the entropy coefficient for the given episode.

    Prefers explicit schedules defined via ENTROPY_SCHEDULE and falls back to
    the legacy linear decay controlled by ENTROPY_COEF_START/END.
    """
    if episode_num is None:
        episode_num = 0
    if ENTROPY_SCHEDULE:
        for start_ep, end_ep, entropy_value in ENTROPY_SCHEDULE:
            if start_ep <= episode_num <= end_ep:
                return entropy_value
        return ENTROPY_SCHEDULE[-1][2]
    if episode_num <= ENTROPY_DECAY_START_EP:
        return ENTROPY_COEF_START
    span = max(1.0, float(ENTROPY_DECAY_END_EP - ENTROPY_DECAY_START_EP))
    frac = min(1.0, max(0.0, (episode_num - ENTROPY_DECAY_START_EP) / span))
    return ENTROPY_COEF_START + frac * (ENTROPY_COEF_END - ENTROPY_COEF_START)


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
    cost_values: List = field(default_factory=list)  # List[torch.Tensor] V_cost(s) from cost critic
    rewards: List = field(default_factory=list)  # List[float]
    step_ids: List = field(default_factory=list)  # List[int] global step index for hindsight reward patching
    # step_id -> indices in this window (accelerates hindsight reward patching)
    hindsight_step_to_indices: Dict[int, List[int]] = field(default_factory=dict)
    dones: List = field(default_factory=list)  # List[bool]
    low_zone_relay_flags: List = field(default_factory=list)  # Track diagnostic mask
    # Loss-level Lagrangian: track decomposed signals per step
    delivery_rewards: List = field(default_factory=list)   # delivery-oriented reward component
    overhead_costs: List = field(default_factory=list)     # overhead penalty magnitude (as cost)

    # 초기 Hidden States (시퀀스 시작 **직전**)
    init_actor_h: Optional = None  # torch.Tensor
    init_critic_h: Optional = None  # torch.Tensor
    init_cost_critic_h: Optional = None  # torch.Tensor

    # 메모리 관리
    last_update_time: float = field(default_factory=lambda: time.time())

    def add_step(self, local_obs, global_obs, action, logprob, value,
                 reward, done, actor_h_prev=None, critic_h_prev=None,
                 low_zone_relay=False, delivery_reward: float = 0.0, overhead_cost: float = 0.0,
                 cost_value=None, cost_critic_h_prev=None, step_id: int = -1):
        """
        Transition 추가

        Args:
            actor_h_prev: (h, c) tuple - 이 step **이전** actor hidden
            critic_h_prev: (h, c) tuple - 이 step **이전** critic hidden
            cost_value: V_cost(s) from cost critic (tensor or float)
            cost_critic_h_prev: 이 step **이전** cost critic hidden
        """
        # 첫 step이면 init hidden 저장
        if len(self.local_obs) == 0:
            if actor_h_prev is not None:
                self.init_actor_h = actor_h_prev.detach()
            if critic_h_prev is not None:
                self.init_critic_h = critic_h_prev.detach()
            if cost_critic_h_prev is not None:
                self.init_cost_critic_h = cost_critic_h_prev.detach()

        self.local_obs.append(local_obs)
        self.global_obs.append(global_obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        # Cost critic value (default to zero tensor matching value shape)
        if cost_value is not None:
            self.cost_values.append(cost_value)
        else:
            self.cost_values.append(torch.zeros_like(value) if hasattr(value, 'shape') else torch.tensor(0.0))
        self.rewards.append(reward)
        try:
            self.step_ids.append(int(step_id))
        except Exception:
            self.step_ids.append(-1)
        # Index positions by global step id so hindsight patches don't need to scan every window/sequence.
        try:
            sid = int(self.step_ids[-1])
            if sid >= 0:
                self.hindsight_step_to_indices.setdefault(sid, []).append(len(self.rewards) - 1)
        except Exception:
            pass
        self.dones.append(done)
        self.low_zone_relay_flags.append(bool(low_zone_relay))
        # Decomposed signals (ensure float)
        try:
            self.delivery_rewards.append(float(delivery_reward or 0.0))
        except Exception:
            self.delivery_rewards.append(0.0)
        try:
            self.overhead_costs.append(float(overhead_cost or 0.0))
        except Exception:
            self.overhead_costs.append(0.0)
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
            'actions': torch.stack(self.actions),
            'logprobs': torch.stack(self.logprobs).squeeze(-1),
            'values': torch.stack(self.values),
            'cost_values': torch.stack(self.cost_values),
            'rewards': torch.tensor(self.rewards, dtype=torch.float32),
            'step_ids': torch.tensor(self.step_ids, dtype=torch.int64),
            # Store done mask as float to avoid dtype issues during GAE
            'dones': torch.tensor(self.dones, dtype=torch.float32),
            'low_zone_relay_flags': torch.tensor(self.low_zone_relay_flags, dtype=torch.bool),
            # Dual signals for loss-level Lagrangian
            'delivery_rewards': torch.tensor(self.delivery_rewards, dtype=torch.float32),
            'overhead_costs': torch.tensor(self.overhead_costs, dtype=torch.float32),
            'init_actor_h': self.init_actor_h,
            'init_critic_h': self.init_critic_h,
            'init_cost_critic_h': self.init_cost_critic_h,
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

    def __init__(self, obs_dim=7, global_obs_dim=20, hidden_size=128, num_layers=1, action_dim=1):
        # Hyperparameters
        self.clip_eps = float(os.environ.get("PPO_CLIP", "0.20"))
        # Stability tuning (hardcoded defaults; still overridable via env)
        self.value_coef = float(os.environ.get("PPO_VALUE_COEF", "0.3"))
        self.cost_value_coef = float(os.environ.get("PPO_COST_VALUE_COEF", "0.1"))
        self.entropy_coef = float(os.environ.get("PPO_ENTROPY", str(ENTROPY_COEF_START)))
        self.lr = float(os.environ.get("PPO_LR", str(LR_SCHED_START)))
        # Full-batch PPO over all finalized sequences (no mini-batching)
        self.epochs = int(os.environ.get("PPO_EPOCHS", "5"))
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
        self.action_dim = int(action_dim) if action_dim is not None else 1
        if self.action_dim <= 0:
            self.action_dim = 1
        # ===== Actor Network (Decentralized) =====
        # Input: local observations (7D)
        # GRU processes temporal sequence
        # Output: action distribution parameters

        self.actor_embedding = nn.Linear(obs_dim, hidden_size).to(self.device)
        # Optional LayerNorm before GRU
        self.use_layer_norm = PPO_LAYER_NORM
        if self.use_layer_norm:
            self.actor_ln = nn.LayerNorm(hidden_size).to(self.device)
        else:
            self.actor_ln = None
        self.actor_gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.action_dist = ACTION_DIST
        head_dim = (2 * self.action_dim) if self.action_dist == "beta" else self.action_dim
        self.actor_head = nn.Linear(hidden_size, head_dim).to(self.device)
        # Gaussian-only exploration parameter (retained for compatibility)
        self.log_std = nn.Parameter(torch.full((self.action_dim,), float(LOG_STD_INIT), device=self.device))
        self.log_std_min = -0.5

        # ===== Critic Network (Centralized) =====
        # Input: global observations (20D = 7 local + 13 global)
        # GRU processes temporal sequence
        # Output: state value

        self.critic_embedding = nn.Linear(global_obs_dim, hidden_size * 2).to(self.device)
        if self.use_layer_norm:
            self.critic_ln = nn.LayerNorm(hidden_size * 2).to(self.device)
        else:
            self.critic_ln = None
        self.critic_gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.critic_head = nn.Linear(hidden_size * 2, 1).to(self.device)

        # ===== Cost Critic Network (Centralized) =====
        # Separate value head for overhead cost estimation
        # Used only when LOSS_LAGRANGIAN=true for dual-advantage PPO
        self.cost_critic_embedding = nn.Linear(global_obs_dim, hidden_size * 2).to(self.device)
        if self.use_layer_norm:
            self.cost_critic_ln = nn.LayerNorm(hidden_size * 2).to(self.device)
        else:
            self.cost_critic_ln = None
        self.cost_critic_gru = nn.GRU(
            input_size=hidden_size * 2,
            hidden_size=hidden_size * 2,
            num_layers=num_layers,
            batch_first=True
        ).to(self.device)
        self.cost_critic_head = nn.Linear(hidden_size * 2, 1).to(self.device)

        # Optimizer for all parameters (log_std has its own LR multiplier)
        base_params = (
            list(self.actor_embedding.parameters()) +
            list(self.actor_gru.parameters()) +
            list(self.actor_head.parameters()) +
            list(self.critic_embedding.parameters()) +
            list(self.critic_gru.parameters()) +
            list(self.critic_head.parameters()) +
            list(self.cost_critic_embedding.parameters()) +
            list(self.cost_critic_gru.parameters()) +
            list(self.cost_critic_head.parameters())
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
        self.cost_critic_hidden_manager = HiddenStateManager(num_layers, hidden_size * 2, self.device)

        # Sequence management (NEW: SequenceWindow based)
        self.sequence_windows = {}  # key -> SequenceWindow
        self.ready_sequences = []   # List[dict] - finalized sequences ready for training
        self.max_ready_sequences = int(os.environ.get("MAX_READY_SEQUENCES", "8192"))
        # Hindsight combo patch acceleration: avoid scanning all windows/sequences for each step_id.
        self._hindsight_ready_index = defaultdict(list)  # step_id -> list[(seq_dict, indices_tuple)]
        self._hindsight_open_keys = defaultdict(set)     # step_id -> set(key) of windows that contain step_id

        # Memory management settings
        self.max_window_age = float(os.environ.get("MAX_WINDOW_AGE", "300.0"))  # 5 minutes
        self.max_active_windows = int(os.environ.get("MAX_ACTIVE_WINDOWS", "1000"))
        # Require exactly 32 sequences before training (hardcoded batch size)
        self.min_sequences_to_train = 12
        self.episodic_updates = EPISODIC_UPDATES
        # Cap how many per-step keys (sequences) are ingested into SequenceWindows when episodic updates are enabled.
        # This is separate from the PPO training batch size (max_sequences_per_update).
        self.step_max_sequences = int(STEP_MAX_SEQUENCES)

        # Last actions cache (for reward assignment)
        # Stores tensors in the policy-training space (pre-tanh u for Gaussian, z for Beta)
        self.last_actions = {}  # key -> (local_obs, global_obs, action_train, logprob, value, cost_value, actor_h_prev, critic_h_prev, cost_critic_h_prev)

        self.policy_id = "rmappo_v1"
        print(f"[RecurrentPPOPolicy] Initialized R-MAPPO:")
        print(f"  - Actor GRU: {obs_dim}D -> {hidden_size} (L={num_layers})")
        print(f"  - Critic GRU: {global_obs_dim}D -> {hidden_size*2} (L={num_layers})")
        print(f"  - Cost Critic GRU: {global_obs_dim}D -> {hidden_size*2} (L={num_layers})")
        print(f"  - Truncated BPTT: {TRUNCATED_BPTT_LEN} steps")
        print(f"  - LOSS_LAGRANGIAN: {LOSS_LAGRANGIAN}")
        print(f"  - LayerNorm: {'ENABLED' if self.use_layer_norm else 'disabled'}")

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

        normal = None
        try:
            normal = torch.distributions.Normal(mean64, std)
            u = normal.rsample()
        except Exception as e:
            print(f"[ERROR] Normal distribution failed: {e}")
            u = torch.zeros_like(mean64)

        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        a = torch.tanh(u)
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))

        if normal is None:
            logp = torch.zeros_like(u)
        else:
            logp = normal.log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), min=1e-12))
        logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
        logp = logp.sum(dim=-1)

        return (
            u.to(mean.dtype),
            a.to(mean.dtype),
            logp.to(mean.dtype),
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

        return logp.sum(dim=-1).to(mean.dtype)

    def _beta_sample(self, raw_params):
        """Sample z~Beta(alpha,beta), map to a=2z-1, and compute log_prob(z)."""
        params = torch.nan_to_num(raw_params)
        if params.shape[-1] != (2 * self.action_dim):
            raise ValueError(
                f"Beta raw_params dim mismatch: got {params.shape[-1]} but expected {2 * self.action_dim}"
            )
        sh = list(params.shape[:-1]) + [self.action_dim, 2]
        p2 = params.view(*sh)
        alpha_raw = p2[..., 0]
        beta_raw = p2[..., 1]
        alpha = torch.clamp(F.softplus(alpha_raw) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
        beta = torch.clamp(F.softplus(beta_raw) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
        dist = torch.distributions.Beta(alpha.to(dtype=torch.float64), beta.to(dtype=torch.float64))
        z = dist.rsample().to(dtype=raw_params.dtype)
        z = torch.clamp(z, min=BETA_EPS, max=1.0 - BETA_EPS)
        a = z * 2.0 - 1.0
        logp = dist.log_prob(z.to(dtype=torch.float64)).to(dtype=raw_params.dtype).sum(dim=-1)
        return z, a, logp, alpha, beta

    def _beta_logprob(self, raw_params, z):
        """Compute log_prob(z) under current Beta(alpha,beta)."""
        params = torch.nan_to_num(raw_params)
        if params.shape[-1] != (2 * self.action_dim):
            raise ValueError(
                f"Beta raw_params dim mismatch: got {params.shape[-1]} but expected {2 * self.action_dim}"
            )
        sh = list(params.shape[:-1]) + [self.action_dim, 2]
        p2 = params.view(*sh)
        alpha_raw = p2[..., 0]
        beta_raw = p2[..., 1]
        alpha = torch.clamp(F.softplus(alpha_raw) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
        beta = torch.clamp(F.softplus(beta_raw) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
        dist = torch.distributions.Beta(alpha.to(dtype=torch.float64), beta.to(dtype=torch.float64))
        z64 = torch.nan_to_num(z).to(dtype=torch.float64)
        if z64.dim() == params.dim() and z64.shape[:-1] == params.shape[:-1]:
            pass  # (..., action_dim)
        else:
            # Allow scalar z shape (...,) when action_dim == 1
            if self.action_dim == 1 and z64.shape == params.shape[:-1]:
                z64 = z64.unsqueeze(-1)
        z64 = torch.clamp(z64, min=BETA_EPS, max=1.0 - BETA_EPS)
        logp = dist.log_prob(z64).sum(dim=-1)
        return logp

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
        local_obs_seq = local_obs.unsqueeze(1)  # (batch, 1, obs_dim)

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
        actor_embed = self.actor_embedding(local_obs_seq)
        if self.actor_ln is not None:
            actor_embed = self.actor_ln(actor_embed)
        embedded_actor = torch.tanh(actor_embed)  # (batch, 1, hidden)
        gru_out_actor, h_new_actor = self.actor_gru(
            embedded_actor, h_init_actor
        )  # (batch, 1, hidden)

        # Update hidden states for all agents
        for i, key in enumerate(keys):
            # Extract hidden state for this agent
            h_new_i = h_new_actor[:, i:i+1, :]  # (num_layers, 1, hidden)
            self.actor_hidden_manager.update(key, h_new_i)

        # Policy head (batched)
        raw_params = self.actor_head(gru_out_actor.squeeze(1))
        if self.action_dist == "beta":
            z, actions, logprobs, _, _ = self._beta_sample(raw_params)
            action_train = z
        else:
            u_pre, actions, logprobs = self._tanh_gaussian_sample(raw_params)
            action_train = u_pre

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
        critic_embed = self.critic_embedding(global_obs_seq)
        if self.critic_ln is not None:
            critic_embed = self.critic_ln(critic_embed)
        embedded_critic = torch.tanh(critic_embed)  # (batch, 1, hidden*2)
        gru_out_critic, h_new_critic = self.critic_gru(
            embedded_critic, h_init_critic
        )  # (batch, 1, hidden*2)

        # Update hidden states for all agents
        for i, key in enumerate(keys):
            h_new_i = h_new_critic[:, i:i+1, :]
            self.critic_hidden_manager.update(key, h_new_i)

        # Value head (batched)
        values = self.critic_head(gru_out_critic.squeeze(1)).squeeze(-1)  # (batch,)

        # ===== Cost Critic Forward (BATCHED GRU) =====
        cost_critic_h_list = []
        cost_critic_hiddens_prev = []

        for key in keys:
            h_prev = self.cost_critic_hidden_manager.get_or_init(key)
            cost_critic_hiddens_prev.append(h_prev)
            cost_critic_h_list.append(h_prev)

        h_init_cost_critic = torch.cat(cost_critic_h_list, dim=1)

        cost_critic_embed = self.cost_critic_embedding(global_obs_seq)
        if self.cost_critic_ln is not None:
            cost_critic_embed = self.cost_critic_ln(cost_critic_embed)
        embedded_cost_critic = torch.tanh(cost_critic_embed)
        gru_out_cost_critic, h_new_cost_critic = self.cost_critic_gru(
            embedded_cost_critic, h_init_cost_critic
        )

        for i, key in enumerate(keys):
            h_new_i = h_new_cost_critic[:, i:i+1, :]
            self.cost_critic_hidden_manager.update(key, h_new_i)

        cost_values = self.cost_critic_head(gru_out_cost_critic.squeeze(1)).squeeze(-1)  # (batch,)

        # Store last actions for reward assignment
        act_list = action_scaled.cpu().tolist()
        if self.action_dim == 1:
            try:
                act_list = [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in act_list]
            except Exception:
                pass
        for i, k in enumerate(keys):
            self.last_actions[k] = (
                local_obs[i].detach(),
                global_obs[i].detach(),
                action_train[i].detach(),
                logprobs[i].detach(),
                values[i].detach(),
                cost_values[i].detach(),
                actor_hiddens_prev[i],       # Hidden BEFORE step
                critic_hiddens_prev[i],       # Hidden BEFORE step
                cost_critic_hiddens_prev[i],  # Hidden BEFORE step
            )

        return act_list

    def update(self, rewards_per_host, rewards_per_key, mapping_host_to_keys, delta_limit,
               epochs=None, done_keys=None, low_zone_relays=None,
               delivery_per_key=None, overhead_per_key=None,
               train_now=True, allow_cleanup=True, allow_trim=True,
               step_id: int = -1):
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
            if len(data) != 9:
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
        step_cap = int(getattr(self, "step_max_sequences", STEP_MAX_SEQUENCES))
        # When forcing episode-end-only updates, ingest all keys each step (no per-step cap).
        if (not FORCE_EPISODE_END_ONLY_UPDATES) and self.episodic_updates and step_cap > 0:
            all_keys = list(self.last_actions.keys())
            if len(all_keys) > step_cap:
                random.shuffle(all_keys)
                selected_keys = set(all_keys[:step_cap])

        for host, acted_keys in acted_by_host.items():
            if not acted_keys:
                continue

            r_host = float(rewards_per_host.get(host, 0.0))
            r_each = r_host / len(acted_keys) if len(acted_keys) > 0 else 0.0

            for k in acted_keys:
                if selected_keys is not None and k not in selected_keys:
                    continue
                # Unpack last_actions (9 elements)
                local_obs, global_obs, action, logprob, value, cost_value, actor_h_prev, critic_h_prev, cost_critic_h_prev = self.last_actions[k]
                r_key = float(rewards_per_key.get(k, r_each))
                # Decomposed signals (optional, default 0.0)
                d_rew = 0.0
                o_cost = 0.0
                try:
                    if delivery_per_key is not None:
                        d_rew = float(delivery_per_key.get(k, 0.0) or 0.0)
                except Exception:
                    d_rew = 0.0
                try:
                    if overhead_per_key is not None:
                        o_cost = float(overhead_per_key.get(k, 0.0) or 0.0)
                except Exception:
                    o_cost = 0.0

                # Done 판정: prev_transition에서 전달/드랍 이벤트 발생 시 즉시 종료
                done = k in done_index

                # Get or create sequence window for this agent
                if k not in self.sequence_windows:
                    self.sequence_windows[k] = SequenceWindow(key=k, max_len=TRUNCATED_BPTT_LEN)

                window = self.sequence_windows[k]
                window.add_step(
                    local_obs, global_obs, action, logprob, value, r_key, done,
                    actor_h_prev, critic_h_prev,
                    low_zone_relay=(k in low_zone_relays),
                    delivery_reward=d_rew, overhead_cost=o_cost,
                    cost_value=cost_value, cost_critic_h_prev=cost_critic_h_prev,
                    step_id=step_id,
                )
                # Track which open windows contain this step_id so hindsight patches can find them in O(1).
                if COMBO_HINDSIGHT_CREDIT:
                    try:
                        sid = int(step_id)
                        if sid >= 0:
                            self._hindsight_open_keys[sid].add(k)
                    except Exception:
                        pass
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
                        # Promote this window's step_id indices into the ready-sequence index.
                        self._hindsight_register_finalized_sequence(k, window, seq_data)
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
                        self.cost_critic_hidden_manager.reset([k])
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

    def _hindsight_register_finalized_sequence(self, key: str, window: "SequenceWindow", seq_data: dict):
        """
        Index step_id occurrences for hindsight reward patching.

        Keeps runtime bounded by the number of occurrences at that step (instead of scanning
        every open window / ready sequence for each patch).
        """
        if not COMBO_HINDSIGHT_CREDIT:
            return
        try:
            sid_map = getattr(window, "hindsight_step_to_indices", None) or {}
        except Exception:
            sid_map = {}
        if not sid_map:
            return
        for sid, idxs in sid_map.items():
            try:
                sid_i = int(sid)
            except Exception:
                continue
            try:
                self._hindsight_open_keys[sid_i].discard(key)
            except Exception:
                pass
            try:
                if idxs:
                    self._hindsight_ready_index[sid_i].append((seq_data, tuple(int(i) for i in idxs)))
            except Exception:
                continue

    def apply_hindsight_combo_reward(self, step_id: int, combo_total: float):
        """
        Retroactively add the combo shaping reward to transitions tagged with `step_id`.

        This is needed when we compute delivery success for messages created in a past step
        (hindsight credit assignment) and want the reward to be applied to that creation step.

        Distribution matches the legacy "per-host" combo behavior: compute unique hosts that
        have a transition at `step_id`, then add `combo_total/num_hosts` to *each* key under
        that host at that step.
        """
        try:
            sid = int(step_id)
        except Exception:
            return {"ok": False, "error": "bad_step_id"}
        try:
            total = float(combo_total)
        except Exception:
            total = 0.0
        if not math.isfinite(total) or total == 0.0:
            # Nothing to apply; also clear cached indices for this step so we don't leak memory.
            try:
                self._hindsight_ready_index.pop(sid, None)
                self._hindsight_open_keys.pop(sid, None)
            except Exception:
                pass
            return {"ok": True, "applied": 0, "hosts": 0, "per_host": 0.0}

        # Fast path: use per-step indices populated during updates/finalization.
        try:
            ready_entries = list(self._hindsight_ready_index.get(sid, []))
        except Exception:
            ready_entries = []
        try:
            open_keys = set(self._hindsight_open_keys.get(sid, set()))
        except Exception:
            open_keys = set()

        # Cache-miss fallback (e.g., older checkpoints/state): scan once.
        if not ready_entries and not open_keys:
            try:
                for k, window in (self.sequence_windows or {}).items():
                    try:
                        step_ids = getattr(window, "step_ids", None)
                        if not step_ids:
                            continue
                        for i, s in enumerate(step_ids):
                            if int(s) == sid:
                                open_keys.add(k)
                                break
                    except Exception:
                        continue
            except Exception:
                pass
            try:
                for seq in (self.ready_sequences or []):
                    try:
                        step_ids_t = seq.get("step_ids")
                        if step_ids_t is None:
                            continue
                        mask = (step_ids_t == sid)
                        has_any = bool(mask.any().item()) if hasattr(mask, "any") else False
                        if not has_any:
                            continue
                        # Convert boolean mask to explicit indices for efficient in-place update.
                        try:
                            idx_t = mask.nonzero(as_tuple=False).view(-1)
                            idxs = tuple(int(x.item()) for x in idx_t)
                        except Exception:
                            idxs = ()
                        ready_entries.append((seq, idxs))
                    except Exception:
                        continue
            except Exception:
                pass

        # Filter out stale references and compute hosts from actual occurrences we can still patch.
        hosts = set()
        window_entries = []  # (key, window, indices)
        for k in open_keys:
            window = self.sequence_windows.get(k)
            if window is None:
                continue
            try:
                idxs = (getattr(window, "hindsight_step_to_indices", None) or {}).get(sid) or []
            except Exception:
                idxs = []
            if not idxs:
                continue
            window_entries.append((k, window, idxs))
            try:
                host = k.split("#", 1)[0] if "#" in k else k
                if host:
                    hosts.add(host)
            except Exception:
                pass

        ready_entries_valid = []  # (seq, indices, rewards_obj)
        for seq, idxs in ready_entries:
            if not idxs:
                continue
            try:
                rewards_t = seq.get("rewards")
            except Exception:
                rewards_t = None
            if rewards_t is None:
                continue
            ready_entries_valid.append((seq, idxs, rewards_t))
            try:
                key = str(seq.get("key", ""))
                host = key.split("#", 1)[0] if "#" in key else key
                if host:
                    hosts.add(host)
            except Exception:
                pass

        if not hosts:
            try:
                self._hindsight_ready_index.pop(sid, None)
                self._hindsight_open_keys.pop(sid, None)
            except Exception:
                pass
            return {"ok": True, "applied": 0, "hosts": 0, "per_host": 0.0}

        per_host = total / float(len(hosts))
        applied = 0

        # Apply to open windows for this step_id
        for _k, window, idxs in window_entries:
            for idx in idxs:
                try:
                    window.rewards[int(idx)] = float(window.rewards[int(idx)]) + per_host
                    applied += 1
                except Exception:
                    continue

        # Apply to finalized sequences
        for seq, idxs, rewards_t in ready_entries_valid:
            if isinstance(rewards_t, torch.Tensor):
                try:
                    idx_t = torch.tensor(list(idxs), dtype=torch.long, device=rewards_t.device)
                    add_t = torch.full((idx_t.numel(),), float(per_host), dtype=rewards_t.dtype, device=rewards_t.device)
                    rewards_t.index_add_(0, idx_t, add_t)
                    applied += int(idx_t.numel())
                except Exception:
                    # Fallback to slow per-index update if torch advanced ops fail
                    try:
                        for idx in idxs:
                            rewards_t[int(idx)] = rewards_t[int(idx)] + float(per_host)
                            applied += 1
                    except Exception:
                        continue
            else:
                try:
                    for idx in idxs:
                        rewards_t[int(idx)] = float(rewards_t[int(idx)]) + float(per_host)
                        applied += 1
                except Exception:
                    continue

        # One-step-only: free indices once applied (or attempted).
        try:
            self._hindsight_ready_index.pop(sid, None)
            self._hindsight_open_keys.pop(sid, None)
        except Exception:
            pass

        return {"ok": True, "applied": applied, "hosts": len(hosts), "per_host": per_host}

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
                self._hindsight_register_finalized_sequence(k, window, seq_data)
                if allow_trim:
                    self._trim_ready_sequences()
            del self.sequence_windows[k]
        if keys:
            self.actor_hidden_manager.reset(keys)
            self.critic_hidden_manager.reset(keys)
            self.cost_critic_hidden_manager.reset(keys)

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
            window = self.sequence_windows.get(k)
            if window is None:
                continue
            try:
                last_v = float(window.values[-1].detach().cpu().item()) if window.values else 0.0
            except Exception:
                last_v = 0.0
            seq_data = window.finalize(bootstrap_value=last_v)
            if seq_data is not None:
                self.ready_sequences.append(seq_data)
                self._hindsight_register_finalized_sequence(k, window, seq_data)
                self._trim_ready_sequences()

            # Window 제거
            del self.sequence_windows[k]

            # Hidden state reset
            self.actor_hidden_manager.reset([k])
            self.critic_hidden_manager.reset([k])
            self.cost_critic_hidden_manager.reset([k])

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
                window = self.sequence_windows.get(k)
                if window is None:
                    continue
                try:
                    last_v = float(window.values[-1].detach().cpu().item()) if window.values else 0.0
                except Exception:
                    last_v = 0.0
                seq_data = window.finalize(bootstrap_value=last_v)
                if seq_data is not None:
                    self.ready_sequences.append(seq_data)
                    self._hindsight_register_finalized_sequence(k, window, seq_data)
                    self._trim_ready_sequences()

                del self.sequence_windows[k]
                self.actor_hidden_manager.reset([k])
                self.critic_hidden_manager.reset([k])
                self.cost_critic_hidden_manager.reset([k])

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

    def _compute_dual_gae(self, seq_data):
        """
        Compute separate GAEs for delivery reward and overhead cost.

        - Delivery advantage uses reward critic baseline (standard GAE).
        - Overhead advantage uses cost critic baseline (proper variance reduction).

        Returns:
            adv_delivery: Tensor[seq_len]
            adv_overhead: Tensor[seq_len]
            delivery_returns: Tensor[seq_len] (value targets for reward critic)
            cost_returns: Tensor[seq_len] (value targets for cost critic)
        """
        # tensors already on CPU in seq_data; move/cast to device/dtype
        values = torch.as_tensor(seq_data['values'], dtype=torch.float32, device=self.device)
        cost_values = torch.as_tensor(seq_data.get('cost_values', torch.zeros_like(values)), dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(seq_data['dones'], dtype=torch.float32, device=self.device)
        bootstrap_value = torch.as_tensor(seq_data['bootstrap_value'], dtype=torch.float32, device=self.device)
        del_rewards = torch.as_tensor(seq_data.get('delivery_rewards', torch.zeros_like(values)), dtype=torch.float32, device=self.device)
        ovh_costs = torch.as_tensor(seq_data.get('overhead_costs', torch.zeros_like(values)), dtype=torch.float32, device=self.device)

        seq_len = len(values)
        adv_delivery = torch.zeros(seq_len, dtype=torch.float32, device=self.device)
        adv_overhead = torch.zeros(seq_len, dtype=torch.float32, device=self.device)

        # Delivery path (standard GAE with reward critic baseline)
        next_value = bootstrap_value
        next_adv = torch.zeros(1, dtype=torch.float32, device=self.device)
        for t in reversed(range(seq_len)):
            delta = del_rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            adv_delivery[t] = delta + self.gamma * self.gae_lambda * next_adv * (1.0 - dones[t])
            next_value = values[t]
            next_adv = adv_delivery[t]
        delivery_returns = adv_delivery + values

        # Overhead path (GAE with cost critic baseline)
        # Bootstrap cost value = 0 at terminal (same convention as reward critic)
        next_value_c = torch.zeros(1, dtype=torch.float32, device=self.device)
        next_adv_c = torch.zeros(1, dtype=torch.float32, device=self.device)
        for t in reversed(range(seq_len)):
            delta_c = ovh_costs[t] + self.gamma * next_value_c * (1.0 - dones[t]) - cost_values[t]
            adv_overhead[t] = delta_c + self.gamma * self.gae_lambda * next_adv_c * (1.0 - dones[t])
            next_value_c = cost_values[t]
            next_adv_c = adv_overhead[t]
        cost_returns = adv_overhead + cost_values

        return adv_delivery, adv_overhead, delivery_returns, cost_returns

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

        # 1. Compute GAE for each sequence (dual-advantage if enabled)
        seq_data_list = []
        total_steps = 0

        invalid_seqs = 0
        for seq in self.ready_sequences[:n_train]:
            local = seq['local_obs']
            global_obs = seq['global_obs']
            if local.shape[-1] != self.obs_dim or global_obs.shape[-1] != self.global_obs_dim:
                invalid_seqs += 1
                continue

            # Decide advantage path
            if LOSS_LAGRANGIAN and ('delivery_rewards' in seq) and ('overhead_costs' in seq):
                try:
                    adv_d, adv_o, delivery_returns, cost_returns = self._compute_dual_gae(seq)
                    lam = float(getattr(Handler, 'CONSTRAINT_LAMBDA', 0.0)) if 'Handler' in globals() else 0.0
                    # Normalize each advantage to unit variance before combining
                    adv_d_n = adv_d / (adv_d.std(unbiased=False) + 1e-8)
                    adv_o_n = adv_o / (adv_o.std(unbiased=False) + 1e-8)
                    seq['advantages'] = lam * adv_d_n - adv_o_n
                    seq['returns'] = delivery_returns
                    seq['cost_returns'] = cost_returns
                except Exception as dl_exc:
                    # Fallback to standard path if dual fails
                    print(f"[PPO] Dual-GAE failed: {dl_exc}; falling back to standard GAE")
                    advantages, returns, _ = self._compute_gae_for_sequence(seq)
                    seq['advantages'] = advantages
                    seq['returns'] = returns
            else:
                advantages, returns, _ = self._compute_gae_for_sequence(seq)
                seq['advantages'] = advantages
                seq['returns'] = returns
            # Maintain low-zone diagnostic mask if present
            low_zone_mask = seq.get('low_zone_relay_flags')
            if isinstance(low_zone_mask, torch.Tensor):
                seq['low_zone_mask'] = low_zone_mask
            else:
                # Ensure presence for downstream diagnostics
                seq['low_zone_mask'] = torch.zeros_like(seq['returns'], dtype=torch.bool)
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
            padded_actions = torch.zeros(batch_size, max_len, self.action_dim, device=self.device)
            padded_old_logp = torch.zeros(batch_size, max_len, device=self.device)
            padded_adv = torch.zeros(batch_size, max_len, device=self.device)
            padded_ret = torch.zeros(batch_size, max_len, device=self.device)
            padded_cost_ret = torch.zeros(batch_size, max_len, device=self.device)

            # Stack init hidden states
            init_actor_h_list = []
            init_critic_h_list = []
            init_cost_critic_h_list = []

            # Check if any sequence has cost_returns (LOSS_LAGRANGIAN dual path)
            has_dual_path = LOSS_LAGRANGIAN and any('cost_returns' in seq for seq in batch_seqs)

            for i, seq in enumerate(batch_seqs):
                seq_len = seq_lengths[i]

                # Fill padded tensors
                padded_local[i, :seq_len] = seq['local_obs']
                padded_global[i, :seq_len] = seq['global_obs']
                padded_actions[i, :seq_len] = seq['actions']
                padded_old_logp[i, :seq_len] = seq['logprobs']
                padded_adv[i, :seq_len] = seq['advantages']
                padded_ret[i, :seq_len] = seq['returns']
                if has_dual_path and 'cost_returns' in seq:
                    padded_cost_ret[i, :seq_len] = seq['cost_returns']

                # Get init hidden states
                init_actor_h_list.append(
                    seq['init_actor_h'].to(self.device) if seq['init_actor_h'] is not None
                    else torch.zeros(self.num_layers, 1, self.hidden_size, device=self.device)
                )
                init_critic_h_list.append(
                    seq['init_critic_h'].to(self.device) if seq['init_critic_h'] is not None
                    else torch.zeros(self.num_layers, 1, self.hidden_size * 2, device=self.device)
                )
                init_cost_critic_h_list.append(
                    seq.get('init_cost_critic_h', torch.zeros(self.num_layers, 1, self.hidden_size * 2)).to(self.device)
                    if seq.get('init_cost_critic_h') is not None
                    else torch.zeros(self.num_layers, 1, self.hidden_size * 2, device=self.device)
                )

            # Stack hidden states: (num_layers, batch_size, hidden)
            # Each tensor is (num_layers, 1, hidden), concat on dim=1 (batch dimension)
            init_actor_h = torch.cat(init_actor_h_list, dim=1)  # (num_layers, batch, hidden)
            init_critic_h = torch.cat(init_critic_h_list, dim=1)  # (num_layers, batch, hidden*2)
            init_cost_critic_h = torch.cat(init_cost_critic_h_list, dim=1)  # (num_layers, batch, hidden*2)

            # GRU forward pass with pack_padded_sequence
            # Actor
            actor_embed = self.actor_embedding(padded_local)
            if self.actor_ln is not None:
                actor_embed = self.actor_ln(actor_embed)
            actor_embedded = torch.tanh(actor_embed)
            packed_actor = torch.nn.utils.rnn.pack_padded_sequence(
                actor_embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_actor_out, _ = self.actor_gru(packed_actor, init_actor_h)
            actor_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_actor_out, batch_first=True, total_length=max_len
            )

            # Reward Critic
            critic_embed = self.critic_embedding(padded_global)
            if self.critic_ln is not None:
                critic_embed = self.critic_ln(critic_embed)
            critic_embedded = torch.tanh(critic_embed)
            packed_critic = torch.nn.utils.rnn.pack_padded_sequence(
                critic_embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_critic_out, _ = self.critic_gru(packed_critic, init_critic_h)
            critic_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_critic_out, batch_first=True, total_length=max_len
            )

            # Cost Critic (always forward for gradient flow; loss gated by has_dual_path)
            cost_critic_embed = self.cost_critic_embedding(padded_global)
            if self.cost_critic_ln is not None:
                cost_critic_embed = self.cost_critic_ln(cost_critic_embed)
            cost_critic_embedded = torch.tanh(cost_critic_embed)
            packed_cost_critic = torch.nn.utils.rnn.pack_padded_sequence(
                cost_critic_embedded, seq_lengths, batch_first=True, enforce_sorted=False
            )
            packed_cost_critic_out, _ = self.cost_critic_gru(packed_cost_critic, init_cost_critic_h)
            cost_critic_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_cost_critic_out, batch_first=True, total_length=max_len
            )

            # Get actor output (policy parameters)
            raw_actor_out = self.actor_head(actor_out)  # (batch, max_len, head_dim)
            value_pred = self.critic_head(critic_out).squeeze(-1)  # (batch, max_len)
            cost_value_pred = self.cost_critic_head(cost_critic_out).squeeze(-1)  # (batch, max_len)

            # Compute log probs based on action distribution type
            if self.action_dist == "beta":
                # padded_actions stores z (raw Beta sample), shape (B, T, action_dim)
                new_logp = self._beta_logprob(raw_actor_out, padded_actions)
            else:
                new_logp = self._tanh_gaussian_logprob(raw_actor_out, padded_actions)

            # Create mask for valid positions
            mask = torch.zeros(batch_size, max_len, device=self.device)
            for i, length in enumerate(seq_lengths):
                mask[i, :length] = 1.0

            # Recompute advantages/returns using current value predictions for consistency
            if has_dual_path:
                # Dual-critic GAE recomputation: separate delivery and overhead paths
                lam = float(getattr(Handler, 'CONSTRAINT_LAMBDA', 0.0)) if 'Handler' in globals() else 0.0
                for i, seq in enumerate(batch_seqs):
                    L = seq_lengths[i]
                    if L <= 0:
                        continue
                    vals = value_pred[i, :L]
                    cvals = cost_value_pred[i, :L]
                    dns = seq['dones'].to(self.device)
                    del_rews = seq.get('delivery_rewards', seq['rewards']).to(self.device)
                    ovh_costs = seq.get('overhead_costs', torch.zeros(L, device=self.device)).to(self.device)

                    # Delivery GAE (reward critic baseline)
                    gae_d = torch.zeros(1, device=self.device)
                    adv_d = torch.zeros(L, device=self.device)
                    for t in range(L - 1, -1, -1):
                        next_nonterm = 1.0 - dns[t]
                        next_val = vals[t] if t == L - 1 else vals[t + 1]
                        delta = del_rews[t] + self.gamma * next_val * next_nonterm - vals[t]
                        gae_d = delta + self.gamma * self.gae_lambda * next_nonterm * gae_d
                        adv_d[t] = gae_d
                    ret_d = adv_d + vals

                    # Overhead GAE (cost critic baseline)
                    gae_c = torch.zeros(1, device=self.device)
                    adv_c = torch.zeros(L, device=self.device)
                    for t in range(L - 1, -1, -1):
                        next_nonterm = 1.0 - dns[t]
                        next_cval = cvals[t] if t == L - 1 else cvals[t + 1]
                        delta_c = ovh_costs[t] + self.gamma * next_cval * next_nonterm - cvals[t]
                        gae_c = delta_c + self.gamma * self.gae_lambda * next_nonterm * gae_c
                        adv_c[t] = gae_c
                    ret_c = adv_c + cvals

                    # Normalize and combine: A = λ * norm(A_delivery) - norm(A_overhead)
                    adv_d_n = adv_d / (adv_d.std(unbiased=False) + 1e-8)
                    adv_c_n = adv_c / (adv_c.std(unbiased=False) + 1e-8)
                    combined = lam * adv_d_n - adv_c_n

                    padded_adv[i, :L] = combined
                    padded_ret[i, :L] = ret_d
                    padded_cost_ret[i, :L] = ret_c
            else:
                # Standard single-critic GAE recomputation
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
            mask_bool = mask.bool()
            adv_valid_all = torch.masked_select(padded_adv, mask_bool)
            if adv_valid_all.numel() == 0:
                del self.ready_sequences[:n_train]
                return {"updated": False, "reason": "masked_adv_count_zero"}
            adv_mean = adv_valid_all.mean()
            adv_std = adv_valid_all.std(unbiased=False)
            if not torch.isfinite(adv_std):
                raise RuntimeError("Masked advantage std became non-finite during PPO update")
            padded_adv = (padded_adv - adv_mean) / (adv_std + 1e-8)

            # Apply mask and flatten
            flat_mask = mask_bool.view(-1)
            new_logp_valid = torch.masked_select(new_logp.view(-1), flat_mask)
            old_logp_valid = torch.masked_select(padded_old_logp.view(-1), flat_mask)
            adv_valid = torch.masked_select(padded_adv.view(-1), flat_mask)
            ret_valid = torch.masked_select(padded_ret.view(-1), flat_mask)
            value_pred_valid = torch.masked_select(value_pred.view(-1), flat_mask)

            # PPO loss
            ratio = (new_logp_valid - old_logp_valid).exp()
            surr1 = ratio * adv_valid
            surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_valid
            policy_loss = -torch.min(surr1, surr2).mean()

            # Reward critic value loss (optional clipping)
            if PPO_VALUE_CLIP > 0.0:
                # Build padded old values from sequences
                padded_old_values = torch.zeros(batch_size, max_len, device=self.device)
                for i, seq in enumerate(batch_seqs):
                    L = seq_lengths[i]
                    if L > 0:
                        v = seq['values'].to(self.device)
                        padded_old_values[i, :L] = v
                old_value_valid = torch.masked_select(padded_old_values.view(-1), flat_mask)
                v_clipped = old_value_valid + torch.clamp(value_pred_valid - old_value_valid, -PPO_VALUE_CLIP, PPO_VALUE_CLIP)
                value_loss_unclipped = (value_pred_valid - ret_valid).pow(2)
                value_loss_clipped = (v_clipped - ret_valid).pow(2)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped).mean()
            else:
                value_loss = (value_pred_valid - ret_valid).pow(2).mean()

            # Cost critic value loss (only when dual path active)
            cost_value_loss = torch.tensor(0.0, device=self.device)
            if has_dual_path:
                cost_ret_valid = torch.masked_select(padded_cost_ret.view(-1), flat_mask)
                cost_value_pred_valid = torch.masked_select(cost_value_pred.view(-1), flat_mask)
                cost_value_loss = (cost_value_pred_valid - cost_ret_valid).pow(2).mean()

            # Entropy computation
            if self.action_dist == "beta":
                # Beta entropy from current policy parameters (differentiable)
                params = torch.nan_to_num(raw_actor_out)
                if self.action_dim == 1:
                    alpha = torch.clamp(F.softplus(params[..., 0]) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
                    beta_p = torch.clamp(F.softplus(params[..., 1]) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
                    beta_dist = torch.distributions.Beta(
                        alpha.to(torch.float64), beta_p.to(torch.float64))
                    ent_per_step = beta_dist.entropy().to(raw_actor_out.dtype)  # (batch, max_len)
                else:
                    sh = list(params.shape[:-1]) + [self.action_dim, 2]
                    p2 = params.view(*sh)
                    alpha = torch.clamp(F.softplus(p2[..., 0]) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
                    beta_p = torch.clamp(F.softplus(p2[..., 1]) + BETA_MIN_PARAM, min=BETA_MIN_PARAM)
                    beta_dist = torch.distributions.Beta(
                        alpha.to(torch.float64), beta_p.to(torch.float64))
                    ent = beta_dist.entropy().to(raw_actor_out.dtype)  # (batch, max_len, action_dim)
                    ent_per_step = ent.sum(dim=-1)  # (batch, max_len)
                ent_valid = torch.masked_select(ent_per_step, mask_bool)
                entropy = ent_valid.mean() if ent_valid.numel() > 0 else torch.tensor(0.0, device=self.device)
            else:
                # Gaussian entropy from log_std parameter
                log_std_clamped = torch.clamp(self.log_std, min=self.log_std_min)
                if LOG_STD_MAX is not None and LOG_STD_MAX > 0:
                    log_std_clamped = torch.clamp(log_std_clamped, max=float(LOG_STD_MAX))
                entropy = (0.5 + 0.5 * math.log(2 * math.pi)) + log_std_clamped
                entropy = entropy.sum()

            # Total loss (reward critic + cost critic)
            loss = policy_loss + self.value_coef * value_loss + self.cost_value_coef * cost_value_loss - self.entropy_coef * entropy

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
                list(self.critic_head.parameters()) +
                list(self.cost_critic_embedding.parameters()) +
                list(self.cost_critic_gru.parameters()) +
                list(self.cost_critic_head.parameters()),
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

        try:
            cost_vl = float(cost_value_loss.item()) if cost_value_loss is not None else 0.0
        except Exception:
            cost_vl = 0.0

        result = {
            "updated": True,
            "trained_on": total_steps,
            "num_sequences": num_seqs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "cost_value_loss": cost_vl,
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
            self.cost_critic_hidden_manager.reset()
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
            "cost_critic_embedding": self.cost_critic_embedding.state_dict(),
            "cost_critic_gru": self.cost_critic_gru.state_dict(),
            "cost_critic_head": self.cost_critic_head.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "opt": self.opt.state_dict(),
            "actor_hiddens": self.actor_hidden_manager.get_all_states(),
            "critic_hiddens": self.critic_hidden_manager.get_all_states(),
            "cost_critic_hiddens": self.cost_critic_hidden_manager.get_all_states(),
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

            # Cost critic (backward compatible: skip if not in checkpoint)
            if "cost_critic_embedding" in ckpt:
                self.cost_critic_embedding.load_state_dict(ckpt["cost_critic_embedding"])
            if "cost_critic_gru" in ckpt:
                self.cost_critic_gru.load_state_dict(ckpt["cost_critic_gru"])
            if "cost_critic_head" in ckpt:
                self.cost_critic_head.load_state_dict(ckpt["cost_critic_head"])

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

            cost_critic_hiddens = ckpt.get("cost_critic_hiddens")
            if cost_critic_hiddens:
                self.cost_critic_hidden_manager.set_all_states(cost_critic_hiddens)

            meta = ckpt.get("meta", {})
            pid = meta.get("policy_id")
            if pid:
                self.policy_id = pid

            return {"ok": True, "meta": meta}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ---------------------------------------------------------------------------
# CTDE SAC (MA-SAC) default hyperparameters
# ---------------------------------------------------------------------------
# These are code-level defaults (standard SAC baselines). Env vars can still
# override them, but leaving env unset yields these values deterministically.
#
# Note on replay size: SAC commonly uses 1e6, but MP_v1 observations can be
# high-dimensional (hundreds of floats), so 200k is a safer default for memory.
SAC_DEFAULT_GAMMA = 0.99
SAC_DEFAULT_TAU = 0.005
SAC_DEFAULT_BATCH_SIZE = 256
SAC_DEFAULT_REPLAY_SIZE = 200_000
SAC_DEFAULT_MIN_REPLAY_SIZE = 10_000
SAC_DEFAULT_UPDATES_PER_TRAIN_CALL = 1
# Default kept small so `RmappoMaxpropBridgeReport.timeoutMs=3000` won't time out at episode_end.
# Increase via `SAC_UPDATES_PER_EPISODE_END` (and/or raise timeoutMs) for more training per episode.
SAC_DEFAULT_UPDATES_PER_EPISODE_END = 32
SAC_DEFAULT_ACTOR_LR = 3e-4
SAC_DEFAULT_CRITIC_LR = 3e-4
SAC_DEFAULT_ALPHA_LR = 3e-4
SAC_DEFAULT_AUTO_ALPHA = True
SAC_DEFAULT_ALPHA_INIT = 0.2
SAC_DEFAULT_TARGET_ENTROPY = None  # None => -action_dim
SAC_DEFAULT_GRAD_CLIP = 0.0
SAC_DEFAULT_LOG_STD_MIN = -20.0
SAC_DEFAULT_LOG_STD_MAX = 2.0
SAC_DEFAULT_LOGPROB_EPS = 1e-6

# Recurrent (RMA-SAC-style) defaults for MP_v1.
# Note: MP_v1 step is often 100s (RmappoMaxpropBridgeReport.sampleInterval=100),
# so seq_len=32 covers ~3200s of history, burn_in=8 covers ~800s.
SAC_RNN_DEFAULT_ENABLE = True
SAC_RNN_DEFAULT_SEQ_LEN = 32
SAC_RNN_DEFAULT_BURN_IN = 8
SAC_RNN_DEFAULT_BATCH_SEQS = 32
SAC_RNN_DEFAULT_MIN_EPISODES = 64
SAC_RNN_DEFAULT_MAX_EPISODES = 4096
SAC_RNN_DEFAULT_UPDATES_PER_EPISODE_END = 8


class RecurrentSquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)

        self.gru = nn.GRU(self.obs_dim, self.hidden_size, batch_first=True)
        self.mu = nn.Linear(self.hidden_size, self.action_dim)
        self.log_std = nn.Linear(self.hidden_size, self.action_dim)

    def forward(self, obs_seq: "torch.Tensor", h0=None):
        if obs_seq.ndim != 3 or obs_seq.shape[-1] != self.obs_dim:
            raise ValueError(
                f"[RecurrentActor] obs_seq shape mismatch: got {tuple(obs_seq.shape)} expected (*,*,{self.obs_dim})"
            )
        out, h1 = self.gru(obs_seq, h0)
        mean = self.mu(out)
        log_std = self.log_std(out)
        log_std = torch.clamp(log_std, SAC_DEFAULT_LOG_STD_MIN, SAC_DEFAULT_LOG_STD_MAX)
        return mean, log_std, h1

    def sample(self, obs_seq: "torch.Tensor", h0=None):
        mean, log_std, h1 = self.forward(obs_seq, h0=h0)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        x = normal.rsample()
        y = torch.tanh(x)
        logp = normal.log_prob(x) - torch.log(1.0 - y.pow(2) + SAC_DEFAULT_LOGPROB_EPS)
        logp = logp.sum(dim=-1, keepdim=True)
        mean_a = torch.tanh(mean)
        return y, logp, mean_a, h1


class RecurrentSACQNetwork(nn.Module):
    def __init__(self, global_obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.global_obs_dim = int(global_obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)

        self.gru = nn.GRU(self.global_obs_dim, self.hidden_size, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size + self.action_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def encode(self, global_obs_seq: "torch.Tensor", h0=None):
        if global_obs_seq.ndim != 3 or global_obs_seq.shape[-1] != self.global_obs_dim:
            raise ValueError(
                f"[RecurrentCritic] global_obs_seq shape mismatch: got {tuple(global_obs_seq.shape)} "
                f"expected (*,*,{self.global_obs_dim})"
            )
        return self.gru(global_obs_seq, h0)

    def forward(self, global_obs_seq: "torch.Tensor", action_seq: "torch.Tensor", h0=None):
        if action_seq.ndim != 3 or action_seq.shape[-1] != self.action_dim:
            raise ValueError(
                f"[RecurrentCritic] action_seq shape mismatch: got {tuple(action_seq.shape)} expected (*,*,{self.action_dim})"
            )
        out, h1 = self.encode(global_obs_seq, h0=h0)
        x = torch.cat([out, action_seq], dim=-1)
        b, t, d = x.shape
        q = self.head(x.reshape(b * t, d)).reshape(b, t, 1)
        return q, h1


class RecurrentEpisodeReplay:
    def __init__(self, max_episodes: int, obs_dim: int, global_obs_dim: int, action_dim: int):
        self.max_episodes = int(max(1, max_episodes))
        self.obs_dim = int(obs_dim)
        self.global_obs_dim = int(global_obs_dim)
        self.action_dim = int(action_dim)
        self.episodes = []

    def __len__(self):
        return int(len(self.episodes))

    def add_episode(self, ep: dict):
        self.episodes.append(ep)
        overflow = len(self.episodes) - self.max_episodes
        if overflow > 0:
            del self.episodes[:overflow]

    def sample(self, batch_seqs: int, seq_len: int, device: "torch.device"):
        if not self.episodes:
            raise ValueError("No episodes in replay")
        batch_seqs = int(max(1, batch_seqs))
        seq_len = int(max(2, seq_len))

        obs_b = np.zeros((batch_seqs, seq_len, self.obs_dim), dtype=np.float32)
        gobs_b = np.zeros((batch_seqs, seq_len, self.global_obs_dim), dtype=np.float32)
        act_b = np.zeros((batch_seqs, seq_len, self.action_dim), dtype=np.float32)
        rew_b = np.zeros((batch_seqs, seq_len, 1), dtype=np.float32)
        nobs_b = np.zeros((batch_seqs, seq_len, self.obs_dim), dtype=np.float32)
        ngobs_b = np.zeros((batch_seqs, seq_len, self.global_obs_dim), dtype=np.float32)
        done_b = np.ones((batch_seqs, seq_len, 1), dtype=np.float32)
        mask_b = np.zeros((batch_seqs, seq_len, 1), dtype=np.float32)

        for i in range(batch_seqs):
            for _ in range(64):
                ep = random.choice(self.episodes)
                l = int(ep["obs"].shape[0])
                if l >= seq_len:
                    start = random.randint(0, l - seq_len)
                    end = start + seq_len
                    obs_b[i] = ep["obs"][start:end]
                    gobs_b[i] = ep["gobs"][start:end]
                    act_b[i] = ep["act"][start:end]
                    rew_b[i, :, 0] = ep["rew"][start:end]
                    nobs_b[i] = ep["nobs"][start:end]
                    ngobs_b[i] = ep["ngobs"][start:end]
                    done_b[i, :, 0] = ep["done"][start:end]
                    mask_b[i, :, 0] = 1.0
                    break
            else:
                raise ValueError("Not enough episode length for recurrent sampling")

        return (
            torch.as_tensor(obs_b, device=device),
            torch.as_tensor(gobs_b, device=device),
            torch.as_tensor(act_b, device=device),
            torch.as_tensor(rew_b, device=device),
            torch.as_tensor(nobs_b, device=device),
            torch.as_tensor(ngobs_b, device=device),
            torch.as_tensor(done_b, device=device),
            torch.as_tensor(mask_b, device=device),
        )


class SACReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, global_obs_dim: int, action_dim: int):
        self.capacity = max(1, int(capacity))
        self.obs_dim = int(obs_dim)
        self.global_obs_dim = int(global_obs_dim)
        self.action_dim = int(action_dim)

        self._obs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._gobs = np.zeros((self.capacity, self.global_obs_dim), dtype=np.float32)
        self._act = np.zeros((self.capacity, self.action_dim), dtype=np.float32)
        self._rew = np.zeros((self.capacity, 1), dtype=np.float32)
        self._nobs = np.zeros((self.capacity, self.obs_dim), dtype=np.float32)
        self._ngobs = np.zeros((self.capacity, self.global_obs_dim), dtype=np.float32)
        self._done = np.zeros((self.capacity, 1), dtype=np.float32)

        self._ptr = 0
        self._size = 0

    def __len__(self) -> int:
        return int(self._size)

    def push_batch(
        self,
        obs: "np.ndarray",
        gobs: "np.ndarray",
        act: "np.ndarray",
        rew: "np.ndarray",
        nobs: "np.ndarray",
        ngobs: "np.ndarray",
        done: "np.ndarray",
    ) -> None:
        if obs is None:
            return
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[1] != self.obs_dim:
            raise ValueError(f"obs shape mismatch: {obs.shape} (expected (*,{self.obs_dim}))")
        n = int(obs.shape[0])
        if n <= 0:
            return

        gobs = np.asarray(gobs, dtype=np.float32)
        act = np.asarray(act, dtype=np.float32)
        rew = np.asarray(rew, dtype=np.float32).reshape(n, 1)
        nobs = np.asarray(nobs, dtype=np.float32)
        ngobs = np.asarray(ngobs, dtype=np.float32)
        done = np.asarray(done, dtype=np.float32).reshape(n, 1)

        if gobs.shape != (n, self.global_obs_dim):
            raise ValueError(f"gobs shape mismatch: {gobs.shape} (expected ({n},{self.global_obs_dim}))")
        if act.shape != (n, self.action_dim):
            raise ValueError(f"act shape mismatch: {act.shape} (expected ({n},{self.action_dim}))")
        if nobs.shape != (n, self.obs_dim):
            raise ValueError(f"nobs shape mismatch: {nobs.shape} (expected ({n},{self.obs_dim}))")
        if ngobs.shape != (n, self.global_obs_dim):
            raise ValueError(f"ngobs shape mismatch: {ngobs.shape} (expected ({n},{self.global_obs_dim}))")

        start = int(self._ptr)
        end = start + n
        if end <= self.capacity:
            sl = slice(start, end)
            self._obs[sl] = obs
            self._gobs[sl] = gobs
            self._act[sl] = act
            self._rew[sl] = rew
            self._nobs[sl] = nobs
            self._ngobs[sl] = ngobs
            self._done[sl] = done
        else:
            first = self.capacity - start
            sl1 = slice(start, self.capacity)
            sl2 = slice(0, end - self.capacity)
            self._obs[sl1] = obs[:first]
            self._gobs[sl1] = gobs[:first]
            self._act[sl1] = act[:first]
            self._rew[sl1] = rew[:first]
            self._nobs[sl1] = nobs[:first]
            self._ngobs[sl1] = ngobs[:first]
            self._done[sl1] = done[:first]

            self._obs[sl2] = obs[first:]
            self._gobs[sl2] = gobs[first:]
            self._act[sl2] = act[first:]
            self._rew[sl2] = rew[first:]
            self._nobs[sl2] = nobs[first:]
            self._ngobs[sl2] = ngobs[first:]
            self._done[sl2] = done[first:]

        self._ptr = int(end % self.capacity)
        self._size = int(min(self.capacity, self._size + n))

    def sample(self, batch_size: int, device: "torch.device"):
        if self._size <= 0:
            raise ValueError("cannot sample from empty buffer")
        b = int(batch_size)
        b = max(1, min(b, self._size))
        idx = np.random.randint(0, self._size, size=b)
        obs = torch.as_tensor(self._obs[idx], device=device)
        gobs = torch.as_tensor(self._gobs[idx], device=device)
        act = torch.as_tensor(self._act[idx], device=device)
        rew = torch.as_tensor(self._rew[idx], device=device)
        nobs = torch.as_tensor(self._nobs[idx], device=device)
        ngobs = torch.as_tensor(self._ngobs[idx], device=device)
        done = torch.as_tensor(self._done[idx], device=device)
        return obs, gobs, act, rew, nobs, ngobs, done


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)

        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
        )
        self.mean = nn.Linear(self.hidden_size, self.action_dim)
        self.log_std = nn.Linear(self.hidden_size, self.action_dim)

        self.log_std_min = float(os.environ.get("SAC_LOG_STD_MIN", str(SAC_DEFAULT_LOG_STD_MIN)))
        self.log_std_max = float(os.environ.get("SAC_LOG_STD_MAX", str(SAC_DEFAULT_LOG_STD_MAX)))
        self.eps = float(os.environ.get("SAC_LOGPROB_EPS", str(SAC_DEFAULT_LOGPROB_EPS)))

    def forward(self, obs: "torch.Tensor"):
        h = self.net(obs)
        mean = self.mean(h)
        log_std = self.log_std(h)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(self, obs: "torch.Tensor"):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()
        a = torch.tanh(u)
        logp = normal.log_prob(u) - torch.log(torch.clamp(1.0 - a.pow(2), min=self.eps))
        logp = logp.sum(dim=-1, keepdim=True)
        mean_a = torch.tanh(mean)
        return a, logp, mean_a


class SACQNetwork(nn.Module):
    def __init__(self, global_obs_dim: int, action_dim: int, hidden_size: int):
        super().__init__()
        self.global_obs_dim = int(global_obs_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.net = nn.Sequential(
            nn.Linear(self.global_obs_dim + self.action_dim, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 1),
        )

    def forward(self, global_obs: "torch.Tensor", action: "torch.Tensor"):
        x = torch.cat([global_obs, action], dim=-1)
        return self.net(x)


class CTDESACPolicy:
    """
    CTDE Soft Actor-Critic (SAC) for multi-host control.

    - Actor (Decentralized): local obs -> action in [-1,1]^A
    - Critic (Centralized): global obs (local + global_features) + action -> Q

    NOTE: This is feed-forward SAC (non-recurrent). The HTTP protocol, CTDE split,
    and host/key batching are kept compatible with the existing server.
    """

    def __init__(self, obs_dim: int, global_obs_dim: int, hidden_size: int, action_dim: int):
        device_str = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device_str)
        print(f"[CTDESACPolicy] Using device: {self.device}")

        self.obs_dim = int(obs_dim)
        self.global_obs_dim = int(global_obs_dim)
        self.hidden_size = int(hidden_size)
        self.action_dim = int(action_dim) if action_dim is not None else 1
        if self.action_dim <= 0:
            self.action_dim = 1

        self.gamma = float(os.environ.get("SAC_GAMMA", str(SAC_DEFAULT_GAMMA)))
        self.tau = float(os.environ.get("SAC_TAU", str(SAC_DEFAULT_TAU)))
        self.batch_size = int(os.environ.get("SAC_BATCH_SIZE", str(SAC_DEFAULT_BATCH_SIZE)))
        self.min_replay_size = int(os.environ.get("SAC_MIN_REPLAY_SIZE", str(SAC_DEFAULT_MIN_REPLAY_SIZE)))
        self.updates_per_train_call = int(os.environ.get("SAC_UPDATES_PER_TRAIN_CALL", str(SAC_DEFAULT_UPDATES_PER_TRAIN_CALL)))
        self.updates_per_episode_end = int(
            os.environ.get("SAC_UPDATES_PER_EPISODE_END", str(SAC_DEFAULT_UPDATES_PER_EPISODE_END))
        )
        self.grad_clip = float(os.environ.get("SAC_GRAD_CLIP", str(SAC_DEFAULT_GRAD_CLIP)))

        actor_lr = float(os.environ.get("SAC_ACTOR_LR", str(SAC_DEFAULT_ACTOR_LR)))
        critic_lr = float(os.environ.get("SAC_CRITIC_LR", str(SAC_DEFAULT_CRITIC_LR)))
        alpha_lr = float(os.environ.get("SAC_ALPHA_LR", str(SAC_DEFAULT_ALPHA_LR)))

        self.actor = SquashedGaussianActor(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q1 = SACQNetwork(self.global_obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q2 = SACQNetwork(self.global_obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)

        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)

        self.auto_alpha = os.environ.get(
            "SAC_AUTO_ALPHA", "true" if SAC_DEFAULT_AUTO_ALPHA else "false"
        ).lower() in ("1", "true", "yes")
        init_alpha = float(os.environ.get("SAC_ALPHA_INIT", str(SAC_DEFAULT_ALPHA_INIT)))
        if self.auto_alpha:
            self.log_alpha = torch.tensor(math.log(max(1e-8, init_alpha)), device=self.device, requires_grad=True)
            self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)
            default_target_entropy = (
                -float(self.action_dim) if SAC_DEFAULT_TARGET_ENTROPY is None else float(SAC_DEFAULT_TARGET_ENTROPY)
            )
            self.target_entropy = float(os.environ.get("SAC_TARGET_ENTROPY", str(default_target_entropy)))
        else:
            self.log_alpha = None
            self.alpha_opt = None
            self.target_entropy = None
            self._alpha = float(os.environ.get("SAC_ALPHA", str(init_alpha)))

        buf_cap = int(os.environ.get("SAC_REPLAY_SIZE", str(SAC_DEFAULT_REPLAY_SIZE)))
        self.replay_buffer = SACReplayBuffer(
            capacity=buf_cap,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
        )
        self.ready_sequences = self.replay_buffer  # compatibility

        self.policy_id = "masac_maxprop_v1"
        self.last_actions = {}  # key -> (local_obs[np], global_obs[np], action[np])

        self.step_max_sequences = 0

        try:
            te_desc = self.target_entropy if self.auto_alpha else None
            print(
                "[CTDESACPolicy] SAC defaults: "
                f"gamma={self.gamma} tau={self.tau} batch={self.batch_size} "
                f"replay={len(self.replay_buffer)}/{self.replay_buffer.capacity} min_replay={self.min_replay_size} "
                f"lr_actor={actor_lr} lr_critic={critic_lr} "
                f"auto_alpha={self.auto_alpha} alpha_init={init_alpha} target_entropy={te_desc} "
                f"updates_ep_end={self.updates_per_episode_end}"
            )
        except Exception:
            pass

    @property
    def alpha(self) -> float:
        if self.auto_alpha:
            return float(self.log_alpha.detach().exp().cpu().item())
        return float(self._alpha)

    def _to_tensor(self, x):
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _construct_global_obs(self, local_obs_t: "torch.Tensor", global_feat_t: "torch.Tensor"):
        return torch.cat([local_obs_t, global_feat_t], dim=-1)

    def act_batch(self, obs_batch, delta_limit, keys, global_features=None):
        if not obs_batch:
            return []
        local_obs = torch.nan_to_num(self._to_tensor(obs_batch))
        if local_obs.ndim != 2 or local_obs.shape[1] != self.obs_dim:
            raise ValueError(f"[CTDESACPolicy] obs_dim mismatch: got {tuple(local_obs.shape)} expected (*,{self.obs_dim})")
        batch_size = int(local_obs.shape[0])
        if keys is not None and len(keys) != batch_size:
            raise ValueError(f"[CTDESACPolicy] keys length mismatch: {len(keys)} vs batch {batch_size}")

        if global_features is None or len(global_features) == 0:
            gf_dim = max(0, self.global_obs_dim - self.obs_dim)
            global_feat = torch.zeros((batch_size, gf_dim), dtype=torch.float32, device=self.device)
        else:
            global_feat = torch.nan_to_num(self._to_tensor(global_features))
            if global_feat.ndim != 2:
                raise ValueError(f"[CTDESACPolicy] global_features must be 2D; got {tuple(global_feat.shape)}")
            if global_feat.shape[0] != batch_size:
                raise ValueError(f"[CTDESACPolicy] global_features batch mismatch: {global_feat.shape[0]} vs {batch_size}")
            if global_feat.shape[1] != (self.global_obs_dim - self.obs_dim):
                raise ValueError(
                    f"[CTDESACPolicy] global_feat dim mismatch: got {global_feat.shape[1]} expected {self.global_obs_dim - self.obs_dim}"
                )

        with torch.no_grad():
            if EVAL_ONLY:
                mean, _ = self.actor.forward(local_obs)
                a = torch.tanh(mean)
            else:
                a, _, _ = self.actor.sample(local_obs)
        a = torch.nan_to_num(a)

        scale = float(delta_limit) if (delta_limit is not None and float(delta_limit) > 0) else 1.0
        a_scaled = torch.clamp(a * scale, min=-abs(scale), max=abs(scale))

        global_obs = self._construct_global_obs(local_obs, global_feat)
        a_store = a.detach().cpu().numpy().astype(np.float32)
        local_store = local_obs.detach().cpu().numpy().astype(np.float32)
        global_store = global_obs.detach().cpu().numpy().astype(np.float32)
        for i, k in enumerate(keys or []):
            self.last_actions[str(k)] = (local_store[i], global_store[i], a_store[i])

        out = a_scaled.detach().cpu().numpy().tolist()
        if self.action_dim == 1:
            try:
                out = [float(v[0]) if isinstance(v, (list, tuple)) else float(v) for v in out]
            except Exception:
                pass
        return out

    def update(
        self,
        rewards_per_host,
        rewards_per_key,
        mapping_host_to_keys,
        delta_limit,
        epochs=None,
        done_keys=None,
        low_zone_relays=None,
        delivery_per_key=None,
        overhead_per_key=None,
        train_now=True,
        allow_cleanup=True,
        allow_trim=True,
        step_id: int = -1,
        next_obs_batch=None,
        next_keys=None,
        next_global_features=None,
    ):
        if not self.last_actions:
            return {"updated": False, "ready_sequences": len(self.replay_buffer)}

        next_state_local = {}
        next_state_global = {}
        if next_obs_batch and next_keys:
            next_local_t = torch.nan_to_num(self._to_tensor(next_obs_batch))
            if next_local_t.ndim != 2 or next_local_t.shape[1] != self.obs_dim:
                raise ValueError(f"[CTDESACPolicy] next_obs_dim mismatch: got {tuple(next_local_t.shape)} expected (*,{self.obs_dim})")
            batch_n = int(next_local_t.shape[0])
            if len(next_keys) != batch_n:
                raise ValueError(f"[CTDESACPolicy] next_keys length mismatch: {len(next_keys)} vs {batch_n}")

            if next_global_features is None or len(next_global_features) == 0:
                gf_dim = max(0, self.global_obs_dim - self.obs_dim)
                next_gf_t = torch.zeros((batch_n, gf_dim), dtype=torch.float32, device=self.device)
            else:
                next_gf_t = torch.nan_to_num(self._to_tensor(next_global_features))
                if next_gf_t.ndim != 2 or next_gf_t.shape[0] != batch_n:
                    raise ValueError("[CTDESACPolicy] next_global_features batch mismatch")
                if next_gf_t.shape[1] != (self.global_obs_dim - self.obs_dim):
                    raise ValueError("[CTDESACPolicy] next_global_features dim mismatch")

            next_global_t = self._construct_global_obs(next_local_t, next_gf_t)
            next_local_np = next_local_t.detach().cpu().numpy().astype(np.float32)
            next_global_np = next_global_t.detach().cpu().numpy().astype(np.float32)
            for i, k in enumerate(next_keys):
                kk = str(k)
                next_state_local[kk] = next_local_np[i]
                next_state_global[kk] = next_global_np[i]

        done_index = set(done_keys or [])

        acted_by_host = defaultdict(list)
        for k in self.last_actions.keys():
            host = k.split("#", 1)[0] if "#" in k else ""
            acted_by_host[host].append(k)

        step_cap = int(getattr(self, "step_max_sequences", 0) or 0)
        selected_keys = None
        if step_cap > 0:
            all_keys = list(self.last_actions.keys())
            if len(all_keys) > step_cap:
                random.shuffle(all_keys)
                selected_keys = set(all_keys[:step_cap])

        obs_l = []
        gobs_l = []
        act_l = []
        rew_l = []
        nobs_l = []
        ngobs_l = []
        done_l = []

        for host, acted_keys in acted_by_host.items():
            if not acted_keys:
                continue
            r_host = float(rewards_per_host.get(host, 0.0) or 0.0)
            r_each = r_host / float(len(acted_keys)) if acted_keys else 0.0
            for k in acted_keys:
                if selected_keys is not None and k not in selected_keys:
                    continue
                local_obs_np, global_obs_np, action_np = self.last_actions.get(k)
                r = float(rewards_per_key.get(k, r_each) or 0.0)
                done = (k in done_index)

                if k in next_state_local:
                    next_local_np = next_state_local[k]
                    next_global_np = next_state_global[k]
                else:
                    next_local_np = local_obs_np
                    next_global_np = global_obs_np
                    done = True

                obs_l.append(local_obs_np)
                gobs_l.append(global_obs_np)
                act_l.append(action_np)
                rew_l.append(r)
                nobs_l.append(next_local_np)
                ngobs_l.append(next_global_np)
                done_l.append(1.0 if done else 0.0)

        if obs_l:
            self.replay_buffer.push_batch(
                np.stack(obs_l, axis=0),
                np.stack(gobs_l, axis=0),
                np.stack(act_l, axis=0),
                np.asarray(rew_l, dtype=np.float32),
                np.stack(nobs_l, axis=0),
                np.stack(ngobs_l, axis=0),
                np.asarray(done_l, dtype=np.float32),
            )

        self.last_actions.clear()

        if not train_now:
            return {"updated": False, "ready_sequences": len(self.replay_buffer)}

        if len(self.replay_buffer) < max(1, self.min_replay_size):
            return {"updated": False, "ready_sequences": len(self.replay_buffer), "reason": "warmup"}

        return self._train_steps(int(self.updates_per_train_call))

    def _soft_update(self, src: nn.Module, dst: nn.Module):
        tau = float(self.tau)
        with torch.no_grad():
            for p, tp in zip(src.parameters(), dst.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def _train_steps(self, n_updates: int):
        n_updates = max(1, int(n_updates))
        if len(self.replay_buffer) < max(1, self.min_replay_size):
            return {"updated": False, "reason": "warmup", "ready_sequences": len(self.replay_buffer)}

        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_alpha_loss = 0.0
        last_entropy = 0.0
        last_alpha = self.alpha

        for _ in range(n_updates):
            obs, gobs, act, rew, nobs, ngobs, done = self.replay_buffer.sample(self.batch_size, self.device)

            with torch.no_grad():
                next_a, next_logp, _ = self.actor.sample(nobs)
                q1_t = self.q1_target(ngobs, next_a)
                q2_t = self.q2_target(ngobs, next_a)
                min_q_t = torch.min(q1_t, q2_t)
                alpha_t = self.alpha
                target_q = rew + (1.0 - done) * float(self.gamma) * (min_q_t - alpha_t * next_logp)

            q1 = self.q1(gobs, act)
            q2 = self.q2(gobs, act)
            critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.grad_clip)
            self.critic_opt.step()

            a_pi, logp_pi, _ = self.actor.sample(obs)
            q1_pi = self.q1(gobs, a_pi)
            q2_pi = self.q2(gobs, a_pi)
            min_q_pi = torch.min(q1_pi, q2_pi)
            alpha_val = self.alpha
            actor_loss = (alpha_val * logp_pi - min_q_pi).mean()

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()

            alpha_loss = torch.tensor(0.0, device=self.device)
            if self.auto_alpha:
                alpha_loss = -(self.log_alpha * (logp_pi.detach() + float(self.target_entropy))).mean()
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

            last_actor_loss = float(actor_loss.detach().cpu().item())
            last_critic_loss = float(critic_loss.detach().cpu().item())
            last_alpha_loss = float(alpha_loss.detach().cpu().item()) if alpha_loss is not None else 0.0
            last_entropy = float((-logp_pi).mean().detach().cpu().item())
            last_alpha = float(self.alpha)

        return {
            "updated": True,
            "trained_on": int(n_updates),
            "policy_loss": last_actor_loss,
            "value_loss": last_critic_loss,
            "cost_value_loss": 0.0,
            "entropy": last_entropy,
            "alpha": last_alpha,
            "alpha_loss": last_alpha_loss,
        }

    def train_ready_sequences(self, force=False, consume_all=False):
        if len(self.replay_buffer) < max(1, self.min_replay_size):
            return {"updated": False, "reason": "warmup", "ready_sequences": len(self.replay_buffer)}
        n = self.updates_per_episode_end if consume_all else self.updates_per_train_call
        return self._train_steps(int(n))

    def finalize_all_sequences(self, force_terminal=False, allow_trim=False):
        self.last_actions.clear()
        return {"ok": True}

    def reset_hidden_states(self, episode_start=False):
        return

    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "meta": {
                "policy_id": self.policy_id,
                "algo": "sac",
                "obs_dim": self.obs_dim,
                "global_obs_dim": self.global_obs_dim,
                "action_dim": self.action_dim,
                "hidden_size": self.hidden_size,
            },
        }
        if self.auto_alpha:
            ckpt["log_alpha"] = self.log_alpha.detach().cpu()
            ckpt["alpha_opt"] = self.alpha_opt.state_dict()
            ckpt["target_entropy"] = self.target_entropy
        else:
            ckpt["alpha"] = float(self._alpha)

        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            torch.save(ckpt, str(p))
            return {"ok": True, "path": str(p)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def load(self, path):
        if not path or not os.path.isfile(path):
            return {"ok": False, "error": "Missing checkpoint"}
        try:
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt["actor"])
            self.q1.load_state_dict(ckpt["q1"])
            self.q2.load_state_dict(ckpt["q2"])
            self.q1_target.load_state_dict(ckpt.get("q1_target", ckpt["q1"]))
            self.q2_target.load_state_dict(ckpt.get("q2_target", ckpt["q2"]))
            try:
                self.actor_opt.load_state_dict(ckpt.get("actor_opt", {}))
                self.critic_opt.load_state_dict(ckpt.get("critic_opt", {}))
            except Exception:
                pass
            if self.auto_alpha:
                la = ckpt.get("log_alpha")
                if la is not None:
                    with torch.no_grad():
                        self.log_alpha.copy_(la.to(self.device))
                try:
                    self.alpha_opt.load_state_dict(ckpt.get("alpha_opt", {}))
                except Exception:
                    pass
                try:
                    te = ckpt.get("target_entropy")
                    if te is not None:
                        self.target_entropy = float(te)
                except Exception:
                    pass
            else:
                try:
                    self._alpha = float(ckpt.get("alpha", self._alpha))
                except Exception:
                    pass
            meta = ckpt.get("meta", {})
            pid = meta.get("policy_id")
            if pid:
                self.policy_id = pid
            self.last_actions.clear()
            return {"ok": True, "meta": meta}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class CTDERMASACPolicy(CTDESACPolicy):
    """
    Recurrent CTDE SAC (RMA-SAC style) for MP_v1.

    - Actor/critic are GRU-based.
    - Experience is stored as per-key (host) episodes; training samples subsequences.
    - Training timing remains controlled by the server (episode_end-only by default).
    """

    def __init__(self, obs_dim: int, global_obs_dim: int, hidden_size: int, action_dim: int):
        super().__init__(obs_dim, global_obs_dim, hidden_size, action_dim)

        self.recurrent_enable = os.environ.get(
            "MP_V1_SAC_RECURRENT", "true" if SAC_RNN_DEFAULT_ENABLE else "false"
        ).lower() in ("1", "true", "yes")

        seq_len = int(os.environ.get("SAC_RNN_SEQ_LEN", str(SAC_RNN_DEFAULT_SEQ_LEN)))
        burn_in = int(os.environ.get("SAC_RNN_BURN_IN", str(SAC_RNN_DEFAULT_BURN_IN)))
        batch_seqs = int(os.environ.get("SAC_RNN_BATCH_SEQS", str(SAC_RNN_DEFAULT_BATCH_SEQS)))
        min_eps = int(os.environ.get("SAC_RNN_MIN_EPISODES", str(SAC_RNN_DEFAULT_MIN_EPISODES)))
        max_eps = int(os.environ.get("SAC_RNN_MAX_EPISODES", str(SAC_RNN_DEFAULT_MAX_EPISODES)))
        upd_ep_end = int(os.environ.get("SAC_RNN_UPDATES_PER_EPISODE_END", str(SAC_RNN_DEFAULT_UPDATES_PER_EPISODE_END)))

        self.seq_len = max(2, seq_len)
        self.burn_in = max(0, burn_in)
        self.batch_seqs = max(1, batch_seqs)
        self.min_episodes_to_train = max(1, min_eps)
        self.updates_per_episode_end = max(1, upd_ep_end)

        self.actor = RecurrentSquashedGaussianActor(self.obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q1 = RecurrentSACQNetwork(self.global_obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q2 = RecurrentSACQNetwork(self.global_obs_dim, self.action_dim, self.hidden_size).to(self.device)
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target = copy.deepcopy(self.q2).to(self.device)
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)

        actor_lr = float(os.environ.get("SAC_ACTOR_LR", str(SAC_DEFAULT_ACTOR_LR)))
        critic_lr = float(os.environ.get("SAC_CRITIC_LR", str(SAC_DEFAULT_CRITIC_LR)))
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=critic_lr)

        self.replay_ep = RecurrentEpisodeReplay(
            max_episodes=max_eps,
            obs_dim=self.obs_dim,
            global_obs_dim=self.global_obs_dim,
            action_dim=self.action_dim,
        )
        self._cur_ep = defaultdict(lambda: {"obs": [], "gobs": [], "act": [], "rew": [], "nobs": [], "ngobs": [], "done": []})
        self._actor_h = {}

        self.policy_id = "rmasac_maxprop_v1"

        try:
            print(
                "[CTDERMASACPolicy] Recurrent enabled: "
                f"seq_len={self.seq_len} burn_in={self.burn_in} batch_seqs={self.batch_seqs} "
                f"min_eps={self.min_episodes_to_train} max_eps={max_eps} updates_ep_end={self.updates_per_episode_end}"
            )
        except Exception:
            pass

    def reset_hidden_states(self, episode_start=False):
        if episode_start:
            self._actor_h.clear()

    def act_batch(self, obs_batch, delta_limit, keys, global_features=None):
        if not self.recurrent_enable:
            self.policy_id = "masac_maxprop_v1"
            return super().act_batch(obs_batch, delta_limit, keys, global_features=global_features)

        if not obs_batch:
            return []
        local_obs = torch.nan_to_num(self._to_tensor(obs_batch))
        if local_obs.ndim != 2 or local_obs.shape[1] != self.obs_dim:
            raise ValueError(f"[CTDERMASACPolicy] obs_dim mismatch: got {tuple(local_obs.shape)} expected (*,{self.obs_dim})")
        batch_size = int(local_obs.shape[0])
        if keys is not None and len(keys) != batch_size:
            raise ValueError(f"[CTDERMASACPolicy] keys length mismatch: {len(keys)} vs batch {batch_size}")

        if global_features is None or len(global_features) == 0:
            gf_dim = max(0, self.global_obs_dim - self.obs_dim)
            global_feat = torch.zeros((batch_size, gf_dim), dtype=torch.float32, device=self.device)
        else:
            global_feat = torch.nan_to_num(self._to_tensor(global_features))
            if global_feat.ndim != 2 or global_feat.shape[0] != batch_size:
                raise ValueError("[CTDERMASACPolicy] global_features batch mismatch")
            if global_feat.shape[1] != (self.global_obs_dim - self.obs_dim):
                raise ValueError("[CTDERMASACPolicy] global_features dim mismatch")

        x = local_obs.unsqueeze(1)  # [B,1,D]
        h0_list = []
        for k in keys or []:
            kk = str(k)
            h = self._actor_h.get(kk)
            if h is None:
                h = torch.zeros((1, 1, self.hidden_size), dtype=torch.float32, device=self.device)
            h0_list.append(h)
        h0 = torch.cat(h0_list, dim=1) if h0_list else None  # [1,B,H]

        with torch.no_grad():
            if EVAL_ONLY:
                mean, _, h1 = self.actor.forward(x, h0=h0)
                a = torch.tanh(mean)
            else:
                a, _, _, h1 = self.actor.sample(x, h0=h0)
        a = torch.nan_to_num(a[:, 0, :])

        if h1 is not None and keys:
            for i, k in enumerate(keys):
                self._actor_h[str(k)] = h1[:, i:i + 1, :].detach()

        scale = float(delta_limit) if (delta_limit is not None and float(delta_limit) > 0) else 1.0
        a_scaled = torch.clamp(a * scale, min=-abs(scale), max=abs(scale))

        global_obs = self._construct_global_obs(local_obs, global_feat)
        a_store = a.detach().cpu().numpy().astype(np.float32)
        local_store = local_obs.detach().cpu().numpy().astype(np.float32)
        global_store = global_obs.detach().cpu().numpy().astype(np.float32)
        for i, k in enumerate(keys or []):
            self.last_actions[str(k)] = (local_store[i], global_store[i], a_store[i])

        out = a_scaled.detach().cpu().numpy().tolist()
        return out

    def update(self, *args, **kwargs):
        if not self.recurrent_enable:
            self.policy_id = "masac_maxprop_v1"
            return super().update(*args, **kwargs)

        (
            rewards_per_host,
            rewards_per_key,
            mapping_host_to_keys,
            _delta_limit,
        ) = (args[0], args[1], args[2], args[3])

        done_keys = kwargs.get("done_keys")
        train_now = kwargs.get("train_now", True)
        step_id = int(kwargs.get("step_id", -1) or -1)
        next_obs_batch = kwargs.get("next_obs_batch")
        next_keys = kwargs.get("next_keys")
        next_global_features = kwargs.get("next_global_features")

        if not self.last_actions:
            return {"updated": False, "ready_sequences": len(self.replay_ep)}

        next_state_local = {}
        next_state_global = {}
        if next_obs_batch and next_keys:
            next_local_t = torch.nan_to_num(self._to_tensor(next_obs_batch))
            if next_local_t.ndim != 2 or next_local_t.shape[1] != self.obs_dim:
                raise ValueError(
                    f"[CTDERMASACPolicy] next_obs_dim mismatch: got {tuple(next_local_t.shape)} expected (*,{self.obs_dim})"
                )
            batch_n = int(next_local_t.shape[0])
            if len(next_keys) != batch_n:
                raise ValueError(f"[CTDERMASACPolicy] next_keys length mismatch: {len(next_keys)} vs {batch_n}")

            if next_global_features is None or len(next_global_features) == 0:
                gf_dim = max(0, self.global_obs_dim - self.obs_dim)
                next_gf_t = torch.zeros((batch_n, gf_dim), dtype=torch.float32, device=self.device)
            else:
                next_gf_t = torch.nan_to_num(self._to_tensor(next_global_features))
                if next_gf_t.ndim != 2 or next_gf_t.shape[0] != batch_n:
                    raise ValueError("[CTDERMASACPolicy] next_global_features batch mismatch")
                if next_gf_t.shape[1] != (self.global_obs_dim - self.obs_dim):
                    raise ValueError("[CTDERMASACPolicy] next_global_features dim mismatch")

            next_global_t = self._construct_global_obs(next_local_t, next_gf_t)
            next_local_np = next_local_t.detach().cpu().numpy().astype(np.float32)
            next_global_np = next_global_t.detach().cpu().numpy().astype(np.float32)
            for i, k in enumerate(next_keys):
                kk = str(k)
                next_state_local[kk] = next_local_np[i]
                next_state_global[kk] = next_global_np[i]

        done_index = set(done_keys or [])

        acted_by_host = defaultdict(list)
        for k in self.last_actions.keys():
            host = k.split("#", 1)[0] if "#" in k else ""
            acted_by_host[host].append(k)

        step_cap = int(getattr(self, "step_max_sequences", 0) or 0)
        selected_keys = None
        if step_cap > 0:
            all_keys = list(self.last_actions.keys())
            if len(all_keys) > step_cap:
                random.shuffle(all_keys)
                selected_keys = set(all_keys[:step_cap])

        for host, acted_keys in acted_by_host.items():
            if not acted_keys:
                continue
            r_host = float(rewards_per_host.get(host, 0.0) or 0.0)
            r_each = r_host / float(len(acted_keys)) if acted_keys else 0.0
            for k in acted_keys:
                if selected_keys is not None and k not in selected_keys:
                    continue
                local_obs_np, global_obs_np, action_np = self.last_actions.get(k)
                r = float(rewards_per_key.get(k, r_each) or 0.0)
                done = (k in done_index)

                if k in next_state_local:
                    next_local_np = next_state_local[k]
                    next_global_np = next_state_global[k]
                else:
                    next_local_np = local_obs_np
                    next_global_np = global_obs_np
                    done = True

                ep = self._cur_ep[str(k)]
                ep["obs"].append(local_obs_np)
                ep["gobs"].append(global_obs_np)
                ep["act"].append(action_np)
                ep["rew"].append(float(r))
                ep["nobs"].append(next_local_np)
                ep["ngobs"].append(next_global_np)
                ep["done"].append(1.0 if done else 0.0)

                if done:
                    # Host/key ended mid-episode; finalize this sequence early.
                    try:
                        n = len(ep["obs"])
                        if n >= max(2, self.seq_len):
                            self.replay_ep.add_episode(
                                {
                                    "obs": np.asarray(ep["obs"], dtype=np.float32).reshape(n, self.obs_dim),
                                    "gobs": np.asarray(ep["gobs"], dtype=np.float32).reshape(n, self.global_obs_dim),
                                    "act": np.asarray(ep["act"], dtype=np.float32).reshape(n, self.action_dim),
                                    "rew": np.asarray(ep["rew"], dtype=np.float32).reshape(n,),
                                    "nobs": np.asarray(ep["nobs"], dtype=np.float32).reshape(n, self.obs_dim),
                                    "ngobs": np.asarray(ep["ngobs"], dtype=np.float32).reshape(n, self.global_obs_dim),
                                    "done": np.asarray(ep["done"], dtype=np.float32).reshape(n,),
                                }
                            )
                    except Exception:
                        pass
                    try:
                        del self._cur_ep[str(k)]
                    except Exception:
                        pass

        self.last_actions.clear()

        if not train_now:
            return {"updated": False, "ready_sequences": len(self.replay_ep), "step_id": step_id}

        # Training is handled by train_ready_sequences() at episode_end to match existing timing.
        return {"updated": False, "ready_sequences": len(self.replay_ep), "step_id": step_id}

    def finalize_all_sequences(self, force_terminal=False, allow_trim=False):
        if not self.recurrent_enable:
            self.policy_id = "masac_maxprop_v1"
            return super().finalize_all_sequences(force_terminal=force_terminal, allow_trim=allow_trim)

        # Convert any in-progress per-key episodes into replay episodes.
        for k, data in list(self._cur_ep.items()):
            try:
                n = len(data["obs"])
            except Exception:
                n = 0
            if n <= 0:
                continue
            if force_terminal:
                try:
                    data["done"][-1] = 1.0
                except Exception:
                    pass
            try:
                ep = {
                    "obs": np.asarray(data["obs"], dtype=np.float32).reshape(n, self.obs_dim),
                    "gobs": np.asarray(data["gobs"], dtype=np.float32).reshape(n, self.global_obs_dim),
                    "act": np.asarray(data["act"], dtype=np.float32).reshape(n, self.action_dim),
                    "rew": np.asarray(data["rew"], dtype=np.float32).reshape(n,),
                    "nobs": np.asarray(data["nobs"], dtype=np.float32).reshape(n, self.obs_dim),
                    "ngobs": np.asarray(data["ngobs"], dtype=np.float32).reshape(n, self.global_obs_dim),
                    "done": np.asarray(data["done"], dtype=np.float32).reshape(n,),
                }
                if int(ep["obs"].shape[0]) >= max(2, self.seq_len):
                    self.replay_ep.add_episode(ep)
            except Exception:
                pass

        self._cur_ep.clear()
        self.last_actions.clear()
        self._actor_h.clear()
        return {"ok": True, "episodes": len(self.replay_ep)}

    def train_ready_sequences(self, force=False, consume_all=False):
        if not self.recurrent_enable:
            self.policy_id = "masac_maxprop_v1"
            return super().train_ready_sequences(force=force, consume_all=consume_all)

        if len(self.replay_ep) < self.min_episodes_to_train:
            return {"updated": False, "reason": "warmup", "ready_sequences": len(self.replay_ep)}

        n = self.updates_per_episode_end if consume_all else self.updates_per_train_call
        return self._train_steps_recurrent(int(n))

    def _soft_update(self, src: nn.Module, dst: nn.Module):
        tau = float(self.tau)
        with torch.no_grad():
            for p, tp in zip(src.parameters(), dst.parameters()):
                tp.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    def _train_steps_recurrent(self, n_updates: int):
        n_updates = max(1, int(n_updates))

        last_actor_loss = 0.0
        last_critic_loss = 0.0
        last_alpha_loss = 0.0
        last_entropy = 0.0
        last_alpha = self.alpha

        burn = int(max(0, min(self.burn_in, self.seq_len - 1)))

        for _ in range(n_updates):
            obs, gobs, act, rew, nobs, ngobs, done, mask = self.replay_ep.sample(
                self.batch_seqs, self.seq_len, self.device
            )

            if burn > 0:
                obs_b = obs[:, :burn, :]
                gobs_b = gobs[:, :burn, :]
                nobs_b = nobs[:, :burn, :]
                ngobs_b = ngobs[:, :burn, :]
            else:
                obs_b = gobs_b = nobs_b = ngobs_b = None

            obs_l = obs[:, burn:, :]
            gobs_l = gobs[:, burn:, :]
            act_l = act[:, burn:, :]
            rew_l = rew[:, burn:, :]
            nobs_l = nobs[:, burn:, :]
            ngobs_l = ngobs[:, burn:, :]
            done_l = done[:, burn:, :]
            mask_l = mask[:, burn:, :]

            # Build initial h from burn-in (no grad) to avoid BPTT through the entire prefix.
            with torch.no_grad():
                h_a_obs = None
                h_a_next = None
                h_q1 = None
                h_q2 = None
                h_q1t = None
                h_q2t = None
                if burn > 0:
                    _, _, h_a_obs = self.actor.forward(obs_b, h0=None)
                    _, _, h_a_next = self.actor.forward(nobs_b, h0=None)
                    _, h_q1 = self.q1.encode(gobs_b, h0=None)
                    _, h_q2 = self.q2.encode(gobs_b, h0=None)
                    _, h_q1t = self.q1_target.encode(ngobs_b, h0=None)
                    _, h_q2t = self.q2_target.encode(ngobs_b, h0=None)

            with torch.no_grad():
                next_a, next_logp, _, _ = self.actor.sample(nobs_l, h0=h_a_next)
                q1_t, _ = self.q1_target(ngobs_l, next_a, h0=h_q1t)
                q2_t, _ = self.q2_target(ngobs_l, next_a, h0=h_q2t)
                min_q_t = torch.min(q1_t, q2_t)
                alpha_t = self.alpha
                target_q = rew_l + (1.0 - done_l) * float(self.gamma) * (min_q_t - alpha_t * next_logp)

            q1, _ = self.q1(gobs_l, act_l, h0=h_q1)
            q2, _ = self.q2(gobs_l, act_l, h0=h_q2)
            denom = torch.clamp(mask_l.sum(), min=1.0)
            critic_loss = (((q1 - target_q).pow(2) + (q2 - target_q).pow(2)) * mask_l).sum() / denom

            self.critic_opt.zero_grad(set_to_none=True)
            critic_loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), self.grad_clip)
            self.critic_opt.step()

            a_pi, logp_pi, _, _ = self.actor.sample(obs_l, h0=h_a_obs)
            q1_pi, _ = self.q1(gobs_l, a_pi, h0=h_q1)
            q2_pi, _ = self.q2(gobs_l, a_pi, h0=h_q2)
            min_q_pi = torch.min(q1_pi, q2_pi)
            alpha_val = self.alpha
            actor_loss = ((alpha_val * logp_pi - min_q_pi) * mask_l).sum() / denom

            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            if self.grad_clip and self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
            self.actor_opt.step()

            alpha_loss = torch.tensor(0.0, device=self.device)
            if self.auto_alpha:
                alpha_loss = -(
                    self.log_alpha * ((logp_pi.detach() + float(self.target_entropy)) * mask_l).sum() / denom
                )
                self.alpha_opt.zero_grad(set_to_none=True)
                alpha_loss.backward()
                self.alpha_opt.step()

            self._soft_update(self.q1, self.q1_target)
            self._soft_update(self.q2, self.q2_target)

            last_actor_loss = float(actor_loss.detach().cpu().item())
            last_critic_loss = float(critic_loss.detach().cpu().item())
            last_alpha_loss = float(alpha_loss.detach().cpu().item()) if alpha_loss is not None else 0.0
            last_entropy = float(((-logp_pi) * mask_l).sum().detach().cpu().item() / float(denom.detach().cpu().item()))
            last_alpha = float(self.alpha)

        return {
            "updated": True,
            "trained_on": int(n_updates),
            "policy_loss": last_actor_loss,
            "value_loss": last_critic_loss,
            "cost_value_loss": 0.0,
            "entropy": last_entropy,
            "alpha": last_alpha,
            "alpha_loss": last_alpha_loss,
        }


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
    MP_V1_AGENT = None
    MP_V1_EP_COUNT = 0
    MP_V1_TRAINING_ENABLED = not EVAL_ONLY
    MP_V1_EP_REWARD_ACC = 0.0
    MP_V1_EP_COMBO_REWARD_SUM = 0.0
    MP_V1_EP_CREATED_SUM = 0
    MP_V1_EP_TRANSFERRED_SUM = 0
    MP_V1_EP_DELIVERED_SUM = 0
    MP_V1_EP_DROPPED_SUM = 0
    MP_V1_EP_ENERGY_USED_SUM = 0.0
    MP_V1_EP_HOST_SAMPLES_SUM = 0
    MP_V1_EP_STEP_COUNT = 0
    MP_V1_EP_LOSS_UPDATE_COUNT = 0
    MP_V1_EP_LOSS_POLICY_SUM = 0.0
    MP_V1_EP_LOSS_VALUE_SUM = 0.0
    MP_V1_EP_LOSS_COST_VALUE_SUM = 0.0
    MP_V1_EP_LOSS_ENTROPY_SUM = 0.0
    MP_V1_LAST_EPISODIC_TRAIN_STEP_ID = None
    _BOOTSTRAP_MIXER_LOAD = None
    _BOOTSTRAP_MIXER_SAVE = None

    # MP v1 episode-level action stats accumulator (for diagnosis)
    MP_V1_ACT_STATS = None

    # Optional: load initial model if provided
    if MODEL_PATH:
        try:
            if TORCH_OK:
                res = AGENT.load(MODEL_PATH)
                if res.get("ok", False):
                    print(f"[BOOT] Loaded R-MAPPO model from {MODEL_PATH}")
                    _BOOTSTRAP_MIXER_LOAD = MODEL_PATH
                    try:
                        Handler.load_pid_state(MODEL_PATH)
                    except Exception as pid_exc:
                        print(f"[PID] Load at boot failed: {pid_exc}")
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
                    try:
                        Handler.save_pid_state(BASELINE_PATH)
                    except Exception as pid_exc:
                        print(f"[PID] Baseline PID save failed: {pid_exc}")
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
        "constraint_penalty":0.0,
        "combo_reward":0.0,
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
    # Success constraint tracking
    EP_COMBO_REWARD_SUM = 0.0
    # Hindsight credit assignment tracking
    PREV_SIM_TIME = None  # Previous /infer_and_update sim_time (for step interval boundaries)
    HINDSIGHT_STEP_COUNTER = 0  # Current step number within episode
    HINDSIGHT_STEP_TIME_START = {}  # step -> start_time
    HINDSIGHT_STEP_TIME_END = {}    # step -> end_time
    HINDSIGHT_STEP_CREATED = {}     # step -> created count
    HINDSIGHT_STEP_DELIVERED = {}   # step -> delivered count (updated later)
    HINDSIGHT_STEP_TTL_SUM = {}     # step -> sum of ttl_ratios (for TTL-weighted)
    SAW_DELIVERY_EVENTS = False  # Whether we have observed delivery_events in requests this episode
    HINDSIGHT_PENDING_REWARDS = {}  # step -> pending combo reward to apply
    HINDSIGHT_STEP_RELAY_TOTAL = {}  # step -> total relayed events in that step interval
    HINDSIGHT_STEP_DELIVERED_INTERVAL = {}  # step -> delivered events in that step interval (not origin-attributed)
    HINDSIGHT_COMBO_APPLIED_STEPS = set()  # steps for which hindsight combo_reward has been applied
    SUCCESS_PROXY_EMA = None
    SUCCESS_PROXY_BEST = 0.0
    SUCCESS_PROXY_TARGET = SR_CONSTRAINT_BASE_TARGET
    SUCCESS_CONSTRAINT_DEFICIT = 0.0
    CONSTRAINT_LAMBDA = SR_CONSTRAINT_LAMBDA_INIT
    CONSTRAINT_STEP_PENALTY = 0.0
    LAST_CONSTRAINT_LOG_TIME = None
    CONSTRAINT_LAST_DENOM = SR_CONSTRAINT_DENOM
    CONSTRAINT_LAST_CREATED = 0.0
    CONSTRAINT_LAST_SUCCESS = None
    # PID controller for Lagrangian multiplier (initialized below once agent/env set)
    PID_CONTROLLER = PIDLambdaController(
        kp=PID_KP, ki=PID_KI, kd=PID_KD,
        lambda_init=SR_CONSTRAINT_LAMBDA_INIT,
        lambda_max=SR_CONSTRAINT_LAMBDA_MAX,
        integral_max=PID_INTEGRAL_MAX,
        lambda_min=LAMBDA_MIN,
        relax_factor=PID_RELAX_FACTOR,
        neg_integral_decay=PID_NEG_INTEGRAL_DECAY,
    )
    # Adaptive target tracking (dual EMA of delivery rate)
    DELIVERY_RATE_SLOW_EMA = None
    DELIVERY_RATE_FAST_EMA = None
    ADAPTIVE_WARMUP_RATES = []
    # Ratcheted best delivery rate (slow decay, only goes up easily)
    BEST_DELIVERY_RATE = 0.0
    # Overhead-based adaptive lambda floor
    OVERHEAD_EMA = None
    OVERHEAD_PEAK = None
    # Delivery rate EMA for dense reward signal (Stage 3)
    DELIVERY_STEP_EMA = 0.0

    # Episode-level loss accumulators (for plotting)
    EP_LOSS_POLICY_SUM = 0.0
    EP_LOSS_VALUE_SUM = 0.0
    EP_LOSS_COST_VALUE_SUM = 0.0
    EP_LOSS_ENTROPY_SUM = 0.0
    EP_LOSS_UPDATE_COUNT = 0
    LAST_EPISODIC_TRAIN_STEP_ID = None
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
    def _pid_ckpt_path(cls, base_path):
        if not base_path:
            return None
        return f"{base_path}.pid.json"

    @classmethod
    def save_pid_state(cls, base_path):
        """Persist PID controller state alongside model checkpoint."""
        try:
            if not base_path or cls.PID_CONTROLLER is None:
                return
            path = cls._pid_ckpt_path(base_path)
            state = cls.PID_CONTROLLER.state_dict() if hasattr(cls.PID_CONTROLLER, 'state_dict') else None
            if not state:
                return
            # Include ratcheted best delivery rate for continuity
            state["best_delivery_rate"] = cls.BEST_DELIVERY_RATE
            pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(state, f)
        except Exception as exc:
            print(f"[PID] Failed to save PID state: {exc}")

    @classmethod
    def load_pid_state(cls, base_path):
        """Restore PID controller state if present."""
        try:
            if not base_path or cls.PID_CONTROLLER is None:
                return
            path = cls._pid_ckpt_path(base_path)
            if not os.path.isfile(path):
                return
            with open(path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if hasattr(cls.PID_CONTROLLER, 'load_state_dict') and isinstance(state, dict):
                cls.PID_CONTROLLER.load_state_dict(state)
                # Restore ratcheted best delivery rate
                if "best_delivery_rate" in state:
                    cls.BEST_DELIVERY_RATE = float(state["best_delivery_rate"])
                print("[PID] Restored PID controller state from checkpoint")
        except Exception as exc:
            print(f"[PID] Failed to load PID state: {exc}")

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
            if TORCH_OK and Handler.TRAINING_ENABLED and (EPISODIC_UPDATES or FORCE_EPISODE_END_ONLY_UPDATES):
                try:
                    update_start = time.time()
                    Handler.AGENT.finalize_all_sequences(force_terminal=True, allow_trim=False)

                    # Apply any remaining hindsight combo rewards so the last steps of the episode
                    # also receive creation-step credit (they may not have reached the fixed delay yet).
                    if COMBO_REWARD_ENABLE and COMBO_HINDSIGHT_CREDIT:
                        try:
                            created_steps = set(Handler.HINDSIGHT_STEP_CREATED.keys())
                            pending_steps = sorted(created_steps - set(Handler.HINDSIGHT_COMBO_APPLIED_STEPS))
                            for s in pending_steps:
                                try:
                                    created_s = float(Handler.HINDSIGHT_STEP_CREATED.get(s, 0.0) or 0.0)
                                    ttl_sum_s = float(Handler.HINDSIGHT_STEP_TTL_SUM.get(s, 0.0) or 0.0)
                                    delivered_s = float(Handler.HINDSIGHT_STEP_DELIVERED.get(s, 0.0) or 0.0)
                                    S = 0.0
                                    if created_s > 0.0:
                                        S = min(1.0, ttl_sum_s / created_s)

                                    rel_total = float(Handler.HINDSIGHT_STEP_RELAY_TOTAL.get(s, 0.0) or 0.0)
                                    del_total = float(Handler.HINDSIGHT_STEP_DELIVERED_INTERVAL.get(s, 0.0) or 0.0)
                                    step_extra = max(0.0, rel_total - del_total)
                                    denom_rel = max(1.0, rel_total)
                                    O = max(0.0, min(1.0, step_extra / denom_rel))

                                    combo_total = 0.0
                                    if COMBO_FORM == 'log':
                                        import math as _m
                                        term_s = _m.log(max(COMBO_EPS, S))
                                        term_o = _m.log(max(COMBO_EPS, 1.0 - O))
                                        combo_total = COMBO_SCALE * (COMBO_ALPHA * term_s + COMBO_BETA * term_o)
                                    else:
                                        combo_total = COMBO_SCALE * (
                                            (max(COMBO_EPS, S) ** COMBO_ALPHA)
                                            * (max(COMBO_EPS, 1.0 - O) ** COMBO_BETA)
                                        )

                                    patch_res = None
                                    if combo_total != 0.0 and TORCH_OK and Handler.AGENT is not None:
                                        patch_res = Handler.AGENT.apply_hindsight_combo_reward(int(s), combo_total)
                                    Handler.HINDSIGHT_COMBO_APPLIED_STEPS.add(int(s))

                                    applied_ok = bool(isinstance(patch_res, dict) and patch_res.get("ok"))
                                    applied_n = 0
                                    if applied_ok:
                                        try:
                                            applied_n = int(patch_res.get("applied", 0) or 0)
                                        except Exception:
                                            applied_n = 0
                                    if combo_total != 0.0 and applied_ok and applied_n > 0:
                                        try:
                                            Handler.EP_REWARD_ACC = (Handler.EP_REWARD_ACC or 0.0) + float(combo_total)
                                        except Exception:
                                            pass
                                        try:
                                            Handler.EP_COMBO_REWARD_SUM = float(getattr(Handler, 'EP_COMBO_REWARD_SUM', 0.0) or 0.0) + float(combo_total)
                                        except Exception:
                                            pass

                                    # Log a compact tail-apply hint when there's meaningful data.
                                    if delivered_s > 0 or created_s > 0:
                                        raw_sr = delivered_s / max(1.0, created_s)
                                        msg = (f"[HINDSIGHT_EP_END] step={s} created={created_s:.0f} delivered={delivered_s:.0f} "
                                               f"raw_sr={raw_sr:.3f} ttl_weighted_S={S:.3f} O={O:.3f} combo={combo_total:.3f}")
                                        if isinstance(patch_res, dict) and patch_res.get("ok"):
                                            msg += (f" patched={patch_res.get('applied', 0)} hosts={patch_res.get('hosts', 0)}")
                                        print(msg)
                                except Exception:
                                    continue
                        except Exception as tail_exc:
                            print(f"[WARN] Failed to apply tail hindsight combo rewards: {tail_exc}")
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
                        Handler.EP_LOSS_COST_VALUE_SUM += float(episode_update_result.get('cost_value_loss', 0.0) or 0.0)
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
                    # Exploration schedule: episode-based entropy coefficient updates
                    try:
                        new_entropy = entropy_coef_for_episode(Handler.EP_COUNT)
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
                    try:
                        Handler.save_pid_state(path)
                    except Exception as pid_exc:
                        print(f"[PID] Failed to save PID at episode end: {pid_exc}")

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
                drop_rate_metric = None
                try:
                    created_total_for_drop = float(Handler.EP_CREATED_SUM or 0.0)
                    if math.isfinite(created_total_for_drop) and created_total_for_drop > 0.0:
                        drop_rate_metric = float(Handler.EP_DROPPED_SUM or 0.0) / created_total_for_drop
                        if not math.isfinite(drop_rate_metric):
                            drop_rate_metric = None
                except Exception:
                    drop_rate_metric = None
                log_episode_reward(
                    episode_num=Handler.EP_COUNT,
                    sim_id=sim_id,
                    total_reward=round(float(Handler.EP_REWARD_ACC or 0.0), 6),
                    combo_reward=round(float(getattr(Handler, 'EP_COMBO_REWARD_SUM', 0.0) or 0.0), 6),
                    avg_delivery_rate=(None if (avg is None or (isinstance(avg, float) and math.isnan(avg))) else avg),
                    delivered=delivered,
                    created=created,
                    avg_overhead=avg_overhead,
                    avg_delay=avg_delay_metric,
                    buffer_size_mb=buffer_size,
                    drop_rate=(None if drop_rate_metric is None else round(float(drop_rate_metric), 6)),
                    avg_energy_used=None,
                )
                update_success_constraint_from_episode(Handler.EP_COUNT)
                log_episode_core_totals(Handler.EP_COUNT, buffer_size)
                # Compute and log episode loss averages
                try:
                    upd = max(1, int(Handler.EP_LOSS_UPDATE_COUNT or 0))
                    actor_loss_avg = (Handler.EP_LOSS_POLICY_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                    critic_loss_avg = (Handler.EP_LOSS_VALUE_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                    cost_critic_loss_avg = (Handler.EP_LOSS_COST_VALUE_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                    entropy_avg = (Handler.EP_LOSS_ENTROPY_SUM / upd) if Handler.EP_LOSS_UPDATE_COUNT > 0 else None
                except Exception:
                    actor_loss_avg = None
                    critic_loss_avg = None
                    cost_critic_loss_avg = None
                    entropy_avg = None
                log_episode_losses(
                    episode_num=Handler.EP_COUNT,
                    sim_id=sim_id,
                    actor_loss_avg=actor_loss_avg,
                    critic_loss_avg=critic_loss_avg,
                    cost_critic_loss_avg=cost_critic_loss_avg,
                    updates=Handler.EP_LOSS_UPDATE_COUNT or 0,
                    entropy_avg=entropy_avg,
                    buffer_size_mb=buffer_size,
                )
                # Snapshot for response payload before resets
                _actor_loss_avg_snapshot = actor_loss_avg
                _critic_loss_avg_snapshot = critic_loss_avg
                _cost_critic_loss_avg_snapshot = cost_critic_loss_avg
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
            Handler.EP_COMBO_REWARD_SUM = 0.0
            # Reset hindsight credit assignment tracking
            Handler.PREV_SIM_TIME = None
            Handler.HINDSIGHT_STEP_COUNTER = 0
            Handler.HINDSIGHT_STEP_TIME_START = {}
            Handler.HINDSIGHT_STEP_TIME_END = {}
            Handler.HINDSIGHT_STEP_CREATED = {}
            Handler.HINDSIGHT_STEP_DELIVERED = {}
            Handler.HINDSIGHT_STEP_TTL_SUM = {}
            Handler.SAW_DELIVERY_EVENTS = False
            Handler.HINDSIGHT_PENDING_REWARDS = {}
            Handler.HINDSIGHT_STEP_RELAY_TOTAL = {}
            Handler.HINDSIGHT_STEP_DELIVERED_INTERVAL = {}
            Handler.HINDSIGHT_COMBO_APPLIED_STEPS = set()
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
            Handler.EP_LOSS_COST_VALUE_SUM = 0.0
            Handler.EP_LOSS_ENTROPY_SUM = 0.0
            Handler.EP_LOSS_UPDATE_COUNT = 0
            Handler.LAST_EPISODIC_TRAIN_STEP_ID = None

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
                if '_cost_critic_loss_avg_snapshot' in locals() and _cost_critic_loss_avg_snapshot is not None:
                    resp_payload['avg_cost_critic_loss'] = float(_cost_critic_loss_avg_snapshot)
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
                try:
                    Handler.save_pid_state(out_path)
                except Exception as pid_exc:
                    print(f"[PID] Manual save PID failed: {pid_exc}")
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
                try:
                    Handler.load_pid_state(in_path)
                except Exception as pid_exc:
                    print(f"[PID] Manual load PID failed: {pid_exc}")
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
                try:
                    Handler.load_pid_state(Handler.BASELINE_PATH)
                except Exception as pid_exc:
                    print(f"[PID] Baseline PID load failed: {pid_exc}")
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

        # --------------------------------------------------------------------
        # MaxProp++ parameter-control protocol (rmappo_maxprop_v1)
        # --------------------------------------------------------------------
        protocol = str(req.get("protocol", "")).strip()
        if protocol == "rmappo_maxprop_v1":
            try:
                resp_obj = handle_rmappo_maxprop_v1(req)
            except Exception as exc:
                resp_obj = {"policy_id": "error", "error": str(exc), "actions": []}
            body = json.dumps(resp_obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._safe_write(body)
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
        # Decomposed per-key signals for loss-level Lagrangian
        delivery_per_key = {}
        overhead_per_key = {}
        done_keys = set()
        stress_keys = set()
        # Track which keys actually relayed this step and which keys are in low-zone
        relayed_keys = set()
        low_zone_keys = set()
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
            'constraint_penalty': 0.0,
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
        if not math.isfinite(total_messages) or total_messages <= 0.0:
            total_messages = 1.0

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
        total_delivered_combo_ttl = 0.0
        for item in prev_tr:
            try:
                total_delivered += _to_float(item.get("delivered", 0.0), 0.0)
                if COMBO_IMMEDIATE_TTL_WEIGHT_MODE != "none":
                    delivered_i = _to_float(item.get("delivered", 0.0), 0.0)
                    if delivered_i > 0.0:
                        ttl_ratio_i = _to_float(item.get("avg_ttl_ratio", 1.0), 1.0)
                        if not math.isfinite(ttl_ratio_i):
                            ttl_ratio_i = 1.0
                        ttl_ratio_i = max(0.0, min(1.0, ttl_ratio_i))
                        w_i = 1.0
                        if COMBO_IMMEDIATE_TTL_WEIGHT_MODE == "exp":
                            w_i = math.exp(-COMBO_IMMEDIATE_TTL_EXP_K * (1.0 - ttl_ratio_i))
                        total_delivered_combo_ttl += delivered_i * w_i
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

        # Optional: TTL-weighted success-rate EMA dedicated to combo (keeps legacy STEP_SR_SMOOTHED unchanged)
        combo_step_sr = None
        if COMBO_IMMEDIATE_TTL_WEIGHT_MODE != "none" and created_since_last > 0.0:
            try:
                combo_step_sr = max(0.0, min(1.0, float(total_delivered_combo_ttl) / float(created_since_last)))
            except Exception:
                combo_step_sr = None
        if combo_step_sr is not None and math.isfinite(combo_step_sr):
            try:
                if (getattr(Handler, "COMBO_STEP_SR_SMOOTHED", None) is None
                        or not math.isfinite(float(getattr(Handler, "COMBO_STEP_SR_SMOOTHED", 0.0)))):
                    Handler.COMBO_STEP_SR_SMOOTHED = float(combo_step_sr)
                else:
                    Handler.COMBO_STEP_SR_SMOOTHED = (
                        STEP_SUCCESS_RATE_BETA * float(Handler.COMBO_STEP_SR_SMOOTHED)
                        + (1.0 - STEP_SUCCESS_RATE_BETA) * float(combo_step_sr)
                    )
            except Exception:
                pass
        created_increment = created_since_last if (created_since_last > 0.0 and math.isfinite(created_since_last)) else 0.0

        # ===== Hindsight Credit Assignment =====
        # Prefer step_id/prev_time from the Java bridge when present. This keeps step boundaries
        # aligned even if the server restarts mid-episode and avoids expensive time-interval searches.
        step_id = Handler.HINDSIGHT_STEP_COUNTER
        use_req_step_id = False
        try:
            req_step = req.get("step_id", None)
            if req_step is not None and math.isfinite(float(req_step)):
                step_id = int(float(req_step))
                use_req_step_id = (step_id >= 0)
        except Exception:
            use_req_step_id = False

        if not use_req_step_id:
            Handler.HINDSIGHT_STEP_COUNTER += 1
        else:
            # Keep counter monotonic for any internal fallback logic.
            Handler.HINDSIGHT_STEP_COUNTER = max(int(getattr(Handler, "HINDSIGHT_STEP_COUNTER", 0) or 0), int(step_id) + 1)

        prev_sim_time = Handler.PREV_SIM_TIME
        try:
            req_prev_time = req.get("prev_time", None)
            if req_prev_time is not None and math.isfinite(float(req_prev_time)):
                prev_sim_time = float(req_prev_time)
        except Exception:
            pass
        Handler.PREV_SIM_TIME = sim_time

        if COMBO_HINDSIGHT_CREDIT:
            # Record step interval boundaries as (prev_time, now], matching created_since_last / prev_transition.
            if prev_sim_time is None:
                prev_sim_time = sim_time
            Handler.HINDSIGHT_STEP_TIME_START[step_id] = float(prev_sim_time)
            Handler.HINDSIGHT_STEP_TIME_END[step_id] = float(sim_time)
            Handler.HINDSIGHT_STEP_CREATED[step_id] = float(created_increment)
            # Ensure origin-attribution buckets exist
            Handler.HINDSIGHT_STEP_DELIVERED.setdefault(step_id, 0.0)
            Handler.HINDSIGHT_STEP_TTL_SUM.setdefault(step_id, 0.0)

            # Process delivery_events: attribute to original creation step
            delivery_events = req.get("delivery_events", [])
            if delivery_events:
                Handler.SAW_DELIVERY_EVENTS = True
            for ev in delivery_events:
                try:
                    creation_time = float(ev.get("creation_time", 0.0))
                    ttl_ratio = float(ev.get("ttl_ratio", 1.0))
                    if not math.isfinite(ttl_ratio):
                        ttl_ratio = 1.0
                    ttl_ratio = max(0.0, min(1.0, ttl_ratio))

                    # Find the step when this message was created
                    origin_step = None
                    try:
                        csid = ev.get("creation_step_id", None)
                        if csid is not None and math.isfinite(float(csid)):
                            cand = int(float(csid))
                            if cand >= 0:
                                origin_step = cand
                    except Exception:
                        origin_step = None

                    if origin_step is None:
                        # Legacy path: locate origin step by searching recorded time boundaries.
                        for step_num in range(step_id, -1, -1):
                            start_t = Handler.HINDSIGHT_STEP_TIME_START.get(step_num, float('inf'))
                            end_t = Handler.HINDSIGHT_STEP_TIME_END.get(step_num, float('inf'))
                            if start_t <= creation_time < end_t:
                                origin_step = step_num
                                break
                            elif creation_time < start_t and step_num == 0:
                                origin_step = 0  # Before first step
                                break

                    if origin_step is not None:
                        Handler.HINDSIGHT_STEP_DELIVERED.setdefault(origin_step, 0.0)
                        Handler.HINDSIGHT_STEP_TTL_SUM.setdefault(origin_step, 0.0)
                        Handler.HINDSIGHT_STEP_DELIVERED[origin_step] = \
                            Handler.HINDSIGHT_STEP_DELIVERED.get(origin_step, 0.0) + 1.0
                        # TTL-weighted delivered credit:
                        # - exp: penalize late arrivals smoothly (ttl_ratio low => small)
                        # - bonus: legacy linear bonus (always >= 1.0)
                        # - none/other: each delivered counts as 1.0
                        if COMBO_TTL_WEIGHT_MODE == "exp":
                            weighted_value = math.exp(-COMBO_TTL_EXP_K * (1.0 - ttl_ratio))
                        elif COMBO_TTL_WEIGHT_MODE in ("bonus", "linear", "linear_bonus"):
                            weighted_value = 1.0 + COMBO_TTL_BONUS_SCALE * ttl_ratio
                        else:
                            weighted_value = 1.0
                        Handler.HINDSIGHT_STEP_TTL_SUM[origin_step] = \
                            Handler.HINDSIGHT_STEP_TTL_SUM.get(origin_step, 0.0) + weighted_value
                except Exception:
                    continue

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

                # 1) Delivery signal (loss-level) and optional reward shaping
                if relayed > 0:
                    step_relay_total += relayed
                    if key:
                        relayed_keys.add(key)
                ttl_ratio = _to_float(item.get("avg_ttl_ratio", 1.0), 1.0)
                if not math.isfinite(ttl_ratio):
                    ttl_ratio = 1.0
                ttl_ratio = max(0.0, min(1.0, ttl_ratio))
                ttl_weight = 1.0
                if RELAY_TTL_DECAY_K > 0.0:
                    ttl_weight = math.exp(-RELAY_TTL_DECAY_K * max(0.0, 1.0 - ttl_ratio))

                # Loss-level delivery signal: normalized by batch size
                if delivered > 0 and key:
                    base_delivery = (delivered / max(1.0, total_messages)) * ttl_weight
                    delivery_per_key[key] = delivery_per_key.get(key, 0.0) + float(base_delivery)

                # Optional reward shaping (kept for backward-compat)
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
                # 2) Overhead signal (loss-level) and optional reward shaping (disabled when LOSS_LAGRANGIAN)
                if relayed > 0:
                    extra_relays = max(0.0, relayed - delivered)
                    if extra_relays > 0.0:
                        overhead_factor = 1.0 - math.exp(-extra_relays / OVERHEAD_DECAY)
                        if overhead_factor > 0.0:
                            # Base normalized overhead magnitude in [0,1]
                            event_scale = extra_relays / max(1.0, relayed)
                            penalty_scale = max(0.0, min(1.0, overhead_factor * event_scale))
                            # Loss-level cost: dampen when overhead already low relative to peak
                            oh_damp = 1.0
                            if (Handler.OVERHEAD_EMA is not None
                                    and Handler.OVERHEAD_PEAK is not None
                                    and Handler.OVERHEAD_PEAK > 0):
                                oh_damp = min(1.0, Handler.OVERHEAD_EMA / Handler.OVERHEAD_PEAK)
                            if key:
                                overhead_per_key[key] = overhead_per_key.get(key, 0.0) + float(penalty_scale * oh_damp)
                            # Reward shaping path only when explicitly enabled and loss-level off
                            if (not LOSS_LAGRANGIAN) and OVERHEAD_PENALTY_WEIGHT != 0.0 and not (COMBO_REWARD_ENABLE and COMBO_SUPPRESS_OVERHEAD_SHAPING):
                                # Diminish overhead pressure as overhead drops from peak
                                # Prevents runaway convergence to overhead ~1
                                oh_ps = 1.0
                                if (Handler.OVERHEAD_EMA is not None
                                        and Handler.OVERHEAD_PEAK is not None
                                        and Handler.OVERHEAD_PEAK > 0):
                                    oh_ps = min(1.0, Handler.OVERHEAD_EMA / Handler.OVERHEAD_PEAK)
                                comp_over = -abs(OVERHEAD_PENALTY_WEIGHT) * penalty_scale * 10.0 * oh_ps
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
                # Track keys that are in low buffer utilization zone
                if avg_buf_cached is not None and avg_buf_cached <= BUFFER_DIFF_LOW_THRESHOLD:
                    if key:
                        low_zone_keys.add(key)
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

        # Store interval totals for hindsight combo overhead calculation (step-aligned).
        if COMBO_HINDSIGHT_CREDIT:
            try:
                Handler.HINDSIGHT_STEP_RELAY_TOTAL[step_id] = float(step_relay_total)
            except Exception:
                Handler.HINDSIGHT_STEP_RELAY_TOTAL[step_id] = 0.0
            try:
                Handler.HINDSIGHT_STEP_DELIVERED_INTERVAL[step_id] = float(total_delivered)
            except Exception:
                Handler.HINDSIGHT_STEP_DELIVERED_INTERVAL[step_id] = 0.0

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

        # ====================================================================
        # Stage 3: Delivery EMA dense signal
        # ====================================================================
        step_delivery_rate = total_delivered / max(1.0, total_messages)
        if math.isfinite(step_delivery_rate):
            Handler.DELIVERY_STEP_EMA = (
                DELIVERY_EMA_ALPHA * step_delivery_rate
                + (1.0 - DELIVERY_EMA_ALPHA) * Handler.DELIVERY_STEP_EMA
            )
        if DELIVERY_EMA_DENSE_WEIGHT > 0.0 and Handler.DELIVERY_STEP_EMA > 0.0:
            dense_signal = DELIVERY_EMA_DENSE_WEIGHT * Handler.DELIVERY_STEP_EMA
            for k in keys:
                delivery_per_key[k] = delivery_per_key.get(k, 0.0) + dense_signal

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
            if SR_SPARSE_NORMALIZE:
                norm = max(1.0, total_messages)
            else:
                norm = 1.0
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
                base_term = SR_SPARSE_WEIGHT
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

        # Apply Lagrangian constraint penalty with per-host weighting
        # Skip when loss-level Lagrangian is active (penalty flows through loss instead)
        # Also skip when combo shaping is enabled and configured to suppress constraint shaping
        if (not LOSS_LAGRANGIAN) and (not (COMBO_REWARD_ENABLE and COMBO_SUPPRESS_CONSTRAINT_SHAPING)) and Handler.CONSTRAINT_STEP_PENALTY > 0.0:
            target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
            if target_hosts:
                # Build per-host weights emphasizing low-zone keys that did not relay
                weights = {}
                total_w = 0.0
                for host in target_hosts:
                    host_keys = map_host_to_keys.get(host, []) or []
                    miss_low = 0
                    for k in host_keys:
                        if k in low_zone_keys and k not in relayed_keys:
                            miss_low += 1
                    w = float(miss_low)
                    weights[host] = w
                    total_w += w

                # Fallback to uniform if no signal
                if total_w <= 0.0:
                    per_host_penalty = -Handler.CONSTRAINT_STEP_PENALTY / max(1, len(target_hosts))
                    total_penalty = per_host_penalty * len(target_hosts)
                    for host in target_hosts:
                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host_penalty
                        for key in map_host_to_keys.get(host, []):
                            rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host_penalty
                else:
                    total_penalty = 0.0
                    P = Handler.CONSTRAINT_STEP_PENALTY
                    for host in target_hosts:
                        share = weights.get(host, 0.0) / total_w
                        per_host_pen = -P * share
                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host_pen
                        for key in map_host_to_keys.get(host, []):
                            rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host_pen
                        total_penalty += per_host_pen
                step_comp['constraint_penalty'] += total_penalty
                if Handler.CONSTRAINT_LAMBDA > 0.0:
                    should_log = False
                    if Handler.LAST_CONSTRAINT_LOG_TIME is None:
                        should_log = True
                    elif sim_time - Handler.LAST_CONSTRAINT_LOG_TIME >= 600:
                        should_log = True
                    if should_log:
                        Handler.LAST_CONSTRAINT_LOG_TIME = sim_time
                        print(
                            f"[REWARD] t={sim_time:.1f}s constraint penalty applied (weighted) "
                            f"lambda={Handler.CONSTRAINT_LAMBDA:.3f} deficit={Handler.SUCCESS_CONSTRAINT_DEFICIT:.4f}"
                        )

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

        # Accounting-only reward applied retroactively (hindsight patch). This is not included in
        # rewards_per_host, but we include it in step totals / core episode totals so CSV "total_reward"
        # reflects the actual reward credited to the episode.
        retro_reward_for_total = 0.0

        # Combined step-level reward: success × (1 - overhead)
        if COMBO_REWARD_ENABLE:
            try:
                if COMBO_HINDSIGHT_CREDIT:
                    # True hindsight: once the reward for a creation step is knowable, patch it back
                    # onto the transitions of that step (instead of rewarding the current step).
                    apply_step = int(step_id) - int(COMBO_HINDSIGHT_DELAY_STEPS)
                    if apply_step >= 0 and apply_step not in Handler.HINDSIGHT_COMBO_APPLIED_STEPS:
                        created = float(Handler.HINDSIGHT_STEP_CREATED.get(apply_step, 0.0) or 0.0)
                        ttl_sum = float(Handler.HINDSIGHT_STEP_TTL_SUM.get(apply_step, 0.0) or 0.0)
                        delivered_count = float(Handler.HINDSIGHT_STEP_DELIVERED.get(apply_step, 0.0) or 0.0)

                        S = 0.0
                        if created > 0.0:
                            S = min(1.0, ttl_sum / created)

                        # Overhead ratio for the same step interval
                        rel_total = float(Handler.HINDSIGHT_STEP_RELAY_TOTAL.get(apply_step, 0.0) or 0.0)
                        del_total = float(Handler.HINDSIGHT_STEP_DELIVERED_INTERVAL.get(apply_step, 0.0) or 0.0)
                        step_extra = max(0.0, rel_total - del_total)
                        denom_rel = max(1.0, rel_total)
                        O = max(0.0, min(1.0, step_extra / denom_rel))

                        # If we haven't observed origin-attribution delivery_events at all, we can't compute
                        # creation-step success yet. Fall back to interval delivered/created so reward doesn't
                        # collapse to ~0 when running with an older Java bridge.
                        if created > 0.0 and not getattr(Handler, "SAW_DELIVERY_EVENTS", False):
                            try:
                                S = max(0.0, min(1.0, float(del_total) / float(created)))
                            except Exception:
                                S = 0.0

                        # Compute once, but keep it around so we can still account for tail steps.
                        if apply_step in Handler.HINDSIGHT_PENDING_REWARDS:
                            combo_total = float(Handler.HINDSIGHT_PENDING_REWARDS.get(apply_step, 0.0) or 0.0)
                        else:
                            combo_total = 0.0
                            if COMBO_FORM == 'log':
                                import math as _m
                                term_s = _m.log(max(COMBO_EPS, S))
                                term_o = _m.log(max(COMBO_EPS, 1.0 - O))
                                combo_total = COMBO_SCALE * (COMBO_ALPHA * term_s + COMBO_BETA * term_o)
                            else:
                                combo_total = COMBO_SCALE * (
                                    (max(COMBO_EPS, S) ** COMBO_ALPHA)
                                    * (max(COMBO_EPS, 1.0 - O) ** COMBO_BETA)
                                )
                            Handler.HINDSIGHT_PENDING_REWARDS[apply_step] = float(combo_total)

                        patch_res = None
                        applied_ok = False
                        applied_n = 0
                        if TORCH_OK and Handler.TRAINING_ENABLED and Handler.AGENT is not None and hasattr(Handler.AGENT, "apply_hindsight_combo_reward"):
                            if combo_total == 0.0:
                                # Nothing to patch; mark the step as applied to avoid repeated work.
                                applied_ok = True
                            else:
                                patch_res = Handler.AGENT.apply_hindsight_combo_reward(apply_step, combo_total)
                                applied_ok = bool(isinstance(patch_res, dict) and patch_res.get("ok"))
                                if applied_ok:
                                    try:
                                        applied_n = int(patch_res.get("applied", 0) or 0)
                                    except Exception:
                                        applied_n = 0
                            if applied_ok:
                                Handler.HINDSIGHT_COMBO_APPLIED_STEPS.add(apply_step)
                                Handler.HINDSIGHT_PENDING_REWARDS.pop(apply_step, None)
                                # Only account the retro reward if we actually patched at least one transition.
                                if combo_total != 0.0 and applied_n > 0:
                                    step_comp['combo_reward'] += combo_total
                                    retro_reward_for_total += combo_total
                                    # Keep episode total reward accounting consistent even though the reward is applied retroactively.
                                    try:
                                        Handler.EP_REWARD_ACC = (Handler.EP_REWARD_ACC or 0.0) + float(combo_total)
                                    except Exception:
                                        pass
                                    # Episode accumulator for combo reward
                                    try:
                                        Handler.EP_COMBO_REWARD_SUM = float(getattr(Handler, 'EP_COMBO_REWARD_SUM', 0.0) or 0.0) + float(combo_total)
                                    except Exception:
                                        pass
                            else:
                                # Patch failed unexpectedly: fall back to current-step distribution to avoid silent rewards,
                                # and mark as applied so we don't retry forever.
                                target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
                                if target_hosts and combo_total != 0.0:
                                    per_host = combo_total / len(target_hosts)
                                    for host in target_hosts:
                                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host
                                        for key in map_host_to_keys.get(host, []):
                                            rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host
                                step_comp['combo_reward'] += combo_total
                                try:
                                    Handler.EP_COMBO_REWARD_SUM = float(getattr(Handler, 'EP_COMBO_REWARD_SUM', 0.0) or 0.0) + float(combo_total)
                                except Exception:
                                    pass
                                Handler.HINDSIGHT_COMBO_APPLIED_STEPS.add(apply_step)
                                Handler.HINDSIGHT_PENDING_REWARDS.pop(apply_step, None)
                                applied_ok = True
                        else:
                            # Torch unavailable: can't patch a past step transition. Apply for logging/heuristic runs.
                            target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
                            if target_hosts and combo_total != 0.0:
                                per_host = combo_total / len(target_hosts)
                                for host in target_hosts:
                                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host
                                    for key in map_host_to_keys.get(host, []):
                                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host
                            step_comp['combo_reward'] += combo_total
                            # Episode accumulator for combo reward
                            try:
                                Handler.EP_COMBO_REWARD_SUM = float(getattr(Handler, 'EP_COMBO_REWARD_SUM', 0.0) or 0.0) + float(combo_total)
                            except Exception:
                                pass
                            Handler.HINDSIGHT_COMBO_APPLIED_STEPS.add(apply_step)
                            Handler.HINDSIGHT_PENDING_REWARDS.pop(apply_step, None)
                            applied_ok = True

                        # Debug log for hindsight credit (creation-step reward)
                        if delivered_count > 0 or created > 0:
                            raw_sr = delivered_count / max(1.0, created)
                            if created > 0.0 and not getattr(Handler, "SAW_DELIVERY_EVENTS", False):
                                raw_sr = float(del_total) / max(1.0, created)
                            msg = (f"[HINDSIGHT] apply_step={apply_step} created={created:.0f} delivered={delivered_count:.0f} "
                                   f"raw_sr={raw_sr:.3f} ttl_weighted_S={S:.3f} O={O:.3f} combo={combo_total:.3f}")
                            if isinstance(patch_res, dict) and patch_res.get("ok"):
                                msg += (f" patched={patch_res.get('applied', 0)} hosts={patch_res.get('hosts', 0)} "
                                        f"per_host={patch_res.get('per_host', 0.0):.6f}")
                            elif not TORCH_OK:
                                msg += " patched=(torch_disabled_fallback)"
                            elif not Handler.TRAINING_ENABLED:
                                msg += " patched=(training_disabled_fallback)"
                            print(msg)
                else:
                    # Original: immediate step-level success rate shaping (no hindsight credit).
                    S = 0.0
                    if step_sr is not None and math.isfinite(step_sr):
                        S = max(0.0, min(1.0, float(step_sr)))
                    # Prefer EMA-smoothed step success-rate when available.
                    # If requested, use a TTL-weighted EMA dedicated to combo.
                    if COMBO_IMMEDIATE_TTL_WEIGHT_MODE != "none":
                        try:
                            sr_ema = getattr(Handler, "COMBO_STEP_SR_SMOOTHED", None)
                            if sr_ema is not None and math.isfinite(float(sr_ema)):
                                S = max(0.0, min(1.0, float(sr_ema)))
                        except Exception:
                            pass
                    else:
                        # Legacy behavior: use unweighted STEP_SR_SMOOTHED
                        try:
                            sr_ema = getattr(Handler, "STEP_SR_SMOOTHED", None)
                            if sr_ema is not None and math.isfinite(float(sr_ema)):
                                S = max(0.0, min(1.0, float(sr_ema)))
                        except Exception:
                            pass

                    # Overhead ratio this step (excess relays over delivered)
                    step_extra = 0.0
                    try:
                        step_extra = max(0.0, float(step_relay_total) - float(total_delivered))
                    except Exception:
                        step_extra = 0.0
                    denom_rel = max(1.0, float(step_relay_total))
                    O = 0.0
                    if denom_rel > 0.0:
                        O = max(0.0, min(1.0, step_extra / denom_rel))

                    combo_total = 0.0
                    if COMBO_FORM == 'log':
                        import math as _m
                        term_s = _m.log(max(COMBO_EPS, S))
                        term_o = _m.log(max(COMBO_EPS, 1.0 - O))
                        combo_total = COMBO_SCALE * (COMBO_ALPHA * term_s + COMBO_BETA * term_o)
                    else:
                        combo_total = COMBO_SCALE * (
                            (max(COMBO_EPS, S) ** COMBO_ALPHA)
                            * (max(COMBO_EPS, 1.0 - O) ** COMBO_BETA)
                        )

                    # Distribute across active hosts
                    target_hosts = list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
                    if target_hosts and combo_total != 0.0:
                        per_host = combo_total / len(target_hosts)
                        for host in target_hosts:
                            rewards_per_host[host] = rewards_per_host.get(host, 0.0) + per_host
                            for key in map_host_to_keys.get(host, []):
                                rewards_per_key[key] = rewards_per_key.get(key, 0.0) + per_host
                    step_comp['combo_reward'] += combo_total
                    # Episode accumulator for combo reward
                    try:
                        Handler.EP_COMBO_REWARD_SUM = float(getattr(Handler, 'EP_COMBO_REWARD_SUM', 0.0) or 0.0) + float(combo_total)
                    except Exception:
                        pass
            except Exception:
                pass

        # After all contributions: accumulate episode total and log components
        try:
            total_step_reward = sum(rewards_per_host.values()) if rewards_per_host else 0.0
            Handler.EP_REWARD_ACC = (Handler.EP_REWARD_ACC or 0.0) + float(total_step_reward)
            if STEP_REWARD_TRACKING:
                step_comp['total_hosts_reward'] = total_step_reward + retro_reward_for_total
                log_step_reward_components(sim_time, step_comp, buffer_size_mb, Handler.EP_COUNT)
                # Optional simple step log
                log_hosts = list(rewards_per_host.keys()) or list(map_host_to_keys.keys()) or list(Handler.LAST_ACTIVE_HOSTS)
                log_step_rewards(
                    sim_time,
                    rewards_per_host or {},
                    buffer_size_mb,
                    Handler.EP_COUNT,
                    total_reward_override=(total_step_reward + retro_reward_for_total),
                    num_hosts_override=len(log_hosts) if log_hosts else 0,
                )
        except Exception:
            pass

        Handler.EP_HIGH_PENALTY_SUM += step_comp.get('peer_high_pred_pen', 0.0)
        Handler.EP_LOW_PENALTY_SUM += step_comp.get('skip_low_pred_pen', 0.0)

        # Build the exact per-key rewards that will be consumed for the previous step's actions.
        # (Used for Java-side diagnostics via RLBridgeReport/RLStateReport.)
        rewards_used_kv = {}
        try:
            if TORCH_OK and Handler.AGENT is not None and getattr(Handler.AGENT, "last_actions", None):
                acted_by_host = defaultdict(list)
                for k in Handler.AGENT.last_actions.keys():
                    host = k.split('#', 1)[0] if '#' in k else ''
                    acted_by_host[host].append(k)
                for host, acted_keys in acted_by_host.items():
                    if not acted_keys:
                        continue
                    r_host = float(rewards_per_host.get(host, 0.0))
                    r_each = r_host / len(acted_keys) if len(acted_keys) > 0 else 0.0
                    for k in acted_keys:
                        try:
                            rewards_used_kv[k] = float(rewards_per_key.get(k, r_each))
                        except Exception:
                            rewards_used_kv[k] = float(r_each)
        except Exception:
            rewards_used_kv = {}

        # Update policy
        update_result = {}
        if TORCH_OK and Handler.TRAINING_ENABLED:
            try:
                intra_train_enabled = bool((not FORCE_EPISODE_END_ONLY_UPDATES) and EPISODIC_UPDATES and EPISODIC_TRAIN_INTERVAL_STEPS > 0)
                update_result = Handler.AGENT.update(
                    rewards_per_host,
                    rewards_per_key,
                    map_host_to_keys,
                    delta_limit,
                    delivery_per_key=delivery_per_key,
                    overhead_per_key=overhead_per_key,
                    done_keys=done_keys,
                    low_zone_relays=list(Handler.LOW_ZONE_RELAY_KEYS),
                    train_now=(False if FORCE_EPISODE_END_ONLY_UPDATES else (not EPISODIC_UPDATES)),
                    allow_cleanup=True,
                    allow_trim=True,
                    step_id=step_id,
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
                    f"cost_value={update_result.get('cost_value_loss', 0):.4f}, "
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
                    Handler.EP_LOSS_COST_VALUE_SUM += float(update_result.get('cost_value_loss', 0.0) or 0.0)
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
            elif (not FORCE_EPISODE_END_ONLY_UPDATES) and EPISODIC_UPDATES and EPISODIC_TRAIN_INTERVAL_STEPS > 0:
                # Episodic updates: optionally distribute training inside an episode to avoid stagnation.
                try:
                    step_i = int(step_id)
                except Exception:
                    step_i = -1
                if step_i >= 0:
                    last_i = getattr(Handler, "LAST_EPISODIC_TRAIN_STEP_ID", None)
                    if last_i is None:
                        Handler.LAST_EPISODIC_TRAIN_STEP_ID = step_i
                    else:
                        try:
                            due = (step_i - int(last_i)) >= int(EPISODIC_TRAIN_INTERVAL_STEPS)
                        except Exception:
                            due = False
                        if due and getattr(Handler.AGENT, "ready_sequences", None):
                            try:
                                mid_res = Handler.AGENT.train_ready_sequences(force=False)
                            except Exception as mid_exc:
                                mid_res = {"updated": False, "error": str(mid_exc)}
                            if isinstance(mid_res, dict) and mid_res.get("updated"):
                                Handler.LAST_EPISODIC_TRAIN_STEP_ID = step_i
                                print(f"[DEBUG] t={sim_time:.1f}s: episodic mid-episode PPO update completed!")
                                try:
                                    Handler.EP_LOSS_POLICY_SUM += float(mid_res.get('policy_loss', 0.0) or 0.0)
                                    Handler.EP_LOSS_VALUE_SUM += float(mid_res.get('value_loss', 0.0) or 0.0)
                                    Handler.EP_LOSS_COST_VALUE_SUM += float(mid_res.get('cost_value_loss', 0.0) or 0.0)
                                    Handler.EP_LOSS_ENTROPY_SUM += float(mid_res.get('entropy', 0.0) or 0.0)
                                    Handler.EP_LOSS_UPDATE_COUNT += 1
                                except Exception:
                                    pass
                                try:
                                    buffer_size = extract_buffer_size_mb(sim_id)
                                    log_update_metrics(sim_time, mid_res, buffer_size, Handler.EP_COUNT)
                                except Exception:
                                    pass
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
        resp = {"policy_id": Handler.POLICY_ID, "actions": actions, "actions_kv": flat_map, "rewards_kv": rewards_used_kv}


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


def handle_rmappo_maxprop_v1(req):
    """
    MA-SAC controller for the rmappo_maxprop_v1 protocol (MaxProp++ parameter control).
    - Per-step inference: returns 5D action vector per host.
    - Per-step transition ingestion: consumes prev_transition to build rewards.
    - Episode-end update: Java calls mode=episode_end; this triggers training.
    """

    def _to_float(val, default=0.0):
        try:
            f = float(val)
        except Exception:
            return default
        if math.isnan(f) or math.isinf(f):
            return default
        return f

    def _clamp(x, lo, hi):
        if not math.isfinite(x):
            x = lo
        if not math.isfinite(lo):
            lo = x
        if not math.isfinite(hi):
            hi = x
        if lo > hi:
            lo, hi = hi, lo
        if x < lo:
            return lo
        if x > hi:
            return hi
        return x

    def _clamp01(x):
        try:
            x = float(x)
        except Exception:
            return 0.0
        if not math.isfinite(x):
            return 0.0
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x

    def _affine01(a, lo, hi):
        """Map a in [-1,1] to [lo,hi]."""
        try:
            a = float(a)
        except Exception:
            a = 0.0
        if not math.isfinite(a):
            a = 0.0
        a = max(-1.0, min(1.0, a))
        t = 0.5 * (a + 1.0)
        if not math.isfinite(lo) or not math.isfinite(hi):
            return float(lo if math.isfinite(lo) else 0.0)
        return float(lo + t * (hi - lo))

    action_spec = req.get("action_spec") or {}
    lam_min = _to_float(action_spec.get("lambda_cost_min", 0.0), 0.0)
    lam_max = _to_float(action_spec.get("lambda_cost_max", 2.0), 2.0)
    tau_min = _to_float(action_spec.get("tau_age_min", 0.0), 0.0)
    tau_max = _to_float(action_spec.get("tau_age_max", 3600.0), 3600.0)
    beta_min = _to_float(action_spec.get("beta_x_min", 0.0), 0.0)
    beta_max = _to_float(action_spec.get("beta_x_max", 1.0), 1.0)
    kx_min = _to_float(action_spec.get("k_x_min", 0.1), 0.1)
    kx_max = _to_float(action_spec.get("k_x_max", 2.0), 2.0)
    mr_min = _to_float(action_spec.get("m_relay_min", -1e9), -1e9)
    mr_max = _to_float(action_spec.get("m_relay_max", 5.0), 5.0)

    state_batch = req.get("state_batch", []) or []
    prev_transition = req.get("prev_transition", []) or []
    actions = []

    # Episode-end: train on accumulated sequences (episodic update mode).
    mode = str(req.get("mode", "") or "").strip().lower()
    if mode == "episode_end":
        sim_id = str(req.get("sim_id", "") or "")
        try:
            sim_time_end = int(float(req.get("time", 0) or 0))
        except Exception:
            sim_time_end = 0
        buffer_size_mb = None
        try:
            bsm = req.get("buffer_size_mb", None)
            if bsm is not None and math.isfinite(float(bsm)):
                buffer_size_mb = float(bsm)
        except Exception:
            buffer_size_mb = None
        if buffer_size_mb is None:
            buffer_size_mb = extract_buffer_size_mb(sim_id)

        train_result = None
        if TORCH_OK and getattr(Handler, "MP_V1_AGENT", None) is not None and Handler.MP_V1_TRAINING_ENABLED:
            try:
                Handler.MP_V1_AGENT.finalize_all_sequences(force_terminal=True, allow_trim=False)
                train_result = Handler.MP_V1_AGENT.train_ready_sequences(force=True, consume_all=True)
            except Exception as exc:
                train_result = {"updated": False, "error": str(exc)}

        # Episode accounting + logs (match existing *_rmappo.csv formats)
        Handler.MP_V1_EP_COUNT += 1
        try:
            created = int(Handler.MP_V1_EP_CREATED_SUM or 0)
            delivered = int(Handler.MP_V1_EP_DELIVERED_SUM or 0)
            transferred = int(Handler.MP_V1_EP_TRANSFERRED_SUM or 0)
            dropped = int(Handler.MP_V1_EP_DROPPED_SUM or 0)
            total_reward_ep = float(Handler.MP_V1_EP_REWARD_ACC or 0.0)
        except Exception:
            created = delivered = transferred = dropped = 0
            total_reward_ep = 0.0

        avg_delivery_rate = None
        if created > 0:
            avg_delivery_rate = float(delivered) / float(created)

        avg_overhead = None
        if delivered > 0:
            avg_overhead = (float(transferred) - float(delivered)) / float(delivered)

        try:
            algo_tag = "masac"
            try:
                pid = getattr(getattr(Handler, "MP_V1_AGENT", None), "policy_id", "") or ""
                if str(pid).lower().startswith("rmasac"):
                    algo_tag = "rmasac"
            except Exception:
                algo_tag = "masac"
            drop_rate_metric = None
            if created > 0:
                try:
                    drop_rate_metric = float(dropped) / float(created)
                except Exception:
                    drop_rate_metric = None
            avg_energy_used_metric = None
            try:
                denom_samples = int(Handler.MP_V1_EP_HOST_SAMPLES_SUM or 0)
                if denom_samples > 0:
                    avg_energy_used_metric = float(Handler.MP_V1_EP_ENERGY_USED_SUM or 0.0) / float(denom_samples)
                    if not math.isfinite(avg_energy_used_metric):
                        avg_energy_used_metric = None
            except Exception:
                avg_energy_used_metric = None
            log_episode_reward(
                episode_num=Handler.MP_V1_EP_COUNT,
                sim_id=sim_id,
                total_reward=round(total_reward_ep, 6),
                combo_reward=round(float(Handler.MP_V1_EP_COMBO_REWARD_SUM or 0.0), 6),
                avg_delivery_rate=avg_delivery_rate,
                delivered=delivered,
                created=created,
                avg_overhead=avg_overhead,
                avg_delay=None,
                buffer_size_mb=buffer_size_mb,
                drop_rate=(None if drop_rate_metric is None else round(float(drop_rate_metric), 6)),
                avg_energy_used=(None if avg_energy_used_metric is None else round(float(avg_energy_used_metric), 6)),
                algo_tag=algo_tag,
            )
        except Exception:
            pass
        try:
            log_mp_v1_reward_parameters(
                episode_num=Handler.MP_V1_EP_COUNT,
                sim_time=sim_time_end,
                sim_id=sim_id,
                buffer_size_mb=buffer_size_mb,
                algo_tag=algo_tag,
            )
        except Exception:
            pass

        # Action diagnostics (episode-level aggregates)
        try:
            if MP_V1_ACTION_LOG_ENABLE and isinstance(getattr(Handler, "MP_V1_ACT_STATS", None), dict):
                st = Handler.MP_V1_ACT_STATS
                if int(st.get("n", 0) or 0) > 0:
                    # Ensure derived fields exist (means/stds/fracs) even if running in heuristic fallback.
                    if "lam_mean" not in st:
                        n = float(int(st.get("n", 0) or 0))
                        if n > 0:
                            def _final(prefix):
                                s = float(st.get(prefix + "_sum", 0.0) or 0.0)
                                ss = float(st.get(prefix + "_sumsq", 0.0) or 0.0)
                                mean = s / n
                                var = max(0.0, (ss / n) - mean * mean)
                                st[prefix + "_mean"] = mean
                                st[prefix + "_std"] = math.sqrt(var)
                                st[prefix + "_frac_min"] = float(st.get(prefix + "_min_cnt", 0) or 0) / n
                                st[prefix + "_frac_max"] = float(st.get(prefix + "_max_cnt", 0) or 0) / n
                            _final("lam")
                            _final("tau")
                            _final("beta")
                            _final("kx")
                            _final("mr")
                            st["mr_frac_ge0"] = float(st.get("mr_ge0_cnt", 0) or 0) / n
                    log_mp_v1_action_episode_stats(
                        episode_num=Handler.MP_V1_EP_COUNT,
                        sim_id=sim_id,
                        stats=st,
                        buffer_size_mb=buffer_size_mb,
                    )
        except Exception:
            pass

        if train_result is not None and isinstance(train_result, dict) and train_result.get("updated"):
            try:
                Handler.MP_V1_EP_LOSS_POLICY_SUM += float(train_result.get("policy_loss", 0.0) or 0.0)
                Handler.MP_V1_EP_LOSS_VALUE_SUM += float(train_result.get("value_loss", 0.0) or 0.0)
                Handler.MP_V1_EP_LOSS_COST_VALUE_SUM += float(train_result.get("cost_value_loss", 0.0) or 0.0)
                Handler.MP_V1_EP_LOSS_ENTROPY_SUM += float(train_result.get("entropy", 0.0) or 0.0)
                Handler.MP_V1_EP_LOSS_UPDATE_COUNT += 1
            except Exception:
                pass
            try:
                metrics_with_sim = dict(train_result)
                metrics_with_sim["sim_id"] = sim_id
                log_update_metrics(sim_time_end, metrics_with_sim, buffer_size_mb, Handler.MP_V1_EP_COUNT, algo_tag=algo_tag)
            except Exception:
                pass

        try:
            upd = max(1, int(Handler.MP_V1_EP_LOSS_UPDATE_COUNT or 0))
            actor_loss_avg = (Handler.MP_V1_EP_LOSS_POLICY_SUM / upd) if Handler.MP_V1_EP_LOSS_UPDATE_COUNT > 0 else None
            critic_loss_avg = (Handler.MP_V1_EP_LOSS_VALUE_SUM / upd) if Handler.MP_V1_EP_LOSS_UPDATE_COUNT > 0 else None
            cost_critic_loss_avg = (Handler.MP_V1_EP_LOSS_COST_VALUE_SUM / upd) if Handler.MP_V1_EP_LOSS_UPDATE_COUNT > 0 else None
            entropy_avg = (Handler.MP_V1_EP_LOSS_ENTROPY_SUM / upd) if Handler.MP_V1_EP_LOSS_UPDATE_COUNT > 0 else None
        except Exception:
            actor_loss_avg = critic_loss_avg = cost_critic_loss_avg = entropy_avg = None

        try:
            log_episode_losses(
                episode_num=Handler.MP_V1_EP_COUNT,
                sim_id=sim_id,
                actor_loss_avg=actor_loss_avg,
                critic_loss_avg=critic_loss_avg,
                cost_critic_loss_avg=cost_critic_loss_avg,
                updates=int(Handler.MP_V1_EP_LOSS_UPDATE_COUNT or 0),
                entropy_avg=entropy_avg,
                buffer_size_mb=buffer_size_mb,
                algo_tag=algo_tag,
            )
        except Exception:
            pass

        # Save MP v1 checkpoint (per-episode).
        if TORCH_OK and getattr(Handler, "MP_V1_AGENT", None) is not None and Handler.MP_V1_EP_COUNT > 0:
            try:
                os.makedirs(MP_V1_MODEL_DIR, exist_ok=True)
                prefix = "masac"
                try:
                    pid = getattr(getattr(Handler, "MP_V1_AGENT", None), "policy_id", "") or ""
                    if str(pid).lower().startswith("rmasac"):
                        prefix = "rmasac"
                except Exception:
                    prefix = "masac"
                if buffer_size_mb:
                    model_name = f"{prefix}_maxprop_buf{int(buffer_size_mb)}_ep{Handler.MP_V1_EP_COUNT}.pt"
                else:
                    model_name = f"{prefix}_maxprop_{sim_id or 'sim'}_ep{Handler.MP_V1_EP_COUNT}.pt"
                path = MP_V1_MODEL_PATH or os.path.join(MP_V1_MODEL_DIR, model_name)
                res = Handler.MP_V1_AGENT.save(path)
                print(f"[EPISODE_END] Saved {prefix.upper()} MaxProp++ model ep{Handler.MP_V1_EP_COUNT}: {res.get('path')}")
            except Exception as e:
                print(f"[ERROR] Failed to save MP_V1 model: {e}")

        # Reset episode accumulators
        Handler.MP_V1_EP_REWARD_ACC = 0.0
        Handler.MP_V1_EP_COMBO_REWARD_SUM = 0.0
        Handler.MP_V1_EP_CREATED_SUM = 0
        Handler.MP_V1_EP_TRANSFERRED_SUM = 0
        Handler.MP_V1_EP_DELIVERED_SUM = 0
        Handler.MP_V1_EP_DROPPED_SUM = 0
        Handler.MP_V1_EP_ENERGY_USED_SUM = 0.0
        Handler.MP_V1_EP_HOST_SAMPLES_SUM = 0
        Handler.MP_V1_EP_STEP_COUNT = 0
        Handler.MP_V1_EP_LOSS_UPDATE_COUNT = 0
        Handler.MP_V1_EP_LOSS_POLICY_SUM = 0.0
        Handler.MP_V1_EP_LOSS_VALUE_SUM = 0.0
        Handler.MP_V1_EP_LOSS_COST_VALUE_SUM = 0.0
        Handler.MP_V1_EP_LOSS_ENTROPY_SUM = 0.0
        Handler.MP_V1_LAST_EPISODIC_TRAIN_STEP_ID = None
        Handler.MP_V1_ACT_STATS = None

        if TORCH_OK and getattr(Handler, "MP_V1_AGENT", None) is not None:
            try:
                Handler.MP_V1_AGENT.reset_hidden_states(episode_start=True)
            except Exception:
                pass

        out = {
            "policy_id": getattr(getattr(Handler, "MP_V1_AGENT", None), "policy_id", "masac_maxprop_v1"),
            "status": "episode_end_ack",
            "actions": [],
        }
        if train_result is not None:
            out["train"] = train_result
        return out

    # Compute combo reward from prev_transition totals
    reward_debug = None
    total_reward = 0.0
    created = transferred = delivered = dropped = 0
    energy_used = 0.0
    n_hosts = 0
    if MP_V1_REWARD_ENABLE:
        try:
            dt = _to_float(req.get("dt", 1.0), 1.0)
            if not math.isfinite(dt) or dt <= 1e-9:
                dt = 1.0
            created = 0
            transferred = 0
            delivered = 0
            dropped = 0
            energy_used = 0.0
            n_hosts = 0
            for tr in prev_transition:
                if not isinstance(tr, dict):
                    continue
                try:
                    created += int(float(tr.get("created_cnt", 0) or 0))
                except Exception:
                    pass
                try:
                    transferred += int(float(tr.get("transferred_cnt", 0) or 0))
                except Exception:
                    pass
                try:
                    delivered += int(float(tr.get("delivered_cnt", 0) or 0))
                except Exception:
                    pass
                try:
                    dropped += int(float(tr.get("dropped", 0) or 0))
                except Exception:
                    pass
                energy_used += _to_float(tr.get("energy_used", 0.0), 0.0)
                n_hosts += 1

            S = 0.0
            if created > 0:
                S = _clamp01(float(delivered) / float(created))
            extra = max(0.0, float(transferred) - float(delivered))
            O = _clamp01(extra / max(1.0, float(transferred)))
            D = 0.0
            if created > 0:
                D = _clamp01(float(dropped) / float(created))
            e_rate_total = max(0.0, float(energy_used)) / float(dt)
            denom_hosts = float(max(1, int(n_hosts)))
            e_rate_avg = e_rate_total / denom_hosts
            e_rate_used = e_rate_total if MP_V1_ENERGY_RATE_MODE == "total" else e_rate_avg
            E = _clamp01(e_rate_used / max(1e-12, MP_V1_ENERGY_RATE_NORM))

            if MP_V1_REWARD_FORM == "log":
                import math as _m
                term_s = _m.log(max(MP_V1_REWARD_EPS, S))
                term_o = _m.log(max(MP_V1_REWARD_EPS, O))
                term_e = _m.log(max(MP_V1_REWARD_EPS, 1.0 - E))
                term_d = _m.log(max(MP_V1_REWARD_EPS, 1.0 - D))
                total = MP_V1_REWARD_SCALE * (
                    MP_V1_ALPHA_SUCCESS * term_s
                    + MP_V1_BETA_OVERHEAD * term_o
                    + MP_V1_GAMMA_ENERGY * term_e
                    + MP_V1_DELTA_DROP * term_d
                )
            else:
                total = MP_V1_REWARD_SCALE * (
                    (max(MP_V1_REWARD_EPS, S) ** MP_V1_ALPHA_SUCCESS)
                    * (max(MP_V1_REWARD_EPS, O) ** MP_V1_BETA_OVERHEAD)
                    * (max(MP_V1_REWARD_EPS, 1.0 - E) ** MP_V1_GAMMA_ENERGY)
                    * (max(MP_V1_REWARD_EPS, 1.0 - D) ** MP_V1_DELTA_DROP)
                )
            total_reward = float(total) if math.isfinite(float(total)) else 0.0
            reward_debug = {
                "total": float(total_reward),
                "S": float(S),
                "O": float(O),
                "E": float(E),
                "D": float(D),
                "form": str(MP_V1_REWARD_FORM),
                "scale": float(MP_V1_REWARD_SCALE),
                "energy_rate_total": float(e_rate_total),
                "energy_rate_avg": float(e_rate_avg),
                "energy_rate_mode": str(MP_V1_ENERGY_RATE_MODE),
                "energy_rate_norm": float(MP_V1_ENERGY_RATE_NORM),
                "created": int(created),
                "transferred": int(transferred),
                "delivered": int(delivered),
                "dropped": int(dropped),
                "energy_used": float(energy_used),
                "n_hosts": int(n_hosts),
            }
            if MP_V1_REWARD_DEBUG_PRINT:
                try:
                    step_id_dbg = int(float(req.get("step_id", -1) or -1))
                except Exception:
                    step_id_dbg = -1
                print(
                    "[MP_V1_REWARD] "
                    f"step={step_id_dbg} total={total_reward:.6f} "
                    f"S={S:.3f} O={O:.3f} E={E:.3f} D={D:.3f} "
                    f"e_rate_total={e_rate_total:.6f} e_rate_avg={e_rate_avg:.6f} "
                    f"e_mode={MP_V1_ENERGY_RATE_MODE} e_norm={MP_V1_ENERGY_RATE_NORM:.6f}"
                )
        except Exception:
            reward_debug = None
            total_reward = 0.0

    # Accumulate per-episode totals for logging
    try:
        Handler.MP_V1_EP_REWARD_ACC = float(Handler.MP_V1_EP_REWARD_ACC or 0.0) + float(total_reward or 0.0)
        Handler.MP_V1_EP_COMBO_REWARD_SUM = float(Handler.MP_V1_EP_COMBO_REWARD_SUM or 0.0) + float(total_reward or 0.0)
        Handler.MP_V1_EP_CREATED_SUM = int(Handler.MP_V1_EP_CREATED_SUM or 0) + int(created or 0)
        Handler.MP_V1_EP_TRANSFERRED_SUM = int(Handler.MP_V1_EP_TRANSFERRED_SUM or 0) + int(transferred or 0)
        Handler.MP_V1_EP_DELIVERED_SUM = int(Handler.MP_V1_EP_DELIVERED_SUM or 0) + int(delivered or 0)
        Handler.MP_V1_EP_DROPPED_SUM = int(Handler.MP_V1_EP_DROPPED_SUM or 0) + int(dropped or 0)
        Handler.MP_V1_EP_ENERGY_USED_SUM = float(Handler.MP_V1_EP_ENERGY_USED_SUM or 0.0) + float(energy_used or 0.0)
        Handler.MP_V1_EP_HOST_SAMPLES_SUM = int(Handler.MP_V1_EP_HOST_SAMPLES_SUM or 0) + int(max(0, int(n_hosts or 0)))
        Handler.MP_V1_EP_STEP_COUNT = int(Handler.MP_V1_EP_STEP_COUNT or 0) + 1
    except Exception:
        pass

    # Build obs/key batches
    obs_batch = []
    keys = []
    hosts = []
    for item in state_batch:
        if not isinstance(item, dict):
            continue
        host = str(item.get("host", "") or "")
        obs = item.get("obs") or []
        if not host or not isinstance(obs, list) or len(obs) == 0:
            continue
        hosts.append(host)
        obs_batch.append(obs)
        keys.append(f"{host}#v1")

    # Global features (shared across hosts) - used by centralized critic / transition storage
    global_features = req.get("global_features", []) or []
    global_features_batch = []
    if isinstance(global_features, list) and len(global_features) > 0:
        global_features_batch = [global_features for _ in range(len(obs_batch))]

    # Lazy-init MP v1 agent (separate from the legacy delta agent)
    if (not MP_V1_FORCE_HEURISTIC) and TORCH_OK and getattr(Handler, "MP_V1_AGENT", None) is None:
        try:
            obs_dim = int(len(obs_batch[0])) if obs_batch else 371
            gf_dim = int(len(global_features)) if isinstance(global_features, list) else 0
            if gf_dim <= 0:
                gf_dim = 5
            use_recurrent = os.environ.get(
                "MP_V1_SAC_RECURRENT", "true" if SAC_RNN_DEFAULT_ENABLE else "false"
            ).lower() in ("1", "true", "yes")
            if use_recurrent:
                Handler.MP_V1_AGENT = CTDERMASACPolicy(
                    obs_dim=obs_dim,
                    global_obs_dim=obs_dim + gf_dim,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    action_dim=5,
                )
            else:
                Handler.MP_V1_AGENT = CTDESACPolicy(
                    obs_dim=obs_dim,
                    global_obs_dim=obs_dim + gf_dim,
                    hidden_size=LSTM_HIDDEN_SIZE,
                    action_dim=5,
                )
                Handler.MP_V1_AGENT.policy_id = "masac_maxprop_v1"
            # For host-level MaxProp++ control, ingest all hosts each step by default.
            try:
                Handler.MP_V1_AGENT.step_max_sequences = int(MP_V1_STEP_MAX_SEQUENCES)
            except Exception:
                Handler.MP_V1_AGENT.step_max_sequences = 0
        except Exception as exc:
            print(f"[BOOT] Failed to init MP_V1_AGENT: {exc}")
            Handler.MP_V1_AGENT = None

    # Ingest previous step transitions (reward for last action)
    if (not MP_V1_FORCE_HEURISTIC) and TORCH_OK and getattr(Handler, "MP_V1_AGENT", None) is not None and Handler.TRAINING_ENABLED:
        try:
            map_host_to_keys = {h: [f"{h}#v1"] for h in hosts}
            per_host = total_reward / max(1, len(hosts))
            rewards_per_host = {h: per_host for h in hosts}
            rewards_per_key = {f"{h}#v1": per_host for h in hosts}
            step_id = int(float(req.get("step_id", -1) or -1))
            Handler.MP_V1_AGENT.update(
                rewards_per_host,
                rewards_per_key,
                map_host_to_keys,
                delta_limit=1.0,
                epochs=None,
                done_keys=set(),
                low_zone_relays=set(),
                delivery_per_key=None,
                overhead_per_key=None,
                train_now=(False if FORCE_EPISODE_END_ONLY_UPDATES else (not EPISODIC_UPDATES)),
                allow_cleanup=True,
                allow_trim=True,
                step_id=step_id,
                next_obs_batch=obs_batch,
                next_keys=keys,
                next_global_features=global_features_batch,
            )
            # Optional: distribute episodic training inside an episode (instead of only at episode_end).
            if (not FORCE_EPISODE_END_ONLY_UPDATES) and EPISODIC_UPDATES and EPISODIC_TRAIN_INTERVAL_STEPS > 0 and step_id >= 0:
                last_i = getattr(Handler, "MP_V1_LAST_EPISODIC_TRAIN_STEP_ID", None)
                if last_i is None:
                    Handler.MP_V1_LAST_EPISODIC_TRAIN_STEP_ID = step_id
                else:
                    try:
                        due = (int(step_id) - int(last_i)) >= int(EPISODIC_TRAIN_INTERVAL_STEPS)
                    except Exception:
                        due = False
                    if due and getattr(Handler.MP_V1_AGENT, "ready_sequences", None):
                        try:
                            mid_res = Handler.MP_V1_AGENT.train_ready_sequences(force=False)
                        except Exception as mid_exc:
                            mid_res = {"updated": False, "error": str(mid_exc)}
                        if isinstance(mid_res, dict) and mid_res.get("updated"):
                            Handler.MP_V1_LAST_EPISODIC_TRAIN_STEP_ID = step_id
                            try:
                                Handler.MP_V1_EP_LOSS_POLICY_SUM += float(mid_res.get('policy_loss', 0.0) or 0.0)
                                Handler.MP_V1_EP_LOSS_VALUE_SUM += float(mid_res.get('value_loss', 0.0) or 0.0)
                                Handler.MP_V1_EP_LOSS_COST_VALUE_SUM += float(mid_res.get('cost_value_loss', 0.0) or 0.0)
                                Handler.MP_V1_EP_LOSS_ENTROPY_SUM += float(mid_res.get('entropy', 0.0) or 0.0)
                                Handler.MP_V1_EP_LOSS_UPDATE_COUNT += 1
                            except Exception:
                                pass
                            try:
                                sim_time_now = int(float(req.get("time", 0) or 0))
                            except Exception:
                                sim_time_now = 0
                            try:
                                metrics_with_sim = dict(mid_res)
                                sim_id_mid = str(req.get("sim_id", "") or "")
                                metrics_with_sim["sim_id"] = sim_id_mid
                                # Episode number not incremented until episode_end; log under current+1.
                                ep_num = max(1, int(Handler.MP_V1_EP_COUNT or 0) + 1)
                                buf_mid = extract_buffer_size_mb(sim_id_mid)
                                log_update_metrics(sim_time_now, metrics_with_sim, buf_mid, ep_num)
                            except Exception:
                                pass
        except Exception as exc:
            print(f"[MP_V1] update failed: {exc}")

    # Produce actions for current state (uses global_features_batch prepared above)

    actions_raw = None
    if (not MP_V1_FORCE_HEURISTIC) and TORCH_OK and getattr(Handler, "MP_V1_AGENT", None) is not None:
        try:
            actions_raw = Handler.MP_V1_AGENT.act_batch(obs_batch, 1.0, keys, global_features_batch)
        except Exception as exc:
            print(f"[MP_V1] act_batch failed: {exc}")
            actions_raw = None

    # Fallback: heuristic if agent not available
    if actions_raw is None:
        tau_default = 600.0
        beta_default = 0.2
        kx_default = 1.0
        for i, host in enumerate(hosts):
            obs = obs_batch[i] if i < len(obs_batch) else []
            e_norm = _to_float(obs[0], 0.5) if len(obs) > 0 else 0.5
            buf_occ = _to_float(obs[3], 0.0) if len(obs) > 3 else 0.0
            lam = lam_min
            tau = _clamp(tau_default, tau_min, tau_max)
            beta = _clamp(beta_default, beta_min, beta_max)
            kx = _clamp(kx_default, kx_min, kx_max)
            mr = mr_min  # large negative disables relay-margin gating
            if buf_occ >= 0.85 or e_norm <= 0.20:
                lam = _clamp(lam_min + 0.5 * (lam_max - lam_min), lam_min, lam_max)
                kx = _clamp(0.7, kx_min, kx_max)
                mr = _clamp(0.2, mr_min, mr_max)
            actions.append({
                "host": host,
                "lambda_cost": float(lam),
                "tau_age": float(tau),
                "beta_x": float(beta),
                "k_x": float(kx),
                "m_relay": float(mr),
            })
            if MP_V1_ACTION_LOG_ENABLE:
                try:
                    st = Handler.MP_V1_ACT_STATS
                    if not isinstance(st, dict):
                        st = {
                            "n": 0,
                            "lam_sum": 0.0, "lam_sumsq": 0.0, "lam_min_obs": float("inf"), "lam_max_obs": float("-inf"),
                            "lam_min_cnt": 0, "lam_max_cnt": 0,
                            "tau_sum": 0.0, "tau_sumsq": 0.0, "tau_min_obs": float("inf"), "tau_max_obs": float("-inf"),
                            "tau_min_cnt": 0, "tau_max_cnt": 0,
                            "beta_sum": 0.0, "beta_sumsq": 0.0, "beta_min_obs": float("inf"), "beta_max_obs": float("-inf"),
                            "beta_min_cnt": 0, "beta_max_cnt": 0,
                            "kx_sum": 0.0, "kx_sumsq": 0.0, "kx_min_obs": float("inf"), "kx_max_obs": float("-inf"),
                            "kx_min_cnt": 0, "kx_max_cnt": 0,
                            "mr_sum": 0.0, "mr_sumsq": 0.0, "mr_min_obs": float("inf"), "mr_max_obs": float("-inf"),
                            "mr_min_cnt": 0, "mr_max_cnt": 0, "mr_ge0_cnt": 0,
                            "lam_min": float(lam_min), "lam_max": float(lam_max),
                            "tau_min": float(tau_min), "tau_max": float(tau_max),
                            "beta_min": float(beta_min), "beta_max": float(beta_max),
                            "kx_min": float(kx_min), "kx_max": float(kx_max),
                            "mr_min": float(mr_min), "mr_max": float(mr_max),
                        }
                        Handler.MP_V1_ACT_STATS = st

                    eps = 1e-12
                    st["n"] = int(st.get("n", 0) or 0) + 1

                    def _upd(prefix, v, lo, hi):
                        st[prefix + "_sum"] += float(v)
                        st[prefix + "_sumsq"] += float(v) * float(v)
                        st[prefix + "_min_obs"] = min(float(st[prefix + "_min_obs"]), float(v))
                        st[prefix + "_max_obs"] = max(float(st[prefix + "_max_obs"]), float(v))
                        if float(v) <= float(lo) + eps:
                            st[prefix + "_min_cnt"] = int(st.get(prefix + "_min_cnt", 0) or 0) + 1
                        if float(v) >= float(hi) - eps:
                            st[prefix + "_max_cnt"] = int(st.get(prefix + "_max_cnt", 0) or 0) + 1

                    _upd("lam", lam, lam_min, lam_max)
                    _upd("tau", tau, tau_min, tau_max)
                    _upd("beta", beta, beta_min, beta_max)
                    _upd("kx", kx, kx_min, kx_max)
                    _upd("mr", mr, mr_min, mr_max)
                    if float(mr) >= 0.0:
                        st["mr_ge0_cnt"] = int(st.get("mr_ge0_cnt", 0) or 0) + 1
                except Exception:
                    pass
        resp = {"policy_id": "heuristic_maxprop_v1", "actions": actions}
        if reward_debug is not None:
            resp["reward_debug"] = reward_debug
        return resp

    # Map normalized actions to parameter ranges
    for i, host in enumerate(hosts):
        a = actions_raw[i] if i < len(actions_raw) else None
        if not isinstance(a, (list, tuple)) or len(a) < 5:
            a = [0.0, 0.0, 0.0, 0.0, 0.0]
        lam = _clamp(_affine01(a[0], lam_min, lam_max), lam_min, lam_max)
        tau = _clamp(_affine01(a[1], tau_min, tau_max), tau_min, tau_max)
        beta = _clamp(_affine01(a[2], beta_min, beta_max), beta_min, beta_max)
        kx = _clamp(_affine01(a[3], kx_min, kx_max), kx_min, kx_max)
        mr = _clamp(_affine01(a[4], mr_min, mr_max), mr_min, mr_max)
        actions.append({
            "host": host,
            "lambda_cost": float(lam),
            "tau_age": float(tau),
            "beta_x": float(beta),
            "k_x": float(kx),
            "m_relay": float(mr),
        })

        # Accumulate episode-level action stats (after clamping).
        if MP_V1_ACTION_LOG_ENABLE:
            try:
                st = Handler.MP_V1_ACT_STATS
                if not isinstance(st, dict):
                    st = {
                        "n": 0,
                        "lam_sum": 0.0, "lam_sumsq": 0.0, "lam_min_obs": float("inf"), "lam_max_obs": float("-inf"),
                        "lam_min_cnt": 0, "lam_max_cnt": 0,
                        "tau_sum": 0.0, "tau_sumsq": 0.0, "tau_min_obs": float("inf"), "tau_max_obs": float("-inf"),
                        "tau_min_cnt": 0, "tau_max_cnt": 0,
                        "beta_sum": 0.0, "beta_sumsq": 0.0, "beta_min_obs": float("inf"), "beta_max_obs": float("-inf"),
                        "beta_min_cnt": 0, "beta_max_cnt": 0,
                        "kx_sum": 0.0, "kx_sumsq": 0.0, "kx_min_obs": float("inf"), "kx_max_obs": float("-inf"),
                        "kx_min_cnt": 0, "kx_max_cnt": 0,
                        "mr_sum": 0.0, "mr_sumsq": 0.0, "mr_min_obs": float("inf"), "mr_max_obs": float("-inf"),
                        "mr_min_cnt": 0, "mr_max_cnt": 0, "mr_ge0_cnt": 0,
                        "lam_min": float(lam_min), "lam_max": float(lam_max),
                        "tau_min": float(tau_min), "tau_max": float(tau_max),
                        "beta_min": float(beta_min), "beta_max": float(beta_max),
                        "kx_min": float(kx_min), "kx_max": float(kx_max),
                        "mr_min": float(mr_min), "mr_max": float(mr_max),
                    }
                    Handler.MP_V1_ACT_STATS = st

                # Use a tight epsilon since values are clamped and often exactly equal at bounds.
                eps = 1e-12
                st["n"] = int(st.get("n", 0) or 0) + 1

                def _upd(prefix, v, lo, hi):
                    st[prefix + "_sum"] += float(v)
                    st[prefix + "_sumsq"] += float(v) * float(v)
                    st[prefix + "_min_obs"] = min(float(st[prefix + "_min_obs"]), float(v))
                    st[prefix + "_max_obs"] = max(float(st[prefix + "_max_obs"]), float(v))
                    if float(v) <= float(lo) + eps:
                        st[prefix + "_min_cnt"] = int(st.get(prefix + "_min_cnt", 0) or 0) + 1
                    if float(v) >= float(hi) - eps:
                        st[prefix + "_max_cnt"] = int(st.get(prefix + "_max_cnt", 0) or 0) + 1

                _upd("lam", lam, lam_min, lam_max)
                _upd("tau", tau, tau_min, tau_max)
                _upd("beta", beta, beta_min, beta_max)
                _upd("kx", kx, kx_min, kx_max)
                _upd("mr", mr, mr_min, mr_max)
                if float(mr) >= 0.0:
                    st["mr_ge0_cnt"] = int(st.get("mr_ge0_cnt", 0) or 0) + 1
            except Exception:
                pass

    resp = {"policy_id": getattr(getattr(Handler, "MP_V1_AGENT", None), "policy_id", "masac_maxprop_v1"),
            "actions": actions}
    if reward_debug is not None:
        resp["reward_debug"] = reward_debug

    # Finalize derived action stats for the current episode (keep running aggregates; output at episode_end).
    if MP_V1_ACTION_LOG_ENABLE:
        try:
            st = Handler.MP_V1_ACT_STATS
            n = int(st.get("n", 0) or 0) if isinstance(st, dict) else 0
            if isinstance(st, dict) and n > 0:
                def _final(prefix):
                    s = float(st.get(prefix + "_sum", 0.0) or 0.0)
                    ss = float(st.get(prefix + "_sumsq", 0.0) or 0.0)
                    mean = s / float(n)
                    var = max(0.0, (ss / float(n)) - mean * mean)
                    st[prefix + "_mean"] = mean
                    st[prefix + "_std"] = math.sqrt(var)
                    st[prefix + "_frac_min"] = float(st.get(prefix + "_min_cnt", 0) or 0) / float(n)
                    st[prefix + "_frac_max"] = float(st.get(prefix + "_max_cnt", 0) or 0) / float(n)
                _final("lam")
                _final("tau")
                _final("beta")
                _final("kx")
                _final("mr")
                st["mr_frac_ge0"] = float(st.get("mr_ge0_cnt", 0) or 0) / float(n)
        except Exception:
            pass
    return resp


def main():
    main_print(f"Starting DRL server at http://{HOST}:{PORT}/infer_and_update")
    main_print("  - Legacy (non-mp_v1): R-MAPPO (PPO)")
    main_print("  - MaxProp++ (rmappo_maxprop_v1): CTDE SAC (MA-SAC)")
    main_print(f"  - PyTorch: {'enabled' if TORCH_OK else 'disabled'}")
    main_print(f"  - GRU hidden size: {LSTM_HIDDEN_SIZE}")
    main_print(f"  - GRU layers: {LSTM_NUM_LAYERS}")
    main_print(f"  - Truncated BPTT: {TRUNCATED_BPTT_LEN} steps")
    main_print(f"  - Mode: {'EVAL_ONLY' if EVAL_ONLY else 'TRAINING'}")
    if FORCE_EPISODE_END_ONLY_UPDATES:
        main_print("  - Updates: EPISODE_END_ONLY (hardcoded)")
    else:
        main_print(f"  - Updates: {'EPISODIC' if EPISODIC_UPDATES else 'STREAMING'}")
    if (not FORCE_EPISODE_END_ONLY_UPDATES) and EPISODIC_UPDATES and EPISODIC_TRAIN_INTERVAL_STEPS > 0:
        main_print(f"  - Episodic mid-episode train interval: {EPISODIC_TRAIN_INTERVAL_STEPS} steps")
    main_print(f"  - STEP_MAX_SEQUENCES={STEP_MAX_SEQUENCES} (legacy); MP_V1_STEP_MAX_SEQUENCES={MP_V1_STEP_MAX_SEQUENCES} (mp_v1)")
    main_print(
        "  - MP_v1 reward: enable={en} form={form} scale={scale} "
        "alpha={a} beta={b} gamma={g} delta={d} e_norm={enorm}"
        .format(
            en=MP_V1_REWARD_ENABLE,
            form=MP_V1_REWARD_FORM,
            scale=MP_V1_REWARD_SCALE,
            a=MP_V1_ALPHA_SUCCESS,
            b=MP_V1_BETA_OVERHEAD,
            g=MP_V1_GAMMA_ENERGY,
            d=MP_V1_DELTA_DROP,
            enorm=MP_V1_ENERGY_RATE_NORM,
        )
    )
    main_print(f"    * MP_V1_ENERGY_RATE_MODE={MP_V1_ENERGY_RATE_MODE} (avg_per_host recommended for all-host control)")
    if TORCH_OK:
        try:
            main_print(
                "  - MP_v1 SAC defaults: "
                f"gamma={SAC_DEFAULT_GAMMA} tau={SAC_DEFAULT_TAU} "
                f"actor_lr={SAC_DEFAULT_ACTOR_LR} critic_lr={SAC_DEFAULT_CRITIC_LR} alpha_lr={SAC_DEFAULT_ALPHA_LR} "
                f"batch={SAC_DEFAULT_BATCH_SIZE} replay={SAC_DEFAULT_REPLAY_SIZE} min_replay={SAC_DEFAULT_MIN_REPLAY_SIZE} "
                f"auto_alpha={SAC_DEFAULT_AUTO_ALPHA} alpha_init={SAC_DEFAULT_ALPHA_INIT} "
                f"updates_ep_end={SAC_DEFAULT_UPDATES_PER_EPISODE_END}"
            )
        except Exception:
            pass
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
            main_print(f"  - PPO LR={Handler.AGENT.lr}; value_coef={Handler.AGENT.value_coef}; cost_value_coef={Handler.AGENT.cost_value_coef}; gamma={Handler.AGENT.gamma}; gae_lambda={Handler.AGENT.gae_lambda}")
        except Exception:
            pass
    if TORCH_OK:
        try:
            upper_desc = "no upper bound" if LOG_STD_MAX is None or LOG_STD_MAX <= 0 else f"<= {LOG_STD_MAX}"
            main_print(f"  - Exploration clamp: log_std >= {Handler.AGENT.log_std_min} ({upper_desc})")
            if ENTROPY_SCHEDULE:
                entropy_schedule_desc = ", ".join(
                    f"ep{start}-{end}={value}" for start, end, value in ENTROPY_SCHEDULE
                )
                main_print(f"  - Entropy coef schedule: {entropy_schedule_desc}")
            else:
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
