#!/usr/bin/env python3
"""
PPO-based DRL HTTP server (sync, per-step update + action inference).

Endpoint: POST http://127.0.0.1:5000/infer_and_update
Request JSON:
{
  "sim_id": "dynamic_hml_1h",
  "time": 3600,
  "delta_limit": 0.05,
  "prev_transition": [ {"host":"p1","relayed":3,"drops":1,"aborted":0,"delivered":1}, ... ],
  "state_batch": [ {"host":"p1","msg_id":"H176","dest":"w82",
                     "contacts_norm":0.12,"pred":0.5881,"bufocc_mean":0.45,
                     "capacity_norm":0.5,"self_buf_util":0.35}, ... ]
}

Response JSON:
{
  "policy_id":"ppo_v1",
  "actions":[ {"host":"p1","per_message":[{"msg_id":"H176","delta":0.01}, ...]}, ... ]
}

Notes:
- Observations are 5-dim: [contacts_norm, pred, bufocc_mean, capacity_norm, self_buf_util]
- Action is scalar delta in [-delta_limit, +delta_limit]; modeled via tanh(mean) * delta_limit (Gaussian policy optional)
- Per-step update uses weighted rewards from prev_transition: r = W_DELIVER*delivered + W_RELAY*relayed - W_DROP*drops - W_ABORT*aborted (env-configurable). Reward is divided equally among messages that received action in the previous step for that host.
- If torch is unavailable, falls back to a heuristic policy.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from collections import defaultdict
import math
import os
import random
import pathlib

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

HOST = os.environ.get("DRL_HOST", "127.0.0.1")
PORT = int(float(os.environ.get("DRL_PORT", "5000")))

# Action logging configuration
PPO_DEBUG_ACTIONS = os.environ.get("PPO_DEBUG_ACTIONS", "false").lower() == "true"

# Step-by-step reward tracking configuration
STEP_REWARD_TRACKING = os.environ.get("STEP_REWARD_TRACKING", "true").lower() == "true"

def log_step_rewards(sim_time, rewards_per_host, buffer_size_mb=None):
    """Log step-by-step rewards for tracking and graphing."""
    if not STEP_REWARD_TRACKING:
        return

    try:
        # Create reward logs directory
        log_dir = os.path.join("reports", "reward_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Determine log file based on buffer size
        if buffer_size_mb:
            log_file = os.path.join(log_dir, f"step_rewards_buf{int(buffer_size_mb)}M.csv")
        else:
            log_file = os.path.join(log_dir, "step_rewards.csv")

        # Write header if file doesn't exist
        write_header = not os.path.exists(log_file)

        with open(log_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("sim_time,total_reward,avg_reward_per_host,num_hosts\n")

            total_reward = sum(rewards_per_host.values()) if rewards_per_host else 0.0
            num_hosts = len(rewards_per_host) if rewards_per_host else 0
            avg_reward = total_reward / num_hosts if num_hosts > 0 else 0.0

            f.write(f"{sim_time},{total_reward},{avg_reward},{num_hosts}\n")
    except Exception:
        pass  # Silent fail for logging

def log_actions_for_visualization(sim_time, state_batch, actions_raw, flat_map):
    """Log actions for visualization analysis."""
    try:
        # Create action logs directory
        log_dir = os.path.join("reports", "action_logs")
        os.makedirs(log_dir, exist_ok=True)

        # Log to CSV file for visualization
        csv_file = os.path.join(log_dir, "actions.csv")

        # Write header if file doesn't exist
        write_header = not os.path.exists(csv_file)

        with open(csv_file, "a", encoding="utf-8") as f:
            if write_header:
                f.write("sim_time,host,dest,contacts_norm,freebuf_norm,pred,action_delta\n")

            for i, item in enumerate(state_batch):
                try:
                    host = str(item.get("host", ""))
                    dest = str(item.get("dest", ""))
                    c = item.get("contacts_norm", 0.0)
                    f_buf = item.get("freebuf_norm", 0.0)
                    p = item.get("pred", 0.0)
                    delta = float(actions_raw[i] if i < len(actions_raw) else 0.0)

                    f.write(f"{sim_time},{host},{dest},{c},{f_buf},{p},{delta}\n")
                except Exception:
                    continue
    except Exception as e:
        pass  # Silent fail for logging

# Training configuration - 10만초 에피소드 50번 학습 디폴트
EPISODE_SECONDS = int(float(os.environ.get("EPISODE_SECONDS", "100000")))  # 10만초 에피소드
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "50"))  # 50번 학습
SIM_END_SECONDS = int(float(os.environ.get("SIM_END_SECONDS", "5000000")))  # 500만초 (50 에피소드 * 10만초)

# Training / checkpoint control - 에피소드 단위 자동 저장
# TRAIN_UNTIL_SECONDS: 호환성을 위해 유지하되 사용 안함 (에피소드 기반 저장 사용)
TRAIN_UNTIL_SECONDS = int(float(os.environ.get("TRAIN_UNTIL_SECONDS", "0")))  # 비활성화
# SAVE_AT_SECONDS: 에피소드 기반 저장으로 대체되어 사용 안함
SAVE_AT_SECONDS = int(float(os.environ.get("SAVE_AT_SECONDS", "0")))  # 비활성화
# MODEL_DIR / MODEL_PATH: checkpoint directory / explicit file path
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.environ.get("MODEL_PATH", "")  # empty = auto-generate when saving
# EVAL_ONLY=true: disable training updates (inference only)
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() in ("1", "true", "yes")

# Reward weights (CTDE-aligned with RLStateReport)
W_DELIVER = float(os.environ.get("W_DELIVER", "1.0"))
W_DROP    = float(os.environ.get("W_DROP",    "0.0"))
W_DELAY   = float(os.environ.get("W_DELAY",   "0.0"))
CONDITIONAL_RELAY_GAIN = float(os.environ.get("CONDITIONAL_RELAY_GAIN", "1.0"))
DEFAULT_BUFFER_MB = float(os.environ.get("DEFAULT_BUFFER_MB", "60.0"))

# Relay/abort shaping constants (match RLStateReport.java)
RELAY_MIN   = float(os.environ.get("RELAY_MIN",   "-0.4"))
RELAY_MAX   = float(os.environ.get("RELAY_MAX",   "3.0"))
RELAY_SLOPE = float(os.environ.get("RELAY_SLOPE", "0.12"))
RELAY_MID   = float(os.environ.get("RELAY_MID",   "40.0"))
PRESSURE_SLOPE = float(os.environ.get("PRESSURE_SLOPE", "0.2"))
PRESSURE_MID   = float(os.environ.get("PRESSURE_MID",   "35.0"))
PRESSURE_CLIP  = float(os.environ.get("PRESSURE_CLIP",  "0.15"))
ABORT_GAIN  = float(os.environ.get("ABORT_GAIN",  "0.6"))
ABORT_SLOPE = float(os.environ.get("ABORT_SLOPE", "0.18"))
ABORT_MID   = float(os.environ.get("ABORT_MID",   "25.0"))


def buffer_weight_profile(buffer_size_mb):
    """Return (delivery_weight, relay_weight, abort_weight) for a given buffer size."""
    if buffer_size_mb is None or buffer_size_mb <= 0:
        buffer_size_mb = DEFAULT_BUFFER_MB
    sigmoid = 1.0 / (1.0 + math.exp(-RELAY_SLOPE * (buffer_size_mb - RELAY_MID)))
    relay_weight = RELAY_MIN + (RELAY_MAX - RELAY_MIN) * sigmoid * 2.0
    if relay_weight < 0.0:
        relay_weight = min(0.15, abs(relay_weight))
    abort_weight = -ABORT_GAIN / (1.0 + math.exp(ABORT_SLOPE * (buffer_size_mb - ABORT_MID)))
    return W_DELIVER, relay_weight, abort_weight


def relay_pressure_coeff(buffer_size_mb, pressure_diff):
    """Map pressure difference to [-1,1] with buffer-size-dependent gating."""
    if buffer_size_mb is None or buffer_size_mb <= 0:
        buffer_size_mb = DEFAULT_BUFFER_MB
    gate = 0.5 * (1.0 + math.tanh(PRESSURE_SLOPE * (buffer_size_mb - PRESSURE_MID)))
    if PRESSURE_CLIP > 0:
        norm = max(-1.0, min(1.0, pressure_diff / PRESSURE_CLIP))
    else:
        norm = pressure_diff
    return gate * norm

def extract_buffer_size_mb(sim_id):
    """Extract buffer size from sim_id. Returns buffer size in MB or None if not found."""
    try:
        sim_id_lower = sim_id.lower()
        if "buf" in sim_id_lower:
            # Look for patterns like "buf05", "buf10", "buf30", etc.
            import re
            match = re.search(r'buf(\d+)', sim_id_lower)
            if match:
                return float(match.group(1))
        return None
    except Exception:
        return None

class HeuristicPolicy:
    def __init__(self):
        pass

    def act_batch(self, obs_batch, delta_limit):
        # obs: list of (contacts_norm, pred, bufocc_mean, capacity_norm, self_buf_util)
        out = []
        for obs in obs_batch:
            c = float(obs[0] or 0.0)  # contacts_norm
            p = float(obs[1] or 0.0)  # pred
            buf_mean = float(obs[2] or 0.0)  # bufocc_mean
            cap_norm = float(obs[3] or 0.0)  # capacity_norm
            self_util = float(obs[4] or 0.0)  # self_buf_util

            # Simple heuristic: use self_util instead of freebuf
            base = c * max(0.0, 1.0 - self_util) * max(0.0, 1.0 - p)
            if delta_limit is not None and delta_limit > 0:
                delta = delta_limit * base
                delta = max(-delta_limit, min(delta_limit, delta))
            else:
                # Unbounded mode: return base in [-inf, +inf] (still bounded by feature ranges)
                delta = base
            out.append(delta)
        return out

    def update(self, *args, **kwargs):
        return {}


class PPOPolicy:
    def __init__(self, obs_dim=7, hidden1=128, hidden2=128):
        # Hyperparameters (env-overridable)
        self.clip_eps = float(os.environ.get("PPO_CLIP", "0.2"))
        self.value_coef = float(os.environ.get("PPO_VALUE_COEF", "0.5"))
        self.entropy_coef = float(os.environ.get("PPO_ENTROPY", "0.00"))  # 탐험 보수적: 0.015 → 0.005
        self.lr = float(os.environ.get("PPO_LR", "3e-4"))  # 학습률 복원: 2e-4 → 3e-4
        self.batch_size = int(os.environ.get("PPO_BATCH_SIZE", "128"))
        self.minibatch = int(os.environ.get("PPO_MINIBATCH", "128"))
        self.epochs = int(os.environ.get("PPO_EPOCHS", "20"))
        self.gamma = float(os.environ.get("PPO_GAMMA", "0.99"))
        self.gae_lambda = float(os.environ.get("PPO_GAE_LAMBDA", "0.95"))

        self.device = torch.device("cpu")

        # Actor: outputs mean; log_std is a learnable parameter
        # Architecture: 7-128-128-1 (uses local observation only)
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden1), nn.Tanh(),
            nn.Linear(hidden1, hidden2), nn.Tanh(),
            nn.Linear(hidden2, 1)
        ).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device))
        self.log_std_min = -3.0  # 최소 log_std 완화: -2.0 → -3.0

        # CTDE: Critic uses global state (local + global features)
        # Global state = 5 local + 10 global = 15 dimensions
        self.global_feat_dim = 10
        self.global_obs_dim = obs_dim + self.global_feat_dim  # 5 + 10 = 15
        # Architecture: 15-256-128-1 (expanded input and first hidden layer)
        self.critic = nn.Sequential(
            nn.Linear(self.global_obs_dim, hidden1 * 2), nn.Tanh(),  # 15 → 256
            nn.Linear(hidden1 * 2, hidden2), nn.Tanh(),              # 256 → 128
            nn.Linear(hidden2, 1)                                     # 128 → 1
        ).to(self.device)
        self.opt = optim.Adam(list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()), lr=self.lr)

        actor_init = os.environ.get("PPO_ACTOR_INIT", "").strip()
        if actor_init:
            try:
                state = torch.load(actor_init, map_location=self.device)
                self.actor.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[PPOPolicy] WARN: failed to load actor init {actor_init}: {e}")
        critic_init = os.environ.get("PPO_CRITIC_INIT", "").strip()
        if critic_init:
            try:
                state = torch.load(critic_init, map_location=self.device)
                self.critic.load_state_dict(state, strict=False)
            except Exception as e:
                print(f"[PPOPolicy] WARN: failed to load critic init {critic_init}: {e}")

        # Last step cache and on-policy buffer
        # CTDE: last_actions: key -> (local_obs, global_obs, pre_tanh_action, logprob, value)
        self.last_actions = {}
        self.buf_local_obs = []   # Actor uses local obs (5D)
        self.buf_global_obs = []  # Critic uses global obs (15D)
        self.buf_act_pre = []
        self.buf_logp = []
        self.buf_val = []
        self.buf_rew = []

        print(f"[PPOPolicy] Initialized CTDE mode: Actor=5D, Critic=15D (5 local + 10 global)")
        self.policy_id = "ppo_v2"

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _construct_global_obs(self, local_obs, global_features):
        """
        CTDE: Construct global observation by concatenating local obs with global features.

        Args:
            local_obs: Tensor of shape (batch_size, 5) - local observations
            global_features: Tensor of shape (batch_size, 10) - global network features (REQUIRED)

        Returns:
            Tensor of shape (batch_size, 15) - combined global observation
        """
        return torch.cat([local_obs, global_features], dim=1)

    def _tanh_gaussian_sample(self, mean):
        # CRITICAL: detach and clean mean/std completely before Normal distribution
        log_std_clamped = torch.clamp(torch.nan_to_num(self.log_std.detach()), min=self.log_std_min, max=2.0)
        std = log_std_clamped.exp()
        std = torch.clamp(std, min=1e-6, max=10.0).expand_as(mean)

        # Completely sanitize mean (detach from corrupted gradient graph)
        mean = mean.detach()
        mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))

        # Completely sanitize std
        std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1e-2))

        # Final safety check before Normal creation
        if not (torch.isfinite(mean).all() and torch.isfinite(std).all()):
            print(f"[ERROR] Still have NaN/Inf after sanitization - using safe defaults")
            mean = torch.zeros_like(mean)
            std = torch.full_like(std, 1e-2)

        try:
            normal = torch.distributions.Normal(mean, std)
            u = normal.rsample()
        except Exception as e:
            print(f"[ERROR] Normal distribution creation failed: {e}, returning zeros")
            u = torch.zeros_like(mean)

        u = torch.where(torch.isfinite(u), u, torch.zeros_like(u))
        a = torch.tanh(u)
        a = torch.where(torch.isfinite(a), a, torch.zeros_like(a))

        # log-prob with tanh correction
        logp = normal.log_prob(u) - torch.log(torch.clamp(1 - a.pow(2), min=1e-6))
        logp = torch.where(torch.isfinite(logp), logp, torch.zeros_like(logp))
        return u, a, logp.squeeze(-1)

    def _tanh_gaussian_logprob(self, mean, u):
        # Same safety measures as _tanh_gaussian_sample
        log_std_clamped = torch.clamp(torch.nan_to_num(self.log_std.detach()), min=self.log_std_min, max=2.0)
        std = log_std_clamped.exp()
        std = torch.clamp(std, min=1e-6, max=10.0).expand_as(mean)

        # Sanitize inputs
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
            print(f"[ERROR] logprob calculation failed: {e}, returning zeros")
            logp = torch.zeros_like(u)

        return logp.squeeze(-1)

    def act_batch(self, obs_batch, delta_limit, keys, global_features=None):
        """
        Generate actions for a batch of observations using CTDE.

        Args:
            obs_batch: List of local observations (5D)
            delta_limit: Action scale limit
            keys: List of keys for caching
            global_features: Global features (10D) for CTDE (optional, uses zeros if None)

        Returns:
            List of scaled actions
        """
        if len(obs_batch) == 0:
            return []

        if not torch.isfinite(self.log_std.data).all():
            self.log_std.data = torch.where(torch.isfinite(self.log_std.data), self.log_std.data, torch.zeros_like(self.log_std.data))

        local_obs = torch.nan_to_num(self._to_tensor(obs_batch))

        # Handle None or empty global_features
        if global_features is None or len(global_features) == 0:
            batch_size = local_obs.size(0)
            global_feat_tensor = torch.zeros(batch_size, self.global_feat_dim, device=self.device)
        else:
            global_feat_tensor = torch.nan_to_num(self._to_tensor(global_features))

        # Actor uses only local observations (5D)
        mean = torch.nan_to_num(self.actor(local_obs).squeeze(-1))
        if not torch.isfinite(mean).all():
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
        u, a, logp = self._tanh_gaussian_sample(mean.unsqueeze(-1))
        a = a.squeeze(-1)

        # Scale to delta space
        scale = delta_limit if (delta_limit is not None and delta_limit > 0) else 1.0
        action_scaled = (a * scale).detach()

        # CTDE: Critic uses global observation (15D = 5 local + 10 global)
        global_obs = self._construct_global_obs(local_obs, global_feat_tensor)
        val = self.critic(global_obs).squeeze(-1).detach()

        # Store last step actions per key (local_obs, global_obs, pre_tanh_action, logprob, value)
        act_list = action_scaled.cpu().tolist()
        for i, k in enumerate(keys):
            self.last_actions[k] = (local_obs[i].detach(), global_obs[i].detach(), u[i].detach(), logp[i].detach(), val[i].detach())
        return act_list

    def _add_to_buffer(self, local_obs, global_obs, u_pre, logp, val, rew):
        """Add transition to replay buffer (CTDE version)."""
        self.buf_local_obs.append(local_obs)
        self.buf_global_obs.append(global_obs)
        self.buf_act_pre.append(u_pre)
        self.buf_logp.append(logp)
        self.buf_val.append(val)
        self.buf_rew.append(torch.tensor(rew, dtype=torch.float32, device=self.device))

    def update(self, rewards_per_host, rewards_per_key, mapping_host_to_keys, delta_limit, epochs=None):
        # Build transitions from last step using per-host rewards.
        # Credit is split ONLY among keys that actually received actions in the previous step.
        any_added = False
        acted_by_host = defaultdict(list)
        for k in self.last_actions.keys():
            try:
                host = k.split('#', 1)[0]
            except Exception:
                host = ''
            acted_by_host[host].append(k)

        for host, acted_keys in acted_by_host.items():
            if not acted_keys:
                continue
            r_host = float(rewards_per_host.get(host, 0.0))
            r_each = r_host / float(len(acted_keys)) if len(acted_keys) > 0 else 0.0
            for k in acted_keys:
                # CTDE: unpack local_obs, global_obs, pre_tanh_action, logprob, value
                local_obs, global_obs, u_pre, logp, val = self.last_actions[k]
                r_key = float(rewards_per_key.get(k, r_each))
                self._add_to_buffer(local_obs, global_obs, u_pre, logp, val, r_key)
                any_added = True
        # clear last actions cache (we've consumed them)
        self.last_actions.clear()

        if not any_added:
            return {"updated": False}

        # If buffer not large enough, defer optimization
        if len(self.buf_local_obs) < self.batch_size:
            return {"updated": False, "buffer": len(self.buf_local_obs)}

        # Prepare tensors (CTDE: separate local and global obs)
        local_obs = torch.stack(self.buf_local_obs)
        global_obs = torch.stack(self.buf_global_obs)
        u_pre = torch.stack(self.buf_act_pre).squeeze(-1) if self.buf_act_pre[0].dim() > 0 else torch.stack(self.buf_act_pre)
        old_logp = torch.stack(self.buf_logp)
        val_old = torch.stack(self.buf_val)
        rew = torch.stack(self.buf_rew)

        # GAE (Generalized Advantage Estimation) for better temporal credit assignment
        batch_len = rew.size(0)
        advantages = torch.zeros_like(rew)
        returns = torch.zeros_like(rew)

        # Compute next values for bootstrapping
        # For one-step, next_val = 0 (episode boundaries)
        # In reality, we'd need to track episode boundaries, but for simplicity:
        # We'll use a simplified GAE that assumes independent transitions
        with torch.no_grad():
            for t in reversed(range(batch_len)):
                if t == batch_len - 1:
                    next_val = 0.0  # Terminal or boundary
                    next_advantage = 0.0
                else:
                    next_val = val_old[t + 1]
                    next_advantage = advantages[t + 1]

                # TD error: δ_t = r_t + γ * V(s_{t+1}) - V(s_t)
                delta = rew[t] + self.gamma * next_val - val_old[t]

                # GAE: A_t = δ_t + (γλ) * A_{t+1}
                advantages[t] = delta + self.gamma * self.gae_lambda * next_advantage

                # Returns for value function target: R_t = A_t + V(s_t)
                returns[t] = advantages[t] + val_old[t]

        # Normalize advantages for training stability
        adv = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        ret = returns

        n = local_obs.size(0)
        idx = torch.randperm(n)
        mb = self.minibatch if self.minibatch > 0 else n

        for _ in range(self.epochs):
            for start in range(0, n, mb):
                end = min(start + mb, n)
                b = idx[start:end]
                # CTDE: Actor uses local_obs, Critic uses global_obs
                b_local_obs = local_obs[b]
                b_global_obs = global_obs[b]
                b_u = u_pre[b].unsqueeze(-1)
                b_old_logp = old_logp[b]
                b_adv = adv[b]
                b_ret = ret[b]

                # Actor: uses only local observations (5D)
                mean = self.actor(b_local_obs)
                new_logp = self._tanh_gaussian_logprob(mean, b_u)
                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Critic: uses global observations (15D)
                value_pred = self.critic(b_global_obs).squeeze(-1)
                value_loss = (value_pred - b_ret).pow(2).mean()

                # approximate entropy of underlying normal (ignoring tanh)
                log_std_clamped = torch.clamp(self.log_std, min=self.log_std_min)
                std = log_std_clamped.exp()
                entropy = (0.5 + 0.5 * math.log(2 * math.pi)) + log_std_clamped
                entropy = entropy.sum()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                # Gradient clipping with tighter bound for stability
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()), 0.5)
                self.opt.step()

                # NaN safety check after parameter update
                has_nan = False
                for param in list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()):
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        has_nan = True
                        break
                if has_nan:
                    print("[ERROR] NaN/Inf detected in network parameters after update - reverting update")
                    # This update corrupted the network, ideally should reload previous weights
                    # For now, just skip and log the error
                    return {"updated": False, "error": "NaN in parameters"}

        # Diagnostic logging
        std_mean = std.mean().item()
        entropy_mean = entropy.item() / std.numel()  # per dimension
        adv_mean = adv.mean().item()
        adv_var = adv.var().item()

        # Log diagnostics to file
        try:
            import os
            os.makedirs("reports", exist_ok=True)
            with open(os.path.join("reports", "drl_policy_diagnostics.txt"), "a", encoding="utf-8") as f:
                f.write(f"UPDATE: std_mean={std_mean:.6f} entropy_mean={entropy_mean:.6f} "
                       f"adv_mean={adv_mean:.6f} adv_var={adv_var:.6f} batch_size={n}\n")
        except Exception:
            pass

        # Clear buffer after update (CTDE: clear both local and global obs)
        self.buf_local_obs.clear(); self.buf_global_obs.clear(); self.buf_act_pre.clear(); self.buf_logp.clear(); self.buf_val.clear(); self.buf_rew.clear()
        return {"updated": True, "trained_on": n, "std_mean": std_mean, "entropy": entropy_mean, "adv_var": adv_var}

    # --- Checkpoint I/O ---
    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "opt": self.opt.state_dict(),
            "meta": {
                "policy_id": self.policy_id,
                "obs_dim": 5,
            },
        }
        p = pathlib.Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            torch.save(ckpt, str(p))
            return {"ok": True, "path": str(p)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def load(self, path):
        try:
            import torch
            ckpt = torch.load(path, map_location=self.device)
            self.actor.load_state_dict(ckpt.get("actor", {}))
            self.critic.load_state_dict(ckpt.get("critic", {}))
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
            meta = ckpt.get("meta", {})
            pid = meta.get("policy_id")
            if pid:
                self.policy_id = pid
            return {"ok": True, "meta": meta}
        except Exception as e:
            return {"ok": False, "error": str(e)}


class Handler(BaseHTTPRequestHandler):
    # Initialize base policy once
    if TORCH_OK:
        AGENT = PPOPolicy()
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
                    print(f"[BOOT] Loaded model from {MODEL_PATH}")
                else:
                    print(f"[BOOT] Failed to load model from {MODEL_PATH}: {res.get('error', 'unknown')}")
            else:
                print(f"[BOOT] MODEL_PATH set but torch unavailable: {MODEL_PATH}")
        except Exception as e:
            print(f"[BOOT] Exception while loading MODEL_PATH {MODEL_PATH}: {e}")
    # Episode tracking for reward reporting
    EP_CUR = None
    EP_REWARD_ACC = 0.0
    STATE_CACHE = {}  # host#dest -> feature snapshot from previous step
    EP_COUNT = 0  # increments when simulator notifies episode end
    # Detailed reward component tracking
    EP_DELIVERED_SUM = 0.0
    EP_PRESSURE_SUM = 0.0
    EP_RELAYED_SUM = 0.0
    EP_OVERHEAD_PENALTY = 0.0
    # Observation variance tracking
    OBS_CONTACTS = []
    OBS_FREEBUF = []
    OBS_PRED = []
    # Training control
    TRAINING_ENABLED = not EVAL_ONLY
    MODEL_SAVED_AT = None  # first save time

    def do_POST(self):
        # Episode end notification: record average delivery rate per run
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
            # Episode reward: scale to delivered (rate * created)
            reward = delivered if delivered is not None else int((avg if avg==avg else 0.0) * float(created))
            Handler.EP_COUNT += 1
            try:
                # Create directory structure based on sim_id buffer size
                buffer_size = extract_buffer_size_mb(sim_id)
                if buffer_size is not None:
                    report_dir = f"reports_drl_train/buf{int(buffer_size):02d}"
                else:
                    report_dir = "reports_drl_train/unknown"
                os.makedirs(report_dir, exist_ok=True)
                with open(os.path.join(report_dir, "drl_episode_rewards.txt"), "a", encoding="utf-8") as f:
                    f.write(f"episode {Handler.EP_COUNT} reward {reward} avg_delivery_rate {avg:.6f} delivered {delivered} created {created} sim_id {sim_id}\n")

                # Write detailed diagnostic log
                with open(os.path.join(report_dir, "drl_diagnostic_log.txt"), "a", encoding="utf-8") as f:
                    # Calculate observation variances
                    import numpy as np
                    contacts_var = np.var(Handler.OBS_CONTACTS) if Handler.OBS_CONTACTS else 0.0
                    freebuf_var = np.var(Handler.OBS_FREEBUF) if Handler.OBS_FREEBUF else 0.0
                    pred_var = np.var(Handler.OBS_PRED) if Handler.OBS_PRED else 0.0

                    f.write(f"EP{Handler.EP_COUNT}: delivered_sum={Handler.EP_DELIVERED_SUM:.1f} "
                           f"pressure_sum={Handler.EP_PRESSURE_SUM:.1f} relayed_sum={Handler.EP_RELAYED_SUM:.1f} "
                           f"overhead_penalty={Handler.EP_OVERHEAD_PENALTY:.1f} "
                           f"contacts_var={contacts_var:.6f} freebuf_var={freebuf_var:.6f} pred_var={pred_var:.6f}\n")
            except Exception:
                pass

            # Save model at episode completion
            if TORCH_OK and Handler.EP_COUNT > 0:
                try:
                    os.makedirs(MODEL_DIR, exist_ok=True)

                    # Extract buffer size for cleaner filenames
                    buffer_size = extract_buffer_size_mb(sim_id)
                    if buffer_size:
                        model_name = f"buf{int(buffer_size)}_ep{Handler.EP_COUNT}_ppo.pt"
                    else:
                        model_name = f"{sim_id or 'sim'}_ep{Handler.EP_COUNT}_ppo.pt"

                    path = MODEL_PATH or os.path.join(MODEL_DIR, model_name)
                    res = Handler.AGENT.save(path)
                    print(f"[EPISODE_END] Saved model for episode {Handler.EP_COUNT} to {res.get('path', path)} (ok={res.get('ok')})")

                    # Check if training is complete
                    if Handler.EP_COUNT >= TOTAL_EPISODES:
                        Handler.TRAINING_ENABLED = False

                        if buffer_size:
                            final_name = f"buf{int(buffer_size)}_final_ep{TOTAL_EPISODES}_ppo.pt"
                        else:
                            final_name = f"{sim_id or 'sim'}_final_ep{TOTAL_EPISODES}_ppo.pt"

                        final_path = MODEL_PATH or os.path.join(MODEL_DIR, final_name)
                        final_res = Handler.AGENT.save(final_path)
                        print(f"[TRAINING_COMPLETE] Saved final model to {final_res.get('path', final_path)} after {TOTAL_EPISODES} episodes")
                except Exception as e:
                    print(f"[ERROR] Failed to save model at episode {Handler.EP_COUNT}: {e}")

            # Reset episode tracking variables for next episode
            Handler.EP_DELIVERED_SUM = 0.0
            Handler.EP_PRESSURE_SUM = 0.0
            Handler.EP_RELAYED_SUM = 0.0
            Handler.EP_OVERHEAD_PENALTY = 0.0
            Handler.OBS_CONTACTS = []
            Handler.OBS_FREEBUF = []
            Handler.OBS_PRED = []

            body = json.dumps({"ok": True, "episode": Handler.EP_COUNT}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        
        # Manual save/load endpoints
        if self.path == "/save":
            if not TORCH_OK:
                self.send_response(400); self.end_headers(); self.wfile.write(b"Torch not available")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(raw.decode("utf-8"))
            except Exception:
                req = {}
            path = req.get("path") or MODEL_PATH or os.path.join(MODEL_DIR, "ppo_model.pt")
            res = Handler.AGENT.save(path)
            body = json.dumps(res).encode("utf-8")
            self.send_response(200 if res.get("ok") else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/load":
            if not TORCH_OK:
                self.send_response(400); self.end_headers(); self.wfile.write(b"Torch not available")
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                req = json.loads(raw.decode("utf-8"))
            except Exception:
                req = {}
            path = req.get("path") or MODEL_PATH
            if not path:
                self.send_response(400); self.end_headers(); self.wfile.write(b"Missing path")
                return
            res = Handler.AGENT.load(path)
            body = json.dumps(res).encode("utf-8")
            self.send_response(200 if res.get("ok") else 500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

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

        delta_limit = float(req.get("delta_limit", 0.05))
        prev_tr = req.get("prev_transition", []) or []
        state_batch = req.get("state_batch", []) or []
        sim_time = int(float(req.get("time", 0)))
        sim_id = str(req.get("sim_id", ""))
        prev_features = Handler.STATE_CACHE or {}
        next_features = {}

        def _to_float(val, default=0.0):
            try:
                f = float(val)
            except Exception:
                return default
            if math.isnan(f) or math.isinf(f):
                return default
            return f

        # 1) Build rewards per host and per key from prev_transition (CTDE-aligned reward function)
        rewards_per_host = {}
        rewards_per_key = {}
        buffer_hint = extract_buffer_size_mb(sim_id)
        for item in prev_tr:
            try:
                host = str(item.get("host", ""))
                if not host:
                    continue
                dest = str(item.get("dest", "")) if isinstance(item, dict) else ""
                key = f"{host}#{dest}" if dest else None
                delivered = _to_float(item.get("delivered", 0.0), 0.0)
                relayed = _to_float(item.get("relayed", 0.0), 0.0)
                drops = _to_float(item.get("drops", 0.0), 0.0)
                aborted = _to_float(item.get("aborted", 0.0), 0.0)
                avg_delay = _to_float(item.get("avg_delay", 0.0), 0.0)
                buffer_size_mb = _to_float(item.get("buffer_size_mb", 0.0), 0.0)
                if buffer_size_mb <= 0:
                    buffer_bytes = _to_float(item.get("buffer_size", 0.0), 0.0)
                    if buffer_bytes > 0:
                        buffer_size_mb = buffer_bytes / (1024.0 * 1024.0)
                if buffer_size_mb <= 0:
                    buffer_size_mb = buffer_hint if buffer_hint else DEFAULT_BUFFER_MB

                feature = prev_features.get(key) if key else None
                self_util_prev = _to_float(feature.get("self_buf_util"), 0.0) if feature else 0.0
                buf_mean_prev = _to_float(feature.get("bufocc_mean"), 0.0) if feature else 0.0
                global_msg_rate = _to_float(feature.get("global_msg_change_rate"), 0.0) if feature else 0.0

                w_del, w_relay, w_abort = buffer_weight_profile(buffer_size_mb)
                reward = 0.0
                if delivered > 0:
                    reward += w_del * delivered
                if aborted > 0:
                    reward += w_abort * aborted
                if drops > 0 and W_DROP != 0.0:
                    reward -= W_DROP * drops
                if delivered > 0 and avg_delay > 0 and W_DELAY != 0.0:
                    reward -= W_DELAY * delivered * avg_delay

                mean_occ = buf_mean_prev
                if math.isnan(mean_occ) or math.isinf(mean_occ):
                    mean_occ = 0.5

                if relayed > 0:
                    if mean_occ <= NETWORK_AVG_LOW_THRESHOLD:
                        relay_bonus = w_relay * relayed
                    elif mean_occ >= NETWORK_AVG_HIGH_THRESHOLD:
                        relay_bonus = -RELAY_CRIT_MULT * w_relay * relayed
                    else:
                        relay_bonus = 0.0

                    if global_msg_rate > 0.3:
                        warning_scale = min(1.0, max(0.0, global_msg_rate))
                        relay_bonus -= H_WARNING_GAIN * warning_scale * relayed

                    if relay_bonus != 0.0:
                        reward += relay_bonus
                        Handler.EP_PRESSURE_SUM += abs(relay_bonus)

                if mean_occ >= NETWORK_AVG_HIGH_THRESHOLD:
                    reward -= RELAY_CRIT_STATIC_PENALTY

                delta_raw = item.get("rate_delta")
                if delta_raw is not None:
                    rate_delta = _to_float(delta_raw, 0.0)
                    reward += RATE_DELTA_GAIN * rate_delta

                if not math.isnan(self_util_prev):
                    flush_bonus = max(0.0, FLUSH_UTIL_THRESHOLD - self_util_prev)
                    if flush_bonus > 0:
                        reward += FLUSH_GAIN * flush_bonus

                if reward != 0.0:
                    rewards_per_host[host] = rewards_per_host.get(host, 0.0) + reward
                    if key:
                        rewards_per_key[key] = rewards_per_key.get(key, 0.0) + reward

                Handler.EP_DELIVERED_SUM += delivered
                Handler.EP_RELAYED_SUM += relayed
            except Exception:
                continue

        # 2) Prepare obs batch for current state and mapping to keys
        obs_batch = []
        global_features_batch = []  # CTDE: 10D global features per observation
        keys = []
        map_host_to_keys = defaultdict(list)
        for item in state_batch:
            try:
                host = str(item.get("host", ""))
                dest = str(item.get("dest", ""))
                c = _to_float(item.get("contacts_norm", 0.0), 0.0)
                self_util_cur = _to_float(item.get("self_buf_util"), 0.0)
                buf_mean_cur = _to_float(item.get("bufocc_mean"), 0.0)
                cap_norm_cur = _to_float(item.get("capacity_norm"), 0.0)
                freebuf_raw = item.get("freebuf_norm", None)
                if freebuf_raw is None:
                    freebuf_raw = 1.0 - self_util_cur
                f = _to_float(freebuf_raw, 0.0)
                if f < 0.0:
                    f = 0.0
                if f > 1.0:
                    f = 1.0
                p = _to_float(item.get("pred", 0.0), 0.0)
                # 5-dimensional local observation: [contacts_norm, pred, bufocc_mean, capacity_norm, self_buf_util]
                pressure_diff = _to_float(item.get("pressure_diff", self_util_cur - buf_mean_cur), 0.0)
                pressure_diff = max(-1.0, min(1.0, pressure_diff))
                if any(math.isnan(x) or math.isinf(x) for x in (c, p, buf_mean_cur, cap_norm_cur, self_util_cur, pressure_diff)):
                    continue
                rate_delta = _to_float(item.get("rate_delta", 0.0), 0.0)
                obs_batch.append([c, p, buf_mean_cur, cap_norm_cur, self_util_cur, pressure_diff, rate_delta])

                # CTDE: Extract 10D global features
                g_active_conns = _to_float(item.get("global_active_conns", 0.0), 0.0)
                g_total_msgs = _to_float(item.get("global_total_msgs", 0.0), 0.0)
                g_avg_buf = _to_float(item.get("global_avg_buf", 0.0), 0.0)
                g_avg_contacts = _to_float(item.get("global_avg_contacts", 0.0), 0.0)
                g_msg_change_rate = _to_float(item.get("global_msg_change_rate", 0.0), 0.0)
                g_msg_change_momentum = _to_float(item.get("global_msg_change_momentum", 0.0), 0.0)
                g_buf_util_change_rate = _to_float(item.get("global_buf_util_change_rate", 0.0), 0.0)
                g_buf_util_momentum = _to_float(item.get("global_buf_util_momentum", 0.0), 0.0)
                g_avg_free_buf = _to_float(item.get("global_avg_free_buf", 0.0), 0.0)
                g_high_util_frac = _to_float(item.get("global_high_util_frac", 0.0), 0.0)
                if any(math.isnan(x) or math.isinf(x) for x in (
                    g_active_conns, g_total_msgs, g_avg_buf, g_avg_contacts,
                    g_msg_change_rate, g_msg_change_momentum, g_buf_util_change_rate,
                    g_buf_util_momentum, g_avg_free_buf, g_high_util_frac
                )):
                    continue
                global_features_batch.append([
                    g_active_conns, g_total_msgs, g_avg_buf, g_avg_contacts, g_msg_change_rate,
                    g_msg_change_momentum, g_buf_util_change_rate, g_buf_util_momentum,
                    g_avg_free_buf, g_high_util_frac
                ])

                k = f"{host}#{dest}"
                keys.append(k)
                map_host_to_keys[host].append(k)

                # Collect observations for variance analysis
                Handler.OBS_CONTACTS.append(c)
                Handler.OBS_FREEBUF.append(f)
                Handler.OBS_PRED.append(p)
                next_features[k] = {
                    "self_buf_util": self_util_cur,
                    "bufocc_mean": buf_mean_cur,
                    "capacity_norm": cap_norm_cur,
                    "freebuf_norm": f,
                    "global_msg_change_rate": g_msg_change_rate,
                }
            except Exception:
                continue

        Handler.STATE_CACHE = next_features
        # 2b) Episode reward accumulation and report
        try:
            ep_idx = (sim_time // EPISODE_SECONDS) if EPISODE_SECONDS > 0 else 0
        except Exception:
            ep_idx = 0
        sum_r = sum(rewards_per_host.values()) if rewards_per_host else 0.0

        # Log step-by-step rewards
        if STEP_REWARD_TRACKING and rewards_per_host:
            buffer_size = extract_buffer_size_mb(sim_id)
            log_step_rewards(sim_time, rewards_per_host, buffer_size)
        if Handler.EP_CUR is None:
            Handler.EP_CUR = ep_idx
            Handler.EP_REWARD_ACC = 0.0
        if ep_idx == Handler.EP_CUR:
            Handler.EP_REWARD_ACC += sum_r
        else:
            try:
                # This section logs step-by-step reward accumulation (less important)
                # Will write to general reports directory for now
                os.makedirs("reports", exist_ok=True)
                with open(os.path.join("reports", "drl_step_rewards.txt"), "a", encoding="utf-8") as f:
                    f.write(f"episode {Handler.EP_CUR} reward_sum {Handler.EP_REWARD_ACC:.4f}\n")
            except Exception:
                pass
            Handler.EP_CUR = ep_idx
            Handler.EP_REWARD_ACC = sum_r

        # 3) Training / checkpoint logic
        # Note: Model saving is now handled in /episode_end endpoint for accuracy

        # Update policy only if training enabled
        if TORCH_OK:
            if Handler.TRAINING_ENABLED:
                Handler.AGENT.update(rewards_per_host, rewards_per_key, map_host_to_keys, delta_limit, epochs=2)
        else:
            Handler.AGENT.update()

        # 4) Infer actions for current batch (CTDE: pass global_features)
        if TORCH_OK:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit, keys, global_features_batch)
        else:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit)

        # 5) Group actions by host and also provide a flat map for robustness
        per_host = defaultdict(list)
        flat_map = {}
        for i, item in enumerate(state_batch):
            try:
                host = str(item.get("host", ""))
                dest = str(item.get("dest", ""))
                delta = float(actions_raw[i] if i < len(actions_raw) else 0.0)
                if delta_limit is not None and delta_limit > 0:
                    delta_c = round(max(-delta_limit, min(delta_limit, delta)), 4)
                else:
                    # Unbounded mode: don't clamp
                    delta_c = round(delta, 4)
                per_host[host].append({"dest": dest, "delta": delta_c})
                flat_map[f"{host}#{dest}"] = delta_c
            except Exception:
                continue

        actions = [{"host": h, "per_message": v} for h, v in per_host.items()]
        resp = {"policy_id": Handler.POLICY_ID, "actions": actions, "actions_kv": flat_map}

        # Log actions for visualization if enabled
        if PPO_DEBUG_ACTIONS:
            log_actions_for_visualization(req.get('time', 0), state_batch, actions_raw, flat_map)

        body = json.dumps(resp).encode("utf-8")
        try:
            # Concise per-request trace
            total_actions = sum(len(v) for v in per_host.values())
            print(f"t={req.get('time')} states={len(state_batch)} actions={total_actions} policy={Handler.POLICY_ID}")
        except Exception:
            pass
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    print(f"Starting PPO server at http://{HOST}:{PORT}/infer_and_update (torch={'on' if TORCH_OK else 'off'})")
    if EVAL_ONLY:
        print("Mode: EVAL_ONLY (no training updates)")
    elif TRAIN_UNTIL_SECONDS > 0:
        print(f"Mode: TRAIN_UNTIL_SECONDS={TRAIN_UNTIL_SECONDS} (auto-save and freeze)")
    elif SAVE_AT_SECONDS > 0:
        print(f"Mode: SAVE_AT_SECONDS={SAVE_AT_SECONDS} (one-shot save)")
    srv = HTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
RELAY_CRIT_MULT = float(os.environ.get("RELAY_CRIT_MULT", "2.0"))
H_WARNING_GAIN = float(os.environ.get("H_WARNING_GAIN", "0.4"))
RELAY_CRIT_STATIC_PENALTY = float(os.environ.get("RELAY_CRIT_STATIC_PENALTY", "1.0"))
FLUSH_UTIL_THRESHOLD = float(os.environ.get("FLUSH_UTIL_THRESHOLD", "0.30"))
FLUSH_GAIN = float(os.environ.get("FLUSH_GAIN", "0.8"))
RATE_DELTA_GAIN = float(os.environ.get("RATE_DELTA_GAIN", "10.0"))

# Network average-based relay thresholds (matching heuristic: BufferLoadHeuristicReport)
NETWORK_AVG_LOW_THRESHOLD = float(os.environ.get("NETWORK_AVG_LOW_THRESHOLD", "0.35"))
NETWORK_AVG_HIGH_THRESHOLD = float(os.environ.get("NETWORK_AVG_HIGH_THRESHOLD", "0.75"))
