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
                     "contacts_norm":0.12,"freebuf_norm":0.93,"pred":0.5881}, ... ]
}

Response JSON:
{
  "policy_id":"ppo_v1",
  "actions":[ {"host":"p1","per_message":[{"msg_id":"H176","delta":0.01}, ...]}, ... ]
}

Notes:
- Observations are 3-dim: [contacts_norm, freebuf_norm, pred]
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

HOST = "127.0.0.1"
PORT = 5000

# Epsilon-greedy configuration (overridable via environment)
EPS_POLICY = os.environ.get("DRL_POLICY", "eps_greedy").lower()  # ppo | heuristic | eps_greedy
EPS_START = float(os.environ.get("EPS_START", "0.30"))
EPS_END = float(os.environ.get("EPS_END", "0.05"))
EPISODE_SECONDS = int(float(os.environ.get("EPISODE_SECONDS", "3600")))
TOTAL_EPISODES = int(os.environ.get("TOTAL_EPISODES", "12"))
SIM_END_SECONDS = int(float(os.environ.get("SIM_END_SECONDS", "1200000")))  # single-run horizon

# Training / checkpoint control (env)
# TRAIN_UNTIL_SECONDS: 학습을 이 시뮬레이션 시간까지 수행하고 자동 저장 후 평가 모드로 전환
TRAIN_UNTIL_SECONDS = int(float(os.environ.get("TRAIN_UNTIL_SECONDS", "0")))
# SAVE_AT_SECONDS: 이 시뮬레이션 시간을 초과하면 1회 저장(학습 지속)
SAVE_AT_SECONDS = int(float(os.environ.get("SAVE_AT_SECONDS", "0")))
# MODEL_DIR / MODEL_PATH: 저장 위치 지정(디렉토리 / 파일)
MODEL_DIR = os.environ.get("MODEL_DIR", "models")
MODEL_PATH = os.environ.get("MODEL_PATH", "")  # 비어있으면 자동 네이밍 사용
# EVAL_ONLY=true: 시작부터 학습 비활성화(평가 전용)
EVAL_ONLY = os.environ.get("EVAL_ONLY", "false").lower() in ("1", "true", "yes")

# Reward weights (overridable via environment)
# Delivery reward increased by 3x (now 27.0 for better performance)
W_DELIVER = float(os.environ.get("W_DELIVER", "20.0"))
W_RELAY   = float(os.environ.get("W_RELAY",   "10.0"))
W_DROP    = float(os.environ.get("W_DROP",    "2.0"))
W_ABORT   = float(os.environ.get("W_ABORT",   "0.5"))
W_DELAY   = float(os.environ.get("W_DELAY",   "0.001"))  # penalty for high delay (direct relationship)


class HeuristicPolicy:
    def __init__(self):
        pass

    def act_batch(self, obs_batch, delta_limit):
        # obs: list of (contacts_norm, freebuf_norm, pred)
        out = []
        for c, f, p in obs_batch:
            c = float(c or 0.0); f = float(f or 0.0); p = float(p or 0.0)
            base = c * max(0.0, 1.0 - f) * max(0.0, 1.0 - p)
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
    def __init__(self, obs_dim=3, hidden=64):
        # Hyperparameters (env-overridable)
        self.clip_eps = float(os.environ.get("PPO_CLIP", "0.2"))
        self.value_coef = float(os.environ.get("PPO_VALUE_COEF", "0.5"))
        self.entropy_coef = float(os.environ.get("PPO_ENTROPY", "0.0"))
        self.lr = float(os.environ.get("PPO_LR", "3e-4"))
        self.batch_size = int(os.environ.get("PPO_BATCH_SIZE", "1024"))
        self.minibatch = int(os.environ.get("PPO_MINIBATCH", "128"))
        self.epochs = int(os.environ.get("PPO_EPOCHS", "6"))

        self.device = torch.device("cpu")

        # Actor: outputs mean; log_std is a learnable parameter
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        ).to(self.device)
        self.log_std = nn.Parameter(torch.zeros(1, device=self.device))
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        ).to(self.device)
        self.opt = optim.Adam(list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()), lr=self.lr)

        # Last step cache and on-policy buffer
        # last_actions: key -> (obs_tensor, pre_tanh_action, logprob, value)
        self.last_actions = {}
        self.buf_obs = []
        self.buf_act_pre = []
        self.buf_logp = []
        self.buf_val = []
        self.buf_rew = []

        self.policy_id = "ppo_v2"

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _tanh_gaussian_sample(self, mean):
        std = self.log_std.exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        u = normal.rsample()  # reparameterized
        a = torch.tanh(u)
        # log-prob with tanh correction
        logp = normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return u, a, logp.squeeze(-1)

    def _tanh_gaussian_logprob(self, mean, u):
        std = self.log_std.exp().expand_as(mean)
        normal = torch.distributions.Normal(mean, std)
        a = torch.tanh(u)
        logp = normal.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        return logp.squeeze(-1)

    def act_batch(self, obs_batch, delta_limit, keys):
        if len(obs_batch) == 0:
            return []
        obs = self._to_tensor(obs_batch)
        mean = self.actor(obs).squeeze(-1)
        u, a, logp = self._tanh_gaussian_sample(mean.unsqueeze(-1))
        a = a.squeeze(-1)
        # scale to delta space
        scale = delta_limit if (delta_limit is not None and delta_limit > 0) else 1.0
        action_scaled = (a * scale).detach()
        val = self.critic(obs).squeeze(-1).detach()

        # store last step actions per key
        act_list = action_scaled.cpu().tolist()
        for i, k in enumerate(keys):
            self.last_actions[k] = (obs[i].detach(), u[i].detach(), logp[i].detach(), val[i].detach())
        return act_list

    def _add_to_buffer(self, obs, u_pre, logp, val, rew):
        self.buf_obs.append(obs)
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
                obs, u_pre, logp, val = self.last_actions[k]
                r_key = float(rewards_per_key.get(k, r_each))
                self._add_to_buffer(obs, u_pre, logp, val, r_key)
                any_added = True
        # clear last actions cache (we've consumed them)
        self.last_actions.clear()

        if not any_added:
            return {"updated": False}

        # If buffer not large enough, defer optimization
        if len(self.buf_obs) < self.batch_size:
            return {"updated": False, "buffer": len(self.buf_obs)}

        # Prepare tensors
        obs = torch.stack(self.buf_obs)
        u_pre = torch.stack(self.buf_act_pre).squeeze(-1) if self.buf_act_pre[0].dim() > 0 else torch.stack(self.buf_act_pre)
        old_logp = torch.stack(self.buf_logp)
        val_old = torch.stack(self.buf_val)
        rew = torch.stack(self.buf_rew)

        # One-step advantage baseline
        adv = (rew - val_old).detach()
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = rew.detach()

        n = obs.size(0)
        idx = torch.randperm(n)
        mb = self.minibatch if self.minibatch > 0 else n

        for _ in range(self.epochs):
            for start in range(0, n, mb):
                end = min(start + mb, n)
                b = idx[start:end]
                b_obs = obs[b]
                b_u = u_pre[b].unsqueeze(-1)
                b_old_logp = old_logp[b]
                b_adv = adv[b]
                b_ret = ret[b]

                mean = self.actor(b_obs)
                new_logp = self._tanh_gaussian_logprob(mean, b_u)
                ratio = (new_logp - b_old_logp).exp()
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_pred = self.critic(b_obs).squeeze(-1)
                value_loss = (value_pred - b_ret).pow(2).mean()

                # approximate entropy of underlying normal (ignoring tanh)
                std = self.log_std.exp()
                entropy = (0.5 + 0.5 * math.log(2 * math.pi)) + self.log_std
                entropy = entropy.sum()

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor.parameters()) + [self.log_std] + list(self.critic.parameters()), 1.0)
                self.opt.step()

        # Clear buffer after update
        self.buf_obs.clear(); self.buf_act_pre.clear(); self.buf_logp.clear(); self.buf_val.clear(); self.buf_rew.clear()
        return {"updated": True, "trained_on": n}

    # --- Checkpoint I/O ---
    def save(self, path):
        ckpt = {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "log_std": self.log_std.detach().cpu(),
            "opt": self.opt.state_dict(),
            "meta": {
                "policy_id": self.policy_id,
                "obs_dim": 3,
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
    POLICY_ID = BASE_POLICY_ID if EPS_POLICY != "eps_greedy" else f"eps_greedy({BASE_POLICY_ID})"
    # Episode tracking for reward reporting
    EP_CUR = None
    EP_REWARD_ACC = 0.0
    EP_COUNT = 0  # increments when simulator notifies episode end
    # Training control
    TRAINING_ENABLED = not EVAL_ONLY
    MODEL_SAVED_AT = None  # first save time
    ONESHOT_SAVED = False

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
                os.makedirs("reports", exist_ok=True)
                with open(os.path.join("reports", "drl_episode_rewards.txt"), "a", encoding="utf-8") as f:
                    f.write(f"episode {Handler.EP_COUNT} reward {reward} avg_delivery_rate {avg:.6f} delivered {delivered} created {created} sim_id {sim_id}\n")
            except Exception:
                pass
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

        # 1) Build rewards per host from prev_transition
        rewards_per_host = {}
        for item in prev_tr:
            try:
                host = str(item.get("host", ""))
                relayed = float(item.get("relayed", 0) or 0)
                drops = float(item.get("drops", 0) or 0)
                aborted = float(item.get("aborted", 0) or 0)
                # weights aligned with Java: wRelay=+1, wDrop=−1, wAbort=−0.5
                r = relayed - drops - 0.5 * aborted
                rewards_per_host[host] = rewards_per_host.get(host, 0.0) + r
            except Exception:
                continue

        # Apply optional additional weighting if 'delivered' is present
        try:
            # Adjust rewards with configurable weights while preserving legacy base r
            has_delivered = any((isinstance(it, dict) and ('delivered' in it)) for it in prev_tr)
            if has_delivered:
                for item in prev_tr:
                    try:
                        host = str(item.get("host", ""))
                        delivered = float(item.get("delivered", 0) or 0)
                        relayed = float(item.get("relayed", 0) or 0)
                        drops = float(item.get("drops", 0) or 0)
                        aborted = float(item.get("aborted", 0) or 0)
                        avg_delay = float(item.get("avg_delay", 0) or 0)  # average delivery delay in seconds
                        
                        # Delay penalty: direct relationship - higher delay = higher penalty
                        delay_penalty = 0
                        if delivered > 0 and avg_delay > 0:
                            delay_penalty = W_DELAY * delivered * avg_delay
                        
                        r_adj = (W_DELIVER * delivered) + (W_RELAY - 1.0) * relayed - (W_DROP - 1.0) * drops - (W_ABORT - 0.5) * aborted - delay_penalty
                        rewards_per_host[host] = rewards_per_host.get(host, 0.0) + r_adj
                    except Exception:
                        continue
        except Exception:
            pass

        # Build per-key rewards map (host#dest) for finer credit assignment
        rewards_per_key = {}
        try:
            for item in prev_tr:
                host = str(item.get("host", ""))
                dest = str(item.get("dest", "")) if isinstance(item, dict) else ""
                if not dest:
                    continue
                relayed = float(item.get("relayed", 0) or 0)
                drops = float(item.get("drops", 0) or 0)
                aborted = float(item.get("aborted", 0) or 0)
                delivered = float(item.get("delivered", 0) or 0)
                avg_delay = float(item.get("avg_delay", 0) or 0)
                base = relayed - drops - 0.5 * aborted
                
                # Include delay penalty in per-key calculation
                delay_penalty = 0
                if delivered > 0 and avg_delay > 0:
                    delay_penalty = W_DELAY * delivered * avg_delay
                
                r = (W_DELIVER * delivered) + (W_RELAY * relayed) - (W_DROP * drops) - (W_ABORT * aborted) - delay_penalty if ('delivered' in item) else base
                key = f"{host}#{dest}"
                rewards_per_key[key] = rewards_per_key.get(key, 0.0) + r
        except Exception:
            pass

        # 2) Prepare obs batch for current state and mapping to keys
        obs_batch = []
        keys = []
        map_host_to_keys = defaultdict(list)
        for item in state_batch:
            try:
                host = str(item.get("host", ""))
                dest = str(item.get("dest", ""))
                c = float(item.get("contacts_norm", 0.0) or 0.0)
                f = float(item.get("freebuf_norm", 0.0) or 0.0)
                p = float(item.get("pred", 0.0) or 0.0)
                obs_batch.append([c, f, p])
                k = f"{host}#{dest}"
                keys.append(k)
                map_host_to_keys[host].append(k)
            except Exception:
                continue

        # 2b) Episode reward accumulation and report
        try:
            ep_idx = (sim_time // EPISODE_SECONDS) if EPISODE_SECONDS > 0 else 0
        except Exception:
            ep_idx = 0
        sum_r = sum(rewards_per_host.values()) if rewards_per_host else 0.0
        if Handler.EP_CUR is None:
            Handler.EP_CUR = ep_idx
            Handler.EP_REWARD_ACC = 0.0
        if ep_idx == Handler.EP_CUR:
            Handler.EP_REWARD_ACC += sum_r
        else:
            try:
                os.makedirs("reports", exist_ok=True)
                with open(os.path.join("reports", "drl_episode_rewards.txt"), "a", encoding="utf-8") as f:
                    f.write(f"episode {Handler.EP_CUR} reward_sum {Handler.EP_REWARD_ACC:.4f}\n")
            except Exception:
                pass
            Handler.EP_CUR = ep_idx
            Handler.EP_REWARD_ACC = sum_r

        # 3) Training / checkpoint logic
        # Autosave at SAVE_AT_SECONDS (one-shot)
        if TORCH_OK and (not Handler.ONESHOT_SAVED) and SAVE_AT_SECONDS > 0 and sim_time >= SAVE_AT_SECONDS:
            # Prefer user-defined path; else auto name by sim_id and time
            path = MODEL_PATH or os.path.join(MODEL_DIR, f"{sim_id or 'sim'}_t{sim_time}_ppo.pt")
            res = Handler.AGENT.save(path)
            Handler.ONESHOT_SAVED = True
            Handler.MODEL_SAVED_AT = sim_time
            try:
                print(f"[CHKPT] Saved model at t={sim_time} to {res.get('path', path)} (ok={res.get('ok')})")
            except Exception:
                pass

        # Train-until mode: after threshold, save and switch to eval-only
        if TORCH_OK and TRAIN_UNTIL_SECONDS > 0 and sim_time >= TRAIN_UNTIL_SECONDS and Handler.TRAINING_ENABLED:
            # Save once at the cutoff
            path = MODEL_PATH or os.path.join(MODEL_DIR, f"{sim_id or 'sim'}_t{TRAIN_UNTIL_SECONDS}_ppo.pt")
            res = Handler.AGENT.save(path)
            Handler.MODEL_SAVED_AT = TRAIN_UNTIL_SECONDS
            Handler.TRAINING_ENABLED = False
            try:
                print(f"[TRAIN->EVAL] Saved model at cutoff t={TRAIN_UNTIL_SECONDS} to {res.get('path', path)}; switch to eval-only")
            except Exception:
                pass

        # Update policy only if training enabled
        if TORCH_OK:
            if Handler.TRAINING_ENABLED:
                Handler.AGENT.update(rewards_per_host, rewards_per_key, map_host_to_keys, delta_limit, epochs=2)
        else:
            Handler.AGENT.update()

        # 4) Infer actions for current batch (optionally epsilon-greedy)
        if EPS_POLICY == "eps_greedy":
            # Compute epsilon schedule
            if TOTAL_EPISODES <= 1:
                # Single-episode: decay over wall-clock sim progress
                progress = 0.0
                if SIM_END_SECONDS > 0:
                    progress = max(0.0, min(1.0, float(sim_time) / float(SIM_END_SECONDS)))
                eps_now = EPS_START - (EPS_START - EPS_END) * progress
                eps_now = max(EPS_END, min(EPS_START, eps_now))
            else:
                # Multi-episode: decay per episode index
                idx_clamped = max(0, min(ep_idx, TOTAL_EPISODES - 1))
                eps_now = EPS_START - (EPS_START - EPS_END) * (idx_clamped / (TOTAL_EPISODES - 1))
                eps_now = max(EPS_END, min(EPS_START, eps_now))
            # Exploitation via base policy
            base_actions = Handler.AGENT.act_batch(obs_batch, delta_limit) if not TORCH_OK else Handler.AGENT.act_batch(obs_batch, delta_limit, keys)
            actions_raw = []
            for a in base_actions:
                if random.random() < eps_now:
                    actions_raw.append(random.uniform(-delta_limit, delta_limit))
                else:
                    actions_raw.append(a)
        else:
            actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit) if not TORCH_OK else Handler.AGENT.act_batch(obs_batch, delta_limit, keys)

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

        body = json.dumps(resp).encode("utf-8")
        try:
            # Concise per-request trace to ease debugging
            total_actions = sum(len(v) for v in per_host.values())
            extra = ""
            if EPS_POLICY == "eps_greedy":
                if TOTAL_EPISODES <= 1:
                    extra = f" eps=[{EPS_START}->{EPS_END}] progress={sim_time}/{SIM_END_SECONDS}"
                else:
                    extra = f" eps=[{EPS_START}->{EPS_END}] ep={Handler.EP_COUNT}/{TOTAL_EPISODES}"
            print(f"t={req.get('time')} states={len(state_batch)} actions={total_actions} policy={Handler.POLICY_ID}{extra}")
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
