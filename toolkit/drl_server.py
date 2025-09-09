#!/usr/bin/env python3
"""
PPO-based DRL HTTP server (sync, per-step update + action inference).

Endpoint: POST http://127.0.0.1:5000/infer_and_update
Request JSON:
{
  "sim_id": "dynamic_hml_1h",
  "time": 3600,
  "delta_limit": 0.05,
  "prev_transition": [ {"host":"p1","relayed":3,"drops":1,"aborted":0}, ... ],
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
- Synchronous per-step update using prev_transition: node-level reward is divided equally among messages that received action in previous step for that host.
- If torch is unavailable, falls back to a heuristic policy.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from collections import defaultdict
import math

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    TORCH_OK = True
except Exception:
    TORCH_OK = False

HOST = "127.0.0.1"
PORT = 5000


class HeuristicPolicy:
    def __init__(self):
        pass

    def act_batch(self, obs_batch, delta_limit):
        # obs: list of (contacts_norm, freebuf_norm, pred)
        out = []
        for c, f, p in obs_batch:
            c = float(c or 0.0); f = float(f or 0.0); p = float(p or 0.0)
            delta = delta_limit * c * max(0.0, 1.0 - f) * max(0.0, 1.0 - p)
            out.append(max(-delta_limit, min(delta_limit, delta)))
        return out

    def update(self, *args, **kwargs):
        return {}


class PPOPolicy:
    def __init__(self, obs_dim=3, hidden=64, lr=3e-4, clip_eps=0.2, value_coef=0.5, entropy_coef=0.0):
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.device = torch.device("cpu")

        self.policy = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1), nn.Tanh()  # output in [-1,1]
        ).to(self.device)
        self.value = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, 1)
        ).to(self.device)
        self.opt = optim.Adam(list(self.policy.parameters()) + list(self.value.parameters()), lr=lr)

        # last step cache: key host#msg_id -> (obs_tensor, action_tensor, logprob, value)
        self.last_actions = {}
        self.policy_id = "ppo_v1"

    def _to_tensor(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def act_batch(self, obs_batch, delta_limit, keys):
        # obs_batch: list of obs vectors; keys: parallel list of unique keys
        if len(obs_batch) == 0:
            return []
        obs = self._to_tensor(obs_batch)
        mean_raw = self.policy(obs).squeeze(-1)  # [-1,1]
        # Simple deterministic policy (tanh mean) scaled by delta_limit
        action = mean_raw.tanh() * 1.0  # already in [-1,1]
        action_scaled = action * delta_limit

        # logprob (deterministic surrogate): use log(1 - a^2 + eps) as a pseudo prior (not used strongly)
        eps = 1e-6
        logp = torch.log(1.0 - action.pow(2) + eps).sum() * 0.0  # placeholder 0
        val = self.value(obs).squeeze(-1).detach()

        act_list = action_scaled.detach().cpu().numpy().tolist()
        for i, k in enumerate(keys):
            self.last_actions[k] = (obs[i].detach(), action[i].detach(), logp if isinstance(logp, torch.Tensor) else torch.tensor(0.0), val[i])
        return act_list

    def update(self, rewards_per_host, mapping_host_to_keys, delta_limit, epochs=2):
        # Build one-step batch from last_actions and distribute rewards
        obs_list, act_list, old_logp_list, val_list, adv_list, ret_list = [], [], [], [], [], []
        gamma = 0.99
        lam = 0.95

        for host, keys in mapping_host_to_keys.items():
            if not keys:
                continue
            r_host = float(rewards_per_host.get(host, 0.0))
            r_each = r_host / float(len(keys)) if len(keys) > 0 else 0.0
            for k in keys:
                if k not in self.last_actions:
                    continue
                obs, act_raw, old_logp, val = self.last_actions[k]
                # scale raw action [-1,1] to delta space for update stability
                act = act_raw  # in [-1,1]
                # One-step target
                ret = torch.tensor(r_each, dtype=torch.float32)
                adv = ret - val.cpu()
                obs_list.append(obs)
                act_list.append(act)
                old_logp_list.append(torch.tensor(0.0))
                val_list.append(val)
                adv_list.append(adv)
                ret_list.append(ret)

        if not obs_list:
            return {"updated": False}

        obs = torch.stack(obs_list)
        act = torch.stack(act_list)
        old_logp = torch.stack(old_logp_list)
        val_old = torch.stack(val_list).detach()
        adv = torch.stack(adv_list).detach()
        ret = torch.stack(ret_list).detach()
        # normalize advantage
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(epochs):
            mean_raw = self.policy(obs).squeeze(-1)
            # deterministic surrogate: encourage mean_raw toward act
            policy_loss = ((mean_raw - act).pow(2)).mean()
            value_pred = self.value(obs).squeeze(-1)
            value_loss = (value_pred - ret).pow(2).mean()
            entropy = 0.0
            loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * (entropy if isinstance(entropy, torch.Tensor) else 0.0)
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value.parameters()), 1.0)
            self.opt.step()

        # Clear last actions that were used
        self.last_actions.clear()
        return {"updated": True}


class Handler(BaseHTTPRequestHandler):
    # Initialize policy once
    if TORCH_OK:
        AGENT = PPOPolicy()
        POLICY_ID = "ppo_v1"
    else:
        AGENT = HeuristicPolicy()
        POLICY_ID = "heuristic_v1"

    def do_POST(self):
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

        # 2) Prepare obs batch for current state and mapping to keys
        obs_batch = []
        keys = []
        map_host_to_keys = defaultdict(list)
        for item in state_batch:
            try:
                host = str(item.get("host", ""))
                msg_id = str(item.get("msg_id", ""))
                c = float(item.get("contacts_norm", 0.0) or 0.0)
                f = float(item.get("freebuf_norm", 0.0) or 0.0)
                p = float(item.get("pred", 0.0) or 0.0)
                obs_batch.append([c, f, p])
                k = f"{host}#{msg_id}"
                keys.append(k)
                map_host_to_keys[host].append(k)
            except Exception:
                continue

        # 3) Update policy from prev transitions (sync per-step)
        if TORCH_OK:
            Handler.AGENT.update(rewards_per_host, map_host_to_keys, delta_limit, epochs=2)
        else:
            Handler.AGENT.update()

        # 4) Infer actions for current batch
        actions_raw = Handler.AGENT.act_batch(obs_batch, delta_limit) if not TORCH_OK else Handler.AGENT.act_batch(obs_batch, delta_limit, keys)

        # 5) Group actions by host
        per_host = defaultdict(list)
        for i, item in enumerate(state_batch):
            try:
                host = str(item.get("host", ""))
                msg_id = str(item.get("msg_id", ""))
                delta = float(actions_raw[i] if i < len(actions_raw) else 0.0)
                per_host[host].append({"msg_id": msg_id, "delta": round(max(-delta_limit, min(delta_limit, delta)), 4)})
            except Exception:
                continue

        actions = [{"host": h, "per_message": v} for h, v in per_host.items()]
        resp = {"policy_id": Handler.POLICY_ID, "actions": actions}

        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    print(f"Starting PPO server at http://{HOST}:{PORT}/infer_and_update (torch={'on' if TORCH_OK else 'off'})")
    srv = HTTPServer((HOST, PORT), Handler)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
