# DRL Agent Specification

This note captures the current DRL integration as implemented in `toolkit/drl_server.py` and the matching Java bridge classes. It reflects the code as it exists today (see commit history for updates) and is meant to be a concise reference for experiments and future papers.

## State Space
- **Per message features** (5 floats, see `Handler.do_POST` in `toolkit/drl_server.py:742`):
  1. `contacts_norm` – recent contact rate normalised to `[0, 1]`.
  2. `pred` – current Prophet predictability toward the destination.
  3. `bufocc_mean` – average buffer occupancy of neighbours.
  4. `capacity_norm` – neighbour-send capacity normalised to `[0, 1]`.
  5. `self_buf_util` – local buffer utilisation.
- Extra context cached per `host#dest` for reward shaping (`STATE_CACHE` stores `self_buf_util`, `bufocc_mean`, `capacity_norm`, `freebuf_norm`).
- Each POST batches observations for all pending messages; the order of `state_batch` matches the order of returned actions.

## Action Space
- The PPO actor emits a scalar per message (`toolkit/drl_server.py:303-330`).
- Actions are sampled from a Gaussian with learnable mean and log-std, passed through `tanh`, and scaled by the scenario’s `delta_limit` (default `0.05`, but overridable via environment or request).
- Final action domain: `[-delta_limit, +delta_limit]`. Java applies this as an additive adjustment to Prophet predictability for that message.
- If PyTorch is unavailable (`torch=off`), the server falls back to `HeuristicPolicy` (same range, deterministic heuristic).

## Reward Structure
Reward shaping now uses buffer-dependent weights (`buffer_weight_profile`) and a pressure coefficient (`relay_pressure_coeff`) inside `Handler.do_POST` (`toolkit/drl_server.py:696-785`):

1. **Weight profile:** for each simulation ID, the buffer size `B` is inferred (`extract_buffer_size_mb`).  
  `buffer_weight_profile(B)` returns three scalars:
  - `w_del(B)` – 현재 1.0 (전달 성공은 항상 +1).
  - `w_relay(B)` – `RELAY_MIN + (RELAY_MAX - RELAY_MIN) / (1 + exp(-RELAY_SLOPE * (B - RELAY_MID)))`; 기본 파라미터 (`RELAY_MIN=-0.4`, `RELAY_MAX=3.0`, `RELAY_SLOPE=0.12`, `RELAY_MID=40.0`)에서는 5 MB≈−0.35, 25 MB≈0.08, 40 MB≈1.30, 50 MB≈2.21, 100 MB≈3.00으로 버퍼가 커질수록 보상이 빠르게 증가한다.
  - `w_abort(B)` – `-ABORT_GAIN / (1 + exp(ABORT_SLOPE * (B - ABORT_MID)))`; 기본값 (`ABORT_GAIN=0.8`, `ABORT_SLOPE=0.18`, `ABORT_MID=25.0`)에서는 10 MB≈−0.75, 25 MB≈−0.40, 40 MB≈−0.05, 60 MB 이후에는 ≈0으로 수렴한다.
  Fallback weights `W_DELIVER`, `W_RELAY`, `W_ABORT` apply only when the buffer size cannot be parsed.

```
B(MB)  w_relay  w_abort  gate
  5     -0.350   -0.779   0.000
 10     -0.310   -0.750   0.000
 15     -0.239   -0.687   0.000
 20     -0.117   -0.569   0.002
 25      0.082   -0.400   0.018
 30      0.387   -0.231   0.119
 35      0.805   -0.113   0.500
 40      1.300   -0.050   0.881
 50      2.213   -0.009   0.998
 60      2.717   -0.001   1.000
 70      2.910   -0.000   1.000
 80      2.972   -0.000   1.000
 90      2.992   -0.000   1.000
100      2.997   -0.000   1.000
```

2. **Reward attribution per transition item (`prev_transition`):**
   - Delivery: `reward += w_del(B) * delivered`.
   - Abort: `reward += w_abort(B) * aborted` (negative when buffer is tight).
   - Relay: `reward += w_relay(B) * relayed + CONDITIONAL_RELAY_GAIN * relay_pressure_coeff(B, pressure_diff) * relayed`.
     - `pressure_diff` = `self_buf_util_prev - bufocc_mean_prev` from cached state.
     - `relay_pressure_coeff` clamps the pressure diff to [-1, 1] (using `PRESSURE_CLIP`) and multiplies it by a sigmoid gate `0.5*(1+tanh(PRESSURE_SLOPE*(B-PRESSURE_MID)))`, so mid-sized buffers rely on pressure cues while large buffers approach unconditional relay rewards.
   - Drops are no longer part of the reward.

3. **Tracking:** delivery, relay, pressure and abort contributions are accumulated into `Handler.EP_*` fields for diagnostics.

4. **Per-key distribution:** the same scalar reward is aggregated for both `rewards_per_host[host]` and `rewards_per_key[host#dest]`, which feed the PPO advantage calculation.

5. **Episodes:** `/episode_end` (`toolkit/drl_server.py:525-606`) now logs both delivery-based reward and the accumulated PPO reward (`ppo_reward`), saves checkpoints (`models/bufXX_epN_ppo.pt`), and stops training after `TOTAL_EPISODES` (default `50`).

## Training Loop
- PPO policy/value networks: MLP 5→128→128→1 with `tanh` activations (`toolkit/drl_server.py:256-280`).
- Hyperparameters (default, configurable via env vars):
  - `PPO_CLIP=0.2`, `PPO_LR=3e-4`, `PPO_BATCH_SIZE=1024`, `PPO_MINIBATCH=128`, `PPO_EPOCHS=6`, `PPO_ENTROPY=0.005`, `PPO_VALUE_COEF=0.5`.
- On each call to `/infer_and_update`, transitions are buffered; once the buffer reaches `PPO_BATCH_SIZE`, PPO optimisation runs and the buffer is cleared (`toolkit/drl_server.py:331-421`).
- Entropy regularisation and gradient norm clipping (1.0) guard stability.

## Environment Toggles
- `DRL_PORT`, `DRL_HOST`: server bind address.
- `MODEL_DIR`, `MODEL_PATH`: checkpoint handling.
- `TRAIN_UNTIL_SECONDS`, `EVAL_ONLY`: training schedule control.
- Reward scaling can be tuned via `RELAY_MIN`, `RELAY_MAX`, `RELAY_SLOPE`, `RELAY_MID`, `CONDITIONAL_RELAY_GAIN`, `ABORT_GAIN`, `ABORT_SLOPE`, `ABORT_MID`, `PRESSURE_SLOPE`, `PRESSURE_MID`, `PRESSURE_CLIP`, `DEFAULT_BUFFER_MB`, 등. Fallback static weights remain available through `W_DELIVER`, `W_RELAY`, `W_ABORT`.
- Logging options: `STEP_REWARD_TRACKING`, `PPO_DEBUG_ACTIONS`.

## Java Bridge Notes
- The ONE simulator reports per-step stats through `RLBridgeReport` and acknowledges episodes via `RLStateReport`.
- Action deltas returned by the server are applied inside Prophet router with per-message adjustments, respecting the scenario’s `delta_limit`.

For further tweaks (e.g., continuous buffer-dependent scaling), plan out new helper functions in `toolkit/drl_server.py` and ensure the Java bridge conveys any additional fields needed from the environment.***
