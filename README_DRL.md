# DRL Live Integration (PPO, synchronous per-step)

This repo contains a live DRL (PPO) bridge between The ONE simulator (Java) and a Python server.

## Components

- Java
  - `src/report/RLBridgeReport.java`: Sends state to Python and applies per-message predictability deltas (?) returned by the DRL server.
  - `src/report/RLStateReport.java`: Logs per-step rewards and per-message state for inspection.
  - `src/routing/ProphetRouter.java`: Supports external predictability offsets via `setExternalOffset()`.
- Python
  - `toolkit/drl_server.py`: HTTP server (CPU by default) with a minimal PPO skeleton using PyTorch.

## Run (Windows)

1) Install Java (JDK 8+) and Python 3.
2) (Optional) Create venv and install PyTorch (CPU or CUDA as needed):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r toolkit/requirements.txt
   ```
3) Start the DRL server:
   ```bash
   python toolkit/drl_server.py
   ```
   Console shows: `Starting PPO server at http://localhost:5000/infer_and_update (torch=on|off)`
4) In another terminal, run the simulation:
   ```bash
   one.bat -b 1 dynamic_traffic_hml_1h_settings.txt
   ```
   Or use the helper:
   ```bash
   run_with_drl.bat
   ```

## Configuration

`dynamic_traffic_hml_1h_settings.txt` includes:
- `Report.report6 = RLBridgeReport`
- `RLBridgeReport.sampleInterval = 30` (step/reward period)
- `RLBridgeReport.windowSize = 600` (state window average)
- `RLBridgeReport.deltaLimit = 0.05` (? clamp)
- `RLBridgeReport.maxMessagesPerNode = 20` (inference cap)
- `RLBridgeReport.url = http://localhost:5000/infer_and_update`
- `RLBridgeReport.activationTime = 0` (optional delay in sim-seconds before DRL takes over; Prophet runs alone before this time)

The buffer-based heuristic overlay (`BufferLoadHeuristicReport`) accepts the same
`activationTime` knob, so you can let vanilla PRoPHET drive the network and only
enable heuristics after, e.g., `14400` seconds.

### Observation Vector (DRL)

The server expects 5-dim observations per host-destination pair:

- `[contacts_norm, pred, bufocc_mean, capacity_norm, self_buf_util]`
  - `contacts_norm`: normalized average contacts over a time window
  - `pred`: PRoPHET predictability to the destination
  - `bufocc_mean`: mean buffer occupancy derived from the shared tracker (0-1)
  - `capacity_norm`: node buffer capacity normalized by the scenario's maximum
  - `self_buf_util`: current node buffer utilization (used / total)

## Pushing to GitHub

```bash
git status
git add -A
git commit -m "drl: add PPO server + RL bridge + state/reward reports"
git remote add origin https://github.com/<YOUR-ORG>/<YOUR-REPO>.git
git push -u origin main  # or master, depending on your repo
```

> Note: ignore files include `/target/`, Python caches, and report logs. Adjust `.gitignore` as needed.

## Model Save/Load & Train-Then-Eval

- Env controls (set before starting the server):
  - `TRAIN_UNTIL_SECONDS`: train until this sim-time, then auto-save and freeze to eval-only.
  - `SAVE_AT_SECONDS`: one-shot autosave after crossing this sim-time (training continues).
  - `MODEL_DIR` / `MODEL_PATH`: checkpoint location (dir or explicit file).
  - `EVAL_ONLY=true`: start in evaluation mode (no training updates).

- Examples
  - Train 60 minutes (3600s), save, then evaluate with the frozen policy:
    ```bash
    TRAIN_UNTIL_SECONDS=3600 DRL_POLICY=ppo python toolkit/drl_server.py
    ```
  - Train for a long run (e.g., 600k s) but save once at 60k s:
    ```bash
    SAVE_AT_SECONDS=60000 DRL_POLICY=ppo python toolkit/drl_server.py
    ```
  - Manual save/load:
    ```bash
    curl -X POST http://127.0.0.1:5000/save  -d '{"path":"models/ppo_model.pt"}' -H 'Content-Type: application/json'
    curl -X POST http://127.0.0.1:5000/load  -d '{"path":"models/ppo_model.pt"}' -H 'Content-Type: application/json'
    ```

## Local R-MAPPO Inference (No HTTP Hop)

You can now run the trained recurrent actor directly inside the Java simulator for
almost realtime inference (e.g., 0.1 s sampling) without depending on the Python server.

1. **Export the actor weights** from a checkpoint produced by `drl_server_rmappo.py`:
   ```bash
   python toolkit/export_rmappo_actor.py models_rmappo/buf30_policy.pt models_rmappo/buf30_actor_local.txt
   ```
   The output is a plain text file (`arch=rmappo_gru`) that contains the embedding,
   GRU, and head weights needed for deterministic inference.
2. **Point the scenario to the local policy** by editing your `.txt` settings:
   ```
   RLBridgeReport.localPolicyPath = models_rmappo/buf30_actor_local.txt
   RLBridgeReport.sampleInterval = 0.1      # optional: 0.1 s for near-realtime actions
   RLBridgeReport.url =                     # leave empty when running fully local
   ```
   When the file carries the `rmappo_gru` flag, `RLBridgeReport` automatically loads
   `LocalRmappoPolicy`, keeps per host-destination hidden states, and bypasses the HTTP call.
3. **Run the simulator** normally. All deltas are produced locally, so the Python server
   can stay offline for evaluation-only runs. Place the exported file in `Report.reportDir`
   or any readable path; the bridge logs `policy=(local)` when the local mode is active.

> The exporter only writes the actor (policy) weights. Training should still be done via
> the Python server; after training completes, export the actor you want to evaluate
> and reconfigure `RLBridgeReport` as shown above.



