# DRL Live Integration (PPO, synchronous per-step)

This repo contains a live DRL (PPO) bridge between The ONE simulator (Java) and a Python server.

## Components

- Java
  - `src/report/RLBridgeReport.java`: Sends state to Python and applies per-message predictability deltas (Δ) returned by the DRL server.
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
- `RLBridgeReport.deltaLimit = 0.05` (Δ clamp)
- `RLBridgeReport.maxMessagesPerNode = 20` (inference cap)
- `RLBridgeReport.url = http://localhost:5000/infer_and_update`

## Outputs

- `reports/*_RLStateReport.txt`: State and per-step reward logs.
- `reports/*_RLBridgeReport.txt`: Bridge diagnostics (HTTP ok/timeouts, applied action counts).

## Pushing to GitHub

```bash
git status
git add -A
git commit -m "drl: add PPO server + RL bridge + state/reward reports"
git remote add origin https://github.com/<YOUR-ORG>/<YOUR-REPO>.git
git push -u origin main  # or master, depending on your repo
```

> Note: ignore files include `/target/`, Python caches, and report logs. Adjust `.gitignore` as needed.

