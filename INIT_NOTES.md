# /init Quick Reference

## Build & Core Simulator
- `compile.bat` / `./compile.sh`: compile Java sources to `target/` with dependencies from `lib/`
- `one.bat` / `./one.sh`: launch simulator (`-b` for batch runs, settings files as trailing arguments)

## DRL Integration Workflow
- Start server: `python toolkit/drl_server.py` (override with `DRL_HOST`, `DRL_PORT`, `TRAIN_UNTIL_SECONDS`, `EVAL_ONLY`, etc.)
- Windows end-to-end: `run_with_drl.bat` (starts the server then runs The ONE)
- Automation scripts:
  - `run_buffer_experiments.ps1` - full train/eval/analyze pipeline (`-BufferSizes @("10","30","50")`)
  - `run_all_buffer_training.ps1` - parallel training across buffer sizes
- Monitoring & analysis: `python toolkit/realtime_monitor.py`, `python toolkit/quick_check.py`, `python toolkit/plot_delivery_rates.py`, `python toolkit/plot_losses.py`, `python toolkit/plot_rewards.py`
- Policy export: `python toolkit/export_policy.py`, `python toolkit/export_actor_txt.py`

## DRL Reward System Snapshot (from CLAUDE.md)
- 5M/10M/15M/20M: base `delivered * 10.0`, pressure-distance bonus or penalty when `|pressure_diff| >= 0.1`
- 25M: base `delivered * 10.0`; bonus `relayed * 5.0` if occupancy >= 0.6, penalty `relayed * -3.0` if occupancy >= 0.8
- 30M+: base `delivered * 10.0 + relayed * 10.0`, extra `relayed * 5.0` when occupancy >= 0.7
- Small buffers (<= 20M) should keep pressure rewards and train with multiple seeds for robustness

## 1000k Experiment Notes
- Automated script: `./start_1000k_experiment.ps1 -BufferSize 10`
- Manual flow:
  1. `python toolkit/drl_server.py`
  2. `./one.sh scenarios/dynamic/drl_train/drl_train_buf10_1000k.txt`
  3. `python toolkit/plot_step_rewards_1000k.py --buffer 10 --interval 10`
- Suggested environment variables: `EPISODE_SECONDS=1000000`, `TOTAL_EPISODES=1`, `SAVE_AT_SECONDS=100000`, `MODEL_DIR=models/buf10_1000k`

## Miscellaneous Reminders
- Key modules: `core/`, `movement/`, `routing/`, `report/`, `gui/`, `toolkit/`
- Java dependencies: `lib/ECLA.jar`, `lib/DTNConsoleConnection.jar`
- Python dependencies: `toolkit/requirements.txt`
- Reference this note after `/init` to stay aligned with the latest CLAUDE updates.
