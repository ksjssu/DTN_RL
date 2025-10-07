# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Build and Compilation
- **Compile the project**: `compile.bat` (Windows) or `./compile.sh` (Linux/Unix)
  - Compiles all Java source files from `src/` to `target/` directory
  - Includes necessary JAR dependencies from `lib/` (ECLA.jar, DTNConsoleConnection.jar)
  - Copies GUI resources to target directory

### Running the Simulator
- **Run with GUI**: `one.bat` (Windows) or `./one.sh` (Linux/Unix)
- **Run in batch mode**: `./one.sh -b <runcount>` or `./one.sh -b <start>:<end>`
- **Specify settings**: `./one.sh [settings_files]`

### DRL Integration Commands
- **Run with DRL integration**: `run_with_drl.bat` (starts DRL server automatically and runs simulation)
- **Start DRL server manually**: `python toolkit/drl_server.py`
- **Install Python dependencies**: `pip install -r toolkit/requirements.txt`
- **Run comprehensive buffer experiments**: `.\run_buffer_experiments.ps1` (complete training→evaluation→analysis pipeline)
- **Run all buffer training**: `.\run_all_buffer_training.ps1` (parallel training across buffer sizes)
- **Monitor training progress**: `python toolkit/realtime_monitor.py` or `python toolkit/quick_check.py`
- **Export trained policy**: `python toolkit/export_policy.py` or `python toolkit/export_actor_txt.py`

### Example Usage
```bash
# Compile the project
./compile.sh

# Run with default settings and GUI
./one.sh

# Run in batch mode for 10 runs
./one.sh -b 10

# Run with custom settings
./one.sh example_settings/epidemic_settings.txt

# Run with DRL integration (Windows)
run_with_drl.bat

# Start DRL server manually
python toolkit/drl_server.py

# Run DRL with environment variables for training control
TRAIN_UNTIL_SECONDS=3600 python toolkit/drl_server.py

# Monitor training progress in real-time
python toolkit/realtime_monitor.py

# Quick check of training status
python toolkit/quick_check.py

# Run comprehensive buffer size experiments (training + evaluation + analysis)
.\run_buffer_experiments.ps1 -BufferSizes @("10", "30", "50")

# Analyze results and generate plots
python toolkit/plot_delivery_rates.py
python toolkit/plot_losses.py
python toolkit/aggregate_delivery_rates.py
```

## Project Architecture

### Core Components
The ONE simulator is organized into several key packages:

- **`core/`** - Fundamental simulation classes:
  - `DTNSim.java` - Main entry point and simulation coordinator
  - `DTNHost.java` - Represents network nodes/devices
  - `Message.java` - Message objects passed between nodes
  - `Connection.java` - Network connections between hosts
  - `World.java` - Simulation world containing all hosts
  - `Settings.java` - Configuration management
  - `NetworkInterface.java` - Network interface abstraction

- **`movement/`** - Node movement models (RandomWalk, MapBasedMovement, etc.)
- **`routing/`** - DTN routing protocols (Epidemic, Prophet, Spray and Wait, etc.)
- **`interfaces/`** - Network interface implementations (SimpleBroadcastInterface, etc.)
- **`applications/`** - Application layer protocols (PingApplication, etc.)
- **`report/`** - Simulation result reporters and analyzers
- **`gui/`** - Graphical user interface components
- **`input/`** - External data input handling

### Configuration System
The simulator uses text-based configuration files:
- `default_settings.txt` - Default simulation parameters
- `example_settings/` - Pre-configured scenarios for different use cases
- Settings use hierarchical naming with dots (e.g., `Scenario.endTime`, `btInterface.transmitRange`)

### Extension Points
The ONE is designed for extensibility through:
- **Movement models** - Implement new node mobility patterns
- **Routing protocols** - Add new DTN routing algorithms  
- **Network interfaces** - Create new communication interface types
- **Applications** - Build application-layer protocols
- **Reports** - Generate custom simulation metrics and outputs

### Data Flow
1. `DTNSim` initializes the simulation from settings
2. `World` manages all `DTNHost` instances
3. Each `DTNHost` has movement model, routing protocol, and network interfaces
4. Messages are created by applications and routed between hosts
5. Reports collect statistics and generate output

### Key Libraries
- `ECLA.jar` - External library dependency
- `DTNConsoleConnection.jar` - Console connection functionality

### Toolkit
The `toolkit/` directory contains Python scripts for DRL analysis and original Perl scripts:
- **DRL Analysis**: `plot_delivery_rates.py`, `plot_losses.py`, `plot_rewards.py`, `aggregate_delivery_rates.py`
- **DRL Monitoring**: `realtime_monitor.py`, `quick_check.py`, `simple_monitor.py`, `ultra_monitor.py`
- **DRL Policy Export**: `export_policy.py`, `export_actor_txt.py`
- **Advanced Analysis**: `buffer_strategy_analyzer.py`, `analyze_catastrophic_forgetting.py`, `inference_quality_analyzer.py`
- **Original Analysis**: ccdfPlotter.pl, getStats.pl (data analysis and plotting)
- **Format conversion**: dtnsim2parser.pl, dieselnetConverter.pl
- **Visualization**: messageLocationAnimator.pl, `visualize_actions.py`

This simulator is primarily used for Delay/Disruption Tolerant Networking (DTN) research and supports both interactive GUI mode and batch execution for large-scale experiments.

## DRL (Deep Reinforcement Learning) Integration

This codebase extends the ONE simulator with Deep Reinforcement Learning capabilities using Proximal Policy Optimization (PPO).

### DRL Architecture
- **Java Components**:
  - `src/report/RLBridgeReport.java` - Synchronous bridge between Java simulator and Python DRL server
  - `src/report/RLStateReport.java` - Logs per-step rewards and state information for analysis
  - `src/routing/ProphetRouter.java` - Enhanced to support external predictability offsets from DRL
  - `src/report/BufferOccupancyTracker.java` - Shared buffer occupancy tracking across nodes

- **Python Components**:
  - `toolkit/drl_server.py` - HTTP server implementing PPO agent with PyTorch
  - `toolkit/requirements.txt` - Python dependencies (torch, numpy)
  - `toolkit/*.py` - Analysis and plotting scripts for results

### DRL Workflow
1. Java simulator sends state observations and previous transitions to Python server via HTTP
2. Python PPO agent processes 5-dimensional observations: `[contacts_norm, pred, bufocc_mean, capacity_norm, self_buf_util]`
3. Agent returns predictability delta actions within `[-delta_limit, +delta_limit]`
4. Java applies deltas to Prophet routing protocol predictability values
5. Rewards are calculated based on delivered messages, relays, drops, and delays

### DRL Configuration
- **Environment Variables**: Control training behavior (TRAIN_UNTIL_SECONDS, SAVE_AT_SECONDS, EVAL_ONLY, etc.)
- **Buffer-specific Experiments**: Scripts support different buffer sizes (5M-50M) for comprehensive analysis
- **Automated Workflows**: PowerShell scripts for training, evaluation, and analysis pipelines

### Key DRL Reports and Output
- **Training Metrics**: `reports_drl_train/` - Loss curves, rewards, and learning statistics
- **Evaluation Results**: `reports_drl_eval/` - Performance metrics during evaluation
- **Model Checkpoints**: `models/` - Saved PPO models organized by buffer size (buf05/ through buf50/)
- **Summary Reports**: `reports_summary/` - Aggregated results across experiments
- **Scenario Outputs**: Various `.txt` files for different buffer sizes and routing protocols
- **Step Rewards**: `reports/reward_logs/` - Step-by-step reward tracking for 1000k simulations
- **Reward Plots**: `reports/reward_plots/` - Real-time reward visualization graphs

## Current DRL Reward System (Latest Update)

### Buffer-Specific Reward Strategies
The DRL system uses differentiated reward strategies based on buffer sizes:

#### 5M, 10M, 15M, 20M Buffers: Pressure-Based Rewards ⭐ LATEST
- **Base**: `delivered × 10.0`
- **Pressure-based relay**: Uses `pressure_diff = my_occupancy - network_avg_occupancy`
- **Rewards/Penalties**: Applied when `|pressure_diff| ≥ 0.1`
  - `pressure_diff ≥ 0.1`: Relay bonus = `min(pressure_diff × 2.0 × relayed, relayed × 5.0)`
  - `pressure_diff ≤ -0.1`: Relay penalty = `max(pressure_diff × 1.0 × relayed, -relayed × 2.0)`
- **Strategy**: "Dynamic cooperation based on network situation - help when relatively available, restrain when constrained"

#### 25M Buffer: Threshold-Based Smart Relay
- **Base**: `delivered × 10.0`
- **Rewards**: `relayed × 5.0` when `buffer_occupancy ≤ 0.6`
- **Penalties**: `relayed × -3.0` when `buffer_occupancy ≥ 0.8`

#### 30M+ Buffers: Traditional Smart Relay
- **Base**: `delivered × 10.0 + relayed × 10.0`
- **Bonus**: `relayed × 5.0` when `buffer_occupancy ≤ 0.7`

### 1000k Simulation Commands
For long-term (1,000,000 seconds) DRL training:

```bash
# Automatic execution with real-time monitoring
.\start_1000k_experiment.ps1 -BufferSize 10

# Manual execution
python toolkit/drl_server.py  # Start DRL server first
./one.sh scenarios/dynamic/drl_train/drl_train_buf10_1000k.txt

# Real-time reward monitoring (separate terminal)
python toolkit/plot_step_rewards_1000k.py --buffer 10 --interval 10
```

### Environment Variables for 1000k Training
```bash
EPISODE_SECONDS=1000000        # 1M second episodes
TOTAL_EPISODES=1               # Single long episode
SAVE_AT_SECONDS=100000         # Save every 100k seconds
STEP_REWARD_TRACKING=true      # Enable step-by-step reward logging
MODEL_DIR=models/buf10_1000k   # Model save directory
```

## Performance Issues and Solutions

### Small Buffer Performance Problem (5M-20M)
**Issue**: Good performance during training but significant degradation during evaluation

**Root Causes**:
- **Seed Sensitivity**: Small buffers highly sensitive to initial conditions
- **Overfitting**: Adaptation to specific training scenarios
- **Complex Reward Signals**: Pressure-based rewards can be unstable
- **Exploration/Exploitation Imbalance**: Deterministic policies during evaluation

**Solutions Implemented**:
1. **Pressure-Based Rewards Restored**: Dynamic network situation awareness for 5M, 10M, 15M, 20M
2. **Multi-Seed Learning**: Plan to train with 3-5 different seeds for robust policies
3. **Transfer Learning**: Potential knowledge transfer from stable 30M+ models

### Buffer-Specific Characteristics
- **5M-20M**: High seed sensitivity, require pressure-based rewards, multi-seed training recommended
- **25M**: Intermediate stage, threshold-based approach sufficient
- **30M+**: Stable, robust with single seed, simple reward structure works well

### Experimental Workflows
- **Buffer Size Studies**: Use PowerShell scripts to run systematic experiments across buffer sizes (5M-50M)
- **Baseline Comparisons**: Prophet router baselines and heuristic strategies (e.g., DupCA-aware Prophet)
- **Training Monitoring**: Real-time monitoring tools track loss curves, rewards, and performance metrics
- **Policy Analysis**: Export and visualize trained policies, analyze catastrophic forgetting patterns
- **Result Aggregation**: Automated tools combine results across multiple runs and buffer configurations