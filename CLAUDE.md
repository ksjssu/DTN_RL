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
The `toolkit/` directory contains Perl scripts for:
- Data analysis and plotting (ccdfPlotter.pl, getStats.pl)
- Format conversion (dtnsim2parser.pl, dieselnetConverter.pl)
- Visualization (messageLocationAnimator.pl)

This simulator is primarily used for Delay/Disruption Tolerant Networking (DTN) research and supports both interactive GUI mode and batch execution for large-scale experiments.