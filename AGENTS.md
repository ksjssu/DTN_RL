# Repository Guidelines

## Project Structure & Module Organization
Source packages live under `src/core` for domain logic, `src/routing` for routers (e.g., `ProphetRouter`), and `src/movement` for mobility behaviors. Mirrored tests sit in `src/test/...` using identical package paths, so `src/routing/ProphetRouter.java` pairs with `src/test/routing/ProphetRouterTest.java`. Scenario inputs reside in `data/`, DRL helpers in `toolkit/`, pretrained policies in `models/`, and third-party JARs in `lib/`. Build artifacts land in `target/`; keep it ignored and empty in commits.

## Build, Test, and Development Commands
- `./compile.sh`: Compiles Java 8 sources and stages outputs plus test classes in `target/`.
- `java -cp "target;lib/*;target/test" test.AllTests`: Runs the aggregated JUnit suite after compiling.
- `./one.sh -b 3 default_settings.txt`: Executes a deterministic headless simulation; omit `-b` for GUI.
- `python toolkit/drl_server.py` then `./one.sh -b 1 dynamic_traffic_hml_1h_settings.txt`: Launches DRL scenarios with the server running first.
Re-run `./compile.sh` whenever dependencies change and clean `target/` if builds drift.

## Coding Style & Naming Conventions
Java code uses four-space indentation, ~100-character lines, and Java 8 APIs. Packages are lowercase, classes UpperCamelCase, members lowerCamelCase, and constants UPPER_SNAKE_CASE. Provide Javadoc for public routers, reports, and movement models, and keep inline comments minimal, focusing on non-obvious scheduling logic. Python utilities in `toolkit/` follow PEP 8 and stay `black`-compatible.

## Testing Guidelines
Tests rely on JUnit 3/4, live under `src/test/...`, and end with `*Test`. Seed randomness explicitly (see `default_settings.txt`) to keep DRL comparisons reproducible. After changing routing or DRL logic, run `./compile.sh`, `java -cp "target;lib/*;target/test" test.AllTests`, and at least one representative `./one.sh` scenario.

## Commit & Pull Request Guidelines
Use imperative, scoped subjects such as `routing: tighten relay policy` or `report: add drl logging`. Pull requests summarize intent, list modules touched (e.g., `src/routing`, `toolkit`, `data`), and attach evidence for `./compile.sh`, AllTests, and relevant `./one.sh` runs. Flag any `data/` or configuration updates explicitly and never stage `target/` outputs or new JARs.

## Configuration & Debugging Tips
Adjust defaults in `default_settings.txt` and override per-scenario knobs via `wdm_settings/`. Keep `Report.report6 = RLBridgeReport` synchronized with the DRL server endpoint, and clear `drl_server.out` before diagnosing failures. If builds misbehave, delete `target/`, rerun `./compile.sh`, and re-test before landing.
