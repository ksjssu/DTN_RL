# Repository Guidelines

## Project Structure & Module Organization
- `src/` — Java sources organized by package: `core/`, `routing/`, `movement/`, `input/`, `report/`, `gui/`, `applications/`, `interfaces/`, `ui/`, `util/`, and tests in `src/test/`.
- `target/` — Compiled `.class` files produced by the build scripts.
- `lib/` — Third‑party jars used at compile/runtime (`ECLA.jar`, `DTNConsoleConnection.jar`).
- `data/` — Example maps and POIs in WKT used by simulation settings.
- `wdm_settings/` and `*.txt` — Ready‑to‑run configuration files (see `README.txt` for override order).
- `toolkit/` — Utility scripts (Perl) for analysis/conversion.

## Build, Test, and Development Commands
- Build (Windows): `./compile.bat`
- Build (Linux/macOS): `./compile.sh`
  - Compiles from `src/` to `target/` with classpath `lib/`.
- Run (GUI or batch):
  - Windows: `./one.bat default_settings.txt`
  - Linux/macOS: `./one.sh -b 3 default_settings.txt` (batch with 3 runs)
- Clean: delete `target/` (no dedicated clean script).
- Requirements: JDK 8+ recommended (project works without Maven/Gradle).

## Coding Style & Naming Conventions
- Java, 4‑space indentation, UTF‑8, ~100‑char line length.
- Packages: lowercase (`routing.util`); Classes: UpperCamelCase; methods/fields: lowerCamelCase; constants: UPPER_SNAKE_CASE.
- One public class per file; keep classes in matching package folders.
- Settings keys use `Namespace.Key = Value` (CamelCase), e.g., `MovementModel.maxSpeed`.
- Write Javadoc for public APIs; prefer immutable collections when reasonable.

## Testing Guidelines
- Tests live in `src/test/`; JUnit 3/4 styles (`AllTests.java`, `@Test`).
- Run via IDE or CLI by adding JUnit to classpath, compiling `src/test`, and running `test.AllTests`.
- Name tests `*Test.java`; keep tests deterministic and GUI‑independent.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise; add scope when helpful (e.g., `routing: fix relay policy`).
- Reference issues (e.g., `Fixes #123`); explain rationale, risks, and config impacts.
- PRs include a description, run instructions (commands/config files), screenshots for GUI changes, and updated docs/settings when relevant.
- Do not commit `target/`, large generated outputs, or local data.

## Configuration Tips
- Layer configs: keep shared values in `default_settings.txt`; override in scenario‑specific files.
- Use forward slashes in settings paths (portable across platforms).
- WKT assets reside under `data/`; ensure referenced files exist when sharing configs.
