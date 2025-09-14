# Repository Guidelines

## Project Structure & Module Organization
- `src/` — Java sources by package: `core/`, `routing/`, `movement/`, `input/`, `report/`, `gui/`, `applications/`, `interfaces/`, `ui/`, `util/`. Tests live in `src/test/`.
- `target/` — Compiled classes (safe to delete to clean builds).
- `lib/` — Third‑party jars for compile/runtime (e.g., `ECLA.jar`, `DTNConsoleConnection.jar`).
- `data/` — Example maps/POIs (WKT) referenced by settings.
- `wdm_settings/` and `*.txt` — Ready‑to‑run configuration files (see `README.txt` for override order).
- `toolkit/` — Utility scripts for analysis/conversion.

## Build, Test, and Development Commands
- Build (Windows): `./compile.bat` — compiles `src/` into `target/` using jars from `lib/` (JDK 8+).
- Build (Linux/macOS): `./compile.sh` — same as above.
- Run (GUI or batch):
  - Windows: `./one.bat default_settings.txt`
  - Linux/macOS: `./one.sh -b 3 default_settings.txt` (batch with 3 runs)
- Clean: delete the `target/` directory.

## Coding Style & Naming Conventions
- Java, 4‑space indentation, UTF‑8, ~100‑char line length.
- Packages: lowercase; Classes: UpperCamelCase; methods/fields: lowerCamelCase; constants: UPPER_SNAKE_CASE.
- One public class per file; keep classes in matching package folders.
- Settings keys: `Namespace.Key = Value` (CamelCase), e.g., `MovementModel.maxSpeed`.
- Public APIs include Javadoc; prefer immutable collections when reasonable.

## Testing Guidelines
- Frameworks: JUnit 3/4 supported.
- Location: `src/test/`; name tests `*Test.java`; keep deterministic and GUI‑independent.
- Run: add JUnit to the classpath, compile `src/test`, then run `test.AllTests` (via IDE or CLI).

## Commit & Pull Request Guidelines
- Commits: imperative, concise; add scope when helpful (e.g., `routing: fix relay policy`).
- Reference issues (e.g., `Fixes #123`); explain rationale, risks, and config impacts.
- PRs: include description, run instructions (commands/config files), screenshots for GUI changes, and updated docs/settings as needed.
- Do not commit `target/`, large generated outputs, or local data.

## Configuration Tips
- Keep shared values in `default_settings.txt`; override in scenario files.
- Use forward slashes in settings paths (portable across OSes).
- WKT assets reside under `data/`; ensure referenced files exist when sharing configs.

## Agent Notes
- Scope applies to the entire repository tree.
- Keep changes minimal and focused; avoid unrelated refactors or renames.
- Fix root causes; match existing style and module layout.
