# Repository Guidelines

## Project Structure & Module Organization
Keep production code under `src/`. Place shared models in `src/core`, routing logic in `src/routing`, and mobility behaviors in `src/movement`. Mirror packages under `src/test/...` for test sources so they compile alongside the code being exercised. Keep scenario data in `data/` (maps, POIs, WKT shapes) and rely on staged jars in `lib/`. Build artifacts live in `target/`; do not add generated outputs to version control.

## Build, Test, and Development Commands
Run `./compile.sh` (or `compile.bat` on Windows) to produce fresh Java 8 classes in `target/`. Use `./one.sh -b 3 default_settings.txt` for a three-run headless regression; drop `-b` to launch the GUI. After compiling tests, execute `java -cp "target;lib/*;target/test" test.AllTests` to run the aggregate JUnit suite.

## Coding Style & Naming Conventions
Indent Java code with four spaces and keep lines near 100 characters. Packages stay lowercase, classes UpperCamelCase, and members lowerCamelCase; reserve `UPPER_SNAKE_CASE` for constants. Limit features to Java 8 and supply Javadoc for public APIs. Add concise inline comments only where logic is subtle.

## Testing Guidelines
Write deterministic JUnit 3/4 cases named `*Test.java` under the mirrored package in `src/test`. Seed randomness explicitly so regressions remain reproducible. Re-run `./compile.sh` before invoking `test.AllTests`, and expand coverage when adjusting routing or movement behavior.

## Commit & Pull Request Guidelines
Author commits with imperative, scoped messages such as `routing: tighten relay policy`. Pull requests should summarize intent, list affected modules, and include verification evidence (for example `./compile.sh`, regression runs). Provide GUI screenshots when visuals change and flag configuration or data updates. Never commit jars or `target/` outputs.

## Configuration & Debugging Tips
Tune defaults in `default_settings.txt`, override scenarios via matching keys in `wdm_settings/`, and delete `target/` followed by `./compile.sh` if builds drift. Prefer dependencies staged in `lib/` to avoid drift and keep iterative changes small so they are easy to review.
