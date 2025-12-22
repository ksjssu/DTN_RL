from pathlib import Path

lines = Path("toolkit/drl_server.py").read_text(encoding="utf-8-sig").splitlines()
for idx, line in enumerate(lines, 1):
    if "def evaluate_decision_quality" in line:
        print("evaluate_decision_quality", idx)
    if line.strip() == "key = f\"{host}#{dest}\"":
        print("key_assignment", idx)
