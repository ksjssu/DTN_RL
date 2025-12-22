#!/usr/bin/env python3
"""
R-MAPPO TBPTT Flow Test Script

Tests the complete flow:
1. done_keys mechanism (event-based done detection)
2. SequenceWindow TBPTT management
3. Hidden state continuity
4. PPO training triggers

Monitors drl_server_rmappo.py output for key events.
"""
import subprocess
import time
import re
import sys

print("=" * 80)
print("🧪 R-MAPPO TBPTT Flow Test")
print("=" * 80)
print("Testing drl_train_buf20_100k.txt scenario\n")

print("📋 Expected Flow:")
print("  1. DRL server starts on port 5020")
print("  2. Java simulator connects")
print("  3. Every 100s: /infer_and_update called")
print("  4. done_keys collected from prev_transition events")
print("  5. SequenceWindows finalized when len>=16 or done=True")
print("  6. PPO training when ready_sequences >= 32")
print("  7. Hidden states reset on done=True")
print("=" * 80 + "\n")

print("🚀 Starting DRL server...")
print("Command: USE_GRADIENT_REWARD=true W_GRAD=1.0 W_ALIGN=0.5 W_ARRIVAL=10.0 python3 toolkit/drl_server_rmappo.py\n")

# Start DRL server in background
server_proc = subprocess.Popen(
    ["python3", "toolkit/drl_server_rmappo.py"],
    env={
        **subprocess.os.environ,
        "USE_GRADIENT_REWARD": "true",
        "W_GRAD": "1.0",
        "W_ALIGN": "0.5",
        "W_ARRIVAL": "10.0",
        "STEP_REWARD_TRACKING": "false",  # Disable for cleaner logs
    },
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

print("Waiting 3 seconds for server to initialize...")
time.sleep(3)

print("\n" + "=" * 80)
print("🎮 Starting Java Simulator (30 seconds timeout for testing)")
print("=" * 80 + "\n")

# Start Java simulator in background
sim_proc = subprocess.Popen(
    ["timeout", "30", "./one.sh", "-b", "1", "scenarios/dynamic/drl_train/drl_train_buf20_100k.txt"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Monitoring patterns
patterns = {
    "server_start": re.compile(r"Listening on http://127\.0\.0\.1:(\d+)"),
    "request": re.compile(r'\"POST /infer_and_update HTTP/1\.1\" (\d+)'),
    "episode_end": re.compile(r'\"POST /episode_end HTTP/1\.1\" (\d+)'),
    "done_keys": re.compile(r"done_keys=\{([^}]*)\}"),
    "seq_finalized": re.compile(r"Finalized sequence for key=(\S+), len=(\d+), done=(\w+)"),
    "ppo_training": re.compile(r"Starting PPO training with (\d+) sequences"),
    "hidden_reset": re.compile(r"Reset hidden states for keys=\{([^}]*)\}"),
}

events = {
    "requests": 0,
    "done_keys_total": 0,
    "sequences_finalized": 0,
    "ppo_updates": 0,
    "hidden_resets": 0,
}

print("📊 Monitoring Events:")
print("-" * 80)

try:
    # Monitor server output
    while True:
        line = server_proc.stdout.readline()
        if not line:
            break

        line = line.strip()

        # Check patterns
        if patterns["server_start"].search(line):
            print(f"✅ Server started: {line}")

        if patterns["request"].search(line):
            events["requests"] += 1
            print(f"📥 Request #{events['requests']}: {line}")

        if patterns["episode_end"].search(line):
            print(f"🏁 Episode end: {line}")

        if patterns["done_keys"].search(line):
            match = patterns["done_keys"].search(line)
            done_keys = match.group(1)
            count = len(done_keys.split(",")) if done_keys else 0
            events["done_keys_total"] += count
            print(f"✅ done_keys collected: {count} keys - {done_keys[:100]}...")

        if patterns["seq_finalized"].search(line):
            match = patterns["seq_finalized"].search(line)
            key = match.group(1)
            length = match.group(2)
            done = match.group(3)
            events["sequences_finalized"] += 1
            print(f"📦 Sequence finalized: key={key}, len={length}, done={done}")

        if patterns["ppo_training"].search(line):
            match = patterns["ppo_training"].search(line)
            num_seqs = match.group(1)
            events["ppo_updates"] += 1
            print(f"🎓 PPO Training #{events['ppo_updates']}: {num_seqs} sequences")

        if patterns["hidden_reset"].search(line):
            match = patterns["hidden_reset"].search(line)
            keys = match.group(1)
            count = len(keys.split(",")) if keys else 0
            events["hidden_resets"] += count
            print(f"🔄 Hidden states reset: {count} keys - {keys[:100]}...")

        # Check if simulator finished
        if sim_proc.poll() is not None:
            break

except KeyboardInterrupt:
    print("\n\n⚠️ Interrupted by user")

finally:
    print("\n" + "=" * 80)
    print("📈 Summary")
    print("=" * 80)
    print(f"Total /infer_and_update requests: {events['requests']}")
    print(f"Total done_keys collected: {events['done_keys_total']}")
    print(f"Total sequences finalized: {events['sequences_finalized']}")
    print(f"Total PPO updates: {events['ppo_updates']}")
    print(f"Total hidden state resets: {events['hidden_resets']}")
    print("=" * 80 + "\n")

    # Cleanup
    print("🧹 Cleaning up processes...")
    server_proc.terminate()
    sim_proc.terminate()

    try:
        server_proc.wait(timeout=5)
        sim_proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_proc.kill()
        sim_proc.kill()

    print("✅ Test complete!")
