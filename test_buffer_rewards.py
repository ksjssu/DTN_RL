#!/usr/bin/env python3
"""
Test the new buffer-specific reward functions
"""
import sys
sys.path.append('toolkit')

# Import the functions from drl_server
from drl_server import extract_buffer_size_mb, get_buffer_specific_reward

def test_buffer_size_extraction():
    """Test buffer size extraction from sim_id"""
    test_cases = [
        ("scenario_buf05_eval_100k", 5.0),
        ("scenario_buf10_eval_100k", 10.0),
        ("scenario_buf30_eval_100k", 30.0),
        ("scenario_buf50_eval_100k", 50.0),
        ("rl_training_overlay_buf25", 25.0),
        ("dynamic_traffic_100k", None),
        ("", None)
    ]

    print("=== Buffer Size Extraction Test ===")
    for sim_id, expected in test_cases:
        result = extract_buffer_size_mb(sim_id)
        status = "PASS" if result == expected else "FAIL"
        print(f"{status}: '{sim_id}' -> {result} (expected {expected})")

def test_reward_calculations():
    """Test buffer-specific reward calculations"""
    # Test data: delivered=2, relayed=5, drops=1, aborted=0, delay_penalty=0
    delivered, relayed, drops, aborted, delay_penalty = 2, 5, 1, 0, 0

    test_cases = [
        ("scenario_buf05_eval", 5.0, 2 * 50.0),  # 5M: delivery only
        ("scenario_buf10_eval", 10.0, 2 * 30.0),  # 10M: conservative
        ("scenario_buf15_eval", 15.0, 2 * 25.0 + 5 * 8.0),  # 15M: balanced
        ("scenario_buf20_eval", 20.0, 2 * 20.0 + 5 * 12.0),  # 20M: intermediate
        ("scenario_buf25_eval", 25.0, 2 * 15.0 + 5 * 15.0),  # 25M: aggressive
        ("scenario_buf30_eval", 30.0, 2 * 10.0 + 5 * 10.0),  # 30M: smart relay
        ("scenario_buf50_eval", 50.0, 2 * 10.0 + 5 * 10.0),  # 50M: smart relay
    ]

    print("\n=== Reward Calculation Test ===")
    print(f"Test data: delivered={delivered}, relayed={relayed}, drops={drops}, aborted={aborted}")

    for sim_id, buffer_size, expected in test_cases:
        result = get_buffer_specific_reward(sim_id, delivered, relayed, drops, aborted, delay_penalty)
        status = "PASS" if abs(result - expected) < 0.01 else "FAIL"
        print(f"{status}: {buffer_size}M buffer -> {result:.1f} (expected {expected:.1f})")

if __name__ == "__main__":
    test_buffer_size_extraction()
    test_reward_calculations()