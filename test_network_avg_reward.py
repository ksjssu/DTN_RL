#!/usr/bin/env python3
"""
Test script for network average-based relay reward logic.
Validates that the reward calculation uses locally observed network average (buf_mean_prev).
"""

# Test constants (matching drl_server.py defaults)
NETWORK_AVG_LOW_THRESHOLD = 0.33
NETWORK_AVG_HIGH_THRESHOLD = 0.66
RELAY_CRIT_MULT = 2.0

def calculate_relay_reward(buf_mean_observed, w_relay, relayed):
    """
    Network average-based relay reward calculation (matching drl_server.py:841-852)
    Each node uses its locally observed network average.
    """
    if buf_mean_observed <= NETWORK_AVG_LOW_THRESHOLD:
        # Observed network average is low (<=33%) -> reward relay
        relay_bonus = w_relay * relayed
    elif buf_mean_observed >= NETWORK_AVG_HIGH_THRESHOLD:
        # Observed network average is high (>=66%) -> penalize relay
        relay_bonus = -RELAY_CRIT_MULT * w_relay * relayed
    else:
        # Observed network average is medium (33%-66%) -> neutral
        relay_bonus = 0.0

    return relay_bonus

def run_tests():
    """Run test cases"""
    print("=== Network Average-Based Relay Reward Tests ===\n")

    w_relay = 3.0  # typical relay weight
    relayed = 10   # 10 messages relayed

    test_cases = [
        # (buf_mean_observed, expected_behavior)
        (0.20, "REWARD (observed avg 20% <= 33%)"),
        (0.33, "REWARD (observed avg 33% <= 33%)"),
        (0.40, "NEUTRAL (observed avg 40% in 33-66%)"),
        (0.50, "NEUTRAL (observed avg 50% in 33-66%)"),
        (0.60, "NEUTRAL (observed avg 60% in 33-66%)"),
        (0.66, "PENALTY (observed avg 66% >= 66%)"),
        (0.75, "PENALTY (observed avg 75% >= 66%)"),
        (0.90, "PENALTY (observed avg 90% >= 66%)"),

        # Edge cases
        (0.00, "REWARD (observed avg 0%)"),
        (1.00, "PENALTY (observed avg 100%)"),
    ]

    all_passed = True
    for buf_mean_observed, expected in test_cases:
        bonus = calculate_relay_reward(buf_mean_observed, w_relay, relayed)

        # Determine actual behavior
        if bonus > 0:
            actual = f"REWARD ({bonus:.1f})"
        elif bonus < 0:
            actual = f"PENALTY ({bonus:.1f})"
        else:
            actual = "NEUTRAL (0.0)"

        # Check if expectation matches
        passed = expected.split('(')[0].strip() == actual.split('(')[0].strip()
        status = "✓" if passed else "✗"

        if not passed:
            all_passed = False

        print(f"{status} buf_mean={buf_mean_observed:.2f} → {actual:25s} (expected: {expected})")

    # Test scenario: Different nodes, same time, different observations
    print(f"\n{'='*70}")
    print("Scenario: Multiple nodes at same time with different observations\n")

    scenarios = [
        ("Node A (isolated)", 0.25, "REWARD"),
        ("Node B (urban)", 0.70, "PENALTY"),
        ("Node C (suburban)", 0.45, "NEUTRAL"),
    ]

    for name, obs_avg, expected in scenarios:
        bonus = calculate_relay_reward(obs_avg, w_relay, relayed)
        if bonus > 0:
            actual = "REWARD"
        elif bonus < 0:
            actual = "PENALTY"
        else:
            actual = "NEUTRAL"

        status = "✓" if actual == expected else "✗"
        print(f"{status} {name:20s} observed avg={obs_avg:.2f} → {actual}")

    print(f"\n{'='*70}")
    if all_passed:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")

    return all_passed

if __name__ == "__main__":
    import sys
    success = run_tests()
    sys.exit(0 if success else 1)
