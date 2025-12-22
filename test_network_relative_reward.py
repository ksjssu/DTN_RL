#!/usr/bin/env python3
"""
Test script for network-relative relay reward logic.
Validates that the new reward calculation uses network average buffer occupancy.
"""

# Test constants (matching drl_server.py defaults)
RELAY_NETWORK_LOW_DIFF = -0.15
RELAY_NETWORK_WARN_DIFF = 0.10
RELAY_NETWORK_CRIT_DIFF = 0.20
RELAY_CRIT_MULT = 2.0

def calculate_relay_reward(self_util, buf_mean, w_relay, relayed):
    """
    Network-relative relay reward calculation (matching drl_server.py:841-856)
    """
    relative_util = self_util - buf_mean

    if relative_util <= RELAY_NETWORK_LOW_DIFF:
        # Much below network average: reward relay
        relay_bonus = w_relay * relayed
    elif relative_util >= RELAY_NETWORK_CRIT_DIFF:
        # Much above network average: heavy penalty
        relay_bonus = -RELAY_CRIT_MULT * w_relay * relayed
    elif relative_util >= RELAY_NETWORK_WARN_DIFF:
        # Above network average: penalty
        relay_bonus = -w_relay * relayed
    else:
        # Near network average: neutral
        relay_bonus = 0.0

    return relay_bonus

def run_tests():
    """Run test cases"""
    print("=== Network-Relative Relay Reward Tests ===\n")

    w_relay = 3.0  # typical relay weight
    relayed = 10   # 10 messages relayed

    test_cases = [
        # (self_util, buf_mean, expected_behavior)
        (0.20, 0.50, "REWARD (much below average)"),
        (0.30, 0.50, "REWARD (below average)"),
        (0.45, 0.50, "NEUTRAL (near average)"),
        (0.55, 0.50, "NEUTRAL (near average)"),
        (0.65, 0.50, "PENALTY (above average)"),
        (0.75, 0.50, "HEAVY PENALTY (much above average)"),

        # Edge cases
        (0.40, 0.40, "NEUTRAL (exactly at average)"),
        (0.00, 0.30, "REWARD (empty vs average 30%)"),
        (0.90, 0.30, "HEAVY PENALTY (90% vs average 30%)"),
    ]

    all_passed = True
    for self_util, buf_mean, expected in test_cases:
        bonus = calculate_relay_reward(self_util, buf_mean, w_relay, relayed)
        relative_util = self_util - buf_mean

        # Determine actual behavior
        if bonus > 0:
            actual = f"REWARD ({bonus:.1f})"
        elif bonus < 0:
            if abs(bonus) > w_relay * relayed:
                actual = f"HEAVY PENALTY ({bonus:.1f})"
            else:
                actual = f"PENALTY ({bonus:.1f})"
        else:
            actual = "NEUTRAL (0.0)"

        # Check if expectation matches
        passed = expected.split('(')[0].strip() == actual.split('(')[0].strip()
        status = "✓" if passed else "✗"

        if not passed:
            all_passed = False

        print(f"{status} self={self_util:.2f}, mean={buf_mean:.2f}, "
              f"diff={relative_util:+.2f} → {actual:30s} (expected: {expected})")

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
