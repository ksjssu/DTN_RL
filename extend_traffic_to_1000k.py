#!/usr/bin/env python3
"""
Traffic pattern extension script for 1000k seconds simulation
Extends the existing traffic pattern by repeating the 100k pattern 10 times
"""

def extend_traffic_pattern():
    input_file = "scenarios/dynamic/drl_train/drl_train_buf10_1000k.txt"

    # Read the current file
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Find the current event count
    event_count = 420  # Current number of events

    # Find the last event line and extract the pattern
    # The pattern repeats every 100k seconds with L/M/H patterns

    # We need to extend from 200k to 1000k (8 more cycles of 100k each)
    additional_events = []
    event_num = event_count + 1

    # L=120,180 interval, M=25,35 interval, H=5,10 interval
    # Each cycle = 7 events per 10k period, 70 events per 100k period

    base_patterns = [
        ("L", 120, 180),   # Low traffic
        ("M", 25, 35),     # Medium traffic
        ("H", 5, 10),      # High traffic
        ("M", 25, 35),     # Medium traffic
        ("L", 120, 180),   # Low traffic
        ("M", 25, 35),     # Medium traffic
        ("H", 5, 10),      # High traffic
    ]

    # Extend for cycles 3-10 (200k to 1000k)
    for cycle in range(2, 10):  # cycles 2-9 (200k-1000k)
        cycle_start = cycle * 100000

        for period in range(10):  # 10 periods per cycle
            period_start = cycle_start + period * 10000

            for i, (traffic_type, min_interval, max_interval) in enumerate(base_patterns):
                event_start = period_start + i * 1429  # ~1429 seconds per event
                event_end = min(period_start + (i + 1) * 1429, period_start + 10000)

                if i == 6:  # Last event in period, extend to period end
                    event_end = period_start + 10000
                    if period == 9:  # Last period in cycle
                        event_end = cycle_start + 100000

                prefix_num = cycle * 20 + period * 2 + (0 if traffic_type in ["L", "M"] else 1)
                prefix = f"{traffic_type}{prefix_num}"

                additional_events.append(
                    f"Events{event_num}.class = MessageEventGenerator\n"
                    f"Events{event_num}.interval = {min_interval},{max_interval}\n"
                    f"Events{event_num}.size = 500k,1M\n"
                    f"Events{event_num}.hosts = 0,126\n"
                    f"Events{event_num}.time = {event_start},{event_end}\n"
                    f"Events{event_num}.prefix = {prefix}\n"
                )
                event_num += 1

    # Update the Events.nrof line
    updated_lines = []
    for line in lines:
        if line.startswith("Events.nrof = "):
            updated_lines.append(f"Events.nrof = {event_num - 1}\n")
        else:
            updated_lines.append(line)

    # Add the additional events
    updated_lines.extend(additional_events)

    # Write back to file
    with open(input_file, 'w', encoding='utf-8') as f:
        f.writelines(updated_lines)

    print(f"Extended traffic pattern to 1000k seconds with {event_num - 1} total events")

if __name__ == "__main__":
    extend_traffic_pattern()