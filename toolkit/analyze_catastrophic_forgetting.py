#!/usr/bin/env python3
"""
Catastrophic Forgetting Analysis for DTN_RL
Analyzes potential catastrophic forgetting in DRL training logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import re
from pathlib import Path

def load_rl_bridge_reports(reports_dir):
    """Load and analyze RLBridgeReport files for state/action patterns"""
    pattern = os.path.join(reports_dir, "*RLBridgeReport.txt")
    files = list(Path(reports_dir).glob("*RLBridgeReport.txt"))

    data = []
    for file in files:
        # Extract RNG seed from filename
        match = re.search(r'rng(\d+)', str(file))
        rng_seed = int(match.group(1)) if match else 0

        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                if "RLBridge OK" in line and "states=" in line:
                    # Parse: # RLBridge OK t=600 policy=ppo_v2 states=9 recv_actions=9 applied_actions=9
                    parts = line.strip().split()
                    try:
                        time_val = int(parts[3].split('=')[1])
                        states = int([p for p in parts if 'states=' in p][0].split('=')[1])
                        actions = int([p for p in parts if 'applied_actions=' in p][0].split('=')[1])

                        data.append({
                            'rng_seed': rng_seed,
                            'time': time_val,
                            'states': states,
                            'actions': actions,
                            'file': str(file.name)
                        })
                    except (IndexError, ValueError):
                        continue

    return pd.DataFrame(data)

def analyze_forgetting_patterns(df_bridge, df_delivery=None):
    """Analyze patterns that indicate catastrophic forgetting"""

    print("=== CATASTROPHIC FORGETTING ANALYSIS ===")

    # 1. RNG Seed Transition Analysis
    print("\n1. RNG SEED ACTIVATION ANALYSIS:")
    seed_stats = df_bridge.groupby('rng_seed').agg({
        'states': ['count', 'mean', 'max'],
        'time': ['min', 'max']
    }).round(2)

    print(seed_stats)

    # Find DRL activation point for each seed
    activation_times = {}
    for seed in df_bridge['rng_seed'].unique():
        seed_data = df_bridge[df_bridge['rng_seed'] == seed]
        first_active = seed_data[seed_data['states'] > 0]
        if not first_active.empty:
            activation_times[seed] = first_active['time'].min()
        else:
            activation_times[seed] = None

    print(f"\n2. DRL ACTIVATION TIMES:")
    for seed, time in activation_times.items():
        status = f"t={time}s" if time is not None else "NEVER ACTIVATED"
        print(f"   RNG {seed}: {status}")

    # 3. State Count Volatility Analysis
    print(f"\n3. STATE COUNT VOLATILITY:")
    for seed in sorted(df_bridge['rng_seed'].unique()):
        seed_data = df_bridge[df_bridge['rng_seed'] == seed]
        active_data = seed_data[seed_data['states'] > 0]

        if not active_data.empty:
            volatility = active_data['states'].std()
            mean_states = active_data['states'].mean()
            cv = volatility / mean_states if mean_states > 0 else 0

            print(f"   RNG {seed}: Mean={mean_states:.1f}, Std={volatility:.1f}, CV={cv:.3f}")

    return activation_times

def plot_forgetting_analysis(df_bridge, output_dir="reports_plots"):
    """Plot forgetting-related metrics"""
    os.makedirs(output_dir, exist_ok=True)

    # Group by RNG seed for analysis
    seeds = sorted(df_bridge['rng_seed'].unique())

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. States over time per RNG seed
    for seed in seeds[:10]:  # Limit to first 10 seeds for readability
        seed_data = df_bridge[df_bridge['rng_seed'] == seed]
        if not seed_data.empty:
            axes[0,0].plot(seed_data['time']/3600, seed_data['states'],
                          alpha=0.7, label=f'RNG {seed}', linewidth=1)

    axes[0,0].set_title('DRL States Over Time by RNG Seed')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].set_ylabel('Number of States')
    axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0,0].grid(True, alpha=0.3)

    # 2. Activation time comparison
    activation_times = {}
    for seed in seeds:
        seed_data = df_bridge[df_bridge['rng_seed'] == seed]
        first_active = seed_data[seed_data['states'] > 0]
        if not first_active.empty:
            activation_times[seed] = first_active['time'].min()

    if activation_times:
        axes[0,1].bar(range(len(activation_times)),
                     [activation_times[s]/3600 for s in sorted(activation_times.keys())],
                     alpha=0.7, color='orange')
        axes[0,1].set_title('DRL Activation Time by RNG Seed')
        axes[0,1].set_xlabel('RNG Seed')
        axes[0,1].set_ylabel('Activation Time (hours)')
        axes[0,1].set_xticks(range(len(activation_times)))
        axes[0,1].set_xticklabels(sorted(activation_times.keys()))
        axes[0,1].grid(True, alpha=0.3)

    # 3. State count volatility (coefficient of variation)
    cv_values = []
    seed_labels = []
    for seed in seeds:
        seed_data = df_bridge[df_bridge['rng_seed'] == seed]
        active_data = seed_data[seed_data['states'] > 0]

        if not active_data.empty and len(active_data) > 5:
            mean_states = active_data['states'].mean()
            std_states = active_data['states'].std()
            cv = std_states / mean_states if mean_states > 0 else 0

            cv_values.append(cv)
            seed_labels.append(seed)

    if cv_values:
        axes[1,0].bar(range(len(cv_values)), cv_values, alpha=0.7, color='red')
        axes[1,0].set_title('State Count Volatility (Coefficient of Variation)')
        axes[1,0].set_xlabel('RNG Seed')
        axes[1,0].set_ylabel('Coefficient of Variation')
        axes[1,0].set_xticks(range(len(cv_values)))
        axes[1,0].set_xticklabels(seed_labels)
        axes[1,0].grid(True, alpha=0.3)

        # Add interpretation
        high_volatility = [i for i, cv in enumerate(cv_values) if cv > 0.3]
        if high_volatility:
            axes[1,0].axhline(y=0.3, color='red', linestyle='--', alpha=0.7,
                            label='High Volatility Threshold')
            axes[1,0].legend()

    # 4. DRL Activity Heatmap (Time vs RNG Seed)
    if len(seeds) > 1:
        # Create heatmap data
        time_bins = np.arange(0, df_bridge['time'].max() + 3600, 3600)  # 1-hour bins
        heatmap_data = np.zeros((len(seeds), len(time_bins)-1))

        for i, seed in enumerate(seeds):
            seed_data = df_bridge[df_bridge['rng_seed'] == seed]
            for j in range(len(time_bins)-1):
                time_mask = (seed_data['time'] >= time_bins[j]) & (seed_data['time'] < time_bins[j+1])
                active_in_bin = seed_data[time_mask & (seed_data['states'] > 0)]
                heatmap_data[i, j] = len(active_in_bin)

        im = axes[1,1].imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        axes[1,1].set_title('DRL Activity Heatmap (RNG Seed vs Time)')
        axes[1,1].set_xlabel('Time (hours)')
        axes[1,1].set_ylabel('RNG Seed')
        xtick_pos = range(0, len(time_bins)-1, 2)
        xtick_labels = [f'{time_bins[i]//3600}' for i in range(0, len(time_bins)-1, 2)]
        axes[1,1].set_xticks(xtick_pos)
        axes[1,1].set_xticklabels(xtick_labels[:len(xtick_pos)])
        axes[1,1].set_yticks(range(len(seeds)))
        axes[1,1].set_yticklabels(seeds)

        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1,1])
        cbar.set_label('DRL Activity Count')

    plt.tight_layout()

    output_file = os.path.join(output_dir, 'catastrophic_forgetting_analysis.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Catastrophic forgetting analysis saved to: {output_file}")

    plt.show()
    return fig

def detect_forgetting_events(df_bridge, threshold_drop=0.5):
    """Detect potential catastrophic forgetting events"""
    print(f"\n4. FORGETTING EVENT DETECTION:")
    print(f"   Threshold: {threshold_drop*100}% drop in activity")

    forgetting_events = []

    for seed in sorted(df_bridge['rng_seed'].unique()):
        seed_data = df_bridge[df_bridge['rng_seed'] == seed]
        active_data = seed_data[seed_data['states'] > 0].sort_values('time')

        if len(active_data) < 10:  # Need sufficient data
            continue

        # Calculate moving average of states
        window = min(20, len(active_data) // 4)
        if window < 5:
            continue

        moving_avg = active_data['states'].rolling(window=window).mean()

        # Find significant drops
        for i in range(window, len(moving_avg)-1):
            current = moving_avg.iloc[i]
            previous = moving_avg.iloc[i-window//2]

            if previous > 0 and current/previous < (1 - threshold_drop):
                forgetting_events.append({
                    'rng_seed': seed,
                    'time': active_data.iloc[i]['time'],
                    'drop_ratio': current/previous,
                    'before_states': previous,
                    'after_states': current
                })

    if forgetting_events:
        print(f"   Found {len(forgetting_events)} potential forgetting events:")
        for event in forgetting_events:
            print(f"     RNG {event['rng_seed']}: t={event['time']:,}s, "
                  f"drop={event['drop_ratio']:.2f} "
                  f"({event['before_states']:.1f}→{event['after_states']:.1f} states)")
    else:
        print("   No significant forgetting events detected")

    return forgetting_events

def main():
    """Main analysis function"""
    if len(sys.argv) > 1:
        reports_dir = sys.argv[1]
    else:
        reports_dir = "reports_drl_train/buf05"

    print(f"Analyzing catastrophic forgetting in: {reports_dir}")

    # Load data
    df_bridge = load_rl_bridge_reports(reports_dir)

    if df_bridge.empty:
        print("No RLBridgeReport data found!")
        return

    print(f"Loaded {len(df_bridge)} RLBridge records from {df_bridge['rng_seed'].nunique()} RNG seeds")

    # Perform analyses
    activation_times = analyze_forgetting_patterns(df_bridge)
    forgetting_events = detect_forgetting_events(df_bridge)
    plot_forgetting_analysis(df_bridge)

    # Summary recommendations
    print(f"\n=== RECOMMENDATIONS ===")

    inactive_seeds = [seed for seed, time in activation_times.items() if time is None]
    if inactive_seeds:
        print(f"⚠️  Seeds {inactive_seeds} never activated DRL - consider increasing simulation time")

    if forgetting_events:
        print(f"⚠️  {len(forgetting_events)} potential forgetting events detected")
        print("   💡 Consider: Experience Replay, L2 Regularization, or Elastic Weight Consolidation")

    active_seeds = [seed for seed, time in activation_times.items() if time is not None]
    if len(active_seeds) > 1:
        activation_variance = np.var([activation_times[s] for s in active_seeds])
        if activation_variance > 3600**2:  # 1 hour variance
            print(f"⚠️  High variance in activation times ({np.sqrt(activation_variance)/3600:.1f}h std)")
            print("   💡 Consider: Multi-task learning or domain adaptation techniques")

    print("   ✅ Monitor entropy coefficient (currently 0.02) for exploration-exploitation balance")

if __name__ == "__main__":
    main()