#!/usr/bin/env python3
"""
Quick action visualization test for 10M buffer
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_actions_from_rlstate(file_path):
    """Extract action-like data from RLStateReport for visualization"""
    try:
        # Read the file
        with open(file_path, 'r') as f:
            lines = f.readlines()

        data = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split()
            if len(parts) >= 13 and parts[1] == 'R':  # Reward line
                try:
                    time = float(parts[0])
                    host = parts[2]
                    contacts_norm = float(parts[5]) if parts[5] != 'NaN' else 0.0
                    pred = float(parts[6]) if parts[6] != 'NaN' else 0.0
                    bufocc_mean = float(parts[7]) if parts[7] != 'NaN' else 0.0
                    capacity_norm = float(parts[8])
                    self_buf_util = float(parts[9])
                    reward = float(parts[13])

                    data.append({
                        'time': time,
                        'host': host,
                        'contacts_norm': contacts_norm,
                        'pred': pred,
                        'bufocc_mean': bufocc_mean,
                        'capacity_norm': capacity_norm,
                        'self_buf_util': self_buf_util,
                        'reward': reward
                    })
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error reading file: {e}")
        return None

def analyze_state_patterns(df):
    """Analyze state patterns instead of actions"""
    print("=== 10M BUFFER STATE PATTERN ANALYSIS ===")
    print(f"Total state records: {len(df)}")
    print(f"Time range: {df['time'].min():.0f} - {df['time'].max():.0f} seconds")
    print(f"Unique hosts: {df['host'].nunique()}")

    print(f"\nState Statistics:")
    print(f"  Contacts (norm): mean={df['contacts_norm'].mean():.3f}, std={df['contacts_norm'].std():.3f}")
    print(f"  Prophet pred: mean={df['pred'].mean():.3f}, std={df['pred'].std():.3f}")
    print(f"  Buffer utilization: mean={df['self_buf_util'].mean():.3f}, std={df['self_buf_util'].std():.3f}")
    print(f"  Rewards: mean={df['reward'].mean():.3f}, std={df['reward'].std():.3f}")

    # Check reward distribution
    positive_rewards = (df['reward'] > 0).sum()
    zero_rewards = (df['reward'] == 0).sum()
    print(f"\nReward Distribution:")
    print(f"  Positive rewards: {positive_rewards} ({positive_rewards/len(df)*100:.1f}%)")
    print(f"  Zero rewards: {zero_rewards} ({zero_rewards/len(df)*100:.1f}%)")

def create_state_visualizations(df, output_dir="reports_plots/buf10_states"):
    """Create state visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('default')

    # 1. State distributions
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Contacts
    axes[0,0].hist(df['contacts_norm'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].set_title('Contacts Distribution')
    axes[0,0].set_xlabel('Contacts (normalized)')
    axes[0,0].grid(True, alpha=0.3)

    # Prophet prediction
    axes[0,1].hist(df['pred'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[0,1].set_title('Prophet Prediction Distribution')
    axes[0,1].set_xlabel('Prophet Score')
    axes[0,1].grid(True, alpha=0.3)

    # Buffer utilization
    axes[0,2].hist(df['self_buf_util'], bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[0,2].set_title('Buffer Utilization Distribution')
    axes[0,2].set_xlabel('Buffer Utilization')
    axes[0,2].grid(True, alpha=0.3)

    # Buffer occupancy
    axes[1,0].hist(df['bufocc_mean'], bins=30, alpha=0.7, edgecolor='black', color='red')
    axes[1,0].set_title('Mean Buffer Occupancy Distribution')
    axes[1,0].set_xlabel('Buffer Occupancy')
    axes[1,0].grid(True, alpha=0.3)

    # Rewards
    axes[1,1].hist(df['reward'], bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[1,1].set_title('Reward Distribution')
    axes[1,1].set_xlabel('Reward')
    axes[1,1].grid(True, alpha=0.3)

    # Capacity
    axes[1,2].hist(df['capacity_norm'], bins=30, alpha=0.7, edgecolor='black', color='brown')
    axes[1,2].set_title('Capacity Distribution')
    axes[1,2].set_xlabel('Capacity (normalized)')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/state_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. State evolution over time
    plt.figure(figsize=(15, 12))

    # Sample data if too large
    if len(df) > 5000:
        df_sample = df.sample(n=5000).sort_values('time')
    else:
        df_sample = df.sort_values('time')

    plt.subplot(3, 2, 1)
    plt.plot(df_sample['time'], df_sample['contacts_norm'], alpha=0.6, linewidth=0.5)
    plt.title('Contacts Over Time')
    plt.ylabel('Contacts (norm)')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 2)
    plt.plot(df_sample['time'], df_sample['pred'], alpha=0.6, linewidth=0.5, color='orange')
    plt.title('Prophet Predictions Over Time')
    plt.ylabel('Prophet Score')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 3)
    plt.plot(df_sample['time'], df_sample['self_buf_util'], alpha=0.6, linewidth=0.5, color='green')
    plt.title('Buffer Utilization Over Time')
    plt.ylabel('Buffer Util')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 4)
    plt.plot(df_sample['time'], df_sample['bufocc_mean'], alpha=0.6, linewidth=0.5, color='red')
    plt.title('Buffer Occupancy Over Time')
    plt.ylabel('Buffer Occupancy')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 5)
    plt.plot(df_sample['time'], df_sample['reward'], alpha=0.6, linewidth=0.5, color='purple')
    plt.title('Rewards Over Time')
    plt.ylabel('Reward')
    plt.xlabel('Time (seconds)')
    plt.grid(True, alpha=0.3)

    plt.subplot(3, 2, 6)
    # Reward rate (moving average)
    window = min(100, len(df_sample)//10)
    if window > 1:
        reward_ma = df_sample['reward'].rolling(window=window).mean()
        plt.plot(df_sample['time'], reward_ma, linewidth=2, color='darkred')
        plt.title(f'Reward Moving Average (window={window})')
        plt.ylabel('Avg Reward')
        plt.xlabel('Time (seconds)')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/state_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"State visualizations saved to: {output_dir}/")

def main():
    """Main function"""
    print("10M Buffer State Pattern Analyzer")
    print("=" * 40)

    # Use the most recent RLStateReport file
    state_file = "reports_drl_train/buf10/drl_train_buf10_100k_rng1_run1_RLStateReport.txt"

    if not os.path.exists(state_file):
        print(f"State file not found: {state_file}")
        return

    print(f"Loading state data from: {state_file}")
    df = extract_actions_from_rlstate(state_file)

    if df is None or len(df) == 0:
        print("No state data could be extracted")
        return

    # Analyze patterns
    analyze_state_patterns(df)

    # Create visualizations
    create_state_visualizations(df)

    print(f"\n✅ Analysis complete! Check reports_plots/buf10_states/ for visualizations")

if __name__ == "__main__":
    main()