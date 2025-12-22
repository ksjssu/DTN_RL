#!/usr/bin/env python3
"""
Action Pattern Visualization Tool for DTN DRL
Analyzes and visualizes DRL agent actions to identify patterns and issues.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
from pathlib import Path

def load_action_data(action_file="reports/action_logs/actions.csv"):
    """Load action data from CSV file"""
    if not os.path.exists(action_file):
        print(f"Action file not found: {action_file}")
        return None

    try:
        # CSV format: timestamp,key,action,intrinsic,obs1,obs2,obs3,obs4,obs5
        df = pd.read_csv(action_file, header=None,
                        names=['timestamp', 'key', 'action', 'intrinsic',
                               'contacts', 'prophet', 'buffer', 'capacity', 'self_util'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"Error loading action data: {e}")
        return None

def analyze_action_patterns(df):
    """Analyze action patterns for potential issues"""
    print("=== ACTION PATTERN ANALYSIS ===")

    # Basic statistics
    print(f"Total actions recorded: {len(df)}")
    print(f"Action range: [{df['action'].min():.3f}, {df['action'].max():.3f}]")
    print(f"Action mean: {df['action'].mean():.3f}, std: {df['action'].std():.3f}")

    # Check for tap dancing (repeated identical actions)
    action_counts = df['action'].value_counts()
    most_common = action_counts.head()
    print(f"\nMost common actions:")
    for action, count in most_common.items():
        percentage = (count / len(df)) * 100
        print(f"  {action:.3f}: {count} times ({percentage:.1f}%)")

    # Check for exploration vs exploitation
    action_variance = df['action'].var()
    print(f"\nAction variance: {action_variance:.6f}")
    if action_variance < 0.01:
        print("⚠️  WARNING: Very low action variance - possible convergence to single action")
    elif action_variance > 1.0:
        print("✅ Good action diversity")

    # RND intrinsic reward analysis
    if 'intrinsic' in df.columns and df['intrinsic'].sum() > 0:
        print(f"\nIntrinsic rewards:")
        print(f"  Mean: {df['intrinsic'].mean():.4f}")
        print(f"  Max: {df['intrinsic'].max():.4f}")
        print(f"  Non-zero rate: {(df['intrinsic'] > 0).mean()*100:.1f}%")

def create_visualizations(df, output_dir="reports/action_viz"):
    """Create action visualization plots"""
    os.makedirs(output_dir, exist_ok=True)

    # Set style
    plt.style.use('default')
    sns.set_palette("husl")

    # 1. Action distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(df['action'], bins=50, alpha=0.7, edgecolor='black')
    plt.title('DRL Action Distribution')
    plt.xlabel('Action Value (Delta)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/action_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Action over time
    plt.figure(figsize=(12, 8))

    # Sample data if too large
    if len(df) > 10000:
        df_sample = df.sample(n=10000).sort_values('timestamp')
    else:
        df_sample = df.sort_values('timestamp')

    plt.subplot(2, 1, 1)
    plt.plot(df_sample['timestamp'], df_sample['action'], alpha=0.6, linewidth=0.5)
    plt.title('Action Values Over Time')
    plt.ylabel('Action (Delta)')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    if 'intrinsic' in df.columns:
        plt.plot(df_sample['timestamp'], df_sample['intrinsic'], alpha=0.6, linewidth=0.5, color='orange')
        plt.title('Intrinsic Rewards Over Time')
        plt.ylabel('Intrinsic Reward')
    plt.xlabel('Time')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_timeline.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Action vs State relationship
    plt.figure(figsize=(15, 10))

    state_vars = ['contacts', 'prophet', 'buffer', 'capacity', 'self_util']
    for i, var in enumerate(state_vars):
        plt.subplot(2, 3, i+1)
        plt.scatter(df[var], df['action'], alpha=0.3, s=1)
        plt.xlabel(var.title())
        plt.ylabel('Action')
        plt.title(f'Action vs {var.title()}')
        plt.grid(True, alpha=0.3)

    # 6th plot: Action vs Intrinsic Reward
    if 'intrinsic' in df.columns:
        plt.subplot(2, 3, 6)
        plt.scatter(df['intrinsic'], df['action'], alpha=0.3, s=1, color='red')
        plt.xlabel('Intrinsic Reward')
        plt.ylabel('Action')
        plt.title('Action vs Intrinsic Reward')
        plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/action_vs_state.png", dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Visualizations saved to: {output_dir}/")

def main():
    """Main analysis and visualization function"""
    print("DTN DRL Action Pattern Visualizer")
    print("=" * 40)

    # Load data
    df = load_action_data()
    if df is None:
        print("No action data available. Enable action logging with:")
        print("set PPO_DEBUG_ACTIONS=true")
        return

    # Analyze patterns
    analyze_action_patterns(df)

    # Create visualizations
    create_visualizations(df)

    print("\n✅ Analysis complete! Check reports/action_viz/ for plots.")

if __name__ == "__main__":
    main()