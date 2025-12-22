#!/usr/bin/env python3
"""
Real-time step reward plotting for 1000k simulation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import argparse
from pathlib import Path

def plot_step_rewards(buffer_size=10, refresh_interval=30, max_points=10000):
    """Plot step-by-step rewards with real-time updates."""

    # Setup file paths
    log_dir = Path("reports/reward_logs")
    log_file = log_dir / f"step_rewards_buf{buffer_size}M.csv"

    if not log_file.exists():
        print(f"Reward log file not found: {log_file}")
        print("Enable reward tracking with STEP_REWARD_TRACKING=true")
        return

    plt.ion()  # Interactive mode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    last_modified = 0

    print(f"실시간 리워드 모니터링 시작 - {buffer_size}M 버퍼")
    print(f"갱신 간격: {refresh_interval}초")
    print("Ctrl+C로 종료")

    try:
        while True:
            # Check if file was modified
            current_modified = os.path.getmtime(log_file)

            if current_modified > last_modified:
                try:
                    # Read the CSV data
                    df = pd.read_csv(log_file)

                    if len(df) == 0:
                        time.sleep(refresh_interval)
                        continue

                    # Limit points for performance
                    if len(df) > max_points:
                        df = df.iloc[-max_points:]

                    # Convert sim_time to hours for better readability
                    df['sim_hours'] = df['sim_time'] / 3600

                    # Clear previous plots
                    ax1.clear()
                    ax2.clear()

                    # Plot 1: Total reward over time
                    ax1.plot(df['sim_hours'], df['total_reward'], 'b-', alpha=0.7, linewidth=1)
                    ax1.set_xlabel('시뮬레이션 시간 (시간)')
                    ax1.set_ylabel('총 리워드')
                    ax1.set_title(f'{buffer_size}M 버퍼 - 시간별 총 리워드')
                    ax1.grid(True, alpha=0.3)

                    # Add moving average
                    if len(df) > 100:
                        window = min(100, len(df) // 10)
                        moving_avg = df['total_reward'].rolling(window=window, center=True).mean()
                        ax1.plot(df['sim_hours'], moving_avg, 'r-', linewidth=2, alpha=0.8, label=f'{window}점 이동평균')
                        ax1.legend()

                    # Plot 2: Average reward per host
                    ax2.plot(df['sim_hours'], df['avg_reward_per_host'], 'g-', alpha=0.7, linewidth=1)
                    ax2.set_xlabel('시뮬레이션 시간 (시간)')
                    ax2.set_ylabel('호스트당 평균 리워드')
                    ax2.set_title(f'{buffer_size}M 버퍼 - 시간별 호스트당 평균 리워드')
                    ax2.grid(True, alpha=0.3)

                    # Add statistics
                    latest_hour = df['sim_hours'].iloc[-1]
                    latest_total = df['total_reward'].iloc[-1]
                    latest_avg = df['avg_reward_per_host'].iloc[-1]
                    total_points = len(df)

                    fig.suptitle(f'1000K 시뮬레이션 실시간 리워드 모니터링\\n'
                               f'현재: {latest_hour:.1f}시간, 총 리워드: {latest_total:.2f}, '
                               f'평균: {latest_avg:.4f}, 데이터포인트: {total_points}',
                               fontsize=12)

                    plt.tight_layout()
                    plt.draw()

                    last_modified = current_modified

                    print(f"\\r[{time.strftime('%H:%M:%S')}] "
                          f"시간: {latest_hour:.1f}h, 총 리워드: {latest_total:.2f}, "
                          f"평균: {latest_avg:.4f}, 포인트: {total_points}", end="")

                except Exception as e:
                    print(f"\\n데이터 읽기 오류: {e}")

            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\\n\\n모니터링 종료")

        # Save final plot
        output_dir = Path("reports/reward_plots")
        output_dir.mkdir(exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"step_rewards_buf{buffer_size}M_{timestamp}.png"

        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"최종 그래프 저장: {output_file}")

    finally:
        plt.ioff()
        plt.show()

def main():
    parser = argparse.ArgumentParser(description="1000k 시뮬레이션 실시간 리워드 모니터링")
    parser.add_argument("--buffer", "-b", type=int, default=10, help="버퍼 크기 (MB)")
    parser.add_argument("--interval", "-i", type=int, default=30, help="갱신 간격 (초)")
    parser.add_argument("--max-points", "-m", type=int, default=10000, help="최대 표시 포인트")

    args = parser.parse_args()

    plot_step_rewards(
        buffer_size=args.buffer,
        refresh_interval=args.interval,
        max_points=args.max_points
    )

if __name__ == "__main__":
    main()