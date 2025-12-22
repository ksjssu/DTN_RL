#!/usr/bin/env python3
"""
Ultra-Strong Exploration Monitor (ASCII only)
초강력 탐험 모니터링 - 유니코드 문제 해결
"""

import os
import time
import pandas as pd
import numpy as np
from datetime import datetime

def load_episode_data(buffer_size):
    """에피소드 데이터 로드"""
    file_path = f"reports_drl_train/buf{buffer_size:02d}/drl_episode_rewards.txt"

    if not os.path.exists(file_path):
        return None

    try:
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('episode'):
                    parts = line.split()
                    episode = int(parts[1])
                    delivery_rate = float(parts[5])
                    reward = int(parts[3])
                    data.append({
                        'episode': episode,
                        'delivery_rate': delivery_rate,
                        'reward': reward
                    })
        return pd.DataFrame(data) if data else None
    except Exception as e:
        return None

def analyze_ultra_exploration(df, target_performance=0.42):
    """초강력 탐험 효과 분석"""
    if df is None or len(df) < 2:
        return "데이터 부족", False, 0.0

    # 최근 성능
    recent = df.tail(min(5, len(df)))
    current_perf = recent['delivery_rate'].mean()
    std_dev = recent['delivery_rate'].std()

    # 탐험 강도 분석
    if pd.isna(std_dev):
        exploration_level = "계산불가"
        is_exploring = False
    elif std_dev < 0.001:
        exploration_level = "과수렴 상태 - 탐험 부족!"
        is_exploring = False
    elif std_dev < 0.005:
        exploration_level = "약한 탐험"
        is_exploring = True
    elif std_dev < 0.015:
        exploration_level = "적절한 탐험"
        is_exploring = True
    elif std_dev < 0.025:
        exploration_level = "강한 탐험 (초강력 모드 효과)"
        is_exploring = True
    else:
        exploration_level = "극강 탐험 (초강력 모드 최대 효과)"
        is_exploring = True

    # 목표 달성 여부
    target_achieved = current_perf >= target_performance

    return exploration_level, is_exploring, current_perf

def format_trend_analysis(df):
    """트렌드 분석"""
    if df is None or len(df) < 6:
        return "트렌드 분석 불가능"

    # 전반부 vs 후반부 비교
    mid = len(df) // 2
    first_half = df.head(mid)['delivery_rate'].mean()
    second_half = df.tail(len(df) - mid)['delivery_rate'].mean()
    diff = second_half - first_half

    if diff > 0.005:
        return f"강한 상승 추세 (+{diff:.4f})"
    elif diff > 0.002:
        return f"상승 추세 (+{diff:.4f})"
    elif diff < -0.005:
        return f"강한 하락 추세 ({diff:.4f})"
    elif diff < -0.002:
        return f"하락 추세 ({diff:.4f})"
    else:
        return f"안정 ({diff:+.4f})"

def clear_screen():
    """화면 클리어"""
    os.system('cls' if os.name == 'nt' else 'clear')

def ultra_monitor(buffer_size=10, interval=3, target_perf=0.42):
    """초강력 탐험 모니터"""
    print(f"Ultra-Strong Exploration Monitor - {buffer_size}M Buffer")
    print("Target: Beat heuristic d30 performance (>0.42)")
    print("Ctrl+C to stop")

    last_episode = 0
    baseline_episodes = []
    ultra_start_episode = None

    try:
        while True:
            df = load_episode_data(buffer_size)

            clear_screen()

            # 헤더
            timestamp = datetime.now().strftime("%H:%M:%S")
            print("=" * 60)
            print(f"ULTRA-STRONG EXPLORATION MONITOR - {buffer_size}M [{timestamp}]")
            print("=" * 60)

            if df is None or len(df) == 0:
                print("Waiting for training data...")
                time.sleep(interval)
                continue

            latest = df.iloc[-1]
            current_episode = latest['episode']

            # 새 에피소드 감지
            if current_episode > last_episode:
                new_episode_marker = " [NEW!]"
                last_episode = current_episode
            else:
                new_episode_marker = ""

            # 기본 정보
            print(f"Current Episode: {current_episode}{new_episode_marker}")
            print(f"Current Performance: {latest['delivery_rate']:.4f}")
            print(f"Current Reward: {latest['reward']:,}")

            # 초강력 탐험 효과 분석
            exploration_level, is_exploring, current_perf = analyze_ultra_exploration(df, target_perf)

            print(f"\nULTRA EXPLORATION STATUS:")
            print(f"  Level: {exploration_level}")
            print(f"  Exploring: {'YES' if is_exploring else 'NO'}")
            print(f"  Performance: {current_perf:.4f}")

            # 목표 달성 체크
            if current_perf >= target_perf:
                print(f"  TARGET ACHIEVED! Performance > {target_perf:.2f}")
                print(f"  Heuristic d30 performance beaten!")
            else:
                gap = target_perf - current_perf
                print(f"  Gap to target: {gap:.4f}")

            # 최근 통계
            recent_count = min(10, len(df))
            recent = df.tail(recent_count)

            print(f"\nRecent {recent_count} Episodes Stats:")
            print(f"  Average: {recent['delivery_rate'].mean():.4f}")

            std_val = recent['delivery_rate'].std()
            if pd.isna(std_val):
                std_str = "N/A"
            else:
                std_str = f"{std_val:.4f}"
            print(f"  Std Dev: {std_str}")
            print(f"  Range: [{recent['delivery_rate'].min():.4f}, {recent['delivery_rate'].max():.4f}]")

            # 트렌드 분석
            trend = format_trend_analysis(df)
            print(f"  Trend: {trend}")

            # 초강력 탐험 파라미터 효과 예측
            print(f"\nULTRA PARAMETERS ACTIVE:")
            print(f"  Entropy: 4.0x (ultra-strong)")
            print(f"  RND Strength: 3.0x (maximum)")
            print(f"  Forced Exploration: 40%")
            print(f"  Reset Interval: 8 episodes")

            # 다음 리셋 예측
            next_reset = 8 - (current_episode % 8)
            if next_reset == 8:
                next_reset = 0

            if next_reset == 0:
                print(f"  RESET EPISODE! Exploration will be maximized")
            elif next_reset <= 2:
                print(f"  Reset in {next_reset} episodes")

            # 성능 평가 및 권장사항
            if not pd.isna(std_val):
                if std_val < 0.001:
                    print(f"\nWARNING: Over-convergence detected!")
                    print(f"         Ultra parameters may need adjustment")
                elif std_val > 0.02:
                    print(f"\nGOOD: High exploration activity detected")
                    print(f"      Ultra-strong mode is working")
                elif current_perf > target_perf:
                    print(f"\nSUCCESS: Target performance achieved!")
                    print(f"         Ultra-exploration successful")

            # 베이스라인 vs 현재 비교
            if len(df) >= 5:
                if not baseline_episodes:
                    baseline_episodes = df.head(5)['delivery_rate'].tolist()
                    baseline_avg = np.mean(baseline_episodes)
                    print(f"\nBaseline (first 5 episodes): {baseline_avg:.4f}")

                baseline_avg = np.mean(baseline_episodes)
                improvement = current_perf - baseline_avg
                if improvement > 0:
                    print(f"Improvement from baseline: +{improvement:.4f}")
                else:
                    print(f"Change from baseline: {improvement:.4f}")

            print(f"\nNext update in {interval} seconds...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\nUltra-Strong Exploration Monitoring stopped")

        # 최종 결과 요약
        if df is not None and len(df) > 0:
            final_perf = df.iloc[-1]['delivery_rate']
            print(f"\nFINAL RESULTS:")
            print(f"  Episodes completed: {len(df)}")
            print(f"  Final performance: {final_perf:.4f}")
            if final_perf >= target_perf:
                print(f"  TARGET ACHIEVED: Performance > {target_perf:.2f}")
            else:
                print(f"  Target gap: {target_perf - final_perf:.4f}")

    except Exception as e:
        print(f"\nMonitoring error: {e}")

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='Ultra-Strong Exploration Monitor')
    parser.add_argument('--buffer', '-b', type=int, default=10, help='Buffer size')
    parser.add_argument('--interval', '-i', type=int, default=3, help='Update interval (seconds)')
    parser.add_argument('--target', '-t', type=float, default=0.42, help='Target performance')

    args = parser.parse_args()

    ultra_monitor(args.buffer, args.interval, args.target)

if __name__ == "__main__":
    main()