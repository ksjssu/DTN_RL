#!/usr/bin/env python3
"""
Quick Training Status Check
현재 훈련 상태를 빠르게 확인하는 도구
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def load_latest_episodes(buffer_size, count=10):
    """최근 에피소드 데이터 로드"""
    file_path = f"reports_drl_train/buf{buffer_size:02d}/drl_episode_rewards.txt"

    if not os.path.exists(file_path):
        return None

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

    if not data:
        return None

    df = pd.DataFrame(data)
    return df.tail(count)

def analyze_exploration_status(df):
    """탐험 상태 분석"""
    if df is None or len(df) < 5:
        return "데이터 부족"

    std_dev = df['delivery_rate'].std()

    if std_dev < 0.001:
        return "🚨 과수렴 (Over-converged)"
    elif std_dev < 0.005:
        return "🎯 안정적 (Stable)"
    elif std_dev < 0.015:
        return "🚀 탐험중 (Exploring)"
    else:
        return "⚡ 고강도 탐험 (High Exploration)"

def quick_check(buffer_size=10):
    """빠른 상태 체크"""
    print(f"{buffer_size}M 버퍼 현재 상태 체크")
    print("=" * 50)

    # 최근 데이터 로드
    recent_df = load_latest_episodes(buffer_size, 10)

    if recent_df is None:
        print("훈련 데이터가 없습니다.")
        return

    latest = recent_df.iloc[-1]

    # 기본 정보
    print(f"최신 에피소드: {latest['episode']}")
    print(f"현재 성능: {latest['delivery_rate']:.4f}")
    print(f"현재 보상: {latest['reward']:,}")

    # 최근 통계
    print(f"\n최근 10개 에피소드 통계:")
    print(f"   평균: {recent_df['delivery_rate'].mean():.4f}")
    print(f"   표준편차: {recent_df['delivery_rate'].std():.4f}")
    print(f"   최소값: {recent_df['delivery_rate'].min():.4f}")
    print(f"   최대값: {recent_df['delivery_rate'].max():.4f}")

    # 탐험 상태
    exploration_status = analyze_exploration_status(recent_df)
    print(f"\n탐험 상태: {exploration_status}")

    # 성능 트렌드
    if len(recent_df) >= 6:
        first_half = recent_df.head(5)['delivery_rate'].mean()
        second_half = recent_df.tail(5)['delivery_rate'].mean()
        trend_diff = second_half - first_half

        if trend_diff > 0.002:
            trend = "개선중"
        elif trend_diff < -0.002:
            trend = "하락중"
        else:
            trend = "안정"

        print(f"트렌드: {trend} ({trend_diff:+.4f})")

    # 권장사항
    std_dev = recent_df['delivery_rate'].std()
    performance = recent_df['delivery_rate'].mean()

    print(f"\n권장사항:")
    if std_dev < 0.001:
        print("   WARNING: 과수렴 상태! 탐험 강화 필요")
        print("   -> 엔트로피 계수 증가 또는 RND 강화")
    elif performance < 0.35:
        print("   CAUTION: 성능이 낮습니다")
        print("   -> 보상 구조 점검 또는 환경 설정 확인")
    elif performance > 0.42:
        print("   EXCELLENT: 우수한 성능! 현재 설정 유지")
    else:
        print("   OK: 정상 범위. 지속 모니터링 권장")

def compare_buffers():
    """여러 버퍼 크기 비교"""
    print("🔍 버퍼 크기별 비교 분석")
    print("=" * 50)

    buffer_sizes = [10, 40]  # 주요 버퍼 크기들

    for buf_size in buffer_sizes:
        recent_df = load_latest_episodes(buf_size, 10)

        if recent_df is not None:
            latest = recent_df.iloc[-1]
            print(f"\n{buf_size}M 버퍼:")
            print(f"   에피소드: {latest['episode']}")
            print(f"   성능: {latest['delivery_rate']:.4f}")
            print(f"   표준편차: {recent_df['delivery_rate'].std():.4f}")
            print(f"   상태: {analyze_exploration_status(recent_df)}")
        else:
            print(f"\n{buf_size}M 버퍼: 데이터 없음")

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='DTN DRL 빠른 상태 체크')
    parser.add_argument('--buffer', '-b', type=int, default=10, help='버퍼 크기')
    parser.add_argument('--compare', '-c', action='store_true', help='버퍼별 비교')

    args = parser.parse_args()

    print(f"체크 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if args.compare:
        compare_buffers()
    else:
        quick_check(args.buffer)

if __name__ == "__main__":
    main()