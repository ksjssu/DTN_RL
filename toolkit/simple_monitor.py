#!/usr/bin/env python3
"""
Simple Real-time Monitor for DTN DRL
간단하고 효과적인 실시간 모니터링 도구
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

def analyze_exploration_status(df):
    """탐험 상태 분석"""
    if df is None or len(df) < 2:
        return "데이터 부족", "gray"

    recent = df.tail(min(10, len(df)))
    std_dev = recent['delivery_rate'].std()

    if pd.isna(std_dev):
        return "계산 불가", "gray"
    elif std_dev < 0.001:
        return "과수렴 상태", "red"
    elif std_dev < 0.005:
        return "안정적", "yellow"
    elif std_dev < 0.015:
        return "탐험중", "green"
    else:
        return "고강도 탐험", "blue"

def format_trend(df):
    """트렌드 분석"""
    if df is None or len(df) < 4:
        return "판단 불가"

    recent = df.tail(min(6, len(df)))
    if len(recent) < 4:
        return "데이터 부족"

    mid = len(recent) // 2
    first_half = recent.head(mid)['delivery_rate'].mean()
    second_half = recent.tail(len(recent) - mid)['delivery_rate'].mean()
    diff = second_half - first_half

    if diff > 0.003:
        return f"상승 (+{diff:.4f})"
    elif diff < -0.003:
        return f"하락 ({diff:.4f})"
    else:
        return f"안정 ({diff:+.4f})"

def clear_screen():
    """화면 클리어 (Windows용)"""
    os.system('cls' if os.name == 'nt' else 'clear')

def simple_monitor(buffer_size=10, interval=3):
    """간단한 실시간 모니터"""
    print(f"DRL 실시간 모니터 - {buffer_size}M 버퍼")
    print("Ctrl+C로 중단")

    last_episode = 0
    baseline = None

    try:
        while True:
            # 데이터 로드
            df = load_episode_data(buffer_size)

            # 화면 클리어
            clear_screen()

            # 헤더
            timestamp = datetime.now().strftime("%H:%M:%S")
            print("=" * 60)
            print(f"DTN DRL 실시간 모니터 - {buffer_size}M 버퍼 [{timestamp}]")
            print("=" * 60)

            if df is None or len(df) == 0:
                print("훈련 데이터 대기 중...")
                time.sleep(interval)
                continue

            # 최신 데이터
            latest = df.iloc[-1]
            current_episode = latest['episode']

            # 새 에피소드 알림
            if current_episode > last_episode:
                new_episode_marker = " [NEW!]"
                last_episode = current_episode
            else:
                new_episode_marker = ""

            # 현재 상태
            print(f"에피소드: {current_episode}{new_episode_marker}")
            print(f"현재 성능: {latest['delivery_rate']:.4f}")
            print(f"현재 보상: {latest['reward']:,}")

            # 최근 통계
            recent_count = min(10, len(df))
            recent = df.tail(recent_count)

            print(f"\n최근 {recent_count}개 에피소드 통계:")
            print(f"  평균: {recent['delivery_rate'].mean():.4f}")

            std_val = recent['delivery_rate'].std()
            if pd.isna(std_val):
                std_str = "계산불가"
            else:
                std_str = f"{std_val:.4f}"
            print(f"  표준편차: {std_str}")

            print(f"  범위: [{recent['delivery_rate'].min():.4f}, {recent['delivery_rate'].max():.4f}]")

            # 탐험 상태
            exploration_status, color = analyze_exploration_status(df)
            print(f"\n탐험 상태: {exploration_status}")

            # 트렌드
            trend = format_trend(df)
            print(f"트렌드: {trend}")

            # 베이스라인 설정 및 비교
            if baseline is None and len(df) >= 5:
                baseline = df.head(5)['delivery_rate'].mean()
                print(f"\n베이스라인 설정: {baseline:.4f} (첫 5개 에피소드)")

            if baseline is not None:
                current_avg = recent['delivery_rate'].mean()
                diff = current_avg - baseline
                if diff > 0:
                    print(f"베이스라인 대비: +{diff:.4f} (개선)")
                elif diff < -0.005:
                    print(f"베이스라인 대비: {diff:.4f} (하락)")
                else:
                    print(f"베이스라인 대비: {diff:+.4f} (유지)")

            # 성능 평가
            current_perf = latest['delivery_rate']
            if current_perf > 0.42:
                grade = "우수"
            elif current_perf > 0.40:
                grade = "양호"
            elif current_perf > 0.37:
                grade = "보통"
            else:
                grade = "개선필요"

            print(f"\n성능 등급: {grade}")

            # 중요 이벤트 예측
            next_reset = 15 - (current_episode % 15)
            if next_reset == 15:
                next_reset = 0

            if next_reset <= 3 and next_reset > 0:
                print(f"\n주의: {next_reset}개 에피소드 후 탐험 리셋 예정!")
            elif next_reset == 0:
                print(f"\n현재 탐험 리셋 에피소드입니다!")

            # 권장사항
            if not pd.isna(std_val):
                if std_val < 0.001:
                    print(f"\n권장: 과수렴 상태 - 탐험 강화 필요")
                elif std_val > 0.02:
                    print(f"\n정보: 활발한 탐험 중 - 성능 변동 정상")

            print(f"\n다음 업데이트: {interval}초 후...")
            time.sleep(interval)

    except KeyboardInterrupt:
        print(f"\n\n모니터링 종료")
    except Exception as e:
        print(f"\n오류 발생: {e}")

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='간단한 DRL 실시간 모니터')
    parser.add_argument('--buffer', '-b', type=int, default=10, help='버퍼 크기')
    parser.add_argument('--interval', '-i', type=int, default=3, help='업데이트 간격(초)')

    args = parser.parse_args()

    simple_monitor(args.buffer, args.interval)

if __name__ == "__main__":
    main()