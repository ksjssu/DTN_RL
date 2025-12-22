#!/usr/bin/env python3
"""
Real-time Training Monitor for DTN DRL
탐험 강화 후 실시간 성능 변화를 모니터링하는 도구
"""

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import threading
from datetime import datetime

class RealTimeMonitor:
    def __init__(self, buffer_size=10, refresh_interval=5):
        """
        실시간 모니터 초기화

        Args:
            buffer_size: 모니터링할 버퍼 크기
            refresh_interval: 업데이트 간격 (초)
        """
        self.buffer_size = buffer_size
        self.refresh_interval = refresh_interval
        self.episode_file = f"reports_drl_train/buf{buffer_size:02d}/drl_episode_rewards.txt"
        self.running = False

        # 이전 상태 저장
        self.last_episode = 0
        self.baseline_performance = None  # 탐험 강화 이전 성능
        self.exploration_start_episode = None

        # 성능 추적
        self.performance_history = []
        self.std_history = []
        self.exploration_events = []

        print(f"🔍 실시간 모니터 초기화: {buffer_size}M 버퍼")
        print(f"   파일: {self.episode_file}")
        print(f"   업데이트 간격: {refresh_interval}초")

    def load_current_data(self):
        """현재 에피소드 데이터 로드"""
        if not os.path.exists(self.episode_file):
            return None

        try:
            data = []
            with open(self.episode_file, 'r') as f:
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
            print(f"데이터 로드 오류: {e}")
            return None

    def detect_exploration_boost(self, df):
        """탐험 강화 시점 감지"""
        if len(df) < 10:
            return False

        # 최근 10개 에피소드의 표준편차 확인
        recent = df.tail(10)
        current_std = recent['delivery_rate'].std()

        # 이전 기준 대비 표준편차 증가 확인
        if len(self.std_history) > 0:
            prev_avg_std = np.mean(self.std_history[-5:]) if len(self.std_history) >= 5 else self.std_history[-1]
            if current_std > prev_avg_std * 2:  # 2배 이상 증가
                return True

        return current_std > 0.005  # 절대적 기준

    def analyze_current_state(self, df):
        """현재 상태 분석"""
        if df is None or len(df) == 0:
            return None

        recent_episodes = df.tail(10)
        latest_episode = df.iloc[-1]

        # 기본 통계
        stats = {
            'latest_episode': latest_episode['episode'],
            'latest_performance': latest_episode['delivery_rate'],
            'latest_reward': latest_episode['reward'],
            'recent_mean': recent_episodes['delivery_rate'].mean(),
            'recent_std': recent_episodes['delivery_rate'].std(),
            'recent_min': recent_episodes['delivery_rate'].min(),
            'recent_max': recent_episodes['delivery_rate'].max(),
            'trend': 'unknown'
        }

        # 트렌드 분석
        if len(recent_episodes) >= 5:
            first_half = recent_episodes.head(5)['delivery_rate'].mean()
            second_half = recent_episodes.tail(5)['delivery_rate'].mean()
            if second_half > first_half + 0.002:
                stats['trend'] = 'improving'
            elif second_half < first_half - 0.002:
                stats['trend'] = 'declining'
            else:
                stats['trend'] = 'stable'

        # 탐험 상태 감지
        stats['exploration_active'] = self.detect_exploration_boost(df)

        return stats

    def print_status_update(self, stats):
        """상태 업데이트 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        print(f"\n{'='*60}")
        print(f"🕐 {timestamp} - 에피소드 {stats['latest_episode']} 업데이트")
        print(f"{'='*60}")

        # 현재 성능
        print(f"📈 현재 성능:")
        print(f"   전달률: {stats['latest_performance']:.4f}")
        print(f"   보상: {stats['latest_reward']:,}")

        # 최근 통계 (10개 에피소드)
        print(f"\n📊 최근 통계 (10개 에피소드):")
        print(f"   평균: {stats['recent_mean']:.4f}")
        print(f"   표준편차: {stats['recent_std']:.4f}")
        print(f"   범위: [{stats['recent_min']:.4f}, {stats['recent_max']:.4f}]")
        print(f"   트렌드: {stats['trend'].upper()}")

        # 탐험 상태
        exploration_icon = "🚀" if stats['exploration_active'] else "🎯"
        exploration_status = "활성화됨" if stats['exploration_active'] else "정상"
        print(f"\n{exploration_icon} 탐험 상태: {exploration_status}")

        # 베이스라인과 비교
        if self.baseline_performance:
            diff = stats['recent_mean'] - self.baseline_performance
            trend_icon = "📈" if diff > 0 else "📉" if diff < 0 else "➡️"
            print(f"\n{trend_icon} 베이스라인 대비: {diff:+.4f}")

        # 성능 단계 평가
        if stats['recent_mean'] > 0.42:
            grade = "🌟 우수"
        elif stats['recent_mean'] > 0.40:
            grade = "✅ 양호"
        elif stats['recent_mean'] > 0.38:
            grade = "⚠️ 보통"
        else:
            grade = "🚨 개선필요"

        print(f"\n성능 평가: {grade}")

    def save_monitoring_report(self, df):
        """모니터링 리포트 저장"""
        if df is None:
            return

        try:
            # 시각화 생성
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

            # 1. 성능 추이
            ax1.plot(df['episode'], df['delivery_rate'], 'b-', alpha=0.7)
            ax1.set_title('Performance Trend')
            ax1.set_xlabel('Episode')
            ax1.set_ylabel('Delivery Rate')
            ax1.grid(True, alpha=0.3)

            # 베이스라인 표시
            if self.baseline_performance:
                ax1.axhline(y=self.baseline_performance, color='red', linestyle='--',
                           label=f'Baseline: {self.baseline_performance:.3f}')
                ax1.legend()

            # 2. 표준편차 추이 (윈도우 크기 10)
            if len(df) >= 10:
                rolling_std = df['delivery_rate'].rolling(window=10).std()
                ax2.plot(df['episode'], rolling_std, 'g-', alpha=0.7)
                ax2.set_title('Performance Stability (Rolling Std)')
                ax2.set_xlabel('Episode')
                ax2.set_ylabel('Standard Deviation')
                ax2.grid(True, alpha=0.3)

                # 과수렴 경계선
                ax2.axhline(y=0.001, color='red', linestyle='--', label='Over-convergence')
                ax2.axhline(y=0.005, color='orange', linestyle='--', label='Target')
                ax2.legend()

            # 3. 최근 분포
            recent_data = df.tail(20)['delivery_rate']
            ax3.hist(recent_data, bins=10, alpha=0.7, edgecolor='black')
            ax3.set_title('Recent Performance Distribution (Last 20)')
            ax3.set_xlabel('Delivery Rate')
            ax3.set_ylabel('Frequency')
            ax3.grid(True, alpha=0.3)

            # 4. 보상 추이
            ax4.plot(df['episode'], df['reward'], 'purple', alpha=0.7)
            ax4.set_title('Reward Trend')
            ax4.set_xlabel('Episode')
            ax4.set_ylabel('Reward')
            ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            os.makedirs('reports/monitoring', exist_ok=True)
            plt.savefig('reports/monitoring/realtime_training_status.png', dpi=150, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"리포트 저장 오류: {e}")

    def monitor_loop(self):
        """모니터링 메인 루프"""
        print(f"\n🚀 실시간 모니터링 시작!")
        print(f"   Ctrl+C를 눌러서 중단할 수 있습니다.")

        try:
            while self.running:
                df = self.load_current_data()

                if df is not None and len(df) > 0:
                    current_episode = df.iloc[-1]['episode']

                    # 새로운 에피소드가 추가되었을 때만 업데이트
                    if current_episode > self.last_episode:
                        stats = self.analyze_current_state(df)

                        if stats:
                            self.print_status_update(stats)

                            # 베이스라인 설정 (처음 10개 에피소드 평균)
                            if self.baseline_performance is None and len(df) >= 10:
                                self.baseline_performance = df.head(10)['delivery_rate'].mean()
                                print(f"\n📊 베이스라인 성능 설정: {self.baseline_performance:.4f}")

                            # 성능 이력 저장
                            self.performance_history.append(stats['recent_mean'])
                            self.std_history.append(stats['recent_std'])

                            # 리포트 저장
                            self.save_monitoring_report(df)

                        self.last_episode = current_episode

                time.sleep(self.refresh_interval)

        except KeyboardInterrupt:
            print(f"\n\n⏹️ 모니터링 중단됨")
        except Exception as e:
            print(f"\n❌ 모니터링 오류: {e}")

    def start(self):
        """모니터링 시작"""
        self.running = True
        self.monitor_loop()

    def stop(self):
        """모니터링 중단"""
        self.running = False

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='DTN DRL 실시간 모니터링')
    parser.add_argument('--buffer', '-b', type=int, default=10, help='버퍼 크기 (MB)')
    parser.add_argument('--interval', '-i', type=int, default=5, help='업데이트 간격 (초)')

    args = parser.parse_args()

    monitor = RealTimeMonitor(buffer_size=args.buffer, refresh_interval=args.interval)
    monitor.start()

if __name__ == "__main__":
    main()