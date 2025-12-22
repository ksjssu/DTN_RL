#!/usr/bin/env python3
"""
DRL Training Log Tracker
DRL 서버 로그에서 탐험 강화 관련 메시지를 추적하는 도구
"""

import re
import os
import time
from datetime import datetime
from collections import defaultdict

class LogTracker:
    def __init__(self, log_patterns=None):
        """로그 추적기 초기화"""
        self.log_patterns = log_patterns or {
            'exploration': r'\[EXPLORATION\].*강제 탐험 (\d+)회',
            'reset': r'\[RESET\].*에피소드 (\d+): 탐험 리셋',
            'dynamic': r'\[DYNAMIC\].*Buffer:(\d+)M.*Multiplier:([0-9.]+)',
            'entropy': r'\[ENTROPY\].*entropy_coef=([0-9.]+)',
            'buffer_strategy': r'\[BUFFER STRATEGY\].*Buffer size: ([0-9.]+)M'
        }

        self.event_counts = defaultdict(int)
        self.last_events = defaultdict(list)

    def scan_recent_output(self, text):
        """최근 출력에서 패턴 검색"""
        events = []

        for event_type, pattern in self.log_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)

            for match in matches:
                if event_type == 'exploration':
                    count = int(match)
                    events.append({
                        'type': 'exploration',
                        'message': f'강제 탐험 {count}회 실행됨',
                        'data': {'count': count}
                    })

                elif event_type == 'reset':
                    episode = int(match)
                    events.append({
                        'type': 'reset',
                        'message': f'에피소드 {episode}에서 탐험 리셋',
                        'data': {'episode': episode}
                    })

                elif event_type == 'dynamic':
                    buffer_size, multiplier = match
                    events.append({
                        'type': 'dynamic',
                        'message': f'{buffer_size}M 버퍼 동적 보상 배수: {multiplier}',
                        'data': {'buffer_size': int(buffer_size), 'multiplier': float(multiplier)}
                    })

                elif event_type == 'entropy':
                    entropy_coef = float(match)
                    events.append({
                        'type': 'entropy',
                        'message': f'엔트로피 계수: {entropy_coef:.4f}',
                        'data': {'entropy_coef': entropy_coef}
                    })

                elif event_type == 'buffer_strategy':
                    buffer_size = float(match)
                    events.append({
                        'type': 'buffer_strategy',
                        'message': f'버퍼 전략 적용: {buffer_size}M',
                        'data': {'buffer_size': buffer_size}
                    })

        return events

    def monitor_file(self, file_path, tail_lines=100):
        """파일 모니터링 (tail -f 방식)"""
        if not os.path.exists(file_path):
            print(f"파일이 존재하지 않습니다: {file_path}")
            return

        print(f"📋 로그 파일 모니터링 시작: {file_path}")
        print("   Ctrl+C로 중단 가능")

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 파일 끝으로 이동
                f.seek(0, 2)

                while True:
                    line = f.readline()
                    if line:
                        events = self.scan_recent_output(line)
                        for event in events:
                            self.print_event(event)
                    else:
                        time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n📋 로그 모니터링 중단됨")
        except Exception as e:
            print(f"❌ 로그 모니터링 오류: {e}")

    def print_event(self, event):
        """이벤트 출력"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        icons = {
            'exploration': '🚀',
            'reset': '🔄',
            'dynamic': '⚡',
            'entropy': '🎲',
            'buffer_strategy': '📊'
        }

        icon = icons.get(event['type'], '📝')
        print(f"{timestamp} {icon} {event['message']}")

        # 카운트 업데이트
        self.event_counts[event['type']] += 1

        # 최근 이벤트 저장 (최대 10개)
        self.last_events[event['type']].append(event)
        if len(self.last_events[event['type']]) > 10:
            self.last_events[event['type']].pop(0)

    def print_summary(self):
        """이벤트 요약 출력"""
        print("\n" + "="*50)
        print("📊 로그 이벤트 요약")
        print("="*50)

        if not any(self.event_counts.values()):
            print("감지된 이벤트가 없습니다.")
            return

        for event_type, count in self.event_counts.items():
            icon = {
                'exploration': '🚀',
                'reset': '🔄',
                'dynamic': '⚡',
                'entropy': '🎲',
                'buffer_strategy': '📊'
            }.get(event_type, '📝')

            print(f"{icon} {event_type.title()}: {count}회")

        # 최근 이벤트들
        print(f"\n📋 최근 이벤트들:")
        all_recent = []
        for event_type, events in self.last_events.items():
            for event in events[-3:]:  # 각 타입별 최근 3개
                all_recent.append(event)

        all_recent.sort(key=lambda x: x.get('timestamp', 0), reverse=True)

        for event in all_recent[:5]:  # 최근 5개만 표시
            icon = {
                'exploration': '🚀',
                'reset': '🔄',
                'dynamic': '⚡',
                'entropy': '🎲',
                'buffer_strategy': '📊'
            }.get(event['type'], '📝')
            print(f"  {icon} {event['message']}")

def monitor_console_output():
    """콘솔 출력 직접 모니터링 (실시간)"""
    print("🔍 DRL 서버 출력 패턴 감지 대기중...")
    print("   DRL 서버를 실행하면 관련 로그가 표시됩니다.")
    print("   Ctrl+C로 중단")

    tracker = LogTracker()

    try:
        print("\n" + "="*50)
        print("📊 감지 대기중인 패턴들:")
        print("🚀 [EXPLORATION] - 강제 탐험 실행")
        print("🔄 [RESET] - 주기적 탐험 리셋")
        print("⚡ [DYNAMIC] - 동적 보상 적용")
        print("🎲 [ENTROPY] - 엔트로피 계수 변화")
        print("📊 [BUFFER STRATEGY] - 버퍼 전략 적용")
        print("="*50)

        while True:
            time.sleep(1)
            # 실제로는 stdin이나 파이프를 통해 받아야 하지만
            # 여기서는 사용자 안내만 제공
            pass

    except KeyboardInterrupt:
        print("\n📊 모니터링 중단됨")
        tracker.print_summary()

def main():
    """메인 함수"""
    import argparse

    parser = argparse.ArgumentParser(description='DRL 훈련 로그 추적')
    parser.add_argument('--file', '-f', type=str, help='모니터링할 로그 파일')
    parser.add_argument('--console', '-c', action='store_true', help='콘솔 출력 모니터링')

    args = parser.parse_args()

    if args.file:
        tracker = LogTracker()
        tracker.monitor_file(args.file)
    elif args.console:
        monitor_console_output()
    else:
        print("📋 DRL 로그 추적 도구")
        print("사용법:")
        print("  python log_tracker.py --console  # 콘솔 출력 모니터링")
        print("  python log_tracker.py --file <파일명>  # 파일 모니터링")

if __name__ == "__main__":
    main()