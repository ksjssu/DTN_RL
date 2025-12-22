#!/usr/bin/env python3
"""
30M 버퍼의 실제 보상 가중치 테스트
"""
import sys
sys.path.append('toolkit')

from drl_server import (
    get_delivery_weight_for_buffer_size,
    get_relay_weight_for_buffer_size,
    get_drop_weight_for_buffer_size,
    get_abort_weight_for_buffer_size,
    get_delay_weight_for_buffer_size
)

def test_30m_rewards():
    buffer_size = 30.0

    print(f"=== 30M 버퍼 ({buffer_size}MB) 보상 가중치 ===")
    print(f"전달 보상: {get_delivery_weight_for_buffer_size(buffer_size):.1f}점")
    print(f"릴레이 보상: {get_relay_weight_for_buffer_size(buffer_size):.1f}점")
    print(f"드롭 패널티: -{get_drop_weight_for_buffer_size(buffer_size):.1f}점")
    print(f"어보트 패널티: -{get_abort_weight_for_buffer_size(buffer_size):.1f}점")
    print(f"지연 패널티: -{get_delay_weight_for_buffer_size(buffer_size):.1f}점")

    print(f"\n=== 예상 보상 계산 ===")
    print(f"릴레이 1개 → +{get_relay_weight_for_buffer_size(buffer_size):.1f}점")
    print(f"드롭 1개 → -{get_drop_weight_for_buffer_size(buffer_size):.1f}점")

    # 실제 확인된 데이터와 비교
    print(f"\n=== 실제 확인된 30M 데이터 ===")
    print(f"릴레이 1개 → +5.0점 (실제)")
    print(f"드롭 1개 → -0.5점 (실제)")

if __name__ == "__main__":
    test_30m_rewards()