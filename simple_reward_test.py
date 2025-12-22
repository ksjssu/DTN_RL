#!/usr/bin/env python3
"""
간단한 보상 가중치 테스트 (함수만 추출)
"""
import os

# 기본 보상 가중치 (환경변수로 오버라이드 가능)
W_DELIVER = float(os.environ.get("W_DELIVER", "10.0"))
W_RELAY   = float(os.environ.get("W_RELAY",   "5.0"))
W_DROP    = float(os.environ.get("W_DROP",    "0.5"))
W_ABORT   = float(os.environ.get("W_ABORT",   "0.0"))
W_DELAY   = float(os.environ.get("W_DELAY",   "0.0"))

def get_delivery_weight_for_buffer_size(buffer_size_mb):
    """버퍼 크기별 전달 보상 가중치 계산"""
    if 9.0 <= buffer_size_mb <= 11.0:
        return 25.0  # 10M: 전달 성공시 +25점
    else:
        return W_DELIVER  # 다른 버퍼들: 기본값

def get_relay_weight_for_buffer_size(buffer_size_mb):
    """버퍼 크기별 릴레이 보상 가중치 계산"""
    if 9.0 <= buffer_size_mb <= 11.0:
        return 0.0  # 10M: 릴레이 보상 비활성화
    else:
        return W_RELAY  # 다른 버퍼들: 기본값

def get_drop_weight_for_buffer_size(buffer_size_mb):
    """버퍼 크기별 드롭 패널티 가중치 계산"""
    if 9.0 <= buffer_size_mb <= 11.0:
        return 30.0  # 10M: 드롭시 -30점
    else:
        return W_DROP  # 다른 버퍼들: 기본값

def test_buffers():
    for buffer_size in [10.0, 15.0, 30.0, 50.0]:
        print(f"\n=== {buffer_size}M 버퍼 ===")
        print(f"전달: +{get_delivery_weight_for_buffer_size(buffer_size):.1f}")
        print(f"릴레이: +{get_relay_weight_for_buffer_size(buffer_size):.1f}")
        print(f"드롭: -{get_drop_weight_for_buffer_size(buffer_size):.1f}")

if __name__ == "__main__":
    print("현재 보상 체계:")
    print(f"W_DELIVER={W_DELIVER}, W_RELAY={W_RELAY}, W_DROP={W_DROP}")
    test_buffers()