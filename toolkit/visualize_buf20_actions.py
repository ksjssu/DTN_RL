#!/usr/bin/env python3
"""
20M 버퍼 DRL 액션 시각화
네트워크 평균 버퍼 점유율에 따른 릴레이 보상 전략 분석
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_action_logs():
    """액션 로그 파일 로드"""
    log_file = "reports/action_logs/actions.csv"

    if not os.path.exists(log_file):
        print(f"액션 로그 파일이 없습니다: {log_file}")
        return None

    df = pd.read_csv(log_file)
    print(f"로드된 액션 수: {len(df)}")
    return df

def calculate_relay_reward(buf_mean, relayed=1.0):
    """20M 릴레이 보상 계산"""
    if buf_mean <= 0.4:
        return relayed * 4.0
    elif buf_mean >= 0.7:
        return -relayed * 2.0
    else:
        return 0.0

def plot_reward_strategy():
    """20M 보상 전략 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 네트워크 평균 버퍼 점유율별 릴레이 보상
    buf_mean_range = np.linspace(0, 1, 100)
    relay_rewards = [calculate_relay_reward(bm) for bm in buf_mean_range]

    ax = axes[0, 0]
    ax.plot(buf_mean_range, relay_rewards, linewidth=3, color='#2ecc71')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.axvline(x=0.4, color='red', linestyle='--', alpha=0.5, label='buf_mean=0.4 (보상 시작)')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='buf_mean=0.7 (패널티 시작)')
    ax.fill_between(buf_mean_range, 0, relay_rewards,
                     where=(np.array(relay_rewards) > 0), alpha=0.3, color='green', label='보상 영역')
    ax.fill_between(buf_mean_range, 0, relay_rewards,
                     where=(np.array(relay_rewards) < 0), alpha=0.3, color='red', label='패널티 영역')
    ax.set_xlabel('네트워크 평균 버퍼 점유율', fontsize=12, fontweight='bold')
    ax.set_ylabel('릴레이당 보상 (relayed=1)', fontsize=12, fontweight='bold')
    ax.set_title('20M 버퍼: 네트워크 중심 릴레이 보상 전략', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 릴레이 수에 따른 총 보상
    ax = axes[0, 1]
    relay_counts = np.arange(0, 21)
    for buf_mean_val, label, color in [(0.3, 'buf_mean=0.3 (여유)', 'green'),
                                        (0.5, 'buf_mean=0.5 (중간)', 'gray'),
                                        (0.8, 'buf_mean=0.8 (혼잡)', 'red')]:
        total_rewards = [calculate_relay_reward(buf_mean_val, r) for r in relay_counts]
        ax.plot(relay_counts, total_rewards, marker='o', label=label, color=color, linewidth=2)

    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('릴레이 메시지 수', fontsize=12, fontweight='bold')
    ax.set_ylabel('총 릴레이 보상', fontsize=12, fontweight='bold')
    ax.set_title('릴레이 수에 따른 총 보상 변화', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 10M vs 20M 전략 비교
    ax = axes[1, 0]
    buf_mean_range = np.linspace(0, 1, 100)

    # 20M: 네트워크 평균 기준
    rewards_20m = [calculate_relay_reward(bm) for bm in buf_mean_range]

    # 10M: 드랍 패널티만 (릴레이 보상 없음)
    rewards_10m = [0.0] * len(buf_mean_range)

    ax.plot(buf_mean_range, rewards_20m, linewidth=3, label='20M (네트워크 중심)', color='#3498db')
    ax.plot(buf_mean_range, rewards_10m, linewidth=3, label='10M (릴레이 보상 없음)',
            color='#e74c3c', linestyle='--')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('네트워크 평균 버퍼 점유율', fontsize=12, fontweight='bold')
    ax.set_ylabel('릴레이당 보상', fontsize=12, fontweight='bold')
    ax.set_title('10M vs 20M 릴레이 보상 전략 비교', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 4. 전략 설명 텍스트
    ax = axes[1, 1]
    ax.axis('off')

    strategy_text = """
    20M 버퍼 보상 전략 (네트워크 중심)

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    📊 기본 보상:
    • delivered × 10.0
    • drop/abort 패널티 없음

    🔄 릴레이 보상 (네트워크 평균 버퍼 기준):

    ✅ 네트워크 여유 (buf_mean ≤ 0.4):
       → relayed × 4.0
       → 적극적인 협력 장려

    ⏸️ 중간 구간 (0.4 < buf_mean < 0.7):
       → 0.0
       → 중립적 행동

    ❌ 네트워크 혼잡 (buf_mean ≥ 0.7):
       → -relayed × 2.0
       → 협력 자제

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    💡 핵심 아이디어:

    • 본인 버퍼 상태는 무시
    • 네트워크 전체 상황에 집중
    • 작은 버퍼 → 네트워크가 여유로울 때만 협력
    • 선택적 조력자 역할

    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    ax.text(0.05, 0.95, strategy_text, transform=ax.transAxes,
            fontsize=11, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    # 저장
    output_dir = Path('reports_plots/buf20_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'reward_strategy_visualization.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 보상 전략 시각화 저장: {output_path}")
    plt.close()

def plot_action_distribution(df):
    """액션 분포 분석"""
    if df is None or len(df) == 0:
        print("액션 데이터가 없어 분포 분석을 건너뜁니다.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. buf_mean별 action 분포
    ax = axes[0, 0]
    scatter = ax.scatter(df['bufocc_mean'], df['action_delta'],
                        c=df['sim_time'], cmap='viridis', alpha=0.5, s=10)
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=0.4, color='green', linestyle='--', alpha=0.5, label='buf_mean=0.4')
    ax.axvline(x=0.7, color='orange', linestyle='--', alpha=0.5, label='buf_mean=0.7')
    ax.set_xlabel('네트워크 평균 버퍼 점유율', fontsize=12)
    ax.set_ylabel('Action Delta', fontsize=12)
    ax.set_title('네트워크 평균 버퍼 vs Action', fontsize=14, fontweight='bold')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Simulation Time')

    # 2. Action 히스토그램
    ax = axes[0, 1]
    ax.hist(df['action_delta'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.set_xlabel('Action Delta', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Action Delta 분포', fontsize=14, fontweight='bold')
    ax.legend()

    # 3. buf_mean 구간별 평균 action
    ax = axes[1, 0]
    bins = [0, 0.4, 0.7, 1.0]
    labels = ['여유\n(≤0.4)', '중간\n(0.4-0.7)', '혼잡\n(≥0.7)']
    df['buf_category'] = pd.cut(df['bufocc_mean'], bins=bins, labels=labels)

    means = df.groupby('buf_category', observed=True)['action_delta'].mean()
    stds = df.groupby('buf_category', observed=True)['action_delta'].std()

    colors = ['green', 'gray', 'red']
    bars = ax.bar(range(len(means)), means, yerr=stds, capsize=5,
                   color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('평균 Action Delta', fontsize=12)
    ax.set_title('네트워크 상황별 평균 Action', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 시간에 따른 buf_mean 변화
    ax = axes[1, 1]
    time_bins = np.linspace(df['sim_time'].min(), df['sim_time'].max(), 50)
    df['time_bin'] = pd.cut(df['sim_time'], bins=time_bins)

    time_mean = df.groupby('time_bin', observed=True)['bufocc_mean'].mean()
    time_std = df.groupby('time_bin', observed=True)['bufocc_mean'].std()

    x_vals = [interval.mid for interval in time_mean.index]
    ax.plot(x_vals, time_mean.values, linewidth=2, color='#2ecc71')
    ax.fill_between(x_vals,
                     time_mean.values - time_std.values,
                     time_mean.values + time_std.values,
                     alpha=0.3, color='#2ecc71')
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='보상 임계값 (0.4)')
    ax.axhline(y=0.7, color='orange', linestyle='--', alpha=0.5, label='패널티 임계값 (0.7)')
    ax.set_xlabel('Simulation Time (s)', fontsize=12)
    ax.set_ylabel('네트워크 평균 버퍼 점유율', fontsize=12)
    ax.set_title('시간에 따른 네트워크 버퍼 상태 변화', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    output_dir = Path('reports_plots/buf20_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'action_distribution_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 액션 분포 분석 저장: {output_path}")
    plt.close()

def main():
    print("=" * 60)
    print("20M 버퍼 DRL 액션 시각화 및 보상 전략 분석")
    print("=" * 60)

    # 1. 보상 전략 시각화 (항상 생성)
    print("\n[1] 보상 전략 시각화 생성 중...")
    plot_reward_strategy()

    # 2. 실제 액션 로그 분석 (있을 경우)
    print("\n[2] 액션 로그 분석 중...")
    df = load_action_logs()
    if df is not None:
        plot_action_distribution(df)
    else:
        print("→ 액션 로그가 없어 이론적 분석만 수행했습니다.")
        print("→ DRL 학습 중 PPO_DEBUG_ACTIONS=true로 실행하면 액션이 기록됩니다.")

    print("\n" + "=" * 60)
    print("[OK] 분석 완료!")
    print("=" * 60)
    print("\n저장 위치: reports_plots/buf20_analysis/")

if __name__ == "__main__":
    main()
