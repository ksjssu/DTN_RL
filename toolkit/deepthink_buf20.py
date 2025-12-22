#!/usr/bin/env python3
"""
20M DRL Deep Think Analysis
현재 학습 중인 데이터를 깊이 분석하여 성능 개선이 안 되는 근본 원인 파악
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_rlstate_data(file_path):
    """RLStateReport 파일 로드"""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 14:
                continue
            try:
                record = {
                    'time': int(parts[0]),
                    'type': parts[1],
                    'host': parts[2],
                    'contacts_norm': float(parts[5]),
                    'pred': float(parts[6]) if parts[6] != 'NaN' else np.nan,
                    'bufocc_mean': float(parts[7]),
                    'capacity_norm': float(parts[8]),
                    'self_buf_util': float(parts[9]),
                    'relayed': int(parts[10]),
                    'drops': int(parts[11]),
                    'aborted': int(parts[12]),
                    'reward': float(parts[13])
                }
                data.append(record)
            except:
                continue
    return pd.DataFrame(data)

def analyze_episode_rewards():
    """에피소드별 성능 추이 분석"""
    rewards_file = "reports_drl_train/buf20/drl_episode_rewards.txt"

    episodes = []
    delivery_rates = []

    with open(rewards_file, 'r') as f:
        for line in f:
            if line.startswith('episode'):
                parts = line.split()
                ep = int(parts[1])
                dr = float(parts[5])
                episodes.append(ep)
                delivery_rates.append(dr)

    df = pd.DataFrame({'episode': episodes, 'delivery_rate': delivery_rates})

    print("="*80)
    print("DEEP THINK ANALYSIS: 20M DRL 성능 개선 실패 원인 분석")
    print("="*80)

    # 구간별 성능
    early = df[df['episode'] <= 10]
    mid = df[(df['episode'] > 10) & (df['episode'] <= 25)]
    late = df[df['episode'] > 25]

    print("\n[에피소드별 성능 추이]")
    print(f"초기 (1-10):   평균 {early['delivery_rate'].mean():.4f} (std: {early['delivery_rate'].std():.4f})")
    print(f"중기 (11-25):  평균 {mid['delivery_rate'].mean():.4f} (std: {mid['delivery_rate'].std():.4f})")
    print(f"후기 (26-37):  평균 {late['delivery_rate'].mean():.4f} (std: {late['delivery_rate'].std():.4f})")

    # 트렌드 분석
    trend = late['delivery_rate'].mean() - early['delivery_rate'].mean()
    print(f"\n[학습 트렌드]")
    print(f"초기 → 후기 변화: {trend:+.4f} ({trend/early['delivery_rate'].mean()*100:+.2f}%)")

    if trend < 0:
        print(">>> 심각한 문제: 학습할수록 성능 하락! (Catastrophic forgetting or wrong reward)")

    return df

def deep_analysis_relative_spare(df):
    """상대적 여유도 전략의 실제 작동 분석"""

    print("\n" + "="*80)
    print("상대적 여유도(Relative Spare Capacity) 전략 실제 작동 분석")
    print("="*80)

    # relative_spare 계산
    df['spare_self'] = 1.0 - df['self_buf_util']
    df['spare_network'] = 1.0 - df['bufocc_mean']
    df['relative_spare'] = df['spare_self'] - df['spare_network']

    # 구간 분류
    reward_zone = df[df['relative_spare'] >= 0.1]
    penalty_zone = df[df['relative_spare'] <= -0.1]
    neutral_zone = df[(df['relative_spare'] > -0.1) & (df['relative_spare'] < 0.1)]

    print(f"\n[구간 분포]")
    print(f"보상 구간 (relative_spare ≥ 0.1):  {len(reward_zone):8d} ({len(reward_zone)/len(df)*100:5.1f}%)")
    print(f"패널티 구간 (≤ -0.1):              {len(penalty_zone):8d} ({len(penalty_zone)/len(df)*100:5.1f}%)")
    print(f"중립 구간 (-0.1 ~ 0.1):             {len(neutral_zone):8d} ({len(neutral_zone)/len(df)*100:5.1f}%)")

    # 릴레이 분석
    relayed_in_reward = reward_zone[reward_zone['relayed'] > 0]
    relayed_in_penalty = penalty_zone[penalty_zone['relayed'] > 0]

    print(f"\n[릴레이 발생 분석]")
    print(f"보상 구간에서 릴레이:   {len(relayed_in_reward):6d}개 ({len(relayed_in_reward)/len(reward_zone)*100 if len(reward_zone)>0 else 0:.1f}%)")
    print(f"패널티 구간에서 릴레이: {len(relayed_in_penalty):6d}개 ({len(relayed_in_penalty)/len(penalty_zone)*100 if len(penalty_zone)>0 else 0:.1f}%)")

    if len(relayed_in_reward) > 0:
        avg_reward = relayed_in_reward['reward'].mean()
        avg_relay = relayed_in_reward['relayed'].mean()
        print(f"\n보상 구간 릴레이 시:")
        print(f"  평균 릴레이 수: {avg_relay:.2f}")
        print(f"  평균 보상: {avg_reward:.2f}")

    if len(relayed_in_penalty) > 0:
        avg_reward = relayed_in_penalty['reward'].mean()
        avg_relay = relayed_in_penalty['relayed'].mean()
        print(f"\n패널티 구간 릴레이 시:")
        print(f"  평균 릴레이 수: {avg_relay:.2f}")
        print(f"  평균 보상(패널티): {avg_reward:.2f}")

    # 문제 진단
    print(f"\n[문제 진단]")

    # 문제 1: 보상 기회가 충분한가?
    if len(reward_zone) / len(df) < 0.1:
        print(">>> 문제 1: 보상 구간이 너무 적음 (<10%) - 임계값 완화 필요")

    # 문제 2: 릴레이가 발생하는가?
    total_relayed = len(df[df['relayed'] > 0])
    if total_relayed / len(df) < 0.01:
        print(f">>> 문제 2: 릴레이 자체가 거의 없음 ({total_relayed/len(df)*100:.2f}%) - 네트워크 문제 또는 정책 문제")

    # 문제 3: 보상이 delivered보다 작은가?
    if len(relayed_in_reward) > 0:
        max_relay_reward = relayed_in_reward['reward'].max()
        typical_delivery = 10.0
        if max_relay_reward < typical_delivery:
            print(f">>> 문제 3: 릴레이 보상({max_relay_reward:.1f})이 전달 보상(10.0)보다 작음 - 보상 더 강화 필요")

    # 문제 4: relative_spare 값의 실제 범위
    print(f"\n[relative_spare 통계]")
    print(f"평균: {df['relative_spare'].mean():.4f}")
    print(f"중앙값: {df['relative_spare'].median():.4f}")
    print(f"최소: {df['relative_spare'].min():.4f}")
    print(f"최대: {df['relative_spare'].max():.4f}")
    print(f"표준편차: {df['relative_spare'].std():.4f}")

    if abs(df['relative_spare'].mean()) > 0.2:
        if df['relative_spare'].mean() < 0:
            print(">>> 문제 4: relative_spare 평균이 음수 - 대부분 노드가 네트워크보다 차있음")
        else:
            print(">>> 문제 4: relative_spare 평균이 양수 - 대부분 노드가 네트워크보다 비어있음")

    return df

def visualize_problems(df, episode_df):
    """문제점 시각화"""

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # 1. 에피소드별 성능 추이
    ax = axes[0, 0]
    ax.plot(episode_df['episode'], episode_df['delivery_rate'],
            marker='o', linewidth=2, markersize=6, color='#e74c3c')

    # 이동평균
    ma = episode_df['delivery_rate'].rolling(window=5, min_periods=1).mean()
    ax.plot(episode_df['episode'], ma, linewidth=3, color='black',
            label='5-ep Moving Avg', alpha=0.7)

    ax.set_xlabel('Episode', fontweight='bold')
    ax.set_ylabel('Delivery Rate', fontweight='bold')
    ax.set_title('학습 진행: 성능 하락 중!', fontweight='bold', color='red')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. relative_spare 분포
    ax = axes[0, 1]
    ax.hist(df['relative_spare'], bins=50, alpha=0.7, color='#3498db', edgecolor='black')
    ax.axvline(x=0.1, color='green', linestyle='--', linewidth=2, label='보상 임계값 (0.1)')
    ax.axvline(x=-0.1, color='red', linestyle='--', linewidth=2, label='패널티 임계값 (-0.1)')
    ax.axvline(x=df['relative_spare'].mean(), color='orange', linestyle='-',
               linewidth=3, label=f'평균 ({df["relative_spare"].mean():.3f})')
    ax.set_xlabel('Relative Spare (self - network)', fontweight='bold')
    ax.set_ylabel('빈도', fontweight='bold')
    ax.set_title('상대적 여유도 분포', fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. self_buf_util vs bufocc_mean
    ax = axes[0, 2]
    sample = df.sample(n=min(5000, len(df)))
    scatter = ax.scatter(sample['bufocc_mean'], sample['self_buf_util'],
                        c=sample['relative_spare'], cmap='RdYlGn', alpha=0.5, s=10)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='self = network')
    ax.set_xlabel('Network Avg Buf (bufocc_mean)', fontweight='bold')
    ax.set_ylabel('Self Buffer Util', fontweight='bold')
    ax.set_title('버퍼 상태 상관관계', fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Relative Spare')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. 시간에 따른 relative_spare 변화
    ax = axes[1, 0]
    time_groups = df.groupby('time')['relative_spare'].agg(['mean', 'std', 'count'])
    time_groups = time_groups[time_groups['count'] >= 5]

    ax.plot(time_groups.index / 3600, time_groups['mean'], linewidth=2, color='#9b59b6')
    ax.fill_between(time_groups.index / 3600,
                     time_groups['mean'] - time_groups['std'],
                     time_groups['mean'] + time_groups['std'],
                     alpha=0.3, color='#9b59b6')
    ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.1, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('시간 (시)', fontweight='bold')
    ax.set_ylabel('Relative Spare', fontweight='bold')
    ax.set_title('시간에 따른 상대적 여유도 변화', fontweight='bold')
    ax.grid(True, alpha=0.3)

    # 5. 보상 분포
    ax = axes[1, 1]
    reward_data = df[df['reward'] != 0]['reward']
    if len(reward_data) > 0:
        ax.hist(reward_data, bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=2)
        ax.axvline(x=10, color='blue', linestyle='--', linewidth=2,
                   label='Delivery 보상 (10.0)', alpha=0.7)
        ax.set_xlabel('Reward', fontweight='bold')
        ax.set_ylabel('빈도', fontweight='bold')
        ax.set_title('실제 보상 분포', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    # 6. 릴레이 발생 패턴
    ax = axes[1, 2]
    relayed_df = df[df['relayed'] > 0]
    if len(relayed_df) > 0:
        ax.scatter(relayed_df['relative_spare'], relayed_df['reward'],
                   alpha=0.5, s=20, c=relayed_df['relayed'], cmap='viridis')
        ax.axvline(x=0.1, color='green', linestyle='--', linewidth=2)
        ax.axvline(x=-0.1, color='red', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.set_xlabel('Relative Spare', fontweight='bold')
        ax.set_ylabel('Reward', fontweight='bold')
        ax.set_title('릴레이 시 Relative Spare vs Reward', fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_dir = Path('reports_plots/buf20_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'deepthink_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Deep Think 분석 그래프 저장: {output_path}")
    plt.close()

def main():
    # 에피소드 성능 분석
    episode_df = analyze_episode_rewards()

    # 최신 에피소드 State 분석
    latest_state_file = "reports_drl_train/buf20/drl_train_buf20_100k_rng1_run37_RLStateReport.txt"
    print(f"\n최신 에피소드 State 로드: {latest_state_file}")
    df = load_rlstate_data(latest_state_file)
    print(f"총 {len(df)}개 레코드 분석")

    # Deep Think 분석
    df = deep_analysis_relative_spare(df)

    # 시각화
    visualize_problems(df, episode_df)

    print("\n" + "="*80)
    print("DEEP THINK 결론 및 해결 방안")
    print("="*80)
    print("\n학습이 제대로 안 되는 근본 원인을 찾아 개선 방안을 제시하세요.")
    print("\n")

if __name__ == "__main__":
    main()
