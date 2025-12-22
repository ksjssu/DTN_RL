#!/usr/bin/env python3
"""
20M 버퍼 DRL 학습 분석 (보상체계 수정 전)
임계값: buf_mean ≤ 0.4 (보상 4.0), buf_mean ≥ 0.7 (패널티 -2.0)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def analyze_training_progress():
    """학습 진행 분석"""

    # 에피소드 보상 파일 읽기
    rewards_file = "reports_drl_train/buf20/drl_episode_rewards.txt"

    episodes = []
    rewards = []
    delivery_rates = []
    delivered = []

    with open(rewards_file, 'r') as f:
        for line in f:
            if line.startswith('episode'):
                parts = line.split()
                ep = int(parts[1])
                reward = int(parts[3])
                dr = float(parts[5])
                deliv = int(parts[7])

                episodes.append(ep)
                rewards.append(reward)
                delivery_rates.append(dr)
                delivered.append(deliv)

    df = pd.DataFrame({
        'episode': episodes,
        'reward': rewards,
        'delivery_rate': delivery_rates,
        'delivered': delivered
    })

    # 학습 구간 구분 (첫 80은 training, 나머지는 eval로 추정)
    train_df = df[df['episode'] <= 80].copy()
    eval_df = df[df['episode'] > 80].copy()

    print("=" * 60)
    print("20M DRL 학습 분석 (보상체계 수정 전)")
    print("=" * 60)
    print(f"\n총 에피소드: {len(df)}")
    print(f"학습 에피소드: {len(train_df)} (1-80)")
    print(f"평가 에피소드: {len(eval_df)} (81-100)")

    print("\n" + "-" * 60)
    print("학습 구간 성능 (Episode 1-80):")
    print("-" * 60)
    print(f"평균 전달률: {train_df['delivery_rate'].mean():.4f}")
    print(f"표준편차: {train_df['delivery_rate'].std():.4f}")
    print(f"최고: {train_df['delivery_rate'].max():.4f} (Episode {train_df.loc[train_df['delivery_rate'].idxmax(), 'episode']:.0f})")
    print(f"최저: {train_df['delivery_rate'].min():.4f} (Episode {train_df.loc[train_df['delivery_rate'].idxmin(), 'episode']:.0f})")

    # 초기(1-20), 중기(21-50), 후기(51-80) 구간별 분석
    early = train_df[train_df['episode'] <= 20]
    mid = train_df[(train_df['episode'] > 20) & (train_df['episode'] <= 50)]
    late = train_df[train_df['episode'] > 50]

    print(f"\n초기 (1-20): 평균 {early['delivery_rate'].mean():.4f}")
    print(f"중기 (21-50): 평균 {mid['delivery_rate'].mean():.4f}")
    print(f"후기 (51-80): 평균 {late['delivery_rate'].mean():.4f}")

    if len(eval_df) > 0:
        print("\n" + "-" * 60)
        print("평가 구간 성능 (Episode 81-100):")
        print("-" * 60)
        print(f"평균 전달률: {eval_df['delivery_rate'].mean():.4f}")
        print(f"표준편차: {eval_df['delivery_rate'].std():.4f}")
        print(f"최고: {eval_df['delivery_rate'].max():.4f}")
        print(f"최저: {eval_df['delivery_rate'].min():.4f}")

        # 학습 vs 평가 비교
        improvement = eval_df['delivery_rate'].mean() - train_df['delivery_rate'].mean()
        print(f"\n학습 대비 평가 성능 변화: {improvement:+.4f} ({improvement/train_df['delivery_rate'].mean()*100:+.2f}%)")

    return df, train_df, eval_df

def plot_training_analysis(df, train_df, eval_df):
    """학습 분석 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 전달률 추이
    ax = axes[0, 0]
    ax.plot(df['episode'], df['delivery_rate'], marker='o', markersize=4,
            linewidth=1.5, alpha=0.7, color='#3498db')

    # 학습/평가 구간 구분선
    if len(eval_df) > 0:
        split_ep = eval_df['episode'].min()
        ax.axvline(x=split_ep, color='red', linestyle='--', linewidth=2,
                   alpha=0.5, label=f'Training/Eval 구분 (Ep {split_ep})')

    # 이동평균선
    window = 10
    ma = df['delivery_rate'].rolling(window=window, min_periods=1).mean()
    ax.plot(df['episode'], ma, linewidth=3, color='orange',
            label=f'{window}-Episode Moving Avg')

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Delivery Rate', fontsize=12, fontweight='bold')
    ax.set_title('20M DRL 학습 진행 - 전달 성공률 추이', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 2. 보상 추이
    ax = axes[0, 1]
    ax.plot(df['episode'], df['reward'], marker='s', markersize=4,
            linewidth=1.5, alpha=0.7, color='#2ecc71')

    if len(eval_df) > 0:
        ax.axvline(x=split_ep, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ma_reward = df['reward'].rolling(window=window, min_periods=1).mean()
    ax.plot(df['episode'], ma_reward, linewidth=3, color='darkgreen',
            label=f'{window}-Episode Moving Avg')

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Total Reward', fontsize=12, fontweight='bold')
    ax.set_title('20M DRL 학습 진행 - 총 보상 추이', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # 3. 학습 구간별 전달률 분포
    ax = axes[1, 0]

    early = train_df[train_df['episode'] <= 20]
    mid = train_df[(train_df['episode'] > 20) & (train_df['episode'] <= 50)]
    late = train_df[train_df['episode'] > 50]

    data_to_plot = []
    labels = []
    colors = []

    if len(early) > 0:
        data_to_plot.append(early['delivery_rate'])
        labels.append(f'초기\n(1-20)\nn={len(early)}')
        colors.append('#e74c3c')

    if len(mid) > 0:
        data_to_plot.append(mid['delivery_rate'])
        labels.append(f'중기\n(21-50)\nn={len(mid)}')
        colors.append('#f39c12')

    if len(late) > 0:
        data_to_plot.append(late['delivery_rate'])
        labels.append(f'후기\n(51-80)\nn={len(late)}')
        colors.append('#27ae60')

    if len(eval_df) > 0:
        data_to_plot.append(eval_df['delivery_rate'])
        labels.append(f'평가\n(81-100)\nn={len(eval_df)}')
        colors.append('#3498db')

    bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True,
                     showmeans=True, meanline=True)

    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel('Delivery Rate', fontsize=12, fontweight='bold')
    ax.set_title('구간별 전달 성공률 분포', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # 4. 학습 안정성 분석 (Rolling STD)
    ax = axes[1, 1]

    rolling_std = df['delivery_rate'].rolling(window=10, min_periods=1).std()
    ax.plot(df['episode'], rolling_std, linewidth=2, color='purple')
    ax.fill_between(df['episode'], 0, rolling_std, alpha=0.3, color='purple')

    if len(eval_df) > 0:
        ax.axvline(x=split_ep, color='red', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_xlabel('Episode', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling Std (10-ep window)', fontsize=12, fontweight='bold')
    ax.set_title('학습 안정성 분석 - 전달률 변동성', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # 저장
    output_dir = Path('reports_plots/buf20_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'training_analysis_old_reward.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 학습 분석 그래프 저장: {output_path}")
    plt.close()

def main():
    print("\n")
    df, train_df, eval_df = analyze_training_progress()
    plot_training_analysis(df, train_df, eval_df)

    print("\n" + "=" * 60)
    print("[OK] 분석 완료!")
    print("=" * 60)
    print("\n저장 위치: reports_plots/buf20_analysis/")
    print("\n💡 주요 발견:")

    if len(train_df) > 0 and len(eval_df) > 0:
        train_avg = train_df['delivery_rate'].mean()
        eval_avg = eval_df['delivery_rate'].mean()

        if eval_avg > train_avg:
            print(f"✅ 평가 성능이 학습 대비 {(eval_avg-train_avg)/train_avg*100:.1f}% 향상")
        else:
            print(f"⚠️  평가 성능이 학습 대비 {(train_avg-eval_avg)/train_avg*100:.1f}% 저하")

    print("\n보상체계 (수정 전):")
    print("- 임계값: buf_mean ≤ 0.4 (보상), buf_mean ≥ 0.7 (패널티)")
    print("- 보상: relayed × 4.0")
    print("- 패널티: -relayed × 2.0")
    print("\n")

if __name__ == "__main__":
    main()
