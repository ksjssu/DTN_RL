#!/usr/bin/env python3
"""
20M 버퍼 DRL State/Action/Reward 히스토리 분석
보상체계 수정 전 데이터 분석
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

# 한글 폰트 설정
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
                    'type': parts[1],  # R or S
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

def analyze_buffer_occupancy_vs_actions(df):
    """buf_mean 구간별 action/reward 분석"""

    print("\n" + "="*60)
    print("네트워크 평균 버퍼 점유율(buf_mean) 분석")
    print("="*60)

    # buf_mean 구간 분류
    bins = [0, 0.33, 0.4, 0.66, 0.7, 1.0]
    labels = ['매우낮음(≤0.33)', '낮음(0.33-0.4)', '중간(0.4-0.66)', '높음(0.66-0.7)', '매우높음(≥0.7)']
    df['buf_category'] = pd.cut(df['bufocc_mean'], bins=bins, labels=labels)

    # 구간별 통계
    print("\n[구간별 분포]")
    for cat in labels:
        cat_data = df[df['buf_category'] == cat]
        pct = len(cat_data) / len(df) * 100
        print(f"{cat}: {len(cat_data):6d}개 ({pct:5.1f}%)")

    # 보상체계 임계값 분석
    print("\n[보상체계 임계값 분석] (수정 전: 0.4/0.7)")
    below_04 = df[df['bufocc_mean'] <= 0.4]
    above_07 = df[df['bufocc_mean'] >= 0.7]
    middle = df[(df['bufocc_mean'] > 0.4) & (df['bufocc_mean'] < 0.7)]

    print(f"buf_mean ≤ 0.4 (보상 구간): {len(below_04):6d}개 ({len(below_04)/len(df)*100:5.1f}%)")
    print(f"buf_mean ≥ 0.7 (패널티 구간): {len(above_07):6d}개 ({len(above_07)/len(df)*100:5.1f}%)")
    print(f"중간 (0.4-0.7): {len(middle):6d}개 ({len(middle)/len(df)*100:5.1f}%)")

    # 휴리스틱 임계값 비교
    print("\n[휴리스틱 임계값 분석] (0.33/0.66)")
    below_033 = df[df['bufocc_mean'] <= 0.33]
    above_066 = df[df['bufocc_mean'] >= 0.66]
    middle_h = df[(df['bufocc_mean'] > 0.33) & (df['bufocc_mean'] < 0.66)]

    print(f"buf_mean ≤ 0.33 (휴리스틱 LOW): {len(below_033):6d}개 ({len(below_033)/len(df)*100:5.1f}%)")
    print(f"buf_mean ≥ 0.66 (휴리스틱 HIGH): {len(above_066):6d}개 ({len(above_066)/len(df)*100:5.1f}%)")
    print(f"중간 (0.33-0.66): {len(middle_h):6d}개 ({len(middle_h)/len(df)*100:5.1f}%)")

    return df

def analyze_reward_components(df):
    """보상 구성 요소 분석"""

    print("\n" + "="*60)
    print("보상 구성 요소 분석")
    print("="*60)

    # Relayed가 있는 경우만
    relayed_df = df[df['relayed'] > 0]

    if len(relayed_df) > 0:
        print(f"\n릴레이 발생: {len(relayed_df)}개 (전체의 {len(relayed_df)/len(df)*100:.1f}%)")

        # buf_mean별 릴레이 분석
        for threshold, label in [(0.4, '≤0.4 (보상)'), (0.7, '≥0.7 (패널티)')]:
            if threshold == 0.4:
                subset = relayed_df[relayed_df['bufocc_mean'] <= threshold]
            else:
                subset = relayed_df[relayed_df['bufocc_mean'] >= threshold]

            if len(subset) > 0:
                avg_relay = subset['relayed'].mean()
                avg_reward = subset['reward'].mean()
                print(f"\nbuf_mean {label}:")
                print(f"  평균 릴레이 수: {avg_relay:.2f}")
                print(f"  평균 보상: {avg_reward:.2f}")

    # Drop 분석
    drop_df = df[df['drops'] > 0]
    if len(drop_df) > 0:
        print(f"\n드랍 발생: {len(drop_df)}개 (전체의 {len(drop_df)/len(df)*100:.1f}%)")
        print(f"  평균 드랍 수: {drop_df['drops'].mean():.2f}")

def plot_history_analysis(df):
    """히스토리 시각화"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. 시간에 따른 buf_mean 변화
    ax = axes[0, 0]

    # 시간별 평균 buf_mean
    time_groups = df.groupby('time')['bufocc_mean'].agg(['mean', 'std', 'count'])
    time_groups = time_groups[time_groups['count'] >= 5]  # 최소 5개 이상 데이터

    ax.plot(time_groups.index / 3600, time_groups['mean'],
            linewidth=2, color='#3498db', label='평균 buf_mean')
    ax.fill_between(time_groups.index / 3600,
                     time_groups['mean'] - time_groups['std'],
                     time_groups['mean'] + time_groups['std'],
                     alpha=0.3, color='#3498db')

    # 임계값 표시
    ax.axhline(y=0.4, color='green', linestyle='--', linewidth=2,
               alpha=0.7, label='DRL 보상 임계값 (0.4)')
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2,
               alpha=0.7, label='DRL 패널티 임계값 (0.7)')
    ax.axhline(y=0.33, color='lime', linestyle=':', linewidth=2,
               alpha=0.5, label='휴리스틱 LOW (0.33)')
    ax.axhline(y=0.66, color='orange', linestyle=':', linewidth=2,
               alpha=0.5, label='휴리스틱 HIGH (0.66)')

    ax.set_xlabel('시간 (시)', fontsize=12, fontweight='bold')
    ax.set_ylabel('네트워크 평균 버퍼 점유율', fontsize=12, fontweight='bold')
    ax.set_title('시간에 따른 네트워크 버퍼 상태 변화', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    # 2. buf_mean 분포 히스토그램
    ax = axes[0, 1]

    ax.hist(df['bufocc_mean'], bins=50, alpha=0.7, color='#2ecc71', edgecolor='black')
    ax.axvline(x=0.4, color='green', linestyle='--', linewidth=2, label='DRL 0.4')
    ax.axvline(x=0.7, color='red', linestyle='--', linewidth=2, label='DRL 0.7')
    ax.axvline(x=0.33, color='lime', linestyle=':', linewidth=2, label='휴리스틱 0.33')
    ax.axvline(x=0.66, color='orange', linestyle=':', linewidth=2, label='휴리스틱 0.66')

    ax.set_xlabel('네트워크 평균 버퍼 점유율', fontsize=12, fontweight='bold')
    ax.set_ylabel('빈도', fontsize=12, fontweight='bold')
    ax.set_title('buf_mean 분포 (임계값 비교)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # 3. buf_mean vs self_buf_util
    ax = axes[1, 0]

    sample = df.sample(n=min(5000, len(df)))
    scatter = ax.scatter(sample['bufocc_mean'], sample['self_buf_util'],
                        c=sample['reward'], cmap='RdYlGn', alpha=0.5, s=10)

    ax.axvline(x=0.4, color='green', linestyle='--', alpha=0.5)
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=0.5, color='blue', linestyle='--', alpha=0.3)

    ax.set_xlabel('네트워크 평균 버퍼 (buf_mean)', fontsize=12, fontweight='bold')
    ax.set_ylabel('자신의 버퍼 점유율', fontsize=12, fontweight='bold')
    ax.set_title('네트워크 vs 개인 버퍼 상태', fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, label='Reward')
    ax.grid(True, alpha=0.3)

    # 4. 구간별 보상 분포
    ax = axes[1, 1]

    bins = [0, 0.33, 0.4, 0.66, 0.7, 1.0]
    labels = ['≤0.33\n(H-LOW)', '0.33-0.4\n(중립)', '0.4-0.66\n(중립)',
              '0.66-0.7\n(중립)', '≥0.7\n(DRL 패널티)']
    df['buf_cat'] = pd.cut(df['bufocc_mean'], bins=bins, labels=labels)

    box_data = [df[df['buf_cat'] == cat]['reward'].dropna()
                for cat in labels if len(df[df['buf_cat'] == cat]) > 0]
    box_labels = [cat for cat in labels if len(df[df['buf_cat'] == cat]) > 0]

    bp = ax.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     showmeans=True)

    colors = ['lime', 'lightgreen', 'gray', 'orange', 'red']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.set_ylabel('Reward', fontsize=12, fontweight='bold')
    ax.set_title('buf_mean 구간별 보상 분포', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # 저장
    output_dir = Path('reports_plots/buf20_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / 'history_analysis_old_reward.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] 히스토리 분석 그래프 저장: {output_path}")
    plt.close()

def main():
    print("\n20M DRL State/Action/Reward 히스토리 분석")
    print("보상체계 (수정 전): buf_mean ≤ 0.4 (보상 4.0), buf_mean ≥ 0.7 (패널티 -2.0)")

    # RLStateReport 파일 로드 (첫 번째 에피소드만)
    state_file = "reports_drl_train/buf20/drl_train_buf20_100k_rng1_run1_RLStateReport.txt"

    print(f"\n데이터 로드 중: {state_file}")
    df = load_rlstate_data(state_file)
    print(f"총 {len(df)}개 레코드 로드")

    # 분석
    df = analyze_buffer_occupancy_vs_actions(df)
    analyze_reward_components(df)
    plot_history_analysis(df)

    print("\n" + "="*60)
    print("[OK] 히스토리 분석 완료!")
    print("="*60)
    print("\n주요 발견:")
    print("1. DRL 임계값(0.4/0.7)과 휴리스틱 임계값(0.33/0.66) 비교")
    print("2. buf_mean이 각 구간에 머무는 시간 비율")
    print("3. 구간별 실제 보상/패널티 분포")
    print("\n")

if __name__ == "__main__":
    main()
