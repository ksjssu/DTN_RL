#!/usr/bin/env python3
"""
Current 10M Buffer Action Visualization
현재 학습 중인 10M 버퍼의 액션과 상태 패턴 분석
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_latest_state_data():
    """Load the latest RLStateReport data"""
    # Find the most recent RLStateReport file
    state_files = [
        "reports_drl_train/buf10/drl_train_buf10_100k_rng1_run1_RLStateReport.txt",
        "reports_drl_train/buf10/drl_train_buf10_100k_rng1_run2_RLStateReport.txt",
        "reports_drl_train/buf10/drl_train_buf10_100k_rng1_run3_RLStateReport.txt"
    ]

    latest_file = None
    latest_size = 0

    for file in state_files:
        if os.path.exists(file):
            size = os.path.getsize(file)
            if size > latest_size:
                latest_size = size
                latest_file = file

    if not latest_file:
        print("No RLStateReport files found!")
        return None

    print(f"Loading latest state data from: {latest_file}")
    print(f"File size: {latest_size / 1024 / 1024:.1f} MB")

    try:
        data = []
        with open(latest_file, 'r') as f:
            lines = f.readlines()

        for line in lines[-50000:]:  # 최근 50k 라인만 분석 (성능 고려)
            line = line.strip()
            if line.startswith('#') or not line:
                continue

            parts = line.split()
            if len(parts) >= 13 and parts[1] == 'R':  # Reward line
                try:
                    time = float(parts[0])
                    host = parts[2]
                    contacts_norm = float(parts[5]) if parts[5] != 'NaN' else 0.0
                    pred = float(parts[6]) if parts[6] != 'NaN' else 0.0
                    bufocc_mean = float(parts[7]) if parts[7] != 'NaN' else 0.0
                    capacity_norm = float(parts[8])
                    self_buf_util = float(parts[9])
                    relayed = float(parts[10])
                    drops = float(parts[11])
                    aborted = float(parts[12])
                    reward = float(parts[13])

                    data.append({
                        'time': time,
                        'host': host,
                        'contacts_norm': contacts_norm,
                        'pred': pred,
                        'bufocc_mean': bufocc_mean,
                        'capacity_norm': capacity_norm,
                        'self_buf_util': self_buf_util,
                        'relayed': relayed,
                        'drops': drops,
                        'aborted': aborted,
                        'reward': reward
                    })
                except (ValueError, IndexError):
                    continue

        return pd.DataFrame(data)

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def load_step_rewards():
    """Load step reward data"""
    step_file = "reports_drl_train/buf10/drl_step_rewards.csv"
    if os.path.exists(step_file):
        try:
            df = pd.read_csv(step_file)
            print(f"Step rewards loaded: {len(df)} records")
            return df
        except Exception as e:
            print(f"Error loading step rewards: {e}")
    return None

def analyze_current_learning(df, step_df=None):
    """Analyze current learning patterns"""
    print("\n=== 현재 10M 버퍼 학습 상태 분석 ===")

    if len(df) == 0:
        print("No data available!")
        return

    print(f"분석 데이터: {len(df):,}개 기록")
    print(f"시간 범위: {df['time'].min():.0f} - {df['time'].max():.0f}초")
    print(f"학습 시간: {(df['time'].max() - df['time'].min())/3600:.1f}시간")

    # 최근 vs 초기 비교
    recent_data = df[df['time'] > df['time'].quantile(0.8)]  # 최근 20%
    early_data = df[df['time'] < df['time'].quantile(0.2)]   # 초기 20%

    print(f"\n=== 학습 진행 상황 ===")
    print(f"초기 버퍼 사용률: {early_data['self_buf_util'].mean():.3f}")
    print(f"최근 버퍼 사용률: {recent_data['self_buf_util'].mean():.3f}")
    print(f"변화: {recent_data['self_buf_util'].mean() - early_data['self_buf_util'].mean():+.3f}")

    print(f"\n초기 평균 보상: {early_data['reward'].mean():.2f}")
    print(f"최근 평균 보상: {recent_data['reward'].mean():.2f}")
    print(f"보상 변화: {recent_data['reward'].mean() - early_data['reward'].mean():+.2f}")

    # 드롭 분석
    print(f"\n=== 드롭 패턴 분석 ===")
    print(f"초기 드롭율: {(early_data['drops'] > 0).mean()*100:.1f}%")
    print(f"최근 드롭율: {(recent_data['drops'] > 0).mean()*100:.1f}%")

    # 보상 분포
    positive_rewards = (recent_data['reward'] > 0).sum()
    zero_rewards = (recent_data['reward'] == 0).sum()
    negative_rewards = (recent_data['reward'] < 0).sum()
    total = len(recent_data)

    print(f"\n=== 최근 보상 분포 ===")
    print(f"양수 보상: {positive_rewards}/{total} ({positive_rewards/total*100:.1f}%)")
    print(f"제로 보상: {zero_rewards}/{total} ({zero_rewards/total*100:.1f}%)")
    print(f"음수 보상: {negative_rewards}/{total} ({negative_rewards/total*100:.1f}%)")

    if step_df is not None and len(step_df) > 0:
        print(f"\n=== 에피소드 진행 ===")
        print(f"총 에피소드: {step_df['episode'].max()}")
        recent_episodes = step_df[step_df['episode'] >= step_df['episode'].max() - 2]
        print(f"최근 에피소드 평균 보상: {recent_episodes['reward_sum'].mean():.2f}")

def create_current_visualizations(df, step_df=None, output_dir="reports_plots/buf10_current"):
    """Create current learning visualizations"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 학습 진행도 - 시간에 따른 주요 지표
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Sample data for performance
    if len(df) > 10000:
        df_sample = df.sample(n=10000).sort_values('time')
    else:
        df_sample = df.sort_values('time')

    # Buffer utilization over time
    axes[0,0].plot(df_sample['time']/3600, df_sample['self_buf_util'], alpha=0.6, linewidth=0.5)
    axes[0,0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='70% 목표')
    axes[0,0].axhline(y=0.85, color='orange', linestyle='--', alpha=0.7, label='85% 위험')
    axes[0,0].set_title('버퍼 사용률 변화')
    axes[0,0].set_ylabel('Buffer Utilization')
    axes[0,0].set_xlabel('Time (hours)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Rewards over time
    axes[0,1].plot(df_sample['time']/3600, df_sample['reward'], alpha=0.6, linewidth=0.5, color='purple')
    axes[0,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
    axes[0,1].set_title('보상 변화')
    axes[0,1].set_ylabel('Reward')
    axes[0,1].set_xlabel('Time (hours)')
    axes[0,1].grid(True, alpha=0.3)

    # Drops over time
    axes[1,0].plot(df_sample['time']/3600, df_sample['drops'], alpha=0.6, linewidth=0.5, color='red')
    axes[1,0].set_title('드롭 발생 패턴')
    axes[1,0].set_ylabel('Drops')
    axes[1,0].set_xlabel('Time (hours)')
    axes[1,0].grid(True, alpha=0.3)

    # Contacts over time
    axes[1,1].plot(df_sample['time']/3600, df_sample['contacts_norm'], alpha=0.6, linewidth=0.5, color='green')
    axes[1,1].set_title('연결성 변화')
    axes[1,1].set_ylabel('Contacts (normalized)')
    axes[1,1].set_xlabel('Time (hours)')
    axes[1,1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_progress.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 2. 현재 상태 분포
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Recent data for distribution analysis
    recent_data = df[df['time'] > df['time'].quantile(0.8)]

    # Buffer utilization distribution
    axes[0,0].hist(recent_data['self_buf_util'], bins=30, alpha=0.7, edgecolor='black')
    axes[0,0].axvline(x=0.7, color='red', linestyle='--', label='70% 목표')
    axes[0,0].set_title('현재 버퍼 사용률 분포')
    axes[0,0].set_xlabel('Buffer Utilization')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)

    # Reward distribution
    axes[0,1].hist(recent_data['reward'], bins=30, alpha=0.7, edgecolor='black', color='purple')
    axes[0,1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
    axes[0,1].set_title('현재 보상 분포')
    axes[0,1].set_xlabel('Reward')
    axes[0,1].grid(True, alpha=0.3)

    # Drops distribution
    axes[0,2].hist(recent_data['drops'], bins=20, alpha=0.7, edgecolor='black', color='red')
    axes[0,2].set_title('현재 드롭 분포')
    axes[0,2].set_xlabel('Drops')
    axes[0,2].grid(True, alpha=0.3)

    # Contacts distribution
    axes[1,0].hist(recent_data['contacts_norm'], bins=30, alpha=0.7, edgecolor='black', color='green')
    axes[1,0].set_title('현재 연결성 분포')
    axes[1,0].set_xlabel('Contacts (normalized)')
    axes[1,0].grid(True, alpha=0.3)

    # Buffer occupancy
    axes[1,1].hist(recent_data['bufocc_mean'], bins=30, alpha=0.7, edgecolor='black', color='orange')
    axes[1,1].set_title('현재 평균 버퍼 점유율')
    axes[1,1].set_xlabel('Buffer Occupancy')
    axes[1,1].grid(True, alpha=0.3)

    # Prophet prediction
    axes[1,2].hist(recent_data['pred'], bins=30, alpha=0.7, edgecolor='black', color='blue')
    axes[1,2].set_title('현재 Prophet 예측')
    axes[1,2].set_xlabel('Prophet Score')
    axes[1,2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/current_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Episode rewards if available
    if step_df is not None and len(step_df) > 0:
        plt.figure(figsize=(12, 6))
        plt.plot(step_df['episode'], step_df['reward_sum'], marker='o', linewidth=2)
        plt.title('에피소드별 총 보상')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.grid(True, alpha=0.3)
        plt.savefig(f"{output_dir}/episode_rewards.png", dpi=150, bbox_inches='tight')
        plt.close()

    print(f"현재 학습 상태 비주얼라이제이션 저장: {output_dir}/")

def main():
    """Main function"""
    print("=== 현재 학습 중인 10M 버퍼 액션 비주얼라이제이션 ===")

    # Load latest data
    df = load_latest_state_data()
    if df is None or len(df) == 0:
        print("데이터를 로드할 수 없습니다.")
        return

    step_df = load_step_rewards()

    # Analyze current learning
    analyze_current_learning(df, step_df)

    # Create visualizations
    create_current_visualizations(df, step_df)

    print(f"\n✅ 현재 학습 상태 분석 완료!")
    print(f"📊 비주얼라이제이션: reports_plots/buf10_current/")

if __name__ == "__main__":
    main()