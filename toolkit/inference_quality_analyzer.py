#!/usr/bin/env python3
"""
Inference Quality Analyzer for DTN DRL
리워드 성능과 실제 추론(inference) 품질을 분석하는 도구
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_episode_data(buffer_size):
    """에피소드 데이터 로드"""
    file_path = f"reports_drl_train/buf{buffer_size:02d}/drl_episode_rewards.txt"
    if not os.path.exists(file_path):
        return None

    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('episode'):
                parts = line.split()
                episode = int(parts[1])
                reward = int(parts[3])
                delivery_rate = float(parts[5])
                delivered = int(parts[7])
                created = int(parts[9])

                data.append({
                    'episode': episode,
                    'reward': reward,
                    'delivery_rate': delivery_rate,
                    'delivered': delivered,
                    'created': created
                })

    return pd.DataFrame(data)

def analyze_inference_quality(df, buffer_size, window_size=10):
    """추론 품질 분석"""
    if df is None or len(df) < window_size:
        return None

    print(f"\n=== INFERENCE QUALITY ANALYSIS: {buffer_size}M BUFFER ===")

    # 1. 기본 통계
    recent_episodes = df.tail(window_size)
    all_episodes = df

    print(f"\nBASIC STATISTICS (Last {window_size} episodes):")
    print(f"   Mean delivery rate: {recent_episodes['delivery_rate'].mean():.4f}")
    print(f"   Std deviation: {recent_episodes['delivery_rate'].std():.4f}")
    print(f"   Min: {recent_episodes['delivery_rate'].min():.4f}")
    print(f"   Max: {recent_episodes['delivery_rate'].max():.4f}")
    print(f"   Range: {recent_episodes['delivery_rate'].max() - recent_episodes['delivery_rate'].min():.4f}")

    # 2. 수렴 품질 분석
    convergence_quality = analyze_convergence_quality(recent_episodes)
    print(f"\nCONVERGENCE QUALITY:")
    print(f"   Type: {convergence_quality['type']}")
    print(f"   Stability Score: {convergence_quality['stability_score']:.3f}")
    print(f"   Exploration Score: {convergence_quality['exploration_score']:.3f}")

    # 3. 리워드 vs 성능 일치도
    reward_performance_correlation = analyze_reward_performance_correlation(df)
    print(f"\nREWARD-PERFORMANCE CORRELATION:")
    print(f"   Correlation: {reward_performance_correlation['correlation']:.3f}")
    print(f"   Consistency: {reward_performance_correlation['consistency']}")

    # 4. Inference 품질 점수
    inference_score = calculate_inference_quality_score(df, window_size)
    print(f"\nOVERALL INFERENCE QUALITY SCORE: {inference_score:.2f}/10.0")

    # 5. 문제 진단
    diagnose_issues(df, buffer_size, convergence_quality, reward_performance_correlation)

    return {
        'buffer_size': buffer_size,
        'basic_stats': {
            'mean': recent_episodes['delivery_rate'].mean(),
            'std': recent_episodes['delivery_rate'].std(),
            'range': recent_episodes['delivery_rate'].max() - recent_episodes['delivery_rate'].min()
        },
        'convergence_quality': convergence_quality,
        'reward_correlation': reward_performance_correlation,
        'inference_score': inference_score
    }

def analyze_convergence_quality(df):
    """수렴 품질 분석"""
    delivery_rates = df['delivery_rate'].values

    # 안정성 측정 (낮은 분산 = 안정적)
    stability_score = max(0, 1 - (np.std(delivery_rates) / 0.05))  # 0.05를 기준으로 정규화

    # 탐험성 측정 (적당한 분산 = 좋은 탐험)
    std_dev = np.std(delivery_rates)
    if std_dev < 0.001:  # 너무 낮음 = 과수렴
        exploration_score = 0.2
        convergence_type = "OVER_CONVERGED (과수렴)"
    elif std_dev < 0.005:  # 적당함 = 좋은 수렴
        exploration_score = 1.0
        convergence_type = "GOOD_CONVERGENCE (좋은 수렴)"
    elif std_dev < 0.02:  # 약간 높음 = 학습 중
        exploration_score = 0.7
        convergence_type = "LEARNING (학습 중)"
    else:  # 너무 높음 = 불안정
        exploration_score = 0.3
        convergence_type = "UNSTABLE (불안정)"

    return {
        'type': convergence_type,
        'stability_score': stability_score,
        'exploration_score': exploration_score,
        'std_dev': std_dev
    }

def analyze_reward_performance_correlation(df):
    """리워드와 성능의 상관관계 분석"""
    correlation = np.corrcoef(df['reward'], df['delivery_rate'])[0, 1]

    if correlation > 0.8:
        consistency = "EXCELLENT (리워드와 성능 일치)"
    elif correlation > 0.6:
        consistency = "GOOD (리워드와 성능 대체로 일치)"
    elif correlation > 0.4:
        consistency = "MODERATE (부분적 일치)"
    else:
        consistency = "POOR (리워드와 성능 불일치)"

    return {
        'correlation': correlation,
        'consistency': consistency
    }

def calculate_inference_quality_score(df, window_size):
    """전체 추론 품질 점수 계산 (0-10점)"""
    recent = df.tail(window_size)

    # 1. 성능 점수 (0-4점): 절대 성능
    mean_performance = recent['delivery_rate'].mean()
    performance_score = min(4.0, (mean_performance / 0.7) * 4.0)  # 0.7을 만점으로 가정

    # 2. 안정성 점수 (0-3점): 낮은 분산
    std_dev = recent['delivery_rate'].std()
    if std_dev < 0.001:
        stability_score = 1.0  # 과수렴은 낮은 점수
    elif std_dev < 0.005:
        stability_score = 3.0  # 적당한 안정성
    elif std_dev < 0.02:
        stability_score = 2.0  # 약간 불안정
    else:
        stability_score = 0.5  # 매우 불안정

    # 3. 개선 추세 점수 (0-3점): 학습 진전
    first_half = recent.head(window_size//2)['delivery_rate'].mean()
    second_half = recent.tail(window_size//2)['delivery_rate'].mean()
    improvement = second_half - first_half

    if improvement > 0.01:
        trend_score = 3.0  # 개선 중
    elif improvement > 0:
        trend_score = 2.0  # 약간 개선
    elif improvement > -0.005:
        trend_score = 1.5  # 유지
    else:
        trend_score = 0.5  # 악화

    return performance_score + stability_score + trend_score

def diagnose_issues(df, buffer_size, convergence_quality, reward_correlation):
    """문제점 진단 및 해결책 제시"""
    print(f"\nISSUE DIAGNOSIS & RECOMMENDATIONS:")

    issues = []

    # 과수렴 문제
    if convergence_quality['std_dev'] < 0.001:
        issues.append("OVER_CONVERGENCE: 너무 일찍 수렴, 탐험 부족")
        print("   WARNING: 과수렴 감지: 에이전트가 단일 전략에 고착됨")
        print("       -> 엔트로피 계수 증가 필요")
        print("       -> RND 강도 증가 필요")
        print("       -> 학습률 조정 고려")

    # 성능 정체
    recent_mean = df.tail(10)['delivery_rate'].mean()
    if recent_mean < 0.4 and buffer_size >= 10:
        issues.append("LOW_PERFORMANCE: 절대 성능 부족")
        print("   PERFORMANCE: 성능 정체: 휴리스틱 대비 성능 부족")
        print("       -> 상태 정보 확장 필요")
        print("       -> 보상 구조 재검토 필요")
        print("       -> 환경 복잡도 증가 고려")

    # 리워드-성능 불일치
    if reward_correlation['correlation'] < 0.5:
        issues.append("REWARD_MISMATCH: 리워드와 실제 성능 불일치")
        print("   MISMATCH: 리워드 불일치: 최적화 방향과 목표 불일치")
        print("       -> 리워드 함수 재설계 필요")
        print("       -> 즉시 보상 vs 지연 보상 균형 조정")

    if not issues:
        print("   OK: 특별한 문제점이 발견되지 않았습니다.")

def create_inference_quality_plot(buffer_sizes=[10, 40]):
    """추론 품질 시각화"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    results = {}
    for i, buf_size in enumerate(buffer_sizes):
        df = load_episode_data(buf_size)
        if df is None:
            continue

        results[buf_size] = analyze_inference_quality(df, buf_size)

        # 성능 추이
        axes[0, i].plot(df['episode'], df['delivery_rate'], 'b-', alpha=0.7)
        axes[0, i].set_title(f'{buf_size}M Buffer: Performance Trend')
        axes[0, i].set_ylabel('Delivery Rate')
        axes[0, i].grid(True, alpha=0.3)

        # 최근 분포
        recent = df.tail(20)
        axes[1, i].hist(recent['delivery_rate'], bins=10, alpha=0.7, edgecolor='black')
        axes[1, i].set_title(f'{buf_size}M Buffer: Recent Distribution')
        axes[1, i].set_xlabel('Delivery Rate')
        axes[1, i].set_ylabel('Frequency')
        axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('reports/inference_quality_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    return results

def main():
    """메인 분석 함수"""
    print("DTN DRL Inference Quality Analyzer")
    print("=" * 50)

    # 보고서 디렉토리 생성
    os.makedirs('reports', exist_ok=True)

    # 분석 실행
    results = create_inference_quality_plot()

    # 비교 분석
    if len(results) >= 2:
        print(f"\nCOMPARATIVE ANALYSIS:")
        buf_sizes = list(results.keys())
        for i in range(len(buf_sizes)):
            for j in range(i+1, len(buf_sizes)):
                buf1, buf2 = buf_sizes[i], buf_sizes[j]
                print(f"\n{buf1}M vs {buf2}M:")
                print(f"   Performance: {results[buf1]['basic_stats']['mean']:.4f} vs {results[buf2]['basic_stats']['mean']:.4f}")
                print(f"   Stability: {results[buf1]['basic_stats']['std']:.4f} vs {results[buf2]['basic_stats']['std']:.4f}")
                print(f"   Quality Score: {results[buf1]['inference_score']:.2f} vs {results[buf2]['inference_score']:.2f}")

    print(f"\nAnalysis complete! Check reports/inference_quality_analysis.png")

if __name__ == "__main__":
    main()