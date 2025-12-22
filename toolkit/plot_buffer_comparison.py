#!/usr/bin/env python3
"""
버퍼 크기별 시간대에 따른 전달 성공률 비교 그래프
30M, 40M, 50M 버퍼별로 5개 라인 비교: heuristic_d010, d020, d030, prophet, drl_eval
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def load_heuristic_data(buffer_size, delta):
    """휴리스틱 데이터 로드 (reports_summary)"""
    file_path = f"reports_summary/heuristic_d{delta:03d}/buf{buffer_size}.csv"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None

    df = pd.read_csv(file_path)
    # 시간을 시간 단위로 변환 (초 -> 시간)
    df['time_hours'] = df['end'] / 3600
    return df

def load_prophet_data(buffer_size):
    """Prophet 데이터 로드 (reports_summary)"""
    file_path = f"reports_summary/prophet/buf{buffer_size}.csv"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None

    df = pd.read_csv(file_path)
    df['time_hours'] = df['end'] / 3600
    return df

def load_drl_eval_data(buffer_size):
    """DRL 평가 데이터 로드 (reports_drl_eval_summary)"""
    file_path = f"reports_drl_eval_summary/buf{buffer_size}/delivery_success_stats.csv"
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found")
        return None

    df = pd.read_csv(file_path)
    df['time_hours'] = df['end'] / 3600
    return df

def plot_buffer_comparison(buffer_size):
    """특정 버퍼 크기에 대한 비교 그래프"""
    fig, ax = plt.subplots(figsize=(12, 7))

    # 데이터 로드
    data_sources = []

    # 휴리스틱 d010, d020, d030
    for delta in [10, 20, 30]:
        df = load_heuristic_data(buffer_size, delta)
        if df is not None:
            data_sources.append((f'Heuristic d0{delta}', df, 'avg_success_rate'))

    # Prophet
    df_prophet = load_prophet_data(buffer_size)
    if df_prophet is not None:
        data_sources.append(('Prophet', df_prophet, 'avg_success_rate'))

    # DRL Eval
    df_drl = load_drl_eval_data(buffer_size)
    if df_drl is not None:
        data_sources.append(('DRL Eval', df_drl, 'mean_success'))

    # 그래프 그리기
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    line_styles = ['-', '--', '-.', ':', '-']
    markers = ['o', 's', '^', 'v', 'D']

    for idx, (label, df, col_name) in enumerate(data_sources):
        ax.plot(df['time_hours'], df[col_name],
                label=label,
                color=colors[idx],
                linestyle=line_styles[idx],
                marker=markers[idx],
                markersize=4,
                markevery=3,  # 매 3번째 포인트마다 마커 표시
                linewidth=2,
                alpha=0.8)

    # 그래프 꾸미기
    ax.set_xlabel('시간 (시)', fontsize=14, fontweight='bold')
    ax.set_ylabel('전달 성공률', fontsize=14, fontweight='bold')
    ax.set_title(f'버퍼 크기 {buffer_size}M - 시간대별 전달 성공률 비교',
                 fontsize=16, fontweight='bold', pad=20)

    # 범례
    ax.legend(loc='best', fontsize=11, framealpha=0.9, shadow=True)

    # 그리드
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # y축 범위 설정 (0.5 ~ 1.0)
    ax.set_ylim(0.5, 1.0)

    # x축 범위 설정 (0 ~ 28시간)
    ax.set_xlim(0, 28)

    # 축 틱 설정
    ax.tick_params(labelsize=11)

    # 레이아웃 조정
    plt.tight_layout()

    # 저장
    output_dir = Path('reports_plots/buffer_comparison')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'delivery_rate_comparison_buf{buffer_size}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[OK] 그래프 저장: {output_path}")

    plt.close()

def main():
    """메인 함수"""
    print("=" * 60)
    print("버퍼 크기별 전달 성공률 비교 그래프 생성")
    print("=" * 60)

    buffer_sizes = [30, 40, 50]

    for buf_size in buffer_sizes:
        print(f"\n[버퍼 {buf_size}M] 그래프 생성 중...")
        plot_buffer_comparison(buf_size)

    print("\n" + "=" * 60)
    print("[OK] 모든 그래프 생성 완료!")
    print("=" * 60)
    print(f"\n저장 위치: reports_plots/buffer_comparison/")

if __name__ == "__main__":
    main()
