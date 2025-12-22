import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

base_dir = Path('reports_drl_eval_summary/avg_by_buffer')
out_dir = Path('plots/success_rates_eval_avg')
out_dir.mkdir(parents=True, exist_ok=True)

for buf_dir in sorted(base_dir.iterdir()):
    if not buf_dir.is_dir():
        continue
    csv_files = list(buf_dir.glob('*.csv'))
    if not csv_files:
        continue
    csv_path = csv_files[0]
    df = pd.read_csv(csv_path)
    if 'end' in df.columns:
        x = df['end'] / 3600.0
    elif 'start' in df.columns:
        x = df['start'] / 3600.0
    else:
        x = range(len(df))
    y = df['mean_success'] if 'mean_success' in df.columns else df.iloc[:, -1]
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, marker='o', linewidth=2)
    plt.title(f'Buffer {buf_dir.name} - Delivery Success Rate (Avg)')
    plt.xlabel('Time (hours)')
    plt.ylabel('Delivery Success Rate')
    plt.ylim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    out_path = out_dir / f'success_rate_{buf_dir.name}.png'
    plt.savefig(out_path)
    plt.close()
    print(f'saved {out_path}')
