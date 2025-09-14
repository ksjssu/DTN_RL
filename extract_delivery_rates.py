#!/usr/bin/env python3
"""
StepBinMetricsReport 파일들에서 전달성공률만 추출하여 개별 파일로 저장
"""
import os
import glob

def extract_delivery_rates(input_file, output_file):
    """StepBinMetricsReport에서 전달성공률(rate 컬럼)만 추출"""
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        delivery_rates = []
        for line in lines:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
                
            parts = line.split()
            if len(parts) >= 5:  # t0 t1 steps delivered created rate ...
                try:
                    t0 = float(parts[0])
                    rate = float(parts[5])  # rate 컬럼 (전달성공률)
                    delivery_rates.append(f"{t0:.1f}\t{rate:.6f}")
                except (ValueError, IndexError):
                    continue
        
        # 새 파일에 저장
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# time\tdelivery_rate\n")
            for rate_line in delivery_rates:
                f.write(rate_line + "\n")
                
        print(f"OK {os.path.basename(input_file)} -> {os.path.basename(output_file)} ({len(delivery_rates)} entries)")
        
    except Exception as e:
        print(f"ERROR processing {input_file}: {e}")

def main():
    reports_dir = "C:\\Users\\Public\\workspace\\git\\DTN_RL\\reports"
    
    # StepBinMetricsReport 파일들 찾기
    pattern = os.path.join(reports_dir, "*StepBinMetricsReport.txt")
    input_files = glob.glob(pattern)
    
    print(f"Found {len(input_files)} StepBinMetricsReport files")
    
    for input_file in input_files:
        # 출력 파일명 생성
        basename = os.path.basename(input_file)
        output_name = basename.replace("StepBinMetricsReport.txt", "DeliveryRates.txt")
        output_file = os.path.join(reports_dir, output_name)
        
        # 전달성공률 추출
        extract_delivery_rates(input_file, output_file)
    
    print(f"\nCompleted! Generated {len(input_files)} delivery rate files")

if __name__ == "__main__":
    main()