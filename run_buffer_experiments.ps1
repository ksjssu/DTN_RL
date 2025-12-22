# 버퍼 사이즈별 DRL 실험 전체 파이프라인
# 학습 → 평가 → 결과 분석까지 자동화

param(
    [string[]]$BufferSizes = @("05", "10", "15", "20", "25", "30", "35", "40", "45", "50"),
    [switch]$TrainingOnly,
    [switch]$EvaluationOnly,
    [switch]$SkipAnalysis
)

Write-Host "=== DTN DRL Buffer Size Experiments ===" -ForegroundColor Magenta
Write-Host "Buffer sizes: $($BufferSizes -join ', ')MB" -ForegroundColor Cyan

# 실행 권한 확인
$currentPolicy = Get-ExecutionPolicy
if ($currentPolicy -eq "Restricted") {
    Write-Host "⚠ PowerShell execution policy is Restricted. Run the following command:" -ForegroundColor Yellow
    Write-Host "Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
    exit 1
}

# 컴파일 확인
if (!(Test-Path "target")) {
    Write-Host "Compiling Java sources..." -ForegroundColor Yellow
    & .\compile.bat
    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Compilation failed!" -ForegroundColor Red
        exit 1
    }
}

$totalStart = Get-Date

# 1. 학습 단계
if (!$EvaluationOnly) {
    Write-Host "`n📚 Phase 1: Training DRL models" -ForegroundColor Green
    & .\run_all_buffer_training.ps1 -BufferSizes $BufferSizes -StartSeed 1 -EndSeed 30 -Mode "training"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Training phase failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Training phase completed" -ForegroundColor Green
}

# 2. 평가 단계
if (!$TrainingOnly) {
    Write-Host "`n🔬 Phase 2: Evaluating trained models" -ForegroundColor Green
    & .\run_all_buffer_training.ps1 -BufferSizes $BufferSizes -StartSeed 1 -EndSeed 100 -Mode "evaluation"

    if ($LASTEXITCODE -ne 0) {
        Write-Host "✗ Evaluation phase failed!" -ForegroundColor Red
        exit 1
    }
    Write-Host "✓ Evaluation phase completed" -ForegroundColor Green
}

# 3. 결과 분석 단계
if (!$SkipAnalysis -and !$TrainingOnly) {
    Write-Host "`n📊 Phase 3: Analyzing results" -ForegroundColor Green

    # 결과 디렉토리 생성
    if (!(Test-Path "analysis")) {
        New-Item -ItemType Directory -Path "analysis"
    }

    # 리포트 수집 및 분석
    foreach ($bufSize in $BufferSizes) {
        Write-Host "Analyzing results for ${bufSize}MB buffer..." -ForegroundColor Yellow

        # 리포트 파일 패턴
        $reportPattern = "reports/*buf${bufSize}*.txt"
        $reportFiles = Get-ChildItem -Path $reportPattern -ErrorAction SilentlyContinue

        if ($reportFiles.Count -gt 0) {
            Write-Host "Found $($reportFiles.Count) report files for ${bufSize}MB" -ForegroundColor Cyan

            # Python 분석 스크립트 실행 (있다면)
            if (Test-Path "toolkit/aggregate_delivery_rates.py") {
                python toolkit/aggregate_delivery_rates.py --buffer-size "${bufSize}M" --output "analysis/buf${bufSize}_summary.json"
            }
        } else {
            Write-Host "⚠ No report files found for ${bufSize}MB buffer" -ForegroundColor Yellow
        }
    }

    # 종합 비교 분석
    if (Test-Path "toolkit/plot_delivery_rates.py") {
        Write-Host "Generating comparison plots..." -ForegroundColor Yellow
        python toolkit/plot_delivery_rates.py --buffer-sizes $($BufferSizes -join ',') --output "analysis/buffer_comparison.png"
    }

    Write-Host "✓ Analysis phase completed" -ForegroundColor Green
}

$totalEnd = Get-Date
$totalDuration = $totalEnd - $totalStart

Write-Host "`n🎉 All experiments completed!" -ForegroundColor Magenta
Write-Host "Total duration: $($totalDuration.TotalHours.ToString('F1')) hours" -ForegroundColor Cyan
Write-Host "Results saved in:" -ForegroundColor White
Write-Host "  • Models: models/buf{05,10,50}/" -ForegroundColor Gray
Write-Host "  • Reports: reports/" -ForegroundColor Gray
Write-Host "  • Logs: logs/" -ForegroundColor Gray
Write-Host "  • Analysis: analysis/" -ForegroundColor Gray