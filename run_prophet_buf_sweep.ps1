$ErrorActionPreference = "Stop"
$buffers = 5..50 | Where-Object { ($_ % 5) -eq 0 }
foreach ($buf in $buffers) {
    $label = '{0:D2}' -f $buf
    $scenario = "scenario_prophet_buf{0}_100k.txt" -f $label
    $bufferLabel = '{0}M' -f $buf
    Write-Host ("[prophet] buffer={0} :: running {1}" -f $bufferLabel, $scenario) -ForegroundColor Cyan
    $args = @("-b", "100", "default_settings.txt", "runtime_100k_rng1_100.txt", "prophet_baseline_overlay.txt", $scenario)
    & "./one.bat" @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("Run failed for buffer={0} (exit {1})" -f $bufferLabel, $LASTEXITCODE) -ForegroundColor Red
        break
    }
}
