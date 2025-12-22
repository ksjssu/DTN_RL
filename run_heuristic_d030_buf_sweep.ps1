$ErrorActionPreference = "Stop"
$buffers = 5..50 | Where-Object { ($_ % 5) -eq 0 }
foreach ($buf in $buffers) {
    $labelValue = '{0:D2}' -f $buf
    $bufferLabel = '{0}M' -f $buf
    $scenario = "scenario_heuristic_d030_buf{0}_100k.txt" -f $labelValue
    Write-Host ("[heuristic delta0.3] buffer={0} :: running {1}" -f $bufferLabel, $scenario) -ForegroundColor Yellow
    $args = @("-b", "100", "default_settings.txt", "runtime_100k_rng1_100.txt", "heuristic_buffer_overlay.txt", $scenario)
    & "./one.bat" @args
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("Run failed for buffer={0} (exit {1})" -f $bufferLabel, $LASTEXITCODE) -ForegroundColor Red
        break
    }
}
