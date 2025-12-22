$ErrorActionPreference = "Stop"
$buffers = 60, 70, 80, 90, 100
foreach ($buf in $buffers) {
    $bufferLabel = '{0}M' -f $buf
    $scenario = "scenarios/dynamic/prophet/prophet_buf{0}_100k.txt" -f $buf
    Write-Host ("[prophet] buffer={0} :: running {1}" -f $bufferLabel, $scenario) -ForegroundColor Cyan
    & "./one.bat" -b 100 $scenario
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("Run failed for buffer={0} (exit {1})" -f $bufferLabel, $LASTEXITCODE) -ForegroundColor Red
        break
    }
}
