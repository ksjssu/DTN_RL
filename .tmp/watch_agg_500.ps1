$in = 'reports\drl_episode_rewards.txt'
$out = 'reports\drl_episode_rewards_1_500.csv'
$lastCount = -1
while ($true) {
  if (Test-Path $in) {
    $lines = Get-Content $in | Where-Object { $_ -match '^episode ' }
    # 우선 reward 형식 우선, 없으면 delivered로 대체
    $rows = foreach ($l in $lines) {
      if ($l -match 'episode\s+(\d+)\s+reward\s+([\-\d\.]+)') {
        [pscustomobject]@{ ep=[int]$matches[1]; reward=[double]$matches[2] }
      } elseif ($l -match 'episode\s+(\d+).*?delivered\s+(\d+)') {
        [pscustomobject]@{ ep=[int]$matches[1]; reward=[double]$matches[2] }
      }
    }
    $uniq = $rows | Sort-Object ep | Group-Object ep | ForEach-Object { $_.Group[-1] }
    $uniq | Sort-Object ep | Export-Csv -Path $out -NoTypeInformation -Encoding UTF8
    $cnt = ($uniq | Measure-Object).Count
    if ($cnt -ne $lastCount) { Write-Host "현재 집계: $cnt/500 에피소드 (파일: $out)"; $lastCount = $cnt }
    if ($cnt -ge 500) { Write-Host "완료: 500/500 집계"; break }
  }
  Start-Sleep -Seconds 10
}
