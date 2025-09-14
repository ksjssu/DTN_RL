$in = 'reports\drl_episode_rewards.txt'
$out = 'reports\drl_episode_rewards_unbounded_1_500.csv'
$last = -1
while ($true) {
  if (Test-Path $in) {
    $rows = @()
    Get-Content $in | ForEach-Object {
      if ($_ -match '^episode\s+(\d+)\b.*?avg_delivery_rate\s+([\d\.]+).*?delivered\s+(\d+).*?created\s+(\d+)') {
        $ep=[int]$matches[1]; $rate=[double]$matches[2]; $del=[int]$matches[3]; $cre=[int]$matches[4]
        if ($ep -ge 1 -and $ep -le 500) { $rows += [pscustomobject]@{ ep=$ep; reward=$del; rate=$rate; delivered=$del; created=$cre } }
      }
      elseif ($_ -match '^episode\s+(\d+)\b.*?reward\s+([\-\d\.]+).*?avg_delivery_rate\s+([\d\.]+).*?delivered\s+(\d+).*?created\s+(\d+)') {
        $ep=[int]$matches[1]; $rew=[double]$matches[2]; $rate=[double]$matches[3]; $del=[int]$matches[4]; $cre=[int]$matches[5]
        if ($ep -ge 1 -and $ep -le 500) { $rows += [pscustomobject]@{ ep=$ep; reward=$rew; rate=$rate; delivered=$del; created=$cre } }
      }
    }
    $uniq = $rows | Sort-Object ep | Group-Object ep | ForEach-Object { $_.Group[-1] } | Sort-Object ep
    $uniq | Export-Csv -Path $out -NoTypeInformation -Encoding UTF8
    $cnt = ($uniq | Measure-Object).Count
    if ($cnt -ne $last) { Write-Host "현재 집계: $cnt/500 (파일: $out)"; $last = $cnt }
    if ($cnt -ge 500) { Write-Host "완료: 500/500 집계"; break }
  }
  Start-Sleep -Seconds 10
}
