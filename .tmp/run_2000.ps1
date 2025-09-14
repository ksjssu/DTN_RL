$ErrorActionPreference = "Stop"
for ($i=1; $i -le 2000; $i++) {
  $p = ".tmp\seed_$i.txt"
  @(
    "MovementModel.rngSeed = $i",
    "RLStateReport.reportUrl = http://localhost:5000/episode_end"
  ) | Set-Content -Encoding UTF8 $p
  cmd /c one.bat -b 1 dynamic_traffic_hml_1h_settings.txt $p
}
