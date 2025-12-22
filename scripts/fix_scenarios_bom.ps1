$ErrorActionPreference = "Stop"
# Remove UTF-8 BOM from scenario .txt files to prevent 'default_scenario'
# Applies to all files under scenarios\dynamic (including drl_* if desired)

function Remove-BomIfPresent($path) {
  $bytes = [System.IO.File]::ReadAllBytes($path)
  if ($bytes.Length -ge 3 -and $bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191) {
    $out = $bytes[3..($bytes.Length-1)]
    [System.IO.File]::WriteAllBytes($path, $out)
    Write-Host "Fixed BOM:" $path
  }
}

$root = Join-Path $PSScriptRoot '..\scenarios\dynamic'
if (-not (Test-Path $root)) { Write-Error "Folder not found: $root"; exit 1 }

Get-ChildItem -Path $root -Recurse -File -Filter *.txt | ForEach-Object {
  Remove-BomIfPresent $_.FullName
}

Write-Host "BOM check complete."

