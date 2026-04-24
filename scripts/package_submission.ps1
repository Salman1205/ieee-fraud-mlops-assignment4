$zipName = "fraud_mlops_submission.zip"
$exclude = @(
  ".venv",
  "__pycache__",
  ".pytest_cache",
  ".ruff_cache",
  "artifacts",
  ".git"
)

Get-ChildItem -Path . -Recurse | ForEach-Object {
  foreach ($e in $exclude) {
    if ($_.FullName -like "*\$e*") { return }
  }
}

if (Test-Path $zipName) { Remove-Item $zipName -Force }

$items = Get-ChildItem -Path . -Force | Where-Object {
  $_.Name -notin @(".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".git", "artifacts", $zipName)
}

Compress-Archive -Path $items -DestinationPath $zipName -Force
Write-Host "Created $zipName"
