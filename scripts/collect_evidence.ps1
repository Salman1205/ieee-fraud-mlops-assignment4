$paths = @(
  "artifacts/metrics/model_metrics.csv",
  "artifacts/cost_sensitive/comparison.csv",
  "artifacts/drift/drift_report.json",
  "artifacts/retraining/decision.json",
  "artifacts/explainability/feature_importance.csv",
  "artifacts/explainability/shap_summary.png"
)

Write-Host "Evidence files:"
foreach ($p in $paths) {
  if (Test-Path $p) { Write-Host "FOUND: $p" } else { Write-Host "MISSING: $p" }
}
