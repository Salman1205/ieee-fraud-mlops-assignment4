if (Test-Path ".\.venv\Scripts\Activate.ps1") {
  .\.venv\Scripts\Activate.ps1
}
python -m src.train_models --config configs/train_config.yaml
python -m src.cost_sensitive --config configs/train_config.yaml
python -m src.drift_simulation --config configs/train_config.yaml
python -m src.retraining --config configs/train_config.yaml
python -m src.explainability --config configs/train_config.yaml
python -m pipelines.kubeflow_pipeline
