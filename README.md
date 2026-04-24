# Fraud Detection MLOps Assignment (IEEE-CIS)

This project implements an end-to-end fraud detection MLOps system for the IEEE-CIS Fraud Detection dataset with:

- Kubeflow pipeline design assets (7 stages + retries + conditional deploy gate)
- Imbalance handling comparison (`class_weight` vs `SMOTE`)
- Models: `XGBoost`, `LightGBM`, and hybrid (`RF + feature selection`)
- Cost-sensitive learning analysis
- Monitoring (Prometheus + Grafana) and alert-driven intelligent retraining
- CI/CD with GitHub Actions
- Drift simulation and explainability with SHAP

## 1) Prerequisites (Windows local)

Install:

1. Python 3.10+  
2. Git  
3. Docker Desktop (Kubernetes disabled, Minikube used separately)
4. Minikube
5. kubectl
6. Helm (optional, useful for monitoring stack)
7. (Optional) `make` equivalent or use provided PowerShell scripts

## 2) Python setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install mlflow
```

## 3) Dataset setup

1. Download IEEE-CIS data from [Kaggle competition page](https://www.kaggle.com/competitions/ieee-fraud-detection/data).
2. Put the CSVs in:

```text
data/raw/
  train_transaction.csv
  train_identity.csv
  test_transaction.csv
  test_identity.csv
```

Project config currently points to:

```text
data/Data/Data/
  train_transaction.csv
  train_identity.csv
  test_transaction.csv
  test_identity.csv
```

## 4) Local MLflow tracking (recommended on Windows)

Start MLflow UI in a separate terminal:

```powershell
.\.venv\Scripts\Activate.ps1
mlflow ui --backend-store-uri "sqlite:///mlflow.db" --default-artifact-root "./mlruns" --port 5000
```

Open: `http://127.0.0.1:5000`

## 5) Local ML workflow

```powershell
python -m src.train_models --config configs/train_config.yaml
python -m src.cost_sensitive --config configs/train_config.yaml
python -m src.drift_simulation --config configs/train_config.yaml
python -m src.retraining --config configs/train_config.yaml
python -m src.explainability --model-path artifacts/models/best_model.pkl
```

## 6) Run API with metrics

```powershell
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000
```

- Inference: `POST /predict`
- Metrics: `GET /metrics`
- Health: `GET /health`

## 7) Kubernetes + Kubeflow (design and optional local run)

```powershell
minikube start --cpus=4 --memory=8192
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/resource-quota.yaml
kubectl apply -f k8s/pvc.yaml
```

Compile pipeline:

```powershell
python -m pipelines.kubeflow_pipeline
```

If Kubeflow does not run reliably on your local machine, use the local module workflow in Sections 4-5 and keep Kubeflow files/pipeline YAML as design evidence.

## 8) Monitoring

Start API first:

```powershell
uvicorn src.serve_api:app --host 0.0.0.0 --port 8000
```

Then start Prometheus + Grafana:

```powershell
docker compose -f docker-compose.monitoring.yml up -d
```

Open:
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

Import dashboards from `monitoring/grafana_dashboards/`:
- `system_health.json`
- `model_performance.json`
- `data_drift.json`

## 9) CI/CD

GitHub Actions workflow: `.github/workflows/ci-cd.yml`.

It supports:
- CI on push/PR
- Docker build and push
- Kubeflow run trigger
- Intelligent retrain trigger via `repository_dispatch`

## 10) Tests and linting

```powershell
ruff check .
pytest -q
```

## 11) Assignment report

Use `report/report_template.md` and include:
- metrics tables
- confusion matrices
- drift screenshots
- Grafana screenshots
- CI/CD run logs

## 12) Deliverable zip

```powershell
pwsh ./scripts/package_submission.ps1
```

Generated zip excludes venv, caches, binaries, and build artifacts.
