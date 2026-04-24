python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Starting Minikube..."
minikube start --cpus=4 --memory=8192
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/resource-quota.yaml
kubectl apply -f k8s/pvc.yaml

Write-Host "Setup complete."
