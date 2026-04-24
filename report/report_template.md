# Research Report - Fraud MLOps Assignment

## 1. Problem Statement
- Fraud detection objective
- Why recall is critical for business

## 2. System Architecture
- Kubeflow pipeline design overview (namespace, quota, PVC, compiled pipeline spec)
- Local orchestration fallback (MLflow + Python modules) used when Kubeflow runtime is unavailable on local machine
- CI/CD + monitoring integration

## 3. Data Processing
- Missing value strategy
- High-cardinality feature treatment
- Encoding techniques used

## 4. Imbalance Strategy Comparison
- Strategy A: class weighting
- Strategy B: SMOTE
- Comparison table (Precision, Recall, F1, AUC-ROC)

## 5. Model Comparison
- XGBoost
- LightGBM
- Hybrid RF + feature selection
- Confusion matrix analysis (fraud class focus)

## 6. Cost-Sensitive Learning
- Standard vs cost-sensitive result table
- FN/FP cost assumptions
- Business impact interpretation

## 7. Monitoring and Alerting
- Prometheus metrics list
- Grafana dashboard screenshots
- Alert rule behavior

## 8. Drift Simulation and Retraining
- Time-based drift design
- Feature shifts observed
- Retraining trigger strategy (threshold/periodic/hybrid)
- Cost and stability discussion

## 9. Explainability
- Feature importance highlights
- SHAP insights for fraud prediction behavior

## 10. CI/CD Evidence
- Workflow runs
- Image build and deployment logs
- Automated retraining evidence

## 11. Environment Constraint Note (if applicable)
- Briefly explain local Kubeflow runtime constraint
- State that all assignment stages were executed locally via Python modules
- Mention artifacts produced for each stage as objective evidence
- Mention MLflow UI was used for local tracking interface

## 12. Conclusion
- Key outcomes
- Limitations and future improvements
