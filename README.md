# Alzheimer's Disease Prediction | Portfolio Project

A cleaned, leakage-safe rebuild of an end-to-end classification project using the public Alzheimer's Disease Dataset by Rabie El Kharoua.

## What is included

- `alzheimers_prediction_portfolio.ipynb` — fully structured analysis and modelling notebook
- `data/alzheimers_disease_data.csv` — project dataset
- `outputs/model_comparison.csv` — cross-validation comparison after execution
- `outputs/final_test_metrics.csv` — final untouched-test metrics
- `outputs/permutation_importance.csv` — model-agnostic feature importance
- `outputs/best_model_pipeline.joblib` — fitted winning pipeline
- `requirements.txt` — minimal Python dependencies

## Methodology

1. Data quality audit and removal of identifier/confidential metadata fields.
2. Focused EDA rather than exhaustive chart dumping.
3. Corrected statistical tests: Shapiro-Wilk as a diagnostic, Welch's t-test, chi-square + Cramér's V, and point-biserial correlation.
4. Stratified train/test split before any model preprocessing.
5. Scaling kept inside sklearn pipelines to prevent leakage.
6. Five models compared with 3-fold stratified cross-validation using ROC-AUC.
7. Winning model selected from training CV only.
8. One final evaluation on an untouched holdout test set.
9. Permutation importance for model explanation.

## Reproduced result

With `random_state=42`, the rebuild selected **Random Forest** from cross-validation. In the verified run bundled with this project, it achieved approximately:

- CV ROC-AUC: **0.953**
- Holdout ROC-AUC: **0.943**
- Holdout accuracy: **0.939**
- Holdout precision: **0.949**
- Holdout recall: **0.874**
- Holdout F1: **0.910**

Exact results are saved under `outputs/`.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
jupyter notebook alzheimers_prediction_portfolio.ipynb
```

Run all cells from top to bottom.

## Responsible-use note

This repository is an educational portfolio demonstration. It is not clinically validated and must not be used to diagnose patients or make healthcare decisions.

## Dataset source

Rabie El Kharoua (2024), *Alzheimer's Disease Dataset*, Kaggle. DOI: 10.34740/KAGGLE/DSV/8668279.
