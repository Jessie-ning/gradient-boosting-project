# Gradient Boosting Model - Home Credit Default Risk

## Project Overview
A supervised machine learning pipeline to predict loan default risk using the
Home Credit Default Risk Kaggle dataset. The main evaluation metric is AUC.

## Results
| Model | AUC |
|---|---|
| LightGBM Tuned | 0.7718 |
| CatBoost | 0.7657 |
| LightGBM Baseline | 0.7574 |
| XGBoost | 0.7373 |
| Random Forest | 0.7208 |
| Logistic Regression | 0.6322 |

## Project Structure
gradient_boosting_project/
├── data/             # Raw dataset from Kaggle
├── notebooks/        # Jupyter notebook with full analysis
├── src/              # Python scripts
├── outputs/          # Saved charts and results
├── reports/          # Final report
└── requirements.txt  # Python dependencies

## Key Findings
- EXT_SOURCE features are the strongest predictors of default
- Feature engineering improved AUC by 0.007
- LightGBM outperformed all other models
- Class imbalance (8% default rate) makes AUC more appropriate than accuracy

## Setup
```bash
conda create -n gradient_boosting python=3.11
conda activate gradient_boosting
pip install -r requirements.txt
```

## Data
Download from Kaggle: https://www.kaggle.com/c/home-credit-default-risk/data
