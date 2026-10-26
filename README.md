# Diabetes Risk Prediction: Full-Stack ML Application (BRFSS 2015)

Teaching project: scikit-learn model, Flask REST API, Angular form, GitHub Actions CI and GitHub Pages.

- **GitHub repository:** https://github.com/mahnoor-2052/Full-stack-ML-application
- **Static demo (GitHub Pages):** https://mahnoor-2052.github.io/Full-stack-ML-application/

## Task
Binary classification: predict diabetes (0 = no diabetes, 1 = diabetes or pre-diabetes) from 21 health indicators.

## Dataset
CDC BRFSS 2015 "Diabetes Health Indicators" dataset (Kaggle copy), file `diabetes_binary_health_indicators_BRFSS2015.csv`: 253,680 rows, 21 features, target `Diabetes_binary` (218,334 negative, 35,346 positive). Public, anonymous survey data.

## Model and provenance
Trained by me with `notebooks/train_model.py`, following the preprocessing of the published HSSAE paper (Abdussamad et al., Scientific Reports 2025): 70:30 train/test split with `random_state=42`, min-max scaling, and SMOTE oversampling. Scaling and SMOTE are fitted on the training data only; the test set is untouched real data.
Classifier: Random Forest (100 trees, max depth 14) inside a scikit-learn Pipeline (MinMaxScaler + model). Saved as `backend/models/model_bundle.joblib`.

| Metric (30% hold-out test set, 76,104 rows) | Value |
| --- | --- |
| Accuracy | 0.807 |
| AUC | 0.819 |
| Diabetes class precision / recall | 0.37 / 0.58 |

The paper reports 89% accuracy with a deep-learning model (HSSAE). My model is a Random Forest, so the results are not directly comparable.

## Input features (21, in model order)
HighBP, HighChol, CholCheck, BMI, Smoker, Stroke, HeartDiseaseorAttack, PhysActivity, Fruits, Veggies, HvyAlcoholConsump, AnyHealthcare, NoDocbcCost, GenHlth, MentHlth, PhysHlth, DiffWalk, Sex, Age, Education, Income.
Yes/No fields are 0/1. BMI 12-98. GenHlth 1-5. MentHlth and PhysHlth 0-30. Age 1-13. Education 1-6. Income 1-8.

## Versions
Python 3.12, Node.js 22, Angular CLI 21, Flask 3.1, scikit-learn 1.9.1.

## Install