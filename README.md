# Diabetes Risk Prediction (BRFSS 2015)

Full-stack ML teaching project: scikit-learn model, Flask API, Angular form,
GitHub Actions CI and GitHub Pages.

## Task
Binary classification: predict diabetes (0 = no, 1 = diabetes/pre-diabetes)
from 21 health indicators.

## Dataset
CDC BRFSS 2015 "Diabetes Health Indicators", balanced 50/50 file
(70,692 rows) in `data/`.

## Model
RandomForest inside a scikit-learn Pipeline (StandardScaler + model),
trained by `notebooks/train_model.py`. Test accuracy about 75%.

## Run locally
(Fill in later: Flask command, Angular command, test command.)

## Links
- GitHub repository: (add later)
- GitHub Pages site: (add later)

## Limitations
GitHub Pages is static only. It cannot run Flask or the model, so live
predictions work only on a local computer.
This is a teaching prototype, not medical advice.