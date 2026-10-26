from pathlib import Path
import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "diabetes_binary_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"

df = pd.read_csv(CSV)
print("Rows:", len(df), "| Class counts:", df[TARGET].value_counts().to_dict())
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)
features = list(X.columns)

# 70:30 split with random_state=42, as in the paper
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)

# Min-max normalisation, fitted on TRAINING data only
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)

# SMOTE on the TRAINING data only (the test set stays real and untouched)
X_res, y_res = SMOTE(random_state=42).fit_resample(X_train_s, y_train)
print("Training classes after SMOTE:", pd.Series(y_res).value_counts().to_dict())

model = RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=10,
                               n_jobs=-1, random_state=42)
model.fit(X_res, y_res)

# Saved pipeline = scaler + model (no SMOTE needed at prediction time)
pipe = Pipeline([("scale", scaler), ("model", model)])

pred = pipe.predict(X_test)
proba = pipe.predict_proba(X_test)[:, 1]
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print("AUC:", round(roc_auc_score(y_test, proba), 4))
print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))

bundle = {"pipeline": pipe, "features": features, "task": "classification",
          "model_version": "2.0", "classes": ["No diabetes", "Diabetes"]}
(ROOT / "backend" / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, ROOT / "backend" / "models" / "model_bundle.joblib", compress=3)
X_test.head(1).to_csv(ROOT / "data" / "sample_inputs.csv", index=False)
print("Saved bundle with", len(features), "features")