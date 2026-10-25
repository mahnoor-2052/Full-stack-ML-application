from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"
TARGET = "Diabetes_binary"

df = pd.read_csv(CSV)
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)
features = list(X.columns)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

scale_cols = ["BMI", "MentHlth", "PhysHlth", "GenHlth", "Age", "Education", "Income"]
pre = ColumnTransformer([("scale", StandardScaler(), scale_cols)],
                        remainder="passthrough")
model = RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=10,
                               n_jobs=-1, random_state=42)
pipe = Pipeline([("preprocess", pre), ("model", model)])
pipe.fit(X_train, y_train)

pred = pipe.predict(X_test)
print("Accuracy:", round(accuracy_score(y_test, pred), 4))
print(classification_report(y_test, pred, target_names=["No diabetes", "Diabetes"]))

bundle = {"pipeline": pipe, "features": features, "task": "classification",
          "model_version": "1.0", "classes": ["No diabetes", "Diabetes"]}
(ROOT / "backend" / "models").mkdir(parents=True, exist_ok=True)
joblib.dump(bundle, ROOT / "backend" / "models" / "model_bundle.joblib", compress=3)
X_test.head(1).to_csv(ROOT / "data" / "sample_inputs.csv", index=False)
print("Saved bundle with", len(features), "features")