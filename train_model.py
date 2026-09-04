"""
TASK 4: High-Imbalance Clinical Disease Predictor Pipeline
Dataset: Breast Cancer Wisconsin (built into scikit-learn, no download needed)
Techniques: SMOTE oversampling + SVM/RandomForest/XGBoost + GridSearchCV
Metric focus: Recall (catching positive/malignant cases matters most)
"""

import joblib
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier

# 1. Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target  # 0 = malignant, 1 = benign

print("Class distribution:\n", pd.Series(y).value_counts())

# 2. Train/test split (stratified to preserve class ratio)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 3. Define pipelines: SMOTE -> Scale -> Model
models_and_grids = {
    "SVM": (
        ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True))
        ]),
        {"clf__C": [0.1, 1, 10], "clf__kernel": ["rbf", "linear"]}
    ),
    "RandomForest": (
        ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 5, 10]}
    ),
    "XGBoost": (
        ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("clf", XGBClassifier(eval_metric="logloss", random_state=42))
        ]),
        {"clf__n_estimators": [100, 200], "clf__max_depth": [3, 5]}
    ),
}

best_overall = None
best_score = -1
best_name = None

# 4. Tune each model, optimizing for recall
for name, (pipe, grid) in models_and_grids.items():
    print(f"\n--- Tuning {name} ---")
    search = GridSearchCV(pipe, grid, scoring="recall", cv=cv, n_jobs=-1)
    search.fit(X_train, y_train)

    y_pred = search.predict(X_test)
    test_recall = recall_score(y_test, y_pred)
    print(f"Best CV recall: {search.best_score_:.4f}")
    print(f"Test recall: {test_recall:.4f}")
    print(classification_report(y_test, y_pred))

    if test_recall > best_score:
        best_score = test_recall
        best_overall = search.best_estimator_
        best_name = name

print(f"\nBest model: {best_name} with test recall {best_score:.4f}")

# 5. Save best model + feature names
joblib.dump({"model": best_overall, "features": list(X.columns)}, "clinical_model.joblib")
print("Saved as clinical_model.joblib")
