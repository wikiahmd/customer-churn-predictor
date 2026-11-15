from __future__ import annotations

import json 
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
RANDOM_STATE = 42

def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    duplicates = int(df.duplicated().sum())
    id_duplicates = int(df.customerID.duplicated().sum())
    df = df.drop_duplicates().copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing = int(df["TotalCharges"].isna().sum())
    new_customers = df["TotalCharges"].isna() & df["tenure"].eq(0)
    df.loc[new_customers, "TotalCharges"] = 0.0
    if df["TotalCharges"].isna().any():
        raise ValueError("Unexpected missing TotalCharges for nonzero-tenure customers; inspect before training.")
    df = df.drop(columns=["customerID"])
    return df, {"original_rows": len(raw), "duplicate_rows": duplicates,
                "duplicate_customer_ids": id_duplicates, "missing_total_charges": missing,
                "zero_tenure_filled": int(new_customers.sum())}


def main() -> None:
    raw = pd.read_csv(ROOT / "telco_churn.csv")
    df, cleaning = clean_data(raw)
    y = df["Churn"].map({"No": 0, "Yes": 1})
    X = df.drop(columns="Churn")

    numerical = X.select_dtypes(include=np.number).columns.tolist()
    categorical = X.select_dtypes(exclude=np.number).columns.tolist()
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numerical),
        ("categorical", Pipeline({
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        }), categorical),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
    }
    comparison = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocessing", preprocessing), ("model", classifier)])
        scores = cross_validate(pipeline, X_train, y_train, scoring=["recall", "precision", "f1"], cv=cv, n_jobs=-1)
        comparison[name] = {metric: round(float(scores[f"test_{metric}"].mean()), 4)
                            for metric in ("recall", "precision", "f1")}
        print(f"{name:20} {comparison[name]}")

    best_name = max(comparison, key=lambda name: (comparison[name]["recall"], comparison[name]["f1"]))
    tuning_grids = {
        "Logistic Regression": {"model__C": [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]},
        "Decision Tree": {"model__max_depth": [4, 6, 8], "model__min_samples_leaf": [10, 25, 40]},
        "Random Forest": {"model__max_depth": [8, 12, None], "model__min_samples_leaf":  [2, 4]},
    }
    search = GridSearchCV(
        Pipeline([("preprocessing", preprocessing), ("model", candidates[best_name])]),
        tuning_grids[best_name], cv=cv, scoring="recall", n_jobs=-1, refit=True
    )
    search.fit(X_train, y_train)
    final_model = search.best_estimator_
    y_pred = final_model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=["Stayed", "Churned"], output_dict=True)
    confusion = confusion_matrix(y_test, y_pred).tolist()
    holdout = {
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
    }
    print("\nSelected:", best_name, "parameters:", search.best_params_)
    print("Held-out evaluation:", holdout)
    print("Confusion matrix (rows true [stay, churn], columns predicted [stay, churn]):", confusion)
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

    artifacts = {
        "cleaning": cleaning,
        "rows_after_cleaning": len(df),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "historical_churn_rate": round(float(y.mean()), 4),
        "cv_comparison": comparison,
        "selected_model": best_name,
        "tuned_parameters": search.best_params_,
        "grid_cv_recall": round(float(search.best_score_), 4),
        "test_metrics": holdout,
        "confusion_matrix": confusion,
        "classification_report": report,
    }

    joblib.dump(final_model, ROOT / "model.joblib")
    (ROOT / "metrics.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    print("Saved model.joblibe and mertics.json")


if __name__ == "__main__":
    main()