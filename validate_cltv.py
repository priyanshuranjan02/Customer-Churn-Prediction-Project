"""Reproducible CLTV ablation for Customer Churn Prediction.

Run from the repository root:
    ./.venv/bin/python validate_cltv.py

The script reproduces the existing fixed-parameter model and compares tuned
XGBoost pipelines WITH and WITHOUT CLTV on the same held-out test customers.
Candidate artifacts go to validation_results/; live model files are untouched.
"""
from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Telco_customer_churn.csv"
OUT = ROOT / "validation_results"
SEED = 42
TEST_SIZE = 0.20
FOLDS = 5

DROP_COLUMNS = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Churn Label", "Churn Score",
    "Churn Reason",
]

BEST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}

PARAM_GRID = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [3, 4, 5],
    "classifier__learning_rate": [0.03, 0.05, 0.1],
    "classifier__subsample": [0.8, 1.0],
    "classifier__colsample_bytree": [0.8, 1.0],
}


def load_data() -> tuple[pd.DataFrame, pd.Series]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH}. Put this script beside "
            "Telco_customer_churn.csv in the repository root."
        )
    df = pd.read_csv(DATA_PATH)
    required = {"Total Charges", "Churn Value", "CLTV"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")
    df["Churn Value"] = pd.to_numeric(df["Churn Value"], errors="coerce")
    df = df.dropna().copy()  # Matches the notebook's cleaning step.
    df["Churn Value"] = df["Churn Value"].astype(int)
    y = df.pop("Churn Value")
    return df, y


def build_pipeline(X_fit: pd.DataFrame, pipeline: bool = True):
    numeric = X_fit.select_dtypes(include=["number"]).columns.tolist()
    categorical = [col for col in X_fit.columns if col not in numeric]
    prep = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(
                handle_unknown="ignore", drop="first", sparse_output=False
            ), categorical),
        ],
        remainder="drop",
    )
    model = XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=1,
    )
    return Pipeline([("preprocessor", prep), ("classifier", model)])


def metrics_row(name, y_true, pred, prob, raw_count, processed_count,
                cv_auc=None, params=None, elapsed=None):
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "Variant": name,
        "Raw Predictors": int(raw_count),
        "Processed Features": int(processed_count),
        "Best CV ROC-AUC": cv_auc,
        "Test Accuracy": accuracy_score(y_true, pred),
        "Test Precision": precision_score(y_true, pred, zero_division=0),
        "Test Recall": recall_score(y_true, pred, zero_division=0),
        "Test F1": f1_score(y_true, pred, zero_division=0),
        "Test ROC-AUC": roc_auc_score(y_true, prob),
        "Test PR-AUC": average_precision_score(y_true, prob),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Best Parameters": json.dumps(params or {}, sort_keys=True),
        "Runtime Seconds": round(elapsed, 2) if elapsed is not None else None,
    }


def save_variant(name, row, y_test, pred, prob, preprocessor, model, pipeline=None):
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    folder = OUT / safe_name
    folder.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, folder / "preprocessor.pkl")
    joblib.dump(model, folder / "best_xgb_model.pkl")
    if pipeline is not None:
        joblib.dump(pipeline, folder / "best_pipeline.pkl")

    tn, fp, fn, tp = row["TN"], row["FP"], row["FN"], row["TP"]
    cm = pd.DataFrame(
        [[tn, fp], [fn, tp]],
        index=["Actual No Churn", "Actual Churn"],
        columns=["Predicted No Churn", "Predicted Churn"],
    )
    cm.to_csv(folder / "confusion_matrix.csv")
    pred_df = pd.DataFrame({
        "Actual Churn": np.asarray(y_test),
        "Predicted Churn": np.asarray(pred),
        "Churn Probability": np.asarray(prob),
    }, index=y_test.index)
    pred_df.index.name = "Original Row Index"
    pred_df.to_csv(folder / "test_predictions.csv")
    report = classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["No Churn", "Churn"], digits=4, zero_division=0,
    )
    (folder / "evaluation.txt").write_text(
        f"Variant: {name}\n\nMetrics:\n{json.dumps(row, indent=2)}\n\n"
        f"Classification report:\n{report}\n\n"
        f"Confusion matrix:\n{cm.to_string()}\n",
        encoding="utf-8",
    )
    print(f"\n--- {name} ---")
    print("Accuracy={:.4f} | Precision={:.4f} | Recall={:.4f} | F1={:.4f} | ROC-AUC={:.4f} | PR-AUC={:.4f}".format(
        row["Test Accuracy"], row["Test Precision"], row["Test Recall"],
        row["Test F1"], row["Test ROC-AUC"], row["Test PR-AUC"]
    ))
    print("Confusion matrix (rows=actual, columns=predicted):\n", cm)


def reproduce_fixed_with_cltv(X_train, X_test, y_train, y_test):
    """Recreate original notebook flow: fit preprocessor once, then fixed XGB."""
    start = time.time()
    prep = build_pipeline(X_train).named_steps["preprocessor"]
    Xtr = prep.fit_transform(X_train)
    Xte = prep.transform(X_test)
    model = XGBClassifier(
        **BEST_PARAMS, objective="binary:logistic", eval_metric="logloss",
        random_state=SEED, n_jobs=-1,
    )
    model.fit(Xtr, y_train)
    pred = model.predict(Xte)
    prob = model.predict_proba(Xte)[:, 1]
    row = metrics_row(
        "Notebook reproduction with CLTV", y_test, pred, prob,
        X_train.shape[1], Xtr.shape[1], cv_auc=0.8642,
        params=BEST_PARAMS, elapsed=time.time() - start,
    )
    save_variant(row["Variant"], row, y_test, pred, prob, prep, model)
    return row


def tune_variant(name, X_train, X_test, y_train, y_test, cv):
    start = time.time()
    pipe = build_pipeline(X_train)
    search = GridSearchCV(
        estimator=pipe, param_grid=PARAM_GRID, scoring="roc_auc", cv=cv,
        n_jobs=-1, refit=True, verbose=1, return_train_score=False,
    )
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    prep = best_pipe.named_steps["preprocessor"]
    model = best_pipe.named_steps["classifier"]
    pred = best_pipe.predict(X_test)
    prob = best_pipe.predict_proba(X_test)[:, 1]
    feature_count = prep.transform(X_train.iloc[:1]).shape[1]
    params = {k.replace("classifier__", ""): v for k, v in search.best_params_.items()}
    row = metrics_row(
        name, y_test, pred, prob, X_train.shape[1], feature_count,
        cv_auc=search.best_score_, params=params, elapsed=time.time() - start,
    )
    save_variant(name, row, y_test, pred, prob, prep, model, best_pipe)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    X, y = load_data()
    # Split once, then remove CLTV from those exact same rows for the ablation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y,
    )
    X_train_no = X_train.drop(columns=["CLTV"])
    X_test_no = X_test.drop(columns=["CLTV"])
    cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)

    print("Customer Churn CLTV Ablation")
    print("=" * 64)
    print(f"Python={platform.python_version()} | pandas={pd.__version__} | "
          f"scikit-learn={sklearn.__version__} | xgboost={xgboost.__version__}")
    print(f"Cleaned rows={len(X)} | train={len(X_train)} | test={len(X_test)}")
    print(f"Test churn rate={y_test.mean():.4%}")
    print("Both ablation candidates use the same held-out test customers.")
    print("Grid per tuned variant: 108 candidates × 5 folds = 540 fits.")
    print("Preprocessing is inside the CV pipeline for the two tuned variants.")

    results = [reproduce_fixed_with_cltv(X_train, X_test, y_train, y_test)]
    results.append(tune_variant(
        "With CLTV CV-safe", X_train, X_test, y_train, y_test, cv
    ))
    results.append(tune_variant(
        "Without CLTV CV-safe", X_train_no, X_test_no, y_train, y_test, cv
    ))

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "cltv_ablation_metrics.csv", index=False)
    metadata = {
        "dataset": DATA_PATH.name, "cleaned_rows": int(len(X)),
        "train_rows": int(len(X_train)), "test_rows": int(len(X_test)),
        "random_state": SEED, "test_size": TEST_SIZE, "cv_folds": FOLDS,
        "scoring": "roc_auc", "grid_candidates_per_variant": 108,
        "two_tuned_variants_cv_fits": 1080,
        "python": platform.python_version(), "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__, "xgboost": xgboost.__version__,
        "reported_notebook_reference": {
            "accuracy": 0.8024, "precision": 0.6491, "recall": 0.5588,
            "f1": 0.6006, "roc_auc": 0.8481, "pr_auc": 0.6627,
            "cv_roc_auc": 0.8642,
        },
    }
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    cols = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1", "Test ROC-AUC", "Test PR-AUC"]
    print("\n================ FINAL COMPARISON ================")
    print(result_df.set_index("Variant")[cols].round(4).to_string())
    tuned = result_df[result_df["Variant"].isin(["With CLTV CV-safe", "Without CLTV CV-safe"])].set_index("Variant")
    print("\nDelta (without CLTV minus with CLTV):")
    print((tuned.loc["Without CLTV CV-safe", cols] - tuned.loc["With CLTV CV-safe", cols]).round(4).to_string())
    print("\nConfusion-matrix counts:")
    print(result_df[["Variant", "TN", "FP", "FN", "TP"]].to_string(index=False))
    print(f"\nSaved outputs to: {OUT}")
    print("Do not replace deployed artifacts until CLTV availability and results are reviewed.")


if __name__ == "__main__":
    main()