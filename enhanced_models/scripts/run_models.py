from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "xgboost is not installed. Install it with `pip install xgboost`."
    ) from exc


RANDOM_STATE = 42
TEST_SIZE = 0.20

INPUT_FILE = Path("../data/stratified_sample.csv")
RESULTS_FILE = Path("results/model_results.csv")
RUNTIME_FILE = Path("results/model_runtime_summary.csv")
MODELS_DIR = Path("models")


# ---------- Feature engineering ----------
def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Match feature_eng.ipynb
    df["frp_original"] = df["frp"]
    df["frp"] = np.log1p(df["frp"])

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"]).copy()
    df["month"] = df["acq_date"].dt.month

    df["confidence"] = df["confidence"].map({"l": 0, "n": 1, "h": 2})
    df["is_day"] = (df["daynight"] == "D").astype(int)

    df = df.drop(columns=["version", "daynight", "acq_date", "satellite", "instrument"], errors="ignore")
    return df


# ---------- Metrics ----------
def compute_metrics(y_true_log, y_pred_log, y_true_original):
    y_pred_original = np.expm1(y_pred_log)
    y_pred_original = np.clip(y_pred_original, a_min=0, a_max=None)

    metrics = {
        "rmse_original": float(np.sqrt(mean_squared_error(y_true_original, y_pred_original))),
        "mae_original": float(mean_absolute_error(y_true_original, y_pred_original)),
        "r2_original": float(r2_score(y_true_original, y_pred_original)),
        "rmse_log": float(np.sqrt(mean_squared_error(y_true_log, y_pred_log))),
        "mae_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "r2_log": float(r2_score(y_true_log, y_pred_log)),
    }
    return metrics


def clean_params_for_json(params: dict) -> dict:
    cleaned = {}
    for k, v in params.items():
        if isinstance(v, (np.integer,)):
            cleaned[k] = int(v)
        elif isinstance(v, (np.floating,)):
            cleaned[k] = float(v)
        else:
            cleaned[k] = v
    return cleaned


# ---------- Model runners ----------
def run_model(
    model_name: str,
    estimator,
    X_train,
    X_test,
    y_train_log,
    y_test_log,
    y_test_original,
    is_grid_search: bool,
    used_pca: bool,
):
    start = time.time()

    estimator.fit(X_train, y_train_log)
    y_pred_log = estimator.predict(X_test)

    runtime_seconds = time.time() - start
    metrics = compute_metrics(y_test_log, y_pred_log, y_test_original)

    if is_grid_search:
        fitted_model = estimator.best_estimator_
        params = clean_params_for_json(estimator.best_params_)
        pca_components = params.get("pca__n_components") if used_pca else None
    else:
        fitted_model = estimator
        params = clean_params_for_json(estimator.get_params())
        pca_components = None

    joblib.dump(fitted_model, MODELS_DIR / f"{model_name}.joblib")

    result_row = {
        "model_name": model_name,
        "is_grid_search": is_grid_search,
        "used_pca": used_pca,
        "pca_components": pca_components,
        "parameters": json.dumps({k: str(v) for k, v in params.items()}, sort_keys=True),
        **metrics,
    }

    runtime_row = {
        "model_name": model_name,
        "train_rows": int(X_train.shape[0]),
        "test_rows": int(X_test.shape[0]),
        "feature_count": int(X_train.shape[1]),
        "runtime_seconds": float(runtime_seconds),
    }

    return result_row, runtime_row


def main() -> None:
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RUNTIME_FILE.parent.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    df = apply_feature_engineering(df)

    X = df.drop(columns=["frp", "frp_original"])
    y_log = df["frp"]
    y_original = df["frp_original"]

    X_train, X_test, y_train_log, y_test_log, y_train_original, y_test_original = train_test_split(
        X,
        y_log,
        y_original,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    results_rows = []
    runtime_rows = []

    # 1) Linear Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LinearRegression()),
    ])
    row, runtime = run_model(
        model_name="lr",
        estimator=lr_pipeline,
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test_log=y_test_log,
        y_test_original=y_test_original,
        is_grid_search=False,
        used_pca=False,
    )
    results_rows.append(row)
    runtime_rows.append(runtime)

    # 2) Linear Regression with Grid Search -> Ridge + PCA
    lr_gs_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        ("ridge", Ridge(random_state=RANDOM_STATE)),
    ])
    lr_gs_param_grid = {
        "pca__n_components": [5, 8, 10],
        "ridge__alpha": [0.1, 1.0, 10.0],
    }
    lr_gs = GridSearchCV(
        estimator=lr_gs_pipeline,
        param_grid=lr_gs_param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    row, runtime = run_model(
        model_name="lr_gs",
        estimator=lr_gs,
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test_log=y_test_log,
        y_test_original=y_test_original,
        is_grid_search=True,
        used_pca=True,
    )
    results_rows.append(row)
    runtime_rows.append(runtime)

    # 3) XGBoost
    xgb_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    row, runtime = run_model(
        model_name="xgb",
        estimator=xgb_model,
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test_log=y_test_log,
        y_test_original=y_test_original,
        is_grid_search=False,
        used_pca=False,
    )
    results_rows.append(row)
    runtime_rows.append(runtime)

    # 4) XGBoost with Grid Search + PCA
    xgb_gs_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA()),
        (
            "xgb",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        ),
    ])
    xgb_gs_param_grid = {
        "pca__n_components": [5, 8, 10],
        "xgb__n_estimators": [100, 200],
        "xgb__max_depth": [3, 6],
        "xgb__learning_rate": [0.05, 0.1],
    }
    xgb_gs = GridSearchCV(
        estimator=xgb_gs_pipeline,
        param_grid=xgb_gs_param_grid,
        cv=3,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        verbose=1,
    )
    row, runtime = run_model(
        model_name="xgb_gs",
        estimator=xgb_gs,
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test_log=y_test_log,
        y_test_original=y_test_original,
        is_grid_search=True,
        used_pca=True,
    )
    results_rows.append(row)
    runtime_rows.append(runtime)

    # 5) Gradient Boosting
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=RANDOM_STATE,
    )
    row, runtime = run_model(
        model_name="gb",
        estimator=gb_model,
        X_train=X_train,
        X_test=X_test,
        y_train_log=y_train_log,
        y_test_log=y_test_log,
        y_test_original=y_test_original,
        is_grid_search=False,
        used_pca=False,
    )
    results_rows.append(row)
    runtime_rows.append(runtime)

    results_df = pd.DataFrame(results_rows)
    runtime_df = pd.DataFrame(runtime_rows)

    results_df.to_csv(RESULTS_FILE, index=False)
    runtime_df.to_csv(RUNTIME_FILE, index=False)

    print(f"Saved main results to: {RESULTS_FILE}")
    print(f"Saved runtime summary to: {RUNTIME_FILE}")
    print(f"Saved models to: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
