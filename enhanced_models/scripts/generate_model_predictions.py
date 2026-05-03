from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


RANDOM_STATE = 42
TEST_SIZE = 0.20

# Script is intended to be run from enhanced_models/scripts/
INPUT_FILE = Path("../data/stratified_sample.csv")
MODELS_DIR = Path("../models")
RESULTS_DIR = Path("../results")
CLEAN_OUTPUT_FILE = RESULTS_DIR / "model_predictions.csv"
DETAILED_OUTPUT_FILE = RESULTS_DIR / "model_predictions_detailed.csv"

MODEL_ORDER = ["lr", "lr_gs", "xgb", "xgb_gs", "gb"]


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the same feature engineering used in run_models.py."""
    df = df.copy()

    df["frp_original"] = df["frp"]
    df["frp"] = np.log1p(df["frp"])

    df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
    df = df.dropna(subset=["acq_date"]).copy()
    df["month"] = df["acq_date"].dt.month

    df["confidence"] = df["confidence"].map({"l": 0, "n": 1, "h": 2})
    df["is_day"] = (df["daynight"] == "D").astype(int)

    df = df.drop(
        columns=["version", "daynight", "acq_date", "satellite", "instrument"],
        errors="ignore",
    )

    return df


def compute_original_scale_outputs(y_true_log: pd.Series, y_pred_log: np.ndarray) -> dict:
    """Convert log-scale actual/predicted values back to original FRP scale."""
    actual = np.expm1(y_true_log.to_numpy())
    predicted = np.expm1(y_pred_log)
    predicted = np.clip(predicted, a_min=0, a_max=None)

    difference = actual - predicted
    abs_error = np.abs(difference)

    percent_error = np.full_like(actual, fill_value=np.nan, dtype=float)
    nonzero_mask = actual != 0
    percent_error[nonzero_mask] = (abs_error[nonzero_mask] / actual[nonzero_mask]) * 100

    metrics = {
        "R2": float(r2_score(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MSE": float(mean_squared_error(actual, predicted)),
        "MAE": float(mean_absolute_error(actual, predicted)),
    }

    return {
        "actual": actual,
        "predicted": predicted,
        "difference": difference,
        "abs_error": abs_error,
        "percent_error": percent_error,
        "log_actual": y_true_log.to_numpy(),
        "log_predicted": y_pred_log,
        "metrics": metrics,
    }


def add_model_columns(
    clean_df: pd.DataFrame,
    detailed_df: pd.DataFrame,
    model_name: str,
    outputs: dict,
) -> None:
    """Add clean and detailed prediction columns for one model."""
    prefix = f"enhanced_{model_name}"
    metrics = outputs["metrics"]

    clean_df[f"{prefix}_frp"] = outputs["actual"]
    clean_df[f"{prefix}_predict_frp"] = outputs["predicted"]
    clean_df[f"{prefix}_frp_difference"] = outputs["difference"]
    clean_df[f"{prefix}_R2"] = metrics["R2"]
    clean_df[f"{prefix}_RMSE"] = metrics["RMSE"]
    clean_df[f"{prefix}_MSE"] = metrics["MSE"]
    clean_df[f"{prefix}_MAE"] = metrics["MAE"]

    detailed_df[f"{prefix}_frp"] = outputs["actual"]
    detailed_df[f"{prefix}_predict_frp"] = outputs["predicted"]
    detailed_df[f"{prefix}_frp_difference"] = outputs["difference"]
    detailed_df[f"{prefix}_R2"] = metrics["R2"]
    detailed_df[f"{prefix}_RMSE"] = metrics["RMSE"]
    detailed_df[f"{prefix}_MSE"] = metrics["MSE"]
    detailed_df[f"{prefix}_MAE"] = metrics["MAE"]
    detailed_df[f"{prefix}_abs_error"] = outputs["abs_error"]
    detailed_df[f"{prefix}_percent_error"] = outputs["percent_error"]
    detailed_df[f"{prefix}_log_frp"] = outputs["log_actual"]
    detailed_df[f"{prefix}_log_predict_frp"] = outputs["log_predicted"]


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_FILE}")

    missing_models = [name for name in MODEL_ORDER if not (MODELS_DIR / f"{name}.joblib").exists()]
    if missing_models:
        raise FileNotFoundError(
            "Missing model file(s): "
            + ", ".join(str(MODELS_DIR / f"{name}.joblib") for name in missing_models)
        )

    df = pd.read_csv(INPUT_FILE)
    df = apply_feature_engineering(df)

    X = df.drop(columns=["frp", "frp_original"])
    y_log = df["frp"]

    _, X_test, _, y_test_log = train_test_split(
        X,
        y_log,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    clean_df = pd.DataFrame(index=range(len(X_test)))
    detailed_df = pd.DataFrame(index=range(len(X_test)))

    for model_name in MODEL_ORDER:
        model_path = MODELS_DIR / f"{model_name}.joblib"
        model = joblib.load(model_path)

        y_pred_log = model.predict(X_test)
        outputs = compute_original_scale_outputs(y_test_log, y_pred_log)
        add_model_columns(clean_df, detailed_df, model_name, outputs)

        print(f"Added predictions for {model_name}")

    clean_df.to_csv(CLEAN_OUTPUT_FILE, index=False)
    detailed_df.to_csv(DETAILED_OUTPUT_FILE, index=False)

    print(f"Saved clean predictions to: {CLEAN_OUTPUT_FILE}")
    print(f"Saved detailed predictions to: {DETAILED_OUTPUT_FILE}")
    print(f"Rows saved: {len(clean_df)}")


if __name__ == "__main__":
    main()
