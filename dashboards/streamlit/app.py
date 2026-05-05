# ================================
# Imports
# ================================
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

try:
    import shap
    SHAP_AVAILABLE = True
except:
    SHAP_AVAILABLE = False


# ================================
# App Config
# ================================
st.set_page_config(
    page_title="FRP Dashboard",
    page_icon="🔥",
    layout="wide"
)


# ================================
# File Paths
# ================================
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
ENHANCED_MODEL_DIR = ROOT / "enhanced_models" / "models"
BASELINE_MODEL_DIR = ROOT / "baseline" / "models"
MODEL_RESULTS_PATH = DATA_DIR / "model_results.csv"
ENHANCED_PREDS_PATH = DATA_DIR / "model_predictions_detailed.csv"
X_TEST_PATH = DATA_DIR / "enhanced_X_test.csv"
X_TEST_BASELINE_PATH = DATA_DIR / "X_test_baseline.csv"


# ================================
# Feature Columns
# ================================
ENHANCED_FEATURES = [
    "latitude", "longitude", "brightness", "scan", "track",
    "acq_time", "confidence", "bright_t31", "type",
    "month", "is_day"
]
BASELINE_FEATURES = [
    "latitude", "longitude", "brightness", "scan", "track",
    "acq_date", "acq_time", "satellite", "instrument",
    "confidence", "version", "bright_t31", "daynight", "type"
]

# ================================
# Enhanced Model Options Only
# ================================
MODEL_OPTIONS = {
    "Enhanced - LR": {
        "path": ENHANCED_MODEL_DIR / "lr.joblib",
        "key": "enhanced_lr",
        "model_name": "lr",
    },
    "Enhanced - LR GS": {
        "path": ENHANCED_MODEL_DIR / "lr_gs.joblib",
        "key": "enhanced_lr_gs",
        "model_name": "lr_gs",
    },
    "Enhanced - XGB": {
        "path": ENHANCED_MODEL_DIR / "xgb.joblib",
        "key": "enhanced_xgb",
        "model_name": "xgb",
    },
    "Enhanced - XGB GS": {
        "path": ENHANCED_MODEL_DIR / "xgb_gs.joblib",
        "key": "enhanced_xgb_gs",
        "model_name": "xgb_gs",
    },
    "Enhanced - GB": {
        "path": ENHANCED_MODEL_DIR / "gb.joblib",
        "key": "enhanced_gb",
        "model_name": "gb",
    },
}
SHAP_MODEL_OPTIONS = {
    "Enhanced - XGB": {
        "path": ENHANCED_MODEL_DIR / "xgb.joblib",
        "features": ENHANCED_FEATURES,
        "space": "log(FRP)",
    },
    "Enhanced - GB": {
        "path": ENHANCED_MODEL_DIR / "gb.joblib",
        "features": ENHANCED_FEATURES,
        "space": "log(FRP)",
    },
    "Baseline - XGB": {
        "path": BASELINE_MODEL_DIR / "xgb_model.joblib",
        "features": BASELINE_FEATURES,
        "space": "original FRP",
    },
    "Baseline - GB": {
        "path": BASELINE_MODEL_DIR / "gb_model.joblib",
        "features": BASELINE_FEATURES,
        "space": "original FRP",
    },
}

# ================================
# Helper Functions
# ================================
@st.cache_data
def load_csv(path):
    """Load CSV file once."""
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    """Load model once."""
    return joblib.load(path)


def prepare_features(df):
    """Keep only enhanced model features."""
    missing = [col for col in ENHANCED_FEATURES if col not in df.columns]

    if missing:
        raise ValueError(f"Missing enhanced feature columns: {missing}")

    return df[ENHANCED_FEATURES]


def predict_both_scales(model, df):
    """
    Enhanced models predict log(FRP).
    Convert log(FRP) back to original FRP.
    """
    X_ready = prepare_features(df)

    pred_log = float(model.predict(X_ready)[0])
    pred_frp = np.expm1(pred_log)

    return pred_log, pred_frp



def get_model_metrics(model_choice, results_df):
    """Get enhanced model metrics."""
    model_name = MODEL_OPTIONS[model_choice]["model_name"]
    row = results_df[results_df["model_name"] == model_name]

    if row.empty:
        return None, None, None

    r2 = float(row.iloc[0]["r2_original"])
    rmse = float(row.iloc[0]["rmse_original"])
    mae = float(row.iloc[0]["mae_original"])

    return r2, rmse, mae

def get_actual_values(preds_df, model_choice, idx):
    """Get actual FRP and actual log(FRP) for selected row."""
    model_key = MODEL_OPTIONS[model_choice]["key"]
    actual_col = f"{model_key}_frp"

    if actual_col not in preds_df.columns:
        return None, None

    actual_frp = float(preds_df.loc[idx, actual_col])
    actual_log = np.log1p(actual_frp)

    return actual_frp, actual_log


# ================================
# SHAP Functions
# ================================
def shap_sample(X, n):
    """Use smaller sample so SHAP is quicker."""
    if len(X) > n:
        return X.sample(n, random_state=42)
    return X


def prepare_shap_features(df, shap_model_choice):
    """Keep the correct feature columns for selected SHAP model."""
    features = SHAP_MODEL_OPTIONS[shap_model_choice]["features"]

    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing SHAP feature columns: {missing}")

    return df[features]


def plot_shap_local_and_global(model, X, idx, shap_model_choice, plot_type):
    """Show local and global SHAP together."""
    X_ready = prepare_shap_features(X, shap_model_choice)
    sample = shap_sample(X_ready, 500)

    explainer = shap.Explainer(model, sample)

    local_values = explainer(X_ready.iloc[[idx]])
    global_values = explainer(sample)

    left, right = st.columns(2)

    with left:
        st.subheader("Local SHAP")
        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(local_values[0], show=False, max_display=10)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("Global SHAP")
        fig = plt.figure(figsize=(8, 5))

        if plot_type == "Bar":
            shap.plots.bar(global_values, show=False, max_display=15)
        else:
            shap.plots.beeswarm(global_values, show=False, max_display=15)

        st.pyplot(fig, clear_figure=True)

# ================================
# Insight Plot Functions
# ================================
def build_error_dataframe(preds_df, model_choice, X_test):
    """Create dataframe with features, actual log(FRP), predicted log(FRP), and error."""
    model_key = MODEL_OPTIONS[model_choice]["key"]

    actual_col = f"{model_key}_frp"
    pred_col = f"{model_key}_predict_frp"

    if actual_col not in preds_df.columns or pred_col not in preds_df.columns:
        raise ValueError("Actual or predicted FRP columns not found.")

    df_plot = prepare_features(X_test).copy()

    df_plot["actual_frp"] = preds_df[actual_col].values
    df_plot["predicted_frp"] = preds_df[pred_col].values

    df_plot["actual_log_frp"] = np.log1p(df_plot["actual_frp"])
    df_plot["predicted_log_frp"] = np.log1p(df_plot["predicted_frp"].clip(lower=0))

    df_plot["error_log"] = df_plot["actual_log_frp"] - df_plot["predicted_log_frp"]
    df_plot["abs_error_log"] = df_plot["error_log"].abs()

    return df_plot


def plot_error_vs_feature(df_plot, feature):
    """Scatter plot showing prediction error against a selected feature."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df_plot[feature], df_plot["error_log"], alpha=0.35)
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel(feature)
    ax.set_ylabel("Log Error: Actual log(FRP) - Predicted log(FRP)")
    ax.set_title(f"Prediction Error vs {feature}")

    st.pyplot(fig, clear_figure=True)


def plot_error_distribution_by_group(df_plot, group_feature):
    """Boxplot showing how error changes across groups."""
    fig, ax = plt.subplots(figsize=(8, 5))

    groups = sorted(df_plot[group_feature].dropna().unique())
    data = [df_plot[df_plot[group_feature] == g]["error_log"] for g in groups]

    ax.boxplot(data, labels=groups)
    ax.axhline(0, linestyle="--", linewidth=1)

    ax.set_xlabel(group_feature)
    ax.set_ylabel("Log Error")
    ax.set_title(f"Error Distribution by {group_feature}")

    st.pyplot(fig, clear_figure=True)


def plot_feature_vs_target(df_plot, feature):
    """Scatter plot showing relationship between a feature and actual log(FRP)."""
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.scatter(df_plot[feature], df_plot["actual_log_frp"], alpha=0.35)

    ax.set_xlabel(feature)
    ax.set_ylabel("Actual log(FRP)")
    ax.set_title(f"{feature} vs Actual log(FRP)")

    st.pyplot(fig, clear_figure=True)


# ================================
# Load Data
# ================================
required_files = [
    MODEL_RESULTS_PATH,
    ENHANCED_PREDS_PATH,
    X_TEST_PATH,
    X_TEST_BASELINE_PATH
]

missing_files = [str(path) for path in required_files if not path.exists()]

if missing_files:
    st.error("Missing required files:")
    st.code("\n".join(missing_files))
    st.stop()

results_df = load_csv(MODEL_RESULTS_PATH)
enhanced_preds_df = load_csv(ENHANCED_PREDS_PATH)
X_test = load_csv(X_TEST_PATH)
X_test_baseline = load_csv(X_TEST_BASELINE_PATH)

# ================================
# Header
# ================================
st.title("🔥 FRP Prediction Dashboard")

st.info(
    "Predictions use enhanced feature-engineered models only. "
    "These models predict log(FRP)."
)


# ================================
# Tabs
# ================================
tab_pred, tab_shap, tab_insights = st.tabs([
    "Prediction",
    "SHAP Explainability",
    "Model Insights"
])


# ================================
# Prediction Tab
# ================================
with tab_pred:
    st.subheader("Prediction Settings")

    c1, c2 = st.columns(2)

    with c1:
        model_choice = st.selectbox(
            "Select enhanced model",
            list(MODEL_OPTIONS.keys()),
            key="pred_model"
        )

    with c2:
        case_index = st.slider(
            "Selected test row",
            min_value=0,
            max_value=len(X_test) - 1,
            value=0,
            step=1,
            key="pred_case"
        )

    model_path = MODEL_OPTIONS[model_choice]["path"]

    if not model_path.exists():
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = load_model(model_path)

    r2, rmse, mae = get_model_metrics(model_choice, results_df)

    m1, m2, m3 = st.columns(3)
    m1.metric("R²", f"{r2:.3f}" if r2 is not None else "N/A")
    m2.metric("RMSE", f"{rmse:.3f}" if rmse is not None else "N/A")
    m3.metric("MAE", f"{mae:.3f}" if mae is not None else "N/A")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Input")

        input_df = X_test.iloc[[case_index]]
        display_df = prepare_features(input_df)

        st.write(f"Using dataset row: **{case_index}**")
        st.dataframe(display_df, use_container_width=True)

    with right:
        st.subheader("Prediction Output")

        try:
            pred_log, pred_frp = predict_both_scales(model, input_df)

            actual_frp, actual_log = get_actual_values(
                enhanced_preds_df,
                model_choice,
                case_index
            )

            p1, p2 = st.columns(2)

            if actual_log is not None:
                p1.metric("Actual log(FRP)", f"{actual_log:.3f}")
            else:
                p1.metric("Actual log(FRP)", "N/A")

            p2.metric("Predicted log(FRP)", f"{pred_log:.3f}")

            st.caption(
                f"Converted predicted FRP: {pred_frp:.3f}"
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ================================
# SHAP Tab
# ================================
with tab_shap:
    st.subheader("SHAP Settings")

    c1, c2, c3 = st.columns(3)

    with c1:
        shap_model_choice = st.selectbox(
            "Select SHAP model",
            list(SHAP_MODEL_OPTIONS.keys()),
            key="shap_model"
        )

    with c2:
        shap_case_index = st.slider(
            "Selected test row",
            min_value=0,
            max_value=(len(X_test) - 1 if shap_model_choice.startswith("Enhanced") else len(X_test_baseline) - 1),
            value=0,
            step=1,
            key="shap_case"
        )

    with c3:
        plot_type = st.radio(
            "Global plot type",
            ["Bar", "Beeswarm"],
            horizontal=True,
            key="shap_plot_type"
        )

    shap_model_path = SHAP_MODEL_OPTIONS[shap_model_choice]["path"]

    if not shap_model_path.exists():
        st.error(f"Model file not found: {shap_model_path}")
        st.stop()

    shap_model = load_model(shap_model_path)

    st.caption(
        f"SHAP explains this model in {SHAP_MODEL_OPTIONS[shap_model_choice]['space']} space."
    )

    if not SHAP_AVAILABLE:
        st.warning("SHAP is not installed. Run: pip install shap")
    else:
        if st.button("Generate SHAP Explanations", type="primary"):
            with st.spinner("Generating local and global SHAP plots..."):
                try:
                    shap_X = X_test if shap_model_choice.startswith("Enhanced") else X_test_baseline

                    plot_shap_local_and_global(
                        shap_model,
                        shap_X,
                        shap_case_index,
                        shap_model_choice,
                        plot_type
                    )
                except Exception as e:
                    st.error("SHAP failed for this model.")
                    st.exception(e)
# ================================
# Model Insights Tab
# ================================
with tab_insights:
    st.subheader("Model Insight Settings")

    c1, c2, c3 = st.columns(3)

    with c1:
        insight_model_choice = st.selectbox(
            "Select enhanced model",
            list(MODEL_OPTIONS.keys()),
            key="insight_model"
        )

    with c2:
        numeric_features = [
            "brightness", "bright_t31", "scan", "track",
            "acq_time", "month", "confidence"
        ]

        error_feature = st.selectbox(
            "Error vs feature",
            numeric_features,
            key="error_feature"
        )

    with c3:
        group_feature = st.selectbox(
            "Error group",
            ["is_day", "month", "confidence", "type"],
            key="group_feature"
        )

    df_plot = build_error_dataframe(
        enhanced_preds_df,
        insight_model_choice,
        X_test
    )

    st.caption(
        "These plots use log(FRP) error because enhanced models were trained in log(FRP) space."
    )

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Error vs Feature")
        plot_error_vs_feature(df_plot, error_feature)

    with right:
        st.subheader("Error Distribution by Group")
        plot_error_distribution_by_group(df_plot, group_feature)

    st.divider()

    st.subheader("Feature vs Target")

    target_feature = st.selectbox(
        "Select feature for target relationship",
        numeric_features,
        key="target_feature"
    )

    plot_feature_vs_target(df_plot, target_feature)

# ================================
# Footer
# ================================
st.caption(
    "Enhanced models only. Predictions are shown in both log(FRP) and converted original FRP."
)