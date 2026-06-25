"""
app.py — Production-ready Streamlit frontend for High-Risk Pregnancy Prediction
==============================================================================
Deployment target : Streamlit Community Cloud
Entry point       : app.py (repo root)
Models supported  : Logistic Regression, Decision Tree, Random Forest, MLP (.keras)
Scaler            : models/scaler.joblib  (fitted on 9 continuous features)

Run locally:
    streamlit run app.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─── Page config (MUST be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="High-Risk Pregnancy Predictor",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── File paths (relative to repo root) ───────────────────────────────────────
MODELS_DIR      = "models"
REPORTS_DIR     = "reports"
FIGURES_DIR     = os.path.join(REPORTS_DIR, "figures")
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.joblib")
COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")
ALL_METRICS_PATH= os.path.join(REPORTS_DIR, "all_metrics.json")
CV_RESULTS_PATH = os.path.join(REPORTS_DIR, "cv_results.json")
THRESHOLDS_PATH = os.path.join(REPORTS_DIR, "optimal_thresholds.json")

SKLEARN_MODELS = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree":       os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "Random Forest":       os.path.join(MODELS_DIR, "random_forest.joblib"),
}
MLP_PATH = os.path.join(MODELS_DIR, "mlp_model.keras")

# ─── Feature configuration ────────────────────────────────────────────────────
# Layout: label, dtype, min, max, default, step, unit_or_options_dict
FEATURE_CONFIG = {
    "maternal_age":          ("Maternal age",               "int",   14,   55,  28,   1,    "years"),
    "systolic_bp":           ("Systolic BP",                "float", 70,  200, 118,   1.0,  "mmHg"),
    "diastolic_bp":          ("Diastolic BP",               "float", 40,  130,  76,   1.0,  "mmHg"),
    "haemoglobin":           ("Haemoglobin",                "float",  5,   18,  11.5, 0.1,  "g/dL"),
    "bmi":                   ("BMI",                        "float", 15,   55,  26.0, 0.1,  "kg/m²"),
    "fasting_blood_glucose": ("Fasting blood glucose",      "float",  3,   20,   5.2, 0.1,  "mmol/L"),
    "weight_gain_kg":        ("Gestational weight gain",    "float",  0,   30,  12.0, 0.5,  "kg"),
    "parity":                ("Parity",                     "int",    0,   10,   1,   1,    "deliveries"),
    "antenatal_visits":      ("Antenatal care visits",      "int",    0,   20,   4,   1,    "visits"),
    "socioeconomic_status":  ("Socioeconomic status",       "cat",   None,None,None,  None, {"Low": 0, "Middle": 1, "High": 2}),
    "facility_type":         ("Facility type",              "cat",   None,None,None,  None, {"Primary": 0, "Secondary": 1, "Tertiary": 2}),
    "history_hypertension":  ("History of hypertension",    "bin",   None,None, 0,   None, None),
    "gestational_diabetes":  ("Gestational diabetes",       "bin",   None,None, 0,   None, None),
    "previous_caesarean":    ("Previous caesarean section", "bin",   None,None, 0,   None, None),
    "previous_preeclampsia": ("Previous pre-eclampsia",     "bin",   None,None, 0,   None, None),
}

# Exact column order used during training (see preprocess.py)
ALL_KEYS = list(FEATURE_CONFIG.keys())

# Continuous features passed to StandardScaler (binary/categorical excluded)
# The scaler was fitted on these 9 features only (n_features_in_ == 9)
CONTINUOUS_KEYS = [
    "maternal_age", "systolic_bp", "diastolic_bp", "haemoglobin",
    "bmi", "fasting_blood_glucose", "weight_gain_kg", "parity", "antenatal_visits",
]
SCALE_IDX = [ALL_KEYS.index(k) for k in CONTINUOUS_KEYS]

# ─── Cached resource loaders ──────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_scaler():
    """Load the StandardScaler fitted during training."""
    import joblib
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler not found at '{SCALER_PATH}'. Run run_pipeline.py first.")
        st.stop()
    return joblib.load(SCALER_PATH)


@st.cache_resource(show_spinner=False)
def load_sklearn_model(path: str):
    """Load a scikit-learn model from a .joblib file."""
    import joblib
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    """Load the MLP (.keras) model.  TensorFlow import is deferred to avoid
    loading the heavy library on pages that only use sklearn models."""
    try:
        from tensorflow import keras
    except ImportError:
        st.error("TensorFlow is required for the MLP model but is not installed.")
        st.stop()
    if not os.path.exists(path):
        st.error(f"MLP model file not found: {path}")
        st.stop()
    return keras.models.load_model(path)


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH, index_col=0)


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ─── Preprocessing helpers ────────────────────────────────────────────────────
def build_input_vector(values: dict) -> np.ndarray:
    """
    Convert the user's input dict into a (1, 15) float array in the exact
    column order used during training.  This order must match ALL_KEYS.
    """
    ordered = [float(values[k]) for k in ALL_KEYS]
    return np.array(ordered, dtype=np.float64).reshape(1, -1)


def apply_scaler(X_raw: np.ndarray) -> np.ndarray:
    """
    Scale only the continuous columns; leave binary and categorical columns
    unchanged.

    Two cases handled:
      1. scaler.n_features_in_ == 9  → scaler was fitted on the 9 continuous
         columns only (the standard pipeline).  Scale only those columns.
      2. scaler.n_features_in_ == 15 → scaler was fitted on all features.
         Scale the whole row (legacy / edge-case).

    This function is the *only* place scaling is applied and is the fix for
    the scaler feature-mismatch ValueError that caused previous deployment
    failures.
    """
    scaler = load_scaler()
    n_scaler_features = scaler.n_features_in_
    X_out = X_raw.copy()

    if n_scaler_features == len(ALL_KEYS):
        # Scaler covers all 15 features — scale everything
        X_out = scaler.transform(X_raw)
    elif n_scaler_features == len(CONTINUOUS_KEYS):
        # Standard case: scaler covers only the 9 continuous columns
        cont_slice = X_raw[:, SCALE_IDX]          # shape (1, 9)
        scaled_cont = scaler.transform(cont_slice) # shape (1, 9)
        X_out[0, SCALE_IDX] = scaled_cont[0]
    else:
        # Partial match — use however many features the scaler was trained on
        idx_subset = SCALE_IDX[:n_scaler_features]
        cont_slice = X_raw[:, idx_subset]
        scaled_cont = scaler.transform(cont_slice)
        X_out[0, idx_subset] = scaled_cont[0]

    return X_out


# ─── Visual helpers ───────────────────────────────────────────────────────────
def risk_colour(prob: float):
    """Return (hex_colour, risk_label) based on probability."""
    if prob < 0.40:
        return "#2ecc71", "Low Risk"
    elif prob < 0.65:
        return "#f39c12", "Moderate Risk"
    else:
        return "#e74c3c", "High Risk"


def gauge_svg(prob: float) -> str:
    """Render an SVG semi-circular gauge for the risk probability."""
    angle  = prob * 180
    rad    = np.radians(180 - angle)
    cx, cy, r = 120, 110, 90
    nx = cx + r * np.cos(rad)
    ny = cy - r * np.sin(rad)
    colour, label = risk_colour(prob)
    return f"""
<svg viewBox="0 0 240 135" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:300px;display:block;margin:auto">
  <!-- Background arc -->
  <path d="M 30 110 A 90 90 0 0 1 210 110"
        fill="none" stroke="#e0e0e0" stroke-width="20" stroke-linecap="round"/>
  <!-- Coloured progress arc -->
  <path d="M 30 110 A 90 90 0 0 1 {nx:.2f} {ny:.2f}"
        fill="none" stroke="{colour}" stroke-width="20" stroke-linecap="round"/>
  <!-- Needle tip -->
  <circle cx="{nx:.2f}" cy="{ny:.2f}" r="11" fill="{colour}" stroke="white" stroke-width="2"/>
  <!-- Probability label -->
  <text x="120" y="100" text-anchor="middle" font-size="28"
        font-weight="bold" fill="{colour}" font-family="sans-serif">{prob*100:.0f}%</text>
  <!-- Risk label -->
  <text x="120" y="124" text-anchor="middle" font-size="13"
        fill="{colour}" font-family="sans-serif" font-weight="600">{label}</text>
  <!-- Scale labels -->
  <text x="30"  y="130" font-size="11" fill="#aaa" font-family="sans-serif">0%</text>
  <text x="180" y="130" font-size="11" fill="#aaa" font-family="sans-serif">100%</text>
</svg>"""


def feature_bar_chart(values: dict) -> plt.Figure:
    """
    Visualise normalised input values against typical clinical ranges.
    Only continuous features are plotted.
    """
    clinical_ranges = {
        "maternal_age":          (15, 45, "years"),
        "systolic_bp":           (90, 160, "mmHg"),
        "diastolic_bp":          (60, 110, "mmHg"),
        "haemoglobin":           (7,  16,  "g/dL"),
        "bmi":                   (16, 45,  "kg/m²"),
        "fasting_blood_glucose": (3,  15,  "mmol/L"),
        "weight_gain_kg":        (4,  25,  "kg"),
        "parity":                (0,  8,   ""),
        "antenatal_visits":      (0,  12,  "visits"),
    }

    labels, normed = [], []
    for key, (lo, hi, unit) in clinical_ranges.items():
        val = float(values[key])
        n   = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        labels.append(FEATURE_CONFIG[key][0])
        normed.append(n)

    colours = []
    for n in normed:
        if n < 0.35:
            colours.append("#3498db")
        elif n < 0.70:
            colours.append("#2ecc71")
        else:
            colours.append("#e74c3c")

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, normed, color=colours, height=0.55)
    ax.axvline(0.35, color="#f39c12", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.axvline(0.70, color="#e74c3c", linestyle="--", linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Normalised value within clinical range")
    ax.set_title("Input feature summary", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3)
    low_p  = mpatches.Patch(color="#3498db", label="Low range")
    mid_p  = mpatches.Patch(color="#2ecc71", label="Mid range")
    high_p = mpatches.Patch(color="#e74c3c", label="High range")
    ax.legend(handles=[low_p, mid_p, high_p], loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


# ─── Sidebar navigation ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤰 High-Risk Pregnancy\nPrediction System")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["🏠 Home", "🔮 Prediction", "📊 Model Performance", "ℹ️ About"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(
        "**Final Year Project** · Predictive Model for High-Risk Pregnancy "
        "Identification Using Machine Learning\n\n"
        "_Synthetic Nigerian Maternal Health Data_"
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.title("Predictive Model for High-Risk Pregnancy Identification")
    st.subheader("A Machine Learning Decision-Support Tool for Antenatal Risk Screening")

    st.markdown("""
This system uses trained machine learning models to estimate the probability
that a patient's pregnancy profile falls into the **high-risk** category.
It is designed as a **clinical decision-support aid** for midwives, nurses,
and obstetricians at primary health centres and general hospitals in Nigeria.

> ⚠️ **Important disclaimer:** This tool was trained on **synthetic data**
> only. It is intended for academic demonstration and must **not** be used
> for autonomous clinical decisions without external validation on real
> patient data.
""")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Training records", "5,000")
    with col2:
        st.metric("Input features", "15")
    with col3:
        st.metric("Best model F1", "0.777 (MLP)")
    with col4:
        st.metric("Best model AUC", "0.933 (LR)")

    st.markdown("---")
    st.markdown("### Model summary")

    summary = pd.DataFrame({
        "Model":     ["MLP Neural Network", "Logistic Regression", "Random Forest", "Decision Tree"],
        "F1 Score":  [0.777, 0.759, 0.728, 0.619],
        "AUC-ROC":   [0.929, 0.933, 0.904, 0.819],
        "Precision": [0.712, 0.690, 0.698, 0.484],
        "Recall":    [0.856, 0.842, 0.761, 0.860],
    }).set_index("Model")
    st.dataframe(
        summary.style
            .highlight_max(axis=0, props="background-color:#d4edda;font-weight:bold;")
            .format("{:.3f}"),
        use_container_width=True,
    )

    st.markdown("### How to use")
    st.markdown("""
1. Navigate to **🔮 Prediction** in the sidebar.
2. Select a model and adjust the decision threshold if needed.
3. Fill in the patient's clinical measurements and medical history.
4. Click **Run Prediction** to see the risk probability and classification.
5. Review the feature summary chart for a visual snapshot of the inputs.
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Prediction":
    st.title("🔮 Patient Risk Prediction")
    st.markdown(
        "Enter the patient's clinical measurements and demographic details below. "
        "The selected model will output a **risk probability** and a "
        "**HIGH RISK / LOW RISK** classification."
    )

    # ── Model + threshold selector ──────────────────────────────────────────
    available_models = dict(SKLEARN_MODELS)
    if os.path.exists(MLP_PATH):
        available_models["MLP Neural Network"] = MLP_PATH

    col_sel, col_thresh = st.columns([2, 1])

    with col_sel:
        selected_model_name = st.selectbox("Select model", list(available_models.keys()))

    with col_thresh:
        # Load saved F1-optimal threshold for this model if available
        default_thresh = 0.5
        if os.path.exists(THRESHOLDS_PATH):
            try:
                saved = load_json_file(THRESHOLDS_PATH)
                key_map = {
                    "Logistic Regression": "Logistic Regression",
                    "Decision Tree":       "Decision Tree",
                    "Random Forest":       "Random Forest",
                    "MLP Neural Network":  "MLP",
                }
                default_thresh = float(saved.get(key_map.get(selected_model_name, ""), 0.5))
            except Exception:
                pass
        threshold = st.number_input(
            "Decision threshold",
            min_value=0.01, max_value=0.99,
            value=round(default_thresh, 3), step=0.01,
            help="F1-optimal threshold from validation set. Lower = more sensitive.",
        )

    st.markdown("---")

    # ── Input fields ────────────────────────────────────────────────────────
    values = {}

    with st.expander("📋 Clinical Measurements", expanded=True):
        c1, c2, c3 = st.columns(3)
        clinical_keys = [
            "systolic_bp", "diastolic_bp", "haemoglobin",
            "bmi", "fasting_blood_glucose", "weight_gain_kg",
        ]
        for i, key in enumerate(clinical_keys):
            label, dtype, mn, mx, default, step, unit = FEATURE_CONFIG[key]
            with [c1, c2, c3][i % 3]:
                values[key] = st.number_input(
                    f"{label} ({unit})",
                    min_value=float(mn), max_value=float(mx),
                    value=float(default), step=float(step), key=key,
                )

    with st.expander("👤 Demographics & Obstetric History", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            values["maternal_age"] = st.number_input(
                "Maternal age (years)", min_value=14, max_value=55, value=28, step=1
            )
            values["parity"] = st.number_input(
                "Parity (prior deliveries)", min_value=0, max_value=10, value=1, step=1
            )
        with c2:
            values["antenatal_visits"] = st.number_input(
                "Antenatal care visits", min_value=0, max_value=20, value=4, step=1
            )
            ses_map = FEATURE_CONFIG["socioeconomic_status"][6]
            values["socioeconomic_status"] = ses_map[
                st.selectbox("Socioeconomic status", list(ses_map.keys()))
            ]
        with c3:
            fac_map = FEATURE_CONFIG["facility_type"][6]
            values["facility_type"] = fac_map[
                st.selectbox("Facility type", list(fac_map.keys()))
            ]

    with st.expander("🏥 Medical History", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        for key, label, col in [
            ("history_hypertension",  "History of hypertension",    c1),
            ("gestational_diabetes",  "Gestational diabetes",        c2),
            ("previous_caesarean",    "Previous caesarean section",  c3),
            ("previous_preeclampsia", "Previous pre-eclampsia",      c4),
        ]:
            with col:
                values[key] = int(st.checkbox(label, value=False, key=key))

    # ── Validation ──────────────────────────────────────────────────────────
    warnings_list = []
    if values.get("systolic_bp", 0) <= values.get("diastolic_bp", 0):
        warnings_list.append("⚠️ Systolic BP should be greater than Diastolic BP.")
    if values.get("maternal_age", 28) >= 35:
        warnings_list.append("ℹ️ Advanced maternal age (≥35) is a known risk factor.")
    if values.get("haemoglobin", 11.5) < 8:
        warnings_list.append("⚠️ Haemoglobin < 8 g/dL — severe anaemia range.")

    for w in warnings_list:
        st.warning(w)

    st.markdown("---")

    # ── Run prediction ───────────────────────────────────────────────────────
    if st.button("▶ Run Prediction", type="primary", use_container_width=True):

        # 1. Build raw feature vector (1 × 15)
        X_raw = build_input_vector(values)

        # 2. Apply selective scaler — the core fix for the mismatch error
        try:
            X_scaled = apply_scaler(X_raw)
        except Exception as e:
            st.error(f"**Scaler error:** {e}")
            st.info(
                "Possible cause: the scaler artifact does not match this app version. "
                "Re-run `python run_pipeline.py` to regenerate model artifacts."
            )
            st.stop()

        # 3. Load model and predict
        try:
            model_path = available_models[selected_model_name]
            if selected_model_name == "MLP Neural Network":
                model = load_keras_model(model_path)
                prob  = float(model.predict(X_scaled, verbose=0)[0, 0])
            else:
                model = load_sklearn_model(model_path)
                prob  = float(model.predict_proba(X_scaled)[0, 1])
            label = int(prob >= threshold)
        except Exception as e:
            st.error(f"**Prediction error:** {e}")
            st.stop()

        # 4. Display results ─────────────────────────────────────────────────
        colour, risk_text = risk_colour(prob)

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(gauge_svg(prob), unsafe_allow_html=True)

        with res_col2:
            verdict_icon = "🔴" if label == 1 else "🟢"
            st.markdown(
                f"<h2 style='color:{colour};margin-top:0.5rem'>"
                f"{verdict_icon} {'HIGH RISK' if label == 1 else 'LOW RISK'}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Risk probability:** `{prob * 100:.1f}%`  \n"
                f"**Decision threshold:** `{threshold}`  \n"
                f"**Model used:** `{selected_model_name}`"
            )
            if label == 1:
                st.warning(
                    "⚠️ This patient's profile is associated with **elevated maternal risk**. "
                    "Clinical review and closer monitoring are recommended.",
                    icon="⚠️",
                )
            else:
                st.success(
                    "✅ This patient's profile does **not** indicate elevated risk based on "
                    "the current model and threshold.",
                    icon="✅",
                )

        # 5. Feature summary chart
        st.markdown("#### Feature input summary")
        fig = feature_bar_chart(values)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # 6. Input data table
        with st.expander("View full input summary"):
            rows = []
            for key, val in values.items():
                label_str = FEATURE_CONFIG[key][0]
                dtype     = FEATURE_CONFIG[key][1]
                unit      = FEATURE_CONFIG[key][6]
                if dtype == "cat":
                    display = {v: k for k, v in unit.items()}.get(val, val)
                elif dtype == "bin":
                    display = "Yes" if val else "No"
                else:
                    display = f"{val} {unit}"
                rows.append({"Feature": label_str, "Value": display})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Model Performance":
    st.title("📊 Model Performance")

    # Test-set results table
    st.subheader("Test-set results (n = 750)")
    if os.path.exists(COMPARISON_PATH):
        df = load_comparison()
        metric_cols = [c for c in df.columns
                       if c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        styled = (
            df[metric_cols]
            .style
            .highlight_max(axis=0, props="background-color:#d4edda;font-weight:bold;")
            .format("{:.4f}")
        )
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("No results found. Run `python run_pipeline.py` to generate `reports/model_comparison.csv`.")

    # Metric comparison bar chart
    if os.path.exists(COMPARISON_PATH):
        st.markdown("---")
        st.subheader("Metric comparison")
        df = load_comparison()
        metric_cols = [c for c in df.columns
                       if c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        metric_choice = st.selectbox(
            "Select metric",
            metric_cols,
            index=metric_cols.index("f1_score") if "f1_score" in metric_cols else 0,
        )
        fig, ax = plt.subplots(figsize=(8, 3.5))
        palette = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
        bars = ax.barh(df.index, df[metric_choice], color=palette[:len(df)], height=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric_choice.replace("_", " ").title())
        ax.set_title(f"{metric_choice.replace('_', ' ').title()} by model")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Cross-validation results
    st.markdown("---")
    st.subheader("Cross-validation results (stratified 5-fold)")
    if os.path.exists(CV_RESULTS_PATH):
        cv = load_json_file(CV_RESULTS_PATH)
        cv_rows = []
        for model_name, metrics in cv.items():
            row = {"Model": model_name}
            for metric, vals in metrics.items():
                row[metric.replace("_", " ").title()] = f"{vals['mean']:.4f} ± {vals['std']:.4f}"
            cv_rows.append(row)
        st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)
    else:
        st.info("Cross-validation results not found. Run the pipeline to generate them.")

    # Per-model detailed metrics
    st.markdown("---")
    st.subheader("Detailed per-model metrics")
    if os.path.exists(ALL_METRICS_PATH):
        all_m = load_json_file(ALL_METRICS_PATH)
        detail_model = st.selectbox("Select model", list(all_m.keys()), key="detail_model")
        detail_data  = all_m[detail_model]
        metric_items = {
            k: v for k, v in detail_data.items()
            if isinstance(v, (int, float)) and k != "confusion_matrix"
        }
        cols = st.columns(len(metric_items))
        for col, (k, v) in zip(cols, metric_items.items()):
            col.metric(k.replace("_", " ").title(), f"{v:.4f}")
    else:
        st.info("Detailed metrics not found. Run `python run_pipeline.py`.")

    # Report figures
    st.markdown("---")
    st.subheader("Report figures")
    if os.path.exists(FIGURES_DIR):
        figure_catalog = {
            "ROC curve comparison":             "roc_comparison.png",
            "Feature importance comparison":    "feature_importance_comparison.png",
            "MLP training history":             "mlp_training_history.png",
            "Confusion matrix — Logistic Regression": "cm_logistic_regression.png",
            "Confusion matrix — Decision Tree": "cm_decision_tree.png",
            "Confusion matrix — Random Forest": "cm_random_forest.png",
            "Confusion matrix — MLP":           "cm_mlp.png",
        }
        available_figs = {
            t: f for t, f in figure_catalog.items()
            if os.path.exists(os.path.join(FIGURES_DIR, f))
        }
        if available_figs:
            titles = list(available_figs.keys())
            for i in range(0, len(titles), 2):
                cols = st.columns(2)
                for j, col in enumerate(cols):
                    if i + j < len(titles):
                        title = titles[i + j]
                        with col:
                            st.markdown(f"**{title}**")
                            st.image(
                                os.path.join(FIGURES_DIR, available_figs[title]),
                                use_container_width=True,
                            )
        else:
            st.info("No figures yet. Run `python run_pipeline.py`.")
    else:
        st.info("Figures directory not found. Run `python run_pipeline.py`.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")

    st.markdown("""
### Project overview
This system is the Streamlit deployment of a Final Year Project titled:

> **"Predictive Model for High-Risk Pregnancy Identification Using Machine
> Learning: A Comparative Study with Synthetic Nigerian Maternal Health Data"**

### Dataset
- **5,000 synthetic records** generated from published Nigerian and West African
  epidemiological parameters (15 peer-reviewed studies)
- **15 clinical features**: 9 continuous, 4 binary, 2 categorical
- **Target prevalence**: ~29.6% high-risk

### Models evaluated
| Model | Type |
|---|---|
| Logistic Regression | Linear, interpretable baseline |
| Decision Tree | Rule-based, transparent |
| Random Forest | Ensemble (bagged trees) |
| MLP Neural Network | Feedforward deep learning |

### Preprocessing pipeline
1. Categorical encoding: one-hot (facility type), ordinal (SES)
2. Feature scaling: StandardScaler on 9 continuous features only
3. Class imbalance: SMOTE for sklearn models; class weights for MLP
4. Threshold optimisation: F1-optimal threshold on validation set

### Key result
The **MLP** achieves the best F1 score (0.777) while **Logistic Regression**
achieves the best AUC-ROC (0.933) — demonstrating that simpler interpretable
models can perform comparably to neural networks for this task.

### Ethical considerations
- No real patient data was used; no privacy concerns
- Clinical deployment requires external validation on real hospital data
- False-negative rate ~14% (MLP) — this tool supplements, not replaces, clinical judgement
- Feature distributions are calibrated for Nigeria; transfer to other populations
  requires recalibration

### Repository
[github.com/mbanwusifrancisca/high-risk-pregnancy-prediction-deploy](https://github.com/mbanwusifrancisca/high-risk-pregnancy-prediction-deploy)
""")
