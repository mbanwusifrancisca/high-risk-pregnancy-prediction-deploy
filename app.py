"""
app.py — Streamlit frontend for High-Risk Pregnancy Prediction
Drop this file into the root of your project and run:
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

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="High-Risk Pregnancy Predictor",
    page_icon="🤰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Paths (relative to project root) ─────────────────────────────────────────

MODELS_DIR   = "models"
REPORTS_DIR  = "reports"
FIGURES_DIR  = os.path.join(REPORTS_DIR, "figures")

MODEL_FILES = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree":       os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "Random Forest":       os.path.join(MODELS_DIR, "random_forest.joblib"),
    "MLP Neural Network":  os.path.join(MODELS_DIR, "mlp_model.keras"),
}
SCALER_PATH     = os.path.join(MODELS_DIR, "scaler.joblib")
COMPARISON_PATH = os.path.join(REPORTS_DIR, "model_comparison.csv")
ALL_METRICS_PATH = os.path.join(REPORTS_DIR, "all_metrics.json")
CV_RESULTS_PATH  = os.path.join(REPORTS_DIR, "cv_results.json")
THRESHOLDS_PATH  = os.path.join(REPORTS_DIR, "optimal_thresholds.json")

# ─── Feature definitions (matches pipeline exactly) ───────────────────────────

FEATURE_CONFIG = {
    # name: (label, type, min, max, default, step, unit/options)
    "maternal_age":           ("Maternal age",             "int",    14,  55,   28,   1,   "years"),
    "systolic_bp":            ("Systolic blood pressure",  "float",  70,  200,  118,  1.0, "mmHg"),
    "diastolic_bp":           ("Diastolic blood pressure", "float",  40,  130,  76,   1.0, "mmHg"),
    "haemoglobin":            ("Haemoglobin",              "float",  5,   18,   11.5, 0.1, "g/dL"),
    "bmi":                    ("BMI",                      "float",  15,  55,   26.0, 0.1, "kg/m²"),
    "fasting_blood_glucose":  ("Fasting blood glucose",   "float",  3,   20,   5.2,  0.1, "mmol/L"),
    "weight_gain_kg":         ("Gestational weight gain", "float",  0,   30,   12.0, 0.5, "kg"),
    "parity":                 ("Parity",                  "int",    0,   10,   1,    1,   "deliveries"),
    "antenatal_visits":       ("Antenatal care visits",   "int",    0,   20,   4,    1,   "visits"),
    "socioeconomic_status":   ("Socioeconomic status",    "cat",    None,None,  None, None,
                               {"Low": 0, "Middle": 1, "High": 2}),
    "facility_type":          ("Facility type",           "cat",    None,None,  None, None,
                               {"Primary": 0, "Secondary": 1, "Tertiary": 2}),
    "history_hypertension":   ("History of hypertension", "bin",    None,None,  0,    None, None),
    "gestational_diabetes":   ("Gestational diabetes",    "bin",    None,None,  0,    None, None),
    "previous_caesarean":     ("Previous caesarean section","bin",  None,None,  0,    None, None),
    "previous_preeclampsia":  ("Previous pre-eclampsia",  "bin",    None,None,  0,    None, None),
}

BINARY_FEATURES = {k for k, v in FEATURE_CONFIG.items() if v[1] == "bin"}
CATEGORICAL_FEATURES = {k for k, v in FEATURE_CONFIG.items() if v[1] == "cat"}

# ─── Load resources (cached) ──────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_sklearn_model(path):
    import joblib
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_keras_model(path):
    from tensorflow import keras
    return keras.models.load_model(path)

@st.cache_resource(show_spinner=False)
def load_scaler():
    import joblib
    return joblib.load(SCALER_PATH)

@st.cache_data(show_spinner=False)
def load_comparison():
    return pd.read_csv(COMPARISON_PATH, index_col=0)

@st.cache_data(show_spinner=False)
def load_json(path):
    with open(path) as f:
        return json.load(f)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def build_input_vector(values: dict) -> np.ndarray:
    """Return a (1, 15) array in the exact column order used during training."""
    ordered = [values[k] for k in FEATURE_CONFIG]
    return np.array(ordered, dtype=float).reshape(1, -1)


def predict_sklearn(model, X_scaled, threshold=0.5):
    prob = model.predict_proba(X_scaled)[0, 1]
    label = int(prob >= threshold)
    return label, prob


def predict_mlp(model, X_scaled, threshold=0.5):
    prob = float(model.predict(X_scaled, verbose=0)[0, 0])
    label = int(prob >= threshold)
    return label, prob


def risk_colour(prob):
    if prob < 0.4:
        return "#2ecc71", "Low risk"
    elif prob < 0.65:
        return "#f39c12", "Moderate risk"
    else:
        return "#e74c3c", "High risk"


def gauge_svg(prob):
    """Simple SVG semicircle gauge."""
    angle = prob * 180          # 0° = left, 180° = right
    rad   = np.radians(180 - angle)
    cx, cy, r = 120, 110, 90
    nx = cx + r * np.cos(rad)
    ny = cy - r * np.sin(rad)
    colour, _ = risk_colour(prob)
    return f"""
    <svg viewBox="0 0 240 130" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:280px">
      <path d="M 30 110 A 90 90 0 0 1 210 110" fill="none" stroke="#eee" stroke-width="18" stroke-linecap="round"/>
      <path d="M 30 110 A 90 90 0 0 1 {nx:.1f} {ny:.1f}" fill="none" stroke="{colour}"
            stroke-width="18" stroke-linecap="round"/>
      <circle cx="{nx:.1f}" cy="{ny:.1f}" r="10" fill="{colour}"/>
      <text x="120" y="102" text-anchor="middle" font-size="26" font-weight="bold" fill="{colour}">
        {prob*100:.0f}%
      </text>
      <text x="30"  y="126" font-size="11" fill="#aaa">0%</text>
      <text x="185" y="126" font-size="11" fill="#aaa">100%</text>
    </svg>"""

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/64/pregnant.png", width=56)
    st.markdown("## High-Risk Pregnancy\nPrediction System")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🔮 Predict", "📊 Model performance", "🖼 Report figures"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Final Year Project · Nigerian Maternal Health · Synthetic data")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════

if page == "🔮 Predict":
    st.title("🔮 Patient risk prediction")
    st.markdown(
        "Enter the patient's clinical and demographic details below. "
        "The selected model will output a **risk probability** and a binary **high-risk / low-risk** classification."
    )

    # Model selector
    col_sel, col_thresh = st.columns([2, 1])
    with col_sel:
        selected_model_name = st.selectbox("Model", list(MODEL_FILES.keys()))
    with col_thresh:
        # Try to load saved optimal threshold for this model
        thresh_key_map = {
            "Logistic Regression": "Logistic Regression",
            "Decision Tree":       "Decision Tree",
            "Random Forest":       "Random Forest",
            "MLP Neural Network":  "MLP",
        }
        default_thresh = 0.5
        if os.path.exists(THRESHOLDS_PATH):
            try:
                saved = load_json(THRESHOLDS_PATH)
                key   = thresh_key_map[selected_model_name]
                default_thresh = float(saved.get(key, 0.5))
            except Exception:
                pass
        threshold = st.number_input(
            "Decision threshold",
            min_value=0.01, max_value=0.99,
            value=round(default_thresh, 3), step=0.01,
            help="Optimal F1 threshold from validation set. Adjust if needed.",
        )

    st.markdown("---")
    st.subheader("Patient details")

    values = {}

    # ── Clinical measurements ──────────────────────────────────────────────
    with st.expander("📋 Clinical measurements", expanded=True):
        c1, c2, c3 = st.columns(3)
        cols = [c1, c2, c3]
        clinical_keys = [
            "systolic_bp", "diastolic_bp", "haemoglobin",
            "bmi", "fasting_blood_glucose", "weight_gain_kg",
        ]
        for i, key in enumerate(clinical_keys):
            label, dtype, mn, mx, default, step, unit = FEATURE_CONFIG[key]
            with cols[i % 3]:
                val = st.number_input(
                    f"{label} ({unit})",
                    min_value=float(mn), max_value=float(mx),
                    value=float(default), step=float(step), key=key,
                )
                values[key] = val

    # ── Demographic ───────────────────────────────────────────────────────
    with st.expander("👤 Demographics & obstetric history", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            values["maternal_age"] = st.number_input(
                "Maternal age (years)", min_value=14, max_value=55, value=28, step=1,
            )
            values["parity"] = st.number_input(
                "Parity (deliveries)", min_value=0, max_value=10, value=1, step=1,
            )
        with c2:
            values["antenatal_visits"] = st.number_input(
                "Antenatal care visits", min_value=0, max_value=20, value=4, step=1,
            )
            ses_map = FEATURE_CONFIG["socioeconomic_status"][6]
            ses_label = st.selectbox("Socioeconomic status", list(ses_map.keys()))
            values["socioeconomic_status"] = ses_map[ses_label]
        with c3:
            fac_map = FEATURE_CONFIG["facility_type"][6]
            fac_label = st.selectbox("Facility type", list(fac_map.keys()))
            values["facility_type"] = fac_map[fac_label]

    # ── Medical history ────────────────────────────────────────────────────
    with st.expander("🏥 Medical history", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        bin_keys = [
            ("history_hypertension",  "History of hypertension",   c1),
            ("gestational_diabetes",  "Gestational diabetes",      c2),
            ("previous_caesarean",    "Previous caesarean section", c3),
            ("previous_preeclampsia", "Previous pre-eclampsia",    c4),
        ]
        for key, label, col in bin_keys:
            with col:
                values[key] = int(st.checkbox(label, value=False, key=key))

    st.markdown("---")

    # ── Predict button ─────────────────────────────────────────────────────
    if st.button("Run prediction", type="primary", use_container_width=True):
        X_raw = build_input_vector(values)

        # Scale
        try:
            scaler = load_scaler()
            SCALE_COLS = [0,1,2,3,4,5,6,7,8]
        except Exception as e:
            st.error(f"Could not load scaler: {e}")
            st.stop()

        # Load and run model
        try:
            model_path = MODEL_FILES[selected_model_name]
            if selected_model_name == "MLP Neural Network":
                model = load_keras_model(model_path)
                label, prob = predict_mlp(model, X_scaled, threshold)
            else:
                model = load_sklearn_model(model_path)
                label, prob = predict_sklearn(model, X_scaled, threshold)
        except Exception as e:
            st.error(f"Could not load model `{model_path}`: {e}")
            st.stop()

        colour, risk_text = risk_colour(prob)

        # ── Result layout ──────────────────────────────────────────────────
        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.markdown(gauge_svg(prob), unsafe_allow_html=True)

        with res_col2:
            verdict_icon = "🔴" if label == 1 else "🟢"
            st.markdown(
                f"<h2 style='color:{colour};margin-top:1rem'>"
                f"{verdict_icon} {'HIGH RISK' if label == 1 else 'LOW RISK'}</h2>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**Risk probability:** `{prob*100:.1f}%`  \n"
                f"**Threshold used:** `{threshold}`  \n"
                f"**Model:** `{selected_model_name}`"
            )

            if label == 1:
                st.warning(
                    "⚠️ This patient's profile is associated with elevated maternal risk. "
                    "Clinical review and closer monitoring are recommended.",
                    icon="⚠️",
                )
            else:
                st.success(
                    "✅ This patient's profile does not indicate elevated risk based on the current model and threshold.",
                    icon="✅",
                )

        # ── Input summary table ────────────────────────────────────────────
        with st.expander("View input summary"):
            summary_rows = []
            for key, val in values.items():
                label_str = FEATURE_CONFIG[key][0]
                dtype     = FEATURE_CONFIG[key][1]
                unit      = FEATURE_CONFIG[key][6]
                if dtype == "cat":
                    reverse = {v: k for k, v in unit.items()}
                    display = reverse.get(val, val)
                elif dtype == "bin":
                    display = "Yes" if val else "No"
                else:
                    display = f"{val} {unit}"
                summary_rows.append({"Feature": label_str, "Value": display})
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "📊 Model performance":
    st.title("📊 Model performance")

    # ── Comparison table ───────────────────────────────────────────────────
    st.subheader("Test-set results")
    if os.path.exists(COMPARISON_PATH):
        df = load_comparison()

        # Highlight best value per metric
        metric_cols = [c for c in df.columns if c in
                       ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        styled = df[metric_cols].style.highlight_max(
            axis=0, props="background-color:#d4edda; font-weight:bold;"
        ).format("{:.4f}")
        st.dataframe(styled, use_container_width=True)
    else:
        st.info("Run `python run_pipeline.py` first to generate `reports/model_comparison.csv`.")

    st.markdown("---")

    # ── Bar chart comparison ───────────────────────────────────────────────
    if os.path.exists(COMPARISON_PATH):
        st.subheader("Visual comparison")
        df = load_comparison()
        metric_cols = [c for c in df.columns if c in
                       ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        metric_choice = st.selectbox("Metric to compare", metric_cols, index=metric_cols.index("f1_score") if "f1_score" in metric_cols else 0)

        fig, ax = plt.subplots(figsize=(8, 3.5))
        colours = ["#3498db", "#e67e22", "#2ecc71", "#9b59b6"]
        bars = ax.barh(df.index, df[metric_choice], color=colours[:len(df)], height=0.5)
        ax.bar_label(bars, fmt="%.4f", padding=4, fontsize=10)
        ax.set_xlim(0, 1.05)
        ax.set_xlabel(metric_choice.replace("_", " ").title())
        ax.set_title(f"{metric_choice.replace('_', ' ').title()} by model")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    st.markdown("---")

    # ── Cross-validation results ───────────────────────────────────────────
    st.subheader("Cross-validation results (5-fold)")
    if os.path.exists(CV_RESULTS_PATH):
        cv = load_json(CV_RESULTS_PATH)
        cv_rows = []
        for model_name, metrics in cv.items():
            row = {"Model": model_name}
            for metric, vals in metrics.items():
                mean = vals["mean"]
                std  = vals["std"]
                row[metric.replace("_", " ").title()] = f"{mean:.4f} ± {std:.4f}"
            cv_rows.append(row)
        st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)
    else:
        st.info("Cross-validation results not found. Run the pipeline to generate them.")

    st.markdown("---")

    # ── Detailed per-model metrics ─────────────────────────────────────────
    st.subheader("Detailed metrics")
    if os.path.exists(ALL_METRICS_PATH):
        all_m = load_json(ALL_METRICS_PATH)
        detail_model = st.selectbox("Select model", list(all_m.keys()), key="detail_model")
        detail_data = all_m[detail_model]
        metric_items = {k: v for k, v in detail_data.items()
                        if isinstance(v, (int, float)) and k not in ["confusion_matrix"]}

        cols = st.columns(len(metric_items))
        for col, (k, v) in zip(cols, metric_items.items()):
            col.metric(k.replace("_", " ").title(), f"{v:.4f}")
    else:
        st.info("Run the pipeline to generate `reports/all_metrics.json`.")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — REPORT FIGURES
# ══════════════════════════════════════════════════════════════════════════════

elif page == "🖼 Report figures":
    st.title("🖼 Report figures")
    st.markdown(
        "All figures below are generated automatically by the training pipeline and saved to `reports/figures/`."
    )

    if not os.path.exists(FIGURES_DIR):
        st.warning("Figures directory not found. Run `python run_pipeline.py` first.")
        st.stop()

    figure_catalog = {
        "ROC curve comparison": "roc_comparison.png",
        "Feature importance comparison": "feature_importance_comparison.png",
        "MLP training history": "mlp_training_history.png",
        "Confusion matrix — Logistic Regression": "cm_logistic_regression.png",
        "Confusion matrix — Decision Tree": "cm_decision_tree.png",
        "Confusion matrix — Random Forest": "cm_random_forest.png",
        "Confusion matrix — MLP": "cm_mlp.png",
    }

    available = {title: fname for title, fname in figure_catalog.items()
                 if os.path.exists(os.path.join(FIGURES_DIR, fname))}

    if not available:
        st.warning("No figures found yet. Run the pipeline to generate them.")
        st.stop()

    # Show figures in a 2-column grid
    titles = list(available.keys())
    for i in range(0, len(titles), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(titles):
                title = titles[i + j]
                path  = os.path.join(FIGURES_DIR, available[title])
                with col:
                    st.markdown(f"**{title}**")
                    st.image(path, use_container_width=True)

    # Also list any extra figures not in the catalog
    all_pngs = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
    extra    = [f for f in all_pngs if f not in figure_catalog.values()]
    if extra:
        st.markdown("---")
        st.subheader("Additional figures")
        for f in extra:
            st.image(os.path.join(FIGURES_DIR, f), caption=f, use_container_width=True)
