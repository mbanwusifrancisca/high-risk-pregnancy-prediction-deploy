"""
app.py — High-Risk Pregnancy Prediction (Clinical Decision-Support Tool)
========================================================================
Deployment : Streamlit Community Cloud
Audience   : Midwives, nurses, and obstetricians at Nigerian PHCs and hospitals

PIPELINE (verified against scaler.feature_names_in_ and model.feature_names_in_):

  Step 1 — Scale 10 features with StandardScaler (order matters):
    maternal_age, parity, antenatal_visits, socioeconomic_status,
    systolic_bp, diastolic_bp, hemoglobin, bmi, fasting_blood_glucose, weight_gain

  Step 2 — Append 4 binary features (unscaled):
    history_hypertension, gestational_diabetes,
    previous_cesarean, previous_preeclampsia

  Step 3 — One-hot encode facility_type into 3 columns:
    facility_primary_health_centre, facility_private_clinic,
    facility_tertiary_hospital

  Step 4 — Final 17-column row → model.predict_proba()
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

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be the first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MaternaCare — High-Risk Pregnancy Predictor",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ACCESSIBILITY-FIRST UI THEME
# Palette: #E91E63 accent · #212121 text · #FFFFFF bg · WCAG AA contrast
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ─── Base reset ─── */
.stApp {
    background: #FFFFFF;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                 'Helvetica Neue', sans-serif;
    color: #212121;
}
.stApp p, .stApp li, .stApp span, .stApp div, .stApp label {
    color: #212121;
    line-height: 1.6;
}

/* ─── Typography scale (responsive with clamp) ─── */
h1, h2, h3, h4, h5, h6 {
    color: #212121 !important;
    font-weight: 700;
    line-height: 1.3;
    letter-spacing: -0.01em;
}
.stApp h1 {
    font-size: clamp(1.75rem, 5vw, 2.5rem) !important;
    border-bottom: 3px solid #E91E63;
    padding-bottom: 0.75rem;
    margin: 0 0 1.5rem 0;
}
.stApp h2 {
    font-size: clamp(1.4rem, 4vw, 1.875rem) !important;
    margin: 2rem 0 1rem 0;
}
.stApp h3 {
    font-size: clamp(1.15rem, 3.5vw, 1.375rem) !important;
    margin: 1.5rem 0 0.75rem 0;
}
.stApp h4 {
    font-size: clamp(1rem, 3vw, 1.125rem) !important;
    margin: 1rem 0 0.5rem 0;
}

/* ─── Sidebar — clean white with pink accent border ─── */
[data-testid="stSidebar"] {
    background: #FAFAFA;
    border-right: 1px solid #E0E0E0;
}
[data-testid="stSidebar"] * { color: #212121 !important; }
[data-testid="stSidebar"] h1 {
    color: #E91E63 !important;
    border-bottom: 2px solid #F8BBD0;
    font-size: 1.5rem !important;
}
[data-testid="stSidebar"] .stRadio label {
    padding: 0.5rem 0;
    font-weight: 500;
    color: #212121 !important;
}
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #616161 !important;
}

/* ─── Primary button (action button) ─── */
.stButton > button[kind="primary"] {
    background: #E91E63;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    padding: 0.875rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    min-height: 48px;             /* WCAG touch target */
    width: 100%;
    box-shadow: 0 2px 6px rgba(233, 30, 99, 0.25);
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover {
    background: #C2185B;
    box-shadow: 0 4px 10px rgba(233, 30, 99, 0.35);
}
.stButton > button[kind="primary"]:disabled {
    background: #BDBDBD;
    color: #FFFFFF !important;
    box-shadow: none;
}

/* ─── Secondary buttons ─── */
.stButton > button:not([kind="primary"]) {
    background: #FFFFFF;
    color: #212121 !important;
    border: 1.5px solid #E0E0E0;
    border-radius: 8px;
    font-weight: 500;
    min-height: 44px;
}
.stButton > button:not([kind="primary"]):hover {
    border-color: #E91E63;
    color: #E91E63 !important;
}

/* ─── Expanders as cards ─── */
[data-testid="stExpander"] {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    margin-bottom: 1.25rem;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    color: #212121 !important;
    font-weight: 600;
    font-size: 1.05rem;
    padding: 1rem 1.25rem !important;
    background: #FAFAFA;
}
[data-testid="stExpander"] summary:hover {
    background: #F5F5F5;
}
[data-testid="stExpander"] > div > div {
    padding: 1rem 1.25rem !important;
}

/* ─── Metric cards ─── */
[data-testid="stMetric"] {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-left: 4px solid #E91E63;
    border-radius: 10px;
    padding: 1.25rem;
}
[data-testid="stMetricLabel"] {
    color: #616161 !important;
    font-weight: 500;
    font-size: 0.875rem;
}
[data-testid="stMetricValue"] {
    color: #212121 !important;
    font-weight: 700;
    font-size: 1.5rem !important;
}
[data-testid="stMetricDelta"] {
    color: #616161 !important;
    font-size: 0.8rem;
}

/* ─── Input fields — visible borders, large touch targets ─── */
.stNumberInput input,
.stTextInput input,
.stSelectbox > div > div {
    background: #FFFFFF !important;
    color: #212121 !important;
    border: 1.5px solid #BDBDBD !important;
    border-radius: 8px !important;
    min-height: 44px;
    font-size: 1rem !important;
}
.stNumberInput input:focus,
.stTextInput input:focus {
    border-color: #E91E63 !important;
    box-shadow: 0 0 0 3px rgba(233, 30, 99, 0.15) !important;
    outline: none !important;
}
.stNumberInput label,
.stTextInput label,
.stSelectbox label {
    color: #212121 !important;
    font-weight: 500;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}

/* ─── Checkboxes — larger, clearer ─── */
.stCheckbox {
    padding: 0.5rem 0;
}
.stCheckbox label {
    color: #212121 !important;
    font-weight: 500;
    font-size: 0.95rem;
    line-height: 1.5;
}
.stCheckbox label > div:first-child {
    border: 2px solid #BDBDBD;
}

/* ─── Alerts (info/warning/success/error) ─── */
.stAlert {
    border-radius: 10px;
    border-left-width: 5px;
    padding: 1rem 1.25rem !important;
}
.stAlert p { color: #212121 !important; font-weight: 500; }

/* ─── DataFrames ─── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E0E0E0;
}

/* ─── Result cards ─── */
.result-card {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
}
.result-card p, .result-card div, .result-card strong {
    color: #212121 !important;
}
.result-card strong { font-weight: 700; }

/* ─── Risk badges (replaces colored headings) ─── */
.risk-badge {
    display: inline-block;
    padding: 0.625rem 1.25rem;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.02em;
    margin-bottom: 1rem;
}
.risk-badge-low {
    background: #E8F5E9;
    color: #1B5E20;
    border: 2px solid #2E7D32;
}
.risk-badge-moderate {
    background: #FFF3E0;
    color: #B85A00;
    border: 2px solid #ED6C02;
}
.risk-badge-high {
    background: #FFEBEE;
    color: #B71C1C;
    border: 2px solid #D32F2F;
}

/* ─── Probability bars ─── */
.prob-label {
    color: #616161;
    font-size: 0.875rem;
    font-weight: 600;
    margin-bottom: 0.375rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.prob-bar-track {
    background: #E0E0E0;
    border-radius: 8px;
    height: 28px;
    overflow: hidden;
    margin-bottom: 0.5rem;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.4s ease;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 0.75rem;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.9rem;
    min-width: 3rem;
}
.prob-thresholds {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #616161;
    margin-top: -0.25rem;
}

/* ─── Hero banner — clean, dark text on light bg ─── */
.hero {
    background: #FAFAFA;
    border-left: 6px solid #E91E63;
    border-radius: 12px;
    padding: 2rem 1.5rem;
    margin-bottom: 2rem;
}
.hero h1 {
    color: #212121 !important;
    border-bottom: none !important;
    margin: 0 !important;
    padding-bottom: 0 !important;
}
.hero p {
    color: #616161 !important;
    font-size: 1.05rem;
    margin-top: 0.75rem;
}
.hero p strong { color: #212121 !important; }

/* ─── Information panels (Clinical Guide page) ─── */
.info-panel {
    border-radius: 12px;
    padding: 1.25rem;
    border-left-width: 5px;
    border-left-style: solid;
    height: 100%;
}
.info-panel h4 {
    margin: 0 0 0.5rem 0 !important;
    font-weight: 700 !important;
}
.info-panel p { margin: 0 !important; font-size: 0.95rem; }

.info-panel-low {
    background: #E8F5E9;
    border-left-color: #2E7D32;
}
.info-panel-low h4, .info-panel-low p { color: #1B5E20 !important; }

.info-panel-moderate {
    background: #FFF3E0;
    border-left-color: #ED6C02;
}
.info-panel-moderate h4, .info-panel-moderate p { color: #B85A00 !important; }

.info-panel-high {
    background: #FFEBEE;
    border-left-color: #D32F2F;
}
.info-panel-high h4, .info-panel-high p { color: #B71C1C !important; }

/* ─── Section dividers ─── */
hr {
    border: none;
    border-top: 1px solid #E0E0E0;
    margin: 2rem 0 !important;
}

/* ─── Links ─── */
a {
    color: #E91E63 !important;
    text-decoration: underline;
    font-weight: 500;
}
a:hover {
    color: #C2185B !important;
}

/* ─── Mobile responsiveness ─── */
@media (max-width: 768px) {
    .stApp { padding: 0 !important; }
    .hero { padding: 1.5rem 1rem; }
    .stApp h1 { font-size: 1.625rem !important; }
    .stApp h2 { font-size: 1.25rem !important; }
    .stApp h3 { font-size: 1.1rem !important; }
    [data-testid="stExpander"] summary {
        padding: 0.875rem 1rem !important;
        font-size: 1rem;
    }
    [data-testid="stExpander"] > div > div { padding: 0.875rem 1rem !important; }
    .result-card { padding: 1.25rem; }
    .risk-badge { font-size: 0.95rem; padding: 0.5rem 1rem; }
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    [data-testid="stMetric"] { padding: 1rem; margin-bottom: 0.75rem; }
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    .stButton > button[kind="primary"] { font-size: 0.95rem; padding: 0.75rem 1rem; }
}

@media (max-width: 480px) {
    .stApp h1 { font-size: 1.5rem !important; }
    .hero { padding: 1.25rem 1rem; }
    .hero p { font-size: 0.95rem; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR       = "models"
REPORTS_DIR      = "reports"
FIGURES_DIR      = os.path.join(REPORTS_DIR, "figures")
SCALER_PATH      = os.path.join(MODELS_DIR, "scaler.joblib")
COMPARISON_PATH  = os.path.join(REPORTS_DIR, "model_comparison.csv")
ALL_METRICS_PATH = os.path.join(REPORTS_DIR, "all_metrics.json")
CV_RESULTS_PATH  = os.path.join(REPORTS_DIR, "cv_results.json")
THRESHOLDS_PATH  = os.path.join(REPORTS_DIR, "optimal_thresholds.json")

SKLEARN_MODELS = {
    "Logistic Regression": os.path.join(MODELS_DIR, "logistic_regression.joblib"),
    "Decision Tree":       os.path.join(MODELS_DIR, "decision_tree.joblib"),
    "Random Forest":       os.path.join(MODELS_DIR, "random_forest.joblib"),
}
MLP_PATH = os.path.join(MODELS_DIR, "mlp_model.keras")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE  (verified from scaler.feature_names_in_ and model.feature_names_in_)
# ══════════════════════════════════════════════════════════════════════════════

# These 10 features are scaled — order matches scaler.feature_names_in_ exactly
SCALED_FEATURES = [
    "maternal_age", "parity", "antenatal_visits", "socioeconomic_status",
    "systolic_bp", "diastolic_bp", "hemoglobin", "bmi",
    "fasting_blood_glucose", "weight_gain",
]

# These 4 binary features are appended after scaling (unscaled)
BINARY_FEATURES = [
    "history_hypertension", "gestational_diabetes",
    "previous_cesarean", "previous_preeclampsia",
]

# Facility type is one-hot encoded into 3 columns
FACILITY_OHE_COLUMNS = [
    "facility_primary_health_centre",
    "facility_private_clinic",
    "facility_tertiary_hospital",
]

# Maps the user's facility choice to its one-hot column
FACILITY_OPTIONS = {
    "Primary Health Centre":  "facility_primary_health_centre",
    "Private Clinic":         "facility_private_clinic",
    "Tertiary Hospital":      "facility_tertiary_hospital",
}

# Final column order seen by the model (17 columns total)
FINAL_FEATURE_ORDER = SCALED_FEATURES + BINARY_FEATURES + FACILITY_OHE_COLUMNS

SES_OPTIONS = {"Low": 0, "Middle": 1, "High": 2}


# ══════════════════════════════════════════════════════════════════════════════
# CACHED LOADERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_scaler():
    import joblib
    if not os.path.exists(SCALER_PATH):
        st.error(f"Scaler not found at '{SCALER_PATH}'.")
        st.stop()
    return joblib.load(SCALER_PATH)


@st.cache_resource(show_spinner=False)
def load_sklearn_model(path: str):
    import joblib
    if not os.path.exists(path):
        st.error(f"Model file not found: {path}")
        st.stop()
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def load_keras_model(path: str):
    try:
        from tensorflow import keras
    except ImportError:
        st.error("TensorFlow not installed — MLP unavailable.")
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


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING — the critical fix
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_dataframe(user_inputs: dict) -> pd.DataFrame:
    """
    Construct the 17-column feature DataFrame in the exact order the model
    expects. This is THE function that must mirror preprocess.py exactly.

    Steps:
      1. Take the 10 scaled-feature values, scale with StandardScaler
      2. Take the 4 binary features as-is (0/1)
      3. One-hot encode facility_type into 3 columns
      4. Concatenate in the order: [scaled] + [binary] + [facility_ohe]
    """
    scaler = load_scaler()

    # ─── Step 1: Build the 10 scaled features in correct order ───
    scaled_row = pd.DataFrame(
        [[user_inputs[f] for f in SCALED_FEATURES]],
        columns=SCALED_FEATURES,
    )

    # Apply scaler — n_features_in_ must equal 10
    if scaler.n_features_in_ != len(SCALED_FEATURES):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but app is "
            f"passing {len(SCALED_FEATURES)}. Check feature order."
        )
    scaled_values = scaler.transform(scaled_row)
    scaled_df = pd.DataFrame(scaled_values, columns=SCALED_FEATURES)

    # ─── Step 2: Binary features (unscaled) ───
    binary_df = pd.DataFrame(
        [[int(user_inputs[f]) for f in BINARY_FEATURES]],
        columns=BINARY_FEATURES,
    )

    # ─── Step 3: One-hot encode facility_type ───
    chosen_facility_col = FACILITY_OPTIONS[user_inputs["facility_type"]]
    facility_dict = {col: 0 for col in FACILITY_OHE_COLUMNS}
    facility_dict[chosen_facility_col] = 1
    facility_df = pd.DataFrame([facility_dict])[FACILITY_OHE_COLUMNS]

    # ─── Step 4: Concatenate and enforce final column order ───
    final_df = pd.concat([scaled_df, binary_df, facility_df], axis=1)
    final_df = final_df[FINAL_FEATURE_ORDER]  # explicit reordering safety
    return final_df


# ══════════════════════════════════════════════════════════════════════════════
# VISUAL HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def risk_colour(prob: float):
    if prob < 0.40:
        return "#2E7D32", "Low Risk", "🟢"
    elif prob < 0.65:
        return "#ED6C02", "Moderate Risk", "🟡"
    else:
        return "#D32F2F", "High Risk", "🔴"


def gauge_svg(prob: float) -> str:
    angle  = prob * 180
    rad    = np.radians(180 - angle)
    cx, cy, r = 130, 120, 95
    nx = cx + r * np.cos(rad)
    ny = cy - r * np.sin(rad)
    colour, label, _ = risk_colour(prob)
    return f"""
<svg viewBox="0 0 260 155" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:340px;display:block;margin:auto"
     role="img" aria-label="Risk probability gauge showing {prob*100:.0f} percent">
  <defs>
    <linearGradient id="bgGrad" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%"  stop-color="#2E7D32"/>
      <stop offset="50%" stop-color="#ED6C02"/>
      <stop offset="100%" stop-color="#D32F2F"/>
    </linearGradient>
  </defs>
  <path d="M 35 120 A 95 95 0 0 1 225 120"
        fill="none" stroke="url(#bgGrad)" stroke-width="22"
        stroke-linecap="round" opacity="0.20"/>
  <path d="M 35 120 A 95 95 0 0 1 {nx:.2f} {ny:.2f}"
        fill="none" stroke="{colour}" stroke-width="22" stroke-linecap="round"/>
  <circle cx="{nx:.2f}" cy="{ny:.2f}" r="13" fill="{colour}"
          stroke="#FFFFFF" stroke-width="3"/>
  <text x="130" y="105" text-anchor="middle" font-size="34"
        font-weight="bold" fill="#212121" font-family="-apple-system, Segoe UI, sans-serif">
    {prob*100:.0f}%
  </text>
  <text x="130" y="135" text-anchor="middle" font-size="14"
        fill="{colour}" font-family="-apple-system, Segoe UI, sans-serif" font-weight="700">
    {label}
  </text>
  <text x="35"  y="148" font-size="11" fill="#616161" font-family="-apple-system, Segoe UI, sans-serif">Low</text>
  <text x="200" y="148" font-size="11" fill="#616161" font-family="-apple-system, Segoe UI, sans-serif">High</text>
</svg>"""


def feature_bar_chart(user_inputs: dict) -> plt.Figure:
    """Normalised view of continuous inputs against typical clinical ranges."""
    clinical_ranges = {
        "maternal_age":          (15, 45, "years",   "Maternal age"),
        "systolic_bp":           (90, 160, "mmHg",   "Systolic BP"),
        "diastolic_bp":          (60, 110, "mmHg",   "Diastolic BP"),
        "hemoglobin":            (7,  16,  "g/dL",   "Haemoglobin"),
        "bmi":                   (16, 45,  "kg/m²",  "BMI"),
        "fasting_blood_glucose": (3,  15,  "mmol/L", "Fasting glucose"),
        "weight_gain":           (4,  25,  "kg",     "Weight gain"),
        "parity":                (0,  8,   "",       "Parity"),
        "antenatal_visits":      (0,  12,  "visits", "ANC visits"),
    }

    labels, normed, colours = [], [], []
    for key, (lo, hi, unit, label) in clinical_ranges.items():
        val = float(user_inputs[key])
        n   = max(0.0, min(1.0, (val - lo) / (hi - lo)))
        labels.append(label)
        normed.append(n)
        if n < 0.35:
            colours.append("#2E7D32")  # success green
        elif n < 0.70:
            colours.append("#E91E63")  # primary pink accent
        else:
            colours.append("#D32F2F")  # danger red
    
    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")

    bars = ax.barh(labels, normed, color=colours, height=0.6, edgecolor="white", linewidth=1.5)
    ax.axvline(0.35, color="#ED6C02", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.axvline(0.70, color="#D32F2F", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Position within typical clinical range", color="#212121", fontsize=10)
    ax.set_title("Patient input summary", fontsize=13, fontweight="bold", color="#212121", pad=12)
    ax.tick_params(colors="#212121")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_color("#616161")
    ax.grid(axis="x", alpha=0.15, color="#616161")

    low_p  = mpatches.Patch(color="#2E7D32", label="Low range")
    mid_p  = mpatches.Patch(color="#E91E63", label="Typical range")
    high_p = mpatches.Patch(color="#D32F2F", label="High range")
    ax.legend(handles=[low_p, mid_p, high_p], loc="lower right",
              fontsize=9, frameon=False)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("# 🌸 MaternaCare")
    st.markdown("##### Clinical Decision-Support")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠  Home",
            "👩‍⚕️  New Assessment",
            "📊  Model Performance",
            "📖  Clinical Guide",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("##### 💡 Quick Reference")
    st.markdown("""
- **Low risk:** <40%  
- **Moderate risk:** 40–65%  
- **High risk:** >65%  
""")
    st.markdown("---")
    st.caption(
        "Predictive ML tool trained on synthetic Nigerian maternal health "
        "data. For academic and clinical decision-support use only."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page.strip().startswith("🏠"):
    st.markdown("""
<div class="hero">
  <h1>🌸 MaternaCare</h1>
  <p><strong>High-Risk Pregnancy Prediction</strong> · A clinical decision-support
  tool for midwives, nurses, and obstetricians</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
### Welcome 👋

**MaternaCare** uses machine learning to help frontline maternal health workers
in Nigerian primary health centres, private clinics, and tertiary hospitals
quickly identify pregnant women whose clinical profile suggests **elevated
maternal risk** — supporting timely referral, closer monitoring, and informed
counselling.

> ⚠️ **This tool supports clinical judgement; it does not replace it.**
> Always combine model output with full clinical history, examination,
> and your professional expertise.
""")

    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
**1️⃣ Enter patient data**

Fill in the patient's vitals, obstetric history, and demographics.
Takes about 60 seconds.
""")
    with c2:
        st.markdown("""
**2️⃣ Get a risk score**

The model returns a probability of high-risk pregnancy along with
a clear Low / Moderate / High classification.
""")
    with c3:
        st.markdown("""
**3️⃣ Make informed decisions**

Use the score as additional input for referral, monitoring frequency,
or specialist consultation.
""")

    st.markdown("---")
    st.markdown("### Model performance at a glance")
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Training records",  "5,000")
    with c2: st.metric("Best F1 score",     "0.78",  "MLP")
    with c3: st.metric("Best AUC-ROC",      "0.93",  "Logistic")
    with c4: st.metric("Models available",  "4")

    st.markdown("---")
    st.info(
        "**Ready to begin?** Navigate to **👩‍⚕️ New Assessment** in the sidebar "
        "to evaluate a patient.",
        icon="👈",
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: NEW ASSESSMENT
# ══════════════════════════════════════════════════════════════════════════════
elif page.strip().startswith("👩‍⚕️"):
    st.markdown("# 👩‍⚕️ New Patient Assessment")
    st.markdown(
        "Enter the patient's information below. All fields are required for an "
        "accurate prediction. Click **Run Assessment** when ready."
    )

    # ─── Model + threshold (compact, in expander to reduce clutter) ─────────
    with st.expander("⚙️  Advanced settings", expanded=False):
        available_models = dict(SKLEARN_MODELS)
        if os.path.exists(MLP_PATH):
            available_models["MLP Neural Network"] = MLP_PATH

        c1, c2 = st.columns([2, 1])
        with c1:
            selected_model_name = st.selectbox(
                "Prediction model",
                list(available_models.keys()),
                index=0,
                help="Logistic Regression is the most interpretable. "
                     "MLP gives the highest F1 score.",
            )
        with c2:
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
                help="Lower threshold = more sensitive (more positives, fewer missed).",
            )

    user_inputs = {}

    # ═══ SECTION 1: Demographics ════════════════════════════════════════════
    with st.expander("👤  Demographics & socioeconomic context", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["maternal_age"] = st.number_input(
                "Maternal age (years)",
                min_value=14, max_value=55, value=28, step=1,
                help="Age <18 or ≥35 carries increased obstetric risk.",
            )
        with c2:
            user_inputs["socioeconomic_status"] = SES_OPTIONS[
                st.selectbox(
                    "Socioeconomic status",
                    list(SES_OPTIONS.keys()),
                    index=1,
                    help="Based on household income, education, occupation.",
                )
            ]
        with c3:
            user_inputs["facility_type"] = st.selectbox(
                "Care facility",
                list(FACILITY_OPTIONS.keys()),
                index=0,
                help="Type of facility providing antenatal care.",
            )

    # ═══ SECTION 2: Obstetric history ═══════════════════════════════════════
    with st.expander("🤱  Obstetric history", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["parity"] = st.number_input(
                "Parity (prior births)",
                min_value=0, max_value=10, value=1, step=1,
                help="Number of previous pregnancies past 28 weeks.",
            )
        with c2:
            user_inputs["antenatal_visits"] = st.number_input(
                "ANC visits to date",
                min_value=0, max_value=20, value=4, step=1,
                help="WHO recommends ≥8 antenatal contacts.",
            )
        with c3:
            user_inputs["weight_gain"] = st.number_input(
                "Gestational weight gain (kg)",
                min_value=0.0, max_value=30.0, value=12.0, step=0.5,
                help="IOM target ~11.5–16 kg for normal BMI.",
            )

    # ═══ SECTION 3: Vital signs and laboratory ══════════════════════════════
    with st.expander("🩺  Vital signs & laboratory", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            user_inputs["systolic_bp"] = st.number_input(
                "Systolic BP (mmHg)",
                min_value=70.0, max_value=200.0, value=118.0, step=1.0,
                help="≥140 suggests hypertension.",
            )
            user_inputs["bmi"] = st.number_input(
                "BMI (kg/m²)",
                min_value=15.0, max_value=55.0, value=26.0, step=0.1,
                help="Pre-pregnancy if available.",
            )
        with c2:
            user_inputs["diastolic_bp"] = st.number_input(
                "Diastolic BP (mmHg)",
                min_value=40.0, max_value=130.0, value=76.0, step=1.0,
                help="≥90 suggests hypertension.",
            )
            user_inputs["fasting_blood_glucose"] = st.number_input(
                "Fasting glucose (mmol/L)",
                min_value=3.0, max_value=20.0, value=5.2, step=0.1,
                help="≥5.1 suggests gestational diabetes.",
            )
        with c3:
            user_inputs["hemoglobin"] = st.number_input(
                "Haemoglobin (g/dL)",
                min_value=5.0, max_value=18.0, value=11.5, step=0.1,
                help="<11 indicates anaemia in pregnancy.",
            )

    # ═══ SECTION 4: Medical history (binary risk factors) ═══════════════════
    with st.expander("🏥  Risk factors & medical history", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            user_inputs["history_hypertension"] = int(st.checkbox(
                "🩸 History of hypertension",
                help="Chronic HTN diagnosed before pregnancy or before 20 weeks.",
            ))
            user_inputs["previous_cesarean"] = int(st.checkbox(
                "🔪 Previous caesarean section",
            ))
        with c2:
            user_inputs["gestational_diabetes"] = int(st.checkbox(
                "🍬 Gestational diabetes (current or prior)",
            ))
            user_inputs["previous_preeclampsia"] = int(st.checkbox(
                "⚠️ Previous pre-eclampsia",
            ))

    # ─── Real-time validation hints ─────────────────────────────────────────
    hints = []
    if user_inputs.get("systolic_bp", 0) <= user_inputs.get("diastolic_bp", 0):
        hints.append(("error", "Systolic BP must be greater than Diastolic BP."))
    if user_inputs.get("maternal_age", 28) >= 35:
        hints.append(("info", "ℹ️ Advanced maternal age (≥35) is a known risk factor."))
    if user_inputs.get("hemoglobin", 11.5) < 8:
        hints.append(("warning", "⚠️ Haemoglobin <8 g/dL — severe anaemia."))
    if user_inputs.get("systolic_bp", 0) >= 140 or user_inputs.get("diastolic_bp", 0) >= 90:
        hints.append(("warning", "⚠️ BP suggests hypertension — review for pre-eclampsia."))
    if user_inputs.get("fasting_blood_glucose", 0) >= 5.1:
        hints.append(("warning", "⚠️ Fasting glucose ≥5.1 suggests gestational diabetes."))

    for level, msg in hints:
        getattr(st, level)(msg)

    st.markdown("")  # spacer

    # ─── Run prediction ─────────────────────────────────────────────────────
    block_predict = any(level == "error" for level, _ in hints)
    if st.button(
        "▶  Run Assessment",
        type="primary",
        use_container_width=True,
        disabled=block_predict,
    ):

        # Build the 17-feature DataFrame
        try:
            X = build_feature_dataframe(user_inputs)
        except Exception as e:
            st.error(f"**Could not build features:** {e}")
            st.stop()

        # Run the model
        try:
            model_path = available_models[selected_model_name]
            if selected_model_name == "MLP Neural Network":
                model = load_keras_model(model_path)
                prob  = float(model.predict(X.values, verbose=0)[0, 0])
            else:
                model = load_sklearn_model(model_path)
                prob  = float(model.predict_proba(X)[0, 1])
            label = int(prob >= threshold)
        except Exception as e:
            st.error(f"**Prediction failed:** {e}")
            st.stop()

        # ─── Display results ───
        st.markdown("---")
        st.markdown("## 📋 Assessment result")

        colour, risk_text, emoji = risk_colour(prob)

        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown(gauge_svg(prob), unsafe_allow_html=True)

        with c2:
            verdict = "🔴  HIGH RISK" if label == 1 else "🟢  LOW RISK"
            badge_class = "risk-badge-high" if label == 1 else "risk-badge-low"
            st.markdown(
                f"<div class='result-card'>"
                f"<div class='risk-badge {badge_class}'>{verdict}</div>"
                f"<div class='prob-label'>Risk probability</div>"
                f"<div class='prob-bar-track'>"
                f"<div class='prob-bar-fill' style='width:{max(prob*100, 8):.1f}%;background:{colour};'>"
                f"{prob*100:.1f}%"
                f"</div></div>"
                f"<div class='prob-thresholds'>"
                f"<span>0%</span><span>Low ↔ Moderate (40%)</span>"
                f"<span>Moderate ↔ High (65%)</span><span>100%</span>"
                f"</div>"
                f"<p style='font-size:1rem;margin-top:1.25rem;line-height:1.8'>"
                f"<strong>Risk category:</strong> {emoji} {risk_text}<br>"
                f"<strong>Model:</strong> {selected_model_name}<br>"
                f"<strong>Threshold:</strong> {threshold}"
                f"</p></div>",
                unsafe_allow_html=True,
            )

            if label == 1:
                st.warning(
                    "**Clinical action suggested:** Patient profile shows elevated "
                    "risk indicators. Consider closer monitoring, referral to a "
                    "higher-level facility, and a comprehensive obstetric review.",
                    icon="🚨",
                )
            else:
                st.success(
                    "**Routine follow-up appropriate:** No elevated risk detected "
                    "based on current inputs. Continue standard antenatal care "
                    "and re-assess at each visit.",
                    icon="✅",
                )

        # ─── Feature summary chart ───
        st.markdown("### 📊 Patient profile snapshot")
        fig = feature_bar_chart(user_inputs)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ─── Detailed input summary ───
        with st.expander("📑  Full input summary"):
            rows = []
            inv_ses = {v: k for k, v in SES_OPTIONS.items()}
            display_map = {
                "maternal_age":          ("Maternal age",         f"{user_inputs['maternal_age']} years"),
                "parity":                ("Parity",               f"{user_inputs['parity']} prior"),
                "antenatal_visits":      ("ANC visits",           f"{user_inputs['antenatal_visits']}"),
                "socioeconomic_status":  ("Socioeconomic status", inv_ses[user_inputs["socioeconomic_status"]]),
                "systolic_bp":           ("Systolic BP",          f"{user_inputs['systolic_bp']} mmHg"),
                "diastolic_bp":          ("Diastolic BP",         f"{user_inputs['diastolic_bp']} mmHg"),
                "hemoglobin":            ("Haemoglobin",          f"{user_inputs['hemoglobin']} g/dL"),
                "bmi":                   ("BMI",                  f"{user_inputs['bmi']} kg/m²"),
                "fasting_blood_glucose": ("Fasting glucose",      f"{user_inputs['fasting_blood_glucose']} mmol/L"),
                "weight_gain":           ("Weight gain",          f"{user_inputs['weight_gain']} kg"),
                "facility_type":         ("Care facility",        user_inputs["facility_type"]),
                "history_hypertension":  ("Hypertension hx",      "Yes" if user_inputs["history_hypertension"] else "No"),
                "gestational_diabetes":  ("Gestational diabetes", "Yes" if user_inputs["gestational_diabetes"] else "No"),
                "previous_cesarean":     ("Previous CS",          "Yes" if user_inputs["previous_cesarean"] else "No"),
                "previous_preeclampsia": ("Previous pre-eclampsia","Yes" if user_inputs["previous_preeclampsia"] else "No"),
            }
            for _, (label, val) in display_map.items():
                rows.append({"Field": label, "Value": val})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: MODEL PERFORMANCE
# ══════════════════════════════════════════════════════════════════════════════
elif page.strip().startswith("📊"):
    st.markdown("# 📊 Model Performance")
    st.markdown("How well do the underlying models perform on held-out test data?")

    if os.path.exists(COMPARISON_PATH):
        st.markdown("### Test-set results (n = 750)")
        df = load_comparison()
        metric_cols = [c for c in df.columns
                       if c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        styled = (df[metric_cols].style
                  .highlight_max(axis=0, props="background-color:#FCE4EC;font-weight:bold;color:#212121")
                  .format("{:.4f}"))
        st.dataframe(styled, use_container_width=True)

        st.markdown("### Metric comparison")
        metric_choice = st.selectbox(
            "Choose metric",
            metric_cols,
            index=metric_cols.index("f1_score") if "f1_score" in metric_cols else 0,
        )
        fig, ax = plt.subplots(figsize=(8, 3.5))
        fig.patch.set_facecolor("#FAFAFA")
        ax.set_facecolor("#FAFAFA")
        palette = ["#E91E63", "#AD1457", "#F06292", "#F8BBD0"]
        bars = ax.barh(df.index, df[metric_choice], color=palette[:len(df)], height=0.55)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10, color="#212121")
        ax.set_xlim(0, 1.1)
        ax.set_xlabel(metric_choice.replace("_", " ").title(), color="#212121")
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_color("#616161")
        ax.tick_params(colors="#212121")
        ax.grid(axis="x", alpha=0.15, color="#616161")
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    else:
        st.info("Run `python run_pipeline.py` to generate performance reports.")

    if os.path.exists(CV_RESULTS_PATH):
        st.markdown("---")
        st.markdown("### Cross-validation (5-fold stratified)")
        cv = load_json_file(CV_RESULTS_PATH)
        cv_rows = []
        for model_name, metrics in cv.items():
            row = {"Model": model_name}
            for metric, vals in metrics.items():
                row[metric.replace("_", " ").title()] = f"{vals['mean']:.4f} ± {vals['std']:.4f}"
            cv_rows.append(row)
        st.dataframe(pd.DataFrame(cv_rows).set_index("Model"), use_container_width=True)

    if os.path.exists(FIGURES_DIR):
        st.markdown("---")
        st.markdown("### Report figures")
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


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLINICAL GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page.strip().startswith("📖"):
    st.markdown("# 📖 Clinical Guide")
    st.markdown(
        "A quick reference for interpreting MaternaCare results and the risk "
        "factors considered by the model."
    )

    st.markdown("### Risk categories")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class='info-panel info-panel-low'>
<h4>🟢 Low Risk (&lt; 40%)</h4>
<p>Continue routine antenatal care. Re-assess at every visit.</p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class='info-panel info-panel-moderate'>
<h4>🟡 Moderate Risk (40–65%)</h4>
<p>Increased monitoring. Consider specialist consultation.</p>
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class='info-panel info-panel-high'>
<h4>🔴 High Risk (&gt; 65%)</h4>
<p>Refer to higher-level facility for comprehensive review.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Risk factors used by the model")
    st.markdown("""
The model considers **15 patient characteristics** across four domains:
""")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**👤 Demographics**
- Maternal age (risk ↑ if <18 or ≥35)
- Socioeconomic status
- Type of care facility

**🤱 Obstetric history**
- Parity (number of prior births)
- Number of ANC visits attended
- Gestational weight gain (kg)
""")
    with c2:
        st.markdown("""
**🩺 Vital signs & labs**
- Systolic / diastolic blood pressure
- BMI
- Haemoglobin (anaemia screening)
- Fasting blood glucose

**🏥 Medical history (yes/no)**
- Chronic hypertension
- Gestational diabetes
- Previous caesarean section
- Previous pre-eclampsia
""")

    st.markdown("---")
    st.markdown("### When to refer to a higher-level facility")
    st.warning(
        "Use clinical judgement. The model is one input among many. "
        "Consider referral when **any** of these are present, even if the model "
        "predicts low risk:",
        icon="⚠️",
    )
    st.markdown("""
- BP ≥140/90 mmHg on two occasions
- Severe anaemia (Hb <8 g/dL)
- Fasting glucose ≥5.1 mmol/L
- Bleeding, severe headache, blurred vision, or epigastric pain
- Reduced foetal movements
- Prior obstetric complication
""")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page.strip().startswith("ℹ️"):
    st.markdown("# ℹ️ About MaternaCare")

    st.markdown("""
### The project
**MaternaCare** is the Streamlit deployment of a Final Year Project titled:

> *"Predictive Model for High-Risk Pregnancy Identification Using Machine
> Learning: A Comparative Study with Synthetic Nigerian Maternal Health Data."*

### Why this tool exists
Maternal mortality in Nigeria remains among the highest globally. Early
identification of high-risk pregnancies allows timely intervention — but in
resource-constrained settings, frontline workers often lack decision-support
tools to systematically weigh multiple risk factors. MaternaCare addresses
that gap with a fast, model-driven second opinion.

### Dataset
- **5,000 synthetic records** generated from published Nigerian and West
  African epidemiological parameters (15 peer-reviewed studies)
- **15 patient characteristics** across demographics, obstetric history,
  vital signs, laboratory values, and medical history
- **Target prevalence:** ~29.6% high-risk pregnancies

### Models compared
| Model | Strength |
|---|---|
| Logistic Regression | Most interpretable, best AUC-ROC (0.93) |
| Decision Tree | Transparent rule-based reasoning |
| Random Forest | Robust ensemble approach |
| MLP Neural Network | Best F1 score (0.78) |

### Ethical use
- 🩹 This tool was trained on **synthetic data** and **must be externally
  validated** on real patient data before clinical deployment
- 🩹 It is a **decision-support aid**, not an autonomous decision-maker
- 🩹 The model has a false-negative rate (~14% for MLP) — clinical vigilance
  remains essential
- 🩹 Feature distributions are calibrated for Nigeria; transfer to other
  populations requires recalibration

### Project repository
[github.com/mbanwusifrancisca/high-risk-pregnancy-prediction-deploy](https://github.com/mbanwusifrancisca/high-risk-pregnancy-prediction-deploy)
""")

    st.markdown("---")
    st.caption(
        "Made with 💗 as a Final Year academic project · "
        "For research and clinical decision-support use only."
    )
