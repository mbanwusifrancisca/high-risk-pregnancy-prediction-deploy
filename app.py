"""
MaternaCare — Clinical Decision-Support for High-Risk Pregnancy Screening
=========================================================================
A final-year Biomedical Engineering project investigating machine learning
for early identification of high-risk pregnancies in Nigerian healthcare.

Deployment    : Streamlit Community Cloud
Inference     : Logistic Regression (scikit-learn) — best AUC-ROC (0.933)
Pipeline      : 10 scaled features + 4 binary features + 3 one-hot facility = 17 inputs
Risk tiers    : <40% Low · 40–65% Moderate · >65% High (single source of truth)
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="MaternaCare — Clinical Decision-Support",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# ACCESSIBLE THEME  (locked to light mode via .streamlit/config.toml)
# Palette: #E91E63 accent · #212121 text · #FFFFFF bg
# WCAG AA contrast targets met on all text.
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
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
.stApp small, .stApp .stCaption { color: #616161; }

/* ─── Headings ─── */
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

/* ─── Sidebar ─── */
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

/* ─── Primary button ─── */
.stButton > button[kind="primary"] {
    background: #E91E63;
    color: #FFFFFF !important;
    border: none;
    border-radius: 8px;
    padding: 0.875rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    min-height: 48px;
    width: 100%;
    box-shadow: 0 2px 6px rgba(233, 30, 99, 0.25);
    transition: background 0.2s;
}
.stButton > button[kind="primary"]:hover { background: #C2185B; }
.stButton > button[kind="primary"]:disabled {
    background: #BDBDBD;
    color: #FFFFFF !important;
    box-shadow: none;
}

/* ─── Expanders / cards ─── */
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
[data-testid="stExpander"] summary:hover { background: #F5F5F5; }

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

/* ─── Inputs ─── */
.stNumberInput input, .stTextInput input, .stSelectbox > div > div {
    background: #FFFFFF !important;
    color: #212121 !important;
    border: 1.5px solid #BDBDBD !important;
    border-radius: 8px !important;
    min-height: 44px;
    font-size: 1rem !important;
}
.stNumberInput input:focus, .stTextInput input:focus {
    border-color: #E91E63 !important;
    box-shadow: 0 0 0 3px rgba(233, 30, 99, 0.15) !important;
    outline: none !important;
}
.stNumberInput label, .stTextInput label, .stSelectbox label {
    color: #212121 !important;
    font-weight: 500;
    font-size: 0.95rem;
}

/* ─── Checkboxes ─── */
.stCheckbox label {
    color: #212121 !important;
    font-weight: 500;
    font-size: 0.95rem;
    line-height: 1.5;
}

/* ─── Alerts ─── */
.stAlert {
    border-radius: 10px;
    border-left-width: 5px;
    padding: 1rem 1.25rem !important;
}
.stAlert p { color: #212121 !important; font-weight: 500; }

/* ─── DataFrame ─── */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid #E0E0E0;
}

/* ─── Hero ─── */
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

/* ─── Feature cards (Home, How It Works, Intended Use) ─── */
.feature-card {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-radius: 12px;
    padding: 1.5rem 1.25rem;
    height: 100%;
    margin-bottom: 1rem;
}
.feature-card .feature-icon {
    font-size: 2rem;
    margin-bottom: 0.5rem;
    display: block;
}
.feature-card h4 {
    color: #212121 !important;
    margin: 0 0 0.5rem 0 !important;
}
.feature-card p { color: #424242 !important; margin: 0; font-size: 0.95rem; }
.feature-card ul {
    color: #424242 !important;
    margin: 0.5rem 0 0 0;
    padding-left: 1.2rem;
}
.feature-card ul li { margin-bottom: 0.4rem; font-size: 0.95rem; }

/* ─── Risk-tinted result cards (single source for low / moderate / high) ─── */
.result-panel {
    border-radius: 16px;
    padding: 1.75rem 1.5rem;
    margin: 1rem 0;
    border-width: 2px;
    border-style: solid;
}
.result-panel-low {
    background: #E8F5E9;
    border-color: #2E7D32;
}
.result-panel-moderate {
    background: #FFF3E0;
    border-color: #ED6C02;
}
.result-panel-high {
    background: #FFEBEE;
    border-color: #D32F2F;
}
.result-panel p, .result-panel li, .result-panel strong {
    color: #212121 !important;
}

/* ─── Risk badges ─── */
.risk-badge {
    display: inline-block;
    padding: 0.625rem 1.25rem;
    border-radius: 24px;
    font-weight: 700;
    font-size: 1.05rem;
    letter-spacing: 0.04em;
    margin-bottom: 1rem;
    background: #FFFFFF;
}
.risk-badge-low      { color: #1B5E20; border: 2px solid #2E7D32; }
.risk-badge-moderate { color: #B85A00; border: 2px solid #ED6C02; }
.risk-badge-high     { color: #B71C1C; border: 2px solid #D32F2F; }

/* ─── Probability bar ─── */
.prob-label {
    color: #424242;
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.375rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
.prob-bar-track {
    background: #FFFFFF;
    border: 1px solid #BDBDBD;
    border-radius: 8px;
    height: 28px;
    overflow: hidden;
    position: relative;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 0.75rem;
    color: #FFFFFF;
    font-weight: 700;
    font-size: 0.9rem;
    min-width: 3rem;
}
.prob-bar-fill-low      { background: #2E7D32; }
.prob-bar-fill-moderate { background: #ED6C02; }
.prob-bar-fill-high     { background: #D32F2F; }
.prob-thresholds {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: #616161;
    margin-top: 0.25rem;
}

/* ─── Recommendation block inside result ─── */
.recommendation-block {
    background: #FFFFFF;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    margin-top: 1.25rem;
    border: 1px solid #E0E0E0;
}
.recommendation-block h4 { margin: 0 0 0.5rem 0 !important; font-size: 1rem !important; }
.recommendation-block p { margin: 0; line-height: 1.7; }

/* ─── Info panels (Clinical Guide) ─── */
.info-panel {
    border-radius: 12px;
    padding: 1.25rem;
    border-left: 5px solid;
    height: 100%;
    margin-bottom: 1rem;
}
.info-panel h4 { margin: 0 0 0.5rem 0 !important; font-weight: 700 !important; }
.info-panel p { margin: 0 !important; font-size: 0.95rem; }
.info-panel-low      { background: #E8F5E9; border-left-color: #2E7D32; }
.info-panel-low h4, .info-panel-low p           { color: #1B5E20 !important; }
.info-panel-moderate { background: #FFF3E0; border-left-color: #ED6C02; }
.info-panel-moderate h4, .info-panel-moderate p { color: #B85A00 !important; }
.info-panel-high     { background: #FFEBEE; border-left-color: #D32F2F; }
.info-panel-high h4, .info-panel-high p         { color: #B71C1C !important; }

/* ─── Best practices checklist ─── */
.practice-item {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-left: 4px solid #E91E63;
    border-radius: 8px;
    padding: 0.875rem 1.25rem;
    margin-bottom: 0.625rem;
    color: #212121;
    font-size: 0.95rem;
}

/* ─── Developer profile card ─── */
.dev-card {
    background: #FAFAFA;
    border: 1px solid #E0E0E0;
    border-top: 4px solid #E91E63;
    border-radius: 12px;
    padding: 1.75rem 1.5rem;
    max-width: 480px;
    margin: 0 auto;
    text-align: center;
}
.dev-avatar {
    width: 72px;
    height: 72px;
    border-radius: 50%;
    background: #E91E63;
    color: #FFFFFF;
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 0 auto 1rem;
    font-size: 1.625rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.dev-card h3 { margin: 0 0 0.25rem 0 !important; }
.dev-card .dev-role { color: #616161 !important; font-size: 0.95rem; margin: 0 0 0.75rem 0; }
.dev-card .dev-email a { color: #E91E63; text-decoration: none; font-weight: 600; }
.dev-card .dev-email a:hover { text-decoration: underline; }

/* ─── Links ─── */
a {
    color: #E91E63 !important;
    text-decoration: underline;
    font-weight: 500;
}
a:hover { color: #C2185B !important; }

/* ─── Section dividers ─── */
hr {
    border: none;
    border-top: 1px solid #E0E0E0;
    margin: 2rem 0 !important;
}

/* ─── Mobile responsiveness ─── */
@media (max-width: 768px) {
    .hero { padding: 1.5rem 1rem; }
    .stApp h1 { font-size: 1.625rem !important; }
    .stApp h2 { font-size: 1.25rem !important; }
    .stApp h3 { font-size: 1.1rem !important; }
    [data-testid="stExpander"] summary {
        padding: 0.875rem 1rem !important;
        font-size: 1rem;
    }
    [data-testid="column"] { width: 100% !important; flex: 1 1 100% !important; }
    [data-testid="stMetric"] { padding: 1rem; margin-bottom: 0.75rem; }
    [data-testid="stMetricValue"] { font-size: 1.25rem !important; }
    .result-panel { padding: 1.25rem 1rem; }
    .risk-badge { font-size: 0.95rem; padding: 0.5rem 1rem; }
    .feature-card { padding: 1.25rem 1rem; }
    .dev-card { padding: 1.5rem 1.25rem; }
}
@media (max-width: 480px) {
    .stApp h1 { font-size: 1.5rem !important; }
    .hero p { font-size: 0.95rem; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# FILE PATHS  (relative to repo root)
# ══════════════════════════════════════════════════════════════════════════════
MODELS_DIR       = "models"
REPORTS_DIR      = "reports"
FIGURES_DIR      = os.path.join(REPORTS_DIR, "figures")
SCALER_PATH      = os.path.join(MODELS_DIR, "scaler.joblib")
MODEL_PATH       = os.path.join(MODELS_DIR, "logistic_regression.joblib")
COMPARISON_PATH  = os.path.join(REPORTS_DIR, "model_comparison.csv")
CV_RESULTS_PATH  = os.path.join(REPORTS_DIR, "cv_results.json")
THRESHOLDS_PATH  = os.path.join(REPORTS_DIR, "optimal_thresholds.json")


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE PIPELINE  (verified against scaler.feature_names_in_ and model.feature_names_in_)
# ══════════════════════════════════════════════════════════════════════════════
SCALED_FEATURES = [
    "maternal_age", "parity", "antenatal_visits", "socioeconomic_status",
    "systolic_bp", "diastolic_bp", "hemoglobin", "bmi",
    "fasting_blood_glucose", "weight_gain",
]
BINARY_FEATURES = [
    "history_hypertension", "gestational_diabetes",
    "previous_cesarean", "previous_preeclampsia",
]
FACILITY_OHE_COLUMNS = [
    "facility_primary_health_centre",
    "facility_private_clinic",
    "facility_tertiary_hospital",
]
# Facility field has been removed from the UI per requirements.
# We default to Primary Health Centre — the dominant deployment context.
DEFAULT_FACILITY_COLUMN = "facility_primary_health_centre"

FINAL_FEATURE_ORDER = SCALED_FEATURES + BINARY_FEATURES + FACILITY_OHE_COLUMNS
SES_OPTIONS = {"Low": 0, "Middle": 1, "High": 2}

# Risk bands (single source of truth — drive every displayed output)
LOW_BAND_UPPER       = 0.40   # below this → Low Risk
MODERATE_BAND_UPPER  = 0.65   # below this → Moderate, at/above → High


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


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """
    Load the Logistic Regression model.
    LR was selected for deployment because it achieved the highest AUC-ROC
    (0.933) of all four models compared, with strong F1 (0.759) and
    the additional benefit of interpretability for clinical settings.
    """
    import joblib
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model file not found at '{MODEL_PATH}'.")
        st.stop()
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH, index_col=0)


@st.cache_data(show_spinner=False)
def load_json_file(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def get_validated_threshold() -> float:
    """Load the Logistic Regression F1-optimal threshold from training (used internally)."""
    if os.path.exists(THRESHOLDS_PATH):
        try:
            data = load_json_file(THRESHOLDS_PATH)
            return float(data.get("Logistic Regression", 0.5))
        except Exception:
            return 0.5
    return 0.5


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (mirrors training pipeline exactly)
# ══════════════════════════════════════════════════════════════════════════════
def build_feature_dataframe(user_inputs: dict) -> pd.DataFrame:
    """
    Build the 17-column feature DataFrame matching the model's training-time
    column order. Facility type defaults to Primary Health Centre (the dominant
    deployment context); this field is no longer collected from the UI.
    """
    scaler = load_scaler()

    # 1. Scale the 10 continuous + ordinal features
    scaled_row = pd.DataFrame(
        [[user_inputs[f] for f in SCALED_FEATURES]],
        columns=SCALED_FEATURES,
    )
    if scaler.n_features_in_ != len(SCALED_FEATURES):
        raise ValueError(
            f"Scaler expects {scaler.n_features_in_} features but received "
            f"{len(SCALED_FEATURES)}. Re-train or update the pipeline."
        )
    scaled_df = pd.DataFrame(
        scaler.transform(scaled_row),
        columns=SCALED_FEATURES,
    )

    # 2. Binary features (unscaled)
    binary_df = pd.DataFrame(
        [[int(user_inputs[f]) for f in BINARY_FEATURES]],
        columns=BINARY_FEATURES,
    )

    # 3. Facility one-hot — defaulted to Primary Health Centre
    facility_dict = {col: 0 for col in FACILITY_OHE_COLUMNS}
    facility_dict[DEFAULT_FACILITY_COLUMN] = 1
    facility_df = pd.DataFrame([facility_dict])[FACILITY_OHE_COLUMNS]

    # 4. Final 17-column row
    final_df = pd.concat([scaled_df, binary_df, facility_df], axis=1)
    return final_df[FINAL_FEATURE_ORDER]


def predict(user_inputs: dict) -> float:
    """
    Single inference entry point.
    Returns one probability (float) — every downstream display derives from it.
    """
    X = build_feature_dataframe(user_inputs)
    model = load_model()
    # predict_proba returns shape (n_samples, n_classes); column 1 = positive class
    prob = float(model.predict_proba(X)[0, 1])
    # Clamp to [0,1] for numerical safety
    return max(0.0, min(1.0, prob))


# ══════════════════════════════════════════════════════════════════════════════
# ASSESSMENT  (SINGLE source of truth — all displayed outputs derive here)
# ══════════════════════════════════════════════════════════════════════════════
def compute_assessment(prob: float) -> dict:
    """
    Map a single probability to a single tier, badge, colour, and recommendation.
    All downstream UI elements use this dictionary — guaranteeing consistency
    between badge, probability, recommendation, and clinical interpretation.
    """
    if prob < LOW_BAND_UPPER:
        return {
            "tier":          "low",
            "label":         "LOW RISK",
            "icon":          "🟢",
            "badge_class":   "risk-badge-low",
            "panel_class":   "result-panel-low",
            "bar_class":     "prob-bar-fill-low",
            "colour":        "#2E7D32",
            "interpretation":
                "Based on the inputs provided, this profile does not currently "
                "indicate elevated maternal risk.",
            "recommendation":
                "Continue routine antenatal care according to national maternal "
                "health guidelines. Re-assess risk at every subsequent visit and "
                "watch for any new clinical signs.",
        }
    elif prob < MODERATE_BAND_UPPER:
        return {
            "tier":          "moderate",
            "label":         "MODERATE RISK",
            "icon":          "🟡",
            "badge_class":   "risk-badge-moderate",
            "panel_class":   "result-panel-moderate",
            "bar_class":     "prob-bar-fill-moderate",
            "colour":        "#ED6C02",
            "interpretation":
                "This profile shows some risk indicators that warrant closer "
                "attention than routine care alone.",
            "recommendation":
                "Schedule more frequent antenatal visits, perform targeted "
                "follow-up investigations where indicated, and consider "
                "consultation with a specialist if any single risk factor is "
                "severe or worsening.",
        }
    else:
        return {
            "tier":          "high",
            "label":         "HIGH RISK",
            "icon":          "🔴",
            "badge_class":   "risk-badge-high",
            "panel_class":   "result-panel-high",
            "bar_class":     "prob-bar-fill-high",
            "colour":        "#D32F2F",
            "interpretation":
                "This profile is associated with substantially elevated maternal "
                "risk and should be acted upon promptly.",
            "recommendation":
                "Refer to a higher-level facility for comprehensive obstetric "
                "review. Initiate close monitoring in the interim and follow "
                "national maternal referral protocols.",
        }


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
            "📖  Clinical Guide",
            "ℹ️  About",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("##### 💡 Risk bands")
    st.markdown(
        "- 🟢 **Low** — below 40%\n"
        "- 🟡 **Moderate** — 40 to 65%\n"
        "- 🔴 **High** — above 65%"
    )
    st.markdown("---")
    st.caption(
        "Predictive ML tool trained on synthetic Nigerian maternal health "
        "data. For research and clinical decision-support use only."
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: HOME
# ══════════════════════════════════════════════════════════════════════════════
if page.strip().startswith("🏠"):
    st.markdown("""
<div class="hero">
  <h1>🌸 MaternaCare</h1>
  <p><strong>High-Risk Pregnancy Screening</strong> · A clinical decision-support
  tool for midwives, nurses, and obstetricians</p>
</div>
""", unsafe_allow_html=True)

    # ─── Welcome ───────────────────────────────────────────────────────────
    st.markdown("### Welcome 👋")
    st.markdown("""
**MaternaCare** helps frontline maternal health workers in Nigerian primary
health centres, private clinics, and tertiary hospitals quickly identify
pregnant women whose clinical profile suggests **elevated maternal risk** —
supporting timely referral, closer monitoring, and informed counselling.

> ⚠️ **This tool supports clinical judgement; it does not replace it.**
> Always combine model output with full clinical history, examination,
> and your professional expertise.
""")

    # ─── How it works ─────────────────────────────────────────────────────
    st.markdown("### How it works")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class='feature-card'>
<span class='feature-icon'>📝</span>
<h4>1. Enter patient data</h4>
<p>Fill in the patient's vitals, obstetric history, and demographics. Takes about 60 seconds.</p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class='feature-card'>
<span class='feature-icon'>📊</span>
<h4>2. Get a risk score</h4>
<p>The model returns a probability of high-risk pregnancy along with a clear Low, Moderate, or High classification.</p>
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class='feature-card'>
<span class='feature-icon'>🩺</span>
<h4>3. Make informed decisions</h4>
<p>Use the score as additional input for referral, monitoring frequency, or specialist consultation.</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # ─── Intended use ─────────────────────────────────────────────────────
    st.markdown("### Intended use")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
<div class='feature-card'>
<span class='feature-icon'>🎯</span>
<h4>Primary use</h4>
<p>Decision-support tool for antenatal risk screening at Primary Health Centres (PHCs) and General Hospitals in Nigeria. Designed to identify pregnancies that may require closer monitoring or referral to higher-level facilities.</p>
</div>
""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
<div class='feature-card'>
<span class='feature-icon'>👥</span>
<h4>Intended users</h4>
<ul>
<li>Midwives and nurses conducting routine antenatal assessments at PHCs.</li>
<li>Obstetricians working in secondary and tertiary hospitals for patient triage.</li>
<li>Researchers studying machine learning applications for maternal health in low-resource settings.</li>
</ul>
</div>
""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
<div class='feature-card'>
<span class='feature-icon'>🚫</span>
<h4>Out-of-scope uses</h4>
<ul>
<li>Not intended for autonomous clinical decision-making.</li>
<li>The model is a screening aid, not a diagnostic tool.</li>
<li>The model is designed around Nigerian maternal health characteristics and should not be assumed to generalize to other populations without validation.</li>
</ul>
</div>
""", unsafe_allow_html=True)

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

    user_inputs = {}

    # ═══ Demographics ═══════════════════════════════════════════════════════
    with st.expander("👤  Demographics & socioeconomic context", expanded=True):
        c1, c2 = st.columns(2)
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

    # ═══ Obstetric history ══════════════════════════════════════════════════
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

    # ═══ Vital signs & laboratory ═══════════════════════════════════════════
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

    # ═══ Medical history ════════════════════════════════════════════════════
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

    # ─── Real-time clinical hints (not gating) ──────────────────────────────
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

    st.markdown("")

    block_predict = any(level == "error" for level, _ in hints)

    if st.button(
        "▶  Run Assessment",
        type="primary",
        use_container_width=True,
        disabled=block_predict,
    ):
        # ─── Single prediction call ─────────────────────────────────────────
        try:
            prob = predict(user_inputs)
        except Exception as e:
            st.error(f"**Prediction failed:** {e}")
            st.stop()

        # ─── Derive every output from the same probability ──────────────────
        a = compute_assessment(prob)
        prob_pct = prob * 100

        st.markdown("---")
        st.markdown("## 📋 Assessment result")

        # ─── Risk-tinted result panel ───────────────────────────────────────
        st.markdown(
            f"""
<div class='result-panel {a['panel_class']}'>
  <div class='risk-badge {a['badge_class']}'>{a['icon']}  {a['label']}</div>

  <div class='prob-label'>Risk probability</div>
  <div class='prob-bar-track'>
    <div class='prob-bar-fill {a['bar_class']}' style='width:{max(prob_pct, 8):.1f}%'>
      {prob_pct:.1f}%
    </div>
  </div>
  <div class='prob-thresholds'>
    <span>0%</span>
    <span>40% — Moderate</span>
    <span>65% — High</span>
    <span>100%</span>
  </div>

  <div class='recommendation-block'>
    <h4>🩺 Clinical interpretation</h4>
    <p>{a['interpretation']}</p>
  </div>

  <div class='recommendation-block'>
    <h4>✅ Recommended action</h4>
    <p>{a['recommendation']}</p>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        # ─── Patient input summary (collapsed by default) ───────────────────
        with st.expander("📑  Full patient input summary"):
            inv_ses = {v: k for k, v in SES_OPTIONS.items()}
            display_rows = [
                ("Maternal age",            f"{user_inputs['maternal_age']} years"),
                ("Parity",                  f"{user_inputs['parity']} prior"),
                ("ANC visits",              f"{user_inputs['antenatal_visits']}"),
                ("Socioeconomic status",    inv_ses[user_inputs["socioeconomic_status"]]),
                ("Systolic BP",             f"{user_inputs['systolic_bp']} mmHg"),
                ("Diastolic BP",            f"{user_inputs['diastolic_bp']} mmHg"),
                ("Haemoglobin",             f"{user_inputs['hemoglobin']} g/dL"),
                ("BMI",                     f"{user_inputs['bmi']} kg/m²"),
                ("Fasting glucose",         f"{user_inputs['fasting_blood_glucose']} mmol/L"),
                ("Weight gain",             f"{user_inputs['weight_gain']} kg"),
                ("History of hypertension", "Yes" if user_inputs["history_hypertension"] else "No"),
                ("Gestational diabetes",    "Yes" if user_inputs["gestational_diabetes"] else "No"),
                ("Previous caesarean",      "Yes" if user_inputs["previous_cesarean"] else "No"),
                ("Previous pre-eclampsia",  "Yes" if user_inputs["previous_preeclampsia"] else "No"),
            ]
            st.dataframe(
                pd.DataFrame(display_rows, columns=["Field", "Value"]),
                use_container_width=True,
                hide_index=True,
            )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: CLINICAL GUIDE
# ══════════════════════════════════════════════════════════════════════════════
elif page.strip().startswith("📖"):
    st.markdown("# 📖 Clinical Guide")
    st.markdown(
        "A quick reference for interpreting MaternaCare results and using the "
        "tool safely in routine antenatal practice."
    )

    # ─── Risk categories ───────────────────────────────────────────────────
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

    # ─── When to refer ─────────────────────────────────────────────────────
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

    st.markdown("---")

    # ─── Best practices ────────────────────────────────────────────────────
    st.markdown("### Best practices when using MaternaCare")
    practices = [
        "✅  Verify patient information before assessment.",
        "✅  Use MaternaCare alongside routine antenatal evaluation.",
        "✅  Repeat assessment if the patient's condition changes.",
        "✅  Clinical judgement always takes priority over model predictions.",
        "✅  High-risk predictions should prompt timely referral according to national maternal health guidelines.",
    ]
    for p in practices:
        st.markdown(f"<div class='practice-item'>{p}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page.strip().startswith("ℹ️"):
    st.markdown("# ℹ️ About MaternaCare")

    st.markdown("""
MaternaCare is a clinical decision-support application developed as part of a
final-year Biomedical Engineering project. The project investigates the
application of machine learning techniques for early identification of
high-risk pregnancies using routinely collected maternal health indicators
within the Nigerian healthcare context.
""")

    # ─── Why this tool exists ──────────────────────────────────────────────
    st.markdown("### Why this tool exists")
    st.markdown("""
Maternal mortality in Nigeria remains among the highest globally. Early
identification of high-risk pregnancies allows timely intervention — but in
resource-constrained settings, frontline workers often lack decision-support
tools to systematically weigh multiple risk factors. MaternaCare addresses
that gap with a fast, model-driven second opinion.
""")

    st.markdown("---")

    # ─── Risk factors used by the model (moved from Clinical Guide) ────────
    st.markdown("### Risk factors used by the model")
    st.markdown(
        "The model considers **14 patient characteristics** across four domains:"
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**👤 Demographics**
- Maternal age (risk ↑ if <18 or ≥35)
- Socioeconomic status

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

    # ─── Model performance at a glance (moved from Home) ───────────────────
    st.markdown("### Model performance at a glance")
    if os.path.exists(COMPARISON_PATH):
        df = load_comparison()
        metric_cols = [c for c in df.columns
                       if c in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
        styled = (df[metric_cols].style
                  .highlight_max(axis=0, props="background-color:#FCE4EC;font-weight:bold;color:#212121")
                  .format("{:.4f}"))
        st.dataframe(styled, use_container_width=True)
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("Training records",   "5,000")
        with c2: st.metric("LR F1 score",        "0.76")
        with c3: st.metric("LR AUC-ROC",         "0.93")
        with c4: st.metric("Patient features",   "14")

    st.markdown("---")

    # ─── Models compared ───────────────────────────────────────────────────
    st.markdown("### Models compared")
    st.markdown(
        "This project compared four machine learning models to determine the "
        "best-performing model. The deployed application performs inference "
        "using only the **Logistic Regression** model, which achieved the "
        "highest AUC-ROC (0.933) and offers strong interpretability for "
        "clinical decision-support. The other models remain documented here "
        "for transparency:"
    )
    models_df = pd.DataFrame({
        "Model":    ["Logistic Regression", "MLP Neural Network", "Random Forest", "Decision Tree"],
        "Type":     ["Linear, interpretable baseline", "Feedforward deep learning",
                     "Ensemble (bagged trees)", "Rule-based, transparent"],
        "F1 score": [0.759, 0.777, 0.728, 0.619],
        "AUC-ROC":  [0.933, 0.929, 0.904, 0.819],
        "Deployed": ["✅ Yes", "—", "—", "—"],
    }).set_index("Model")
    st.dataframe(models_df, use_container_width=True)

    st.markdown("---")

    # ─── Ethical use ───────────────────────────────────────────────────────
    st.markdown("### Ethical use")
    st.markdown("""
- 🩹 This tool was trained on **synthetic data** and **must be externally
  validated** on real patient data before clinical deployment
- 🩹 It is a **decision-support aid**, not an autonomous decision-maker
- 🩹 The model has a false-negative rate (~14%) — clinical vigilance remains essential
- 🩹 Feature distributions are calibrated for Nigeria; transfer to other
  populations requires recalibration
""")

    st.markdown("---")

    # ─── Project repository ────────────────────────────────────────────────
    st.markdown("### Project repository")
    st.markdown(
        "Source code, training pipeline, and documentation are available at: "
        "[github.com/mbanwusifrancisca/high-risk-pregnancy-prediction-deploy]"
        "(https://github.com/mbanwusifrancisca/high-risk-pregnancy-prediction-deploy)"
    )

    st.markdown("---")

    # ─── Developer ─────────────────────────────────────────────────────────
    st.markdown("### Developer")
    st.markdown("""
<div class='dev-card'>
  <div class='dev-avatar'>MF</div>
  <h3>Mbanwusi Francisca</h3>
  <p class='dev-role'>Final-Year Biomedical Engineering Student</p>
  <p class='dev-email'>📧 <a href='mailto:mbanwusifrancisca@gmail.com'>mbanwusifrancisca@gmail.com</a></p>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "Made with 💗 as a Final Year academic project · "
        "For research and clinical decision-support use only."
    )
