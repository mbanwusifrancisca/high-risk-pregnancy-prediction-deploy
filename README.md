# Predictive Model for High-Risk Pregnancy Identification Using Machine Learning

> A Comparative Study with Synthetic Nigerian Maternal Health Data — Final Year Project

---

## Overview

This project develops and comparatively evaluates four machine learning models for binary high-risk pregnancy classification using synthetic maternal health data parameterised from 15 peer-reviewed Nigerian and West African epidemiological studies.

### Key Results

| Model | F1 Score | AUC-ROC | Precision | Recall |
|-------|----------|---------|-----------|--------|
| **MLP** | **0.777** | 0.929 | **0.712** | **0.856** |
| Logistic Regression | 0.759 | **0.933** | 0.690 | 0.842 |
| Random Forest | 0.728 | 0.904 | 0.698 | 0.761 |
| Decision Tree | 0.619 | 0.819 | 0.484 | 0.860 |

The MLP achieves the best F1 score while Logistic Regression achieves the best AUC — demonstrating that simpler, interpretable models can perform comparably to neural networks for this task.

---

## Project Structure

```
high-risk-pregnancy-prediction/
├── config/                     # Feature definitions, risk criteria (YAML)
├── data/
│   ├── synthetic/              # Generated synthetic dataset (5,000 records)
│   └── processed/              # Train/val/test splits
├── src/
│   ├── data/                   # Data generation & preprocessing
│   │   ├── generate_synthetic_data.py
│   │   └── preprocess.py
│   ├── models/                 # Model implementations
│   │   ├── logistic_regression.py
│   │   ├── decision_tree.py
│   │   ├── random_forest.py
│   │   └── lstm_model.py       # MLP (feedforward neural network)
│   └── evaluation/             # Metrics, threshold optimisation, plotting
│       └── metrics.py
├── reports/
│   ├── figures/                # ROC curves, confusion matrices, feature importance
│   ├── model_comparison.csv    # Test set results
│   ├── cv_results.json         # Cross-validation results
│   └── all_metrics.json        # Detailed per-model metrics
├── paper/                      # Full academic paper (Markdown + DOCX)
│   ├── CHAPTER_1_INTRODUCTION.md
│   ├── CHAPTER_2_LITERATURE_REVIEW.md
│   ├── CHAPTER_3_METHODOLOGY.md
│   ├── CHAPTER_4_RESULTS_AND_DISCUSSION.md
│   ├── CHAPTER_5_CONCLUSION.md
│   ├── REFERENCES.md
│   └── High_Risk_Pregnancy_Prediction_Paper.docx
├── models/                     # Saved trained models (.joblib, .keras)
├── notebooks/                  # Jupyter notebooks (exploration)
├── run_pipeline.py             # End-to-end pipeline execution
├── MODEL_CARD.md               # Technical model & dataset card
├── PROJECT_PLAN.md             # Project timeline
├── requirements.txt            # Python dependencies
└── README.md
```

## Features

15 clinically relevant maternal health features:

| Category | Features |
|----------|----------|
| **Clinical** | Systolic BP, Diastolic BP, Haemoglobin, BMI, Fasting blood glucose, Weight gain |
| **Medical History** | History of hypertension, Gestational diabetes, Previous caesarean, Previous pre-eclampsia |
| **Demographic** | Maternal age, Parity, Antenatal visits, Socioeconomic status, Facility type |

## Models

| Model | Type | Tuning |
|-------|------|--------|
| Logistic Regression | Linear, interpretable baseline | GridSearchCV (C, penalty) |
| Decision Tree | Rule-based, transparent | GridSearchCV (depth, pruning) |
| Random Forest | Ensemble (bagged trees) | RandomizedSearchCV (50 iter) |
| MLP Neural Network | Feedforward deep learning | EarlyStopping, ReduceLR |

### Pipeline Features
- **Probabilistic risk labelling** — logistic model + Bernoulli sampling (no target circularity)
- **Selective feature scaling** — binary features excluded from StandardScaler
- **SMOTE oversampling** for sklearn models; **class weighting** for MLP
- **F1-optimal threshold tuning** on validation set
- **Stratified 5-fold cross-validation** with 95% confidence intervals

## Quick Start

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (data generation → training → evaluation)
python run_pipeline.py
```

All results, figures, and model files are generated automatically.

## Paper

The complete academic paper is available in `paper/`:
- **Markdown chapters**: individual `.md` files for each chapter
- **DOCX**: `paper/High_Risk_Pregnancy_Prediction_Paper.docx` — merged document with embedded figures

## Model Card

See [`MODEL_CARD.md`](MODEL_CARD.md) for detailed technical documentation including dataset card, model architectures, performance metrics, ethical considerations, and reproducibility information.

---

## License

Academic use only — Final Year Project.
