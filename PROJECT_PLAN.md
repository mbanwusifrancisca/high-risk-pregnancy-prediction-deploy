# Project Plan: Predictive Model for High-Risk Pregnancy Identification

## Timeline Overview

| Phase | Description | Duration |
| ----- | ----------- | -------- |
| Phase 1 | Literature Review & Variable Selection | Week 1–2 |
| Phase 2 | Synthetic Data Generation | Week 3–4 |
| Phase 3 | Exploratory Data Analysis & Preprocessing | Week 5–6 |
| Phase 4 | Model Development & Training | Week 7–9 |
| Phase 5 | Evaluation & Comparison | Week 10–11 |
| Phase 6 | Documentation & Report Writing | Week 12–14 |

---

## Phase 1: Literature Review & Variable Selection (Week 1–2)

### Tasks

- Review peer-reviewed literature on maternal health risk factors in Nigeria and Sub-Saharan Africa.
- Identify clinically relevant variables and their normal/abnormal reference ranges.
- Document statistical distributions (mean, std, prevalence rates) for each variable from published sources.
- Define the target variable: binary classification (high-risk vs. low-risk).
- Establish clinical rules for labeling pregnancies as high-risk based on medical guidelines.

### Deliverables

- `config/feature_definitions.yaml` — Variable names, types, ranges, and source references.
- `config/risk_criteria.yaml` — Rules for classifying high-risk pregnancies.
- Literature review summary (for project report).

---

## Phase 2: Synthetic Data Generation (Week 3–4)

### Tasks

- Implement synthetic data generator using statistical distributions from Phase 1.
- Simulate realistic correlations between variables (e.g., age ↔ hypertension, BMI ↔ gestational diabetes).
- Generate N = 5,000–10,000 synthetic patient records.
- Apply risk-labeling logic to assign binary target labels.
- Validate that generated data distributions match published reference ranges.

### Deliverables

- `src/data/generate_synthetic_data.py` — Data generation module.
- `notebooks/01_data_generation.ipynb` — Interactive notebook for generation and validation.
- `data/synthetic/maternal_health_synthetic.csv` — Generated dataset.

---

## Phase 3: Exploratory Data Analysis & Preprocessing (Week 5–6)

### Tasks

- Perform univariate analysis (distributions, summary statistics) for all features.
- Perform bivariate analysis (correlations, group comparisons by risk label).
- Visualize distributions, box plots, correlation heatmaps, class balance.
- Handle any missing values, outliers, or inconsistencies in the synthetic data.
- Feature engineering: encode categorical variables, scale numerical features.
- Split data into training (70%), validation (15%), and test (15%) sets.
- Address class imbalance if present (SMOTE or class weighting).

### Deliverables

- `notebooks/02_eda.ipynb` — EDA notebook with visualizations.
- `notebooks/03_preprocessing.ipynb` — Preprocessing pipeline.
- `src/data/preprocess.py` — Reusable preprocessing module.
- `data/processed/` — Train/validation/test splits saved as CSV.
- `reports/figures/` — Saved EDA plots.

---

## Phase 4: Model Development & Training (Week 7–9)

### Tasks

- Implement four classification models:
  1. **Logistic Regression** — Baseline interpretable model.
  2. **Decision Tree Classifier** — Rule-based interpretable model.
  3. **Random Forest Classifier** — Ensemble model for improved accuracy.
  4. **LSTM Neural Network** — Deep learning approach (sequence-based or feature-based).
- Perform hyperparameter tuning using cross-validation (GridSearchCV / RandomizedSearchCV).
- Train each model on the training set, validate on the validation set.
- Save trained model artifacts.

### Deliverables

- `src/models/logistic_regression.py`
- `src/models/decision_tree.py`
- `src/models/random_forest.py`
- `src/models/lstm_model.py`
- `notebooks/04_model_training.ipynb` — Training notebook.
- `config/hyperparameters.yaml` — Hyperparameter configurations.
- Saved model files in `models/` directory.

### Notes on LSTM

- Since the data is tabular (not sequential time-series per patient), the LSTM will be applied by treating each patient record's features as a short sequence or by using a simple feedforward network with LSTM layers.
- Alternatively, if temporal antenatal visit data is simulated, the LSTM can leverage visit-level sequences.

---

## Phase 5: Evaluation & Comparison (Week 10–11)

### Tasks

- Evaluate all four models on the held-out test set.
- Compute: Accuracy, Precision, Recall, F1 Score, ROC-AUC.
- Plot ROC curves for all models on a single figure.
- Generate confusion matrices for each model.
- Create a comparative performance table.
- Perform statistical significance tests if appropriate.
- Analyze feature importance (Logistic Regression coefficients, tree feature importances).
- Discuss strengths, weaknesses, and clinical applicability of each model.

### Deliverables

- `src/evaluation/metrics.py` — Evaluation functions.
- `src/evaluation/compare_models.py` — Model comparison utilities.
- `notebooks/05_evaluation.ipynb` — Evaluation and comparison notebook.
- `reports/figures/roc_curves.png`
- `reports/figures/confusion_matrices.png`
- `reports/tables/model_comparison.csv`

---

## Phase 6: Documentation & Report Writing (Week 12–14)

### Tasks

- Write the final project report following the required academic format:
  - Chapter 1: Introduction (background, problem statement, aim, objectives, scope)
  - Chapter 2: Literature Review
  - Chapter 3: Methodology (data generation, preprocessing, model development)
  - Chapter 4: Results and Discussion
  - Chapter 5: Conclusion and Recommendations
- Prepare presentation slides.
- Final code cleanup, commenting, and documentation.

### Deliverables

- Final project report (Word/LaTeX).
- Presentation slides.
- Clean, documented codebase.

---

## Risk Mitigation

| Risk | Mitigation |
| ---- | ---------- |
| Synthetic data does not reflect realistic patterns | Validate against published clinical statistics; iterate on generation parameters |
| Class imbalance skews model performance | Apply SMOTE, class weights, or stratified sampling |
| LSTM overfits on tabular data | Use dropout, early stopping, and regularization; compare to simpler models |
| Poor model performance across all algorithms | Re-examine feature selection; engineer interaction features; revisit data generation |

---

## Tools & Technologies

- **Language**: Python 3.10+
- **Data**: NumPy, Pandas, SciPy
- **Visualization**: Matplotlib, Seaborn
- **ML**: Scikit-learn
- **Deep Learning**: TensorFlow / Keras (for LSTM)
- **Notebooks**: Jupyter
- **Version Control**: Git
