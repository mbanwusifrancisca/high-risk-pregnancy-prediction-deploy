# Model Card: High-Risk Pregnancy Prediction Pipeline

## Model Overview

| Field | Value |
|-------|-------|
| **Task** | Binary classification — high-risk pregnancy identification |
| **Target** | `risk_label` ∈ {0: low-risk, 1: high-risk} |
| **Models** | Logistic Regression, Decision Tree, Random Forest, Multi-Layer Perceptron (MLP) |
| **Best Model** | MLP (F1 = 0.777, AUC = 0.929) |
| **Dataset** | Synthetic, 5,000 records, 15 clinical features |
| **Population** | Nigerian maternal health (parameterised from published epidemiological data) |
| **Version** | 1.0 |
| **License** | MIT |
| **Last Updated** | March 2026 |

---

## Intended Use

### Primary Use
Decision-support tool for antenatal risk screening at primary health centres (PHCs) and general hospitals in Nigeria. Designed to flag pregnancies that may benefit from closer monitoring or referral to higher-level facilities.

### Intended Users
- Midwives and nurses at PHCs conducting routine antenatal assessments
- Obstetricians at secondary/tertiary facilities for triage prioritisation
- Researchers developing maternal health ML pipelines for low-resource settings

### Out-of-Scope Uses
- **Not for autonomous clinical decision-making** — the model is a screening aid, not a diagnostic tool
- **Not validated on real patient data** — results are based on synthetic data only
- **Not for non-Nigerian populations** — feature distributions are specific to the Nigerian obstetric population

---

## Dataset Card

### Generation Method
Parametric synthetic data generation using published marginal distributions, with:
- **Shared latent factors** for realistic inter-feature correlations (BMI → BP, glucose, weight gain)
- **Conditional probabilities** for medical history (hypertension conditioned on BP; GDM conditioned on glucose)
- **Clinical constraints** (nulliparous women cannot have previous caesarean or pre-eclampsia)
- **Probabilistic risk labelling** via logistic model + Bernoulli sampling (not deterministic rules)

### Dataset Statistics

| Property | Value |
|----------|-------|
| Total records | 5,000 |
| Features | 15 (9 continuous, 4 binary, 2 categorical) |
| Target prevalence | 29.6% high-risk |
| Train / Val / Test split | 70% / 15% / 15% (stratified) |
| Random seed | 42 |

### Feature Definitions

| Feature | Type | Distribution | Clinical Source |
|---------|------|-------------|-----------------|
| `maternal_age` | Continuous | N(29.0, 5.5), [15, 48] | Senbanjo et al. (2021) |
| `parity` | Discrete | Poisson(λ=2.5), [0, 10] | NDHS 2018 |
| `antenatal_visits` | Discrete | Poisson(λ=4.0), [0, 15] | Fagbamigbe et al. (2020) |
| `socioeconomic_status` | Ordinal | Low 35%, Mid 40%, High 25% | NDHS 2018 |
| `facility_type` | Nominal | PHC 40%, GH 30%, TH 15%, PC 15% | Bolarinwa et al. (2021) |
| `systolic_bp` | Continuous | N(112, 12), [85, 200] | Amoakoh-Coleman et al. (2017) |
| `diastolic_bp` | Continuous | N(70, 10), [50, 130] | Amoakoh-Coleman et al. (2017) |
| `hemoglobin` | Continuous | N(10.9, 1.8), [4, 17] | Akinbami et al. (2013) |
| `bmi` | Continuous | N(27.0, 5.4), [14, 50] | Senbanjo et al. (2021) |
| `fasting_blood_glucose` | Continuous | N(74.5, 11.5), [40, 200] | Olokor et al. (2016) |
| `weight_gain` | Continuous | N(10.7, 3.4), [0, 30] | Eze et al. (2017) |
| `history_hypertension` | Binary | ~8% prevalence | Okonofua et al. (2024) |
| `gestational_diabetes` | Binary | ~5% prevalence | Mustapha et al. (2021) |
| `previous_cesarean` | Binary | ~3% prevalence | Adewuyi et al. (2019) |
| `previous_preeclampsia` | Binary | ~4% prevalence | Ogunlaja et al. (2024) |

### Risk Labelling Mechanism

Risk probability is computed via a logistic model:

```
P(high-risk) = σ(1.4·z(SBP) + 0.9·z(DBP) − 1.1·z(Hb) + 1.2·z(BMI)
               + 0.7·z(glucose) + 0.5·I(WG>18) + 0.5·I(WG<7)
               + 2.8·HTN + 2.4·GDM + 1.5·CS + 2.6·PE
               + 0.7·I(age≥35) + 0.9·I(age<18) + 0.4·I(parity≥5)
               − 0.5·z(ANC) − 2.6)
```

Where `z()` denotes z-score standardisation and `σ` is the logistic sigmoid. The binary label is then:

```
risk_label ~ Bernoulli(P(high-risk))
```

This produces ~87% agreement between the probability and the label, simulating realistic clinical uncertainty.

### Known Biases and Limitations

1. **Synthetic data** — cannot capture rare comorbidities, measurement errors, or missing data patterns present in real clinical data
2. **Risk labelling favours linear models** — the logistic labelling mechanism shares functional form with Logistic Regression, potentially inflating its relative performance
3. **No temporal dimension** — single-timepoint snapshot, not longitudinal across antenatal visits
4. **Population-specific** — distributions are calibrated for Nigeria; direct transfer to other populations is not recommended

---

## Model Details

### Preprocessing Pipeline
1. **Categorical encoding**: One-hot (facility_type), ordinal (SES)
2. **Feature scaling**: StandardScaler on continuous features only; binary features excluded
3. **Class imbalance**: SMOTE for sklearn models (balanced to 1:1); class weights for MLP
4. **Split**: 70/15/15 stratified train/val/test

### Model Architectures

#### Logistic Regression
- Solver: LBFGS, L2 penalty
- Tuned: C ∈ {0.01, 0.1, 1, 10, 100} via 5-fold GridSearchCV

#### Decision Tree
- Tuned: max_depth ∈ {3,5,7,10}, min_samples_split ∈ {5,10,20,30}, min_samples_leaf ∈ {5,10,20}, criterion ∈ {gini, entropy}, ccp_alpha ∈ {0, 0.001, 0.005, 0.01}
- Cost-complexity pruning to prevent SMOTE overfitting

#### Random Forest
- Tuned: 50 iterations RandomizedSearchCV over n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, bootstrap

#### MLP (Multi-Layer Perceptron)
- Architecture: Dense(64)→BN→Drop(0.3)→Dense(32)→BN→Drop(0.3)→Dense(16)→Drop(0.2)→Dense(1, sigmoid)
- Optimizer: Adam (lr=0.001), binary cross-entropy loss
- Callbacks: EarlyStopping (patience=10), ReduceLROnPlateau (factor=0.5, patience=5)
- Epochs: up to 100, batch_size=32, class weights instead of SMOTE

### Threshold Optimisation
F1-optimal threshold tuned on **validation set** by grid search (0.10–0.90, step 0.01), then applied to test set.

---

## Performance Metrics

### Test Set Results (n = 750)

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | Threshold |
|-------|----------|-----------|--------|----|---------|-----------|
| Logistic Regression | 0.8413 | 0.6900 | 0.8423 | 0.7586 | **0.9334** | 0.54 |
| Decision Tree | 0.6867 | 0.4835 | **0.8604** | 0.6191 | 0.8194 | 0.17 |
| Random Forest | 0.8320 | 0.6983 | 0.7613 | 0.7284 | 0.9043 | 0.47 |
| **MLP** | **0.8547** | **0.7116** | 0.8559 | **0.7771** | 0.9294 | 0.52 |

### Confusion Matrices

| Model | TN | FP | FN | TP |
|-------|----|----|----|-----|
| Logistic Regression | 444 | 84 | 35 | 187 |
| Decision Tree | 324 | 204 | 31 | 191 |
| Random Forest | 455 | 73 | 53 | 169 |
| MLP | 451 | 77 | 32 | 190 |

### Cross-Validation (Stratified 5-Fold)

| Model | F1 (mean ± std) | AUC (mean ± std) |
|-------|-----------------|-------------------|
| Logistic Regression | 0.748 ± 0.021 | 0.923 ± 0.011 |
| Decision Tree | 0.626 ± 0.015 | 0.775 ± 0.015 |
| Random Forest | 0.717 ± 0.021 | 0.890 ± 0.015 |

### Top Predictive Features (Consistent Across Models)
1. Systolic blood pressure
2. BMI
3. Haemoglobin (inverse — lower = higher risk)
4. History of hypertension
5. Previous pre-eclampsia
6. Gestational diabetes
7. Fasting blood glucose
8. Diastolic blood pressure

---

## Ethical Considerations

- **No real patient data** was used; no privacy or consent concerns
- **Clinical deployment requires external validation** on real Nigerian hospital data before any use in patient care
- **Algorithmic bias**: the synthetic data reflects published population averages and may not capture subpopulation-specific risk profiles (e.g., sickle cell trait carriers, HIV-positive women)
- **False negatives**: at 85.6% recall (MLP), approximately 14% of true high-risk pregnancies would be missed — the model must be used as a supplement to, not replacement for, clinical judgement
- **Equity**: the model was specifically designed for the Nigerian population to address the gap in locally relevant tools; applying it to other populations without recalibration could introduce bias

---

## Reproducibility

| Component | Details |
|-----------|---------|
| Python | 3.10 |
| scikit-learn | 1.4 |
| TensorFlow | 2.16 |
| imbalanced-learn | 0.12 |
| Random seed | 42 (all operations) |
| Code | `run_pipeline.py` — end-to-end execution |
| Data generation | `src/data/generate_synthetic_data.py` |
| Preprocessing | `src/data/preprocess.py` |
| Models | `src/models/` |
| Evaluation | `src/evaluation/metrics.py` |

To reproduce all results:
```bash
python run_pipeline.py
```

---

## Citation

If you use this work, please cite:

```
@misc{high_risk_pregnancy_2026,
  title={Predictive Model for High-Risk Pregnancy Identification Using Machine Learning:
         A Comparative Study with Synthetic Nigerian Maternal Health Data},
  year={2026},
  url={https://github.com/kossiso/high-risk-pregnancy-prediction}
}
```
