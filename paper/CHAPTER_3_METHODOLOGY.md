# CHAPTER THREE

## METHODOLOGY

### 3.1 Research Design

The study adopts an experimental research design involving the comparative evaluation of four supervised machine learning classification algorithms on the task of binary high-risk pregnancy prediction. The independent variables are the maternal health features (demographic, clinical, and obstetric), while the dependent variable is the binary risk label (0 = low-risk, 1 = high-risk). Model performance is assessed using standard classification metrics on a held-out test set, with statistical stability confirmed through stratified k-fold cross-validation.

### 3.2 Data Generation

#### 3.2.1 Rationale

Due to the limited availability of large-scale, digitised maternal health records in Nigeria, the study employs synthetic data generation using statistical distributions derived from peer-reviewed Nigerian and West African epidemiological studies. This approach enables controlled experimentation while preserving the population-specific distributional characteristics essential for developing locally relevant predictive models.

#### 3.2.2 Feature Selection and Parameterisation

Fifteen maternal health features were selected based on clinical relevance and availability in routine antenatal care settings. Each feature was parameterised using published values from Nigerian studies, as summarised in Table 3.1.

**Table 3.1: Feature Definitions and Data Sources**

| Feature | Type | Distribution | Source |
|---------|------|-------------|--------|
| Maternal age (years) | Continuous | N(29.0, 5.5), [15, 48] | Senbanjo et al. (2021); Ugwuja et al. (2011) |
| Parity | Discrete | Poisson(λ=2.5), [0, 10] | NDHS 2018; Olokor et al. (2016) |
| Antenatal visits | Discrete | Poisson(λ=4.0), [0, 15] | Fagbamigbe et al. (2020) |
| Socioeconomic status | Categorical | Low 35%, Middle 40%, High 25% | NDHS 2018; World Bank (2022) |
| Facility type | Categorical | PHC 40%, GH 30%, TH 15%, PC 15% | Bolarinwa et al. (2021) |
| Systolic BP (mmHg) | Continuous | N(112.0, 12.0), [85, 200] | Amoakoh-Coleman et al. (2017) |
| Diastolic BP (mmHg) | Continuous | N(70.0, 10.0), [50, 130] | Amoakoh-Coleman et al. (2017) |
| Haemoglobin (g/dL) | Continuous | N(10.9, 1.8), [4.0, 17.0] | Akinbami et al. (2013) |
| BMI (kg/m²) | Continuous | N(27.0, 5.4), [14.0, 50.0] | Senbanjo et al. (2021) |
| Fasting blood glucose (mg/dL) | Continuous | N(74.5, 11.5), [40, 200] | Olokor et al. (2016) |
| Gestational weight gain (kg) | Continuous | N(10.7, 3.4), [0, 30] | Eze et al. (2017) |
| History of hypertension | Binary | Prevalence 8% | Okonofua et al. (2024); Adeloye et al. (2015) |
| Gestational diabetes | Binary | Prevalence 5% | Mustapha et al. (2021); Anzaku & Musa (2013) |
| Previous caesarean | Binary | Prevalence 3% | Adewuyi et al. (2019) |
| Previous pre-eclampsia | Binary | Prevalence 4% | Ogunlaja et al. (2024) |

#### 3.2.3 Inter-Feature Correlations

To produce epidemiologically realistic data, inter-feature correlations were introduced through three mechanisms:

1. **Shared latent factor**: A latent body mass index (BMI) variable was drawn first and used to drive correlated effects on blood pressure, fasting blood glucose, and gestational weight gain. This reflects the established cardiovascular-metabolic syndrome in which obesity drives hypertension and glucose dysregulation (Poston et al., 2016).

2. **Shared and independent noise components for blood pressure**: Systolic and diastolic blood pressure were generated using a shared cardiovascular noise component and independent noise terms, producing a realistic correlation (r ≈ 0.65) without near-perfect collinearity.

3. **Conditional probabilities for medical history**: Binary medical history features were generated with probabilities conditioned on related clinical features (e.g., hypertension probability increases with higher systolic BP; gestational diabetes probability increases with higher fasting glucose). A critical clinical constraint was enforced: nulliparous women (parity = 0) were assigned zero probability for previous caesarean section and previous pre-eclampsia, as these conditions require a prior pregnancy.

#### 3.2.4 Probabilistic Risk Labelling

The binary risk label was assigned using a probabilistic logistic model rather than deterministic rules:

$$P(\text{high-risk}) = \sigma\left(\sum_{i} \beta_i x_i + \beta_0\right)$$

where σ is the logistic sigmoid function, x_i are the standardised features, and β_i are clinically-weighted coefficients reflecting the relative importance of each risk factor based on WHO and Nigerian obstetric literature. The actual risk label was then sampled from a Bernoulli distribution:

$$\text{risk\_label} \sim \text{Bernoulli}(P(\text{high-risk}))$$

This probabilistic approach ensures that the risk label is not a deterministic function of the features, introducing realistic clinical uncertainty and preventing the trivial case where a model could perfectly reconstruct the labelling rule. The coefficients were calibrated to produce a high-risk prevalence of approximately 29.6%, consistent with estimates from the Nigerian maternal health literature.

**Table 3.2: Logistic Model Coefficients for Risk Labelling**

| Feature | Coefficient | Interpretation |
|---------|------------|----------------|
| Systolic BP (z-scored) | 1.4 | Elevated BP strongly increases risk |
| Diastolic BP (z-scored) | 0.9 | Elevated diastolic BP increases risk |
| Haemoglobin (z-scored) | −1.1 | Lower Hb (anaemia) increases risk |
| BMI (z-scored) | 1.2 | Higher BMI increases risk |
| Fasting glucose (z-scored) | 0.7 | Higher glucose increases risk |
| Excessive weight gain (>18 kg) | 0.5 | Binary indicator |
| Insufficient weight gain (<7 kg) | 0.5 | Binary indicator |
| History of hypertension | 2.8 | Strong risk factor |
| Gestational diabetes | 2.4 | Strong risk factor |
| Previous caesarean | 1.5 | Moderate risk factor |
| Previous pre-eclampsia | 2.6 | Strong risk factor |
| Advanced maternal age (≥35) | 0.7 | Moderate risk factor |
| Very young mother (<18) | 0.9 | Moderate risk factor |
| Grand multiparity (≥5) | 0.4 | Mild risk factor |
| Antenatal visits (z-scored) | −0.5 | Fewer visits increases risk |
| Intercept | −2.6 | Calibrated for ~28% prevalence |

#### 3.2.5 Dataset Summary

A total of 5,000 synthetic patient records were generated with a fixed random seed (42) for reproducibility. The resulting dataset contained 15 features and 1 binary target variable, with a high-risk prevalence of 29.6%.

### 3.3 Data Preprocessing

#### 3.3.1 Categorical Encoding

Two categorical features required encoding:
- **Facility type** (nominal, 4 categories): One-hot encoded with the first category dropped to avoid multicollinearity, producing three binary indicator variables (primary_health_centre as reference).
- **Socioeconomic status** (ordinal, 3 levels): Ordinal encoded as Low = 0, Middle = 1, High = 2.

After encoding, the feature set comprised 17 numeric columns.

#### 3.3.2 Train–Validation–Test Split

The dataset was split into training (70%, n = 3,500), validation (15%, n = 750), and test (15%, n = 750) sets using stratified random sampling to preserve the class distribution across all splits. The validation set was used for hyperparameter tuning and threshold optimisation, while the test set was held out exclusively for final evaluation.

#### 3.3.3 Feature Scaling

Continuous features were standardised using z-score normalisation (StandardScaler), fitted on the training set and applied to all sets:

$$x_{\text{scaled}} = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$

Binary features (including one-hot encoded variables and original binary indicators) were excluded from scaling to preserve their interpretability and avoid distorting their natural {0, 1} range. This selective scaling approach prevents information leakage and ensures that binary features retain their clinically meaningful values.

#### 3.3.4 Class Imbalance Handling

The training data exhibited moderate class imbalance (29.6% high-risk vs. 70.4% low-risk). Two strategies were employed depending on the model type:

1. **SMOTE oversampling** (for sklearn models): The Synthetic Minority Over-sampling Technique was applied to the training set only, generating synthetic high-risk examples to achieve a balanced 1:1 class ratio (n = 4,928 after resampling). SMOTE was applied after the train–test split to prevent data leakage.

2. **Class weighting** (for MLP): The MLP neural network was trained with inverse-frequency class weights (w₀ = 0.71, w₁ = 1.69), which adjusts the loss function to penalise misclassification of the minority class more heavily. This approach is preferred for neural networks as it avoids the potential for SMOTE-generated artefacts to affect gradient-based learning.

### 3.4 Model Development

Four classification algorithms of increasing complexity were implemented and compared:

#### 3.4.1 Logistic Regression

Logistic Regression was selected as the baseline interpretable model due to its widespread use in clinical prediction and its ability to produce interpretable odds ratios. The model estimates the log-odds of high-risk classification as a linear function of the features:

$$\log\left(\frac{P(y=1|X)}{1-P(y=1|X)}\right) = \beta_0 + \sum_{j=1}^{p}\beta_j x_j$$

Hyperparameter tuning was performed using 5-fold GridSearchCV over regularisation strength C ∈ {0.01, 0.1, 1, 10, 100} with L2 penalty and LBFGS solver (max iterations = 1,000).

#### 3.4.2 Decision Tree Classifier

The Decision Tree Classifier was included as a rule-based model that produces human-readable decision rules, making it suitable for clinical settings where model transparency is essential. Decision trees recursively partition the feature space using information gain or Gini impurity criteria.

Hyperparameter tuning was performed using 5-fold GridSearchCV over:
- Maximum depth: {3, 5, 7, 10}
- Minimum samples per split: {5, 10, 20, 30}
- Minimum samples per leaf: {5, 10, 20}
- Splitting criterion: {Gini, Entropy}
- Cost-complexity pruning alpha: {0.0, 0.001, 0.005, 0.01}

The pruning parameter (ccp_alpha) was included to prevent overfitting on the SMOTE-augmented training data.

#### 3.4.3 Random Forest Classifier

Random Forest, an ensemble method that aggregates predictions from multiple decorrelated decision trees, was included for its well-established performance on tabular classification tasks. Each tree is trained on a bootstrap sample with a random subset of features, reducing variance and improving generalisation.

Hyperparameter tuning was performed using RandomizedSearchCV (50 iterations, 5-fold CV) over:
- Number of estimators: {50, 100, 200, 300, 500}
- Maximum depth: {5, 10, 15, 20, None}
- Minimum samples per split: {2, 5, 10}
- Minimum samples per leaf: {1, 2, 4}
- Maximum features: {sqrt, log2, None}
- Bootstrap: {True, False}

#### 3.4.4 Multi-Layer Perceptron (MLP)

A feedforward Multi-Layer Perceptron neural network was selected as the deep learning approach. Unlike recurrent architectures such as LSTM networks, the MLP is the appropriate architecture for tabular data that does not possess temporal or sequential structure.

The MLP architecture consisted of:

| Layer | Output Shape | Parameters | Details |
|-------|-------------|------------|---------|
| Dense (input) | 64 | 1,152 | ReLU activation |
| BatchNormalization | 64 | 256 | |
| Dropout | 64 | — | Rate = 0.3 |
| Dense (hidden 1) | 32 | 2,080 | ReLU activation |
| BatchNormalization | 32 | 128 | |
| Dropout | 32 | — | Rate = 0.3 |
| Dense (hidden 2) | 16 | 528 | ReLU activation |
| Dropout | 16 | — | Rate = 0.2 |
| Dense (output) | 1 | 17 | Sigmoid activation |

The model was compiled with the Adam optimiser (learning rate = 0.001), binary cross-entropy loss, and trained for up to 100 epochs with batch size 32. Two callbacks were employed:
- **Early stopping** (patience = 10, monitoring validation loss, restoring best weights)
- **Learning rate reduction** (factor = 0.5, patience = 5, minimum LR = 10⁻⁶)

The MLP was trained with class weights rather than SMOTE to avoid introducing synthetic artefacts into gradient-based optimisation.

### 3.5 Evaluation Framework

#### 3.5.1 Metrics

All models were evaluated using the following standard classification metrics:

1. **Accuracy**: The proportion of correctly classified instances.
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$$

2. **Precision**: The proportion of predicted high-risk cases that are truly high-risk.
$$\text{Precision} = \frac{TP}{TP + FP}$$

3. **Recall (Sensitivity)**: The proportion of true high-risk cases correctly identified.
$$\text{Recall} = \frac{TP}{TP + FN}$$

4. **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure.
$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

5. **AUC-ROC**: The area under the receiver operating characteristic curve, measuring discriminative ability across all probability thresholds.

In the clinical context of pregnancy risk screening, recall is prioritised over precision, as the cost of missing a true high-risk pregnancy (false negative) is substantially higher than the cost of a false alarm (false positive), which results in additional — but non-harmful — clinical review.

#### 3.5.2 Threshold Optimisation

Rather than using the default 0.5 probability threshold, the F1-optimal threshold was determined for each model by scanning thresholds from 0.10 to 0.90 in increments of 0.01 on the **validation set**. The threshold maximising the F1 score on the validation set was then applied to the held-out test set. This approach avoids information leakage while optimising the precision–recall trade-off for each model's probability calibration.

#### 3.5.3 Stratified K-Fold Cross-Validation

To assess the statistical stability of model performance, stratified 5-fold cross-validation was conducted on the training data for the three sklearn models (Logistic Regression, Decision Tree, Random Forest). In each fold:
1. SMOTE was applied only to the fold's training portion.
2. Feature scaling was fitted on the fold's training portion and applied to the fold's validation portion.
3. All five metrics were computed.

The mean and standard deviation across folds were reported, along with 95% confidence intervals calculated as:

$$CI_{95\%} = \bar{x} \pm 1.96 \cdot \frac{s}{\sqrt{k}}$$

where x̄ is the mean metric across k folds and s is the standard deviation.

#### 3.5.4 Feature Importance Analysis

Feature importance was extracted from each model type:
- **Logistic Regression**: Absolute values of learned coefficients.
- **Decision Tree and Random Forest**: Gini importance (mean decrease in impurity).

This analysis identifies which clinical features contribute most to the risk prediction, providing clinically interpretable insights.

### 3.6 Implementation Environment

All experiments were implemented in Python 3.10 with the following key libraries:

| Library | Version | Purpose |
|---------|---------|---------|
| NumPy | 1.26 | Numerical computation |
| Pandas | 2.2 | Data manipulation |
| Scikit-learn | 1.4 | ML models and evaluation |
| TensorFlow/Keras | 2.16 | MLP neural network |
| Imbalanced-learn | 0.12 | SMOTE oversampling |
| SciPy | 1.12 | Statistical distributions |
| Matplotlib/Seaborn | 3.8/0.13 | Visualisation |

The complete codebase was developed as a modular Python project with separate modules for data generation, preprocessing, model training, and evaluation. All experiments used a fixed random seed (42) for full reproducibility.
