# CHAPTER FOUR

## RESULTS AND DISCUSSION

### 4.1 Dataset Characteristics

The synthetic dataset comprised 5,000 patient records with 15 features and a binary risk label. After categorical encoding, the feature space expanded to 17 numeric columns. The target variable exhibited moderate class imbalance, with 70.4% low-risk and 29.6% high-risk cases.

**Table 4.1: Dataset Split Summary**

| Split | Samples | High-Risk (%) | Purpose |
|-------|---------|---------------|---------|
| Training | 3,500 | 29.6 | Model training |
| Validation | 750 | 29.6 | Hyperparameter tuning, threshold optimisation |
| Test | 750 | 29.6 | Final evaluation (held-out) |

After SMOTE oversampling, the training set for sklearn models was balanced to 4,928 samples (2,464 per class). The MLP model was trained on the original imbalanced training set with class weights (w₀ = 0.71, w₁ = 1.69).

### 4.2 Model Performance on Test Set

All four models were evaluated on the held-out test set (n = 750) using F1-optimised thresholds tuned on the validation set. Table 4.2 presents the comprehensive results.

**Table 4.2: Model Performance on Test Set**

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Threshold |
|-------|----------|-----------|--------|----------|---------|-----------|
| Logistic Regression | 0.8413 | 0.6900 | 0.8423 | 0.7586 | 0.9334 | 0.54 |
| Decision Tree | 0.6867 | 0.4835 | 0.8604 | 0.6191 | 0.8194 | 0.17 |
| Random Forest | 0.8320 | 0.6983 | 0.7613 | 0.7284 | 0.9043 | 0.47 |
| MLP | 0.8547 | 0.7116 | 0.8559 | 0.7771 | 0.9294 | 0.52 |

#### 4.2.1 MLP Performance

The Multi-Layer Perceptron achieved the highest F1 score (0.7771) and the second-highest AUC-ROC (0.9294), demonstrating strong overall discriminative performance. The model achieved a balanced trade-off between precision (0.7116) and recall (0.8559), correctly identifying 85.6% of true high-risk pregnancies while maintaining a positive predictive value of 71.2%. The optimal threshold (0.52) was close to the default 0.5, indicating well-calibrated probability outputs.

The MLP's training history (Figure 4.1) showed convergence within approximately 25 epochs, with the training loss decreasing from 0.73 to 0.48 and the validation loss stabilising around 0.47. The close alignment of training and validation curves indicates good generalisation without significant overfitting, attributed to the dropout layers (0.2–0.3) and batch normalisation employed in the architecture.

#### 4.2.2 Logistic Regression Performance

Logistic Regression achieved the highest AUC-ROC (0.9334) among all models, indicating excellent discriminative ability across all probability thresholds. Its F1 score (0.7586) was the second highest, with notably high recall (0.8423) and moderate precision (0.6900). The optimal threshold was 0.54.

The strong performance of Logistic Regression — comparable to the more complex MLP — is a noteworthy finding. It suggests that the relationship between the features and the risk label is substantially linear, which is consistent with the logistic risk labelling mechanism used in data generation. This finding has practical implications: in clinical deployment, the simpler and fully interpretable Logistic Regression model may be preferred over the MLP when performance differences are marginal.

#### 4.2.3 Random Forest Performance

The Random Forest achieved an AUC-ROC of 0.9043 and an F1 score of 0.7284, placing it third among the four models. Its precision (0.6983) was comparable to Logistic Regression, but its recall (0.7613) was lower, indicating a somewhat more conservative classification strategy. The optimal threshold (0.47) was slightly below 0.5, reflecting the ensemble's probability calibration.

While Random Forest is often considered the strongest off-the-shelf classifier for tabular data, its performance here was slightly below Logistic Regression and MLP. This may be attributed to the SMOTE-augmented training data, which can introduce artefacts in the synthetic minority examples that tree-based ensemble methods are particularly sensitive to (Blagus & Lusa, 2013).

#### 4.2.4 Decision Tree Performance

The Decision Tree classifier achieved the lowest overall performance, with an F1 score of 0.6191 and an AUC-ROC of 0.8194. While its recall was the highest among all models (0.8604), this came at the expense of very low precision (0.4835), meaning that more than half of its positive predictions were false alarms. The optimal threshold (0.17) was far below 0.5, indicating that the model's probability outputs were poorly calibrated.

The Decision Tree's inferior performance is expected and well-documented in the literature. Single trees are prone to high variance and are limited in their ability to capture complex feature interactions compared to ensemble methods. However, the tree's highest recall suggests it is aggressive in flagging potential high-risk cases, which may be acceptable in a triage screening context where sensitivity is prioritised.

### 4.3 ROC Curve Analysis

Figure 4.2 presents the ROC curves for all four models plotted on a single axis. The ROC curve plots the True Positive Rate (sensitivity) against the False Positive Rate (1 − specificity) at varying probability thresholds.

The MLP and Logistic Regression curves are nearly overlapping in the upper-left region of the ROC space, both demonstrating excellent discrimination with AUC values exceeding 0.93. The Random Forest curve follows closely with an AUC of 0.90. The Decision Tree curve shows visibly lower performance, with a stepped pattern characteristic of models that produce discrete probability outputs rather than smooth probability estimates.

All four models substantially outperform the random classifier (AUC = 0.5, diagonal reference line), confirming that the models have learned meaningful patterns from the data rather than relying on random chance or class frequency.

### 4.4 Confusion Matrix Analysis

The confusion matrices for each model reveal distinct classification patterns:

**Table 4.3: Confusion Matrix Summary**

| Model | True Neg | False Pos | False Neg | True Pos |
|-------|----------|-----------|-----------|----------|
| Logistic Regression | 444 | 84 | 35 | 187 |
| Decision Tree | 324 | 204 | 31 | 191 |
| Random Forest | 455 | 73 | 53 | 169 |
| MLP | 451 | 77 | 32 | 190 |

The Random Forest produced the fewest false positives but missed more true positives (lower recall). The Decision Tree produced the most false positives (lowest precision) but missed the fewest true positives (highest recall). The MLP and Logistic Regression struck the best balance between false positive and false negative rates.

In a clinical screening context, the MLP's confusion matrix profile is most desirable: it correctly identifies approximately 86% of high-risk pregnancies (high sensitivity) while maintaining a manageable false positive rate of approximately 25%, meaning that roughly one in four flagged patients would be falsely classified as high-risk and would receive unnecessary — but non-harmful — additional clinical review.

### 4.5 Cross-Validation Results

Stratified 5-fold cross-validation was conducted on the training data for the three sklearn models to assess performance stability. Table 4.4 presents the cross-validation results with 95% confidence intervals.

**Table 4.4: Stratified 5-Fold Cross-Validation Results**

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 0.832 ± 0.017 | 0.673 ± 0.030 | 0.844 ± 0.014 | 0.748 ± 0.021 | 0.923 ± 0.011 |
| Decision Tree | 0.763 ± 0.013 | 0.588 ± 0.023 | 0.670 ± 0.014 | 0.626 ± 0.015 | 0.775 ± 0.015 |
| Random Forest | 0.828 ± 0.013 | 0.701 ± 0.025 | 0.735 ± 0.031 | 0.717 ± 0.021 | 0.890 ± 0.015 |

The narrow confidence intervals across all metrics confirm the stability of model performance. Logistic Regression demonstrated the most consistent performance (lowest variance in F1 and AUC). The Decision Tree showed the highest variance in recall (SD = 0.014) and the lowest overall metrics, consistent with its known sensitivity to data partitioning. The Random Forest exhibited slightly higher variance than Logistic Regression, particularly in recall (SD = 0.031).

The cross-validation F1 scores are consistent with the test set results, with differences of less than 0.02 for all models, indicating that the test set evaluation is representative and not biased by a favourable split.

### 4.6 Feature Importance Analysis

Feature importance was analysed across three model types to identify the most influential predictors of pregnancy risk.

**Table 4.5: Top 10 Features by Importance**

| Rank | Logistic Regression (|Coefficient|) | Decision Tree (Gini) | Random Forest (Gini) |
|------|--------------------------------------|---------------------|----------------------|
| 1 | History of hypertension (2.00) | Systolic BP (0.276) | Systolic BP (0.157) |
| 2 | Previous pre-eclampsia (1.95) | BMI (0.237) | BMI (0.139) |
| 3 | Gestational diabetes (1.55) | Haemoglobin (0.126) | Diastolic BP (0.125) |
| 4 | Systolic BP (1.21) | Fasting glucose (0.092) | Haemoglobin (0.100) |
| 5 | BMI (1.14) | Diastolic BP (0.081) | Fasting glucose (0.093) |
| 6 | Haemoglobin (0.95) | Weight gain (0.054) | Weight gain (0.083) |
| 7 | Diastolic BP (0.71) | Parity (0.033) | Maternal age (0.072) |
| 8 | Fasting glucose (0.44) | Maternal age (0.032) | Antenatal visits (0.065) |
| 9 | Antenatal visits (0.41) | Antenatal visits (0.031) | Parity (0.055) |
| 10 | Facility type (0.39) | SES (0.015) | SES (0.019) |

Several patterns emerge from the feature importance analysis:

1. **Medical history features dominate in Logistic Regression**: History of hypertension, previous pre-eclampsia, and gestational diabetes are the top three features by coefficient magnitude. This is clinically intuitive — these are established strong risk factors that act as independent binary indicators of elevated risk.

2. **Continuous clinical features dominate in tree-based models**: Systolic BP and BMI are the top two features in both Decision Tree and Random Forest. This reflects the ability of tree-based models to exploit non-linear thresholds in continuous variables (e.g., splitting at SBP ≥ 140 mmHg).

3. **Consistent importance of key features**: Despite using different importance measures, all three models agree that systolic blood pressure, BMI, haemoglobin, and fasting blood glucose are among the most important predictors. This convergence across model types strengthens confidence in the clinical relevance of these features.

4. **Socioeconomic status and facility type have low importance**: These contextual features rank lowest across all models, suggesting that — after controlling for clinical indicators — socioeconomic and healthcare access factors have limited additional predictive value for risk classification. This is consistent with the finding that clinical indicators are more proximal determinants of obstetric risk.

### 4.7 Discussion

#### 4.7.1 Overall Model Performance

The results demonstrate that machine learning models can achieve good to excellent discriminative performance (AUC 0.82–0.93) on the task of binary high-risk pregnancy classification using clinically relevant features derived from Nigerian epidemiological data. These AUC values are comparable to those reported in international maternal risk prediction studies: Jhee et al. (2019) reported AUC values of 0.73–0.89 for pre-eclampsia prediction, while Sufriyana et al. (2020) reported AUC values of 0.82–0.92 for general obstetric risk assessment.

The finding that the MLP neural network achieved the highest F1 score (0.78) while Logistic Regression achieved the highest AUC (0.93) highlights an important nuance: discriminative ability (AUC) and classification performance at a fixed threshold (F1) are distinct measures. Logistic Regression's superior AUC indicates better overall probability calibration, while the MLP's higher F1 suggests better classification performance after threshold optimisation.

#### 4.7.2 Interpretability versus Complexity Trade-off

A key finding of this study is that Logistic Regression — the simplest model evaluated — performed comparably to or better than more complex alternatives on several metrics. This has significant implications for clinical deployment in the Nigerian context:

- **At primary health centres**, where computational resources are limited and model transparency is essential for clinician trust, Logistic Regression provides a practical, interpretable solution that can be implemented as a simple scoring sheet or embedded in a basic calculator application.
- **At tertiary hospitals**, where computational infrastructure is available and the highest possible sensitivity is prioritised, the MLP neural network may be preferred for its slightly superior recall (85.6% vs. 84.2%).

This finding is consistent with Christodoulou et al. (2019), who found that logistic regression performed comparably to ML algorithms on clinical prediction tasks in many settings, particularly when sample sizes are moderate and the true relationship is approximately linear.

#### 4.7.3 Clinical Utility

From a clinical standpoint, the most important performance characteristic for a screening tool is high recall (sensitivity), ensuring that true high-risk pregnancies are not missed. The MLP achieved a recall of 85.6%, meaning that approximately 14% of true high-risk pregnancies would not be flagged by the model. While this is a meaningful limitation, it represents a substantial improvement over ad hoc clinical screening in overburdened PHCs.

The precision of 71.2% (for the MLP) means that approximately 29% of flagged patients would not truly be high-risk. In a screening context, this false positive rate is manageable, as the consequence of a false positive is additional clinical review rather than unnecessary treatment. The cost-benefit ratio therefore favours high sensitivity even at the expense of some precision.

#### 4.7.4 The Role of Threshold Optimisation

Threshold optimisation had a meaningful impact on model performance. The default 0.5 threshold, which assumes equal misclassification costs for both classes, is suboptimal for imbalanced clinical prediction tasks. By tuning the threshold on the validation set to maximise F1 score, each model's precision–recall trade-off was optimised for the specific class distribution and probability calibration.

The Decision Tree's optimal threshold (0.17) was notably low, indicating that the model's probability outputs are poorly calibrated — the tree assigns relatively low probabilities to many true high-risk cases. This is a known limitation of single decision trees, whose probability estimates are based on leaf node class frequencies rather than well-calibrated continuous probabilities.

#### 4.7.5 Limitations

Several limitations should be acknowledged:

1. **Synthetic data**: The results are based on synthetic data generated from published distributions. While the data was designed to be epidemiologically realistic, it cannot fully capture the complexity of real clinical data, including missing values, measurement errors, comorbidity interactions, and temporal patterns.

2. **Risk labelling mechanism**: The probabilistic logistic model used for risk labelling, while avoiding deterministic circularity, inherently favours linear models (particularly Logistic Regression) that share a similar functional form. This may partially explain Logistic Regression's strong performance and should be considered when interpreting the relative model rankings.

3. **Single-timepoint prediction**: The study considers risk assessment at a single point in time, whereas real clinical practice involves longitudinal monitoring across multiple antenatal visits. Future work could incorporate temporal models if sequential visit data were available.

4. **External validation**: The models have not been validated on real patient data. External validation on Nigerian clinical datasets would be essential before any clinical deployment.

5. **Feature set**: The 15 features used in this study, while clinically relevant, do not capture all potential risk factors (e.g., multiple pregnancy, placenta praevia, Rh incompatibility, HIV status). A more comprehensive feature set could improve model performance.
