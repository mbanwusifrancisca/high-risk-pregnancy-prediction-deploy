# Predictive Model for High-Risk Pregnancy Identification Using Machine Learning: A Comparative Study with Synthetic Nigerian Maternal Health Data

---

## ABSTRACT

Maternal mortality remains a critical public health challenge in Nigeria, with a maternal mortality ratio of approximately 512 per 100,000 live births. Early identification of high-risk pregnancies through systematic screening could significantly reduce preventable maternal deaths, yet the limited availability of digitised health records hinders the development of locally relevant predictive models. This study developed and comparatively evaluated four machine learning models — Logistic Regression, Decision Tree, Random Forest, and Multi-Layer Perceptron (MLP) neural network — for binary high-risk pregnancy classification using synthetic data parameterised from 15 peer-reviewed Nigerian and West African epidemiological studies.

A synthetic dataset of 5,000 maternal health records was generated with 15 clinically relevant features spanning demographic, clinical, and obstetric domains. Realistic inter-feature correlations were introduced through shared latent factors, and risk labels were assigned via a probabilistic logistic model with Bernoulli sampling to simulate clinical uncertainty. The dataset was preprocessed with selective feature scaling, stratified train–validation–test splitting (70/15/15), SMOTE oversampling for traditional models, and class weighting for the neural network. F1-optimal threshold tuning was performed on the validation set.

The MLP achieved the highest F1 score (0.777) with a recall of 0.856 and precision of 0.712 (AUC-ROC = 0.929). Logistic Regression achieved the highest AUC-ROC (0.933) with an F1 of 0.759, demonstrating that simpler, interpretable models can perform comparably to neural networks on this task. Random Forest achieved an AUC-ROC of 0.904 and F1 of 0.728, while the Decision Tree was the weakest performer (AUC-ROC = 0.819, F1 = 0.619). Stratified 5-fold cross-validation confirmed performance stability, with Logistic Regression achieving F1 = 0.748 ± 0.021 and AUC = 0.923 ± 0.011. Feature importance analysis consistently identified systolic blood pressure, BMI, haemoglobin, history of hypertension, and previous pre-eclampsia as the most predictive features across all models.

The comparable performance of Logistic Regression and MLP suggests that interpretable models may be sufficient for maternal risk screening in resource-constrained Nigerian primary health centres, while more complex models may be reserved for tertiary facilities. This study establishes a reproducible machine learning pipeline and population-specific data generation methodology that can be directly applied to real clinical data as Nigerian health information systems mature.

**Keywords:** machine learning, maternal health, high-risk pregnancy, predictive modelling, synthetic data, Nigeria, logistic regression, neural network, clinical decision support

---

## TABLE OF CONTENTS

- **Abstract**
- **Table of Contents**
- **List of Tables**
- **List of Figures**

### Chapter One: Introduction
- 1.1 Background of Study
- 1.2 Statement of the Problem
- 1.3 Aim of the Study
- 1.4 Objectives of the Study
- 1.5 Scope of the Study
- 1.6 Significance of the Study
- 1.7 Definition of Terms
- 1.8 Organisation of the Study

### Chapter Two: Literature Review
- 2.1 Maternal Mortality and Morbidity in Nigeria
- 2.2 Clinical Risk Factors for High-Risk Pregnancy
- 2.3 Machine Learning in Maternal Health
- 2.4 Synthetic Data in Healthcare Research
- 2.5 Summary of Literature Review

### Chapter Three: Methodology
- 3.1 Research Design
- 3.2 Data Generation
- 3.3 Data Preprocessing
- 3.4 Model Development
- 3.5 Evaluation Framework
- 3.6 Implementation Environment

### Chapter Four: Results and Discussion
- 4.1 Dataset Characteristics
- 4.2 Model Performance on Test Set
- 4.3 ROC Curve Analysis
- 4.4 Confusion Matrix Analysis
- 4.5 Cross-Validation Results
- 4.6 Feature Importance Analysis
- 4.7 Discussion

### Chapter Five: Conclusion and Recommendations
- 5.1 Summary of Findings
- 5.2 Conclusion
- 5.3 Recommendations
- 5.4 Contribution to Knowledge

### References

---

## LIST OF TABLES

- Table 3.1: Feature Definitions and Data Sources
- Table 3.2: Logistic Model Coefficients for Risk Labelling
- Table 4.1: Dataset Split Summary
- Table 4.2: Model Performance on Test Set
- Table 4.3: Confusion Matrix Summary
- Table 4.4: Stratified 5-Fold Cross-Validation Results
- Table 4.5: Top 10 Features by Importance

## LIST OF FIGURES

- Figure 4.1: MLP Training History (Loss and Accuracy)
- Figure 4.2: ROC Curve Comparison — All Models
- Figure 4.3: Confusion Matrices — All Models
- Figure 4.4: Feature Importance Comparison (Top 10)
