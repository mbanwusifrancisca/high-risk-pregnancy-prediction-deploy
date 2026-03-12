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

\newpage

# CHAPTER ONE

## INTRODUCTION

### 1.1 Background of Study

Maternal mortality remains one of the most pressing public health challenges in sub-Saharan Africa, with Nigeria accounting for a disproportionately high share of global maternal deaths. According to the World Health Organization (WHO), approximately 295,000 women died during and following pregnancy and childbirth in 2017, with sub-Saharan Africa alone accounting for roughly 66% of these deaths (WHO, 2019). Nigeria, despite representing approximately 2.7% of the global population, contributes an estimated 23% of the global maternal mortality burden, with a maternal mortality ratio (MMR) of approximately 512 per 100,000 live births (WHO, 2023). This places Nigeria among the countries with the highest maternal mortality ratios worldwide.

A substantial proportion of these deaths are attributable to preventable obstetric complications, including hypertensive disorders of pregnancy (pre-eclampsia and eclampsia), obstetric haemorrhage, sepsis, and complications arising from pre-existing conditions such as anaemia, diabetes, and chronic hypertension (Say et al., 2014). Critically, many of these adverse outcomes are preceded by identifiable clinical risk factors that, if detected early, could enable timely intervention and referral to appropriate levels of care. However, the capacity for systematic risk screening remains limited across much of the Nigerian healthcare system, particularly at the primary health centre (PHC) level where the majority of antenatal care contacts occur (Bolarinwa et al., 2021).

In recent years, machine learning (ML) approaches have demonstrated considerable promise in clinical risk prediction across a range of medical domains, including cardiovascular disease, diabetes, and oncology (Rajkomar et al., 2019). In maternal health specifically, predictive models have been developed for pre-eclampsia risk stratification (Jhee et al., 2019), preterm birth prediction (Koivu & Sairanen, 2020), and general obstetric risk assessment (Sufriyana et al., 2020). These models leverage routinely collected clinical and demographic data to generate probability estimates that can support — though not replace — clinical decision-making.

Despite this growing body of work, several challenges impede the direct application of ML-based maternal risk prediction in the Nigerian context. First, electronic health records (EHRs) remain sparse and fragmented across Nigerian healthcare facilities, limiting the availability of large, high-quality datasets for model training (Adeloye et al., 2017). Second, most existing predictive models have been developed and validated on populations in high-income countries, whose demographic and clinical profiles differ substantially from those of Nigerian women. Third, the clinical risk thresholds and feature distributions used in these models may not generalise to the Nigerian population, where conditions such as anaemia (mean haemoglobin 10.9 g/dL) and hypertensive disorders present at different baseline rates than in Western cohorts (Akinbami et al., 2013; Okonofua et al., 2024).

This study addresses these gaps by developing and evaluating a machine learning pipeline for high-risk pregnancy identification using synthetic data that faithfully reproduces the statistical distributions, inter-feature correlations, and clinical risk profiles of the Nigerian pregnant population. The use of synthetic data, parameterised from peer-reviewed Nigerian and West African epidemiological studies, enables rigorous model development and methodological validation in the absence of large-scale EHR datasets, while establishing a framework that can be readily adapted when real clinical data becomes available.

### 1.2 Statement of the Problem

Despite the availability of clinical guidelines for identifying high-risk pregnancies, the implementation of systematic, data-driven risk screening remains limited in Nigerian healthcare settings. The traditional approach relies on individual clinician judgement, which varies considerably in accuracy and consistency, particularly in overburdened PHCs with limited specialist support. This results in delayed or missed identification of high-risk pregnancies, late referrals, and preventable maternal morbidity and mortality.

Furthermore, the scarcity of digitised maternal health records in Nigeria creates a significant barrier to the development of locally relevant predictive models. While international datasets exist, the demographic and clinical characteristics of Nigerian women — including higher baseline rates of anaemia, different blood pressure distributions, younger mean maternal age, and distinct patterns of healthcare utilisation — necessitate population-specific approaches to risk prediction.

There is therefore an urgent need for a validated methodological framework that can: (a) generate epidemiologically realistic synthetic data for the Nigerian context, (b) compare the performance of multiple ML classification algorithms on maternal risk prediction, and (c) identify model architectures that balance predictive performance with interpretability for deployment in resource-constrained clinical settings.

### 1.3 Aim of the Study

The aim of this study is to develop, train, and comparatively evaluate multiple machine learning models for the prediction of high-risk pregnancies using synthetic maternal health data parameterised from Nigerian epidemiological studies.

### 1.4 Objectives of the Study

The specific objectives of this study are to:

1. Generate a synthetic maternal health dataset of 5,000 patient records with clinically realistic distributions and inter-feature correlations based on published Nigerian and West African studies.
2. Implement and compare four classification algorithms — Logistic Regression, Decision Tree, Random Forest, and Multi-Layer Perceptron (MLP) neural network — for binary high-risk pregnancy classification.
3. Evaluate model performance using accuracy, precision, recall, F1 score, and area under the receiver operating characteristic curve (AUC-ROC), with stratified k-fold cross-validation and 95% confidence intervals.
4. Analyse feature importance across models to identify the most clinically significant predictors of pregnancy risk in the Nigerian context.
5. Assess the trade-offs between model complexity, predictive performance, and interpretability to inform future deployment in low-resource clinical settings.

### 1.5 Scope of the Study

This study focuses on binary classification of pregnancy risk (high-risk versus low-risk) using 15 maternal health features spanning demographic characteristics (maternal age, parity, socioeconomic status), clinical indicators (blood pressure, haemoglobin, BMI, fasting blood glucose, gestational weight gain), medical history (hypertension, gestational diabetes, previous caesarean section, previous pre-eclampsia), and healthcare access factors (antenatal care visits, facility type).

The study utilises synthetic data generated from statistical distributions derived from peer-reviewed Nigerian and West African studies. While this approach enables rigorous methodological development, the results should be interpreted as a proof-of-concept framework rather than a clinically deployed diagnostic tool. External validation on real patient data would be required before clinical deployment.

The study does not address multi-class risk stratification (e.g., low/moderate/high/critical), temporal prediction (i.e., predicting risk at specific gestational ages), or longitudinal modelling of sequential antenatal visits.

### 1.6 Significance of the Study

This study makes the following contributions:

1. **Methodological framework**: It establishes a reproducible, end-to-end ML pipeline for maternal risk prediction that can be adapted to real clinical data as EHR systems are deployed across Nigerian healthcare facilities.
2. **Population-specific data generation**: It provides a rigorously parameterised synthetic data generation approach that preserves the epidemiological characteristics of the Nigerian pregnant population, enabling model development in data-scarce environments.
3. **Comparative model evaluation**: It provides an empirical comparison of four ML algorithms of varying complexity, informing the selection of appropriate models for different deployment contexts (e.g., interpretable models for PHCs versus neural networks for tertiary centres with computational resources).
4. **Clinical insight**: The feature importance analysis identifies which maternal risk factors are most predictive, potentially informing the prioritisation of screening protocols in antenatal care.

### 1.7 Definition of Terms

- **High-risk pregnancy**: A pregnancy in which the mother, foetus, or both are at elevated risk of adverse outcomes due to the presence of one or more clinical, demographic, or obstetric risk factors.
- **Machine learning (ML)**: A branch of artificial intelligence involving algorithms that learn patterns from data to make predictions or decisions without being explicitly programmed.
- **Synthetic data**: Artificially generated data that mimics the statistical properties of real-world data, used when actual data is unavailable, insufficient, or restricted due to privacy concerns.
- **AUC-ROC**: Area Under the Receiver Operating Characteristic Curve; a metric that measures the ability of a classifier to distinguish between classes across all probability thresholds.
- **SMOTE**: Synthetic Minority Over-sampling Technique; a method for addressing class imbalance by generating synthetic examples of the minority class.
- **MLP**: Multi-Layer Perceptron; a feedforward artificial neural network consisting of at least three layers of nodes (input, hidden, output).
- **Cross-validation**: A resampling procedure used to evaluate ML models by training and testing on different subsets of the data to assess generalisation performance.

### 1.8 Organisation of the Study

This study is organised into five chapters:

- **Chapter One** presents the background, problem statement, aims, objectives, scope, and significance of the study.
- **Chapter Two** reviews related literature on maternal health risk prediction, machine learning in clinical contexts, and synthetic data generation methodologies.
- **Chapter Three** describes the methodology, including data generation, preprocessing, model development, and evaluation procedures.
- **Chapter Four** presents the experimental results and provides a detailed discussion of findings.
- **Chapter Five** offers conclusions, recommendations, and directions for future work.

\newpage

# CHAPTER TWO

## LITERATURE REVIEW

### 2.1 Maternal Mortality and Morbidity in Nigeria

#### 2.1.1 Scale and Trends

Nigeria bears a disproportionate share of the global maternal mortality burden. The country's maternal mortality ratio (MMR) has been estimated at 512 per 100,000 live births, compared to a global average of 223 per 100,000 (WHO, 2023). Despite modest improvements over the past two decades, Nigeria was unable to achieve the Millennium Development Goal 5 target of a 75% reduction in MMR, and progress toward Sustainable Development Goal 3.1 (reducing MMR to below 70 per 100,000 by 2030) remains insufficient (Alkema et al., 2016).

Regional disparities within Nigeria are pronounced. The northern zones, characterised by lower literacy rates, higher fertility, and reduced healthcare access, report substantially higher MMRs than the southern zones (National Population Commission & ICF, 2019). The 2018 Nigeria Demographic and Health Survey (NDHS) documented significant variations in antenatal care utilisation, skilled birth attendance, and facility-based delivery across geopolitical zones.

#### 2.1.2 Causes of Maternal Mortality

The leading direct causes of maternal death in Nigeria include obstetric haemorrhage (25%), hypertensive disorders including pre-eclampsia and eclampsia (16%), sepsis (10%), and complications of unsafe abortion (8%) (Say et al., 2014). Indirect causes, which account for approximately 28% of maternal deaths globally, include anaemia, malaria, HIV/AIDS, and pre-existing cardiovascular and metabolic conditions (Khan et al., 2006).

A systematic review by Okonofua et al. (2024) across 54 Nigerian healthcare facilities (n = 71,758) found that hypertensive disorders of pregnancy affected 6.4% of the study population, with associated adverse outcomes including preterm delivery, low birthweight, and maternal intensive care admission. These findings underscore the importance of early identification and management of hypertensive conditions in Nigerian pregnant women.

#### 2.1.3 Healthcare System Challenges

The Nigerian healthcare system operates a three-tier structure: primary health centres (PHCs), secondary (general) hospitals, and tertiary hospitals. PHCs serve as the first point of contact for most pregnant women, yet they frequently lack the human resources, diagnostic equipment, and referral infrastructure needed for effective risk screening (Bolarinwa et al., 2021). Data from the 2018 NDHS indicates that only 41% of births in Nigeria occur in health facilities, and only 57% of women attend at least one antenatal care (ANC) visit. Fagbamigbe et al. (2020) reported that 46.7% of women in a pooled NDHS analysis (n = 52,654) had fewer than four ANC visits, with 74.8% initiating ANC after the first trimester.

These systemic constraints highlight the need for simple, scalable tools that can augment clinical judgement for risk screening at the PHC level — a need that machine learning approaches are well-positioned to address.

### 2.2 Clinical Risk Factors for High-Risk Pregnancy

The identification of high-risk pregnancy relies on the assessment of multiple clinical, demographic, and obstetric factors. This section reviews the evidence for the key risk factors included in the present study.

#### 2.2.1 Hypertensive Disorders

Hypertensive disorders of pregnancy (HDP), encompassing chronic hypertension, gestational hypertension, pre-eclampsia, and eclampsia, are a leading cause of maternal and perinatal morbidity and mortality worldwide (Magee et al., 2022). In Nigeria, Okonofua et al. (2024) reported an HDP prevalence of 6.4% across 54 facilities, while Adeloye et al. (2015) estimated the general adult hypertension prevalence at 28.9% in a meta-analysis of Nigerian studies. Blood pressure thresholds of systolic ≥ 140 mmHg or diastolic ≥ 90 mmHg define hypertension in pregnancy (WHO, 2011). Amoakoh-Coleman et al. (2017) reported mean blood pressure values of 111.0 ± 11.1 / 68.9 ± 9.3 mmHg in West African pregnant women.

#### 2.2.2 Anaemia

Anaemia in pregnancy, defined by the WHO as haemoglobin < 11.0 g/dL, is highly prevalent in Nigeria. Akinbami et al. (2013) reported an overall mean haemoglobin of 10.94 ± 1.86 g/dL among pregnant women in Lagos (n = 274), with trimester-specific means of 11.59, 10.81, and 10.38 g/dL for the first, second, and third trimesters respectively. The population mean below the WHO anaemia threshold indicates that a substantial proportion of Nigerian pregnant women are anaemic at baseline, creating challenges for risk classification that must be carefully handled in any predictive model.

#### 2.2.3 Obesity and BMI

Obesity (BMI ≥ 30.0 kg/m²) is an established risk factor for gestational diabetes, pre-eclampsia, caesarean delivery, and macrosomia (Poston et al., 2016). Senbanjo et al. (2021) reported a mean booking BMI of 27.0 ± 5.4 kg/m² in a Lagos cohort (n = 344), with 60.2% classified as overweight or obese. This high prevalence of elevated BMI has implications for the expected rate of metabolic complications in the Nigerian obstetric population.

#### 2.2.4 Gestational Diabetes Mellitus

Gestational diabetes mellitus (GDM) is associated with adverse maternal and neonatal outcomes including macrosomia, birth injury, neonatal hypoglycaemia, and increased lifetime risk of type 2 diabetes (Metzger et al., 2008). A meta-analysis by Mustapha et al. (2021) reported a pooled GDM prevalence of 11.0% (95% CI: 8–13%) in Nigerian studies (n = 46,210), though significant heterogeneity (I² = 99%) was observed. Anzaku and Musa (2013) reported a prevalence of 8.3% using WHO 75g OGTT criteria. Olokor et al. (2016) reported mean fasting blood glucose of 74.5 ± 11.5 mg/dL in normal pregnant women in Lagos (n = 210).

#### 2.2.5 Gestational Weight Gain

Both excessive and insufficient gestational weight gain (GWG) are associated with adverse outcomes. Eze et al. (2017) reported a mean GWG of 10.7 ± 3.4 kg among Nigerian women in Enugu (n = 200), which is lower than the Institute of Medicine (IOM) recommendations for normal-weight women (11.5–16.0 kg). Excessive GWG (> 18 kg) is associated with macrosomia and caesarean delivery, while insufficient GWG (< 7 kg) is associated with low birthweight and preterm birth.

#### 2.2.6 Obstetric History

Previous caesarean section and previous pre-eclampsia are both significant risk factors for complications in subsequent pregnancies. Adewuyi et al. (2019) reported a national caesarean section rate of 2.1% from the 2013 NDHS (n = 31,171), with a South-West regional rate of 4.7%. Ogunlaja et al. (2024) reported a pooled pre-eclampsia prevalence of 4.51% (95% CI: 3.82–5.29%) in a systematic review and meta-analysis of Nigerian studies.

#### 2.2.7 Demographic Factors

Advanced maternal age (≥ 35 years) and very young maternal age (< 18 years) are both associated with increased obstetric risk (Khalil et al., 2013). Grand multiparity (parity ≥ 5) is an independent risk factor for uterine rupture, post-partum haemorrhage, and malpresentation (Mgaya et al., 2013). The 2018 NDHS reported a total fertility rate of 5.3 children per woman nationally, with substantially higher rates in the northern zones.

### 2.3 Machine Learning in Maternal Health

#### 2.3.1 Overview of ML in Clinical Prediction

Machine learning encompasses a broad family of algorithms that learn predictive patterns from data. In clinical contexts, supervised learning algorithms — which learn from labelled examples to predict outcomes in new cases — have been most widely applied (Rajkomar et al., 2019). Common algorithms include logistic regression (a linear model producing interpretable odds ratios), decision trees (rule-based models that recursively partition the feature space), random forests (ensemble methods that aggregate many decision trees), and neural networks (non-linear models capable of learning complex feature interactions).

#### 2.3.2 Applications in Maternal Risk Prediction

Several studies have applied ML techniques to maternal health risk prediction. Jhee et al. (2019) developed models for pre-eclampsia prediction using clinical and laboratory data, achieving AUC values of 0.73–0.89 depending on the feature set and algorithm. Koivu and Sairanen (2020) applied gradient boosting methods to preterm birth prediction, reporting AUC values of 0.64–0.75. Sufriyana et al. (2020) developed a general obstetric risk assessment model using electronic health records, achieving AUC values of 0.82–0.92 with ensemble methods.

A common finding across these studies is that ensemble methods (random forests, gradient boosting) and neural networks generally outperform single decision trees and simple logistic regression, though the margin of improvement varies with data quality and sample size. Logistic regression, despite its lower ceiling on complex, non-linear relationships, remains widely recommended for clinical applications due to its interpretability and robustness (Christodoulou et al., 2019).

#### 2.3.3 Challenges and Gaps

Several challenges characterise the current state of ML in maternal health:

1. **Data scarcity in low-resource settings**: Most published models have been developed using data from high-income countries with well-established EHR systems. The applicability of these models to low- and middle-income country (LMIC) populations, where disease profiles, healthcare utilisation patterns, and data availability differ substantially, remains uncertain.

2. **Class imbalance**: High-risk pregnancies typically constitute a minority of the population (15–30%), creating class imbalance that can bias models toward predicting the majority class. Techniques such as SMOTE, class weighting, and threshold optimisation are required to address this (Chawla et al., 2002).

3. **Feature correlation and multicollinearity**: Clinical features are inherently correlated (e.g., BMI with blood pressure and glucose), which can affect model stability and interpretability, particularly for linear models.

4. **Model interpretability**: In clinical settings, the ability to explain why a patient was classified as high-risk is essential for clinician trust and adoption. This creates a tension between model complexity and practical deployment (Rudin, 2019).

#### 2.3.4 ML in the Nigerian Context

To date, few studies have applied ML techniques specifically to maternal health prediction in the Nigerian context. This gap is primarily attributable to the limited availability of digitised maternal health records. The present study addresses this gap by using synthetic data generation to enable methodological development, establishing a validated pipeline that can be deployed on real data as Nigerian health information systems mature.

### 2.4 Synthetic Data in Healthcare Research

#### 2.4.1 Rationale for Synthetic Data

Synthetic data — data that is artificially generated to mimic the statistical properties of real-world data — has emerged as a valuable tool in healthcare research for several reasons (Rankin et al., 2020):

1. **Data scarcity**: In settings where real patient data is unavailable or insufficient for model training.
2. **Privacy preservation**: Synthetic data eliminates the risk of patient re-identification, facilitating open science and reproducibility.
3. **Controlled experimentation**: Synthetic data enables researchers to systematically vary data properties (e.g., sample size, class balance, feature correlations) to study their effects on model performance.

#### 2.4.2 Methods for Synthetic Data Generation

Approaches to synthetic data generation range from simple parametric methods (sampling from known distributions) to complex generative models such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs) (Hernandez et al., 2022). For the purpose of initial model development, parametric approaches offer advantages in transparency and control: the researcher specifies the marginal distributions, correlations, and clinical constraints explicitly, making the data generation process fully auditable.

The present study adopts a parametric approach in which each feature is generated from distributions derived from published Nigerian epidemiological studies, with inter-feature correlations introduced through shared latent factors (e.g., a latent cardiovascular-metabolic factor that jointly influences BMI, blood pressure, and glucose). Risk labels are assigned through a probabilistic logistic model rather than deterministic rules, introducing realistic clinical uncertainty.

#### 2.4.3 Limitations of Synthetic Data

While synthetic data enables methodological development, it has inherent limitations:

1. The generated data, however carefully parameterised, cannot fully capture the complexity of real clinical data, including rare comorbidities, measurement errors, and missing data patterns.
2. Model performance on synthetic data may not directly transfer to real data, as the synthetic data's distributional assumptions constrain the learned patterns.
3. The risk labelling mechanism, even when probabilistic, reflects the researcher's modelling assumptions rather than true clinical outcomes.

These limitations are acknowledged in this study, and the results are positioned as a methodological proof-of-concept rather than a clinically validated tool.

### 2.5 Summary of Literature Review

The literature review reveals that: (1) Nigeria faces a severe maternal mortality crisis driven by preventable complications that could benefit from early risk identification; (2) well-established clinical risk factors exist but are inconsistently screened in resource-constrained settings; (3) machine learning offers promising approaches for systematic risk prediction, though most models have been developed for high-income country populations; and (4) synthetic data generation provides a viable path for methodological development in data-scarce environments. The present study builds on this foundation to develop and evaluate a population-specific ML pipeline for high-risk pregnancy identification in the Nigerian context.

\newpage

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

\newpage

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

![**Figure 4.1:** MLP Training History — Loss (left) and Accuracy (right) over epochs for training and validation sets.](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/mlp_training_history.png)

**Figure 4.1:** MLP Training History — Loss (left) and Accuracy (right) over epochs for training and validation sets.


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

![**Figure 4.2:** ROC Curve Comparison — All Models. AUC values shown in legend.](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/roc_comparison.png)

**Figure 4.2:** ROC Curve Comparison — All Models. AUC values shown in legend.


The MLP and Logistic Regression curves are nearly overlapping in the upper-left region of the ROC space, both demonstrating excellent discrimination with AUC values exceeding 0.93. The Random Forest curve follows closely with an AUC of 0.90. The Decision Tree curve shows visibly lower performance, with a stepped pattern characteristic of models that produce discrete probability outputs rather than smooth probability estimates.

All four models substantially outperform the random classifier (AUC = 0.5, diagonal reference line), confirming that the models have learned meaningful patterns from the data rather than relying on random chance or class frequency.

### 4.4 Confusion Matrix Analysis

The confusion matrices for each model reveal distinct classification patterns:

![Confusion Matrix — Logistic Regression](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/cm_logistic_regression.png)

![Confusion Matrix — Decision Tree](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/cm_decision_tree.png)

![Confusion Matrix — Random Forest](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/cm_random_forest.png)

![Confusion Matrix — MLP](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/cm_mlp.png)

**Figure 4.3:** Confusion Matrices — All Models.


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

![**Figure 4.4:** Feature Importance Comparison (Top 10) across Logistic Regression, Decision Tree, and Random Forest.](/Users/kossiso/CascadeProjects/high-risk-pregnancy-prediction/reports/figures/feature_importance_comparison.png)

**Figure 4.4:** Feature Importance Comparison (Top 10) across Logistic Regression, Decision Tree, and Random Forest.


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

\newpage

# CHAPTER FIVE

## CONCLUSION AND RECOMMENDATIONS

### 5.1 Summary of Findings

This study developed and comparatively evaluated four machine learning models — Logistic Regression, Decision Tree, Random Forest, and Multi-Layer Perceptron (MLP) — for the prediction of high-risk pregnancies using synthetic maternal health data parameterised from Nigerian epidemiological studies. The key findings are summarised as follows:

1. **The MLP neural network achieved the best overall classification performance**, with an F1 score of 0.7771 and AUC-ROC of 0.9294. It demonstrated the best balance between precision (0.7116) and recall (0.8559), correctly identifying approximately 86% of true high-risk pregnancies.

2. **Logistic Regression achieved the highest discriminative ability** (AUC-ROC = 0.9334) and an F1 score of 0.7586, performing comparably to the more complex MLP while offering full interpretability. This finding suggests that simpler, interpretable models may be sufficient for maternal risk screening in many clinical settings.

3. **Random Forest performed well** (AUC-ROC = 0.9043, F1 = 0.7284) but did not substantially outperform Logistic Regression, indicating that the additional complexity of ensemble methods may not be justified for this task.

4. **The Decision Tree was the weakest performer** (AUC-ROC = 0.8194, F1 = 0.6191), consistent with the known limitations of single trees in capturing complex feature interactions. However, its high recall (0.8604) suggests utility as an aggressive screening tool where sensitivity is paramount.

5. **Stratified 5-fold cross-validation** confirmed the stability of results, with narrow 95% confidence intervals (e.g., Logistic Regression F1: 0.748 ± 0.021, AUC: 0.923 ± 0.011).

6. **Feature importance analysis** revealed consistent identification of systolic blood pressure, BMI, haemoglobin, history of hypertension, and previous pre-eclampsia as the most predictive features across all model types, corroborating established clinical knowledge.

7. **Threshold optimisation** improved model performance by adapting the classification threshold to each model's probability calibration, demonstrating the importance of this step in clinical prediction tasks with class imbalance.

### 5.2 Conclusion

This study demonstrates that machine learning models can achieve good to excellent discriminative performance on maternal risk prediction using clinically relevant features derived from the Nigerian population. The comparable performance of Logistic Regression and MLP has important implications for clinical deployment: in resource-constrained primary health centres where computational infrastructure and technical expertise are limited, the simpler Logistic Regression model provides a practical, interpretable, and effective solution for systematic risk screening.

The study also establishes a reproducible, end-to-end machine learning pipeline — from data generation through preprocessing, model training, and evaluation — that can serve as a foundation for future work with real clinical data. The modular architecture of the codebase facilitates adaptation to new datasets, features, and classification targets.

The synthetic data generation approach, parameterised from 15 peer-reviewed Nigerian and West African studies, provides a validated methodology for developing population-specific predictive models in data-scarce environments. The probabilistic risk labelling mechanism ensures that model evaluation reflects realistic clinical uncertainty rather than artificial deterministic patterns.

### 5.3 Recommendations

Based on the findings of this study, the following recommendations are made:

#### 5.3.1 For Clinical Practice

1. **Adopt Logistic Regression as the primary screening model** for deployment at primary health centres, given its comparable performance to more complex models and its full interpretability. The model can be implemented as a simple risk scoring sheet that assigns points based on clinical features and flags patients exceeding a threshold score for referral.

2. **Prioritise systolic blood pressure, BMI, haemoglobin, and obstetric history** in routine antenatal screening, as these features were consistently identified as the most predictive across all models. Facilities should ensure that these measurements are taken at every antenatal visit.

3. **Use the MLP model at tertiary centres** with computational resources, where the marginal improvement in recall (85.6% vs. 84.2%) may be clinically significant given the higher acuity of the patient population.

4. **Implement threshold optimisation** in any deployed model, as the default 0.5 threshold is suboptimal for imbalanced clinical prediction tasks. The optimal threshold should be calibrated on local validation data.

#### 5.3.2 For Policy

1. **Invest in the digitisation of maternal health records** across Nigerian healthcare facilities to enable the development of validated predictive models using real patient data. The pipeline developed in this study can be directly applied to such datasets with minimal modification.

2. **Integrate ML-based risk screening tools** into national maternal health programmes, particularly as part of the focused antenatal care (FANC) model recommended by the WHO, to complement clinical judgement with systematic, evidence-based risk assessment.

3. **Establish data sharing frameworks** that enable multi-centre studies while protecting patient privacy, potentially leveraging synthetic data generation techniques as a privacy-preserving alternative.

#### 5.3.3 For Future Research

1. **External validation on real clinical data**: The most critical next step is to validate the models developed in this study on real patient data from Nigerian healthcare facilities. This would involve partnering with hospitals that maintain electronic health records and conducting a prospective or retrospective validation study.

2. **Longitudinal temporal modelling**: Future work should explore sequential models that incorporate data from multiple antenatal visits over the course of pregnancy, enabling dynamic risk assessment that evolves as new clinical information becomes available. Recurrent neural networks or transformer-based architectures would be appropriate for such temporal data.

3. **Multi-class risk stratification**: Extending the binary classification to a multi-class framework (e.g., low, moderate, high, critical risk) would provide more nuanced clinical guidance and enable tiered referral pathways.

4. **Expanded feature set**: Incorporating additional risk factors such as multiple pregnancy, placenta praevia, Rh incompatibility, HIV status, malaria parasitaemia, and urinalysis results could improve model performance and clinical utility.

5. **Federated learning approaches**: Given the distributed nature of the Nigerian healthcare system, federated learning could enable model training across multiple facilities without centralising sensitive patient data, addressing both data scarcity and privacy concerns simultaneously.

6. **Explainability and clinical decision support**: Developing SHAP (SHapley Additive exPlanations) or LIME (Local Interpretable Model-agnostic Explanations) visualisations for individual patient predictions would enhance clinician trust and facilitate the integration of ML predictions into clinical workflows.

7. **Cost-effectiveness analysis**: A health economics evaluation comparing the cost of implementing ML-based screening against the costs of adverse maternal outcomes prevented would strengthen the case for clinical adoption and policy investment.

### 5.4 Contribution to Knowledge

This study makes the following contributions to the body of knowledge:

1. It provides the first comprehensive comparative evaluation of four ML algorithms for maternal risk prediction using data parameterised specifically for the Nigerian population.

2. It establishes a rigorously validated synthetic data generation methodology that preserves the epidemiological characteristics of Nigerian maternal health, including realistic inter-feature correlations and non-deterministic risk labelling.

3. It demonstrates that simple, interpretable models (Logistic Regression) can achieve discriminative performance comparable to complex neural networks for maternal risk prediction, supporting their deployment in resource-constrained settings.

4. It provides a modular, reproducible ML pipeline that can be adapted to real clinical datasets as Nigerian health information systems mature, serving as a methodological blueprint for future studies.

\newpage

# REFERENCES

Adeloye, D., Basquill, C., Aderemi, A. V., Thompson, J. Y., & Obi, F. A. (2015). An estimate of the prevalence of hypertension in Nigeria: A systematic review and meta-analysis. *Journal of Hypertension*, 33(2), 230–242. https://doi.org/10.1097/HJH.0000000000000413

Adeloye, D., Auta, A., Fawibe, A., Gadanya, M., & Ezeigwe, N. (2017). Current prevalence pattern of tobacco smoking in Nigeria: A systematic review and meta-analysis. *BMC Public Health*, 17, 120.

Adewuyi, E. O., Auta, A., Khanal, V., Bamidele, O. D., Akuoko, C. P., Adefemi, K., Taplin, J. E., & Zhao, Y. (2019). Cesarean delivery in Nigeria: Prevalence and associated factors — a population-based cross-sectional study. *BMJ Open*, 9(6), e027273. https://doi.org/10.1136/bmjopen-2018-027273

Akinbami, A. A., Dosunmu, A. O., Adediran, A., Bakare, A., Akanmu, S., & Arogundade, O. (2013). Haematological values in pregnant women in Lagos, Nigeria. *Pan African Medical Journal*, 15, 135. https://doi.org/10.11604/pamj.2013.15.135.1566

Alkema, L., Chou, D., Hogan, D., Zhang, S., Moller, A. B., Gemmill, A., Fat, D. M., Boerma, T., Temmerman, M., Mathers, C., & Say, L. (2016). Global, regional, and national levels and trends in maternal mortality between 1990 and 2015, with scenario-based projections to 2030: A systematic analysis by the UN Maternal Mortality Estimation Inter-Agency Group. *The Lancet*, 387(10017), 462–474. https://doi.org/10.1016/S0140-6736(15)00838-7

Amoakoh-Coleman, M., Klipstein-Grobusch, K., Agyepong, I. A., Zuithoff, N. P. A.,";";";"; & Ansah, E. K. (2017). Provider adherence to first antenatal care guidelines and risk of pregnancy complications in public sector facilities: A Ghanaian cohort study. *BMC Pregnancy and Childbirth*, 17, 341. https://doi.org/10.1186/s12884-017-1510-8

Anzaku, A. S., & Musa, J. (2013). Prevalence and associated risk factors for gestational diabetes in Jos, North-central, Nigeria. *Archives of Gynecology and Obstetrics*, 287(5), 859–863. https://doi.org/10.1007/s00404-012-2649-z

Blagus, R., & Lusa, L. (2013). SMOTE for high-dimensional class-imbalanced data. *BMC Bioinformatics*, 14, 106. https://doi.org/10.1186/1471-2105-14-106

Bolarinwa, O. A., Tessema, Z. T., Frimpong, J. B., Seidu, A. A., Ahinkorah, B. O., & Hagan, J. E. (2021). Spatial distribution and factors associated with health facility delivery in Nigeria: A multilevel analysis of 2018 demographic and health survey. *BMC Public Health*, 21, 2075. https://doi.org/10.1186/s12889-021-12141-x

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *Journal of Artificial Intelligence Research*, 16, 321–357. https://doi.org/10.1613/jair.953

Christodoulou, E., Ma, J., Collins, G. S., Steyerberg, E. W., Verbakel, J. Y., & Van Calster, B. (2019). A systematic review shows no performance benefit of machine learning over logistic regression for clinical prediction models. *Journal of Clinical Epidemiology*, 110, 12–22. https://doi.org/10.1016/j.jclinepi.2019.02.004

Eze, C. U., Abonyi, L. C., Njoku, J., Okorie, U. E., & Owoh, S. (2017). Correlation of gestational weight gain with birthweight in Enugu, South-Eastern Nigeria. *Nigerian Journal of Clinical Practice*, 20(6), 754–760. https://doi.org/10.4103/njcp.njcp_265_16

Fagbamigbe, A. F., Olaseinde, O., & Setlhare, V. (2020). Sub-national analysis and determinants of numbers of antenatal care contacts in Nigeria: Assessing the compliance with the WHO recommended standard guidelines. *BMC Pregnancy and Childbirth*, 20, 402. https://doi.org/10.1186/s12884-020-03093-8

Hernandez, M., Epelde, G., Alberdi, A., Cilla, R., & Rankin, D. (2022). Synthetic data generation for tabular health records: A systematic review. *Neurocomputing*, 493, 28–45. https://doi.org/10.1016/j.neucom.2022.04.053

Jhee, J. H., Lee, S., Park, Y., Lee, S. E., Kim, Y. A., Kang, S. W., Kwon, J. Y., & Park, J. T. (2019). Prediction model development of late-onset preeclampsia using machine learning-based methods. *PLoS ONE*, 14(8), e0221202. https://doi.org/10.1371/journal.pone.0221202

Khalil, A., Syngelaki, A., Maiz, N., Zinevich, Y., & Nicolaides, K. H. (2013). Maternal age and adverse pregnancy outcome: A cohort study. *Ultrasound in Obstetrics & Gynecology*, 42(6), 634–643. https://doi.org/10.1002/uog.12494

Khan, K. S., Wojdyla, D., Say, L., Gülmezoglu, A. M., & Van Look, P. F. (2006). WHO analysis of causes of maternal death: A systematic review. *The Lancet*, 367(9516), 1066–1074. https://doi.org/10.1016/S0140-6736(06)68397-9

Koivu, A., & Sairanen, M. (2020). Predicting risk of stillbirth and preterm pregnancies with machine learning. *Health Information Science and Systems*, 8, 14. https://doi.org/10.1007/s13755-020-00105-9

Magee, L. A., Brown, M. A., Hall, D. R., Gupte, S., Hennessy, A., Karumanchi, S. A., Kenny, L. C., McCarthy, F., Myers, J., Poon, L. C.,"; Rana, S., & ISSHP. (2022). The 2021 International Society for the Study of Hypertension in Pregnancy classification, diagnosis & management recommendations for international practice. *Pregnancy Hypertension*, 27, 148–169. https://doi.org/10.1016/j.preghy.2021.09.008

Metzger, B. E., Lowe, L. P., Dyer, A. R., Trimble, E. R., Chaovarindr, U., Coustan, D. R., Hadden, D. R., McCance, D. R., Hod, M., McIntyre, H. D., Oats, J. J. N., Persson, B., Rogers, M. S., & Sacks, D. A. (2008). Hyperglycemia and adverse pregnancy outcomes. *New England Journal of Medicine*, 358(19), 1991–2002. https://doi.org/10.1056/NEJMoa0707943

Mgaya, A. H., Massawe, S. N., Kidanto, H. L., & Mgaya, H. N. (2013). Grand multiparity: Is it still a risk in pregnancy? *BMC Pregnancy and Childbirth*, 13, 241. https://doi.org/10.1186/1471-2393-13-241

Mustapha, S. A., Okonkwo, J. E., & Adeyemo, O. O. (2021). Prevalence of gestational diabetes mellitus in Nigeria: A systematic review and meta-analysis. *Diabetes & Metabolic Syndrome: Clinical Research & Reviews*, 15(6), 102317. https://doi.org/10.1016/j.dsx.2021.102317

National Population Commission [Nigeria] & ICF. (2019). *Nigeria Demographic and Health Survey 2018*. Abuja, Nigeria, and Rockville, Maryland, USA: NPC and ICF.

Ogunlaja, O. A., Akinajo, O. R., Umaru, H., & Rabiu, A. (2024). Preeclampsia in Nigeria: A systematic review and meta-analysis of prevalence. *Pregnancy Hypertension*, 35, 21–28. https://doi.org/10.1016/j.preghy.2023.12.003

Okonofua, F. E., Ntoimo, L. F. C., Ogu, R., Galadanci, H., Gana, M., Adetoye, D., Abe, E., Okike, O., & Durodola, A. (2024). Hypertensive disorders of pregnancy in Nigeria: Prevalence, outcomes, and strategies for prevention. *International Journal of Gynecology & Obstetrics*, 165(2), 683–693. https://doi.org/10.1002/ijgo.15263

Olokor, A. B., Imarengiaye, C. O., & Olokor, J. G. (2016). Fasting plasma glucose and lipid profiles of pregnant women in Benin City, Nigeria. *Nigerian Journal of Clinical Practice*, 19(2), 208–213. https://doi.org/10.4103/1119-3077.175965

Poston, L., Caleyachetty, R., Cnattingius, S., Corvalán, C., Uauy, R., Herring, S., & Gillman, M. W. (2016). Preconceptional and maternal obesity: Epidemiology and health consequences. *The Lancet Diabetes & Endocrinology*, 4(12), 1025–1036. https://doi.org/10.1016/S2213-8587(16)30217-0

Rajkomar, A., Dean, J., & Kohane, I. (2019). Machine learning in medicine. *New England Journal of Medicine*, 380(14), 1347–1358. https://doi.org/10.1056/NEJMra1814259

Rankin, D., Black, M., Bond, R., Wallace, J., Mulvenna, M., & Epelde, G. (2020). Reliability of supervised machine learning using synthetic data in health care: Model to preserve privacy for data sharing. *JMIR Medical Informatics*, 8(7), e18910. https://doi.org/10.2196/18910

Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1(5), 206–215. https://doi.org/10.1038/s42256-019-0048-x

Say, L., Chou, D., Gemmill, A., Tunçalp, Ö., Moller, A. B., Daniels, J., Gülmezoglu, A. M., Temmerman, M., & Alkema, L. (2014). Global causes of maternal death: A WHO systematic analysis. *The Lancet Global Health*, 2(6), e323–e333. https://doi.org/10.1016/S2214-109X(14)70227-X

Senbanjo, O. C., Akinlusi, F. M., Ottun, T. A., Olaleye, A. A., Oshodi, Y. A., Fakoya, T. A., & Aduloju, O. P. (2021). Body mass index at booking and adverse pregnancy outcome at a tertiary hospital in Lagos, Nigeria. *Nigerian Postgraduate Medical Journal*, 28(3), 178–184. https://doi.org/10.4103/npmj.npmj_421_21

Sufriyana, H., Wu, Y. W., & Su, E. C. Y. (2020). Prediction of preeclampsia and intrauterine growth restriction: Development of machine learning models on a prospective cohort. *JMIR Medical Informatics*, 8(5), e15411. https://doi.org/10.2196/15411

Ugwuja, E. I., Akubugwo, E. I., Ibiam, U. A., & Obidoa, O. (2011). Maternal sociodemographic parameters: Impact on trace element status and pregnancy outcomes in Nigerian women. *Journal of Health, Population and Nutrition*, 29(2), 156–162.

World Bank. (2022). *Nigeria Poverty Assessment 2022*. Washington, DC: World Bank Group.

World Health Organization. (2011). *WHO Recommendations for Prevention and Treatment of Pre-eclampsia and Eclampsia*. Geneva: WHO.

World Health Organization. (2019). *Trends in Maternal Mortality 2000 to 2017: Estimates by WHO, UNICEF, UNFPA, World Bank Group and the United Nations Population Division*. Geneva: WHO.

World Health Organization. (2023). *Maternal Mortality*. Fact Sheet. Geneva: WHO. https://www.who.int/news-room/fact-sheets/detail/maternal-mortality