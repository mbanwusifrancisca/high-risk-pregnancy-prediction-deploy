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
