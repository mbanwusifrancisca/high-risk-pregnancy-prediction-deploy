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
