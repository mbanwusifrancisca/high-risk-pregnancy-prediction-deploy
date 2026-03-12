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
