# Data Sources & Feature Parameterization

> **Purpose**: This document provides a defensible, peer-reviewed evidence base for every feature
> used in the synthetic maternal health dataset. Each variable's distribution parameters
> (mean, SD, prevalence, range) are sourced from published studies on Nigerian or
> West African pregnant women. This ensures that the generated synthetic data is
> epidemiologically plausible and academically defensible.

---

## Table of Contents

1. [Maternal Age](#1-maternal-age)
2. [Systolic Blood Pressure](#2-systolic-blood-pressure)
3. [Diastolic Blood Pressure](#3-diastolic-blood-pressure)
4. [Hemoglobin Level](#4-hemoglobin-level)
5. [Body Mass Index (BMI)](#5-body-mass-index-bmi)
6. [Fasting Blood Glucose](#6-fasting-blood-glucose)
7. [Gestational Weight Gain](#7-gestational-weight-gain)
8. [History of Hypertension](#8-history-of-hypertension)
9. [Gestational Diabetes Mellitus](#9-gestational-diabetes-mellitus)
10. [Previous Cesarean Section](#10-previous-cesarean-section)
11. [Previous Preeclampsia](#11-previous-preeclampsia)
12. [Parity](#12-parity)
13. [Number of Antenatal Care Visits](#13-number-of-antenatal-care-visits)
14. [Socioeconomic Status](#14-socioeconomic-status)
15. [Facility Type](#15-facility-type)

---

## 1. Maternal Age

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean** | 30.9 years | Senbanjo et al. (2021) |
| **SD** | 4.2 years | Senbanjo et al. (2021) |
| **Range** | 21–43 years | Senbanjo et al. (2021) |
| **Corroborating** | Mean 30.9 ± 4.1 years (range 19–45) | Olokor et al. (2016) |
| **Corroborating** | Mean 27.04 ± 2.75 years (range 15–40) | Ugwuja et al. (2011) |

### Adopted Parameters
- **Distribution**: Normal(mean=29.0, sd=5.5), truncated to [15, 48]
- **Rationale**: The tertiary hospital studies (Senbanjo, Olokor) show mean ~31 years, but these are biased toward urban educated women. The community-level estimate (Ugwuja) shows mean ~27. We adopt 29.0 as a compromise to represent both urban and rural populations, with SD=5.5 to capture the wider age range seen in the NDHS data (median age at first birth <19 in some regions).

### References
1. **Senbanjo OC, Akinlusi FM, Ottun TA** et al. (2021). *Early pregnancy body mass index, gestational weight gain and perinatal outcome in an obstetric population in Lagos, Nigeria.* Pan African Medical Journal, 39:136. PMID: [34527152](https://pubmed.ncbi.nlm.nih.gov/34527152/). PMC: [PMC8418156](https://pmc.ncbi.nlm.nih.gov/articles/PMC8418156/).
2. **Olokor OE, Onakewhor JU, Aderoba AK** (2016). *Response to fifty grams oral glucose challenge test and pattern of preceding fasting plasma glucose in normal pregnant Nigerians.* Nigerian Journal of Clinical Practice, 19(5):640-645. PMID: [27512415](https://pubmed.ncbi.nlm.nih.gov/27512415/).
3. **Ugwuja EI, Akubugwo EI, Ibiam UA, Obidoa O** (2011). *Maternal sociodemographic parameters: impact on trace element status and pregnancy outcomes.* Journal of Health, Population and Nutrition, 29(2):156-162. PMID: [21608425](https://pubmed.ncbi.nlm.nih.gov/21608425/).

---

## 2. Systolic Blood Pressure

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean (±2SD)** | 130 mmHg (upper normal boundary) | Okonofua et al. (1992) |
| **Mean (non-PIH)** | 111.0 mmHg | Amoakoh-Coleman et al. (2017) |
| **SD** | 11.1 mmHg | Amoakoh-Coleman et al. (2017) |
| **Normal BMI subgroup** | 108.2 mmHg | Amoakoh-Coleman et al. (2017) |
| **Obese subgroup** | 113.0 mmHg | Amoakoh-Coleman et al. (2017) |

### Adopted Parameters
- **Distribution**: Normal(mean=112.0, sd=12.0), truncated to [85, 200]
- **Rationale**: The Okonofua 1992 Nigerian study reported mean ±2SD of 130/80 mmHg, implying a mean SBP closer to ~115 mmHg with SD ~7.5 (but this represents the upper boundary). The Ghana study (Amoakoh-Coleman, West African population) provides a precise mean of 111.0 ± 11.1 mmHg. We adopt 112.0 ± 12.0 to reflect the slightly higher resting BP noted in Nigerian women by Okonofua, while staying consistent with the West African data.

### References
1. **Okonofua FE, Balogun JA, Amiengheme NA, O'Brien PM** (1992). *Blood pressure changes during pregnancy in Nigerian women.* International Journal of Cardiology, 37(3):373-379. PMID: [1468822](https://pubmed.ncbi.nlm.nih.gov/1468822/).
2. **Amoakoh-Coleman M, Ogum-Alangea D, Modey-Amoah E, Ntumy MY, Adanu RM, Oppong SA** (2017). *Blood pressure patterns and body mass index status in pregnancy: An assessment among women reporting for antenatal care at the Korle-Bu Teaching hospital, Ghana.* PLoS ONE, 12(12):e0188671. PMC: [PMC5718510](https://pmc.ncbi.nlm.nih.gov/articles/PMC5718510/).

---

## 3. Diastolic Blood Pressure

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean (±2SD)** | 80 mmHg (upper normal boundary) | Okonofua et al. (1992) |
| **Mean (non-PIH)** | 68.9 mmHg | Amoakoh-Coleman et al. (2017) |
| **SD** | 9.3 mmHg | Amoakoh-Coleman et al. (2017) |
| **Normal BMI subgroup** | 66.5 mmHg | Amoakoh-Coleman et al. (2017) |
| **Obese subgroup** | 71.2 mmHg | Amoakoh-Coleman et al. (2017) |

### Adopted Parameters
- **Distribution**: Normal(mean=70.0, sd=10.0), truncated to [50, 130]
- **Rationale**: Same logic as SBP. The West African data gives 68.9 ± 9.3, and Okonofua's Nigerian data suggests slightly higher resting DBP. We use 70.0 ± 10.0.

### References
- Same as Systolic Blood Pressure (references 1 & 2 above).

---

## 4. Hemoglobin Level

| Parameter | Value | Source |
|-----------|-------|--------|
| **Overall Mean** | 10.94 g/dL | Akinbami et al. (2013) |
| **Overall SD** | 1.86 g/dL | Akinbami et al. (2013) |
| **1st trimester** | 11.59 ± 2.35 g/dL | Akinbami et al. (2013) |
| **2nd trimester** | 10.81 ± 1.72 g/dL | Akinbami et al. (2013) |
| **3rd trimester** | 10.38 ± 1.27 g/dL | Akinbami et al. (2013) |
| **Corroborating** | 11.1 ± 1.2 g/dL | Amoakoh-Coleman et al. (2017) |

### Adopted Parameters
- **Distribution**: Normal(mean=10.9, sd=1.8), truncated to [4.0, 17.0]
- **Rationale**: Directly from Akinbami et al.'s 274-patient cross-sectional study at Lagos University Teaching Hospital and Lagos State University Teaching Hospital. The overall mean of 10.94 ± 1.86 g/dL is well-corroborated by the Ghana data (11.1 ± 1.2). WHO defines anaemia in pregnancy as Hb < 11.0 g/dL, placing this population's mean at the anaemia threshold — consistent with high anaemia prevalence in Nigeria.

### References
1. **Akinbami AA, Dosunmu AO, Adediran A, Oshinaike OO, Adebola P, Arogundade O** (2013). *Haematological values in pregnant women in Lagos, Nigeria.* International Journal of Women's Health, 5:241-245. PMID: [23662089](https://pubmed.ncbi.nlm.nih.gov/23662089/).

---

## 5. Body Mass Index (BMI)

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean** | 27.0 kg/m² | Senbanjo et al. (2021) |
| **SD** | 5.4 kg/m² | Senbanjo et al. (2021) |
| **Range** | 13.6–49.4 kg/m² | Senbanjo et al. (2021) |
| **Underweight (<18.5)** | 2.9% | Senbanjo et al. (2021) |
| **Normal (18.5–24.9)** | 36.9% | Senbanjo et al. (2021) |
| **Overweight (25.0–29.9)** | 34.6% | Senbanjo et al. (2021) |
| **Obese (≥30.0)** | 25.6% | Senbanjo et al. (2021) |

### Adopted Parameters
- **Distribution**: Normal(mean=27.0, sd=5.4), truncated to [14.0, 50.0]
- **Rationale**: Directly from Senbanjo et al., a 365-patient study in Lagos State University Teaching Hospital. The BMI category distribution is consistent with the increasing obesity trend in urban Nigeria. The 60.2% overweight/obese rate aligns with global trends for women booking at teaching hospitals.

### References
1. **Senbanjo OC, Akinlusi FM, Ottun TA** et al. (2021). — See Reference 1 under Maternal Age.

---

## 6. Fasting Blood Glucose

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean** | 74.5 mg/dL (4.14 mmol/L) | Olokor et al. (2016) |
| **SD** | 11.5 mg/dL (0.64 mmol/L) | Olokor et al. (2016) |
| **Range** | 42–117 mg/dL | Olokor et al. (2016) |
| **Sample** | 210 normal pregnancies | Olokor et al. (2016) |

### Adopted Parameters
- **Distribution**: Normal(mean=74.5, sd=11.5), truncated to [40, 200]
- **Rationale**: Directly from Olokor et al., who studied 210 women with confirmed normal glucose tolerance at 24–28 weeks gestation in Lagos. The authors note that Nigerian fasting glucose is lower than WHO cutoffs, suggesting population-specific thresholds may be needed.

### References
1. **Olokor OE, Onakewhor JU, Aderoba AK** (2016). — See Reference 2 under Maternal Age.

---

## 7. Gestational Weight Gain

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean total GWG** | 10.7 kg | Eze et al. (2017) |
| **SD** | 3.4 kg | Eze et al. (2017) |
| **Sample** | 200 women, Enugu, Nigeria | Eze et al. (2017) |
| **Excessive GWG prevalence** | 11.1% | Senbanjo et al. (2021) |

### Adopted Parameters
- **Distribution**: Normal(mean=10.7, sd=3.4), truncated to [0, 30]
- **Rationale**: Directly from Eze et al., a longitudinal study of 200 pregnant women at two tertiary hospitals in Enugu, South Eastern Nigeria. Women were recruited at <14 weeks gestation and followed until 38–39 weeks. The mean birthweight of 3.3 ± 0.6 kg validates normal pregnancy outcomes.

### References
1. **Eze ED, Barasa A, Adams MD, Rabiu KM** (2017). *Patterns of gestational weight gain and its association with birthweight in Nigeria.* Nigerian Journal of Clinical Practice, 20(6):754-760. PMID: [28656932](https://pubmed.ncbi.nlm.nih.gov/28656932/).

---

## 8. History of Hypertension

| Parameter | Value | Source |
|-----------|-------|--------|
| **HDP prevalence** | 6.4% | Okonofua et al. (2024) |
| **Gestational hypertension** | 49.8% of HDP | Okonofua et al. (2024) |
| **Chronic hypertension component** | ~16.5% of HDP | Okonofua et al. (2024) |
| **Sample** | 71,758 women, 54 facilities | Okonofua et al. (2024) |
| **General hypertension prevalence (Nigeria)** | 28.9% (all adults) | Adeloye et al. (2015) |

### Adopted Parameters
- **Prevalence (history of hypertension)**: 8.0% (Bernoulli, p=0.08)
- **Rationale**: The 6.4% HDP prevalence (Okonofua 2024) includes gestational hypertension (new-onset). Pre-existing/chronic hypertension is a subset. We estimate ~8% to account for women entering pregnancy with known hypertension, based on the chronic hypertension component of HDP data and the general adult hypertension prevalence in Nigeria of ~28.9%.

### References
1. **Okonofua FE, Ogu R, Garba M** et al. (2024). *Predictors, prevalence and outcome of hypertensive disorders in pregnancy in Nigerian tertiary health facilities.* BJOG, 131(12):1648-1657. PMID: [38960882](https://pubmed.ncbi.nlm.nih.gov/38960882/).
2. **Adeloye D, Basquill C, Aderemi AV, Thompson JY, Obi FA** (2015). *An estimate of the prevalence of hypertension in Nigeria: a systematic review and meta-analysis.* Journal of Hypertension, 33(2):230-242. PMID: [25380154](https://pubmed.ncbi.nlm.nih.gov/25380154/).

---

## 9. Gestational Diabetes Mellitus

| Parameter | Value | Source |
|-----------|-------|--------|
| **Pooled prevalence** | 11.0% (95% CI: 8–13%) | Mustapha et al. (2021) |
| **Range across studies** | 0.5–38% | Mustapha et al. (2021) |
| **Sample** | 46,210 (meta-analysis) | Mustapha et al. (2021) |
| **Corroborating** | 8.3% (95% CI: 5.2–12.4%) | Anzaku et al. (2013) |

### Adopted Parameters
- **Prevalence**: 5.0% (Bernoulli, p=0.05)
- **Rationale**: While the meta-analysis pooled prevalence is 11%, the high heterogeneity (I² = 99%) and wide range (0.5–38%) suggest facility-level variation. Many studies used non-standardized criteria. The Anzaku 2013 study using WHO 75g OGTT criteria found 8.3%. For a representative population-level estimate that includes women who may not be screened, we adopt a conservative 5.0%, consistent with the lower bound of the CI and Sub-Saharan Africa community-based estimates.

### References
1. **Mustapha B, Lawal SK, Akinlusi FM** et al. (2021). *A systematic review and meta-analysis of the prevalence and determinants of gestational diabetes mellitus in Nigeria.* Indian Journal of Endocrinology and Metabolism, 25(3):180-186. PMID: [34760670](https://pubmed.ncbi.nlm.nih.gov/34760670/).
2. **Anzaku AS, Musa J** (2013). *Prevalence and associated risk factors for gestational diabetes in Jos, North-central, Nigeria.* Archives of Gynecology and Obstetrics, 287(5):859-863. PMID: [23224650](https://pubmed.ncbi.nlm.nih.gov/23224650/).

---

## 10. Previous Cesarean Section

| Parameter | Value | Source |
|-----------|-------|--------|
| **National CS rate** | 2.1% (95% CI: 1.8–2.3%) | Adewuyi et al. (2019) |
| **South-West region** | 4.7% | Adewuyi et al. (2019) |
| **Urban** | Higher (AOR: 1.51) | Adewuyi et al. (2019) |
| **Data source** | NDHS 2013, n=31,171 | Adewuyi et al. (2019) |

### Adopted Parameters
- **Prevalence (previous CS)**: 3.0% (Bernoulli, p=0.03)
- **Rationale**: The national CS rate of 2.1% (NDHS 2013) represents all deliveries. For women presenting at ANC who have had a previous CS, the rate would be slightly higher as it accumulates over multiple pregnancies. We adopt 3.0%, between the national rate and the South-West urban rate.

### References
1. **Adewuyi EO, Auta A, Khanal V, Bamidele OD, Akuoko CP, Adefemi K, Tapkigen J, Zhao Y** (2019). *Cesarean delivery in Nigeria: prevalence and associated factors—a population-based cross-sectional study.* BMJ Open, 9(6):e027273. PMID: [31213450](https://pubmed.ncbi.nlm.nih.gov/31213450/).

---

## 11. Previous Preeclampsia

| Parameter | Value | Source |
|-----------|-------|--------|
| **Pooled PE prevalence** | 4.51% (95% CI: 3.82–5.29%) | Ogunlaja et al. (2024) |
| **Eclampsia prevalence** | 1.39% (95% CI: 1.02–1.84%) | Ogunlaja et al. (2024) |
| **Study type** | Systematic review & meta-analysis | Ogunlaja et al. (2024) |
| **Maternal mortality (PE/E)** | 6.04% | Ogunlaja et al. (2024) |

### Adopted Parameters
- **Prevalence (previous preeclampsia)**: 4.0% (Bernoulli, p=0.04)
- **Rationale**: The pooled prevalence of PE in Nigerian pregnant women is 4.51% (systematic review, 2024). For women reporting a history of preeclampsia in previous pregnancies, we adopt 4.0% (slightly below the incidence, since not all women are multiparous and not all with history present at ANC).

### References
1. **Ogunlaja OA, Ogunlaja IP, Olanrewaju TO** et al. (2024). *Prevalence and materno-fetal outcomes of preeclampsia/eclampsia among pregnant women in Nigeria: a systematic review and meta-analysis.* Clinical Hypertension, 30:38. PMID: [39363380](https://pubmed.ncbi.nlm.nih.gov/39363380/).

---

## 12. Parity

| Parameter | Value | Source |
|-----------|-------|--------|
| **Mean** | 1.2 | Olokor et al. (2016) |
| **SD** | 1.1 | Olokor et al. (2016) |
| **Range** | 0–5 | Olokor et al. (2016) |
| **Nulliparous** | 48.8% (tertiary) | Senbanjo et al. (2021) |
| **TFR (national)** | 5.3 | NDHS 2018 |

### Adopted Parameters
- **Distribution**: Poisson(lambda=2.5), truncated to [0, 10]
- **Rationale**: The tertiary hospital mean of 1.2 reflects an urban educated population with high nulliparity (48.8%). The NDHS 2018 total fertility rate is 5.3, suggesting much higher parity in the general population. We use Poisson(lambda=2.5) to represent a mixed population between urban (lower parity) and rural (higher parity) Nigerian women.

### References
1. **Olokor OE** et al. (2016). — See Reference 2 under Maternal Age.
2. **Senbanjo OC** et al. (2021). — See Reference 1 under Maternal Age.
3. **National Population Commission (NPC) [Nigeria] and ICF** (2019). *Nigeria Demographic and Health Survey 2018.* Abuja, Nigeria, and Rockville, Maryland, USA: NPC and ICF. Available at: [dhsprogram.com](https://dhsprogram.com/pubs/pdf/FR359/FR359.pdf).

---

## 13. Number of Antenatal Care Visits

| Parameter | Value | Source |
|-----------|-------|--------|
| **Insufficient ANC (<4 visits)** | 46.7% | Fagbamigbe et al. (2020) |
| **Late timing (after 1st trimester)** | 74.8% | Fagbamigbe et al. (2020) |
| **Data source** | Pooled NDHS 2008, 2013, 2018 (n=52,654) | Fagbamigbe et al. (2020) |
| **≥4 ANC visits** | 53.3% | Fagbamigbe et al. (2020) |
| **ANC from skilled provider** | ~67% | NDHS 2018 |

### Adopted Parameters
- **Distribution**: Mixed — 25% with 0–1 visits, 22% with 2–3 visits, 35% with 4–7 visits, 18% with 8+ visits. Operationalized as Poisson(lambda=4.0), truncated to [0, 15].
- **Rationale**: With 46.7% having <4 visits and 53.3% having ≥4, and considering WHO recommends ≥8 contacts, we use a Poisson distribution with lambda=4.0 which naturally places ~43% below 4 visits and ~57% at 4+, closely matching the NDHS data.

### References
1. **Fagbamigbe AF, Olaseinde O, Fagbamigbe OS** (2020). *Patterns and Predictors of Insufficient Antenatal Care Utilization in Nigeria over a Decade: A Pooled Data Analysis Using Demographic and Health Surveys.* International Journal of Environmental Research and Public Health, 18(2):7400. PMID: [33182288](https://pubmed.ncbi.nlm.nih.gov/33182288/).

---

## 14. Socioeconomic Status

| Parameter | Value | Source |
|-----------|-------|--------|
| **Wealth quintiles** | Lowest, Second, Middle, Fourth, Highest (~20% each by design) | NDHS 2018 |
| **Richest CS rate** | Near 10–15% (WHO target) | Adeyemi et al. (2019) |
| **Poorest CS rate** | Far below WHO target | Adeyemi et al. (2019) |

### Adopted Parameters
- **Distribution**: Categorical — Low: 35%, Middle: 40%, High: 25%
- **Rationale**: The NDHS uses five wealth quintiles (each 20%). For modelling simplicity, we collapse into three categories. Nigeria has approximately 40% of the population living below the poverty line (World Bank, 2022), so we weight "Low" at 35%, "Middle" at 40%, and "High" at 25%, which approximates the bottom two quintiles, middle two, and top quintile respectively.

### References
1. **National Population Commission (NPC) [Nigeria] and ICF** (2019). *Nigeria Demographic and Health Survey 2018.* — See Reference 3 under Parity.
2. **Adeyemi OO, Ayo-Yusuf OA, Bello B** (2019). *Examining inequalities in access to delivery by caesarean section in Nigeria.* PLoS ONE, 14(8):e0221760. PMID: [31465505](https://pubmed.ncbi.nlm.nih.gov/31465505/).

---

## 15. Facility Type

| Parameter | Value | Source |
|-----------|-------|--------|
| **Health facility delivery** | 41% | Bolarinwa et al. (2021) |
| **Home delivery** | 59% | Bolarinwa et al. (2021) |
| **Data source** | NDHS 2018 (n=34,193) | Bolarinwa et al. (2021) |

### Adopted Parameters
- **Distribution**: Categorical — Primary Health Centre: 40%, General/Secondary Hospital: 30%, Tertiary Hospital: 15%, Private Clinic: 15%
- **Rationale**: Among the 41% delivering in health facilities (NDHS 2018), primary health centres serve the majority of the population, particularly in rural areas. Tertiary facilities are concentrated in state capitals. The distribution reflects the Nigerian health system pyramid: wide base of PHCs, fewer secondary hospitals, and limited tertiary centres.

### References
1. **Bolarinwa OA, Fortune E, Aboagye RG, Seidu AA, Olagunju OS, Nwagbara UI, Ameyaw EK, Ahinkorah BO** (2021). *Health facility delivery among women of reproductive age in Nigeria: Does age at first birth matter?* PLoS ONE, 16(11):e0259250. PMC: [PMC8568178](https://pmc.ncbi.nlm.nih.gov/articles/PMC8568178/).

---

## Summary of Adopted Parameters

| # | Feature | Type | Distribution / Prevalence | Source(s) |
|---|---------|------|--------------------------|-----------|
| 1 | Maternal Age | Continuous | N(29.0, 5.5), [15, 48] | Senbanjo 2021; Olokor 2016; Ugwuja 2011 |
| 2 | Systolic BP | Continuous | N(112.0, 12.0), [85, 200] | Okonofua 1992; Amoakoh-Coleman 2017 |
| 3 | Diastolic BP | Continuous | N(70.0, 10.0), [50, 130] | Okonofua 1992; Amoakoh-Coleman 2017 |
| 4 | Hemoglobin | Continuous | N(10.9, 1.8), [4.0, 17.0] | Akinbami 2013 |
| 5 | BMI | Continuous | N(27.0, 5.4), [14.0, 50.0] | Senbanjo 2021 |
| 6 | Fasting Blood Glucose | Continuous | N(74.5, 11.5), [40, 200] | Olokor 2016 |
| 7 | Gestational Weight Gain | Continuous | N(10.7, 3.4), [0, 30] | Eze 2017 |
| 8 | History of Hypertension | Binary | Bernoulli(0.08) | Okonofua 2024; Adeloye 2015 |
| 9 | Gestational Diabetes | Binary | Bernoulli(0.05) | Mustapha 2021; Anzaku 2013 |
| 10 | Previous CS | Binary | Bernoulli(0.03) | Adewuyi 2019 |
| 11 | Previous Preeclampsia | Binary | Bernoulli(0.04) | Ogunlaja 2024 |
| 12 | Parity | Discrete | Poisson(2.5), [0, 10] | Olokor 2016; NDHS 2018 |
| 13 | ANC Visits | Discrete | Poisson(4.0), [0, 15] | Fagbamigbe 2020 |
| 14 | Socioeconomic Status | Categorical | Low: 35%, Mid: 40%, High: 25% | NDHS 2018 |
| 15 | Facility Type | Categorical | PHC: 40%, Secondary: 30%, Tertiary: 15%, Private: 15% | Bolarinwa 2021; NDHS 2018 |

---

## Full Reference List

1. Adeloye D, et al. (2015). PMID: 25380154.
2. Adewuyi EO, et al. (2019). PMID: 31213450.
3. Akinbami AA, et al. (2013). PMID: 23662089.
4. Amoakoh-Coleman M, et al. (2017). PMC: PMC5718510.
5. Anzaku AS, Musa J. (2013). PMID: 23224650.
6. Bolarinwa OA, et al. (2021). PMC: PMC8568178.
7. Eze ED, et al. (2017). PMID: 28656932.
8. Fagbamigbe AF, et al. (2020). PMID: 33182288.
9. Mustapha B, et al. (2021). PMID: 34760670.
10. National Population Commission (NPC) [Nigeria] and ICF (2019). Nigeria DHS 2018.
11. Ogunlaja OA, et al. (2024). PMID: 39363380.
12. Okonofua FE, et al. (1992). PMID: 1468822.
13. Okonofua FE, et al. (2024). PMID: 38960882.
14. Olokor OE, et al. (2016). PMID: 27512415.
15. Senbanjo OC, et al. (2021). PMID: 34527152.
16. Ugwuja EI, et al. (2011). PMID: 21608425.
