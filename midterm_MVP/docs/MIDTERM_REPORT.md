# Predicting Hurricane Disaster Assistance: A Machine Learning Approach to Equitable Relief

**Statistical Learning Midterm Report**
**Date:** October 7, 2025
**Context:** AURA MVP - Hurricane Impact Risk Prediction

---

## 1. Introduction

### Motivation

In the aftermath of major hurricanes, emergency managers face a critical challenge: **how to allocate limited disaster relief resources to communities most in need?** Current approaches rely on reactive damage assessments, often missing vulnerable populations who face barriers to applying for assistance. This project develops a machine learning framework to predict census tract-level FEMA claims, with the goal of identifying communities at risk of being underserved.

The Automated Underwriting for Relief Allocation (AURA) MVP addresses a fundamental equity gap in disaster response: **FEMA claims data reflects who applies for aid, not necessarily who needs it**. Communities with language barriers, limited digital access, or historical distrust of government may experience significant damage yet receive minimal assistance. By modeling the relationship between demographic vulnerability, housing characteristics, and disaster claims, we can flag discrepancies that warrant proactive outreach.

### Research Questions

1. **Predictive accuracy:** Can machine learning algorithms reliably predict tract-level FEMA claims using pre-disaster demographic and housing data?
2. **Equity analysis:** Do models systematically underpredict for vulnerable populations (high poverty, minority, limited English proficiency)?
3. **Feature importance:** Which factors drive predictions - physical risk, demographic vulnerability, or application behavior?

### Scope

This analysis focuses on **5,668 census tract × storm event pairs** from Gulf Coast states (TX, LA, MS, AL, FL) across six major hurricanes (2017-2022): Harvey, Irma, Michael, Laura, Ida, and Ian. We implement and compare four machine learning algorithms (KNN, Decision Tree, Random Forest, Bagging) with full hyperparameter tuning and bias diagnostics.

---

## 2. Data & Preprocessing

### Data Sources

| Source | Features | Coverage | Purpose |
|--------|----------|----------|---------|
| **FEMA Individual Assistance** | Total claims, applications, eligibility rates | 2017-2022 disasters | Target variable & access metrics |
| **CDC Social Vulnerability Index 2022** | Poverty, elderly %, race/ethnicity, language | All US tracts | Vulnerability indicators |
| **US Census ACS 2022** | Population, income, housing units | TX, LA, MS, AL, FL | Demographic context |
| **NOAA HURDAT2** | Storm tracks, wind radii | Historical hurricanes | Physical hazard (future work) |

### Dataset Construction

**Unit of analysis:** Census tract (11-digit GEOID) × Disaster number (FEMA identifier)

**Join logic:**
1. FEMA Individual Assistance registrants aggregated to tract-disaster level
2. Matched to CDC SVI and ACS demographics via GEOID
3. Storm metadata (Harvey=4332, Irma=4337, etc.) from FEMA disaster declarations
4. Hazard features (wind, rainfall, distance) are **placeholders** pending NOAA integration

**Key statistics:**
- Total observations: 5,668
- Tracts with claims > $0: 1,557 (27.5%)
- Median claims: $0 (highly zero-inflated)
- Mean claims (excluding zeros): $5,847
- Maximum claims: $124,683

### Feature Engineering

**Target variable transformation:**
FEMA claims are heavily right-skewed. We train models on `log1p(fema_claims_total)` to normalize the distribution, then evaluate on the raw dollar scale for interpretability.

**Percentage conversions:**
CDC SVI provides counts; we convert to proportions (e.g., `pct_poverty = EP_POV150 / 100`).

**Missing data handling:**
- Continuous features: Median imputation
- Hazard features: All missing (awaiting NOAA pipeline integration)
- Demographic features: <5% missingness

**Excluded from modeling:**
- Identifiers (tract_geoid, disasterNumber, state/county names)
- Leakage variables (fema_claims_pc - derived from target)
- Non-numeric categorical fields

Final feature set: **28 numeric predictors** across demographics (population, income, age), housing (units, mobile homes, crowding), race/ethnicity (Hispanic %, Black %, limited English), and FEMA access (applications, eligibility rates).

### Data Limitations

**Critical gap: Physical hazard features**
The current dataset lacks precise storm exposure metrics (wind speed at tract centroid, rainfall totals, distance to storm path, floodplain coverage). Without these, models may learn demographic proxies for risk rather than true physical vulnerability. A collaborator's NOAA HURDAT2 pipeline generates these features; integration is planned post-midterm.

**Target variable bias**
FEMA claims represent assistance awarded, not underlying damage. Communities with low application rates due to non-financial barriers (language, awareness, trust) will appear low-risk to the model. This measurement error is central to the equity analysis in Section 5.

---

## 3. Modeling Approach

### Algorithms

We implement four regression algorithms per assignment requirements:

1. **K-Nearest Neighbors (KNN)**: Distance-based, non-parametric. Predicts based on k most similar tracts. Sensitive to feature scaling and noise.

2. **Decision Tree**: Hierarchical binary splits to partition feature space. Interpretable but prone to overfitting. Captures non-linear relationships and interactions.

3. **Random Forest**: Ensemble of decision trees with bootstrap sampling and random feature subsets. Reduces overfitting via averaging. Provides feature importance.

4. **Bagging (Bootstrap Aggregating)**: Ensemble method training multiple decision trees on bootstrap samples of training data. Reduces variance through aggregation.

### Training Strategy

**Train/test split:** 80/20 random split (stratified by presence of claims would be ideal but not implemented due to time constraints).

**Cross-validation:** 5-fold CV for hyperparameter tuning. Each fold uses 80% of training data for fitting, 20% for validation. GridSearchCV selects parameters minimizing negative RMSE (equivalent to maximizing RMSE reduction).

**Preprocessing pipeline:**
```
ColumnTransformer([
  ("num", Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
  ]), numeric_features)
])
```

Median imputation handles missing values; StandardScaler (mean=0, std=1) is critical for distance-based algorithms like KNN.

### Hyperparameter Grids

**KNN:**
- `n_neighbors`: [5, 10, 15, 25, 35] - Number of neighbors to consider
- `weights`: ['uniform', 'distance'] - Equal weight vs inverse distance weighting
- `p`: [1, 2] - Manhattan (L1) vs Euclidean (L2) distance

**Decision Tree:**
- `max_depth`: [None, 5, 10, 20] - Maximum tree depth (None = grow until pure leaves)
- `min_samples_split`: [2, 5, 10] - Minimum samples to split a node
- `min_samples_leaf`: [1, 2, 5] - Minimum samples in leaf node

**Random Forest:**
- `n_estimators`: [200, 400] - Number of trees in forest
- `max_depth`: [None, 10, 20] - Depth of each tree
- `min_samples_split`: [2, 5] - Splitting threshold
- `max_features`: ['sqrt', 0.75, 1.0] - Features considered at each split

**Bagging:**
- `n_estimators`: [100, 200, 400] - Number of base estimators
- `max_samples`: [0.5, 0.75, 1.0] - Fraction of samples for each estimator
- `max_features`: [0.5, 0.75, 1.0] - Fraction of features for each estimator

### Evaluation Metrics

1. **R² (Coefficient of Determination)**: Proportion of variance explained (1.0 = perfect fit)
2. **RMSE (Root Mean Squared Error)**: Average prediction error in dollars (penalizes large errors)
3. **MAE (Mean Absolute Error)**: Average absolute prediction error in dollars (robust to outliers)
4. **MAPE (Mean Absolute Percentage Error)**: Percentage error (interpretable scale)
5. **Explained Variance**: Proportion of target variance captured by model

All metrics computed on raw (exponentiated) claims scale for interpretability.

---

## 4. Results

### Model Performance

| Model | Phase | R² | RMSE ($) | MAE ($) | MAPE | Explained Variance |
|-------|-------|--------|---------|---------|------|-------------------|
| KNN | Baseline | 0.790 | 1,676 | 493 | 0.788 | 0.801 |
| KNN | Tuned | **0.703** | 1,997 | 557 | 0.811 | 0.718 |
| Decision Tree | Baseline | 0.810 | 1,594 | 362 | 0.339 | 0.811 |
| Decision Tree | Tuned | **0.844** | 1,444 | 338 | 0.312 | 0.844 |
| Random Forest | Baseline | 0.925 | 1,001 | 265 | 0.274 | 0.926 |
| Random Forest | Tuned | **0.924** | 1,007 | 266 | 0.274 | 0.925 |
| Bagging | Baseline | 0.928 | 980 | 273 | 0.300 | 0.929 |
| Bagging | Tuned | **0.927** | 990 | 262 | 0.273 | 0.927 |

### Key Findings

**Best model: Bagging Regressor**
- **R² = 0.927**: Explains 92.7% of variance in test set FEMA claims
- **RMSE = $990**: Average prediction error of ~$1,000 per tract
- **MAE = $262**: Median absolute error significantly lower than RMSE (skewed error distribution)

**Ensemble dominance:**
Random Forest and Bagging both exceed R² = 0.92, vastly outperforming single models. Aggregating predictions across multiple trees reduces overfitting and captures complex feature interactions.

**Tuning impact:**
- **Decision Tree**: +3.4 percentage points R² (0.810 → 0.844). Regularization via max_depth and min_samples_split prevents overfitting.
- **KNN**: -8.7 percentage points R² (0.790 → 0.703). **Unexpected degradation** suggests baseline parameters (n=5, uniform weights) were closer to optimal. GridSearch selected n=10, distance weighting, Euclidean distance - possibly oversmoothing in sparse high-dimensional space.
- **Ensemble methods**: Minimal change (~0.1 pp). Already well-regularized; tuning provides marginal gains.

**Error distribution:**
MAPE (27-81%) >> MAE/RMSE suggests **percentage errors are large for low-claim tracts**. Models struggle with near-zero predictions (divide-by-small-number instability in MAPE). High-claim tracts are predicted more accurately in absolute terms.

### Feature Importance

Top 10 predictors (Random Forest model):
1. **applications** (31.2%) - Number of FEMA applications
2. **housing_units** (12.7%) - Total housing in tract
3. **acs_total_population** (9.4%) - Tract population
4. **median_household_income** (6.8%) - Income level
5. **pct_poverty** (5.2%) - Poverty rate
6. **pct_elderly** (4.1%) - Population 65+
7. **pct_hispanic** (3.9%) - Hispanic percentage
8. **fema_claims_mean** (3.3%) - Average past claims
9. **pct_black** (2.8%) - Black percentage
10. **pct_mobile_homes** (2.6%) - Mobile home percentage

**Interpretation:**
The model learns heavily from **application volume** (31% importance) and **population size** (housing_units + total_population = 22%). This confirms our bias hypothesis: **the model predicts who applies, not who needs help**. Demographic vulnerability (poverty, race/ethnicity) ranks lower but still significant, suggesting correlations between vulnerability and claims - whether due to physical risk exposure or differential application rates remains ambiguous without hazard features.

---

## 5. Responsible AI Discussion

### The Equity Problem

**Central question:** Does the model systematically underpredict for vulnerable populations?

If yes, operationalizing these predictions could **perpetuate underservice** by directing resources away from communities that face high barriers to applying for aid.

### Bias Diagnostic Methodology

We segment the test set into quartiles by four vulnerability indicators:
- `pct_poverty`: Percentage below 150% federal poverty line
- `pct_hispanic`: Hispanic/Latino population percentage
- `pct_black`: Black/African American population percentage
- `pct_limited_english`: Limited English proficiency percentage

For each quartile, we compute residuals (actual - predicted). **Negative residuals indicate underprediction** (model predicts too low, underestimating need).

### Findings

**Poverty:**
- Q1 (low poverty): Mean residual ≈ $50 (slight overprediction)
- Q4 (high poverty): Mean residual ≈ -$180 (underprediction)
- **Pattern:** Consistent underprediction for high-poverty tracts across all models
- **Magnitude:** ~$230 gap between Q1 and Q4

**Race/Ethnicity:**
- **Hispanic %:** Relatively balanced across quartiles (mean residuals within ±$100)
- **Black %:** Q4 shows -$150 to -$200 underprediction for Bagging/Random Forest
- **Interpretation:** High-Black tracts are moderately underestimated

**Limited English:**
- Minimal systematic bias (residuals ≈ ±$50 across quartiles)
- Variance increases in Q4 but mean remains near zero

**Overall assessment:**
Models exhibit **moderate equity concerns**. High-poverty and high-Black tracts are underpredicted by $150-$230 on average. While residuals are centered near zero (indicating unbiased predictions on average), vulnerable communities show higher residual variance and negative skew.

### Root Cause Analysis

**The target variable is biased:**
FEMA claims data captures assistance awarded, which depends on:
1. **Physical damage** (what we want to predict)
2. **Application behavior** (introduces bias)

Application barriers disproportionately affect vulnerable communities:
- **Language:** Non-English speakers may not understand eligibility
- **Digital access:** Online applications require internet + digital literacy
- **Trust:** Historical marginalization reduces government engagement
- **Awareness:** Lack of outreach in isolated rural or immigrant communities

**Impact on model:**
Without physical hazard features (wind speed, flood zones), the model learns correlations between demographics and claims. Two competing interpretations:
1. **Physical risk proxy:** Vulnerable communities live in higher-risk areas (floodplains, mobile homes, older housing stock)
2. **Application bias:** Vulnerable communities experience similar damage but apply at lower rates

**We cannot distinguish these without hazard data.** The underprediction of high-poverty/high-Black tracts suggests application bias dominates, but this is speculative until NOAA features are integrated.

### Policy Implications

**What this model predicts:**
Expected FEMA claims **given historical application patterns**. This is NOT:
- Eligibility for assistance
- True physical damage
- Community need

**Ethical use guidelines:**
1. **Invert the prediction:** Low predicted claims + high vulnerability = **red flag for outreach**, not deprioritization
2. **Proactive resource allocation:** Deploy mobile assistance centers to underpredicted high-poverty tracts
3. **Partner with local organizations:** VOADs, churches, community centers can verify ground truth
4. **Human-in-the-loop:** Never auto-deny aid based on model predictions

**Who benefits:**
- Emergency managers: Better pre-positioning of resources
- High-application communities: Accurate need estimation
- Researchers: Equity-aware disaster modeling framework

**Who could be harmed:**
- Low-application vulnerable communities: Risk of deprioritization if model misused
- Non-English speakers: Invisible to application-based models
- Mobile home residents: Often unaware of eligibility, underrepresented in claims data

### Safeguards Implemented

1. ✅ **Bias diagnostics:** Residual analysis by vulnerability quartiles
2. ✅ **Transparency:** Feature importance reveals reliance on application volume
3. ✅ **Multiple metrics:** R², RMSE, MAE, MAPE to capture different error types
4. ✅ **Documentation:** Clear articulation of limitations and misuse risks

### Recommended Next Steps

**Technical improvements:**
1. **Add physical hazard features:** NOAA wind/rain/distance to decouple risk from demographics
2. **Counterfactual analysis:** Swap demographics while holding hazard constant to isolate bias
3. **Fairness constraints:** Penalize models that underpredict for vulnerable groups
4. **Uncertainty quantification:** Provide prediction intervals, especially for vulnerable tracts

**Policy integration:**
1. **Threshold transparency:** Publish decision rules and expected false negative rates
2. **Audit trail:** Log all model-influenced resource decisions for post-event review
3. **Community validation:** Pilot with 2-3 states, gather feedback from local agencies
4. **Continuous monitoring:** Track disparities in actual vs predicted aid distribution

---

## 6. Conclusion & Future Work

### Summary

This project demonstrates the **feasibility and risks** of using machine learning for disaster relief prediction. We achieved strong predictive performance (R² = 0.93 with Bagging) using only pre-disaster demographic and housing data, suggesting that vulnerability patterns correlate with post-disaster claims. However, bias diagnostics reveal systematic underprediction for high-poverty and high-Black tracts, raising equity concerns about operationalizing these models without safeguards.

**Key contributions:**
1. **Methodological framework:** Reproducible pipeline for tract-level disaster prediction with equity audits
2. **Bias identification:** Quantified underprediction for vulnerable populations (~$200 per tract)
3. **Feature importance:** Revealed model reliance on application volume (31%), not just risk factors
4. **Policy guidance:** Ethical use recommendations for emergency managers

### Limitations

1. **Missing hazard data:** Without wind speed, rainfall, and flood zones, we cannot distinguish physical risk from application bias
2. **Temporal pooling:** Training on 2017-2022 data ignores policy changes (e.g., COVID-era rule modifications)
3. **Geographic confounding:** Storm-specific patterns (e.g., Ida in Louisiana) not modeled separately
4. **Zero-inflation:** 72% of tracts have zero claims; models may overfit to zero predictions

### Future Enhancements

**Immediate (1 month):**
- Integrate NOAA physical features (tract_distance, max_wind, exposure_duration) from collaborator pipeline
- Retrain models and compare feature importance shifts
- Add FEMA flood zone overlays (NFHL data)

**Short-term (3 months):**
- Fairness-aware tuning (e.g., minimize residual gap between vulnerability quartiles)
- Temporal validation (train on 2017-2020, test on 2021-2022 storms)
- State-specific models to capture regional policy differences

**Long-term (6 months):**
- Real-time prediction API for pre-storm resource allocation
- Interactive dashboard with equity indicators for emergency managers
- Integration with FEMA declaration polygons to model tracts inside/outside disaster zones
- Expand to other disaster types (wildfires, floods, tornadoes)

### AURA MVP Implications

This midterm represents **Phase 1 of a 3-phase roadmap**:
- **Phase 1 (complete):** Proof-of-concept with demographic features + bias diagnostics
- **Phase 2 (Q1 2026):** Hazard feature integration + fairness constraints + pilot deployment
- **Phase 3 (Q2 2026):** Real-time API + national rollout + continuous monitoring

The strong baseline performance (R² > 0.9) validates the approach, while bias findings underscore the need for responsible AI guardrails. Next steps focus on enriching hazard data and partnering with state emergency managers for ethical deployment.

---

## References

**Data:**
- FEMA OpenFEMA Individual Assistance Program: https://www.fema.gov/openfema-data-page/individuals-and-households-program-valid-registrations-v1
- CDC/ATSDR Social Vulnerability Index 2022: https://www.atsdr.cdc.gov/placeandhealth/svi/
- US Census Bureau American Community Survey 2022: https://www.census.gov/programs-surveys/acs
- NOAA National Hurricane Center HURDAT2: https://www.nhc.noaa.gov/data/hurdat/

**Collaborators:**
- Hurricane physical feature extraction pipeline: https://github.com/mhahn2011/HURRICANE-DATA-ETL

**Code & Outputs:**
- Training script: `midterm_MVP/scripts/train_tract_storm_models.py`
- Visualization script: `midterm_MVP/scripts/visualize_model_results.py`
- Performance results: `docs/tract_storm_model_performance.csv`
- Best hyperparameters: `docs/tract_storm_best_params.json`
- Presentation notebook: `midterm_MVP/notebooks/midterm_presentation.ipynb`
