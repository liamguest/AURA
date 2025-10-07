# Midterm MVP - Final Deliverables Summary

**Date:** October 6, 2025
**Status:** ✅ Complete and ready for presentation

---

## 📁 File Locations

### Core Deliverables

**1. Final Report (3-4 pages)**
- Location: `docs/MIDTERM_REPORT.md`
- Sections: Introduction, Data, Modeling, Results, Responsible AI, Conclusion
- Format: Markdown (easily converted to PDF if needed)

**2. Presentation Notebook (10-minute presentation)**
- Location: `notebooks/midterm_presentation.ipynb`
- Interactive Jupyter notebook with all results, visualizations, and narrative
- To run: Open in Jupyter and execute cells sequentially

### Supporting Materials

**3. Model Performance Results**
- Performance table: `../../docs/tract_storm_model_performance.csv`
- Best hyperparameters: `../../docs/tract_storm_best_params.json`
- Performance markdown: `../../docs/tract_storm_model_performance.md`

**4. Visualizations**
- Model results: `visuals/model_results/`
  - `01_performance_comparison.png` - Bar charts comparing baseline vs tuned
  - `02_actual_vs_predicted.png` - Scatter plots for best 3 models
  - `03_residuals_by_vulnerability.png` - Bias diagnostic box plots
  - `04_feature_importance.png` - Top 15 features for each model

- Hurricane features (from collaborator): `visuals/hurricane_features/`
  - `ida_interactive_map.html` - Interactive Folium map
  - `qaqc_01_distance.html` - Distance QA/QC
  - `qaqc_02_wind.html` - Wind speed QA/QC

### Scripts

**5. Training & Visualization Code**
- Model training: `scripts/train_tract_storm_models.py` (already run)
- Visualization generation: `scripts/visualize_model_results.py` (already run)
- Data pipeline: `scripts/build_tract_storm.py` (already run)

---

## 🎯 Key Results Summary

### Model Performance
| Model | R² | RMSE | MAE | Best Parameters |
|-------|-----|------|-----|-----------------|
| **Bagging** | **0.927** | $990 | $262 | n_estimators=400, max_samples=1.0, max_features=1.0 |
| Random Forest | 0.924 | $1,007 | $266 | n_estimators=200, max_depth=20, max_features=1.0 |
| Decision Tree | 0.844 | $1,444 | $338 | max_depth=10, min_samples_split=10 |
| KNN | 0.703 | $1,997 | $557 | n_neighbors=10, weights='distance', p=2 |

**Winner:** Bagging explains 92.7% of variance with ~$1,000 average error

### Responsible AI Findings
- **Moderate bias detected:** High-poverty tracts underpredicted by ~$180
- **High-Black tracts:** Underpredicted by ~$150-$200
- **Root cause:** Target variable (FEMA claims) reflects application behavior, not just damage
- **Safeguards:** Clear policy guidance on ethical use (see report Section 5)

### Top Predictive Features
1. **applications** (31.2%) - Number of FEMA applications submitted
2. **housing_units** (12.7%) - Total housing units in tract
3. **acs_total_population** (9.4%) - Tract population
4. **median_household_income** (6.8%)
5. **pct_poverty** (5.2%)

---

## 📊 Dataset Overview

**Observations:** 5,668 census tract × storm event pairs
**Geography:** TX, LA, MS, AL, FL (Gulf Coast)
**Storms:** Harvey (2017), Irma (2017), Michael (2018), Laura (2020), Ida (2021), Ian (2022)
**Target:** FEMA personal property verified losses (dollars)
**Features:** 28 numeric predictors (demographics, housing, vulnerability)

**Key limitation:** Physical hazard features (wind speed, rainfall, distance) are placeholders - awaiting NOAA integration

---

## 🚀 How to Present This Tomorrow

### Option 1: Use the Jupyter Notebook (Recommended)
1. Navigate to `midterm_MVP/notebooks/`
2. Open `midterm_presentation.ipynb` in Jupyter
3. Run all cells (Kernel → Restart & Run All)
4. Use live notebook for 10-minute presentation
5. Visualizations embedded, code hidden for non-technical audience

### Option 2: Export to Slides
```bash
# Install nbconvert if needed
pip install nbconvert

# Convert to HTML slides
jupyter nbconvert midterm_presentation.ipynb --to slides --post serve
```

### Option 3: Use Visuals Directly
- Open PNG files in `visuals/model_results/` for static presentation
- Open HTML files in browser for interactive maps
- Reference `docs/MIDTERM_REPORT.md` for talking points

---

## 📖 Presentation Outline (10 minutes)

**1. Opening (1 min)**
- Problem: Predicting hurricane disaster assistance needs
- Why: Vulnerable communities under-apply, need proactive outreach
- Context: AURA MVP for equitable disaster relief

**2. Data Story (2 min)**
- 5,668 tract-storm pairs from 6 Gulf Coast hurricanes
- Data sources: FEMA, CDC SVI, Census ACS
- Target: FEMA claims (dollars)
- Key limitation: Missing NOAA physical hazard features

**3. Model Strategy (2 min)**
- 4 algorithms: KNN, Decision Tree, Random Forest, Bagging
- Hyperparameter tuning with 5-fold CV
- Evaluated on 5 metrics (R², RMSE, MAE, MAPE, Explained Variance)
- Best: Bagging (R²=0.93)

**4. Results Snapshot (2.5 min)**
- Show `01_performance_comparison.png` - ensemble methods win
- Show `02_actual_vs_predicted.png` - strong correlation
- Show `04_feature_importance.png` - applications dominate (31%)
- Interpretation: Model learns "who applies" not "who needs help"

**5. Responsible AI Lens (1.5 min)**
- Show `03_residuals_by_vulnerability.png` - bias diagnostics
- Finding: High-poverty tracts underpredicted by $180
- Root cause: FEMA claims ≠ damage (application barriers)
- Safeguard: Use predictions to increase outreach, not ration aid

**6. Call to Action (1 min)**
- Next steps: Integrate NOAA hazard features
- Policy implication: Flag low-prediction + high-vulnerability tracts
- AURA roadmap: Pilot with Gulf Coast states, then national rollout

---

## 🛠️ Technical Details

### Environment Setup
```bash
# Activate virtual environment
source ../../.venv/bin/activate

# Already installed packages:
# - pandas, numpy, scikit-learn, matplotlib, seaborn
# - geopandas, pyarrow (for data pipeline)
```

### Reproducing Results
```bash
# 1. Rebuild dataset (if needed)
python scripts/build_tract_storm.py

# 2. Train models
python scripts/train_tract_storm_models.py

# 3. Generate visualizations
python scripts/visualize_model_results.py

# Results saved to:
# - ../../docs/tract_storm_model_performance.csv
# - ../../docs/tract_storm_best_params.json
# - visuals/model_results/*.png
```

---

## 📧 Collaborator Acknowledgment

Hurricane physical feature extraction pipeline:
- GitHub: https://github.com/mhahn2011/HURRICANE-DATA-ETL
- Visualizations borrowed: Ida interactive map, wind/distance QA/QC
- Future integration: tract_distance, max_wind_mph, exposure_duration

---

## ✅ Assignment Requirements Met

**Technical Requirements:**
- ✅ 4 ML algorithms: KNN, Decision Tree, Random Forest, Bagging
- ✅ Full hyperparameter tuning with GridSearchCV
- ✅ 5+ evaluation metrics: R², RMSE, MAE, MAPE, Explained Variance
- ✅ Train/test split with cross-validation
- ✅ Applied to real public health problem (disaster assistance equity)

**Responsible AI Requirements:**
- ✅ Bias analysis by demographic groups (poverty, race/ethnicity, language)
- ✅ Discussion of who benefits and who could be harmed
- ✅ Practical safeguards (policy guidance, threshold transparency)
- ✅ Data limitations documented (missing hazard features, target bias)
- ✅ Health promotion recommendations (proactive outreach to vulnerable tracts)

**Deliverables:**
- ✅ 3-4 page report (MIDTERM_REPORT.md)
- ✅ Presentation materials (midterm_presentation.ipynb + visualizations)
- ✅ Code repository (scripts fully documented)

---

## 🎓 Final Notes

**Strengths:**
- Strong predictive performance (R² > 0.9)
- Rigorous bias diagnostics
- Clear policy implications
- Reproducible pipeline

**Acknowledged Weaknesses:**
- Missing physical hazard features (explicitly discussed as future work)
- Target variable bias (FEMA claims ≠ true need)
- Zero-inflation handling could be improved (e.g., two-stage model)

**Confidence Level:** High - all requirements met, results are solid, equity discussion is thorough

**Good luck with your presentation tomorrow! 🚀**
