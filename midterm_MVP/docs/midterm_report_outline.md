# Midterm Report & Presentation Outline

## Report (3–4 pages)
1. **Introduction (0.5 page)**
   - Project motivation: equitable disaster relief & AURA MVP alignment.
   - Research questions and scope of tract_storm dataset.
2. **Data & Preprocessing (1 page)**
   - Sources (FEMA IA, ACS, CDC SVI, Hazard placeholders) and join keys.
   - Feature engineering: percentage conversions, log transform of claims, handling missing hazard fields.
   - Limitations: absence of full hazard metrics, declaration coverage gaps.
3. **Modeling Approach (0.75 page)**
   - Algorithms: KNN, Decision Tree, Random Forest, Bagging.
   - Train/test split, cross-validation, hyperparameter grids.
   - Evaluation metrics (R², RMSE, MAE, MAPE, Explained Variance).
4. **Results (0.75 page)**
   - Baseline vs tuned performance table.
   - Key takeaways (e.g., best-performing algorithm, error distribution insights).
   - Visuals to include: actual vs predicted scatter, residuals by vulnerability bucket.
5. **Responsible AI Discussion (0.5 page)**
   - Bias diagnostics results, data limitations, policy interpretation guidance.
6. **Conclusion & Future Work (0.5 page)**
   - Summary of findings and MVP implications.
   - Next steps: hazard enrichment, fairness audits, integration roadmap for AURA.

## Presentation (10 minutes)
1. **Opening (1 min)** – Story hook (e.g., contrasting outcomes for two Gulf Coast tracts).
2. **Data Story (2 min)** – Visual pipeline of sources; highlight gaps.
3. **Model Strategy (2 min)** – Explain algorithm choices and tuning visually.
4. **Results Snapshot (2.5 min)** – Comparative metric chart + key plots.
5. **Responsible AI Lens (1.5 min)** – Bias findings, stakeholder impacts, guardrails.
6. **Call to Action (1 min)** – How this informs FEMA/AURA coordination and next steps.
