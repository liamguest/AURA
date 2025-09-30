# Responsible AI Considerations – tract_storm MVP

**Theme:** Equitable access to disaster relief.

## Key Questions
- Are historically marginalized tracts (e.g., high Black or Hispanic population share, limited English proficiency) systematically under-predicted, potentially masking true need?
- Do FEMA declarations and claim processes introduce policy bias that the model simply reproduces rather than challenges?
- Does the dataset capture non-claiming tracts inside disaster zones so we can distinguish need from access barriers?

## Data Risks
- **Coverage gaps:** Hazard exposure columns are currently placeholders; without precise wind/rain/distance measures, the model may learn socio-demographics as proxies for hazard intensity.
- **Measurement error:** FEMA claims represent assistance awarded, not underlying damage. Communities with low application rates (lack of awareness, trust, or access) will appear low-risk.
- **Temporal drift:** Disasters span multiple years with evolving policies; pooling them risks overweighting older regimes.

## Modeling Safeguards
1. **Bias diagnostics:** After training, segment residuals by vulnerability buckets (e.g., quartiles of `pct_poverty`, `pct_hispanic`, `pct_black`, `pct_limited_english`) to confirm no systematic underestimation.
2. **Counterfactual stress tests:** Swap hazard features while holding demographics constant to ensure predictions respond primarily to physical risk once exposure features are populated.
3. **Explainability:** Use permutation feature importance and partial dependence/accumulated local effects for tree ensembles; report which variables drive predictions and whether they align with policy intent.
4. **Threshold transparency:** If models inform triage/resource thresholds, publish cut-points and expected false negative rates for vulnerable tracts.

## Policy Framing
- Emphasize that predictions indicate *expected claims given historical patterns*, not eligibility. Low predicted claims in high-poverty tracts should trigger investigation, not deprioritization.
- Partner with state VOADs and local agencies to contextualize findings before operational use.
- Maintain a human-in-the-loop review for deployment decisions affecting aid distribution.

## Next Steps
- Integrate FEMA declaration polygons to capture tracts inside/outside formal disaster zones and audit disparities in declaration coverage.
- Add NFHL floodplain overlays and NOAA track distances to decouple hazard intensity from demographic correlates.
- Document data lineage and transformation decisions in a structured datasheet for the midterm report appendix.
