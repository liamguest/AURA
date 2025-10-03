# AURA: AI for Urban Resilience & Alerts

AURA is a research project building census-tract–level models of **hurricane impacts and disaster preparedness** across Gulf Coast communities. It integrates FEMA disaster declarations and Individual Assistance (IA) claims with NOAA hazard data, Census TIGER tracts, and social vulnerability indicators to produce explainable risk models.

---

## Project Overview

- **Core Goal:** Predict and map storm-related risk at the tract level for major Gulf Coast hurricanes.
- **Coverage Feeds:**
  - `tract_storm_fema`: FEMA disaster declarations and IA claims joined to TIGER/Line tracts.
  - `tract_storm_noaa`: NOAA HURDAT2 storm tracks intersected with tract boundaries (hazard metrics delivered separately).
- **Storm Shortlist (14 Gulf Coast storms):** Katrina (2005), Rita (2005), Dennis (2005), Gustav (2008), Ike (2008), Harvey (2017), Irma (2017), Michael (2018), Laura (2020), Delta (2020), Zeta (2020), Sally (2020), Ida (2021), Ian (2022).
- **Key FEMA Disaster Numbers:**
  - Katrina — DR-1603-LA, DR-1604-MS, DR-1605-AL
  - Rita — DR-1606-TX, DR-1607-LA
  - Dennis — DR-1595-FL, DR-1594-MS, DR-1593-AL
  - Gustav — DR-1786-LA
  - Ike — DR-1791-TX, DR-1792-LA
  - Harvey — DR-4332-TX
  - Irma — DR-4337-FL
  - Michael — DR-4399-FL
  - Laura — DR-4559-LA
  - Delta — DR-4570-LA
  - Zeta — DR-4577-LA
  - Sally — DR-4563-AL
  - Ida — DR-4611-LA
  - Ian — DR-4673-FL (and DR-4675 Seminole Tribe)

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies include `geopandas`, `pyogrio`, `pandas`, `scikit-learn`, `numpy`, and supporting libraries captured in `requirements.txt`.

**Prerequisite Data:** Ensure TIGER/Line tract shapefiles are unzipped under `data/raw/shapefiles/` with their `.shp`, `.shx`, `.dbf`, and `.prj` components. Useful sources:

- [Alabama (01)](https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_01_tract.zip)
- [Florida (12)](https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_12_tract.zip)
- [Louisiana (22)](https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_22_tract.zip)
- [Mississippi (28)](https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_28_tract.zip)
- [Texas (48)](https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_48_tract.zip)

---

## Workflow

1. **Refresh FEMA IA Data**

   ```bash
   source .venv/bin/activate
   python scripts/fetch_fema_ia.py --coverage data/processed/tract_storm_fema_coverage.csv
   ```

   This CLI discovers the disaster numbers from the FEMA tract coverage file, pulls all IA records for the shortlist states, and writes raw, cleaned, and aggregated outputs under `data/`.

2. **Drop NOAA Hazard Feed**

   Place the NOAA-derived table at `data/processed/hazard_features_by_tract.csv` with columns:
   - `tract_geoid`, `disasterNumber`
   - `max_wind_experienced`, `exposure_duration_hours`, `intensification_rate`, `distance_to_tract`
   - Additional hazard metrics may be appended as needed.

3. **Rebuild Tract-Level Dataset**

   ```bash
   python scripts/build_tract_storm.py --fema data/processed/fema_housing_by_tract_disaster_focus.csv
   ```

   Output: `data/processed/tract_storm_features.csv` (.parquet optional) containing all engineered features plus coverage flags `in_fema_decl`, `in_noaa_exposure`, and the categorical `coverage_label` (`both`, `fema_only`, `noaa_only`, `none`).

4. **Train Models**

   ```bash
   python scripts/train_tract_storm_models.py
   ```

   Trains KNN, Decision Tree, Random Forest, and Bagging regressors on `fema_claims_total_log1p`; writes evaluation artifacts to:
   - `docs/tract_storm_model_performance.csv`
   - `docs/tract_storm_model_performance.md`
   - `docs/tract_storm_best_params.json`

---

## Documentation & References

- [`docs/data_dictionary.md`](docs/data_dictionary.md) — end-to-end definitions of tracts, FEMA aggregates, coverage tables, hazard expectations, and master dataset columns.
- FEMA declarations: [Disaster Declarations Summaries (v2)](https://www.fema.gov/openfema-data-page/disaster-declarations-summaries-v2)
- FEMA IA API: [Individual Assistance Housing Registrants Large Disasters](https://www.fema.gov/openfema-data-page/individual-assistance-housing-registrants-large-disasters)

For schema details or midterm deliverables, see `midterm_MVP/docs/tract_storm_schema.md` and `midterm_MVP/docs/midterm_report_outline.md`.

---

## Next Steps

- Integrate the NOAA hazard feed as soon as it lands, rerun the builder, and re-evaluate models.
- Expand fairness and residual diagnostics once both coverage sources are populated.
- Update this README and the data dictionary when new features or storms are added.
