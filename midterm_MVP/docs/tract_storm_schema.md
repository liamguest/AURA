# tract_storm Dataset Blueprint

This note captures the working schema and data sources for the joint statistical learning midterm and AURA MVP effort.

## Target
- **fema_claims_total**: Sum of FEMA personal property verified losses (`ppfvl_sum`) for a tract–storm pair.
- **fema_claims_pc**: `fema_claims_total` divided by tract total population (falling back to households when population is missing).

## Keys
- **tract_geoid**: 11-digit Census tract GEOID.
- **disasterNumber**: FEMA disaster identifier, treated as the storm key (Harvey=4332, Irma=4337, Laura=4559, Ida=4611, Ian=4673, etc.).

## Feature Groups & Planned Sources

| Theme | Fields | Source & Notes |
| --- | --- | --- |
| Demographics | total_population, median_household_income, pct_elderly (65+), pct_children (<18), pct_poverty (≤150% FPL), pct_no_vehicle, pct_limited_english, pct_hispanic, pct_black | CDC SVI 2022 bulk file (`data/raw/svi/SVI_2022_US.csv`). Use `EP_*` percentage fields divided by 100 to convert to proportions. Median income comes from ACS clean table (`data/processed/acs_2022_tx_la_ms_al_fl_clean.csv`). |
| Housing | housing_units, pct_mobile_homes, pct_multi_unit, pct_crowded | CDC SVI `E_HU`, `EP_MOBILE`, `EP_MUNIT`, `EP_CROWD` (converted to proportions). |
| Hazard Exposure | storm_max_wind_mph, storm_total_rain_in, min_distance_km, pct_floodplain | Placeholder table `data/processed/hazard_features_by_tract.csv` to be constructed from NOAA HURDAT2 tracks + precipitation grids + floodplain overlays. Current pipeline will expect these columns and allow NaNs until calculations are implemented. |
| Policy & Access | declaration (binary), applications, tsa_eligible_rate, owner_rate, renter_rate | FEMA IA housing registrant aggregates (`data/processed/fema_housing_by_tract_disaster.csv`). Declaration defaults to 1 for tracts in declared counties and 0 otherwise once the full tract universe is joined; stub implementation flags any matched tract–storm pair as declared. |
| Geography | state_fips, county_name, area_sqmi | From SVI and ACS tables for contextual filtering. |

## Processing Notes
- Normalize percentage fields to decimals (0–1) for modeling.
- Apply log1p transform to `fema_claims_total` before model fitting to manage heavy right tail; store both raw and log outcome.
- Standardize continuous predictors for distance-based learners (KNN) using a `ColumnTransformer` pipeline.
- Keep original outcome columns for reporting/evaluation.

## Outstanding Data Gaps
1. Hazard exposure metrics require NOAA storm tracks, rainfall rasters, and tract centroid distances. A placeholder CSV will unblock modeling but should be replaced with engineered values.
2. Floodplain coverage requires FEMA NFHL (or NRI) overlays intersected with tract polygons; currently out of scope for this iteration.
3. Declaration flag should incorporate FEMA declaration shapefiles to tag unaffected tracts with 0; interim approach uses the presence/absence of claims.

The upcoming pipeline steps will materialize this schema into `data/processed/tract_storm_features.parquet` and a modeling-ready `data/processed/tract_storm_model_input.csv`.
