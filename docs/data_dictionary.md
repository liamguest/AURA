# Data Dictionary (WIP)

This document describes the columns currently produced in processed datasets and where source files are stored.

## Directory conventions
- data/raw: Exact downloads as received (immutable provenance).
- data/interim: Lightly transformed/staged files (parsed text, column subsets).
- data/processed: Analysis‑ready tables with standardized keys, types, and names.

## ACS 5‑year Tracts (2022)
- Source (interim): data/interim/acs_2022_tx_la_ms_al_fl.csv
- Output (processed): data/processed/acs_2022_tx_la_ms_al_fl_clean.csv

Columns (processed):
- tract_geoid: 11‑digit census tract identifier (string). Derived from GEO_ID suffix.
- NAME: Human‑readable tract label (tract, county, state).
- state_fips: 2‑digit state FIPS code (string) for TX, LA, MS, AL, FL.
- total_population: ACS total population estimate (B01003_001E). Numeric.
- median_household_income: ACS median household income in USD (B19013_001E). Numeric.

Upstream fields (interim):
- GEO_ID: ACS geographic identifier (e.g., 1400000US48157671202). Last 11 chars are the tract GEOID.
- NAME: As above.
- B01003_001E: Total population.
- B19013_001E: Median household income.
- state_fips: Injected during pull for state provenance.

## FEMA Housing Registrants (IA) — subset and aggregation
- Source (raw subset): data/raw/fema_housing_subset.json
- Clean (interim): data/interim/fema_housing_subset_clean.csv
- Aggregated (processed):
  - data/processed/fema_housing_by_tract_disaster.csv (all pulled disasters)
  - data/processed/fema_housing_by_tract_disaster_focus.csv (14-storm shortlist)

Processed columns (by tract):
- tract_geoid: 11‑digit census tract identifier (string). From censusBlockId.
- applications: Count of IA records associated with the tract.
- ppfvl_sum: Sum of personal property funding verified loss (ppfvl).
- ppfvl_mean: Mean ppfvl.
- tsa_eligible_rate: Share of records eligible for TSA (transitional sheltering assistance).
- owner_rate / renter_rate: Share of Owner vs Renter in ownRent.

Notes:
- The subset is paged from the v1 Housing Registrants endpoint and may be filtered by disasterNumber.
- `scripts/fetch_fema_ia.py` now automates pulls for the shortlist storms, discovering disaster numbers
  from the FEMA tract coverage table (`data/processed/tract_storm_fema_coverage.csv`).
- The `*_focus.csv` file feeds `build_tract_storm.py` so only the 14 named storms enter the master dataset.

## FEMA Declared Counties → Tract Coverage
- Source (interim counties): `data/interim/fema_declared_counties_storms.csv`
- Processed tract mapping: `data/processed/tract_storm_fema_coverage.csv`

Columns:
- storm_name: Hurricane name matching shortlist.
- disasterNumber: FEMA event identifier.
- state: Two-letter postal abbreviation.
- fipsStateCode / fipsCountyCode: 2- and 3-digit state/county FIPS strings.
- designatedArea: County/parish label as listed in the declaration.
- tract_geoid: 11-digit tract GEOID derived via TIGER/Line tract shapefiles (AL, FL, LA, MS, TX).

Notes:
- Coverage is restricted to the Gulf states where TIGER tract boundaries are staged.
- Table is the basis for the `in_fema_decl` flag inside the master `tract_storm_features` dataset.

## NOAA Hazard Coverage (expected feed)
- Processed location: `data/processed/hazard_features_by_tract.csv`
- Required key columns: `tract_geoid`, `disasterNumber`
- Expected feature columns (v1):
  - `max_wind_experienced`
  - `exposure_duration_hours`
  - `intensification_rate`
  - `distance_to_tract`
- Additional hazard metrics can be appended later; script will pass them through automatically.
- Presence of a row yields `in_noaa_exposure = 1` when merged by `build_tract_storm.py`.

## CDC Social Vulnerability Index (SVI)
- Source (raw): `data/raw/svi/*.csv` (e.g., `SVI_2022_US.csv` placed under `data/raw/svi/`)
- Combined (interim, filtered to Gulf states): `data/interim/svi_combined.csv`
- Simple (processed, filtered): `data/processed/svi_simple.csv`
- States kept: Alabama (AL), Louisiana (LA), Florida (FL), Texas (TX), Mississippi (MS)

Key columns (present in simple and/or combined files):
- tract_geoid: 11‑digit census tract identifier (string). Derived from `FIPS`/`GEOID` in source CSV.
- ST / ST_ABBR: State code or abbreviation (e.g., AL, LA, FL, TX, MS).
- STATE: State name.
- STCNTY: State+county FIPS.
- COUNTY: County name.
- FIPS: Full geographic identifier from SVI file (may be tract‑level).
- LOCATION: Human‑readable place label.
- AREA_SQMI: Area in square miles.
- E_TOTPOP / M_TOTPOP: Estimated total population and margin of error.
- E_HU / M_HU: Estimated housing units and margin of error.
- E_HH / M_HH: Estimated households and margin of error.
- E_POV150 / M_POV150: Population at or below 150% of poverty and margin.
- E_UNEMP / M_UNEMP: Unemployed civilian population (16+) and margin.
- E_HBURD / M_HBURD: Housing cost burdened households and margin.
- RPL_THEME1..RPL_THEME4: Thematic SVI percentile rankings (socioeconomic; household composition & disability; minority status & language; housing type & transportation).
- RPL_THEMES: Overall SVI percentile ranking.

Notes:
- The interim file retains all columns post‑filter; `svi_simple.csv` is a starter subset for faster exploration. You can add more SVI variables by editing the keep‑list in `notebooks/data_cleaning/40_svi_ingest.ipynb` and re‑running.

## tract_storm Master Dataset
- Builder script: `scripts/build_tract_storm.py`
- Output: `data/processed/tract_storm_features.csv` (and `.parquet` when dependencies exist)

Key columns (current snapshot):
- Keys & targets:
  - `tract_geoid`, `disasterNumber`
  - `fema_claims_total`, `fema_claims_total_log1p`, `fema_claims_pc`, `applications`, `fema_claims_mean`
- Policy & coverage:
  - `declaration` (presence of claims proxy)
  - `tsa_eligible_rate`, `owner_rate`, `renter_rate`
  - `in_fema_decl`, `in_noaa_exposure`, `coverage_label` (`both`, `fema_only`, `noaa_only`, `none`)
- Demographics & housing (SVI/ACS driven):
  - `acs_total_population`, `svi_total_population`, `population_for_pc`
  - `median_household_income`, `pct_poverty`, `pct_elderly`, `pct_children`, `pct_black`, `pct_hispanic`,
    `pct_no_vehicle`, `pct_limited_english`
  - `housing_units`, `pct_mobile_homes`, `pct_multi_unit`, `pct_crowded_housing`
- Hazard placeholders (if NOAA feed absent these remain NaN):
  - `max_wind_mph`, `total_rainfall_in`, `min_distance_km`, `pct_floodplain`
- Geography/context: `state_abbr`, `county_name`, `acs_state_fips`, `state_fips`, `area_sq_mi`

Notes:
- FEMA coverage flag defaults to 0 when a tract isn’t in the declarations table; NOAA coverage defaults
  to 0 when the hazard feed lacks a matching row.
- `coverage_label` makes it easy to filter to tracts present in one or both coverage regimes.
- Input FEMA claims should use the `_focus.csv` output so we stay scoped to the 14 named storms.

Change log
- 2025‑09‑28: Added FEMA IA shortlist outputs, tract coverage, hazard expectations, and master dataset fields.
- 2025‑09‑08: Initial ACS dictionary added.
