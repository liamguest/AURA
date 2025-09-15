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

## Roadmap (to be added next)
- FEMA Housing Registrants (IA): will produce tract_geoid, program eligibility flags, assistance amounts; joinable on tract_geoid.
- NOAA HURDAT2 exposure: parsed track points; used to derive tract‑level exposure features in processed outputs.
- CDC SVI: tract‑level vulnerability indices; joinable on tract_geoid.

Change log
- 2025‑09‑08: Initial ACS dictionary added.

## FEMA Housing Registrants (IA) — subset and aggregation
- Source (raw subset): data/raw/fema_housing_subset.json
- Clean (interim): data/interim/fema_housing_subset_clean.csv
- Aggregated (processed): data/processed/fema_housing_by_tract.csv

Processed columns (by tract):
- tract_geoid: 11‑digit census tract identifier (string). From censusBlockId.
- applications: Count of IA records associated with the tract.
- ppfvl_sum: Sum of personal property funding verified loss (ppfvl).
- ppfvl_mean: Mean ppfvl.
- tsa_eligible_rate: Share of records eligible for TSA (transitional sheltering assistance).
- owner_rate / renter_rate: Share of Owner vs Renter in ownRent.

Notes:
- The subset is paged from the v1 Housing Registrants endpoint and may be filtered by disasterNumber.
- disaster filters used (example): Harvey=4332, Laura=4559, Ida=4611, Ian=4673 (adjust as needed).

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
