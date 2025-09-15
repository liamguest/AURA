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
- Source (raw): data/raw/svi/*.csv (drop state CSVs here)
- Combined (interim): data/interim/svi_combined.csv
- Simple (processed): data/processed/svi_simple.csv

Keys/columns:
- tract_geoid: Derived from an SVI GEOID/FIPS column (normalized to 11 digits).
- Additional SVI variables depend on the release; we’ll select/rename in a later pass.
