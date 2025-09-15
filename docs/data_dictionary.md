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
