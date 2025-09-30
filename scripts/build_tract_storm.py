"""Build the tract_storm modeling dataset for the AURA midterm MVP.

This script assembles tract-level demographic, housing, hazard, and policy
features for each FEMA disaster (storm) represented in the processed datasets.
It expects the following inputs under the repository root:

- data/raw/svi/SVI_2022_US.csv (CDC Social Vulnerability Index bulk file)
- data/processed/acs_2022_tx_la_ms_al_fl_clean.csv (ACS tract attributes)
- data/processed/fema_housing_by_tract_disaster.csv (FEMA IA aggregates)
- data/processed/hazard_features_by_tract.csv (optional placeholder)

Outputs:
- data/processed/tract_storm_features.csv
- data/processed/tract_storm_features.parquet (if pyarrow/fastparquet is available)

Notes:
- Hazard exposure columns are merged when present; otherwise they are created
  with NaNs so downstream modeling can impute them.
- Percentage fields from SVI (EP_*) are supplied as 0–100 percentages; we
  convert them to proportions for modeling convenience.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

# Define constants for reuse and to avoid scattering literals through the script.
REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

SVI_STATES = {"AL", "FL", "LA", "MS", "TX"}

SVI_PERCENT_COLUMNS = {
    "EP_POV150": "pct_poverty",
    "EP_AGE65": "pct_elderly",
    "EP_AGE17": "pct_children",
    "EP_NOVEH": "pct_no_vehicle",
    "EP_LIMENG": "pct_limited_english",
    "EP_HISP": "pct_hispanic",
    "EP_AFAM": "pct_black",
    "EP_MOBILE": "pct_mobile_homes",
    "EP_MUNIT": "pct_multi_unit",
    "EP_CROWD": "pct_crowded_housing",
}

SVI_ADDITIONAL_COLUMNS = {
    "E_TOTPOP": "svi_total_population",
    "E_HU": "housing_units",
    "AREA_SQMI": "area_sq_mi",
    "COUNTY": "county_name",
    "ST_ABBR": "state_abbr",
    "ST": "state_fips",
}

ACS_COLUMNS = {
    "tract_geoid": "tract_geoid",
    "median_household_income": "median_household_income",
    "total_population": "acs_total_population",
    "state_fips": "acs_state_fips",
}

HAZARD_COLUMNS = [
    "max_wind_mph",
    "total_rainfall_in",
    "min_distance_km",
    "pct_floodplain",
]


@dataclass
class BuildConfig:
    """Configuration for building the tract_storm dataset."""

    svi_path: Path = RAW_DIR / "svi" / "SVI_2022_US.csv"
    acs_path: Path = PROCESSED_DIR / "acs_2022_tx_la_ms_al_fl_clean.csv"
    fema_path: Path = PROCESSED_DIR / "fema_housing_by_tract_disaster.csv"
    hazard_path: Path = PROCESSED_DIR / "hazard_features_by_tract.csv"
    output_csv: Path = PROCESSED_DIR / "tract_storm_features.csv"
    output_parquet: Path = PROCESSED_DIR / "tract_storm_features.parquet"


def load_fema_claims(path: Path) -> pd.DataFrame:
    """Load FEMA IA aggregates and prep outcome columns."""
    df = pd.read_csv(
        path,
        dtype={"tract_geoid": "string", "disasterNumber": "Int64"},
    )
    df = df.rename(
        columns={
            "ppfvl_sum": "fema_claims_total",
            "ppfvl_mean": "fema_claims_mean",
        }
    )
    numeric_cols = [
        "fema_claims_total",
        "fema_claims_mean",
        "applications",
        "tsa_eligible_rate",
        "owner_rate",
        "renter_rate",
    ]
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    df["fema_claims_total"] = df["fema_claims_total"].fillna(0.0)
    df["fema_claims_mean"] = df["fema_claims_mean"].fillna(0.0)
    df["applications"] = df["applications"].fillna(0)
    return df


def load_svi_features(path: Path) -> pd.DataFrame:
    """Load SVI raw file and extract Gulf tract-level features."""
    use_cols = {"FIPS", *SVI_PERCENT_COLUMNS.keys(), *SVI_ADDITIONAL_COLUMNS.keys()}
    svi = pd.read_csv(path, dtype={"FIPS": "string"}, usecols=list(use_cols))
    svi = svi.rename(columns={"FIPS": "tract_geoid"})

    # Filter to the states of interest.
    svi = svi[svi["ST_ABBR"].isin(SVI_STATES)].copy()

    for raw_col, out_col in SVI_PERCENT_COLUMNS.items():
        # Convert 0–100 percentages into proportions; leave NaNs untouched.
        svi[out_col] = svi[raw_col] / 100.0

    svi = svi.rename(columns=SVI_ADDITIONAL_COLUMNS)
    keep_cols = [
        "tract_geoid",
        *SVI_PERCENT_COLUMNS.values(),
        *SVI_ADDITIONAL_COLUMNS.values(),
    ]
    svi = svi[keep_cols]
    return svi


def load_acs_features(path: Path) -> pd.DataFrame:
    acs = pd.read_csv(path, dtype={"tract_geoid": "string", "state_fips": "string"})
    acs = acs.rename(columns=ACS_COLUMNS)
    acs = acs[list(ACS_COLUMNS.values())]
    return acs


def load_hazard_features(path: Path) -> pd.DataFrame:
    if path.exists():
        hazard = pd.read_csv(
            path,
            dtype={"tract_geoid": "string", "disasterNumber": "Int64"},
        )
    else:
        hazard = pd.DataFrame(columns=["tract_geoid", "disasterNumber", *HAZARD_COLUMNS])
    for col in HAZARD_COLUMNS:
        if col not in hazard.columns:
            hazard[col] = pd.NA
    return hazard[["tract_geoid", "disasterNumber", *HAZARD_COLUMNS]]


def attach_population(df: pd.DataFrame) -> pd.DataFrame:
    df["population_for_pc"] = df["acs_total_population"].fillna(df["svi_total_population"])
    df["fema_claims_pc"] = df.apply(
        lambda row: row["fema_claims_total"] / row["population_for_pc"]
        if pd.notna(row["population_for_pc"]) and row["population_for_pc"] > 0
        else pd.NA,
        axis=1,
    )
    return df


def mark_declaration(df: pd.DataFrame) -> pd.DataFrame:
    df["declaration"] = df["disasterNumber"].notna().astype(int)
    return df


def compute_transforms(df: pd.DataFrame) -> pd.DataFrame:
    df["fema_claims_total"] = df["fema_claims_total"].astype(float)
    df["fema_claims_total_log1p"] = np.log1p(df["fema_claims_total"].clip(lower=0))
    return df


def build_dataset(config: BuildConfig) -> pd.DataFrame:
    fema = load_fema_claims(config.fema_path)
    svi = load_svi_features(config.svi_path)
    acs = load_acs_features(config.acs_path)
    hazard = load_hazard_features(config.hazard_path)

    df = fema.merge(svi, on="tract_geoid", how="left")
    df = df.merge(acs, on="tract_geoid", how="left")
    df = df.merge(hazard, on=["tract_geoid", "disasterNumber"], how="left")

    df = mark_declaration(df)
    df = attach_population(df)
    df = compute_transforms(df)

    outcome_cols = [
        "tract_geoid",
        "disasterNumber",
        "fema_claims_total",
        "fema_claims_total_log1p",
        "fema_claims_pc",
        "applications",
        "fema_claims_mean",
    ]
    policy_cols = [
        "declaration",
        "tsa_eligible_rate",
        "owner_rate",
        "renter_rate",
    ]
    demo_cols = [
        "acs_total_population",
        "svi_total_population",
        "population_for_pc",
        "median_household_income",
        "pct_poverty",
        "pct_elderly",
        "pct_children",
        "pct_black",
        "pct_hispanic",
        "pct_no_vehicle",
        "pct_limited_english",
    ]
    housing_cols = [
        "housing_units",
        "pct_mobile_homes",
        "pct_multi_unit",
        "pct_crowded_housing",
    ]
    hazard_cols = HAZARD_COLUMNS
    geography_cols = [
        "state_abbr",
        "county_name",
        "acs_state_fips",
        "state_fips",
        "area_sq_mi",
    ]

    ordered_cols = outcome_cols + policy_cols + demo_cols + housing_cols + hazard_cols + geography_cols
    existing_cols = [col for col in ordered_cols if col in df.columns]
    df = df[existing_cols]
    return df


def export(df: pd.DataFrame, config: BuildConfig) -> None:
    config.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.output_csv, index=False)
    try:
        df.to_parquet(config.output_parquet, index=False)
    except (ImportError, ValueError):
        # Parquet support is optional; skip if dependencies are unavailable.
        pass


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--svi", dest="svi_path", type=Path, default=BuildConfig.svi_path)
    parser.add_argument("--acs", dest="acs_path", type=Path, default=BuildConfig.acs_path)
    parser.add_argument("--fema", dest="fema_path", type=Path, default=BuildConfig.fema_path)
    parser.add_argument("--hazard", dest="hazard_path", type=Path, default=BuildConfig.hazard_path)
    parser.add_argument("--out-csv", dest="output_csv", type=Path, default=BuildConfig.output_csv)
    parser.add_argument("--out-parquet", dest="output_parquet", type=Path, default=BuildConfig.output_parquet)
    args = parser.parse_args()
    return BuildConfig(
        svi_path=args.svi_path,
        acs_path=args.acs_path,
        fema_path=args.fema_path,
        hazard_path=args.hazard_path,
        output_csv=args.output_csv,
        output_parquet=args.output_parquet,
    )


def main() -> None:
    config = parse_args()
    df = build_dataset(config)
    export(df, config)


if __name__ == "__main__":
    main()
