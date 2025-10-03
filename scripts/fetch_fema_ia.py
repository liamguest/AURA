"""Pull FEMA Individual Assistance Housing Registrants for selected disasters.

This script replaces the notebook scraping step with a reproducible CLI flow:
- Discovers target disaster numbers from the FEMA tract coverage table.
- Calls the OpenFEMA API per-disaster with a state filter (AL, FL, LA, MS, TX).
- Stores the raw JSON subset, a cleaned CSV with tract GEOIDs, and the tract-level aggregates.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

OPENFEMA_URL = "https://www.fema.gov/api/open/v1/IndividualAssistanceHousingRegistrantsLargeDisasters"
STATE_ABBR = ["AL", "FL", "LA", "MS", "TX"]
SELECT_COLS = [
    "id",
    "disasterNumber",
    "damagedStateAbbreviation",
    "damagedZipCode",
    "ownRent",
    "residenceType",
    "grossIncome",
    "specialNeeds",
    "tsaEligible",
    "repairAssistanceEligible",
    "replacementAssistanceEligible",
    "personalPropertyEligible",
    "ppfvl",
    "censusBlockId",
    "censusYear",
]


def discover_disasters(coverage_path: Path) -> pd.DataFrame:
    df = pd.read_csv(coverage_path, dtype={"disasterNumber": "Int64"})
    needed_cols = {"storm_name", "disasterNumber"}
    missing = needed_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"Coverage file {coverage_path} missing required columns: {', '.join(sorted(missing))}"
        )
    return df[["storm_name", "disasterNumber"]].drop_duplicates().sort_values("disasterNumber")


def fetch_disaster(disaster_number: int, top: int, states: Iterable[str]) -> List[dict]:
    items: List[dict] = []
    skip = 0
    states_str = "','".join(states)
    while True:
        params = {
            "$filter": f"disasterNumber eq {disaster_number} and damagedStateAbbreviation in ('{states_str}')",
            "$select": ",".join(SELECT_COLS),
            "$format": "json",
            "$top": top,
            "$skip": skip,
        }
        response = requests.get(OPENFEMA_URL, params=params, timeout=120)
        response.raise_for_status()
        batch = response.json().get("IndividualAssistanceHousingRegistrantsLargeDisasters", [])
        if not batch:
            break
        items.extend(batch)
        skip += top
        time.sleep(0.2)
    return items


def to_tract_geoid(value) -> str | None:
    if pd.isna(value):
        return None
    text = str(value)
    return text[:11] if len(text) >= 11 else None


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["tract_geoid"] = df["censusBlockId"].map(to_tract_geoid)
    df = df.dropna(subset=["tract_geoid"])  # ensure groupby keys present

    def rate_bool(series: pd.Series) -> float:
        try:
            return pd.Series(series).astype("boolean").mean(skipna=True)
        except Exception:
            return pd.to_numeric(series == True, errors="coerce").mean()

    grouped = (
        df.groupby(["tract_geoid", "disasterNumber"], as_index=False)
        .agg(
            applications=("id", "count"),
            ppfvl_sum=("ppfvl", "sum"),
            ppfvl_mean=("ppfvl", "mean"),
            tsa_eligible_rate=("tsaEligible", rate_bool),
            owner_rate=("ownRent", lambda s: (s == "Owner").mean()),
            renter_rate=("ownRent", lambda s: (s == "Renter").mean()),
        )
        .sort_values(["disasterNumber", "tract_geoid"])
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--coverage",
        type=Path,
        default=PROCESSED_DIR / "tract_storm_fema_coverage.csv",
        help="Coverage file with storm_name and disasterNumber columns.",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=RAW_DIR / "fema_housing_subset.json",
        help="Path to write combined raw JSON export.",
    )
    parser.add_argument(
        "--clean-out",
        type=Path,
        default=INTERIM_DIR / "fema_housing_subset_clean.csv",
        help="Path to write cleaned applicant-level CSV.",
    )
    parser.add_argument(
        "--agg-out",
        type=Path,
        default=PROCESSED_DIR / "fema_housing_by_tract_disaster.csv",
        help="Path to write tract-level aggregates.",
    )
    parser.add_argument("--top", type=int, default=5000, help="Records per API page (max 5000).")
    args = parser.parse_args()

    coverage = discover_disasters(args.coverage)
    print(f"Discovered {len(coverage)} disaster numbers to pull.")

    records: List[dict] = []
    for _, row in coverage.iterrows():
        disaster_number = int(row["disasterNumber"])
        storm = row["storm_name"]
        print(f"Fetching disaster {disaster_number} ({storm}) ...", flush=True)
        batch = fetch_disaster(disaster_number, args.top, STATE_ABBR)
        print(f"  retrieved {len(batch)} rows")
        for item in batch:
            item.setdefault("storm_name", storm)
        records.extend(batch)

    if not records:
        print("No records retrieved; nothing to write.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(records)
    df.to_json(args.raw_out, orient="records")
    print(f"Wrote raw subset -> {args.raw_out} ({len(df)} rows)")

    df["tract_geoid"] = df["censusBlockId"].map(to_tract_geoid)
    df.to_csv(args.clean_out, index=False)
    print(f"Wrote clean subset -> {args.clean_out} ({len(df)} rows)")

    agg = aggregate(df)
    agg.to_csv(args.agg_out, index=False)
    print(f"Wrote aggregates -> {args.agg_out} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
