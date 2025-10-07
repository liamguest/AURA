"""Build master dataset integrating NOAA HURDAT2 features with existing tract_storm data.

Creates three datasets:
1. tract_storm_master.csv - Full dataset with NOAA columns (mostly NaN)
2. ida_la_complete.csv - Louisiana Ida tracts only (539 rows, 100% NOAA coverage)
3. ida_all_partial.csv - All Ida tracts (811 rows, 66% NOAA coverage)

NOAA features added:
- distance_km: Distance from tract centroid to storm path
- max_wind_experienced_kt: Maximum wind speed at tract (knots)
- duration_in_envelope_hours: Hours of exposure to wind field
- within_64kt: Boolean flag for hurricane-force winds
- Lead time features for Cat 1-5 thresholds
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np

# Paths
AURA_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = AURA_ROOT / "data" / "processed"
NOAA_DIR = AURA_ROOT / "HURRICANE-DATA-ETL" / "03_integration" / "outputs"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data"

# Storm ID mapping
STORM_ID_MAP = {
    "AL092017": 4332,  # Harvey
    "AL112017": 4337,  # Irma
    "AL142018": 4399,  # Michael
    "AL132020": 4559,  # Laura
    "AL092021": 4611,  # Ida
    "AL092022": 4673,  # Ian
}


def load_existing_data():
    """Load the existing tract_storm_features dataset."""
    print("Loading existing tract_storm_features.csv...")
    df = pd.read_csv(PROCESSED_DIR / "tract_storm_features.csv")
    print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def load_noaa_data():
    """Load Ida NOAA features from Michael's pipeline."""
    print("\nLoading NOAA Ida features...")
    noaa_path = NOAA_DIR / "ida_features_complete.csv"

    if not noaa_path.exists():
        raise FileNotFoundError(f"NOAA data not found at {noaa_path}")

    df = pd.read_csv(noaa_path)
    print(f"  ✓ Loaded {len(df)} rows (Louisiana Ida tracts)")

    # Add disaster number for joining
    df['disasterNumber'] = STORM_ID_MAP['AL092021']  # Ida = 4611

    return df


def select_noaa_features(noaa_df):
    """Select and rename NOAA features for merging."""
    print("\nSelecting NOAA features...")

    # Key features to include
    feature_map = {
        'tract_geoid': 'tract_geoid',
        'disasterNumber': 'disasterNumber',
        'distance_km': 'noaa_distance_km',
        'max_wind_experienced_kt': 'noaa_max_wind_kt',
        'duration_in_envelope_hours': 'noaa_duration_hours',
        'within_64kt': 'noaa_within_64kt',
        'wind_source': 'noaa_wind_source',
        'lead_time_cat1_hours': 'noaa_lead_time_cat1',
        'lead_time_cat2_hours': 'noaa_lead_time_cat2',
        'lead_time_cat3_hours': 'noaa_lead_time_cat3',
        'lead_time_cat4_hours': 'noaa_lead_time_cat4',
        'lead_time_cat5_hours': 'noaa_lead_time_cat5',
    }

    # Select available columns
    available_cols = [col for col in feature_map.keys() if col in noaa_df.columns]
    noaa_subset = noaa_df[available_cols].copy()

    # Rename with noaa_ prefix
    rename_map = {old: new for old, new in feature_map.items() if old in available_cols}
    noaa_subset.rename(columns=rename_map, inplace=True)

    print(f"  ✓ Selected {len(rename_map)-2} NOAA features (excluding join keys)")
    print(f"  Features: {[v for k, v in rename_map.items() if k not in ['tract_geoid', 'disasterNumber']]}")

    return noaa_subset


def merge_datasets(base_df, noaa_df):
    """Merge NOAA features into base dataset."""
    print("\nMerging datasets...")
    print(f"  Base dataset: {len(base_df)} rows")
    print(f"  NOAA dataset: {len(noaa_df)} rows")

    # Left join to keep all base rows
    merged = base_df.merge(
        noaa_df,
        on=['tract_geoid', 'disasterNumber'],
        how='left',
        indicator=True
    )

    # Check merge results
    noaa_matched = (merged['_merge'] == 'both').sum()
    print(f"  ✓ Matched {noaa_matched} tracts with NOAA data")
    print(f"  ✓ {len(merged) - noaa_matched} tracts without NOAA data (NaN)")

    # Drop merge indicator
    merged.drop('_merge', axis=1, inplace=True)

    return merged


def create_analysis_subsets(master_df):
    """Create Analysis A and B subsets."""
    print("\nCreating analysis subsets...")

    # Analysis A: Louisiana Ida only (complete NOAA)
    ida_la = master_df[
        (master_df['disasterNumber'] == 4611) &
        (master_df['state_abbr'] == 'LA') &
        (master_df['noaa_distance_km'].notna())
    ].copy()
    print(f"  Analysis A (LA Ida complete): {len(ida_la)} rows")

    # Analysis B: All Ida (partial NOAA)
    ida_all = master_df[master_df['disasterNumber'] == 4611].copy()
    print(f"  Analysis B (All Ida partial): {len(ida_all)} rows")

    # Calculate NOAA coverage
    noaa_coverage_a = (ida_la['noaa_distance_km'].notna().sum() / len(ida_la)) * 100
    noaa_coverage_b = (ida_all['noaa_distance_km'].notna().sum() / len(ida_all)) * 100

    print(f"    → Analysis A NOAA coverage: {noaa_coverage_a:.1f}%")
    print(f"    → Analysis B NOAA coverage: {noaa_coverage_b:.1f}%")

    return ida_la, ida_all


def save_datasets(master_df, ida_la_df, ida_all_df):
    """Save all three datasets."""
    print("\nSaving datasets...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Master dataset
    master_path = OUTPUT_DIR / "tract_storm_master.csv"
    master_df.to_csv(master_path, index=False)
    print(f"  ✓ Saved: {master_path.name} ({len(master_df)} rows)")

    # 2. Analysis A
    ida_la_path = OUTPUT_DIR / "ida_la_complete.csv"
    ida_la_df.to_csv(ida_la_path, index=False)
    print(f"  ✓ Saved: {ida_la_path.name} ({len(ida_la_df)} rows)")

    # 3. Analysis B
    ida_all_path = OUTPUT_DIR / "ida_all_partial.csv"
    ida_all_df.to_csv(ida_all_path, index=False)
    print(f"  ✓ Saved: {ida_all_path.name} ({len(ida_all_df)} rows)")


def generate_summary_stats(master_df, ida_la_df, ida_all_df):
    """Generate summary statistics for documentation."""
    print("\n" + "="*70)
    print("DATASET SUMMARY")
    print("="*70)

    print("\n1. MASTER DATASET (tract_storm_master.csv)")
    print(f"   Total rows: {len(master_df)}")
    print(f"   Total columns: {len(master_df.columns)}")
    noaa_cols = [col for col in master_df.columns if col.startswith('noaa_')]
    print(f"   NOAA columns added: {len(noaa_cols)}")
    print(f"   Storms included: {master_df['disasterNumber'].unique()}")

    print("\n2. ANALYSIS A: Louisiana Ida Only (ida_la_complete.csv)")
    print(f"   Rows: {len(ida_la_df)}")
    print(f"   NOAA coverage: 100% (complete data)")
    print(f"   Target mean: ${ida_la_df['fema_claims_total'].mean():.2f}")
    print(f"   Target median: ${ida_la_df['fema_claims_total'].median():.2f}")

    print("\n3. ANALYSIS B: All Ida Tracts (ida_all_partial.csv)")
    print(f"   Rows: {len(ida_all_df)}")
    noaa_coverage = (ida_all_df['noaa_distance_km'].notna().sum() / len(ida_all_df)) * 100
    print(f"   NOAA coverage: {noaa_coverage:.1f}% ({ida_all_df['noaa_distance_km'].notna().sum()} of {len(ida_all_df)})")
    print(f"   Target mean: ${ida_all_df['fema_claims_total'].mean():.2f}")
    print(f"   Target median: ${ida_all_df['fema_claims_total'].median():.2f}")

    print("\n" + "="*70)


def main():
    """Build master dataset and analysis subsets."""
    print("="*70)
    print("BUILDING MASTER DATASET WITH NOAA FEATURES")
    print("="*70)

    # Load data
    base_df = load_existing_data()
    noaa_df = load_noaa_data()

    # Prepare NOAA features
    noaa_features = select_noaa_features(noaa_df)

    # Merge
    master_df = merge_datasets(base_df, noaa_features)

    # Create subsets
    ida_la_df, ida_all_df = create_analysis_subsets(master_df)

    # Save
    save_datasets(master_df, ida_la_df, ida_all_df)

    # Summary
    generate_summary_stats(master_df, ida_la_df, ida_all_df)

    print("\n✅ Master dataset build complete!")


if __name__ == "__main__":
    main()
