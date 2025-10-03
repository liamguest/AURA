"""Utility for clipping Gulf Coast tract shapefiles to a bounding box."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box


# Bounding box covering: SE Texas → Florida Panhandle
BBOX_COORDS = (-97.5, 27.5, -85.5, 31.5)


# Absolute paths to the tract shapefiles currently staged in the repo
STATE_SHAPEFILES = [
    Path("data/raw/shapefiles/tl_2024_22_tract/tl_2024_22_tract.shp"),  # LA
    Path("data/raw/shapefiles/tl_2024_48_tract/tl_2024_48_tract.shp"),  # TX
    Path("data/raw/shapefiles/tl_2023_01_tract/tl_2023_01_tract.shp"),  # AL
    Path("data/raw/shapefiles/tl_2024_12_tract/tl_2024_12_tract.shp"),  # FL
    Path("data/raw/shapefiles/tl_2024_28_tract/tl_2024_28_tract.shp"),  # MS
]

OUTPUT_PATH = Path("data/interim/tracts_exposed.shp")


def main() -> None:
    print("Loading and combining state tract shapefiles...")
    frames = []
    crs = None
    for path in STATE_SHAPEFILES:
        gdf = gpd.read_file(path)
        frames.append(gdf)
        if crs is None:
            crs = gdf.crs

    combined = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=crs)
    print(f"Total tracts loaded: {len(combined)}")

    bbox_polygon = box(*BBOX_COORDS)
    filtered = combined[combined.intersects(bbox_polygon)]
    print(f"Filtered down to {len(filtered)} tracts.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_file(OUTPUT_PATH)
    print(f"Saved filtered tract shapefile to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
