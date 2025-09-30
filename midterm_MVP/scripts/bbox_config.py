import geopandas as gpd
from shapely.geometry import box
import os

# === CONFIG ===
# Bounding box covering: SE Texas → Florida Panhandle
BBOX_COORDS = [-97.5, 27.5, -85.5, 31.5]

# Folder where state tract shapefiles are stored
SHAPEFILE_DIR = "data/raw/shapefiles/"

# List of state shapefile names (update if different filenames)
STATE_SHAPEFILES = [
    "tl_2024_22_tract.shp",  # LA
    "tl_2024_48_tract.shp",  # TX
    "tl_2024_01_tract.shp",  # AL
    "tl_2024_12_tract.shp",  # FL
    "tl_2024_28_tract.shp"   # MS
]

# Output location
OUTPUT_DIR = "data/interim/"
OUTPUT_NAME = "tracts_exposed.shp"

# === EXECUTION ===

# 1. Load and combine all state shapefiles
print("Loading and combining state tract shapefiles...")
all_tracts = []

for shp in STATE_SHAPEFILES:
    path = os.path.join(SHAPEFILE_DIR, shp)
    tracts = gpd.read_file(path)
    all_tracts.append(tracts)

combined_tracts = gpd.GeoDataFrame(pd.concat(all_tracts, ignore_index=True))
print(f"Total tracts loaded: {len(combined_tracts)}")

# 2. Create bounding box polygon
print("Defining bounding box...")
bbox_polygon = box(*BBOX_COORDS)

# 3. Filter tracts that intersect the bounding box
print("Filtering tracts within bounding box...")
filtered_tracts = combined_tracts[combined_tracts.intersects(bbox_polygon)]
print(f"Filtered down to {len(filtered_tracts)} tracts.")

# 4. Save filtered tracts as new shapefile
output_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME)
filtered_tracts.to_file(output_path)
print(f"Saved filtered tract shapefile to: {output_path}")