import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import geopandas as gpd
import rasterio
from shapely.geometry import box
from wms_utils import download_wms_as_geotiff, DEFAULT_WMS_URL

gdf = gpd.GeoDataFrame(geometry=[box(152000, 381000, 156000, 385000)], crs="EPSG:28992")
tmp = Path(tempfile.mktemp(suffix=".tif"))

print("Downloaden: hitteeiland...")
download_wms_as_geotiff(gdf, wms_url=DEFAULT_WMS_URL,
    layer_name="hitteeiland", resolution_m=50, buffer_m=0, output_path=tmp)

with rasterio.open(tmp) as src:
    b = src.read(1)
    print(f"dtype  : {src.dtypes[0]}")
    print(f"banden : {src.count}")
    print(f"min    : {b.min():.2f}")
    print(f"max    : {b.max():.2f}")
    print(f"mean   : {b.mean():.2f}")
    if src.dtypes[0] in ("float32", "float64") and b.max() < 200:
        print("\n-> VERTROUWBAAR: float waarden (echte meetwaarden)")
    else:
        print("\n-> NIET VERTROUWBAAR: waarschijnlijk kleurpixels (0-255)")

tmp.unlink()