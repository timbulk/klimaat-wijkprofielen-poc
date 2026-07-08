"""test_wms_dtype.py — controleer of de WMS echte meetwaarden of kleurpixels levert.

Gebruik: python3 scripts/test_wms_dtype.py
Geen CBS-bestand nodig — gebruikt een vaste bbox rond Eindhoven centrum.
"""
import sys, tempfile, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import numpy as np
import geopandas as gpd
from shapely.geometry import box
import rasterio
from wms_utils import download_wms_as_geotiff, DEFAULT_WMS_URL

# Vaste bbox: Eindhoven centrum (EPSG:28992 RD New)
# minx, miny, maxx, maxy
BBOX_RD = (152000, 381000, 156000, 385000)
gdf = gpd.GeoDataFrame(geometry=[box(*BBOX_RD)], crs="EPSG:28992")

print("WMS downloaden (hitteeiland_r_hitte, 50m resolutie)...")
with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
    tmp = f.name

download_wms_as_geotiff(gdf, wms_url=DEFAULT_WMS_URL,
    layer_name="hitteeiland_r_hitte", resolution_m=50,
    buffer_m=0, output_path=tmp)

with rasterio.open(tmp) as src:
    band = src.read(1)
    print(f"\ndtype  : {src.dtypes[0]}")
    print(f"banden : {src.count}")
    print(f"min    : {band.min():.2f}")
    print(f"max    : {band.max():.2f}")
    print(f"mean   : {band.mean():.2f}")
    print()
    if src.dtypes[0] in ("float32", "float64") and src.count == 1:
        print("OK: single-band float raster — echte meetwaarden, zonal stats werkt correct")
    elif src.count >= 3 or (band.max() <= 255 and band.dtype == np.uint8):
        print("PROBLEEM: RGB rendered image (kleurwaarden 0-255, geen echte meetwaarden)")
        print("  -> Overweeg WCS in plaats van WMS")
    else:
        print("ONZEKER: controleer dtype en bereik hierboven")
        print("  Verwacht: float32, 1 band, bereik ~26-42 voor hitte in graden")

os.unlink(tmp)