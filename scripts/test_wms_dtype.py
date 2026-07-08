"""test_wms_dtype.py — controleer of de WMS echte meetwaarden of kleurpixels levert."""
import sys, tempfile, os
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

import geopandas as gpd
import rasterio
from wms_utils import download_wms_as_geotiff, DEFAULT_WMS_URL

print("CBS laden...")
gdf = gpd.read_file("data/raw/wijkenbuurten_2023.gpkg", layer="buurten_2023")
gdf = gdf[gdf["gemeentenaam"] == "Eindhoven"].iloc[:3]
print(f"  {len(gdf)} buurten als test-bbox")

with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as f:
    tmp = f.name

print("WMS downloaden...")
download_wms_as_geotiff(gdf, wms_url=DEFAULT_WMS_URL,
    layer_name="hitteeiland_r_hitte", resolution_m=50, output_path=tmp)

with rasterio.open(tmp) as src:
    band = src.read(1)
    print(f"\ndtype  : {src.dtypes[0]}")
    print(f"banden : {src.count}")
    print(f"min    : {band.min():.2f}")
    print(f"max    : {band.max():.2f}")
    print(f"mean   : {band.mean():.2f}")
    print()
    if src.dtypes[0] == "uint8" and src.count >= 3:
        print("PROBLEEM: RGB rendered image (kleurwaarden 0-255, geen meetwaarden)")
    elif src.dtypes[0] in ("float32", "float64") and src.count == 1:
        print("OK: single-band float raster — echte meetwaarden")
    elif src.count == 1 and band.max() <= 255:
        print("ONZEKER: 1 band maar bereik 0-255 — mogelijk kleurwaarden als uint8")
    else:
        print("ONBEKEND: controleer dtype en bereik hierboven")

os.unlink(tmp)